"""Opponent zoo mimicking the expected field (PLAN.md Step 3)."""
import types

import numpy as np

from evaluator import score_batch
from game import Action

_rng = np.random.default_rng()


def eq_vs_random(hero, board, n):
    deck = np.array([c for c in range(36) if c not in set(hero) | set(board)])
    k = 5 - len(board)
    p = _rng.random((n, len(deck))).argsort(1)[:, :5 + k]
    opp, run = deck[p[:, :5]], deck[p[:, 5:]]
    full = np.hstack([np.tile(board, (n, 1)), run]) if board else run
    if k == 0:
        full = np.tile(board, (n, 1))
    a, b = score_batch(hero, full), score_batch(opp, full)
    return float((a > b).mean() + 0.5 * (a == b).mean())


class MC:
    def __init__(self, rollouts=100, thr=0.65, margin=0.0, track=False):
        self.n, self.thr, self.margin, self.track = rollouts, thr, margin, track

    def new_half(self, mem):
        mem.update(seen=0, agg=0)

    def choose_action(self, obs, mem):
        thr, margin = self.thr, self.margin
        if self.track:
            for e in obs.action_history:
                pass
            opp = [e for e in obs.action_history if not e.is_me]
            mem["seen"] += len(opp)
            mem["agg"] += sum(e.action >= Action.RAISE_HALF for e in opp)
            if mem["seen"] > 40:
                rate = mem["agg"] / float(mem["seen"])
                d = 0.05 if rate > 0.3 else -0.05 if rate < 0.15 else 0.0
                thr -= d
                margin -= d / 2
        eq = eq_vs_random(obs.hole_cards, obs.board, self.n)
        po = obs.to_call / float(obs.pot + obs.to_call) if obs.to_call else 0.0
        legal = obs.legal_actions
        if eq > thr and Action.RAISE_POT in legal:
            return Action.RAISE_POT if eq > thr + 0.1 else Action.RAISE_HALF
        if obs.to_call == 0:
            return Action.CHECK_CALL
        return Action.CHECK_CALL if eq > po + margin else Action.FOLD

    def on_hand_end(self, res, mem):
        pass


def frozen_bot(overrides=None):
    from eval.frozen_step2.brain import _name, decide
    from eval.frozen_step2.params import BASE
    import time
    P = dict(BASE)
    P.update(overrides or {})

    def new_half(mem):
        mem.update(t=0.0)

    def choose(obs, mem):
        t = time.perf_counter()
        a = decide(obs, mem, P)
        mem["t"] += time.perf_counter() - t
        return a if a in obs.legal_actions else Action.CHECK_CALL
    return types.SimpleNamespace(new_half=new_half, choose_action=choose,
                                 on_hand_end=lambda r, m: None)


AGGRO = dict(bet_eq=0.48, raise_eq=0.55, raise_over=0.08, semi_bet=0.8, semi_raise=0.4,
             bluff_bet=0.5, bluff_raise=0.25, stab=0.7, open_cuts=(0.25, 0.55, 0.75, 0.9))
PASSIVE = dict(bet_eq=0.8, raise_eq=0.88, raise_over=0.3, semi_bet=0.05, semi_raise=0.0,
               bluff_bet=0.0, bluff_raise=0.0, stab=0.0, call_margin=-0.02,
               open_cuts=(0.93, 0.97, 0.99, 0.995))

ZOO = {
    "mc_naive": lambda: MC(),
    "mc_tight": lambda: MC(thr=0.75, margin=0.05),
    "mc_loose": lambda: MC(thr=0.58, margin=-0.03),
    "mc_track": lambda: MC(track=True),
    "mc_heavy": lambda: MC(rollouts=6000),
    "nn_det": lambda: frozen_bot(dict(det=True)),
    "nn_aggro": lambda: frozen_bot(AGGRO),
    "nn_passive": lambda: frozen_bot(PASSIVE),
    "frozen": lambda: frozen_bot(),
}
