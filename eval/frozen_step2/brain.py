"""Range-aware decision logic. All per-half state lives in ``mem`` (JSON-safe)."""
import math

import numpy as np

from .fast_eval import FLUSH, PAIR, STRAIGHT, TWO_PAIR, score_batch
from .preflop import percentile

_rng = np.random.default_rng()
SOFT_BANK, HARD_BANK = 4.0, 6.5


def _name(a):
    n = getattr(a, "name", a)
    return str(n).upper() if not isinstance(n, int) else {0: "FOLD", 1: "CHECK_CALL", 2: "RAISE_HALF", 3: "RAISE_POT"}.get(n, "")


def pick(obs, names):
    for n in names:
        for a in obs.legal_actions:
            if _name(a) == n:
                return a
    return None


def sig(x, t):
    return 1.0 / (1.0 + math.exp(-max(min(x / t, 30), -30)))


def street_of(e):
    s = e.street
    if isinstance(s, str):
        return {"PREFLOP": 0, "FLOP": 1, "TURN": 2, "RIVER": 3}.get(s.upper(), 0)
    return int(getattr(s, "value", s))


def read_history(obs):
    """Summarise betting so far from the opponent's perspective."""
    pre, post, here = 0, 0, 0
    cur = {0: 0, 3: 1, 4: 2, 5: 3}.get(len(obs.board), 0)
    called_ours, opp_checked = False, False
    for e in obs.action_history:
        n = _name(e.action)
        if e.is_me:
            continue
        st = street_of(e)
        if n.startswith("RAISE"):
            if st == 0:
                pre += 1
            else:
                post += 1
                if st == cur:
                    here += 1
        elif n == "CHECK_CALL":
            if st == 0:
                called_ours = True
            if st == cur and e.amount == 0:
                opp_checked = True
    return pre, post, here, called_ours, opp_checked


def width_for(pre, called, P):
    if pre >= 3:
        return P["w_4bet"]
    if pre == 2:
        return P["w_3bet"]
    if pre == 1:
        return P["w_open"]
    return P["w_call"] if called else P["w_limp"]


def _post_mask(H, board, level):
    cat = score_batch(H, board) >> 20
    if level <= 0:
        return np.ones(len(H), dtype=bool)
    river = len(board) == 5
    made = {1: PAIR, 2: TWO_PAIR}.get(level, STRAIGHT)
    mask = cat >= made
    if not river and level <= 2:
        bs = np.bincount(np.array(board) & 3, minlength=4)
        draw_suits = np.nonzero(bs >= 2)[0]
        if len(draw_suits):
            mask |= ((H & 3)[:, :, None] == draw_suits).sum(1).max(1) >= 2
    return mask


def range_equity(hero, board, width, level, N):
    used = set(hero) | set(board)
    deck = np.array([c for c in range(36) if c not in used])
    k = 5 - len(board)
    M = int(min(max(N * 2, N / max(width, 0.05)), 420))
    perm = _rng.random((M, len(deck))).argsort(1)[:, :5 + k]
    opp = deck[perm[:, :5]]
    keep = percentile(opp) >= 1.0 - width
    if board and level > 0:
        keep2 = keep & _post_mask(opp, board, level)
        if keep2.sum() >= 12:
            keep = keep2
    if keep.sum() >= 12:
        perm, opp = perm[keep][:N], opp[keep][:N]
    else:
        perm, opp = perm[:N], opp[:N]
    run = deck[perm[:, 5:]]
    full = run if not board else np.hstack([np.tile(board, (len(run), 1)), run]) if k else np.tile(board, (len(run), 1))
    a, b = score_batch(hero, full), score_batch(opp, full)
    return float((a > b).mean() + 0.5 * (a == b).mean())


def nut_info(hero, board):
    used = set(hero) | set(board)
    deck = np.array([c for c in range(36) if c not in used])
    mine = int(score_batch(hero, board)[0])
    i, j = np.triu_indices(len(deck), 1)
    others = score_batch(np.column_stack([deck[i], deck[j]]), board)
    beat = float((others > mine).mean())
    bcounts = [0, 0, 0, 0]
    for c in board:
        bcounts[c & 3] += 1
    need = 2 if len(board) < 5 else 3
    fd, blocker = None, False
    for s in range(4):
        if bcounts[s] < need:
            continue
        on_board = {c >> 2 for c in board if c & 3 == s}
        unseen = [r for r in range(8, -1, -1) if r not in on_board]
        mine_s = {c >> 2 for c in hero if c & 3 == s}
        if unseen and unseen[0] in mine_s:
            blocker = True
        if len(mine_s) >= 2 and len(board) < 5:
            pos = min((unseen.index(r) for r in mine_s), default=9)
            fd = pos if fd is None else min(fd, pos)
    outs = 0
    if len(board) < 5:
        boards = np.column_stack([np.tile(board, (len(deck), 1)), deck])
        nxt = score_batch(hero, boards)
        outs = int(((nxt > mine) & ((nxt >> 20) >= STRAIGHT)).sum())
    return dict(score=mine, cat=mine >> 20, beat=beat, fd=fd, outs=outs, blocker=blocker, n=len(deck))


def rollouts_for(obs, mem, P):
    used = mem.get("t", 0.0)
    if used >= HARD_BANK:
        return 0
    left = max(1, 1000 - int(getattr(obs, "hand_index", 0) or 0))
    per = max(SOFT_BANK - used, 0.05) / (left * 5.0)
    return int(max(16, min(P["rollouts"], P["rollouts"] * per / 0.0007)))


def sample(probs, P):
    if P.get("det"):
        return max(probs, key=probs.get)
    x, acc = _rng.random() * sum(probs.values()), 0.0
    for k, v in probs.items():
        acc += v
        if x <= acc:
            return k
    return "CALL"


def to_action(obs, choice, P):
    if choice == "RAISE":
        sz = "RAISE_POT" if (P["p_pot"] >= 0.5 if P.get("det") else _rng.random() < P["p_pot"]) else "RAISE_HALF"
        other = "RAISE_HALF" if sz == "RAISE_POT" else "RAISE_POT"
        return pick(obs, [sz, other, "CHECK_CALL"])
    if choice == "FOLD" and obs.to_call > 0:
        return pick(obs, ["FOLD", "CHECK_CALL"])
    return pick(obs, ["CHECK_CALL", "FOLD"])


def cheap(obs, P):
    """No-rollout rules for when the time bank is nearly gone."""
    if not obs.board:
        pct = float(percentile(np.array(obs.hole_cards))[0])
        return "CALL" if obs.to_call <= 2 or pct > 0.8 else "FOLD"
    cat = int(score_batch(obs.hole_cards, obs.board)[0]) >> 20
    if cat >= TWO_PAIR:
        return "CALL"
    return "FOLD" if obs.to_call > 0 else "CALL"


def decide(obs, mem, P):
    N = rollouts_for(obs, mem, P)
    if N <= 0:
        return to_action(obs, cheap(obs, P), P)
    hero, board = list(obs.hole_cards), list(obs.board)
    pre, post, here, called, checked = read_history(obs)
    width = width_for(pre, called, P)
    pot, to_call = obs.pot, obs.to_call
    po = to_call / float(pot + to_call) if to_call > 0 else 0.0
    if not board:
        return to_action(obs, sample(_preflop(obs, hero, width, po, N, P), P), P)
    info = nut_info(hero, board)
    level = max(0, int(round(min(post, 3) + P["post_shift"])))
    eq = range_equity(hero, board, width, level, N)
    return to_action(obs, sample(_postflop(obs, info, eq, po, post, here, checked, P), P), P)


def _preflop(obs, hero, width, po, N, P):
    pct = float(percentile(np.array(hero))[0])
    eq = range_equity(hero, [], width, 0, N)
    if not obs.is_button:
        eq *= 0.98
    cuts = P["open_cuts"]
    r = sig(pct - cuts[min(obs.raises_this_street, 3)], 0.04)
    if obs.to_call == 0:
        return {"RAISE": r, "CALL": 1 - r}
    margin = P["call_margin"] + P["call_margin_agg"] * max(0, obs.raises_this_street - 1)
    c = sig(eq - po - margin, 0.02)
    return {"RAISE": r, "CALL": (1 - r) * c, "FOLD": (1 - r) * (1 - c)}


def _postflop(obs, info, eq, po, post, here, checked, P):
    n_board = len(obs.board)
    real = {3: 0.97, 4: 0.985, 5: 1.0}[n_board] if not obs.is_button else 1.0
    agg = min(post, 2)
    if obs.to_call > 0 and agg and info["beat"] > 0 and info["cat"] in (STRAIGHT, FLUSH):
        eq -= P["disc"] * agg
    eq *= real
    strong_draw = n_board < 5 and (info["fd"] == 0 or info["outs"] >= 8)
    if obs.to_call == 0:
        thr = P["bet_eq"] - (0.03 if obs.is_button else 0.0)
        r = sig(eq - thr, 0.04)
        if strong_draw:
            r = max(r, P["semi_bet"])
        if info["blocker"]:
            r = max(r, P["bluff_bet"])
        if checked and obs.is_button:
            r = max(r, P["stab"])
        return {"RAISE": r, "CALL": 1 - r}
    thr = max(P["raise_eq"], po + P["raise_over"])
    r = sig(eq - thr, 0.04)
    if strong_draw:
        r = max(r, P["semi_raise"])
    if info["blocker"]:
        r = max(r, P["bluff_raise"])
    margin = P["call_margin"] + P["call_margin_agg"] * max(0, here - 1)
    c = sig(eq - po - margin, 0.02)
    return {"RAISE": r, "CALL": (1 - r) * c, "FOLD": (1 - r) * (1 - c)}
