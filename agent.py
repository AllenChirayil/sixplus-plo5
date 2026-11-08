"""Bot entry point: three functions, wrapped so that they can never raise.

Paste the bodies between the starter repo's ``YOUR CODE HERE`` markers (or import
from here) and keep the ``bot/`` package plus ``evaluator.py`` next to it.
"""
import time

from bot import profile
from bot.brain import _name, decide, pick
from bot.params import BASE


def _fallback(obs):
    try:
        legal = list(obs.legal_actions)
        for want in ("CHECK_CALL", "FOLD"):
            for a in legal:
                if _name(a) == want:
                    return a
        return legal[0]
    except Exception:
        return 1  # CHECK_CALL is always legal


class Agent:
    def __init__(self, params=None):
        self.P = dict(BASE)
        self.P.update(params or {})

    def new_half(self, memory):
        try:
            memory.clear()
            memory.update(t=0.0, hands=0)
        except Exception:
            pass

    def choose_action(self, obs, memory):
        t0 = time.perf_counter()
        try:
            prof = self.P["use_profile"]
            if prof:
                profile.observe(obs, memory)
            if prof and profile.is_timed_out(memory, self.P):
                a = pick(obs, ["RAISE_HALF", "RAISE_POT", "CHECK_CALL"])
            else:
                a = decide(obs, memory, profile.current(memory, self.P) if prof else self.P)
            if a is None or a not in obs.legal_actions:
                a = _fallback(obs)
            if prof:
                profile.after_action(obs, memory, a)
        except Exception:
            a = _fallback(obs)
        try:
            memory["t"] = memory.get("t", 0.0) + time.perf_counter() - t0
        except Exception:
            pass
        return a

    def on_hand_end(self, result, memory):
        try:
            memory["hands"] = memory.get("hands", 0) + 1
            if self.P["use_profile"]:
                profile.on_hand_end(result, memory, self.P)
        except Exception:
            pass


_default = Agent()
new_half = _default.new_half
choose_action = _default.choose_action
on_hand_end = _default.on_hand_end
