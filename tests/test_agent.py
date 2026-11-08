"""Fuzz and sanity checks. Calls decide() directly so the wrapper can't hide exceptions."""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent  # noqa: E402
from bot.brain import decide  # noqa: E402
from bot.params import BASE  # noqa: E402
from game import Action, HistoryEntry, Observation, make_deals, play_hand  # noqa: E402


def fuzz(n=4000):
    for _ in range(n):
        c = random.sample(range(36), 10)
        nb = random.choice([0, 3, 4, 5])
        to_call = random.choice([0, 0, 1, 2, 10, 79, 200])
        legal = [Action.CHECK_CALL] if to_call == 0 else [Action.FOLD, Action.CHECK_CALL]
        rt = {}
        if random.random() < 0.6:
            rt = {Action.RAISE_HALF: random.randint(2, 80), Action.RAISE_POT: random.randint(2, 80)}
            legal += list(rt)
        hist = [HistoryEntry(random.randint(0, 3), random.random() < 0.5, random.choice(list(Action)),
                             random.randint(0, 80), random.randint(0, 80))
                for _ in range(random.choice([0, 0, 1, 5, 12]))]
        o = Observation(max(0, nb - 2), c[:5], c[5:5 + nb], random.choice([0, 3, 40, 160]), to_call,
                        0, to_call, random.choice([0, 1, 40, 80]), random.choice([0, 40, 80]),
                        random.random() < 0.5, legal, rt, random.randint(0, 4), hist, 2,
                        random.choice([0, 999, 5000]))
        mem = {"t": random.choice([0.0, 3.9, 7.0])}
        a = decide(o, mem, BASE)
        assert a in legal, (a, legal)
        assert agent.choose_action(o, mem) in legal


def profile_no_exceptions():
    """agent.py swallows errors, so drive the profile directly over real hands."""
    from bot import profile
    mem = {"t": 0.0}
    for i, d in enumerate(make_deals(400, 9)):
        def act(seat, o):
            if seat == 0:
                profile.observe(o, mem)
                a = random.choice(o.legal_actions)
                profile.after_action(o, mem, a)
                return a
            return random.choice(o.legal_actions)
        _, res = play_hand(d, i % 2, act, i)
        profile.on_hand_end(res[0], mem, BASE)
    assert mem["p"]["hands"] == 400 and mem["p"]["P"] is not None
    json.dumps(mem)


def memory_json_safe():
    mem = {}
    agent.new_half(mem)
    for i, d in enumerate(make_deals(30, 3)):
        def act(seat, o):
            nonlocal mem
            a = agent.choose_action(o, mem)
            mem = json.loads(json.dumps(mem))
            return a
        _, res = play_hand(d, i % 2, act, i)
        before = mem.get("hands", 0)
        agent.on_hand_end(res[0], mem)
        assert mem["hands"] == before + 1
        assert "p" in mem, "profile did not run"


if __name__ == "__main__":
    fuzz()
    profile_no_exceptions()
    memory_json_safe()
    print("ALL TESTS PASSED")
