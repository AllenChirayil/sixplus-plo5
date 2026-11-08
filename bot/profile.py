"""Online opponent model. All state is plain JSON-safe data in mem["p"]."""
import math

import numpy as np

from evaluator import TWO_PAIR, omaha_score
from .brain import _name, street_of
from .preflop import percentile

_SIZE = {"RAISE_HALF": 0, "RAISE_POT": 1}
REFRESH = 25


def new():
    return dict(
        hi=0, pend=None, face=[[[0, 0] for _ in range(4)] for _ in range(2)],
        raises=0, calls=0, checks=0, folds=0, hands=0, vpip=0, pfr=0,
        h_vpip=False, h_pfr=False, h_post=False, h_pre=False,
        sd_r=[0, 0], sd_c=[0, 0], pr=[0.0, 0], streak=0, spots={},
        ex=[0.0, 0.0, 0], off=False, P=None, cls=[],
    )


def observe(obs, mem):
    p = mem.setdefault("p", new())
    hist = obs.action_history
    for e in hist[p["hi"]:]:
        n, st = _name(e.action), street_of(e)
        if e.is_me:
            p["pend"] = [_SIZE[n], st] if n in _SIZE else None
            continue
        _record(p, n, st, e.amount)
    p["hi"] = len(hist)


def _record(p, n, st, amount):
    pend = p["pend"]
    key = "%d|%s" % (st, "-" if pend is None else pend[0])
    spot = p["spots"].setdefault(key, {})
    spot[n] = spot.get(n, 0) + 1
    if n == "FOLD":
        p["folds"] += 1
        p["streak"] += 1
        if pend:
            _face(p, pend, True)
    elif n == "CHECK_CALL":
        if amount > 0 or pend:
            p["calls"] += 1
            p["streak"] = 0
            if st == 0:
                p["h_vpip"] = True
        else:
            p["checks"] += 1
            p["streak"] += 1
        if pend:
            _face(p, pend, False)
    elif n in _SIZE:
        p["raises"] += 1
        p["streak"] = 0
        if st == 0:
            p["h_vpip"] = p["h_pfr"] = p["h_pre"] = True
        else:
            p["h_post"] = True
        if pend:
            _face(p, pend, False)
    p["pend"] = None


def _face(p, pend, folded):
    cell = p["face"][pend[0]][min(pend[1], 3)]
    cell[1] += 1
    if folded:
        cell[0] += 1


def after_action(obs, mem, action):
    p = mem.get("p")
    if p is None:
        return
    n = _name(action)
    st = {0: 0, 3: 1, 4: 2, 5: 3}.get(len(obs.board), 0)
    p["pend"] = [_SIZE[n], st] if n in _SIZE else None


def is_timed_out(mem, P):
    p = mem.get("p")
    return bool(p and p["streak"] >= P["timeout_streak"] and p["calls"] + p["raises"] >= 5)


def on_hand_end(res, mem, P):
    p = mem.setdefault("p", new())
    if p["pend"]:
        if not res.went_to_showdown and res.my_net > 0:
            _face(p, p["pend"], True)
            p["folds"] += 1
            p["streak"] += 1
        elif res.went_to_showdown:
            _face(p, p["pend"], False)
            p["calls"] += 1
            p["streak"] = 0
    p["hands"] += 1
    p["vpip"] += p["h_vpip"]
    p["pfr"] += p["h_pfr"]
    if res.opp_hole and res.board and len(res.board) == 5:
        strong = omaha_score(res.opp_hole, res.board) >> 20 >= TWO_PAIR
        bucket = p["sd_r"] if p["h_post"] else p["sd_c"]
        bucket[0] += 1
        bucket[1] += strong
        if p["h_pre"]:
            p["pr"][0] += float(percentile(np.array(res.opp_hole))[0])
            p["pr"][1] += 1
    if p["P"] is not None and alpha(p, P) > 0.2:
        x = float(res.my_net)
        p["ex"][0] += x
        p["ex"][1] += x * x
        p["ex"][2] += 1
    p.update(hi=0, pend=None, h_vpip=False, h_pfr=False, h_post=False, h_pre=False)
    if p["hands"] % REFRESH == 0:
        _refresh(p, P)


def alpha(p, P):
    if p["off"]:
        return 0.0
    return min(1.0, max(0.0, (p["hands"] - P["ramp_start"]) / float(P["ramp_end"] - P["ramp_start"])))


def _stats(p):
    fh = [sum(c[i] for c in p["face"][0]) for i in (0, 1)]
    fp = [sum(c[i] for c in p["face"][1]) for i in (0, 1)]
    n = fh[1] + fp[1]
    acts = p["raises"] + p["calls"] + p["checks"] + p["folds"]
    return dict(
        n=n, fold=(fh[0] + fp[0]) / float(n) if n else 0.0,
        fh=fh[0] / float(fh[1]) if fh[1] else 0.0, nh=fh[1],
        fp=fp[0] / float(fp[1]) if fp[1] else 0.0, np=fp[1],
        af=p["raises"] / float(acts) if acts else 0.0, acts=acts,
        strong=p["sd_r"][1] / float(p["sd_r"][0]) if p["sd_r"][0] else 0.0, nr=p["sd_r"][0],
    )


def _determinism(p):
    tot, top = 0, 0
    for s in p["spots"].values():
        c = sum(s.values())
        if c >= 5:
            tot += c
            top += max(s.values())
    return top / float(tot) if tot >= 40 else 0.0


def targets(p, base):
    """Exploit parameters implied by the current read; base unchanged if unclassified."""
    s, T, cls = _stats(p), dict(base), []
    if s["n"] >= base["min_face"] and s["nh"] >= 12 and s["np"] >= 12 and s["nr"] >= 8 \
            and s["fp"] - s["fh"] >= 0.08 and s["strong"] >= 0.65:
        cls.append("mc")
        T.update(bet_eq=base["bet_eq"] - 0.04, p_pot=0.75, stab=0.35, raise_eq=base["raise_eq"] - 0.03,
                 post_shift=base["post_shift"] + 0.5, call_margin=base["call_margin"] + 0.02,
                 bluff_raise=0.0)
    if s["n"] >= base["min_face"]:
        if s["fold"] >= 0.60:
            cls.append("folder")
            T.update(bluff_bet=0.40, stab=0.60, semi_bet=0.60, bet_eq=T["bet_eq"] - 0.06)
        elif s["fold"] <= 0.25:
            cls.append("caller")
            T.update(bet_eq=T["bet_eq"] - 0.05, p_pot=0.85, bluff_bet=0.0, bluff_raise=0.0,
                     semi_raise=0.0, stab=0.05)
    if s["acts"] >= 80:
        if s["af"] >= 0.40:
            cls.append("aggressor")
            T.update(call_margin=T["call_margin"] - 0.03, post_shift=T["post_shift"] - 1.0,
                     raise_eq=T["raise_eq"] - 0.04, w_open=max(T["w_open"], 0.60),
                     bluff_bet=T["bluff_bet"] * 0.5)
        elif s["af"] <= 0.12:
            cls.append("passive")
            T.update(stab=max(T["stab"], 0.5), bluff_bet=max(T["bluff_bet"], 0.3),
                     post_shift=T["post_shift"] + 1.0, call_margin=T["call_margin"] + 0.03,
                     bet_eq=T["bet_eq"] - 0.02)
    if _determinism(p) >= 0.85:
        cls.append("det")
        T["post_shift"] = T["post_shift"] + 0.5
    if p["pr"][1] >= 15:  # top-X% has mean percentile 1 - X/2
        T["w_open"] = max(0.15, min(0.85, 2.0 * (1.0 - p["pr"][0] / p["pr"][1])))
    return T, cls


def _refresh(p, base):
    e = p["ex"]
    if not p["off"] and e[2] >= base["revert_min"]:
        var = max(e[1] / e[2] - (e[0] / e[2]) ** 2, 1.0)
        if e[0] < -base["revert_z"] * math.sqrt(var * e[2]):
            p["off"] = True
    a = alpha(p, base)
    if a <= 0.0:
        p["P"], p["cls"] = None, []
        return
    T, p["cls"] = targets(p, base)
    P = dict(base)
    for k, v in T.items():
        if isinstance(v, float) and isinstance(base[k], (int, float)) and not isinstance(base[k], bool):
            P[k] = base[k] + a * (v - base[k])
    p["P"] = P


def current(mem, base):
    p = mem.get("p")
    return p["P"] if p and p["P"] else base
