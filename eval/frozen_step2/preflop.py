"""Vectorised preflop features and percentile (1.0 = best hand)."""
import json
import os

import numpy as np

_R9, _S4 = np.arange(9), np.arange(4)
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preflop_weights.json")
if os.path.exists(_path):
    with open(_path) as f:
        _d = json.load(f)
    _W = np.array(_d["weights"])
    _Q = np.array(_d["quantiles"])
else:
    _W, _Q = None, None


def features(H):
    H = np.atleast_2d(H)
    r, s = H >> 2, H & 3
    cnt = (r[:, :, None] == _R9).sum(1)
    scnt = (s[:, :, None] == _S4).sum(1)
    present = cnt > 0
    near = np.zeros_like(present)
    near[:, 1:] |= present[:, :-1]
    near[:, :-1] |= present[:, 1:]
    near[:, 2:] |= present[:, :-2]
    near[:, :-2] |= present[:, 2:]
    near[:, 0] |= present[:, 8]
    near[:, 8] |= present[:, 0]
    card_rank_cnt = np.take_along_axis(cnt, r, 1)
    card_suit_cnt = np.take_along_axis(scnt, s, 1)
    card_near = np.take_along_axis(near, r, 1)
    dangling = ((card_rank_cnt == 1) & (card_suit_cnt == 1) & ~card_near).sum(1)
    suited_ok = card_suit_cnt >= 2
    return np.column_stack([
        np.ones(len(H)),
        r.sum(1) / 8.0,
        cnt[:, 8],
        (cnt == 2).sum(1),
        ((cnt == 2) * (_R9 + 1)).max(1) / 9.0,
        (cnt == 3).sum(1),
        (cnt == 4).sum(1),
        (present[:, :-1] & present[:, 1:]).sum(1) + (present[:, 8] & present[:, 0]),
        (present[:, :-2] & present[:, 2:]).sum(1),
        (scnt >= 2).sum(1),
        (scnt >= 4).sum(1),
        ((r == 8) & suited_ok).any(1),
        ((r == 7) & suited_ok & ~(r == 8).any(1, keepdims=True)).any(1),
        dangling,
    ]).astype(float)


def score(H):
    return (features(H) * _W).sum(1)


def percentile(H):
    return np.searchsorted(_Q, score(H)) / float(len(_Q))
