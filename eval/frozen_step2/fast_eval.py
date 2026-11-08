"""Short-deck Omaha evaluator. Card int: rank = c >> 2 (0=six .. 8=ace), suit = c & 3.

Ranking: straight flush > quads > flush > full house > trips > straight >
two pair > pair > high card. Category is ``score >> 20``.

Scores come from two 9^5 tables keyed by the *unsorted* ranks, so numpy batches
need no sorting.
"""
from itertools import combinations

import numpy as np

HIGH, PAIR, TWO_PAIR, STRAIGHT, TRIPS, FULL_HOUSE, FLUSH, QUADS, STRAIGHT_FLUSH = range(9)
_POW = 9 ** 5


def _straight_high(uniq):
    if len(uniq) != 5:
        return None
    if uniq[0] - uniq[4] == 4:
        return uniq[0]
    if uniq == [8, 3, 2, 1, 0]:  # A-6-7-8-9
        return 3
    return None


def _encode(cat, ranks):
    v = cat << 20
    for i, r in enumerate(ranks):
        v |= r << (16 - 4 * i)
    return v


def _score_ranks(sorted_desc):
    counts = {}
    for r in sorted_desc:
        counts[r] = counts.get(r, 0) + 1
    order = sorted(counts, key=lambda r: (counts[r], r), reverse=True)
    shape = sorted(counts.values(), reverse=True)
    st = _straight_high(sorted(counts, reverse=True))
    if shape[0] >= 4:
        return _encode(QUADS, order)
    if shape[:2] == [3, 2]:
        return _encode(FULL_HOUSE, order)
    if shape[0] == 3:
        return _encode(TRIPS, order)
    if st is not None:
        return _encode(STRAIGHT, [st])
    if shape[:2] == [2, 2]:
        return _encode(TWO_PAIR, order)
    if shape[0] == 2:
        return _encode(PAIR, order)
    return _encode(HIGH, order)


def _score_flush(sorted_desc):
    uniq = sorted(set(sorted_desc), reverse=True)
    st = _straight_high(uniq)
    if st is not None:
        return _encode(STRAIGHT_FLUSH, [st])
    return _encode(FLUSH, sorted_desc)


def _build_tables():
    idx = np.indices((9,) * 5).reshape(5, -1).T
    srt = -np.sort(-idx, axis=1)
    sorted_key = (srt * (9 ** np.arange(4, -1, -1))).sum(1)
    uniq, inv = np.unique(sorted_key, return_inverse=True)
    n = len(uniq)
    tr = np.zeros(n, dtype=np.int32)
    tf = np.zeros(n, dtype=np.int32)
    for i, k in enumerate(uniq.tolist()):
        ranks = [(k // 9 ** p) % 9 for p in range(4, -1, -1)]
        tr[i] = _score_ranks(ranks)
        tf[i] = _score_flush(ranks) if len(set(ranks)) == 5 else 0
    full_key = (idx * (9 ** np.arange(4, -1, -1))).sum(1)
    rank_tab = np.zeros(_POW, dtype=np.int32)
    flush_tab = np.zeros(_POW, dtype=np.int32)
    rank_tab[full_key] = tr[inv.ravel()]
    flush_tab[full_key] = tf[inv.ravel()]
    return rank_tab, flush_tab


TR, TF = _build_tables()
_COMBOS = {m: np.array(list(combinations(range(m), 3))) for m in (3, 4, 5)}


def _pair_idx(n):
    return np.array(list(combinations(range(n), 2)))


_PAIRS = {n: _pair_idx(n) for n in (2, 5)}


def score_batch(hole, board):
    """Best exactly-2+3 score. hole: (N,h) or (h,), board: (N,m) or (m,), m in 3..5."""
    hole = np.atleast_2d(np.asarray(hole))
    board = np.atleast_2d(np.asarray(board))
    p = _PAIRS[hole.shape[1]]
    t = _COMBOS[board.shape[1]]
    hr, hs, br, bs = hole >> 2, hole & 3, board >> 2, board & 3
    a, b = p[:, 0], p[:, 1]
    pk = hr[:, a] * 9 + hr[:, b]
    ps = np.where(hs[:, a] == hs[:, b], hs[:, a], -1)
    x, y, z = t[:, 0], t[:, 1], t[:, 2]
    tk = br[:, x] * 81 + br[:, y] * 9 + br[:, z]
    ts = np.where((bs[:, x] == bs[:, y]) & (bs[:, y] == bs[:, z]), bs[:, x], -2)
    key = pk[:, :, None] * 729 + tk[:, None, :]
    flush = ps[:, :, None] == ts[:, None, :]
    return np.where(flush, TF[key], TR[key]).max(axis=(1, 2))


def omaha_score(hole, board):
    if len(board) != 5:
        raise ValueError("omaha_score needs a full 5-card board")
    return int(score_batch(list(hole), list(board))[0])
