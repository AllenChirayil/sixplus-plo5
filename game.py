"""Heads-up short-deck PLO5 engine, written from PLAN.md.

NOTE: this is NOT the official game.py (the starter repo was not available).
Raise sizing in particular is an assumption; see PROGRESS.md.
"""
from collections import namedtuple
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional

import numpy as np

from evaluator import omaha_score

STACK, SB, BB, MAX_RAISES = 80, 1, 2, 4


class Action(IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    RAISE_HALF = 2
    RAISE_POT = 3


HistoryEntry = namedtuple("HistoryEntry", "street is_me action amount street_total")


@dataclass
class Observation:
    street: int
    hole_cards: List[int]
    board: List[int]
    pot: int
    to_call: int
    my_bet: int
    opp_bet: int
    my_stack: int
    opp_stack: int
    is_button: bool
    legal_actions: List[Action]
    raise_to: Dict[Action, int]
    raises_this_street: int
    action_history: List[HistoryEntry] = field(default_factory=list)
    big_blind: int = BB
    hand_index: int = 0


@dataclass
class HandResult:
    my_net: int
    went_to_showdown: bool
    board: List[int]
    opp_hole: Optional[List[int]]
    winner: int  # 1 me, -1 opponent, 0 split
    hand_index: int = 0


def make_deals(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = rng.permutation(36)[:15].tolist()
        out.append((c[:5], c[5:10], c[10:15]))
    return out


def raise_options(my_bet, opp_bet, pot, my_stack, opp_stack, last_raise, raises):
    """Total street amounts for the raise actions; empty when raising is illegal."""
    call = opp_bet - my_bet
    if raises >= MAX_RAISES or opp_stack <= 0 or my_stack <= call:
        return {}
    after = pot + call
    min_to = opp_bet + max(last_raise, BB)
    max_to = my_bet + my_stack
    return {
        Action.RAISE_HALF: min(max(opp_bet + after // 2, min_to), max_to),
        Action.RAISE_POT: min(max(opp_bet + after, min_to), max_to),
    }


def play_hand(deal, button, act, hand_index=0):
    """Play one hand. ``act(seat, obs)`` returns an Action. Returns (nets, results)."""
    holes = [deal[0], deal[1]]
    full_board = deal[2]
    stacks = [STACK, STACK]
    bets = [0, 0]
    bb_seat = 1 - button
    bets[button], stacks[button] = SB, STACK - SB
    bets[bb_seat], stacks[bb_seat] = BB, STACK - BB
    pot_done = 0
    history = []
    folded = None
    for street, n_board in enumerate((0, 3, 4, 5)):
        if street > 0:
            pot_done += sum(bets)
            bets = [0, 0]
        if min(stacks) <= 0:
            continue
        seat = button if street == 0 else bb_seat
        acted = [False, False]
        raises, last_raise = 0, BB
        while True:
            if acted[0] and acted[1] and bets[0] == bets[1]:
                break
            o = 1 - seat
            to_call = bets[o] - bets[seat]
            pot = pot_done + sum(bets)
            raise_to = raise_options(bets[seat], bets[o], pot, stacks[seat], stacks[o],
                                     last_raise, raises)
            legal = [Action.CHECK_CALL] + list(raise_to)
            if to_call > 0:
                legal.insert(0, Action.FOLD)
            view = [HistoryEntry(h[0], h[1] == seat, h[2], h[3], h[4]) for h in history]
            obs = Observation(street, list(holes[seat]), list(full_board[:n_board]), pot,
                              to_call, bets[seat], bets[o], stacks[seat], stacks[o],
                              seat == button, legal, dict(raise_to), raises, view, BB,
                              hand_index)
            a = act(seat, obs)
            if a not in legal:
                a = Action.CHECK_CALL
            if a == Action.FOLD:
                history.append((street, seat, a, 0, bets[seat]))
                folded = seat
                break
            if a == Action.CHECK_CALL:
                pay = min(to_call, stacks[seat])
            else:
                target = raise_to[a]
                pay = target - bets[seat]
                last_raise = max(target - bets[o], BB)
                raises += 1
                acted[o] = False
            bets[seat] += pay
            stacks[seat] -= pay
            history.append((street, seat, a, pay, bets[seat]))
            acted[seat] = True
            if a == Action.CHECK_CALL and min(stacks) <= 0 and bets[0] == bets[1]:
                break
            seat = o
        if folded is not None:
            break
    contrib = [STACK - stacks[0], STACK - stacks[1]]
    pot_total = sum(contrib)
    showdown = folded is None
    if folded is not None:
        winner_seat = 1 - folded
        nets = [0, 0]
        nets[winner_seat] = contrib[folded]
        nets[folded] = -contrib[folded]
        codes = [1 if winner_seat == 0 else -1, 1 if winner_seat == 1 else -1]
        shown = None
    else:
        s0, s1 = omaha_score(holes[0], full_board), omaha_score(holes[1], full_board)
        if s0 == s1:
            nets, codes = [0, 0], [0, 0]
        else:
            w = 0 if s0 > s1 else 1
            nets = [0, 0]
            nets[w] = pot_total // 2
            nets[1 - w] = -pot_total // 2
            codes = [1 if w == 0 else -1, 1 if w == 1 else -1]
        shown = holes
    n_seen = len(full_board) if showdown else 0
    results = [HandResult(nets[s], showdown, list(full_board) if showdown else [],
                          list(shown[1 - s]) if shown else None, codes[s], hand_index)
               for s in (0, 1)]
    assert n_seen in (0, 5)
    return nets, results
