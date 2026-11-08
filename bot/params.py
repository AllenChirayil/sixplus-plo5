"""Tunable strategy parameters (Step 2 base)."""
BASE = dict(
    # opponent preflop range widths (top X%)
    w_limp=0.80, w_open=0.40, w_3bet=0.15, w_4bet=0.07, w_call=0.65,
    post_shift=0.0,      # shifts the postflop range filter level
    open_cuts=(0.50, 0.80, 0.92, 0.97),
    bet_eq=0.60, raise_eq=0.70, raise_over=0.20,
    call_margin=0.06, call_margin_agg=0.02,
    semi_bet=0.45, semi_raise=0.05, bluff_bet=0.14, bluff_raise=0.05, stab=0.20,
    disc=0.05, p_pot=0.5,
    det=False, rollouts=280,
)

BASE.update(
    use_profile=True, ramp_start=60, ramp_end=300, min_face=30,
    timeout_streak=20, revert_z=2.0, revert_min=100,
)

# Leak fixes found from hand logs (see PROGRESS.md).
BASE.update(
    rp_margin=0.12, rp_weak=0.08,                 # calling pot-size river bets
    tp_call=0.04, tp_raise=0.05, tp_bet=0.03,     # flop/turn with two pair or worse
    bluff_pot_fltn=0.15, bluff_pot_river=0.85,    # bluff sizing by street
    river_stab=0.25, river_bluff_eq=0.45,         # river bluffs with no showdown value
)
