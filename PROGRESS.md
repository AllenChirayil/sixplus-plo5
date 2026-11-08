# Progress

Built from PLAN.md in a new repo, without the official starter repo.

## Done
- Step 1: evaluator, engine, in-process duplicate runner with CI, bank and error reporting.
- Step 2: range-aware base bot with nut awareness, mixed sizing and a time guard.
- Step 3: opponent zoo (`eval/zoo.py`), frozen copy in `eval/frozen_step2/`.
- Step 4: online profiling and exploitation (`bot/profile.py`). 4 seeds vs the zoo:
  26 wins, 10 chops, 0 losses (+1.79 bb/deal) vs 22/10/0 (+1.37) for the frozen base.
- Step 5: current vs frozen over 8 seeds: 1 win, 7 chops, 0 losses, 0 errors.
- Step 7 (partial): fuzz and JSON-safe memory tests in `tests/test_agent.py`.

## Not done
- Step 6: sweep ran, but its results are not applied. Candidates: `post_shift=1`,
  `call_margin=0.09`, `semi_bet=0.25`. Need testing together.
- Step 7: final timing and release checks.
- Never run against the official judge. `game.py` is written from PLAN.md, and raise sizing
  and the `agent.py` function signatures are assumptions.
- Port to the official `agent.py`, push, submit.

`eval/last_gauntlet.txt` is from the base bot, before Step 4.
