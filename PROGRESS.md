# Progress

## Layout
- Root: official starter files (`game.py`, `evaluator.py`, `judge_engine.py`, `test_agent.py`,
  `docker/`, `tools/`, `baseline-bots/`). `agent.py` is the official server; its
  `YOUR CODE HERE` section calls `bot/core.py`.
- `bot/`: strategy, fast evaluator, profiling.
- `eval/`: in-process simulator (`sim_game.py`), opponent zoo, sweeps, frozen Step 2 bot.
  Not used at match time.

## Done
- Steps 1-3: engine, base bot, opponent zoo.
- Step 4: profiling and exploitation. 4 seeds vs the zoo: 26 wins, 10 chops, 0 losses
  (+1.79 bb/deal) vs 22/10/0 (+1.37) for the frozen base.
- Step 5: current vs frozen over 8 seeds: 1 win, 7 chops, 0 losses, 0 errors.
- Official checks: `pytest test_agent.py` (12 passed), `tools/smoke_test.py` passed against a
  local server, 0 mismatches vs the official evaluator over 40,000 matchups.
- Official judge vs the team's Stallion bot: 7 seeds, all chops, sixplus ahead in all 7.
- Official judge vs `baseline_value_bettor`: win, +6.80 bb/deal.

## Not done
- Step 6: sweep ran, but results are not applied. Candidates: `post_shift=1`,
  `call_margin=0.09`, `semi_bet=0.25`. Need testing together.
- `tools/check_build.sh`: not run, Docker is not installed on this machine.
- Decisions take about 0.4 ms (median), so there is room for more rollouts.

`eval/last_gauntlet.txt` is from the base bot, before Step 4.

## Leak fixes from hand logs (kept)
Found from 16,000 logged hands vs Stallion, tested on 16 different seeds (official judge):

| Variant | bb/deal vs Stallion | W / C / L |
|---|---|---|
| All off | +0.184 | 0 / 16 / 0 |
| Fix 1: tighter calls vs pot-size river bets | +0.384 | 3 / 13 / 0 |
| Fix 2: less money in with two pair or worse | +0.461 | 3 / 13 / 0 |
| Fix 3: bluff sizing by street, more river bluffs | +0.380 | 2 / 14 / 0 |
| All on | +0.545 | 4 / 12 / 0 |

All on vs all off: +0.362 bb/deal (se 0.131). Zoo (32 matches): +2.33 vs +2.16, 24 wins vs 23,
0 losses. Docker check was run before these Python-only changes.
