# Poker Bot — FAST BUILD (a few hours) — TAMU Datathon HU Short Deck PLO5

> **For Claude Code:** We have only a few hours in total. Work through the steps below IN ORDER, sticking to the time boxes. Reliability comes first, then strategy. After each step, update `PROGRESS.md` (what's done, test results, next step). Do NOT start Deep CFR, neural-network training or any other long-running project.

## The situation

The starter repo is https://github.com/tamu-datathon-org/heads-up-bundle. Our bot is `agent.py`. We fill in only `new_half`, `choose_action` and `on_hand_end`, between the `YOUR CODE HERE` markers. The server plumbing must NOT be modified. We may import our own modules and load files committed to the repo.

**Format:** a round robin of 2,000-hand duplicate matches, each 1,000 deals played twice with seats swapped. Both bots get exactly the same cards, so **only strategy matters**.

**Verdict:** a 95% confidence interval on our bb/deal.
- Interval entirely above 0: win, 1 point.
- Interval touches 0: chop, 0.5 points.
- Interval entirely below 0: loss, 0 points.

**Errors:** 5 errors in a half = forfeit = MATCH LOSS.

**The opponents are ALL advanced.** There are no trivial bots. Expect two families:

1. **Monte Carlo equity bots**, the guide's own suggestion: they estimate equity against a random hand, compare it to pot odds, and maybe track the opponent a little.
2. **Deep learning bots**: neural networks trained by self-play, reinforcement learning or a quick Deep CFR. Given the hackathon time limit, they will be under-trained.

**Our edge comes from two sources:**
- **(A)** a base strategy that fixes the known flaws of naive Monte Carlo play: it reads the opponent's likely range, values nut hands correctly, and is hard to read;
- **(B)** online detection and exploitation of each opponent's specific leaks within a single 1,000-hand half.

**Priorities:**
1. Never error.
2. Never lose a match by a margin the judge can confirm.
3. Win as many matches as possible.

## Rules (verify against game.py; the code wins any conflict)

- **Deck:** 36 cards (6 through A). Card int: `rank = c >> 2` (0 = six … 8 = ace), `suit = c & 3`.
- **Ranking, best to worst:** straight flush > quads > FLUSH > FULL HOUSE > TRIPS > STRAIGHT > two pair > pair > high card. A-6-7-8-9 is the lowest straight.
- **Omaha rule:** EXACTLY 2 hole cards + 3 board cards. `omaha_score(hole, board)` handles this, needs a full 5-card board, and its category is `score >> 20`.
- **Hand setup:**
  - Heads-up, 5 hole cards each.
  - Stacks are 80 chips (40 bb), reset every hand.
  - Blinds are 1 and 2. The button posts the small blind and alternates every hand.
- **Betting order:** the button acts first preflop, and the big blind has its option. The non-button player acts first postflop.
- **Actions:**
  - FOLD: only when `to_call > 0`.
  - CHECK_CALL: always legal.
  - RAISE_HALF and RAISE_POT: the game computes the amounts in `obs.raise_to`.
  - Max 4 raises per street.
- **Rule for our code:** ALWAYS return something in `obs.legal_actions`.
- **Observation:**
  - `street`, `hole_cards`, `board`
  - `pot`, `to_call`, `my_bet`, `opp_bet`, `my_stack`, `opp_stack`, `is_button`
  - `legal_actions`, `raise_to`, `raises_this_street`
  - `action_history` (street, is_me, action, amount, street_total)
  - `big_blind`, `hand_index`
- **HandResult:** `my_net`, `went_to_showdown`, `board`, `opp_hole` (showdown only), `winner`.

## Hard limits

- **Time:** an 8 s time bank per half (1,000 hands), about 2–4 ms per decision. Any request over 1 s = error. An empty bank = auto check/fold.
  - **Target: under 4 s per half locally**, because the tournament machine may be slower.
- **Machine:** 1 CPU, 4 GB RAM, no internet. Writes only to `/tmp`.
- **State:** a fresh process every half. `memory` is the only state; nothing carries over between halves.
- **Dependencies:** declare extras in a `requirements.txt` at the repo root. Prefer the standard library plus numpy.
- **Environment:** the user is on Windows. Use PowerShell commands (`$env:PORT=5010; python agent.py`).

---

## STEP 1 — Setup and reading (≈15 min)

- Clone the starter repo into the user's own GitHub repo.
- Read `game.py`, `evaluator.py`, `agent.py` and `baseline-bots/`.
- Flag any rule that differs from this file.
- Benchmark `omaha_score`, and how many Monte Carlo rollouts fit in 1–2 ms in Python.
- Write `eval/run_match.ps1` (or a .py) that starts two bots on chosen ports, runs `judge_engine.py --deals N --seed S`, and prints the RESULT line, bb/deal, the CI, the time bank used and the error count.
  - Support running several seeds back to back.
  - Report the CI, because a chop vs a win is what matters.

## STEP 2 — Range-aware base bot (≈75 min) → SUBMIT

The goal is a base strategy that beats naive Monte Carlo bots on its own, without exploiting anything.

**Bulletproof wrapper and time guard:**
- Wrap all three functions entirely in try/except.
- Check every returned action against `obs.legal_actions`.
- Fallback: CHECK_CALL if legal, else FOLD, else `legal_actions[0]`.
- Accumulate request time in `memory` with `time.perf_counter()`.
- Scale the rollouts to the remaining bank. Near the limit, use cheap rules only.

**Preflop strength:**
- A fast feature score:
  - nut-flush potential (suited aces, double-suited hands),
  - high pairs,
  - connectivity and straight potential,
  - penalties for trips or quads in hand, dangling cards, and non-nut flush potential.
- Calibrate the weights with a quick offline Monte Carlo on a few thousand hands.
- Also compute a percentile for EVERY hand, so ranges can be expressed as "top X%". Store this as a small table or a fast formula.

**Range-aware equity (the key upgrade over naive Monte Carlo bots):**
- Sample opponent hands from the RANGE their actions imply, not from all hands:
  - Use rejection sampling, or a weighted sample by preflop percentile.
  - Tighten the range with each raise. For example: a preflop pot raise means roughly the top 40%; a 3-bet means roughly the top 15%.
  - Postflop, a bet or raise means they hold a made hand or strong draw above some strength. Use a cheap check on the sampled hand.
- Make the starting widths parameters, and let Step 4 adjust them per opponent from showdown data.
- Watch the time budget: cap the rejection attempts, and fall back to plain random sampling.

**Nut awareness (PLO-specific):**
- Detect when we hold the nuts, or near-nuts, on the current board.
- Detect nut and non-nut flush draws, and straight-draw quality.
- Discount non-nut flushes and low straights when the opponent shows strength. In PLO, the second-best hand loses big pots.

**Decision rule:**
- Call when range-equity beats the pot odds by a margin.
- Value-raise when the hand is strong against their range.
- Semi-bluff strong nut draws.
- Bluff at a controlled frequency where our cards block the nuts (for example, holding the ace of the flush suit).

**Be hard to exploit:**
- Use both raise sizes with both value hands and bluffs, so our size doesn't reveal our strength.
- Choose randomly between close options, using a non-deterministic RNG.
- Never become a pure function of our cards that an opponent model could learn.

**Test:**
- `python test_agent.py` passes.
- 1,000-deal matches vs the baselines: 0 errors and under 4 s of bank per half.

**Then tell the user to commit, push and SUBMIT NOW** (team registration must be done first). This is the safety net.

## STEP 3 — Realistic opponent zoo (≈25 min)

Build bots that mimic the real field, all speaking the judge protocol:
- **a) Naive Monte Carlo:** equity vs a random hand against pot odds. Raises above a fixed equity threshold, with ~100 rollouts.
- **b) Tight Monte Carlo and loose Monte Carlo:** (a) with different thresholds.
- **c) Monte Carlo with simple tracking:** (a) that loosens or tightens based on our aggression.
- **d) Heavy Monte Carlo:** (a) with large rollout counts, so it may burn its time bank late in a half. Real Monte Carlo bots may do this too.
- **e) "Neural-style" stand-ins for under-trained deep learning bots:**
  - a deterministic version of our own Step 2 bot that always takes its top action,
  - an over-aggressive variant,
  - an over-passive variant.
- **f) A frozen copy of our Step 2 bot.**

## STEP 4 — Opponent profiling and exploitation (≈60 min)

Track per opponent, cheaply (O(1) per request) in `memory`:
- **Frequencies:**
  - fold rate vs RAISE_HALF and vs RAISE_POT, per street,
  - call rate when facing a bet,
  - raise and re-raise frequency, VPIP and preflop raise rate.
- **Showdown evidence:** how strong their hand was, relative to the board, when they bet, raised, or just called (`opp_hole` is gold).
- **Determinism:** whether they always respond the same way in matching spots.
- **Time-out signs:** a sudden switch to check/fold behavior late in the half.

**Exploits for Monte Carlo–equity bots** (detected by a pot-odds pattern: their calling depends on bet size, and their raises are almost always strong at showdown):
- They over-call against raising ranges, because equity vs random overstates their hand. Value-bet thinner and bet bigger, and cut bluffs against them.
- Their raises are mostly value. Fold medium hands, especially non-nut ones, when they raise pot, and widen our folding as they raise more.
- Their checks mean weakness. Bet or stab often after they check.
- They fold whenever their random-hand equity is below the pot odds. Bluff with RAISE_POT on boards where most hands miss, since a pot-sized bet demands the most equity to call.
- They ignore nuttiness. Win big pots with nut hands, and avoid paying off with second-best hands.

**Exploits for neural or under-trained bots** (detected by statistical deviation from balanced play):
- **Over-folder** (folds to raises well above a balanced rate): raise and bluff much more.
- **Over-caller** (calls far too often): value only, bigger sizes, no bluffs.
- **Over-aggressor** (bets and raises far too often): call down wider, trap with strong hands, re-raise for value.
- **Over-passive** (rarely bets): steal pots, fold to its rare raises unless we're strong.
- **Deterministic or size-revealing:** read their range from their action or size, and narrow our range-aware equity accordingly.

**If an opponent burns its time bank and starts auto check/folding:** raise every hand.

**Safe exploitation:**
- Blend from the Step 2 base toward the exploit as evidence accumulates: start after about 50–100 relevant observations, and ramp up gradually.
- If an exploit's results turn negative, revert to the base strategy.
- Against an opponent we can't classify, or a balanced one, play the Step 2 base unchanged.

**Feed showdown data back into the range model:** adjust the "top X%" widths from Step 2 to match what this opponent actually shows down.

**Test:** the bot must beat every zoo bot by more than the frozen Step 2 version does, never lose a match, and stay within the time and error limits.

## STEP 5 — Limit losses against stronger bots (≈20 min)

- Against an opponent we can't exploit, play the balanced, randomized base and avoid marginal high-variance spots that are negative EV. The aim is to keep the match a chop.
- Never let an exploit run unchecked. The revert in Step 4 is the safeguard.
- Run the frozen Step 2 bot vs the current bot on several seeds. The current bot must not lose. If it does, the exploitation layer is leaking, so tighten the ramp-up thresholds.

## STEP 6 — Tune (≈20 min, if time allows)

- Quickly sweep the key thresholds against the zoo across several seeds: range widths, raise-equity margins, bluff frequency and exploit ramp-up speed.
- Keep a change only if the bot wins at least as many matches, loses none, and its bb/deal doesn't drop.

## STEP 7 — Final checks and resubmit (≈20 min)

- **Fuzz** `choose_action` with weird states: all-in, the raise cap reached, `to_call` larger than the stack, empty history. Zero exceptions.
- **Final timing:** confirm a full 1,000-deal match uses under 4 s of bank per half, even against the most aggressive zoo bot (more decisions per hand).
- **Release:**
  - `test_agent.py` passes.
  - `requirements.txt` is correct.
  - All files are committed and pushed.
- **Print the submission steps:**
  - The same team name as before.
  - The public repo URL.
  - The full 40-character SHA from `git rev-parse HEAD`.

---

## Start now

Begin with STEP 1. If time runs short at any point, stop adding features and go straight to STEP 7 with the last version that passed its tests.
