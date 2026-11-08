"""One-at-a-time parameter sweep. python eval/sweep.py 'bluff_bet=0.05' 'stab=0.1;bet_eq=0.58' ..."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.run_match import run_many, verdict  # noqa: E402
from eval.zoo import ZOO  # noqa: E402

NAMES = [n for n in ZOO if n != "mc_heavy"]
SEEDS = [21, 22, 23, 24]
if __name__ == "__main__":
    for v in [""] + sys.argv[1:]:
        bot = "agent|" + v if v else "agent"
        rs = run_many([(bot, n, 1000, s) for n in NAMES for s in SEEDS], 9)
        by = {n: sum(r["mean"] for r in rs if r["b"] == n) / len(SEEDS) for n in NAMES}
        vd = [verdict(r)[0] for r in rs]
        print("%-40s mean %+.3f W%d C%d L%d | %s" % (v or "BASE", sum(by.values()) / len(by),
              vd.count("W"), vd.count("C"), vd.count("L"),
              " ".join("%s %+.2f" % (k.replace("mc_", "").replace("nn_", "n"), x) for k, x in by.items())), flush=True)
