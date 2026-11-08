"""agent vs the whole zoo over several seeds. python eval/gauntlet.py [--bot agent] [--deals N] [--seeds 1,2]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.run_match import line, run_many, verdict  # noqa: E402
from eval.zoo import ZOO  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="agent")
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--seeds", default="11,12")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    names = [n for n in ZOO if n != a.bot and (not a.only or n in a.only.split(","))]
    rs = run_many([(a.bot, n, a.deals, int(s)) for n in names for s in a.seeds.split(",")], a.jobs)
    tally = {"WIN": 0, "CHOP": 0, "LOSS": 0}
    for r in rs:
        print(line(r))
        tally[verdict(r).split("(")[0]] += 1
    print("TOTAL", tally, "mean bb/deal %.3f" % (sum(r["mean"] for r in rs) / len(rs)))
