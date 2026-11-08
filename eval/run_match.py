"""Duplicate match runner (in-process stand-in for the judge).

python eval/run_match.py A B --deals 1000 --seeds 1,2,3 [--jobs 8]
Bot names are defined in eval/roster.py.
"""
import argparse
import math
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from game import BB, Action, make_deals, play_hand  # noqa: E402

BANK, MAX_REQ, MAX_ERR = 8.0, 1.0, 5


class Player:
    def __init__(self, bot):
        self.bot, self.mem, self.bank, self.errors, self.max_req = bot, {}, BANK, 0, 0.0

    def _charge(self, dt):
        self.bank -= dt
        self.max_req = max(self.max_req, dt)
        if dt > MAX_REQ:
            self.errors += 1

    def new_half(self):
        self.mem, self.bank, self.errors, self.max_req = {}, BANK, 0, 0.0
        t = time.perf_counter()
        self.bot.new_half(self.mem)
        self._charge(time.perf_counter() - t)

    def act(self, obs):
        if self.bank <= 0:
            return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
        t = time.perf_counter()
        try:
            a = self.bot.choose_action(obs, self.mem)
            bad = a not in obs.legal_actions
        except Exception:
            a, bad = Action.CHECK_CALL, True
        self._charge(time.perf_counter() - t)
        if bad:
            self.errors += 1
            return Action.CHECK_CALL
        return a

    def end(self, res):
        t = time.perf_counter()
        try:
            self.bot.on_hand_end(res, self.mem)
        except Exception:
            self.errors += 1
        self._charge(time.perf_counter() - t)


def play_half(pa, pb, deals, a_seat):
    """Returns A's net chips per deal. Seat 0 has the button on even hands."""
    pa.new_half()
    pb.new_half()
    players = [pb, pa] if a_seat else [pa, pb]
    nets = []
    for i, d in enumerate(deals):
        n, res = play_hand(d, i % 2, lambda s, o: players[s].act(o), i)
        players[0].end(res[0])
        players[1].end(res[1])
        nets.append(n[a_seat])
    return nets


def run_match(name_a, name_b, deals, seed):
    from eval.roster import make_bot
    ds = make_deals(deals, seed)
    pa, pb = Player(make_bot(name_a)), Player(make_bot(name_b))
    h1 = play_half(pa, pb, ds, 0)
    a1, b1 = (BANK - pa.bank, pa.errors), (BANK - pb.bank, pb.errors)
    h2 = play_half(pa, pb, ds, 1)
    a2, b2 = (BANK - pa.bank, pa.errors), (BANK - pb.bank, pb.errors)
    vals = [(x + y) / BB for x, y in zip(h1, h2)]
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
    hw = 1.96 * sd / math.sqrt(len(vals))
    return dict(a=name_a, b=name_b, seed=seed, mean=mean, lo=mean - hw, hi=mean + hw,
                bank_a=max(a1[0], a2[0]), bank_b=max(b1[0], b2[0]),
                err_a=max(a1[1], a2[1]), err_b=max(b1[1], b2[1]))


def verdict(r):
    if r["err_a"] >= MAX_ERR:
        return "LOSS(forfeit)"
    if r["err_b"] >= MAX_ERR:
        return "WIN(forfeit)"
    return "WIN" if r["lo"] > 0 else "LOSS" if r["hi"] < 0 else "CHOP"


def _job(args):
    return run_match(*args)


def run_many(jobs_list, jobs):
    if jobs <= 1:
        return [_job(j) for j in jobs_list]
    with mp.Pool(jobs) as pool:
        return pool.map(_job, jobs_list, chunksize=1)


def line(r):
    return ("RESULT %-14s vs %-14s seed=%-4d %-6s bb/deal=%+7.3f CI[%+.3f,%+.3f] "
            "bank=%.2fs/%.2fs err=%d/%d" % (r["a"], r["b"], r["seed"], verdict(r), r["mean"],
                                            r["lo"], r["hi"], r["bank_a"], r["bank_b"],
                                            r["err_a"], r["err_b"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    for r in run_many([(args.a, args.b, args.deals, s) for s in seeds], args.jobs):
        print(line(r))
