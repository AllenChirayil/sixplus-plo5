"""Fit preflop feature weights to equity vs a random hand; write percentile table."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evaluator import score_batch  # noqa: E402
from bot.preflop import features  # noqa: E402


def main(hands=8000, rollouts=120, seed=7):
    rng = np.random.default_rng(seed)
    H = rng.random((hands, 36)).argsort(1)[:, :5]
    eq = np.zeros(hands)
    for i in range(hands):
        deck = np.setdiff1d(np.arange(36), H[i])
        p = rng.random((rollouts, 31)).argsort(1)
        c = deck[p[:, :10]]
        board, opp = c[:, :5], c[:, 5:10]
        a, b = score_batch(H[i], board), score_batch(opp, board)
        eq[i] = (a > b).mean() + 0.5 * (a == b).mean()
    X = features(H)
    w = np.linalg.solve(np.dot(X.T, X) + 1e-3 * np.eye(X.shape[1]), np.dot(X.T, eq))
    pred = (X * w).sum(1)
    r2 = 1 - ((eq - pred) ** 2).sum() / ((eq - eq.mean()) ** 2).sum()
    sample = rng.random((60000, 36)).argsort(1)[:, :5]
    q = np.quantile((features(sample) * w).sum(1), np.linspace(0, 1, 501))
    out = {"weights": [round(x, 5) for x in w.tolist()],
           "quantiles": [round(x, 5) for x in q.tolist()]}
    with open(os.path.join(ROOT, "bot", "preflop_weights.json"), "w") as f:
        json.dump(out, f)
    print("r2", round(r2, 3), "eq mean", round(eq.mean(), 3), "weights", out["weights"])


if __name__ == "__main__":
    main()
