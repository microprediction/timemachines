"""Ablation: which of the two scalars carries the value, mu or z?

The front-end replaces each stream with (predictive mean, surprise z).
This harness decomposes the pair on the section-6 simulation: same
generator, scenarios, seeds and RLS learner as regression_frontend.py,
feature sets varied.

    raw      [1, y_{t-1}, x_{t-1}]                 (control)
    mu       [1, mu_y, mu_x]                        means only
    z        [1, z_y, z_x]                          surprises only
    both     [1, mu_y, z_y, mu_x, z_x]              the recipe (= zin)
    both_nt  [1, y_{t-1}, mu_x, z_x]                inputs fixed, target
                                                    pair replaced by the
                                                    raw lag

Metric: excess MSE vs the clean conditional mean, median over seeds.

Usage:
    python benchmarks/ablation_frontend.py --seeds 30 --workers 6
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

from regression_frontend import (  # noqa: E402
    N, BURN, SCENARIOS, make_series, body_pass)

CONDS = ("raw", "mu", "z", "both", "both_nt")


class RLS:
    def __init__(self, d, lam=0.999, p0=100.0):
        self.d, self.lam = d, lam
        self.w = [0.0] * d
        self.P = [[(p0 if i == j else 0.0) for j in range(d)] for i in range(d)]

    def predict(self, phi):
        return sum(w * p for w, p in zip(self.w, phi))

    def update(self, phi, target):
        import math
        err = target - self.predict(phi)
        Pp = [sum(self.P[i][j] * phi[j] for j in range(self.d))
              for i in range(self.d)]
        denom = self.lam + sum(p * pp for p, pp in zip(phi, Pp))
        g = [pp / denom for pp in Pp]
        for i in range(self.d):
            self.w[i] += g[i] * err
        for i in range(self.d):
            for j in range(self.d):
                self.P[i][j] = (self.P[i][j] - g[i] * Pp[j]) / self.lam
        tr = sum(self.P[i][i] for i in range(self.d))
        if not math.isfinite(tr) or tr > 1e7:
            self.__init__(self.d, self.lam)


def run_one(job):
    scenario, seed = job
    obs_x, obs_y, m, y_clean, tau = make_series(scenario, seed)
    t0 = time.time()
    zy, ypred = body_pass(obs_y)
    zx, xpred = body_pass(obs_x)

    dims = {"raw": 3, "mu": 3, "z": 3, "both": 5, "both_nt": 4}
    learners = {c: RLS(d) for c, d in dims.items()}
    acc = {c: {"se": 0.0, "n": 0} for c in CONDS}

    for t in range(2, N):
        zy1 = zy[t - 1] if zy[t - 1] is not None else 0.0
        zx1 = zx[t - 1] if zx[t - 1] is not None else 0.0
        mu_y, mu_x = ypred[t].mean, xpred[t].mean
        feats = {
            "raw": [1.0, obs_y[t - 1], obs_x[t - 1]],
            "mu": [1.0, mu_y, mu_x],
            "z": [1.0, zy1, zx1],
            "both": [1.0, mu_y, zy1, mu_x, zx1],
            "both_nt": [1.0, obs_y[t - 1], mu_x, zx1],
        }
        for c in CONDS:
            p = learners[c].predict(feats[c])
            if t >= BURN:
                acc[c]["se"] += (p - m[t]) ** 2
                acc[c]["n"] += 1
            learners[c].update(feats[c], obs_y[t])

    res = {"scenario": scenario, "seed": seed}
    for c in CONDS:
        res[c] = acc[c]["se"] / acc[c]["n"]
    res["seconds"] = round(time.time() - t0, 1)
    return res


def summarize(results):
    print(f"\n=== mu-vs-z ablation, excess MSE vs conditional mean "
          f"(median over seeds; n={len(results)} rows) ===")
    print(f"{'scenario':12s} " + " ".join(f"{c:>9s}" for c in CONDS))
    for sc in SCENARIOS:
        rows = [r for r in results if r["scenario"] == sc]
        if not rows:
            continue
        line = f"{sc:12s} "
        for c in CONDS:
            vals = sorted(r[c] for r in rows)
            line += f"{vals[len(vals) // 2]:9.4f} "
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, "ablation_frontend_results.jsonl"))
    args = ap.parse_args()

    jobs = [(sc, sd) for sc in SCENARIOS for sd in range(args.seeds)]
    results = []
    if os.path.exists(args.out):
        with open(args.out) as fh:
            results = [json.loads(line) for line in fh if line.strip()]
        done = {(r["scenario"], r["seed"]) for r in results}
        jobs = [j for j in jobs if j not in done]
        print(f"resuming: {len(results)} done, {len(jobs)} to go", flush=True)

    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, jobs)):
            results.append(res)
            with open(args.out, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            print(f"[{i + 1}/{len(jobs)}] {res['scenario']:12s} "
                  f"seed {res['seed']:<3d}", flush=True)

    summarize(results)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
