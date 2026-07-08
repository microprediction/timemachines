"""River front-end: the Laplace sandwich vs river's standardization pipeline.

Section 6 (`regression_frontend.py`) held the learner fixed (our RLS) and
varied coordinates. This harness flips it: the learners are third-party —
river's, the reference streaming-ML library — and the baseline is river's
own recommended preprocessing, `StandardScaler` on features plus
`TargetStandardScaler` on the target. The question a river ticket would
need answered: does a Laplace front-end lift river's learners over river's
existing pipeline, on the same streams?

Same generator, scenarios and metric as regression_frontend.py (excess MSE
vs the clean conditional mean). Learners: LinearRegression (SGD),
HoeffdingTreeRegressor, KNNRegressor, each at river defaults. Conditions:

    raw     bare learner on {y_{t-1}, x_{t-1}}
    std     TargetStandardScaler(StandardScaler | learner), same features
            (river's existing standardization pipeline)
    lapin   the SAME std pipeline, features replaced by the Laplace bodies'
            representation {mu_y, z_y, mu_x, z_x} — the front-end composes
            with, not replaces, their pipeline
    sand    StandardScaler | learner predicting z_y from {z_y, z_x},
            mapped back through the y-body's predictive CDF. Plug-in
            point map (river learners are point predictors, so no
            quadrature; this is the median-flavoured map, biased for the
            mean under skew — disclosed, not corrected)

Strictly causal, prequential (predict, score, then learn). Resumable jsonl.

Usage:
    python benchmarks/river_frontend.py --seeds 30 --workers 6
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
    N, BURN, SCENARIOS, make_series, body_pass, inverse_cdf_grid, _norm_cdf)

LEARNERS = ("lin", "tree", "knn")
CONDS = ("raw", "std", "lapin", "sand")


def _make(learner, cond):
    from river import linear_model, preprocessing, tree, neighbors
    base = {"lin": linear_model.LinearRegression,
            "tree": tree.HoeffdingTreeRegressor,
            "knn": neighbors.KNNRegressor}[learner]()
    if cond == "raw":
        return base
    scaled = preprocessing.StandardScaler() | base
    if cond in ("std", "lapin"):
        return preprocessing.TargetStandardScaler(regressor=scaled)
    return scaled                                    # sand: z target, unit scale


def run_one(job):
    scenario, seed = job
    obs_x, obs_y, m, y_clean, tau = make_series(scenario, seed)
    t0 = time.time()

    zy, ypred = body_pass(obs_y)
    zx, xpred = body_pass(obs_x)

    models = {(lr, c): _make(lr, c) for lr in LEARNERS for c in CONDS}
    acc = {k: {"se": 0.0, "ae": 0.0, "n": 0} for k in models}

    for t in range(2, N):
        zy1 = zy[t - 1] if zy[t - 1] is not None else 0.0
        zx1 = zx[t - 1] if zx[t - 1] is not None else 0.0
        feats = {
            "raw": {"y1": obs_y[t - 1], "x1": obs_x[t - 1]},
            "std": {"y1": obs_y[t - 1], "x1": obs_x[t - 1]},
            "lapin": {"mu_y": ypred[t].mean, "zy": zy1,
                      "mu_x": xpred[t].mean, "zx": zx1},
            "sand": {"zy": zy1, "zx": zx1},
        }
        qgrid = inverse_cdf_grid(ypred[t])

        for (lr, c), model in models.items():
            p = model.predict_one(feats[c])
            p = p if p is not None else 0.0
            if c == "sand":
                p = qgrid(_norm_cdf(max(-7.0, min(7.0, p))))
            if t >= BURN:
                acc[(lr, c)]["se"] += (p - m[t]) ** 2
                acc[(lr, c)]["ae"] += abs(p - m[t])
                acc[(lr, c)]["n"] += 1
            if c == "sand":
                if zy[t] is not None:
                    model.learn_one(feats[c], zy[t])
            else:
                model.learn_one(feats[c], obs_y[t])

    res = {"scenario": scenario, "seed": seed, "n": N}
    for (lr, c), a in acc.items():
        res[f"{lr}_{c}"] = {"mse": a["se"] / a["n"], "mae": a["ae"] / a["n"]}
    res["seconds"] = round(time.time() - t0, 1)
    return res


def summarize(results):
    print(f"\n=== river front-end, excess MSE vs conditional mean "
          f"(median over seeds; n={len(results)} rows) ===")
    for lr in LEARNERS:
        print(f"\n--- learner: {lr} ---")
        print(f"{'scenario':12s} " + " ".join(f"{c:>9s}" for c in CONDS)
              + f" {'lapin>std':>10s} {'sand>std':>9s}")
        for sc in SCENARIOS:
            rows = [r for r in results if r["scenario"] == sc]
            if not rows:
                continue
            line = f"{sc:12s} "
            for c in CONDS:
                vals = sorted(r[f"{lr}_{c}"]["mse"] for r in rows)
                line += f"{vals[len(vals) // 2]:9.3f} "
            w1 = sum(1 for r in rows
                     if r[f"{lr}_lapin"]["mse"] < r[f"{lr}_std"]["mse"])
            w2 = sum(1 for r in rows
                     if r[f"{lr}_sand"]["mse"] < r[f"{lr}_std"]["mse"])
            line += f" {w1:>6d}/{len(rows):<3d} {w2:>5d}/{len(rows):<3d}"
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--scenarios", type=str, default="")
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, "river_frontend_results.jsonl"))
    args = ap.parse_args()

    scens = [s for s in args.scenarios.split(",") if s] or list(SCENARIOS)
    jobs = [(sc, sd) for sc in scens for sd in range(args.seeds)]

    results = []
    if os.path.exists(args.out):
        with open(args.out) as fh:
            results = [json.loads(line) for line in fh if line.strip()]
        done = {(r["scenario"], r["seed"]) for r in results}
        jobs = [j for j in jobs if j not in done]
        print(f"resuming: {len(results)} done in {args.out}, "
              f"{len(jobs)} to go", flush=True)

    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, jobs)):
            results.append(res)
            with open(args.out, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            print(f"[{i + 1}/{len(jobs)}] {res['scenario']:12s} "
                  f"seed {res['seed']:<3d} {res['seconds']:5.1f}s  "
                  f"lin std {res['lin_std']['mse']:7.3f}  "
                  f"lin lapin {res['lin_lapin']['mse']:7.3f}", flush=True)

    summarize(results)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
