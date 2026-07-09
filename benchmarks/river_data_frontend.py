"""River real-data front-end: the ticket-gate experiment.

`river_frontend.py` established the lift on simulated contamination. This
harness repeats the question on river's own regression datasets with
river's progressive-validation protocol (predict, score, then learn):
does the Laplace front-end beat river's recommended standardization
pipeline where a river user would actually stand?

Datasets (numeric features only, identical treatment for every condition;
strings/datetimes dropped symmetrically — disclosed, not hidden):

    TrumpApproval      1,001 samples, 6 numeric features
    ChickWeights         578 samples, 3 numeric features
    AirlinePassengers    144 samples, 1 numeric feature
    Bikes             20,000 samples (capped from 182,470 — logged), 5
                       numeric weather features, stations interleaved

Variants:
    asis     the dataset untouched (does real-world dirt alone pay for
             the front-end, or does the clean toll win?)
    spikes   2% measurement spikes, 6-10 full-sample sigmas, injected on
             numeric FEATURES only (never the target; scoring stays
             against the real y), 10 seeds — the insurance claim on real
             backgrounds, as in fred_anomaly.py

Learners: river LinearRegression and HoeffdingTreeRegressor at defaults
(KNN excluded: established counterexample, see RESULTS.md section 6).
Conditions per learner:

    std     TargetStandardScaler(StandardScaler | learner) on raw numeric
            features — river's existing pipeline
    lapin   same pipeline, features replaced by per-stream Laplace body
            representations (predictive mean + parade z), target body's
            mean and surprise appended
    sand    StandardScaler | learner predicting the target's parade z,
            plug-in map back through the target body's predictive CDF

A body-alone reference column (the target's Laplace predictive mean,
no regression, no features) is scored alongside: it attributes each win
to the body vs the regression on top of it.

Metric: MAE (river's convention) and MSE against the true target,
progressive validation, burn-in max(30, min(300, n/10)). Resumable jsonl.

Usage:
    python benchmarks/river_data_frontend.py --workers 6
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
import zlib
import random as _random
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

from regression_frontend import inverse_cdf_grid, _norm_cdf  # noqa: E402

DATASETS = ("TrumpApproval", "ChickWeights", "AirlinePassengers", "Bikes")
CAP = 20000
LEARNERS = ("lin", "tree")
CONDS = ("std", "lapin", "sand")
SPIKE_SEEDS = 10


def _load(dsname):
    from river import datasets
    rows = []
    for x, y in getattr(datasets, dsname)():
        rows.append((x, float(y)))
        if len(rows) >= CAP:
            break
    numeric = [k for k, v in rows[0][0].items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return rows, numeric


def _make(learner, cond):
    from river import linear_model, preprocessing, tree
    base = {"lin": linear_model.LinearRegression,
            "tree": tree.HoeffdingTreeRegressor}[learner]()
    scaled = preprocessing.StandardScaler() | base
    if cond in ("std", "lapin"):
        return preprocessing.TargetStandardScaler(regressor=scaled)
    return scaled                                    # sand: z target


def run_one(job):
    dsname, variant, seed = job
    from timemachines import laplace

    rows, numeric = _load(dsname)
    n = len(rows)
    burn = max(30, min(300, n // 10))
    t0 = time.time()

    xs = [{k: float(x[k]) for k in numeric} for x, _ in rows]
    ys = [y for _, y in rows]
    if variant == "spikes":
        rng = _random.Random(zlib.crc32(f"{dsname}:{seed}".encode()))
        for k in numeric:
            col = [r[k] for r in xs]
            mu = sum(col) / n
            sd = math.sqrt(sum((v - mu) ** 2 for v in col) / n) or 1e-8
            for t in range(n):
                if rng.random() < 0.02:
                    xs[t][k] = col[t] + (6.0 + 4.0 * rng.random()) * sd \
                        * rng.choice((-1, 1))

    models = {(lr, c): _make(lr, c) for lr in LEARNERS for c in CONDS}
    acc = {k: {"se": 0.0, "ae": 0.0, "n": 0} for k in models}
    bacc = {"se": 0.0, "ae": 0.0, "n": 0}

    y_body, y_st = laplace(1), None
    f_bodies = {k: (laplace(1), None) for k in numeric}
    f_prev = {k: None for k in numeric}              # predictive Dist for x_t
    y_prev = None                                    # predictive Dist for y_t
    zy_prev = 0.0

    for t in range(n):
        # feature bodies consume x_t (known at prediction time)
        rep = {}
        for k in numeric:
            body, st = f_bodies[k]
            mu = f_prev[k].mean if f_prev[k] is not None else xs[t][k]
            dists, st = body(xs[t][k], st)
            f_bodies[k] = (body, st)
            z = st["z"][0]
            rep[k] = (mu, z if z is not None else 0.0)
            f_prev[k] = dists[0]
        mu_y = y_prev.mean if y_prev is not None else 0.0

        feats = {
            "std": xs[t],
            "lapin": {**{f"mu_{k}": rep[k][0] for k in numeric},
                      **{f"z_{k}": rep[k][1] for k in numeric},
                      "mu_y": mu_y, "zy": zy_prev},
            "sand": {**{f"z_{k}": rep[k][1] for k in numeric},
                     "zy": zy_prev},
        }
        qgrid = inverse_cdf_grid(y_prev) if y_prev is not None else None
        if t >= burn:
            bacc["se"] += (mu_y - ys[t]) ** 2
            bacc["ae"] += abs(mu_y - ys[t])
            bacc["n"] += 1

        preds = {}
        for (lr, c), model in models.items():
            p = model.predict_one(feats[c])
            p = p if p is not None else 0.0
            if c == "sand":
                p = qgrid(_norm_cdf(max(-7.0, min(7.0, p)))) \
                    if qgrid is not None else 0.0
            preds[(lr, c)] = p
            if t >= burn:
                acc[(lr, c)]["se"] += (p - ys[t]) ** 2
                acc[(lr, c)]["ae"] += abs(p - ys[t])
                acc[(lr, c)]["n"] += 1

        # learn: target body first (sand needs z of y_t), then the models
        dists, y_st = y_body(ys[t], y_st)
        zy_t = y_st["z"][0]
        for (lr, c), model in models.items():
            if c == "sand":
                if zy_t is not None:
                    model.learn_one(feats[c], zy_t)
            else:
                model.learn_one(feats[c], ys[t])
        y_prev = dists[0]
        zy_prev = zy_t if zy_t is not None else 0.0

    res = {"ds": dsname, "variant": variant, "seed": seed,
           "n": n, "burn": burn}
    for (lr, c), a in acc.items():
        res[f"{lr}_{c}"] = {"mae": a["ae"] / a["n"], "mse": a["se"] / a["n"]}
    res["body"] = {"mae": bacc["ae"] / bacc["n"], "mse": bacc["se"] / bacc["n"]}
    res["seconds"] = round(time.time() - t0, 1)
    return res


def summarize(results):
    print(f"\n=== river real data, MAE progressive validation "
          f"(n={len(results)} rows) ===")
    for variant in ("asis", "spikes"):
        rows_v = [r for r in results if r["variant"] == variant]
        if not rows_v:
            continue
        print(f"\n--- {variant} ---")
        for lr in LEARNERS:
            print(f"{lr}: {'dataset':18s} "
                  + " ".join(f"{c:>9s}" for c in CONDS)
                  + f" {'body':>9s} {'lapin>std':>10s}")
            for ds in DATASETS:
                rows = [r for r in rows_v if r["ds"] == ds]
                if not rows:
                    continue
                line = f"    {ds:18s} "
                for c in CONDS:
                    vals = sorted(r[f"{lr}_{c}"]["mae"] for r in rows)
                    line += f"{vals[len(vals) // 2]:9.3f} "
                bv = sorted(r["body"]["mae"] for r in rows)
                line += f"{bv[len(bv) // 2]:9.3f} "
                w = sum(1 for r in rows
                        if r[f"{lr}_lapin"]["mae"] < r[f"{lr}_std"]["mae"])
                line += f" {w:>6d}/{len(rows):<3d}"
                print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE,
                                         "river_data_frontend_results.jsonl"))
    args = ap.parse_args()

    jobs = [(ds, "asis", 0) for ds in DATASETS]
    jobs += [(ds, "spikes", s) for ds in DATASETS for s in range(SPIKE_SEEDS)]

    results = []
    if os.path.exists(args.out):
        with open(args.out) as fh:
            results = [json.loads(line) for line in fh if line.strip()]
        done = {(r["ds"], r["variant"], r["seed"]) for r in results}
        jobs = [j for j in jobs if j not in done]
        print(f"resuming: {len(results)} done in {args.out}, "
              f"{len(jobs)} to go", flush=True)

    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, jobs)):
            results.append(res)
            with open(args.out, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            print(f"[{i + 1}/{len(jobs)}] {res['ds']:18s} "
                  f"{res['variant']:6s} seed {res['seed']:<2d} "
                  f"{res['seconds']:6.1f}s  "
                  f"lin std {res['lin_std']['mae']:8.3f}  "
                  f"lin lapin {res['lin_lapin']['mae']:8.3f}", flush=True)

    summarize(results)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
