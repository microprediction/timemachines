"""Ablation on river's datasets, dogfooding the published ice-skaters.

Decomposes the front-end's value on real data using the actual package a
river user would install (`pip install ice-skaters`), same datasets,
injection and protocol as river_data_frontend.py:

    std      river's StandardScaler/TargetStandardScaler pipeline
    lt       raw features + LaplaceTarget only (the target pair alone)
    mu       LaplaceFeatures(emit="mu") + LaplaceTarget
    z        LaplaceFeatures(emit="z") + LaplaceTarget
    both     LaplaceFeatures(emit="both") + LaplaceTarget (the recipe)
    both_nt  LaplaceFeatures(emit="both") without LaplaceTarget

Usage:
    python benchmarks/ablation_river.py --workers 6
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

from river_data_frontend import _load, DATASETS, SPIKE_SEEDS  # noqa: E402

LEARNERS = ("lin", "tree")
CONDS = ("std", "lt", "mu", "z", "both", "both_nt")


def _make(learner, cond):
    from river import linear_model, preprocessing, tree
    from ice_skaters import LaplaceFeatures, LaplaceTarget
    base = {"lin": linear_model.LinearRegression,
            "tree": tree.HoeffdingTreeRegressor}[learner]()
    inner = preprocessing.StandardScaler() | base
    if cond in ("mu", "z", "both", "both_nt"):
        emit = cond if cond in ("mu", "z") else "both"
        inner = LaplaceFeatures(emit=emit) | inner
    model = preprocessing.TargetStandardScaler(regressor=inner)
    if cond in ("lt", "mu", "z", "both"):
        model = LaplaceTarget(regressor=model)
    return model


def run_one(job):
    dsname, variant, seed = job
    rows, numeric = _load(dsname)
    n = len(rows)
    burn = max(30, min(300, n // 10))
    t0 = time.time()

    xs = [{k: float(x[k]) for k in numeric} for x, _ in rows]
    ys = [float(y) for _, y in rows]
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
    acc = {k: {"ae": 0.0, "n": 0} for k in models}
    for t in range(n):
        for key, model in models.items():
            p = model.predict_one(xs[t])
            p = p if p is not None else 0.0
            if t >= burn:
                acc[key]["ae"] += abs(p - ys[t])
                acc[key]["n"] += 1
            model.learn_one(xs[t], ys[t])

    res = {"ds": dsname, "variant": variant, "seed": seed}
    for (lr, c), a in acc.items():
        res[f"{lr}_{c}"] = a["ae"] / a["n"]
    res["seconds"] = round(time.time() - t0, 1)
    return res


def summarize(results):
    print(f"\n=== river ablation, MAE (n={len(results)} rows) ===")
    for variant in ("asis", "spikes"):
        rows_v = [r for r in results if r["variant"] == variant]
        if not rows_v:
            continue
        print(f"\n--- {variant} ---")
        for lr in LEARNERS:
            print(f"{lr}: {'dataset':18s} "
                  + " ".join(f"{c:>8s}" for c in CONDS))
            for ds in DATASETS:
                rows = [r for r in rows_v if r["ds"] == ds]
                if not rows:
                    continue
                line = f"    {ds:18s} "
                for c in CONDS:
                    vals = sorted(r[f"{lr}_{c}"] for r in rows)
                    line += f"{vals[len(vals) // 2]:8.3f} "
                print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, "ablation_river_results.jsonl"))
    args = ap.parse_args()

    jobs = [(ds, "asis", 0) for ds in DATASETS]
    jobs += [(ds, "spikes", s) for ds in DATASETS for s in range(SPIKE_SEEDS)]
    results = []
    if os.path.exists(args.out):
        with open(args.out) as fh:
            results = [json.loads(line) for line in fh if line.strip()]
        done = {(r["ds"], r["variant"], r["seed"]) for r in results}
        jobs = [j for j in jobs if j not in done]
        print(f"resuming: {len(results)} done, {len(jobs)} to go", flush=True)

    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, jobs)):
            results.append(res)
            with open(args.out, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            print(f"[{i + 1}/{len(jobs)}] {res['ds']:18s} {res['variant']:6s} "
                  f"seed {res['seed']:<2d} {res['seconds']:6.1f}s", flush=True)

    summarize(results)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
