"""The cheap-rollup control: EWMA (mu, z) pairs vs the Laplace ones.

Peter's question: how much of the front-end's value is the rollup SHAPE
(any expectation-and-clipped-surprise pair) versus the calibrated
forecaster specifically? An EWMA rollup is two lines and free; if it
matches, the forecaster is not earning its 900x cost here. Same
datasets, injection, seeds and protocol as river_data_frontend.py.

    std     river's pipeline, raw features            (reference)
    elt     raw features + EWMA target pair            (control for lt)
    efull   EWMA pairs for features + EWMA target pair (control for the
                                                        full recipe)

Compare against the lt / both columns of ablation_river_results.jsonl
(identical seeds). EWMA alpha 0.1 with 1/n warmup, z clipped at 7.03.

Usage:
    python benchmarks/ablation_ewma.py --workers 6
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
CONDS = ("std", "elt", "efull")
ALPHA = 0.1
ZCAP = 7.03


class _Roll:
    """Streaming EWMA mean/var per key; (mu, z) read before update."""

    def __init__(self):
        self.m, self.v, self.n = {}, {}, {}

    def pair(self, k, x, update=True):
        n = self.n.get(k, 0)
        m = self.m.get(k, x)
        v = self.v.get(k, 1.0)
        sd = math.sqrt(max(v, 1e-12))
        z = max(-ZCAP, min(ZCAP, (x - m) / sd)) if n > 3 else 0.0
        if update:
            n += 1
            a = max(ALPHA, 1.0 / n)
            d = x - m
            m2 = m + a * d
            self.m[k], self.v[k], self.n[k] = m2, (1 - a) * v + a * d * (x - m2), n
        return m, z


def _pipeline():
    from river import linear_model, preprocessing
    return preprocessing.TargetStandardScaler(
        regressor=preprocessing.StandardScaler()
        | linear_model.LinearRegression())


def _tree_pipeline():
    from river import preprocessing, tree
    return preprocessing.TargetStandardScaler(
        regressor=preprocessing.StandardScaler()
        | tree.HoeffdingTreeRegressor())


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

    makers = {"lin": _pipeline, "tree": _tree_pipeline}
    models = {(lr, c): makers[lr]() for lr in LEARNERS for c in CONDS}
    rolls = {(lr, c): _Roll() for lr in LEARNERS for c in CONDS}
    zy_last = {(lr, c): 0.0 for lr in LEARNERS for c in CONDS}
    acc = {k: {"ae": 0.0, "n": 0} for k in models}

    for t in range(n):
        for (lr, c), model in models.items():
            roll = rolls[(lr, c)]
            if c == "std":
                feats = xs[t]
            elif c == "elt":
                mu_y, _ = roll.pair("__y", ys[t], update=False)
                feats = {**xs[t], "mu_y": mu_y, "z_y": zy_last[(lr, c)]}
            else:
                feats = {"mu_y": roll.pair("__y", ys[t], update=False)[0],
                         "z_y": zy_last[(lr, c)]}
                for k, v in xs[t].items():
                    m, z = roll.pair(k, v, update=False)
                    feats["mu_" + k] = m
                    feats["z_" + k] = z
            p = model.predict_one(feats)
            p = p if p is not None else 0.0
            if t >= burn:
                acc[(lr, c)]["ae"] += abs(p - ys[t])
                acc[(lr, c)]["n"] += 1
            model.learn_one(feats, ys[t])
            if c == "efull":
                for k, v in xs[t].items():
                    roll.pair(k, v, update=True)
            if c in ("elt", "efull"):
                _, zy = roll.pair("__y", ys[t], update=True)
                zy_last[(lr, c)] = zy

    res = {"ds": dsname, "variant": variant, "seed": seed}
    for (lr, c), a in acc.items():
        res[f"{lr}_{c}"] = a["ae"] / a["n"]
    res["seconds"] = round(time.time() - t0, 1)
    return res


def summarize(results):
    print(f"\n=== EWMA rollup control, MAE (n={len(results)} rows) ===")
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
                    default=os.path.join(_HERE, "ablation_ewma_results.jsonl"))
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
