"""The sufficiency study: are five calibrated numbers typically enough?

The broadened question behind the regression front-end: across many real
cross-series regressions, is the five-number rollup

    mu_x, z_x            (the covariate's forecaster: expectation + news)
    mu_y, z_y, mu_y_prev (the target's forecaster via LaplaceTarget(mus=2))

typically within noise of the best raw-information configuration? Pairs
are FRED change-series siblings: same family (leading letters of the id,
the fred_universe rule), aligned on common dates, y regressed on the
concurrent x change.

Conditions, per learner (river LinearRegression, HoeffdingTreeRegressor):

    body    the target's Laplace forecast alone (attribution floor)
    five    the five numbers, via ice-skaters LaplaceFeatures +
            LaplaceTarget(mus=2)
    ewma5   the same five shapes from EWMA forecasters (cheap control)
    base1   raw [x_t, y_{t-1}] (the parsimonious baseline)
    ladder  raw [x_t, x lags 1..4, y lags 1..8] (the generous baseline)

Variants: asis (real dirt only) and spikes (2% measurement spikes at
6-10 full-sample sigmas on the observed x; the target and the scoring
stay untouched). MAE, burn-in max(100, n/10), resumable jsonl.

Price series (equity, fx, commodity, classified by the skaters repo's
fred_universe.asset_class title rule) are EXCLUDED: their changes are
near-martingale, so they measure noise rather than sufficiency; the
skaters paper applies the same split. Data: point TIMEMACHINES_FRED_DATA
at a FRED cache (the skaters repo carries one). Usage:

    TIMEMACHINES_FRED_DATA=~/github/skaters/benchmarks/data \\
        python benchmarks/sufficiency_study.py --pairs 150 --workers 6
"""

from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys
import time
import zlib
import random as _random
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

from fred import _load_levels, _CACHE  # noqa: E402

MIN_LEN = 1200
LEARNERS = ("lin", "tree")
CONDS = ("body", "five", "ewma5", "base1", "ladder")
ZCAP = 7.03
EW_ALPHA = 0.1


def family(series_id):
    m = re.match(r"[A-Za-z]+", series_id)
    return (m.group(0)[:4] if m else series_id)


def _asset_class():
    """Load the house price rule from beside the data cache; None if absent."""
    bench = os.path.dirname(os.path.abspath(_CACHE))
    path = os.path.join(bench, "fred_universe.py")
    if not os.path.exists(path):
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("fred_universe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.asset_class


PRICE = {"equity", "fx", "commodity"}


def build_pairs(cap):
    universe = os.path.join(_CACHE, "universe_daily.json")
    meta = json.load(open(universe))
    cls = _asset_class()
    if cls is None:
        raise SystemExit("fred_universe.py not found beside the data cache; "
                         "the price exclusion needs its asset_class rule")
    ids = [u["id"] for u in meta if cls(u.get("title", "")) not in PRICE]
    cached = [s for s in ids
              if os.path.exists(os.path.join(_CACHE, f"{s}.csv"))]
    fams = {}
    for s in cached:
        fams.setdefault(family(s), []).append(s)
    # round-robin across families so no giant panel (BAML has ~200 members)
    # floods the cap; one pair per family per round
    per_fam = {f: [(m[i], m[i + 1]) for i in range(0, len(m) - 1, 2)]
               for f, m in sorted(fams.items())}
    pairs = []
    rnd = 0
    while len(pairs) < cap and any(len(v) > rnd for v in per_fam.values()):
        for f in sorted(per_fam):
            if len(per_fam[f]) > rnd and len(pairs) < cap:
                pairs.append(per_fam[f][rnd])
        rnd += 1
    return pairs


def aligned_changes(y_id, x_id):
    """Common-date level alignment, then per-series change transform."""
    ly = dict(_load_levels(y_id) or [])
    lx = dict(_load_levels(x_id) or [])
    dates = sorted(set(ly) & set(lx))
    if len(dates) < MIN_LEN + 1:
        return None, None
    vy = [ly[d] for d in dates]
    vx = [lx[d] for d in dates]

    def changes(vals):
        pos = all(v > 0 for v in vals)
        return [(math.log(b) - math.log(a)) if pos else (b - a)
                for a, b in zip(vals[:-1], vals[1:])]

    return changes(vy), changes(vx)


class _EwmaRoll:
    def __init__(self):
        self.m, self.v, self.n = {}, {}, {}
        self.mu_prev_y = 0.0

    def pair(self, k, x, update=True):
        n = self.n.get(k, 0)
        m = self.m.get(k, x)
        v = self.v.get(k, 1.0)
        sd = math.sqrt(max(v, 1e-12))
        z = max(-ZCAP, min(ZCAP, (x - m) / sd)) if n > 3 else 0.0
        if update:
            n += 1
            a = max(EW_ALPHA, 1.0 / n)
            d = x - m
            m2 = m + a * d
            self.m[k], self.v[k], self.n[k] = m2, (1 - a) * v + a * d * (x - m2), n
        return m, z


def _learner(kind):
    from river import linear_model, preprocessing, tree
    base = {"lin": linear_model.LinearRegression,
            "tree": tree.HoeffdingTreeRegressor}[kind]()
    return preprocessing.TargetStandardScaler(
        regressor=preprocessing.StandardScaler() | base)


def _five(kind):
    from river import linear_model, preprocessing, tree
    from ice_skaters import LaplaceFeatures, LaplaceTarget
    base = {"lin": linear_model.LinearRegression,
            "tree": tree.HoeffdingTreeRegressor}[kind]()
    return LaplaceTarget(
        regressor=preprocessing.TargetStandardScaler(
            regressor=LaplaceFeatures()
            | preprocessing.StandardScaler() | base),
        mus=2)


def run_one(job):
    y_id, x_id, variant = job
    ys, xs = aligned_changes(y_id, x_id)
    if ys is None:
        return None
    from timemachines import laplace

    n = len(ys)
    burn = max(100, n // 10)
    t0 = time.time()

    if variant == "spikes":
        rng = _random.Random(zlib.crc32(f"{y_id}:{x_id}".encode()))
        mu = sum(xs) / n
        sd = math.sqrt(sum((v - mu) ** 2 for v in xs) / n) or 1e-12
        xs = [v + ((6.0 + 4.0 * rng.random()) * sd * rng.choice((-1, 1))
                   if rng.random() < 0.02 else 0.0) for v in xs]

    models = {}
    for lr in LEARNERS:
        models[(lr, "five")] = _five(lr)
        models[(lr, "ewma5")] = _learner(lr)
        models[(lr, "base1")] = _learner(lr)
        models[(lr, "ladder")] = _learner(lr)
    rolls = {lr: _EwmaRoll() for lr in LEARNERS}
    ew_zy = {lr: 0.0 for lr in LEARNERS}

    f_body, st_body, pend = laplace(1), None, None
    acc = {(lr, c): {"ae": 0.0, "n": 0} for lr in LEARNERS for c in CONDS}

    for t in range(n):
        feats = {}
        feats["base1"] = {"x": xs[t], "y_1": ys[t - 1] if t >= 1 else 0.0}
        lad = {"x": xs[t]}
        for l in range(1, 5):
            lad[f"x_{l}"] = xs[t - l] if t >= l else 0.0
        for l in range(1, 9):
            lad[f"y_{l}"] = ys[t - l] if t >= l else 0.0
        feats["ladder"] = lad
        feats["five"] = {"x": xs[t]}

        body_pred = pend.mean if pend is not None else 0.0

        for lr in LEARNERS:
            roll = rolls[lr]
            mu_x, z_x = roll.pair("x", xs[t], update=False)
            mu_y, _ = roll.pair("__y", ys[t], update=False)
            ew_feats = {"mu_x": mu_x, "z_x": z_x, "mu_y": mu_y,
                        "z_y": ew_zy[lr], "mu_y_prev1": roll.mu_prev_y}
            for c in CONDS:
                if c == "body":
                    p = body_pred
                else:
                    f = ew_feats if c == "ewma5" else feats[c]
                    p = models[(lr, c)].predict_one(f)
                    p = p if p is not None else 0.0
                if t >= burn:
                    acc[(lr, c)]["ae"] += abs(p - ys[t])
                    acc[(lr, c)]["n"] += 1
            for c in ("five", "base1", "ladder"):
                models[(lr, c)].learn_one(feats[c], ys[t])
            models[(lr, "ewma5")].learn_one(ew_feats, ys[t])
            roll.pair("x", xs[t], update=True)
            roll.mu_prev_y = roll.m.get("__y", 0.0)
            _, zy = roll.pair("__y", ys[t], update=True)
            ew_zy[lr] = zy

        dists, st_body = f_body(ys[t], st_body)
        pend = dists[0]

    res = {"y": y_id, "x": x_id, "variant": variant, "n": n,
           "fam": family(y_id)}
    for (lr, c), a in acc.items():
        res[f"{lr}_{c}"] = a["ae"] / max(a["n"], 1)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def summarize(results):
    results = [r for r in results if r]
    print(f"\n=== sufficiency study (n={len(results)} pair-variants) ===")
    for variant in ("asis", "spikes"):
        rows = [r for r in results if r["variant"] == variant]
        if not rows:
            continue
        print(f"\n--- {variant} (pairs={len(rows)}) ---")
        for lr in LEARNERS:
            best_raw = [min(r[f"{lr}_base1"], r[f"{lr}_ladder"]) for r in rows]
            five = [r[f"{lr}_five"] for r in rows]
            ratios = sorted(f / max(b, 1e-15) for f, b in zip(five, best_raw))
            med = ratios[len(ratios) // 2]
            suff = sum(1 for x in ratios if x <= 1.05) / len(ratios)
            w1 = sum(1 for r in rows if r[f"{lr}_five"] < r[f"{lr}_base1"])
            wl = sum(1 for r in rows if r[f"{lr}_five"] < r[f"{lr}_ladder"])
            we = sum(1 for r in rows if r[f"{lr}_five"] < r[f"{lr}_ewma5"])
            wb = sum(1 for r in rows if r[f"{lr}_five"] < r[f"{lr}_body"])
            fams = {}
            for r, f, b in zip(rows, five, best_raw):
                fams.setdefault(r["fam"], []).append(f <= 1.05 * b)
            fw = sum(sum(v) / len(v) for v in fams.values()) / len(fams)
            print(f"  {lr}: median five/best-raw {med:.3f}; "
                  f"sufficient(<=1.05x) {100*suff:.0f}% raw / "
                  f"{100*fw:.0f}% family-weighted; wins vs base1 "
                  f"{w1}/{len(rows)}, ladder {wl}/{len(rows)}, "
                  f"ewma5 {we}/{len(rows)}, body {wb}/{len(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=150)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, "sufficiency_results.jsonl"))
    args = ap.parse_args()

    pairs = build_pairs(args.pairs)
    print(f"{len(pairs)} family pairs", flush=True)
    jobs = [(y, x, v) for (y, x) in pairs for v in ("asis", "spikes")]

    results = []
    if os.path.exists(args.out):
        with open(args.out) as fh:
            results = [json.loads(line) for line in fh if line.strip()]
        done = {(r["y"], r["x"], r["variant"]) for r in results}
        jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
        print(f"resuming: {len(results)} done, {len(jobs)} to go", flush=True)

    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, jobs)):
            if res is None:
                continue
            results.append(res)
            with open(args.out, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            print(f"[{i + 1}/{len(jobs)}] {res['y'][:14]:14s}~{res['x'][:14]:14s} "
                  f"{res['variant']:6s} {res['seconds']:6.1f}s", flush=True)

    summarize(results)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
