"""The calibrated Granger test, round two: the harsher grid.

Round one (granger_study.py) established size everywhere and power
statistically indistinguishable from the focused classical test, on a
design kind to the classical side. This grid is where the implicit-GLS
mechanism predicts real separation, and it carries the proper literature
opponent: a Cheung-Ng-style cross-correlation of AR-filtered,
EWMA-volatility-standardized residuals (both sides standardized, per
their design), so the target-only calibration rule is tested against
the both-sides choice.

Nulls (x never enters y's mean):
    n_harsh   near-integrated GARCH on y (alpha .25, beta .72)
    n_regime  regime-switching volatility (sigma 1 <-> 4, mean dwell 150)
    n_t3g     t(3) innovations riding harsh GARCH
    n_volc    x drives y's VARIANCE but not its mean (the classic trap
              for mean-causality tests)
Alternatives (mean causality):
    a_harsh   beta=0.08 at lag 1 under harsh GARCH
    a_weak    beta=0.04 at lag 1 under harsh GARCH
    a_t3g     beta=0.08 at lag 1 under t(3)+GARCH
    a_lag3    beta=0.08 at lag 3 under harsh GARCH (misspecified lag:
              the lag-scan columns are the fair readers here)

Tests:
    F      classical Granger F, x lags 1..4 (spans the lag)
    F1     focused classical, x lag 1 only
    HAC    Newey-West Wald on the x block
    CN     Cheung-Ng-style, lag 1 (AR(4)-filtered, EWMA-vol standardized
           both sides, sqrt(T) cross-correlation)
    CNM    CN scanned over lags 1..4, Bonferroni
    Z2H    calibrated target vs raw lagged x, Newey-West variance, lag 1
    Z2M    Z2H scanned over lags 1..4, Bonferroni

200 reps per cell, T=1200, rejection at nominal 5%.

Usage:
    python benchmarks/granger2_study.py --reps 200 --workers 6
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
import random as _random
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from granger_study import (  # noqa: E402
    _solve, classical_tests, classical_f1, chi2_sf, P, T)

SCENARIOS = ("n_harsh", "n_regime", "n_t3g", "n_volc",
             "a_harsh", "a_weak", "a_t3g", "a_lag3")


def simulate(scenario, seed):
    rng = _random.Random(seed)
    n = T + 200
    x = [0.0] * n
    y = [0.0] * n
    h = 0.7
    sig_regime = 1.0
    beta, lag = 0.0, 1
    if scenario == "a_harsh":
        beta = 0.08
    elif scenario == "a_weak":
        beta = 0.04
    elif scenario == "a_t3g":
        beta = 0.08
    elif scenario == "a_lag3":
        beta, lag = 0.08, 3

    def t3():
        c = sum(rng.gauss(0, 1) ** 2 for _ in range(3))
        return rng.gauss(0, 1) / math.sqrt(c / 3.0)

    e_prev = 0.0
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + rng.gauss(0, 1)
        if scenario == "n_regime":
            if rng.random() < 1.0 / 150.0:
                sig_regime = 4.0 if sig_regime == 1.0 else 1.0
            e = sig_regime * rng.gauss(0, 1)
        elif scenario == "n_volc":
            h = 0.05 + 0.20 * x[t - 1] ** 2 + 0.75 * h
            e = math.sqrt(max(h, 1e-8)) * rng.gauss(0, 1)
        else:
            h = 0.02 + 0.25 * e_prev ** 2 + 0.72 * h
            innov = t3() if scenario in ("n_t3g", "a_t3g") else rng.gauss(0, 1)
            e = math.sqrt(max(h, 1e-8)) * innov
        e_prev = e
        drive = beta * x[t - lag] if t >= lag else 0.0
        y[t] = 0.5 * y[t - 1] + drive + e
    return y[200:], x[200:]


# --------------- Cheung-Ng style: AR-filter + EWMA-vol standardize ----------

def _ar_resid(series):
    n = len(series)
    rows = range(P, n)
    d = 1 + P
    X = [[1.0] + [series[t - l] for l in range(1, P + 1)] for t in rows]
    Y = [series[t] for t in rows]
    m = len(Y)
    A = [[sum(X[i][a] * X[i][b] for i in range(m)) for b in range(d)]
         for a in range(d)]
    b = [sum(X[i][a] * Y[i] for i in range(m)) for a in range(d)]
    beta = _solve(A, b)
    return [Y[i] - sum(beta[k] * X[i][k] for k in range(d)) for i in range(m)]


def _ewma_standardize(resid, lam=0.94):
    out = []
    v = sum(r * r for r in resid[:20]) / 20.0 or 1e-12
    for r in resid:
        out.append(r / math.sqrt(max(v, 1e-12)))
        v = lam * v + (1 - lam) * r * r
    return out

def cn_stats(y, x):
    """Cheung-Ng style: cross-correlations of both-sides standardized
    residuals at lags 1..P; returns list of sqrt(T)*ccf values."""
    ey = _ewma_standardize(_ar_resid(y))
    ex = _ewma_standardize(_ar_resid(x))
    out = []
    for lag in range(1, P + 1):
        a = ey[lag:]
        b = ex[:-lag]
        n = len(a)
        ma = sum(a) / n
        mb = sum(b) / n
        ca = [v - ma for v in a]
        cb = [v - mb for v in b]
        den = math.sqrt(sum(v * v for v in ca) * sum(v * v for v in cb)) or 1e-30
        out.append(math.sqrt(n) * sum(p * q for p, q in zip(ca, cb)) / den)
    return out


# --------------- calibrated: target-side z, raw x, NW variance --------------

def z2_stats(y, x):
    from timemachines import laplace

    f, st = laplace(1), None
    zy = []
    for v in y:
        dists, st = f(v, st)
        z = st["z"][0]
        zy.append(0.0 if z is None else z)

    out = []
    for lag in range(1, P + 1):
        a = zy[lag:]
        b = x[:-lag]
        n = len(a)
        ma = sum(a) / n
        mb = sum(b) / n
        ca = [v - ma for v in a]
        cb = [v - mb for v in b]
        prod = [p * q for p, q in zip(ca, cb)]
        mpr = sum(prod) / n
        cprod = [v - mpr for v in prod]
        var = sum(v * v for v in cprod) / n
        for L in range(1, 5):
            w = 1.0 - L / 5.0
            var += 2.0 * w * sum(cprod[i] * cprod[i - L]
                                 for i in range(L, n)) / n
        se = math.sqrt(max(var * n, 1e-30))
        out.append(sum(prod) / se)
    return out


def run_one(job):
    scenario, seed = job
    y, x = simulate(scenario, seed)
    t0 = time.time()
    chi_F, wald = classical_tests(y, x)
    f1 = classical_f1(y, x)
    cn = cn_stats(y, x)
    z2 = z2_stats(y, x)

    def p2(g):
        return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(g) / math.sqrt(2.0))))

    res = {"scenario": scenario, "seed": seed,
           "rej_F": chi2_sf(chi_F, P) < 0.05,
           "rej_HAC": chi2_sf(wald, P) < 0.05,
           "rej_F1": chi2_sf(f1, 1) < 0.05,
           "rej_CN": p2(cn[0]) < 0.05,
           "rej_CNM": min(p2(g) for g in cn) < 0.05 / P,
           "rej_Z2H": p2(z2[0]) < 0.05,
           "rej_Z2M": min(p2(g) for g in z2) < 0.05 / P,
           "seconds": round(time.time() - t0, 1)}
    return res


TESTS = ("rej_F", "rej_HAC", "rej_F1", "rej_CN", "rej_CNM",
         "rej_Z2H", "rej_Z2M")


def summarize(results):
    print(f"\n=== calibrated Granger, round two (n={len(results)}) ===")
    header = f"{'scenario':10s}" + "".join(f"{t[4:]:>7s}" for t in TESTS)
    print(header + "   (rejection at nominal 5%)")
    for sc in SCENARIOS:
        rows = [r for r in results if r["scenario"] == sc]
        if not rows:
            continue
        line = f"{sc:10s}"
        for t in TESTS:
            line += f"{sum(r[t] for r in rows)/len(rows):7.3f}"
        print(line + f"  ({len(rows)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, "granger2_results.jsonl"))
    args = ap.parse_args()

    jobs = [(sc, s) for sc in SCENARIOS for s in range(args.reps)]
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
            if (i + 1) % 50 == 0:
                print(f"[{i + 1}/{len(jobs)}]", flush=True)

    summarize(results)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
