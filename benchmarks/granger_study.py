"""The calibrated Granger test: is z-space causality testing better?

Classical Granger causality tests whether lagged x improves an
autoregression of y, and its null distribution is only as good as the
AR specification: volatility clustering, heavy tails and breaks distort
the size. The calibrated alternative reduces y to its parade surprises
first: under the null (x carries nothing beyond y's own past) and a
calibrated body, z_y is iid N(0,1) by construction (the oracle
decomposition in the ice-skaters theory note), so

    G = sqrt(T) * corr(z_y[t], z_x[t-1])

is standard normal under the null with no lag selection and no
homoskedasticity assumption. This harness measures empirical size at
nominal 5% under four nulls and power under lagged causality, for three
tests:

    F      classical Granger F (OLS, y lags 1..4, x lags 1..4)
    F1     the focused classical test (x lag 1 only, one degree of
           freedom, the fair sharp opponent for Z)
    HAC    the practitioner's fix: t-test on the x coefficients with
           Newey-West standard errors (4 lags)
    Z      the calibrated test above (laplace(1) bodies on y and x)

Nulls: iid Gaussian AR(1); GARCH(1,1) volatility; t(3) innovations; a
mid-sample variance break (x independent AR(1) throughout). Alternative:
y_t += beta * x_{t-1} with GARCH volatility, beta in {0.05, 0.1, 0.2}
(x unit-scale). T=1200 after burn, 200 replications per cell.

Usage:
    python benchmarks/granger_study.py --reps 200 --workers 6
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

T = 1200
P = 4                    # lags in the classical tests
SCENARIOS = ("null_iid", "null_garch", "null_t3", "null_break",
             "alt_b05", "alt_b10", "alt_b20")


def simulate(scenario, seed):
    rng = _random.Random(seed)
    n = T + 200
    x = [0.0] * n
    y = [0.0] * n
    h = 1.0
    beta = {"alt_b05": 0.05, "alt_b10": 0.10, "alt_b20": 0.20}.get(scenario, 0.0)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + rng.gauss(0, 1)
        if scenario == "null_iid":
            e = rng.gauss(0, 1)
        elif scenario == "null_t3":
            # t(3): N / sqrt(chi2_3/3)
            c = sum(rng.gauss(0, 1) ** 2 for _ in range(3))
            e = rng.gauss(0, 1) / math.sqrt(c / 3.0)
        elif scenario == "null_break":
            e = rng.gauss(0, 3.0 if t > n // 2 else 1.0)
        else:                                   # garch nulls and alternatives
            h = 0.05 + 0.12 * (y[t - 1] - 0.5 * y[t - 2]) ** 2 + 0.85 * h
            e = math.sqrt(max(h, 1e-8)) * rng.gauss(0, 1)
        y[t] = 0.5 * y[t - 1] + beta * x[t - 1] + e
    return y[200:], x[200:]


# ---------------- classical tests: OLS with and without HAC -----------------

def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        piv = M[i][i] or 1e-30
        for r in range(n):
            if r != i:
                f = M[r][i] / piv
                for c_ in range(i, n + 1):
                    M[r][c_] -= f * M[i][c_]
    return [M[i][n] / (M[i][i] or 1e-30) for i in range(n)]


def classical_f1(y, x):
    """Focused Granger: y lags 1..P plus x lag 1 only; chi2_1 on P*F."""
    n = len(y)
    rows = range(P, n)
    d = 2 + P

    def design(t):
        return [1.0] + [y[t - l] for l in range(1, P + 1)] + [x[t - 1]]

    X = [design(t) for t in rows]
    Y = [y[t] for t in rows]
    m = len(Y)
    A = [[sum(X[i][a] * X[i][b] for i in range(m)) for b in range(d)]
         for a in range(d)]
    bvec = [sum(X[i][a] * Y[i] for i in range(m)) for a in range(d)]
    beta = _solve(A, bvec)
    rss1 = sum((Y[i] - sum(beta[k] * X[i][k] for k in range(d))) ** 2
               for i in range(m))
    dr = 1 + P
    Ar = [[A[a][b] for b in range(dr)] for a in range(dr)]
    beta_r = _solve(Ar, bvec[:dr])
    rss0 = sum((Y[i] - sum(beta_r[k] * X[i][k] for k in range(dr))) ** 2
               for i in range(m))
    F = (rss0 - rss1) / (rss1 / max(m - d, 1))
    return max(F, 0.0)


def classical_tests(y, x):
    """Returns (F_pvalue_proxy, HAC_pvalue_proxy) for H0: x lags add nothing.

    F: standard Granger F(P, T-2P-1) statistic converted via the chi2
    approximation P*F ~ chi2_P (T large). HAC: Wald chi2 on the x-lag
    coefficients with Newey-West covariance (P lags).
    """
    n = len(y)
    rows = range(P, n)
    d = 1 + 2 * P

    def design(t):
        return [1.0] + [y[t - l] for l in range(1, P + 1)] \
                     + [x[t - l] for l in range(1, P + 1)]

    X = [design(t) for t in rows]
    Y = [y[t] for t in rows]
    m = len(Y)
    A = [[sum(X[i][a] * X[i][b] for i in range(m)) for b in range(d)]
         for a in range(d)]
    bvec = [sum(X[i][a] * Y[i] for i in range(m)) for a in range(d)]
    beta = _solve(A, bvec)
    resid = [Y[i] - sum(beta[k] * X[i][k] for k in range(d)) for i in range(m)]
    rss1 = sum(r * r for r in resid)

    dr = 1 + P
    Ar = [[A[a][b] for b in range(dr)] for a in range(dr)]
    br = bvec[:dr]
    beta_r = _solve(Ar, br)
    resid_r = [Y[i] - sum(beta_r[k] * X[i][k] for k in range(dr))
               for i in range(m)]
    rss0 = sum(r * r for r in resid_r)

    F = ((rss0 - rss1) / P) / (rss1 / max(m - d, 1))
    chi_F = max(P * F, 0.0)

    # HAC Wald on the x-lag block: cov(beta) = A^-1 S A^-1 (sandwich)
    Ainv_cols = [_solve(A, [1.0 if r == c_ else 0.0 for r in range(d)])
                 for c_ in range(d)]
    Ainv = [[Ainv_cols[c_][r] for c_ in range(d)] for r in range(d)]
    L = P
    S = [[0.0] * d for _ in range(d)]
    for lag in range(0, L + 1):
        w = 1.0 - lag / (L + 1.0)
        for i in range(lag, m):
            u = resid[i] * resid[i - lag]
            Xi, Xj = X[i], X[i - lag]
            for a in range(d):
                ua = u * Xi[a]
                for b in range(d):
                    S[a][b] += w * ua * Xj[b] * (1.0 if lag == 0 else 1.0)
            if lag > 0:
                for a in range(d):
                    ua = u * Xj[a]
                    for b in range(d):
                        S[a][b] += w * ua * Xi[b]
    # V = Ainv S Ainv, restricted to the x block
    AS = [[sum(Ainv[a][k] * S[k][b] for k in range(d)) for b in range(d)]
          for a in range(d)]
    V = [[sum(AS[a][k] * Ainv[k][b] for k in range(d)) for b in range(d)]
         for a in range(d)]
    idx = list(range(1 + P, d))
    bx = [beta[i] for i in idx]
    Vx = [[V[i][j] for j in idx] for i in idx]
    Vinv_cols = [_solve(Vx, [1.0 if r == c_ else 0.0 for r in range(P)])
                 for c_ in range(P)]
    wald = sum(bx[a] * Vinv_cols[b][a] * bx[b]
               for a in range(P) for b in range(P))
    return chi_F, max(wald, 0.0)


def chi2_sf(xv, k):
    # regularized upper gamma via series/CF (small k), quick and adequate
    from timemachines import chi2_sf as f
    return f(xv, k)


def z_tests(y, x):
    """Three calibrated statistics. Z: corr(z_y[t], z_x[t-1]) — news on
    news. Z2: corr(z_y[t], x[t-1]) — calibrated target, raw covariate
    level (the level-vs-news fix). Z2H: Z2 with a Newey-West variance on
    the product terms (L=4), repairing residual dependence in z_y under
    heavy tails."""
    from timemachines import laplace
    from skaters.dist import Dist
    std_normal = Dist.gaussian(0.0, 1.0)

    def surprises(series):
        f, st = laplace(1), None
        out = []
        for v in series:
            dists, st = f(v, st)
            z = st["z"][0]
            out.append(0.0 if z is None else z)
        return out

    zy = surprises(y)
    zx = surprises(x)

    def gstat(a, b):
        n = len(a)
        ma = sum(a) / n
        mb = sum(b) / n
        ca = [v - ma for v in a]
        cb = [v - mb for v in b]
        den = math.sqrt(sum(v * v for v in ca) * sum(v * v for v in cb)) or 1e-30
        r = sum(p * q for p, q in zip(ca, cb)) / den
        return math.sqrt(n) * r, ca, cb, den

    g1, _, _, _ = gstat(zy[1:], zx[:-1])
    g2, ca, cb, den = gstat(zy[1:], x[:-1])

    # Newey-West variance of sum(ca*cb)/den, L=4
    prod = [p * q for p, q in zip(ca, cb)]
    n = len(prod)
    mpr = sum(prod) / n
    cprod = [v - mpr for v in prod]
    var = sum(v * v for v in cprod) / n
    for lag in range(1, 5):
        w = 1.0 - lag / 5.0
        var += 2.0 * w * sum(cprod[i] * cprod[i - lag]
                             for i in range(lag, n)) / n
    se = math.sqrt(max(var * n, 1e-30))
    g2h = sum(prod) / se
    return g1, g2, g2h


def run_one(job):
    scenario, seed = job
    y, x = simulate(scenario, seed)
    t0 = time.time()
    chi_F, wald = classical_tests(y, x)
    f1 = classical_f1(y, x)
    g1, g2, g2h = z_tests(y, x)
    pF = chi2_sf(chi_F, P)
    pH = chi2_sf(wald, P)
    pF1 = chi2_sf(f1, 1)

    def p2(g):
        return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(g) / math.sqrt(2.0))))

    return {"scenario": scenario, "seed": seed,
            "rej_F": pF < 0.05, "rej_HAC": pH < 0.05,
            "rej_F1": pF1 < 0.05, "rej_Z": p2(g1) < 0.05,
            "rej_Z2": p2(g2) < 0.05, "rej_Z2H": p2(g2h) < 0.05,
            "seconds": round(time.time() - t0, 1)}


def summarize(results):
    print(f"\n=== calibrated Granger study (n={len(results)}) ===")
    print(f"{'scenario':12s} {'F':>7s} {'HAC':>7s} {'F1':>7s} {'Z':>7s} {'Z2':>7s} {'Z2H':>7s}   (rejection at nominal 5%)")
    for sc in SCENARIOS:
        rows = [r for r in results if r["scenario"] == sc]
        if not rows:
            continue
        line = f"{sc:12s} "
        for t in ("rej_F", "rej_HAC", "rej_F1", "rej_Z", "rej_Z2", "rej_Z2H"):
            line += f"{sum(r[t] for r in rows)/len(rows):7.3f} "
        print(line + f"  ({len(rows)} reps)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, "granger_results.jsonl"))
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
