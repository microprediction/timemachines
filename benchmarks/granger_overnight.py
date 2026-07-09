"""The calibrated Granger test: the overnight examination.

Everything the two day-runs left open, in one resumable grid:

- The full-strength opponent: Cheung-Ng with maximum-likelihood
  GARCH(1,1) standardization (CNG), alongside the EWMA proxy (CNE).
- Sample sizes T in {300, 1200, 4800} for the core scenarios.
- Nulls: iid; harsh GARCH; regime-switching vol; t(3)+GARCH; the
  variance-causality trap; stochastic (log-AR) volatility; volatility
  clocks in BOTH series; near-unit-root target.
- Alternatives: a five-point power curve under harsh GARCH; wrong-lag;
  time-varying beta (on/off blocks); nonlinear (tanh) causality;
  heteroskedastic x with weak causality.
- P-VALUES stored per test per rep, so size/power can be read at any
  level afterwards; 500 reps for nulls, 400 for alternatives.

Tests: F (4-lag Granger), HAC (Newey-West Wald), F1 (focused 1-lag),
CNE (Cheung-Ng, EWMA-standardized), CNG (Cheung-Ng, ML-GARCH
standardized), Z2 (calibrated target vs raw lagged x, NW variance), and
the lag-1..4 Bonferroni scans CNGM and Z2M.

Usage:
    python benchmarks/granger_overnight.py --workers 6
Summarize any time:
    python -c "import json,sys; sys.path.insert(0,'benchmarks'); \\
        from granger_overnight import summarize; \\
        summarize([json.loads(l) for l in open('benchmarks/granger_overnight_results.jsonl')])"
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

from granger_study import _solve, chi2_sf  # noqa: E402

P = 4

# (name, T, reps) — nulls first
CELLS = [
    ("n_iid", 1200, 500), ("n_harsh", 1200, 500), ("n_regime", 1200, 500),
    ("n_t3g", 1200, 500), ("n_volc", 1200, 500), ("n_sv", 1200, 500),
    ("n_xyvol", 1200, 500), ("n_nearunit", 1200, 500),
    ("n_harsh", 300, 500), ("n_harsh", 4800, 500),
    ("n_volc", 300, 500), ("n_volc", 4800, 500),
    ("a_b02", 1200, 400), ("a_b03", 1200, 400), ("a_b04", 1200, 400),
    ("a_b06", 1200, 400), ("a_b08", 1200, 400),
    ("a_lag3", 1200, 400), ("a_tvbeta", 1200, 400),
    ("a_nonlin", 1200, 400), ("a_xvol", 1200, 400),
    ("a_b04", 300, 400), ("a_b04", 4800, 400),
]


def simulate(scenario, T, seed):
    rng = _random.Random((hash((scenario, T)) & 0xffffff) * 1000003 + seed)
    n = T + 200
    x = [0.0] * n
    y = [0.0] * n
    h = 0.7
    hx = 1.0
    sig = 1.0
    logv = 0.0
    e_prev = 0.0
    ex_prev = 0.0
    beta = {"a_b02": 0.02, "a_b03": 0.03, "a_b04": 0.04, "a_b06": 0.06,
            "a_b08": 0.08, "a_lag3": 0.08, "a_tvbeta": 0.08,
            "a_xvol": 0.04}.get(scenario, 0.0)
    lag = 3 if scenario == "a_lag3" else 1
    rho = 0.98 if scenario == "n_nearunit" else 0.5

    def t3():
        c = sum(rng.gauss(0, 1) ** 2 for _ in range(3))
        return rng.gauss(0, 1) / math.sqrt(c / 3.0)

    for t in range(1, n):
        if scenario in ("n_xyvol", "a_xvol"):
            hx = 0.05 + 0.2 * ex_prev ** 2 + 0.75 * hx
            ex = math.sqrt(max(hx, 1e-8)) * rng.gauss(0, 1)
        else:
            ex = rng.gauss(0, 1)
        ex_prev = ex
        x[t] = 0.6 * x[t - 1] + ex

        if scenario == "n_iid":
            e = rng.gauss(0, 1)
        elif scenario == "n_regime":
            if rng.random() < 1.0 / 150.0:
                sig = 4.0 if sig == 1.0 else 1.0
            e = sig * rng.gauss(0, 1)
        elif scenario == "n_volc":
            h = 0.05 + 0.20 * x[t - 1] ** 2 + 0.75 * h
            e = math.sqrt(max(h, 1e-8)) * rng.gauss(0, 1)
        elif scenario == "n_sv":
            logv = 0.97 * logv + 0.35 * rng.gauss(0, 1)
            e = math.exp(0.5 * logv) * rng.gauss(0, 1)
        else:
            h = 0.02 + 0.25 * e_prev ** 2 + 0.72 * h
            innov = t3() if scenario == "n_t3g" else rng.gauss(0, 1)
            e = math.sqrt(max(h, 1e-8)) * innov
        e_prev = e

        b = beta
        if scenario == "a_tvbeta" and (t // 200) % 2 == 1:
            b = 0.0
        if scenario == "a_nonlin":
            drive = 0.12 * math.tanh(2.0 * x[t - 1])
        else:
            drive = b * x[t - lag] if t >= lag else 0.0
        y[t] = rho * y[t - 1] + drive + e
    return y[200:], x[200:]


# ------------------------------ classical tests ------------------------------

def _ols_tests(y, x):
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
    beta_r = _solve(Ar, bvec[:dr])
    rss0 = sum((Y[i] - sum(beta_r[k] * X[i][k] for k in range(dr))) ** 2
               for i in range(m))
    F = ((rss0 - rss1) / P) / (rss1 / max(m - d, 1))
    pF = chi2_sf(max(P * F, 0.0), P)

    # focused
    d1 = 2 + P
    X1 = [[1.0] + [y[t - l] for l in range(1, P + 1)] + [x[t - 1]]
          for t in rows]
    A1 = [[sum(X1[i][a] * X1[i][b] for i in range(m)) for b in range(d1)]
          for a in range(d1)]
    b1 = [sum(X1[i][a] * Y[i] for i in range(m)) for a in range(d1)]
    bt1 = _solve(A1, b1)
    rss1a = sum((Y[i] - sum(bt1[k] * X1[i][k] for k in range(d1))) ** 2
                for i in range(m))
    F1 = (rss0 - rss1a) / (rss1a / max(m - d1, 1))
    pF1 = chi2_sf(max(F1, 0.0), 1)

    # HAC Wald on the x block of the full model
    Ainv_cols = [_solve(A, [1.0 if r == c_ else 0.0 for r in range(d)])
                 for c_ in range(d)]
    Ainv = [[Ainv_cols[c_][r] for c_ in range(d)] for r in range(d)]
    S = [[0.0] * d for _ in range(d)]
    for lagk in range(0, P + 1):
        w = 1.0 - lagk / (P + 1.0)
        for i in range(lagk, m):
            u = resid[i] * resid[i - lagk]
            Xi, Xj = X[i], X[i - lagk]
            for a in range(d):
                ua = u * Xi[a]
                for bb in range(d):
                    S[a][bb] += w * ua * Xj[bb]
            if lagk > 0:
                for a in range(d):
                    ua = u * Xj[a]
                    for bb in range(d):
                        S[a][bb] += w * ua * Xi[bb]
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
    pH = chi2_sf(max(wald, 0.0), P)
    return pF, pH, pF1, resid


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


def _ewma_std(resid, lam=0.94):
    out = []
    v = sum(r * r for r in resid[:20]) / 20.0 or 1e-12
    for r in resid:
        out.append(r / math.sqrt(max(v, 1e-12)))
        v = lam * v + (1 - lam) * r * r
    return out


def _garch_nll(resid, om, al, be):
    v = sum(r * r for r in resid) / len(resid) or 1e-12
    nll = 0.0
    for r in resid:
        v = max(om + al * 0.0 + v * 0.0, 1e-12)  # placeholder, replaced below
    return nll


def _garch_fit_std(resid):
    """GARCH(1,1) Gaussian QMLE via Nelder-Mead on transformed params;
    returns standardized residuals."""
    m = len(resid)
    uvar = sum(r * r for r in resid) / m or 1e-12

    def nll(params):
        lo, la, lb = params
        al = 0.3 / (1.0 + math.exp(-la))
        be = 0.995 / (1.0 + math.exp(-lb)) * (1.0 - al)
        om = math.exp(lo)
        v = uvar
        total = 0.0
        r_prev = 0.0
        for r in resid:
            v = max(om + al * r_prev * r_prev + be * v, 1e-12)
            total += math.log(v) + r * r / v
            r_prev = r
        return total

    # Nelder-Mead, 3 params, ~80 iterations
    pts = [[math.log(uvar * 0.05), 0.0, 2.0]]
    for i in range(3):
        q = pts[0][:]
        q[i] += 0.7
        pts.append(q)
    vals = [nll(p) for p in pts]
    for _ in range(80):
        order = sorted(range(4), key=lambda i: vals[i])
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        cen = [sum(pts[i][j] for i in range(3)) / 3.0 for j in range(3)]
        refl = [cen[j] + (cen[j] - pts[3][j]) for j in range(3)]
        fr = nll(refl)
        if fr < vals[0]:
            exp_ = [cen[j] + 2.0 * (cen[j] - pts[3][j]) for j in range(3)]
            fe = nll(exp_)
            pts[3], vals[3] = (exp_, fe) if fe < fr else (refl, fr)
        elif fr < vals[2]:
            pts[3], vals[3] = refl, fr
        else:
            con = [cen[j] + 0.5 * (pts[3][j] - cen[j]) for j in range(3)]
            fc = nll(con)
            if fc < vals[3]:
                pts[3], vals[3] = con, fc
            else:
                for i in range(1, 4):
                    pts[i] = [(pts[i][j] + pts[0][j]) / 2.0 for j in range(3)]
                    vals[i] = nll(pts[i])
    lo, la, lb = pts[0]
    al = 0.3 / (1.0 + math.exp(-la))
    be = 0.995 / (1.0 + math.exp(-lb)) * (1.0 - al)
    om = math.exp(lo)
    v = uvar
    out = []
    r_prev = 0.0
    for r in resid:
        v = max(om + al * r_prev * r_prev + be * v, 1e-12)
        out.append(r / math.sqrt(v))
        r_prev = r
    return out


def _ccf_pvals(a_series, b_series):
    """p-values of sqrt(T)*ccf at lags 1..P (a leads: corr(a[t], b[t-l]))."""
    out = []
    for lagk in range(1, P + 1):
        a = a_series[lagk:]
        b = b_series[:-lagk]
        n = len(a)
        ma = sum(a) / n
        mb = sum(b) / n
        ca = [v - ma for v in a]
        cb = [v - mb for v in b]
        den = math.sqrt(sum(v * v for v in ca) * sum(v * v for v in cb)) or 1e-30
        g = math.sqrt(n) * sum(p * q for p, q in zip(ca, cb)) / den
        out.append(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(g) / math.sqrt(2.0)))))
    return out


def _z2_pvals(y, x):
    from timemachines import laplace
    f, st = laplace(1), None
    zy = []
    for v in y:
        dists, st = f(v, st)
        z = st["z"][0]
        zy.append(0.0 if z is None else z)
    out = []
    for lagk in range(1, P + 1):
        a = zy[lagk:]
        b = x[:-lagk]
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
        g = sum(prod) / se
        out.append(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(g) / math.sqrt(2.0)))))
    return out


def run_one(job):
    scenario, T, seed = job
    y, x = simulate(scenario, T, seed)
    t0 = time.time()
    pF, pH, pF1, _ = _ols_tests(y, x)
    ey = _ar_resid(y)
    ex = _ar_resid(x)
    cne = _ccf_pvals(_ewma_std(ey), _ewma_std(ex))
    cng = _ccf_pvals(_garch_fit_std(ey), _garch_fit_std(ex))
    z2 = _z2_pvals(y, x)
    return {"scenario": scenario, "T": T, "seed": seed,
            "pF": pF, "pHAC": pH, "pF1": pF1,
            "pCNE": cne[0], "pCNG": cng[0], "pCNGM": min(cng) * P,
            "pZ2": z2[0], "pZ2M": min(z2) * P,
            "seconds": round(time.time() - t0, 1)}


TESTS = ("pF", "pHAC", "pF1", "pCNE", "pCNG", "pCNGM", "pZ2", "pZ2M")


def summarize(results, alpha=0.05):
    print(f"\n=== overnight grid (n={len(results)}), alpha={alpha} ===")
    header = f"{'scenario':12s}{'T':>6s}" + "".join(f"{t[1:]:>7s}" for t in TESTS)
    print(header)
    seen = []
    for sc, T, _ in CELLS:
        if (sc, T) in seen:
            continue
        seen.append((sc, T))
        rows = [r for r in results if r["scenario"] == sc and r["T"] == T]
        if not rows:
            continue
        line = f"{sc:12s}{T:>6d}"
        for t in TESTS:
            line += f"{sum(1 for r in rows if r[t] < alpha)/len(rows):7.3f}"
        print(line + f"  ({len(rows)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, "granger_overnight_results.jsonl"))
    args = ap.parse_args()

    jobs = [(sc, T, s) for sc, T, reps in CELLS for s in range(reps)]
    results = []
    if os.path.exists(args.out):
        with open(args.out) as fh:
            results = [json.loads(line) for line in fh if line.strip()]
        done = {(r["scenario"], r["T"], r["seed"]) for r in results}
        jobs = [j for j in jobs if j not in done]
        print(f"resuming: {len(results)} done, {len(jobs)} to go", flush=True)
    print(f"{len(jobs)} jobs", flush=True)

    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, jobs)):
            results.append(res)
            with open(args.out, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            if (i + 1) % 200 == 0:
                print(f"[{i + 1}/{len(jobs)}]", flush=True)

    summarize(results)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
