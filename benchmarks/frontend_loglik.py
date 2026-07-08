"""Forecasting front-end lift: do existing algorithms improve in held-out
log-likelihood when run on the laplace-transformed series?

laplace defines a causal bijection on paths (the Rosenblatt transform):
    z_t = Phi^{-1}(F_t(y_t)),   F_t = predictive cdf issued at t-1,
so scoring an external algorithm on the z-stream and mapping back to
y-space is EXACT change of variables:
    log p_y(y_t) = log p_z(z_t) + [log f_t(y_t) - log phi(z_t)]
                                   \____________ jacobian ____________/

A bijection cannot create information, so an *optimal* opponent gains
nothing; the measured lift is exactly the structure the opponent misses
that laplace captures. Prediction: large lifts for rigid models (ETS,
ARIMA), smaller for flexible ones (GARCH).

Opponents (rolling one-step, refit every REFIT points, Gaussian
predictive from model mean/variance):
    ets:    statsmodels SimpleExpSmoothing-family (ETS AAN via Holt)
    arima:  statsforecast AutoARIMA
    garch:  arch GARCH(1,1), constant mean
    gauss:  EWMA mean/var Gaussian — control; on z this reproduces
            laplace itself (identity head), a built-in correctness check.

Usage:
    python benchmarks/anomaly/frontend_loglik.py --limit 20 --workers 2
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
import warnings
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from fred import _load_levels, _to_changes  # noqa: E402

UNIVERSE = os.path.join(
    os.environ.get("TIMEMACHINES_FRED_DATA", os.path.join(_HERE, "data")),
    "universe_daily.json")
MIN_LEN = 1500
N_MAX = 3000          # cap series length for refit cost
BURN = 500            # scored region: t >= BURN
REFIT = 200
WINDOW = 1000         # fit window (trailing)
_L2PI = math.log(2.0 * math.pi)


def _clamp(lp):
    """Bounded per-tick loss, the package's own convention: one degenerate
    predictive (e.g. a sticky near-Dirac missed by 8 sigma, logpdf ~ -1e6)
    must not dominate a series mean. Applied to EVERY method symmetrically."""
    return 20.0 if lp > 20.0 else (-20.0 if not (lp >= -20.0) else lp)


def _gll(y, mu, var):
    var = max(var, 1e-12)
    return -0.5 * (_L2PI + math.log(var) + (y - mu) ** 2 / var)


# --- laplace pass: z-series + jacobian, plus laplace's own log-lik ---

def laplace_pass(ys):
    from timemachines import laplace
    from skaters.dist import Dist
    std_normal = Dist.gaussian(0.0, 1.0)
    f = laplace(1)
    state, pend = None, None
    zs, jac, ll = [0.0] * len(ys), [0.0] * len(ys), [0.0] * len(ys)
    for t, y in enumerate(ys):
        if pend is not None:
            u = min(max(pend.cdf(y), 1e-12), 1.0 - 1e-12)
            z = std_normal.quantile(u)
            lp = pend.logpdf(y)
            zs[t] = z
            lphi = -0.5 * (_L2PI + z * z)
            jac[t] = lp - lphi
            ll[t] = lp
        dists, state = f(y, state)
        pend = dists[0]
    return zs, jac, ll


# --- rolling opponents: mean one-step Gaussian log-lik on a series ---

def roll_gauss(xs, dates=None):
    """EWMA mean/var control head."""
    m, v, n = 0.0, 0.0, 0
    out = [0.0] * len(xs)
    for t, x in enumerate(xs):
        n += 1
        a = max(0.02, 1.0 / n)
        out[t] = _gll(x, m, v if n > 3 else 1.0)
        d = x - m
        m += a * d
        v = (1 - a) * v + a * d * (x - m)
    return out

def roll_ets(xs, dates=None):
    from statsmodels.tsa.holtwinters import Holt
    import numpy as np
    out = [0.0] * len(xs)
    mu, var = 0.0, float(np.var(xs[:BURN]) or 1.0)
    fitted = None
    for t in range(BURN, len(xs)):
        if (t - BURN) % REFIT == 0:
            w = np.asarray(xs[max(0, t - WINDOW):t], dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = Holt(w, damped_trend=True).fit(optimized=True)
            var = max(float(np.mean(fitted.resid ** 2)),
                      0.25 * float(np.var(w)), 1e-10)
            preds = fitted.forecast(min(REFIT, len(xs) - t))
        mu = float(preds[(t - BURN) % REFIT])
        out[t] = _gll(xs[t], mu, var)
    return out

def roll_arima(xs, dates=None):
    from statsforecast.models import AutoARIMA
    import numpy as np
    out = [0.0] * len(xs)
    for t in range(BURN, len(xs)):
        if (t - BURN) % REFIT == 0:
            w = np.asarray(xs[max(0, t - WINDOW):t], dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = AutoARIMA()
                fit = m.fit(w)
                h = min(REFIT, len(xs) - t)
                fc = fit.predict(h=h, level=[68.27])
            mus = fc["mean"]
            sds = (fc["hi-68.27"] - fc["lo-68.27"]) / 2.0
        i = (t - BURN) % REFIT
        out[t] = _gll(xs[t], float(mus[i]), float(sds[i]) ** 2)
    return out

def roll_garch(xs, dates=None):
    from arch import arch_model
    import numpy as np
    out = [0.0] * len(xs)
    scale = float(np.std(xs[:BURN])) or 1.0
    for t in range(BURN, len(xs)):
        if (t - BURN) % REFIT == 0:
            w = np.asarray(xs[max(0, t - WINDOW):t], dtype=float) / scale
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                am = arch_model(w, mean="Constant", vol="GARCH", p=1, q=1,
                                rescale=False)
                fr = am.fit(disp="off", show_warning=False)
                fc = fr.forecast(horizon=min(REFIT, len(xs) - t),
                                 reindex=False)
            mus = fc.mean.values[0]
            vars_ = fc.variance.values[0]
        i = (t - BURN) % REFIT
        out[t] = _gll(xs[t] / scale, float(mus[i]), float(vars_[i])) \
            - math.log(scale)
    return out


def roll_prophet(xs, dates=None):
    """Prophet, rolling refit. Prophet models CALENDAR structure (weekday,
    yearly, holidays) from real dates — structure laplace's integer-lag
    seasonal block cannot represent. If prophet_z beats laplace alone, the
    front-end + calendar head genuinely compose."""
    import logging
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    import pandas as pd
    from prophet import Prophet
    out = [0.0] * len(xs)
    for t in range(BURN, len(xs)):
        if (t - BURN) % REFIT == 0:
            j0 = max(0, t - WINDOW)
            df = pd.DataFrame({"ds": pd.to_datetime(dates[j0:t]),
                               "y": xs[j0:t]})
            h = min(REFIT, len(xs) - t)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = Prophet(interval_width=0.6827,
                            uncertainty_samples=100)
                m.fit(df)
                fut = pd.DataFrame({"ds": pd.to_datetime(dates[t:t + h])})
                fc = m.predict(fut)
            mus = fc["yhat"].values
            sds = ((fc["yhat_upper"] - fc["yhat_lower"]) / 2.0).values
        i = (t - BURN) % REFIT
        out[t] = _gll(xs[t], float(mus[i]), max(float(sds[i]), 1e-8) ** 2)
    return out


OPPONENTS = {"gauss": roll_gauss, "ets": roll_ets,
             "arima": roll_arima, "garch": roll_garch,
             "prophet": roll_prophet}


def run_one(args):
    sid, = args
    levels = _load_levels(sid)
    ys = _to_changes(levels) if levels else []
    if len(ys) < MIN_LEN:
        return None
    dates = [d for d, _ in levels][1:]        # aligned with changes
    ys, dates = ys[-N_MAX:], dates[-N_MAX:]
    n = len(ys)
    t0 = time.time()
    zs, jac, lap_ll = laplace_pass(ys)
    res = {"sid": sid, "n": n,
           "laplace": sum(_clamp(v) for v in lap_ll[BURN:]) / (n - BURN)}
    for name, roll in OPPONENTS.items():
        try:
            raw = roll(ys, dates)
            res[f"{name}_raw"] = sum(_clamp(v) for v in raw[BURN:]) / (n - BURN)
            onz = roll(zs, dates)
            res[f"{name}_z"] = sum(_clamp(onz[t] + jac[t])
                                   for t in range(BURN, n)) / (n - BURN)
        except Exception as e:      # a failed fit shouldn't kill the series
            res[f"{name}_err"] = str(e)[:80]
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    ids = [u["id"] for u in json.load(open(UNIVERSE))]
    picked = []
    for sid in ids:
        p = os.path.join(_HERE, "..", "data", f"{sid}.csv")
        if os.path.exists(p) and os.path.getsize(p) > MIN_LEN * 12:
            picked.append(sid)
        if len(picked) >= args.limit:
            break

    results = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(
                run_one, [(s,) for s in picked])):
            if res is None:
                continue
            results.append(res)
            print(f"[{i+1}/{len(picked)}] {res['sid'][:16]:16s} "
                  f"{res['seconds']:6.1f}s  laplace={res['laplace']:+.3f}",
                  flush=True)

    out = os.path.join(_HERE, f"frontend_loglik_n{len(results)}.jsonl")
    with open(out, "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    n = len(results)
    print(f"\n=== mean one-step log-lik per point, y-space (n={n} series) ===")
    print(f"{'opponent':8s} {'raw':>8s} {'laplace-fronted':>16s} "
          f"{'lift':>8s} {'median':>8s} {'wins':>6s}")
    for name in OPPONENTS:
        rows = [r for r in results
                if f"{name}_raw" in r and f"{name}_z" in r]
        if not rows:
            continue
        m_raw = sum(r[f"{name}_raw"] for r in rows) / len(rows)
        m_z = sum(r[f"{name}_z"] for r in rows) / len(rows)
        lifts = sorted(r[f"{name}_z"] - r[f"{name}_raw"] for r in rows)
        med = lifts[len(lifts) // 2]
        wins = sum(l > 0 for l in lifts)
        print(f"{name:8s} {m_raw:8.3f} {m_z:16.3f} {m_z - m_raw:+8.3f} "
              f"{med:+8.3f} {wins}/{len(rows)}")
    lap = sum(r["laplace"] for r in results) / n
    print(f"{'laplace':8s} {'':8s} {lap:16.3f}   (reference, alone)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
