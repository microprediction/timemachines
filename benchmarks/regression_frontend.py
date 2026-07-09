"""Regression front-end: does the Laplace sandwich fix online regression?

The sequel to the detector/forecaster front-end studies (RESULTS.md sections
1-2), with "detector" replaced by "regressor". Thesis under test:

    Run every input stream and the target stream through its own Laplace
    body and regress in parade-z coordinates. The margins are conditional
    predictive CDFs, so each z stream is calibrated, stationary in scale and
    bounded in influence; the regression learns only cross-stream dependence
    (a dynamic Gaussian copula). Predictions map back through the target
    body's predictive distribution, point value by quadrature.

This harness is the contaminated-simulation prong. One fixed dumb learner
(recursive least squares with forgetting) is run under different coordinate
systems on the same synthetic streams; only the coordinates vary. The core
process is a linear Gaussian pair

    x_t = 0.8 x_{t-1} + e_t
    y_t = 0.7 y_{t-1} + 1.0 x_{t-1} + 0.5 n_t

one-step prequential prediction of y_t, contaminated along one axis per
scenario so we learn which failure mode each fix actually repairs:

    clean       nothing (measures the efficiency toll of each fix)
    spikes_x    2% measurement spikes on observed x, 6-10 sigma
    spikes_y    same on observed y
    spikes_both both
    heavy       e and n drawn t(2): infinite-variance driving noise
    drift       noise scales ramp x7 across the series
    distort     regressor observes sinh(x): monotone marginal distortion
                (the nonparanormal generative model); y driven by latent x
    shift       +6 level shift in x at 60%, propagating to y (recovery
                window scored separately)

Conditions (same RLS everywhere; z from ``laplace(1)`` parade state):

    laplace     target body alone, no regression (does x add anything?)
    raw         y_t ~ [1, y_{t-1}, x_{t-1}]
    zscore      same, all variables EWMA-standardized, affine map back
                (the RevIN-style affine special case)
    robust      median/MAD standardized + winsorized at 4 robust units
    huber       raw coordinates, RLS residual clipped in the update
                (fix the loss instead of the coordinates)
    zin         inputs fixed: y_t ~ [1, mu_y, z_y, mu_x, z_x] (body
                predictive means + calibrated surprises)
    zout        output fixed: z_y(t) ~ standardized raw inputs, mapped
                back through the y-body's predictive CDF by quadrature
    sandwich    both: z_y(t) ~ [1, z_y(t-1), z_x(t-1)], quadrature back

Metrics, per (scenario, seed): MSE and MAE against the conditional mean of
the clean process (excess risk; the oracle is the zero point), CRPS against
the clean y. CRPS uses the same 33-node quantile/pinball estimator for every
condition (closed-form quantiles for the Gaussian conditions, a per-tick
inverse-CDF grid for the mixture conditions), so the estimator bias is
shared. Primary metric is MSE vs the conditional mean.

Strictly causal, one pass, no whole-series normalisation. Resumable jsonl,
one line per (scenario, seed).

Usage:
    python benchmarks/regression_frontend.py --seeds 30 --workers 4
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

N = 1200
BURN = 300
PHI_Y, BETA, PHI_X, SIG_Y = 0.7, 1.0, 0.8, 0.5
SCENARIOS = ("clean", "spikes_x", "spikes_y", "spikes_both",
             "heavy", "drift", "distort", "shift")
CONDS = ("laplace", "raw", "zscore", "robust", "huber",
         "zin", "zout", "sandwich")

# 33 mid-quantile nodes and their standard-normal abscissae (Acklam-free:
# bisection on erf once at import).
_M = 33
_US = [(i + 0.5) / _M for i in range(_M)]


def _norm_ppf(p: float) -> float:
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


_ZS = [_norm_ppf(u) for u in _US]


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------- data


def _t2(rng):
    """Student t, 2 degrees of freedom: N / sqrt(chi2_2 / 2)."""
    chi2 = -2.0 * math.log(max(rng.random(), 1e-300))
    return rng.gauss(0.0, 1.0) / math.sqrt(chi2 / 2.0)


def make_series(scenario: str, seed: int):
    """Returns (obs_x, obs_y, m, y_clean, tau).

    m[t] is the conditional mean of clean y_t given the clean past (the
    oracle predictor); tau is the shift onset or None.
    """
    rng = _random.Random(zlib.crc32(f"{scenario}:{seed}".encode()))
    heavy = scenario == "heavy"
    tau = int(0.6 * N) if scenario == "shift" else None
    x = [0.0] * N
    y = [0.0] * N
    m = [0.0] * N
    for t in range(1, N):
        s = math.exp(2.0 * t / N) / math.e if scenario == "drift" else 1.0
        e = (_t2(rng) if heavy else rng.gauss(0.0, 1.0)) * s
        n = (_t2(rng) if heavy else rng.gauss(0.0, 1.0)) * s
        x[t] = PHI_X * x[t - 1] + e
        if tau is not None and t >= tau:
            x[t] += 6.0 * (1.0 - PHI_X)      # shifts the process mean by +6
        m[t] = PHI_Y * y[t - 1] + BETA * x[t - 1]
        y[t] = m[t] + SIG_Y * n
    obs_x, obs_y = list(x), list(y)
    if scenario == "distort":
        obs_x = [math.sinh(v) for v in x]
    if scenario in ("spikes_x", "spikes_both"):
        sx = _pstd(x)
        for t in range(1, N):
            if rng.random() < 0.02:
                obs_x[t] = x[t] + (6.0 + 4.0 * rng.random()) * sx * rng.choice((-1, 1))
    if scenario in ("spikes_y", "spikes_both"):
        sy = _pstd(y)
        for t in range(1, N):
            if rng.random() < 0.02:
                obs_y[t] = y[t] + (6.0 + 4.0 * rng.random()) * sy * rng.choice((-1, 1))
    return obs_x, obs_y, m, y, tau


def _pstd(v):
    mu = sum(v) / len(v)
    return math.sqrt(sum((a - mu) ** 2 for a in v) / len(v)) or 1e-8


# ---------------------------------------------------------------- pieces


class RLS:
    """Recursive least squares with exponential forgetting."""

    def __init__(self, d, lam=0.999, p0=100.0):
        self.d, self.lam = d, lam
        self.w = [0.0] * d
        self.P = [[(p0 if i == j else 0.0) for j in range(d)] for i in range(d)]

    def predict(self, phi):
        return sum(w * p for w, p in zip(self.w, phi))

    def update(self, phi, target, clip=None):
        err = target - self.predict(phi)
        if clip is not None:
            err = max(-clip, min(clip, err))
        Pp = [sum(self.P[i][j] * phi[j] for j in range(self.d))
              for i in range(self.d)]
        denom = self.lam + sum(p * pp for p, pp in zip(phi, Pp))
        g = [pp / denom for pp in Pp]
        for i in range(self.d):
            self.w[i] += g[i] * err
        for i in range(self.d):
            for j in range(self.d):
                self.P[i][j] = (self.P[i][j] - g[i] * Pp[j]) / self.lam
        # windup guard, as in skaters' ar(): reset on inflation/non-finite
        tr = sum(self.P[i][i] for i in range(self.d))
        if not math.isfinite(tr) or tr > 1e7:
            self.__init__(self.d, self.lam)


class Ewma:
    def __init__(self, alpha=0.02):
        self.a, self.m, self.v, self.n = alpha, 0.0, 1.0, 0

    def std(self):
        # unit scale until warm: the first few updates collapse v toward 0
        # (n=1 sets it exactly to 0) and a tiny floor would explode features
        if self.n < 10 or self.v <= 1e-8:
            return 1.0
        return math.sqrt(self.v)

    def update(self, x):
        self.n += 1
        a = max(self.a, 1.0 / self.n)
        d = x - self.m
        self.m += a * d
        self.v = (1 - a) * self.v + a * d * (x - self.m)


class RobustScale:
    """Streaming median/MAD (FAME-style)."""

    def __init__(self, eta=0.05):
        self.eta, self.med, self.mad, self.n = eta, 0.0, 1.0, 0

    def update(self, x):
        self.n += 1
        step = self.eta * self.mad if self.n > 5 else abs(x - self.med) + 1e-8
        self.med += step if x > self.med else -step
        self.mad = (1 - self.eta) * self.mad + self.eta * abs(x - self.med)
        self.mad = max(self.mad, 1e-6)


def inverse_cdf_grid(dist, points=256):
    """Monotone inverse CDF of a Dist by interpolation on a cdf grid."""
    d = dist.prune(12)
    lo = min(m - 8 * s for _, m, s in d.components)
    hi = max(m + 8 * s for _, m, s in d.components)
    if hi <= lo:
        hi = lo + 1e-6
    xs = [lo + (hi - lo) * i / (points - 1) for i in range(points)]
    cs = [d.cdf(v) for v in xs]

    def q(u):
        if u <= cs[0]:
            return xs[0]
        if u >= cs[-1]:
            return xs[-1]
        lo_i, hi_i = 0, points - 1
        while hi_i - lo_i > 1:
            mid = (lo_i + hi_i) // 2
            if cs[mid] < u:
                lo_i = mid
            else:
                hi_i = mid
        c0, c1 = cs[lo_i], cs[hi_i]
        w = (u - c0) / (c1 - c0) if c1 > c0 else 0.5
        return xs[lo_i] + w * (xs[hi_i] - xs[lo_i])

    return q


def pinball_crps(quantiles, y):
    """CRPS approximated as 2/m * sum of pinball losses on the _US grid."""
    total = 0.0
    for u, q in zip(_US, quantiles):
        total += (y - q) * u if y >= q else (q - y) * (1 - u)
    return 2.0 * total / _M


# ---------------------------------------------------------------- run one


def body_pass(series):
    """One Laplace(1) pass: surprise z[t] and the predictive Dist for the
    NEXT point (preds[t] = distribution for time t, issued at t-1)."""
    from timemachines import laplace
    f = laplace(1)
    st = None
    zs = [None] * len(series)
    preds = [None] * (len(series) + 1)
    for t, v in enumerate(series):
        dists, st = f(v, st)
        zs[t] = st["z"][0]
        preds[t + 1] = dists[0]
    return zs, preds


def run_one(job):
    scenario, seed = job

    obs_x, obs_y, m, y_clean, tau = make_series(scenario, seed)
    t0 = time.time()

    zy, ypred = body_pass(obs_y)
    zx, xpred = body_pass(obs_x)

    learners = {
        "raw": RLS(3), "zscore": RLS(3), "robust": RLS(3), "huber": RLS(3),
        "zin": RLS(5), "zout": RLS(3), "sandwich": RLS(3),
    }
    sc_y, sc_x = Ewma(), Ewma()                    # zscore trackers
    rb_y, rb_x = RobustScale(), RobustScale()      # robust trackers
    rvar = {c: Ewma() for c in CONDS if c != "laplace"}   # resid vars
    hub_scale = RobustScale()                      # huber residual scale

    acc = {c: {"se": 0.0, "ae": 0.0, "crps": 0.0, "n": 0} for c in CONDS}
    post = {c: {"se": 0.0, "n": 0} for c in CONDS}

    for t in range(2, N):
        y1, x1 = obs_y[t - 1], obs_x[t - 1]
        zy1 = zy[t - 1] if zy[t - 1] is not None else 0.0
        zx1 = zx[t - 1] if zx[t - 1] is not None else 0.0
        yd = ypred[t]
        mu_y, mu_x = yd.mean, xpred[t].mean

        ys_m, ys_s = sc_y.m, sc_y.std()
        xs_m, xs_s = sc_x.m, sc_x.std()
        yr_m, yr_s = rb_y.med, 1.4826 * rb_y.mad
        xr_m, xr_s = rb_x.med, 1.4826 * rb_x.mad

        feats = {
            "raw": [1.0, y1, x1],
            "zscore": [1.0, (y1 - ys_m) / ys_s, (x1 - xs_m) / xs_s],
            "robust": [1.0,
                       max(-4.0, min(4.0, (y1 - yr_m) / yr_s)),
                       max(-4.0, min(4.0, (x1 - xr_m) / xr_s))],
            "huber": [1.0, y1, x1],
            "zin": [1.0, mu_y, zy1, mu_x, zx1],
            "zout": [1.0, (y1 - ys_m) / ys_s, (x1 - xs_m) / xs_s],
            "sandwich": [1.0, zy1, zx1],
        }

        qgrid = inverse_cdf_grid(yd)

        preds, quants = {}, {}
        preds["laplace"] = mu_y
        quants["laplace"] = [qgrid(u) for u in _US]
        for c in ("raw", "huber", "zin"):
            p = learners[c].predict(feats[c])
            s = rvar[c].std()
            preds[c] = p
            quants[c] = [p + s * z for z in _ZS]
        p = learners["zscore"].predict(feats["zscore"])
        s = rvar["zscore"].std() * ys_s
        preds["zscore"] = ys_m + ys_s * p
        quants["zscore"] = [preds["zscore"] + s * z for z in _ZS]
        p = learners["robust"].predict(feats["robust"])
        s = rvar["robust"].std() * yr_s
        preds["robust"] = yr_m + yr_s * p
        quants["robust"] = [preds["robust"] + s * z for z in _ZS]
        for c in ("zout", "sandwich"):
            zhat = learners[c].predict(feats[c])
            sz = min(rvar[c].std(), 3.0)
            qs = [qgrid(_norm_cdf(zhat + sz * z)) for z in _ZS]
            preds[c] = sum(qs) / _M          # quadrature: mean of pushforward
            quants[c] = qs

        if t >= BURN:
            for c in CONDS:
                a = acc[c]
                a["se"] += (preds[c] - m[t]) ** 2
                a["ae"] += abs(preds[c] - m[t])
                a["crps"] += pinball_crps(quants[c], y_clean[t])
                a["n"] += 1
            if tau is not None and tau <= t < tau + 80:
                for c in CONDS:
                    post[c]["se"] += (preds[c] - m[t]) ** 2
                    post[c]["n"] += 1

        # --- updates (after prediction: prequential) ---
        yt = obs_y[t]
        for c in ("raw", "zin"):
            rvar[c].update(yt - preds[c])
            learners[c].update(feats[c], yt)
        r = yt - preds["huber"]
        hub_scale.update(r)
        rvar["huber"].update(r)
        learners["huber"].update(feats["huber"], yt,
                                 clip=2.5 * 1.4826 * hub_scale.mad)
        tgt = (yt - ys_m) / ys_s
        rvar["zscore"].update(tgt - learners["zscore"].predict(feats["zscore"]))
        learners["zscore"].update(feats["zscore"], tgt)
        tgt = max(-4.0, min(4.0, (yt - yr_m) / yr_s))
        rvar["robust"].update(tgt - learners["robust"].predict(feats["robust"]))
        learners["robust"].update(feats["robust"], tgt)
        if zy[t] is not None:
            for c in ("zout", "sandwich"):
                rvar[c].update(zy[t] - learners[c].predict(feats[c]))
                learners[c].update(feats[c], zy[t])
        sc_y.update(yt)
        sc_x.update(obs_x[t])
        rb_y.update(yt)
        rb_x.update(obs_x[t])

    res = {"scenario": scenario, "seed": seed, "n": N}
    for c in CONDS:
        a = acc[c]
        res[c] = {"mse": a["se"] / a["n"], "mae": a["ae"] / a["n"],
                  "crps": a["crps"] / a["n"]}
        if tau is not None and post[c]["n"]:
            res[c]["mse_post"] = post[c]["se"] / post[c]["n"]
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ---------------------------------------------------------------- main


def summarize(results):
    print(f"\n=== regression front-end, excess MSE vs conditional mean "
          f"(median over seeds; n={len(results)} rows) ===")
    for metric in ("mse", "crps"):
        print(f"\n--- {metric} ---")
        print(f"{'scenario':12s} " + " ".join(f"{c:>9s}" for c in CONDS))
        for sc in SCENARIOS:
            rows = [r for r in results if r["scenario"] == sc]
            if not rows:
                continue
            line = f"{sc:12s} "
            for c in CONDS:
                vals = sorted(r[c][metric] for r in rows)
                line += f"{vals[len(vals) // 2]:9.4f} "
            print(line)
    rows = [r for r in results if r["scenario"] == "shift"
            and "mse_post" in r["raw"]]
    if rows:
        print("\n--- shift recovery (median MSE, 80 ticks post-onset) ---")
        line = f"{'shift+80':12s} "
        for c in CONDS:
            vals = sorted(r[c]["mse_post"] for r in rows)
            line += f"{vals[len(vals) // 2]:9.4f} "
        print(line)
    print("\n--- wins vs raw on mse (per scenario, out of seeds) ---")
    print(f"{'scenario':12s} " + " ".join(f"{c:>9s}" for c in CONDS))
    for sc in SCENARIOS:
        rows = [r for r in results if r["scenario"] == sc]
        if not rows:
            continue
        line = f"{sc:12s} "
        for c in CONDS:
            w = sum(1 for r in rows if r[c]["mse"] < r["raw"]["mse"])
            line += f"{w:>7d}/{len(rows):<2d}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scenarios", type=str, default="")
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, "regression_frontend_results.jsonl"))
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
                  f"raw {res['raw']['mse']:7.3f}  "
                  f"sand {res['sandwich']['mse']:7.3f}", flush=True)

    summarize(results)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
