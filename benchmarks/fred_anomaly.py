"""FRED injection benchmark: anomaly detection on economic series.

The mirror image of UCR: that archive is periodic-waveform home turf for
matrix-profile methods; FRED is trends, regime shifts, fat tails and no clean
periodicity — the regime laplace was built for and the SOTA detectors were
never pointed at. FRED has no anomaly labels, so this is an *injection*
design: real change-series backgrounds (the paper's universe rule, cached
offline), one planted anomaly per series at a known location, argmax protocol
as in ucr_run.py.

Injection types (cycled deterministically; magnitude in local sigmas):
    spike:  x[tau]      += 8 sigma      (one-tick outlier)
    burst:  x[tau:+20]  += N(0, 3 sigma) (variance burst)
    shift:  x[tau:]     += 2 sigma      (persistent change-mean shift =
                                         slope break in levels)

Fairness notes, stated plainly: laplace runs at its defaults, which were
tuned on FRED for *forecasting* (held-out log-lik/CRPS) — never on these
injections; DSPOT and RRCF run at their canonical settings, exactly as on
UCR. Everyone at their defaults, home field disclosed.

Usage:
    python benchmarks/anomaly/fred_anomaly.py --limit 150 --workers 2
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
from collections import deque
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))       # for benchmarks.*
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

from frontend_run import DSpot, rrcf_scores  # noqa: E402
from fred import _load_levels, _to_changes  # noqa: E402

UNIVERSE = os.path.join(
    os.environ.get("TIMEMACHINES_FRED_DATA", os.path.join(_HERE, "data")),
    "universe_daily.json")
MIN_LEN = 1000
TYPES = ("spike", "burst", "shift")


def inject(xs, sid):
    """Plant one anomaly; returns (xs, type, [a_start, a_end])."""
    rng = _random.Random(zlib.crc32(sid.encode()))
    n = len(xs)
    tau = rng.randint(int(0.6 * n), int(0.9 * n))
    w = xs[max(0, tau - 200):tau]
    mu = sum(w) / len(w)
    sigma = math.sqrt(sum((v - mu) ** 2 for v in w) / len(w)) or 1e-8
    kind = TYPES[zlib.crc32(sid.encode()) % len(TYPES)]
    xs = list(xs)
    if kind == "spike":
        xs[tau] += 8.0 * sigma * (1 if rng.random() < 0.5 else -1)
        span = (tau, tau)
    elif kind == "burst":
        end = min(tau + 20, n - 1)
        for t in range(tau, end + 1):
            xs[t] += rng.gauss(0, 3.0 * sigma)
        span = (tau, end)
    else:                                          # shift
        d = 2.0 * sigma * (1 if rng.random() < 0.5 else -1)
        for t in range(tau, n):
            xs[t] += d
        span = (tau, min(tau + 20, n - 1))         # score the onset
    return xs, kind, span


def run_one(args):
    sid, = args
    levels = _load_levels(sid)
    xs = _to_changes(levels) if levels else []
    if len(xs) < MIN_LEN:
        return None
    xs, kind, (a_s, a_e) = inject(xs, sid)
    n = len(xs)
    score_from = int(0.4 * n)
    tol = max(50, a_e - a_s)
    lo, hi = a_s - tol, a_e + tol
    t0 = time.time()

    from timemachines import laplace, mahalanobis

    res = {"sid": sid, "n": n, "kind": kind, "anom": [a_s, a_e]}

    # --- ours: laplace at FRED-tuned defaults ---
    f = mahalanobis(laplace(3), k=3)
    state = None
    mz_m, mz_v, mz_n = 0.0, 0.0, 0
    wbuf, wsum = {8: deque(), 64: deque()}, {8: 0.0, 64: 0.0}
    best = {m: (-1.0, -1) for m in ("mah", "mahS", "z1", "mz")}
    zs = [0.0] * n
    for t, y in enumerate(xs):
        _, state = f(y, state)
        z = state["base"]["z"][0]
        zs[t] = z if z is not None else 0.0
        mz_n += 1
        a = max(0.02, 1.0 / mz_n)
        sd = math.sqrt(mz_v) if mz_v > 0 else 1e-8
        mzs = abs(y - mz_m) / sd if mz_n > 3 else 0.0
        d = y - mz_m
        mz_m += a * d
        mz_v = (1 - a) * mz_v + a * d * (y - mz_m)
        if t < score_from:
            continue
        p = state["pvalue"]
        if p is not None:
            s = -math.log10(max(p, 1e-300))
            if s > best["mah"][0]:
                best["mah"] = (s, t)
            for w in (8, 64):
                wbuf[w].append(s)
                wsum[w] += s
                if len(wbuf[w]) > w:
                    wsum[w] -= wbuf[w].popleft()
                if len(wbuf[w]) == w and wsum[w] / w > best["mahS"][0]:
                    best["mahS"] = (wsum[w] / w, t - w // 2)
        if z is not None and abs(z) > best["z1"][0]:
            best["z1"] = (abs(z), t)
        if mzs > best["mz"][0]:
            best["mz"] = (mzs, t)
    for m, (_, loc) in best.items():
        res[m] = {"loc": loc, "hit": bool(lo <= loc <= hi)}

    # --- SOTA baselines on raw and on the laplace-transformed z ---
    for cond, series in (("raw", xs), ("z", zs)):
        calib = series[:score_from]
        d = DSpot(list(calib))
        b, loc = -1.0, -1
        for t in range(score_from, n):
            s = d.score(series[t])
            if s > b:
                b, loc = s, t
        res[f"dspot_{cond}"] = {"loc": loc, "hit": bool(lo <= loc <= hi)}
        sc = rrcf_scores(series)
        b, loc = -1.0, -1
        for t in range(score_from, n):
            if sc[t] > b:
                b, loc = sc[t], t
        res[f"rrcf_{cond}"] = {"loc": loc, "hit": bool(lo <= loc <= hi)}

    res["seconds"] = round(time.time() - t0, 1)
    return res


KEYS = ("mah", "mahS", "z1", "mz", "dspot_raw", "dspot_z", "rrcf_raw", "rrcf_z")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    ids = [u["id"] for u in json.load(open(UNIVERSE))]
    # keep only series already in the offline cache and long enough
    picked = []
    for sid in ids:
        path = os.path.join(
            os.environ.get("TIMEMACHINES_FRED_DATA",
                           os.path.join(_HERE, "data")), f"{sid}.csv")
        if os.path.exists(path) and os.path.getsize(path) > MIN_LEN * 12:
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
            hits = {m: sum(r[m]["hit"] for r in results) for m in KEYS}
            print(f"[{i+1}/{len(picked)}] {res['sid'][:18]:18s} "
                  f"{res['kind']:5s} {res['seconds']:5.1f}s  "
                  + " ".join(f"{m}:{hits[m]}" for m in KEYS), flush=True)

    out = os.path.join(_HERE, f"fred_anomaly_results_n{len(results)}.jsonl")
    with open(out, "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    n = len(results)
    print(f"\n=== FRED injection accuracy (n={n}) ===")
    print(f"{'method':12s} {'all':>6s}  " + "  ".join(f"{t:>6s}" for t in TYPES))
    for m in KEYS:
        h = sum(r[m]["hit"] for r in results)
        row = f"{m:12s} {h/n:6.3f}  "
        for t in TYPES:
            sub = [r for r in results if r["kind"] == t]
            row += f"{(sum(r[m]['hit'] for r in sub) / len(sub)) if sub else 0:6.3f}  "
        print(row)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
