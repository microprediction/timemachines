"""DAMP-laplace vs DAMP-raw on genuinely martingale series — the non-circular
test of §3b Q3.

The UCR arm (martingale_transform_study.py) de-anchors series whose stationary
template we still know, so "laplace fixes matrix profile" partly means "laplace
undoes the transform we applied." Here there is no ground-truth template:
FRED change series are natively martingale-ish (drift, regime shifts, fat
tails, no clean periodicity), so laplace-z is matrix profile's ONLY viable
normalization — per-window znorm has no stationary recurrence to lock onto.
If DAMP-laplace beats DAMP-raw here, the repair is real in deployment, not an
artifact of the controlled transform.

Same injection design as fred_anomaly.py (deterministic spike/burst/shift per
series). Detectors, all left matrix profile, window M=100:
    damp_raw     znorm MP on the raw change series
    damp_rawnn   non-normalized MP on the raw change series (control)
    damp_lap     non-normalized MP on the laplace-z stream (laplace replaces
                 znorm — the candidate)
Reported per series: argmax hit (planted-window localization) and rank
percentile of the planted window in the score ordering (robust to genuine
crises stealing the argmax, as fred_anomaly_v2.py).

Resumable: one jsonl row per series.

Usage:
    NUMBA_NUM_THREADS=1 python benchmarks/martingale_fred.py --limit 120 --workers 8
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

from fred import _load_levels, _to_changes                 # noqa: E402
from fred_anomaly import inject, UNIVERSE, MIN_LEN, TYPES   # noqa: E402

M = 100


def left_discord(xs, start, score_from, normalize):
    """Streaming left matrix profile over xs[start:]; argmax discord location
    among subsequences whose centre lands in [score_from, n)."""
    import numpy as np
    import stumpy
    T = np.asarray(xs[start:], dtype=np.float64)
    seed_len = max(score_from - start, M + 1)
    if len(T) <= seed_len + 1:
        return -1, -1.0, [0.0] * len(xs)
    stream = stumpy.stumpi(T[:seed_len], m=M, egress=False, normalize=normalize)
    for x in T[seed_len:]:
        stream.update(x)
    lp = stream.left_P_
    full = [0.0] * len(xs)
    best, loc = -1.0, -1
    for i in range(len(lp)):
        t = start + i + M // 2
        s = lp[i]
        if not np.isfinite(s):
            continue
        full[t] = float(s)
        if t >= score_from and s > best:
            best, loc = float(s), t
    return loc, best, full


def laplace_z(xs, start):
    from timemachines import laplace
    f = laplace(3)
    state = None
    zs = [0.0] * len(xs)
    for t in range(start, len(xs)):
        _, state = f(xs[t], state)
        z = state["z"][0]
        zs[t] = z if z is not None else 0.0
    return zs


def rank_pct(full_scores, score_from, n, window):
    lo, hi = window
    cand = list(range(score_from, n))
    if not cand:
        return 0.0
    in_win = [full_scores[t] for t in cand if lo <= t <= hi]
    if not in_win:
        return 0.0
    s_star = max(in_win)
    above = sum(1 for t in cand if full_scores[t] > s_star)
    return 1.0 - above / len(cand)


def run_one(job):
    sid, mode = job
    levels = _load_levels(sid)
    if not levels:
        return None
    if mode == "levels":
        # keep the unit root: the genuine martingale regime (prices), where
        # per-window znorm faces within-window drift and laplace-z is the only
        # stationary representation. This is the analog of UCR cumsum/rw_fast.
        xs = [v for _d, v in levels]
    else:
        xs = _to_changes(levels)      # stationary returns (analog of UCR raw)
    if len(xs) < MIN_LEN:
        return None
    xs, kind, (a_s, a_e) = inject(xs, sid)
    n = len(xs)
    start = 0
    score_from = int(0.4 * n)
    tol = max(M, a_e - a_s)
    window = (a_s - tol, a_e + tol)
    t0 = time.time()

    zs = laplace_z(xs, start)
    res = {"sid": sid, "n": n, "kind": kind, "anom": [a_s, a_e]}
    for cond, series, norm in (("damp_raw", xs, True),
                               ("damp_rawnn", xs, False),
                               ("damp_lap", zs, False)):
        loc, _sc, full = left_discord(series, start, score_from, norm)
        res[cond] = {"hit": bool(window[0] <= loc <= window[1]),
                     "pct": round(rank_pct(full, score_from, n, window), 4)}
    res["seconds"] = round(time.time() - t0, 1)
    return res


KEYS = ("damp_raw", "damp_rawnn", "damp_lap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mode", choices=("levels", "changes"), default="levels",
                    help="levels = genuine unit-root martingale test; "
                         "changes = stationary returns (control)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    if args.out is None:
        args.out = os.path.join(_HERE, f"martingale_fred_{args.mode}_results.jsonl")

    ids = [u["id"] for u in json.load(open(UNIVERSE))]
    data_dir = os.environ.get("TIMEMACHINES_FRED_DATA", os.path.join(_HERE, "data"))
    picked = []
    for sid in ids:
        path = os.path.join(data_dir, f"{sid}.csv")
        if os.path.exists(path) and os.path.getsize(path) > MIN_LEN * 12:
            picked.append(sid)
        if len(picked) >= args.limit:
            break

    results = []
    if os.path.exists(args.out):
        results = [json.loads(l) for l in open(args.out) if l.strip()]
        done = {r["sid"] for r in results}
        picked = [s for s in picked if s not in done]
        print(f"resuming: {len(results)} done, {len(picked)} to go", flush=True)

    ofh = open(args.out, "a")
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(
                run_one, [(s, args.mode) for s in picked]), 1):
            if res is None:
                continue
            results.append(res)
            ofh.write(json.dumps(res) + "\n")
            ofh.flush()
            os.fsync(ofh.fileno())
            print(f"[{i}/{len(picked)}] {res['sid'][:18]:18s} {res['kind']:5s} "
                  f"{res['seconds']:6.1f}s", flush=True)
    ofh.close()

    nser = len(results)
    if not nser:
        return
    print(f"\n=== DAMP-laplace on martingale FRED (n={nser}) ===")
    print(f"{'detector':12s} {'hit':>6s} {'pct':>8s}" + "".join(f"  {t+'_pct':>10s}" for t in TYPES))
    for k in KEYS:
        line = (f"{k:12s} {sum(r[k]['hit'] for r in results)/nser:6.3f} "
                f"{sum(r[k]['pct'] for r in results)/nser:8.4f}")
        for t in TYPES:
            sub = [r for r in results if r["kind"] == t]
            line += f"  {sum(r[k]['pct'] for r in sub)/len(sub) if sub else 0:10.4f}"
        print(line)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
