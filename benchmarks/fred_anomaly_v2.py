"""FRED injection benchmark v2 — rank-percentile scoring (see RESULTS.md §4).

The v1 argmax protocol was confounded: real FRED backgrounds contain genuine
unlabeled anomalies (2008, COVID), so argmax measured "which real crisis did
you prefer". v2 keeps the identical deterministic injections (same seeds,
same planted windows as fred_anomaly.py) and scores each method by the
**rank percentile of the planted window**: the fraction of scored ticks whose
score does not exceed the best in-window score. 1.0 means the argmax is in
the window; 0.99 on a 5000-tick series means ~50 ticks outrank it. This is
robust to a dominant natural event stealing the argmax.

Also reported per method: the v1 argmax hit (reference), and both quantities
with known crisis windows masked out of the ranking set (GFC 2008-06 to
2009-12, COVID 2020-02 to 2021-06) — planted-window ticks are never masked.

Resumable: one jsonl row per series; existing sids are skipped.

Usage:
    python benchmarks/fred_anomaly_v2.py --limit 150 --workers 2
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

from frontend_run import DSpot, rrcf_scores  # noqa: E402
from fred import _load_levels, _to_changes   # noqa: E402
from fred_anomaly import inject, UNIVERSE, MIN_LEN, TYPES  # noqa: E402

CRISES = (("2008-06-01", "2009-12-31"), ("2020-02-01", "2021-06-30"))


def in_crisis(date: str) -> bool:
    return any(lo <= date <= hi for lo, hi in CRISES)


def rolling_mean(xs, w):
    out, buf, s = [0.0] * len(xs), [], 0.0
    for t, v in enumerate(xs):
        buf.append(v)
        s += v
        if len(buf) > w:
            s -= buf.pop(0)
        out[t] = s / len(buf)
    return out


def rank_stats(scores, ticks, window, masked):
    """(hit, pct) of the planted window within the candidate tick set."""
    lo, hi = window
    cand = [t for t in ticks if not masked[t] or lo <= t <= hi]
    if not cand:
        return False, 0.0
    in_win = [scores[t] for t in cand if lo <= t <= hi]
    if not in_win:
        return False, 0.0
    s_star = max(in_win)
    argmax_t = max(cand, key=lambda t: scores[t])
    n_above = sum(1 for t in cand if scores[t] > s_star)
    return bool(lo <= argmax_t <= hi), 1.0 - n_above / len(cand)


def run_one(args):
    sid, = args
    levels = _load_levels(sid)
    xs = _to_changes(levels) if levels else []
    if len(xs) < MIN_LEN:
        return None
    dates = [levels[i + 1][0] for i in range(len(xs))]
    xs, kind, (a_s, a_e) = inject(xs, sid)
    n = len(xs)
    score_from = int(0.4 * n)
    tol = max(50, a_e - a_s)
    window = (a_s - tol, a_e + tol)
    t0 = time.time()

    from timemachines import laplace, mahalanobis

    f = mahalanobis(laplace(3), k=3)
    state = None
    mah = [0.0] * n
    z1 = [0.0] * n
    mz = [0.0] * n
    zs = [0.0] * n
    mz_m, mz_v, mz_n = 0.0, 0.0, 0
    for t, y in enumerate(xs):
        _, state = f(y, state)
        z = state["base"]["z"][0]
        zs[t] = z if z is not None else 0.0
        z1[t] = abs(zs[t])
        p = state["pvalue"]
        if p is not None:
            mah[t] = -math.log10(max(p, 1e-300))
        mz_n += 1
        a = max(0.02, 1.0 / mz_n)
        if mz_n > 3 and mz_v > 0:
            mz[t] = abs(y - mz_m) / math.sqrt(mz_v)
        d = y - mz_m
        mz_m += a * d
        mz_v = (1 - a) * mz_v + a * d * (y - mz_m)

    scores = {"mah": mah, "mahS8": rolling_mean(mah, 8),
              "mahS64": rolling_mean(mah, 64), "z1": z1, "mz": mz}
    for cond, series in (("raw", xs), ("z", zs)):
        d = DSpot(list(series[:score_from]))
        ds = [0.0] * n
        for t in range(score_from, n):
            ds[t] = d.score(series[t])
        scores[f"dspot_{cond}"] = ds
        scores[f"rrcf_{cond}"] = rrcf_scores(series)

    ticks = list(range(score_from, n))
    masked = [in_crisis(dt) for dt in dates]
    res = {"sid": sid, "n": n, "kind": kind, "anom": [a_s, a_e],
           "masked_frac": round(sum(masked[score_from:]) / len(ticks), 3)}
    for m, s in scores.items():
        hit, pct = rank_stats(s, ticks, window, [False] * n)
        hit_m, pct_m = rank_stats(s, ticks, window, masked)
        res[m] = {"hit": hit, "pct": round(pct, 4),
                  "hit_m": hit_m, "pct_m": round(pct_m, 4)}
    res["seconds"] = round(time.time() - t0, 1)
    return res


KEYS = ("mah", "mahS8", "mahS64", "z1", "mz",
        "dspot_raw", "dspot_z", "rrcf_raw", "rrcf_z")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(
        _HERE, "fred_anomaly_v2_results.jsonl"))
    args = ap.parse_args()

    ids = [u["id"] for u in json.load(open(UNIVERSE))]
    data_dir = os.environ.get("TIMEMACHINES_FRED_DATA",
                              os.path.join(_HERE, "data"))
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

    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(
                run_one, [(s,) for s in picked]), 1):
            if res is None:
                continue
            results.append(res)
            with open(args.out, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            print(f"[{i}/{len(picked)}] {res['sid'][:18]:18s} {res['kind']:5s} "
                  f"{res['seconds']:5.1f}s", flush=True)

    n = len(results)
    if not n:
        return
    print(f"\n=== FRED injection v2, rank-percentile (n={n}) ===")
    print(f"{'method':10s} {'hit':>5s} {'pct':>7s} {'hit_m':>6s} {'pct_m':>7s}"
          + "".join(f"  {t+'_pct_m':>11s}" for t in TYPES))
    for m in KEYS:
        line = (f"{m:10s} {sum(r[m]['hit'] for r in results)/n:5.3f} "
                f"{sum(r[m]['pct'] for r in results)/n:7.4f} "
                f"{sum(r[m]['hit_m'] for r in results)/n:6.3f} "
                f"{sum(r[m]['pct_m'] for r in results)/n:7.4f}")
        for t in TYPES:
            sub = [r for r in results if r["kind"] == t]
            line += f"  {sum(r[m]['pct_m'] for r in sub)/len(sub) if sub else 0:11.4f}"
        print(line)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
