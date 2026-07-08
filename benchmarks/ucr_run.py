"""UCR Anomaly Archive runner for the parade Mahalanobis detector.

Protocol (see RESEARCH.md): each of the 250 series contains exactly one
anomaly; the filename encodes ``<trainLen>_<anomStart>_<anomEnd>``. We run
strictly causally over the whole series, use the anomaly-free prefix
[0, trainLen) for state warm-up only, and report the argmax of the score over
the scored region [trainLen, n). A series is a hit iff the argmax falls inside
the anomaly range extended by tolerance max(100, anomaly length). Accuracy is
the fraction of series hit.

Three scorers are computed in the same pass:
    mah:  -log10 p-value of the streaming Mahalanobis detector (the method)
    z1:   |z_1| — the 1-step parade surprise from the SAME forecaster
          (ablation: multivariate geometry vs per-horizon rule)
    mz:   EWMA z-score of the raw series (the community-mandatory trivial
          baseline; no forecaster at all)

Usage:
    python benchmarks/anomaly/ucr_run.py --limit 40 --k 3 --workers 8
    python benchmarks/anomaly/ucr_run.py            # full 250, long run
"""

from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

DATA = os.environ.get(
    "TIMEMACHINES_UCR_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "data", "UCR_Anomaly_FullData"))
_NAME = re.compile(r"^(\d+)_UCR_Anomaly_(.+)_(\d+)_(\d+)_(\d+)\.txt$")


def parse_name(fname: str):
    m = _NAME.match(fname)
    assert m, f"unparseable UCR filename: {fname}"
    sid, name, train_len, a_start, a_end = m.groups()
    return int(sid), name, int(train_len), int(a_start), int(a_end)


def load_series(path: str) -> list:
    vals = []
    with open(path) as f:
        for line in f:
            vals.extend(float(tok) for tok in line.split())
    return vals


def run_one(args):
    fname, k, base, scale_alpha, det_alpha, warmup_tail = args
    sid, name, train_len, a_start, a_end = parse_name(fname)
    ys = load_series(os.path.join(DATA, fname))
    n = len(ys)
    # Skip most of the anomaly-free prefix: every estimator in the stack has
    # effective memory <= ~1/min(alpha) ticks, so only the prefix TAIL warms
    # state. Statistically inert (the scored region is unchanged), large
    # compute saving on UCR's long prefixes. warmup_tail=0 => full prefix.
    start = max(0, train_len - warmup_tail) if warmup_tail else 0

    from skaters import search, parade
    from timemachines import laplace, mahalanobis, zbank

    if base == "zbank":
        # Feature bank: (memory sigma) x (clock stride) grid of engines,
        # concatenated parade surprises, r-factor scatter. z[0] remains the
        # finest engine's 1-step surprise, so the z1 ablation stays valid.
        sigmas, strides = (0.03, 0.003), (1, 4, 16)
        dim = len(sigmas) * len(strides) * k
        f = mahalanobis(zbank(k=k, sigmas=sigmas, strides=strides),
                        k=dim, factors=3, alpha=det_alpha)
    elif base == "search":
        # search() self-discovers periodicity online and injects seas(p)
        # candidates -- essential on UCR's waveform-periodic series, whose
        # periods (~50-400 samples) are invisible to laplace's fixed
        # calendar grid {7,12,24}. parade-wrap to expose z for the detector.
        f = mahalanobis(parade(search(k=k), k=k), k=k, alpha=det_alpha)
    else:
        # scale_alpha: residual-scale EWMA memory ~1/scale_alpha ticks. The
        # 0.03 default was tuned on short monthly FRED series; UCR series are
        # 10k-900k high-frequency points, where a slow scale (long memory)
        # stops the forecaster absorbing anomalies into "normal".
        f = mahalanobis(laplace(k, scale_alpha=scale_alpha), k=k,
                        alpha=det_alpha)
    state = None
    # trivial baseline state: EWMA mean/var of raw y
    mz_m, mz_v, mz_n = 0.0, 0.0, 0
    ALPHA = 0.02
    # scan-statistic buffers: windowed mean of -log10 p (anomalies are
    # intervals, not ticks; w in {8, 64} spans the labeled lengths)
    from collections import deque
    WINDOWS = (8, 64)
    wbuf = {w: deque() for w in WINDOWS}
    wsum = {w: 0.0 for w in WINDOWS}

    best = {"mah": (-1.0, -1.0, -1), "mahS": (-1.0, -1.0, -1),
            "z1": (-1.0, -1.0, -1), "zU": (-1.0, -1.0, -1),
            "mz": (-1.0, -1.0, -1)}
    t0 = time.time()
    for t, y in enumerate(ys[start:], start=start):
        _, state = f(y, state)
        # trivial baseline (causal: score with current estimate, then update)
        mz_n += 1
        a = max(ALPHA, 1.0 / mz_n)
        sd = math.sqrt(mz_v) if mz_v > 0 else 1e-8
        mz_score = abs(y - mz_m) / sd if mz_n > 3 else 0.0
        d = y - mz_m
        mz_m += a * d
        mz_v = (1 - a) * mz_v + a * d * (y - mz_m)

        if t < train_len:                      # warm-up: never scored
            continue
        p = state["pvalue"]
        if p is not None:
            d2 = state["d2"]
            s = -math.log10(max(p, 1e-300))
            if (s, d2) > best["mah"][:2]:      # d2 breaks saturation ties
                best["mah"] = (s, d2, t)
            for w in WINDOWS:
                wbuf[w].append(s)
                wsum[w] += s
                if len(wbuf[w]) > w:
                    wsum[w] -= wbuf[w].popleft()
                if len(wbuf[w]) == w:
                    sc = wsum[w] / w
                    if sc > best["mahS"][0]:
                        best["mahS"] = (sc, 0.0, t - w // 2)
        zvec = state["base"]["z"] if isinstance(state["base"], dict) else None
        z = zvec[0] if zvec else None
        if z is not None:
            if abs(z) > best["z1"][0]:
                best["z1"] = (abs(z), 0.0, t)
        # Union detector: min-p over the Mahalanobis channel and the k margin
        # channels (margins are N(0,1) by parade construction). Max of
        # -log10 p is argmax-equivalent to min-p; a Sidak fold would be
        # needed only for calibrated deployment, not for ranking.
        if p is not None and zvec is not None and all(v is not None for v in zvec):
            su = -math.log10(max(p, 1e-300))
            for zm in zvec:
                pm = math.erfc(abs(zm) / math.sqrt(2.0))
                su = max(su, -math.log10(max(pm, 1e-300)))
            if su > best["zU"][0]:
                best["zU"] = (su, 0.0, t)
        if mz_score > best["mz"][0]:
            best["mz"] = (mz_score, 0.0, t)

    tol = max(100, a_end - a_start)
    lo, hi = a_start - tol, a_end + tol
    res = {"sid": sid, "name": name, "fname": fname,
           "n": n, "train_len": train_len,
           "anom": [a_start, a_end], "seconds": round(time.time() - t0, 1)}
    for key, (score, _tie, loc) in best.items():
        res[key] = {"loc": loc, "score": round(score, 3),
                    "hit": bool(lo <= loc <= hi)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="run only the N shortest series (0 = all 250)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base", default="laplace",
                    choices=("laplace", "search", "zbank"))
    ap.add_argument("--scale-alpha", type=float, default=0.03,
                    help="laplace residual-scale EWMA rate (memory ~1/value)")
    ap.add_argument("--det-alpha", type=float, default=0.02,
                    help="detector location/scatter/null EWMA rate")
    ap.add_argument("--warmup-tail", type=int, default=4000,
                    help="feed only the last N points of the anomaly-free "
                         "prefix (0 = full prefix)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(os.listdir(DATA))
    files = [f for f in files if _NAME.match(f)]
    files.sort(key=lambda f: os.path.getsize(os.path.join(DATA, f)))
    if args.limit:
        files = files[:args.limit]

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"ucr_results_{args.base}_k{args.k}_sa{args.scale_alpha}"
        f"_da{args.det_alpha}.jsonl")

    # resume: skip series already checkpointed in out_path
    results = []
    if os.path.exists(out_path):
        with open(out_path) as fh:
            results = [json.loads(line) for line in fh if line.strip()]
        done = {r.get("fname") for r in results}
        files = [f for f in files if f not in done]
        print(f"resuming: {len(results)} done in {out_path}, "
              f"{len(files)} to go", flush=True)

    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(
                run_one, [(f, args.k, args.base, args.scale_alpha,
                           args.det_alpha, args.warmup_tail)
                          for f in files])):
            results.append(res)
            with open(out_path, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            hits = {m: sum(r[m]["hit"] for r in results) for m in ("mah", "mahS", "z1", "zU", "mz")}
            print(f"[{i+1}/{len(files)}] {res['sid']:03d} {res['name'][:30]:30s} "
                  f"n={res['n']:7d} {res['seconds']:6.1f}s  "
                  f"mah={'HIT ' if res['mah']['hit'] else 'miss'} "
                  f"mahS={'HIT ' if res['mahS']['hit'] else 'miss'} "
                  f"z1={'HIT ' if res['z1']['hit'] else 'miss'} "
                  f"zU={'HIT ' if res['zU']['hit'] else 'miss'} "
                  f"mz={'HIT ' if res['mz']['hit'] else 'miss'}  "
                  f"running: mah {hits['mah']} mahS {hits['mahS']} "
                  f"z1 {hits['z1']} zU {hits['zU']} mz {hits['mz']} "
                  f"of {i+1}", flush=True)

    n = len(results)
    print("\n=== UCR accuracy ===")
    for m, label in (("mah", "parade Mahalanobis (single tick)"),
                     ("mahS", "parade Mahalanobis (scan w=8/64)"),
                     ("z1", "per-horizon |z1| (same forecaster)"),
                     ("zU", "union min-p (Mahalanobis + margins)"),
                     ("mz", "EWMA z-score (trivial baseline)")):
        h = sum(r[m]["hit"] for r in results)
        print(f"{label:38s} {h}/{n} = {h/n:.3f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
