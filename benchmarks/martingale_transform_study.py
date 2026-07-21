"""Does the anchor or the recurrence drive matrix-profile's UCR dominance?

Hypothesis (Cotton): UCR's biological series have an ABSOLUTE ANCHOR — a
level-stationary baseline the waveform returns to — whereas financial series
are martingales with no anchor (unit root). Matrix profile owns UCR because
it exploits the anchor + recurrence; remove the anchor and the question is
(a) does its edge collapse (anchor-artifact) or survive (genuine recurrence
capability), (b) do wald/laplace catch up on the now-martingale series (their
home turf), and (c) can laplace REPAIR a structural detector on de-anchored
series.

We de-anchor each UCR series with several transforms and run every detector
on each. Anomaly LOCATION is preserved by every transform (all pointwise or
cumulative), so the UCR argmax+tolerance protocol still scores "did you
localize the event"; the anomaly TYPE changes under cumsum (spike->step),
tracked in the discussion, not re-labelled.

Transforms (per-step RW sd set by `window_wander` = sigma-fraction the level
drifts over one window M; sigma from the clean prefix):
    raw       control
    rw_slow   y + slow random walk  (window_wander 0.15: MP's per-window
              znorm absorbs it — de-anchors GLOBALLY, keeps local template)
    rw_fast   y + fast random walk  (window_wander 3.0: level moves ~3 sigma
              WITHIN a window, defeating znorm — de-anchors at window scale)
    cumsum    integrate demeaned y  (a genuine unit root; kills the anchor
              AND smears recurrence; spike->step)

Detectors:
    mah, mahS, z1   our head (mahalanobis(laplace) + plain parade surprise)
    damp_raw        SOTA baseline: znorm matrix profile on raw y
    damp_z          znorm matrix profile on laplace-z (DOUBLE normalization)
    damp_lap        DAMP-LAPLACE: non-normalized MP on laplace-z — laplace's
                    causal model-based normalization REPLACES znorm's crude
                    per-window mean/std (the crossover candidate)
    damp_rawnn      control: non-normalized MP on raw y (no normalizer at all)

Usage (gentle, resumable per series):
    NUMBA_NUM_THREADS=1 python benchmarks/martingale_transform_study.py \
        --limit 60 --workers 8
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
import zlib
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from ucr_run import DATA, _NAME, parse_name, load_series  # noqa: E402

M = 100                 # subsequence window; also the hit-tolerance floor
WARMUP_TAIL = 4000      # prefix tail fed for state warm-up (as ucr_run)


def _rng(sid: int):
    seed = zlib.crc32(str(sid).encode()) or 1

    def gauss():
        nonlocal seed
        # two LCG draws -> Box-Muller; deterministic per series
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        u = max(seed / 0x7FFFFFFF, 1e-12)
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        v = seed / 0x7FFFFFFF
        return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)
    return gauss


def transforms(ys, train_len, sid):
    """Return {name: transformed series}. sigma from the clean prefix."""
    n = len(ys)
    prefix = ys[:train_len] if train_len > 10 else ys[:max(10, n // 5)]
    mu = sum(prefix) / len(prefix)
    sigma = math.sqrt(sum((v - mu) ** 2 for v in prefix) / len(prefix)) or 1e-8
    gauss = _rng(sid)

    def random_walk(window_wander):
        step_sd = window_wander * sigma / math.sqrt(M)
        out, w = [0.0] * n, 0.0
        for t in range(n):
            w += step_sd * gauss()
            out[t] = ys[t] + w
        return out

    csum, acc = [0.0] * n, 0.0
    for t in range(n):
        acc += ys[t] - mu
        csum[t] = acc

    return {
        "raw": list(ys),
        "rw_slow": random_walk(0.15),
        "rw_fast": random_walk(3.0),
        "cumsum": csum,
    }


def laplace_pass(xs, start):
    """Plain parade z (for z1 and the DAMP-laplace input) in one pass."""
    from timemachines import laplace
    f = laplace(3)
    state = None
    zs = [0.0] * len(xs)
    z1_best = (-1.0, -1)
    for t in range(start, len(xs)):
        _, state = f(xs[t], state)
        z = state["z"][0]
        zs[t] = z if z is not None else 0.0
    return zs


def mahalanobis_pass(xs, start, train_len):
    """mah (-log10 p argmax) and mahS (scan) and z1 in one causal pass."""
    from timemachines import laplace, mahalanobis
    from collections import deque
    f = mahalanobis(laplace(3), k=3)
    state = None
    best = {"mah": (-1.0, -1), "mahS": (-1.0, -1), "z1": (-1.0, -1)}
    WINDOWS = (8, 64)
    wbuf = {w: deque() for w in WINDOWS}
    wsum = {w: 0.0 for w in WINDOWS}
    for t in range(start, len(xs)):
        _, state = f(xs[t], state)
        if t < train_len:
            continue
        z = state["base"]["z"][0] if isinstance(state["base"], dict) else None
        if z is not None and abs(z) > best["z1"][0]:
            best["z1"] = (abs(z), t)
        p = state["pvalue"]
        if p is None:
            continue
        s = -math.log10(max(p, 1e-300))
        if s > best["mah"][0]:
            best["mah"] = (s, t)
        for w in WINDOWS:
            wbuf[w].append(s)
            wsum[w] += s
            if len(wbuf[w]) > w:
                wsum[w] -= wbuf[w].popleft()
            if len(wbuf[w]) == w and wsum[w] / w > best["mahS"][0]:
                best["mahS"] = (wsum[w] / w, t - w // 2)
    return best


def left_discord(xs, train_len, start, normalize):
    """Streaming left matrix profile; argmax discord location in [train_len,n)."""
    import numpy as np
    import stumpy
    T = np.asarray(xs[start:], dtype=np.float64)
    seed_len = max(train_len - start, M + 1)
    if len(T) <= seed_len + 1:
        return -1, -1.0
    stream = stumpy.stumpi(T[:seed_len], m=M, egress=False, normalize=normalize)
    for x in T[seed_len:]:
        stream.update(x)
    lp = stream.left_P_
    best, loc = -1.0, -1
    for i in range(len(lp)):
        t = start + i
        if t < train_len:
            continue
        s = lp[i]
        if np.isfinite(s) and s > best:
            best, loc = float(s), t + M // 2
    return loc, best


def run_one(job):
    fname, = job
    sid, name, train_len, a_start, a_end = parse_name(fname)
    ys = load_series(os.path.join(DATA, fname))
    n = len(ys)
    start = max(0, train_len - WARMUP_TAIL)
    tol = max(M, a_end - a_start)
    lo, hi = a_start - tol, a_end + tol
    t0 = time.time()

    res = {"sid": sid, "name": name, "n": n, "train_len": train_len}
    for tname, xs in transforms(ys, train_len, sid).items():
        cell = {}
        mb = mahalanobis_pass(xs, start, train_len)
        for m, (_s, locv) in mb.items():
            cell[m] = bool(lo <= locv <= hi)
        zs = laplace_pass(xs, start)
        for cond, series, norm in (("damp_raw", xs, True),
                                   ("damp_rawnn", xs, False),
                                   ("damp_z", zs, True),
                                   ("damp_lap", zs, False)):
            loc, _sc = left_discord(series, train_len, start, norm)
            cell[cond] = bool(lo <= loc <= hi)
        res[tname] = cell
    res["seconds"] = round(time.time() - t0, 1)
    return res


TRANSFORMS = ("raw", "rw_slow", "rw_fast", "cumsum")
DETECTORS = ("mah", "mahS", "z1", "damp_raw", "damp_rawnn", "damp_z", "damp_lap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60,
                    help="N shortest UCR series (0 = all 250)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "martingale_transform_results.jsonl"))
    args = ap.parse_args()
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")

    files = sorted(f for f in os.listdir(DATA) if _NAME.match(f))
    files.sort(key=lambda f: os.path.getsize(os.path.join(DATA, f)))
    if args.limit:
        files = files[:args.limit]

    results = []
    if os.path.exists(args.out):
        results = [json.loads(l) for l in open(args.out) if l.strip()]
        done = {r["sid"] for r in results}
        files = [f for f in files if parse_name(f)[0] not in done]
        print(f"resuming: {len(results)} done, {len(files)} to go", flush=True)
    total = len(results) + len(files)

    ofh = open(args.out, "a")
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(run_one, [(f,) for f in files]):
            results.append(res)
            ofh.write(json.dumps(res) + "\n")
            ofh.flush()
            os.fsync(ofh.fileno())
            print(f"[{len(results)}/{total}] {res['sid']:03d} "
                  f"{res['name'][:24]:24s} {res['seconds']:6.1f}s", flush=True)
    ofh.close()

    nseries = len(results)
    print(f"\n=== martingale transform study (UCR n={nseries}) ===")
    print(f"argmax accuracy; rows=detector, cols=transform\n")
    hdr = f"{'detector':12s}" + "".join(f"{t:>10s}" for t in TRANSFORMS)
    print(hdr)
    for det in DETECTORS:
        row = f"{det:12s}"
        for t in TRANSFORMS:
            hits = sum(1 for r in results if r.get(t, {}).get(det))
            row += f"{hits/nseries:10.3f}"
        print(row)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
