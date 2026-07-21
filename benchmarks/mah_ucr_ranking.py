"""UCR ranking gate for the mah scatter-model change (§4c).

The FRED dual-axis sweep showed pot95_shrink calibrates best with argmax flat
within noise — but FRED injection is the easy ranking regime. The factor
scatter model was chosen precisely to keep sensitivity in the "horizons
disagreed" directions that matter for localization on WAVEFORMS, which
identity shrinkage suppresses. So the genuine ranking test for the `shrink`
component is UCR: if shrink does not regress argmax accuracy here, it is safe
to adopt; if it does, the repair is pot_level only (which is ranking-invariant
by construction).

Configs (identical laplace/detector settings, only the scatter/pot_level
differ): base (factor, pot 0.98), shrink, pot95_shrink. Argmax of
-log10(pvalue) over the scored region [train_len, n); hit within
max(100, anomaly length). pot_level does not affect the argmax (p is monotone
in d2), so pot95_shrink vs shrink also double-checks that invariance on UCR.

Usage:
    NUMBA_NUM_THREADS=1 python benchmarks/mah_ucr_ranking.py --limit 40 --workers 8
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

from ucr_run import DATA, _NAME, parse_name, load_series  # noqa: E402

WARMUP_TAIL = 4000
CONFIGS = ("base", "shrink", "pot95_shrink")


def _make(name):
    from timemachines import laplace, mahalanobis
    b = laplace(3)
    if name == "base":
        return mahalanobis(b, k=3)
    if name == "shrink":
        return mahalanobis(b, k=3, scatter="shrink")
    if name == "pot95_shrink":
        return mahalanobis(b, k=3, pot_level=0.95, scatter="shrink")
    raise ValueError(name)


def argmax_hit(ys, train_len, start, lo, hi, name):
    f = _make(name)
    state = None
    best_s, best_loc = -1.0, -1
    for t in range(start, len(ys)):
        _dists, state = f(ys[t], state)
        if t < train_len:
            continue
        p = state["pvalue"]
        if p is None:
            continue
        s = -math.log10(max(p, 1e-300))
        if s > best_s:
            best_s, best_loc = s, t
    return bool(lo <= best_loc <= hi)


def run_one(job):
    fname, = job
    sid, name, train_len, a_start, a_end = parse_name(fname)
    ys = load_series(os.path.join(DATA, fname))
    start = max(0, train_len - WARMUP_TAIL)
    tol = max(100, a_end - a_start)
    lo, hi = a_start - tol, a_end + tol
    t0 = time.time()
    res = {"sid": sid, "name": name, "n": len(ys)}
    for cfg in CONFIGS:
        res[cfg] = argmax_hit(ys, train_len, start, lo, hi, cfg)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(_HERE, "mah_ucr_ranking_results.jsonl"))
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

    ofh = open(args.out, "a")
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(run_one, [(f,) for f in files]):
            results.append(res)
            ofh.write(json.dumps(res) + "\n")
            ofh.flush()
            os.fsync(ofh.fileno())
            print(f"[{len(results)}] {res['sid']:03d} {res['name'][:22]:22s} "
                  f"{res['seconds']:6.1f}s  "
                  + " ".join(f"{c}:{int(res[c])}" for c in CONFIGS), flush=True)
    ofh.close()

    n = len(results)
    print(f"\n=== mah UCR ranking gate (n={n}) — argmax accuracy ===")
    for cfg in CONFIGS:
        print(f"  {cfg:14s} {sum(r[cfg] for r in results)}/{n} = "
              f"{sum(r[cfg] for r in results)/n:.3f}")
    print("\nGate: shrink must not regress argmax vs base. pot95_shrink==shrink "
          "confirms pot_level is ranking-invariant.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
