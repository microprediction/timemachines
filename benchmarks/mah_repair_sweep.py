"""Dual-axis evaluation of mah calibration fixes: FPR *and* ranking together.

The mah diagnostic (mah_diagnostic.py) localizes the overconfidence to the
bulk Satterthwaite null (d2's tail is heavier than a scaled chi2 — the
finite-sample Hotelling effect: d2 from an EWMA-estimated inverse-covariance
follows a heavy scaled-F, and a two-moment match keeps a too-thin tail). Two
public-knob levers can widen the effective tail, and BOTH have a job-1/job-2
tension, so neither may be adopted on calibration alone:

  pot_level (lower)  hand the tail to the empirical GPD earlier, before the
                     chi2 approximation diverges. Risk: the GPD region creeps
                     toward the mode where its assumption is weaker.
  scatter="shrink"   inflate the small "horizons disagreed" eigenvalues ->
                     lower d2 -> better calibration, but suppresses the very
                     directions the factor model preserves for localization.

This harness plants one anomaly per FRED series (same design as
fred_anomaly.py) and, in ONE pass per config, measures both axes on the same
series:
  * ranking: argmax of -log10(pvalue) over the scored region hits the planted
    window (extended by tolerance);
  * calibration: FPR of pvalue < alpha on the CLEAN region strictly before the
    injection (real events remain, but the comparison is paired across configs).

Configs (existing knobs only — no library change; a winner becomes a proposed
default): base (pot0.98/factor), pot95, pot90, shrink, pot95_shrink.

Resumable: one jsonl row per series.

Usage:
    python benchmarks/mah_repair_sweep.py --limit 120 --workers 8
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

from fred import _load_levels, _to_changes                  # noqa: E402
from fred_anomaly import inject, UNIVERSE, MIN_LEN, TYPES    # noqa: E402

ALPHAS = (1e-2, 1e-3, 1e-4)
CONFIGS = ("base", "pot95", "pot90", "shrink", "pot95_shrink")


def _make(name):
    from timemachines import laplace, mahalanobis
    b = laplace(3)
    if name == "base":
        return mahalanobis(b, k=3)
    if name == "pot95":
        return mahalanobis(b, k=3, pot_level=0.95)
    if name == "pot90":
        return mahalanobis(b, k=3, pot_level=0.90)
    if name == "shrink":
        return mahalanobis(b, k=3, scatter="shrink")
    if name == "pot95_shrink":
        return mahalanobis(b, k=3, pot_level=0.95, scatter="shrink")
    raise ValueError(name)


def one_config(xs, name, warm, tau, score_from, window):
    """Return (argmax_hit, {alpha: clean_FPR})."""
    f = _make(name)
    state = None
    best_s, best_loc = -1.0, -1
    n_clean = 0
    flags = {a: 0 for a in ALPHAS}
    for t, y in enumerate(xs):
        _dists, state = f(y, state)
        p = state["pvalue"]
        if p is None:
            continue
        s = -math.log10(max(p, 1e-300))
        if t >= score_from and s > best_s:       # ranking: global argmax
            best_s, best_loc = s, t
        if warm <= t < tau:                       # calibration: clean region
            n_clean += 1
            for a in ALPHAS:
                if p < a:
                    flags[a] += 1
    lo, hi = window
    return (bool(lo <= best_loc <= hi),
            {a: (flags[a] / n_clean if n_clean else None) for a in ALPHAS})


def run_one(job):
    sid, = job
    levels = _load_levels(sid)
    xs = _to_changes(levels) if levels else []
    if len(xs) < MIN_LEN:
        return None
    xs, kind, (a_s, a_e) = inject(xs, sid)
    n = len(xs)
    warm = int(0.2 * n)
    score_from = int(0.4 * n)
    tau = a_s
    if tau - warm < 200:                          # need a clean stretch
        return None
    tol = max(50, a_e - a_s)
    window = (a_s - tol, a_e + tol)
    t0 = time.time()
    res = {"sid": sid, "n": n, "kind": kind, "tau": tau}
    for name in CONFIGS:
        hit, fpr = one_config(xs, name, warm, tau, score_from, window)
        res[name] = {"hit": hit, "fpr": fpr}
    res["seconds"] = round(time.time() - t0, 1)
    return res


def _median(v):
    s = sorted(x for x in v if x is not None)
    return s[len(s) // 2] if s else float("nan")


def _get(fpr, a):
    return fpr.get(str(a), fpr.get(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(_HERE, "mah_repair_sweep_results.jsonl"))
    args = ap.parse_args()

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
        for i, res in enumerate(pool.imap_unordered(run_one, [(s,) for s in picked]), 1):
            if res is None:
                continue
            results.append(res)
            ofh.write(json.dumps(res) + "\n")
            ofh.flush()
            os.fsync(ofh.fileno())
            print(f"[{i}/{len(picked)}] {res['sid'][:18]:18s} {res['seconds']:5.1f}s",
                  flush=True)
    ofh.close()

    n = len(results)
    print(f"\n=== mah repair sweep (n={n}) — ranking vs calibration ===")
    print(f"{'config':14s} {'argmax':>7s}" + "".join(f"  x@{a:g}".rjust(9) for a in ALPHAS))
    for name in CONFIGS:
        acc = sum(1 for r in results if r[name]["hit"]) / n
        row = f"{name:14s} {acc:7.3f}"
        for a in ALPHAS:
            row += f"{_median([_get(r[name]['fpr'], a) for r in results]) / a:9.1f}"
        print(row)
    print("\nx@ = empirical/nominal clean-region FPR (1.0 = calibrated). "
          "A fix must not drop argmax to buy calibration.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
