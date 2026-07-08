"""The calibration panel: is the advertised false-alarm rate the real one?

The package's central claim is that alarming on ``p < alpha`` yields a
false-alarm rate of ~alpha with no tuning. No mainstream anomaly benchmark
measures this (RESEARCH.md section 2), so this panel does, prequentially:

- Data: the UCR training prefixes — anomaly-free by construction, real-world,
  250 of them. Warm up on the first WARM points of each prefix, then score up
  to CAP clean points. Every flag on this data is a false alarm.
- wald: flag at p < alpha. Nominal rate = alpha by construction.
- DSPOT: the only baseline with a nominal knob — its EVT risk q is a claimed
  per-tail flag rate; we set q = alpha/2 per tail so the two-sided nominal is
  alpha. (Flag when the tail probability of the drift-adjusted point is
  below alpha/2 on either side.)
- RRCF: no probability semantics — deployed the only way an uncalibrated
  score can be: pre-commit a threshold at the empirical (1 - alpha) quantile
  of the first half of the clean stretch, measure the flag rate on the
  second half (threshold transfer). This row is the panel's point.

Report, per nominal alpha: aggregate empirical FPR, the ratio
empirical/nominal (1.0 = perfectly calibrated), and the per-series median
ratio. All strictly causal; thresholds and nulls never see the data they
are scored on.

Usage:
    python benchmarks/calibration_panel.py --limit 120 --workers 6 [--rrcf]
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
from frontend_run import DSpot  # noqa: E402

ALPHAS = (1e-2, 1e-3, 1e-4)
WARM = 4000
CAP = 10000


def run_one(args):
    fname, use_rrcf = args
    sid, name, train_len, _, _ = parse_name(fname)
    if train_len < WARM + 2000:
        return None
    ys = load_series(os.path.join(DATA, fname))[:train_len]  # clean prefix only
    lo = max(0, train_len - (WARM + CAP))
    xs = ys[lo:]
    warm_end = WARM
    n_scored = len(xs) - warm_end
    t0 = time.time()

    from timemachines import wald
    res = {"sid": sid, "n_scored": n_scored}

    # --- wald: p-values on the clean stretch ---
    f = wald(k=3)
    state = None
    pvals = []
    for t, y in enumerate(xs):
        _, state = f(y, state)
        if t >= warm_end and state["pvalue"] is not None:
            pvals.append(state["pvalue"])
    res["wald"] = {f"{a:g}": sum(1 for p in pvals if p < a) for a in ALPHAS}
    res["wald_n"] = len(pvals)

    # --- DSPOT: q = alpha/2 per tail; score = -ln(min tail prob) ---
    d = DSpot(list(xs[:warm_end]))
    scores = [d.score(x) for x in xs[warm_end:]]
    res["dspot"] = {f"{a:g}": sum(1 for s in scores if s > -math.log(a / 2.0))
                    for a in ALPHAS}
    res["dspot_n"] = len(scores)

    # --- RRCF: threshold transfer (first half sets, second half measures) ---
    if use_rrcf:
        from frontend_run import rrcf_scores
        sc = rrcf_scores(xs)[warm_end:]
        half = len(sc) // 2
        cal, ev = sorted(sc[:half]), sc[half:]
        out = {}
        for a in ALPHAS:
            idx = min(int((1.0 - a) * len(cal)), len(cal) - 1)
            thr = cal[idx]
            out[f"{a:g}"] = sum(1 for s in ev if s > thr)
        res["rrcf"] = out
        res["rrcf_n"] = len(ev)

    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rrcf", action="store_true",
                    help="include the RRCF threshold-transfer row (slow)")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(DATA) if _NAME.match(f))
    # need a prefix long enough to warm up AND score; prefer small files
    files = [f for f in files if parse_name(f)[2] >= WARM + 2000]
    files.sort(key=lambda f: os.path.getsize(os.path.join(DATA, f)))
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} series with prefixes >= {WARM + 2000}")

    results = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(
                run_one, [(f, args.rrcf) for f in files])):
            if r is None:
                continue
            results.append(r)
            print(f"[{i+1}/{len(files)}] {r['sid']:03d} "
                  f"scored={r['wald_n']:6d} {r['seconds']:6.1f}s", flush=True)

    out = os.path.join(_HERE, f"calibration_panel_n{len(results)}.jsonl")
    with open(out, "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    methods = ["wald", "dspot"] + (["rrcf"] if args.rrcf else [])
    print(f"\n=== calibration panel: empirical FPR / nominal alpha "
          f"(n={len(results)} clean prefixes; 1.0 = calibrated) ===")
    hdr = f"{'method':8s}" + "".join(f"{f'a={a:g}':>16s}" for a in ALPHAS)
    print(hdr)
    for m in methods:
        cells = []
        for a in ALPHAS:
            key = f"{a:g}"
            flags = sum(r[m][key] for r in results)
            pts = sum(r[f"{m}_n"] for r in results)
            emp = flags / pts if pts else float("nan")
            ratios = sorted((r[m][key] / r[f"{m}_n"]) / a
                            for r in results if r[f"{m}_n"])
            med = ratios[len(ratios) // 2]
            cells.append(f"{emp/a:7.2f}x med{med:5.1f}")
        print(f"{m:8s}" + "".join(f"{c:>16s}" for c in cells))
    print(f"\n(ratio = aggregate empirical FPR / nominal; med = per-series median)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
