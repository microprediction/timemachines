"""The real-data size check for the calibrated Granger test.

On the 126 non-price FRED sibling pairs, lag-1 cross-correlation is
essentially absent (median |corr| 0.012), so the lagged-causality null
approximately holds and every test SHOULD reject near its nominal 5% on
these 252 directed pairs. A test that rejects at 10% on real economic
data is generating false discoveries in the wild. Degenerate pairs
(constant stretches making a regression fit exactly) are skipped and
counted.

    TIMEMACHINES_FRED_DATA=~/github/skaters/benchmarks/data \\
        python benchmarks/granger_fred.py --workers 6
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

from sufficiency_study import build_pairs, aligned_changes  # noqa: E402
from granger_overnight import (  # noqa: E402
    _ols_tests, _ar_resid, _ewma_std, _garch_fit_std, _ccf_pvals, _z2_pvals)

TESTS = ("pF", "pHAC", "pF1", "pCNE", "pCNG", "pZ2")


def run_one(job):
    y_id, x_id, direction = job
    ys, xs = aligned_changes(y_id, x_id)
    if ys is None:
        return None
    if direction == "xy":
        ys, xs = xs, ys
        y_id, x_id = x_id, y_id
    try:
        pF, pH, pF1, _ = _ols_tests(ys, xs)
        cne = _ccf_pvals(_ewma_std(_ar_resid(ys)), _ewma_std(_ar_resid(xs)))[0]
        cng = _ccf_pvals(_garch_fit_std(_ar_resid(ys)),
                         _garch_fit_std(_ar_resid(xs)))[0]
        z2 = _z2_pvals(ys, xs)[0]
    except (ZeroDivisionError, ValueError, OverflowError) as e:
        return {"y": y_id, "x": x_id, "skip": str(type(e).__name__)}
    return {"y": y_id, "x": x_id, "pF": pF, "pHAC": pH, "pF1": pF1,
            "pCNE": cne, "pCNG": cng, "pZ2": z2}


def summarize(results):
    ok = [r for r in results if r and "skip" not in r]
    skipped = [r for r in results if r and "skip" in r]
    print(f"\n=== FRED real-data rejection at nominal 5% "
          f"({len(ok)} directed pairs; {len(skipped)} degenerate skipped) ===")
    print("(the lagged-causality null approximately holds here: median "
          "lag-1 |corr| 0.012)")
    for k in TESTS:
        rate = sum(1 for r in ok if r[k] < 0.05) / max(len(ok), 1)
        print(f"  {k:6s} {rate:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, "granger_fred_results.jsonl"))
    args = ap.parse_args()

    pairs = build_pairs(150)
    jobs = [(y, x, d) for (y, x) in pairs for d in ("yx", "xy")]
    results = []
    if os.path.exists(args.out):
        with open(args.out) as fh:
            results = [json.loads(line) for line in fh if line.strip()]
        done = {(r["y"], r["x"]) for r in results}
        jobs = [j for j in jobs
                if ((j[0], j[1]) if j[2] == "yx" else (j[1], j[0])) not in done]
        print(f"resuming: {len(results)} done, {len(jobs)} to go", flush=True)

    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, jobs)):
            if res is None:
                continue
            results.append(res)
            with open(args.out, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            if (i + 1) % 40 == 0:
                print(f"[{i + 1}/{len(jobs)}]", flush=True)

    summarize(results)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
