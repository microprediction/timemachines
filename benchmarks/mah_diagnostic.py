"""Where does the mah (Mahalanobis-layer) overconfidence come from?

Fact (RESULTS.md §4/§8): plain z1 on the default laplace holds ~2x nominal at
1e-3 (≈ the genuine-anomaly base rate in FRED), but the Mahalanobis p-value
(`mah`) runs ~9x on the same data. §4b ruled out the z-stream (sticky) as the
cause, so the extra overshoot is inside the head. The p-value path is:

    d2 (factor scatter, Woodbury)                  <- scatter estimation
      -> Satterthwaite two-moment null c*chi2_nu   <- bulk null
      -> GPD tail beyond the 98% quantile          <- tail machinery
      -> Bonferroni min(p, 2*p_n), nlp channel     <- second tail + multiplicity

The public knobs isolate the stages without reimplementing anything:

  full    defaults (factor scatter, GPD + nlp on)
  bulk    min_exc=1e9 -> disables BOTH the d2 GPD and the nlp GPD channel
          (both gated by len(...) >= min_exc), leaving pure Satterthwaite chi2.
          full vs bulk splits "the tail machinery over-rejects" from
          "d2/null is inflated in the bulk already".
  shrink  scatter="shrink" -> identity-shrinkage instead of the 1-factor
          model; tests whether the factor scatter under-estimates variance and
          inflates d2.
  f2      factors=2 -> tests whether one factor is too few (residual
          cross-horizon correlation left in the diagonal inflates d2).

Per FRED non-price series, one pass per config, paired (same real events, so
the difference isolates the machinery). Report median two-sided... (mah p is
one-sided upper on d2) per-series FPR of state["pvalue"] < alpha at each
nominal alpha, plus the empirical quantile ratio at 1e-3 so we see the size of
the miscalibration directly.

Resumable: one jsonl row per series.

Usage:
    python benchmarks/mah_diagnostic.py --limit 120 --workers 8
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

from fred import _load_levels, _to_changes            # noqa: E402
from fred_anomaly import UNIVERSE, MIN_LEN             # noqa: E402

ALPHAS = (1e-2, 1e-3, 1e-4)
_BIG = 10 ** 9


def _make(name):
    from timemachines import laplace, mahalanobis
    base = lambda: laplace(3)
    if name == "full":
        return mahalanobis(base(), k=3)
    if name == "bulk":
        return mahalanobis(base(), k=3, min_exc=_BIG)   # chi2 only, no GPD/nlp
    if name == "shrink":
        return mahalanobis(base(), k=3, scatter="shrink")
    if name == "f2":
        return mahalanobis(base(), k=3, factors=2)
    raise ValueError(name)


CONFIGS = ("full", "bulk", "shrink", "f2")


def one_config(xs, name):
    f = _make(name)
    state = None
    n_scored = 0
    flags = {a: 0 for a in ALPHAS}
    z1_flags = {a: 0 for a in ALPHAS}
    z1_n = 0
    for y in xs:
        _dists, state = f(y, state)
        p = state["pvalue"]
        if p is not None:
            n_scored += 1
            for a in ALPHAS:
                if p < a:
                    flags[a] += 1
        # plain z1 reference from the same head's parade (one-sided-equiv two
        # tailed erfc), only meaningful once for the 'full' config
        if name == "full":
            z = state["base"]["z"][0] if isinstance(state["base"], dict) else None
            if z is not None and math.isfinite(z):
                z1_n += 1
                pz = math.erfc(abs(z) / math.sqrt(2.0))
                for a in ALPHAS:
                    if pz < a:
                        z1_flags[a] += 1
    out = {"fpr": {a: (flags[a] / n_scored if n_scored else None) for a in ALPHAS},
           "n_scored": n_scored}
    if name == "full":
        out["z1_fpr"] = {a: (z1_flags[a] / z1_n if z1_n else None) for a in ALPHAS}
    return out


def run_one(job):
    sid, = job
    levels = _load_levels(sid)
    xs = _to_changes(levels) if levels else []
    if len(xs) < MIN_LEN:
        return None
    t0 = time.time()
    res = {"sid": sid, "n": len(xs)}
    for name in CONFIGS:
        res[name] = one_config(xs, name)
    res["z1"] = {"fpr": res["full"].pop("z1_fpr")}
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
    ap.add_argument("--out", default=os.path.join(_HERE, "mah_diagnostic_results.jsonl"))
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
    print(f"\n=== mah diagnostic (n={n}) — median per-series FPR / nominal ===")
    print(f"{'config':10s}" + "".join(f"  fpr@{a:g}".rjust(12) for a in ALPHAS)
          + "".join(f"  x@{a:g}".rjust(9) for a in ALPHAS))
    order = ("z1", "bulk", "shrink", "f2", "full")
    for name in order:
        row = f"{name:10s}"
        fprs = [_get(r[name]["fpr"], a) for a in ALPHAS for r in results]
        meds = {a: _median([_get(r[name]["fpr"], a) for r in results]) for a in ALPHAS}
        for a in ALPHAS:
            row += f"{meds[a]:12.2e}"
        for a in ALPHAS:
            row += f"{meds[a] / a:9.1f}"
        print(row)
    print("\nReading: 'x@' = empirical/nominal ratio (1.0 = calibrated). "
          "full vs bulk splits tail-machinery from d2/null; "
          "shrink/f2 test the scatter model.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
