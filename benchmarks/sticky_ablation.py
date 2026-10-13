"""Does laplace's lattice projection (`sticky`) pollute the detection z-stream?

Hypothesis (Cotton): `sticky=True` (on by default) adds near-Dirac atoms at
revisited values. On lattice series (administrative rates whose CHANGE series
is a run of exact zeros, quoted-on-a-grid variables) an atom is a step in the
predictive CDF; a normal point landing near-but-not-on the atom is mapped
through a locally near-vertical F_t and gets a spurious large |z| (2-4 sigma).
Those pollute the SHOULDER / near-tail of the z distribution at 1e-2..1e-3 —
which the GPD tail splice cannot repair, because the splice only replaces
density BEYOND the frozen tail region and atoms sit in the interior. So the
forecasting win of `sticky` (CRPS/likelihood on grid series) may be a
detection-calibration loss.

Design. Four configs = sticky {on,off} x tails {gpd,gaussian}. Per FRED
non-price change series, one laplace pass per config, strictly causal
(score-before-update is internal to laplace). Report, PAIRED per series (both
configs see the same real events, so the difference isolates the sticky
artifact):
  * empirical two-sided false-alarm rate of erfc(|z1|) at nominal alpha
    (z1 = state["z"][0]);
  * mean 1-step log-likelihood (the forecasting cost of turning sticky off).
Each series is tagged lattice vs continuous by the exact-repeat fraction of
its change series. Falsifiable control: on CONTINUOUS series sticky must make
~no difference; any effect must concentrate on LATTICE series.

Resumable: one jsonl row per series.

Usage:
    python benchmarks/sticky_ablation.py --limit 150 --workers 8
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
CONFIGS = (("sticky_gpd", True, "gpd"),
           ("nosticky_gpd", False, "gpd"),
           ("sticky_gauss", True, "gaussian"),
           ("nosticky_gauss", False, "gaussian"))


def repeat_frac(xs):
    """Fraction of values that are exact repeats of an earlier value — the
    cheap lattice proxy. High on grid/administrative series, ~0 on continuous."""
    seen = set()
    rep = 0
    for v in xs:
        if v in seen:
            rep += 1
        else:
            seen.add(v)
    return rep / len(xs)


def one_config(xs, sticky, tails):
    from timemachines import laplace
    f = laplace(3, sticky=sticky, tails=tails)
    state = None
    n_scored = 0
    flags = {a: 0 for a in ALPHAS}
    ll_sum, ll_n = 0.0, 0
    prev = None
    for y in xs:
        # score y under the predictive issued last tick (score-before-update)
        if prev is not None:
            lp = prev.logpdf(y)
            if lp is not None and math.isfinite(lp):
                ll_sum += max(lp, -20.0)
                ll_n += 1
        dists, state = f(y, state)
        prev = dists[0]
        z = state["z"][0]
        if z is not None:
            n_scored += 1
            p = math.erfc(abs(z) / math.sqrt(2.0))   # two-sided
            for a in ALPHAS:
                if p < a:
                    flags[a] += 1
    fpr = {a: (flags[a] / n_scored if n_scored else None) for a in ALPHAS}
    return {"fpr": fpr, "n_scored": n_scored,
            "mean_ll": (ll_sum / ll_n if ll_n else None)}


def run_one(job):
    sid, = job
    levels = _load_levels(sid)
    xs = _to_changes(levels) if levels else []
    if len(xs) < MIN_LEN:
        return None
    t0 = time.time()
    res = {"sid": sid, "n": len(xs), "repeat_frac": round(repeat_frac(xs), 4)}
    for name, sticky, tails in CONFIGS:
        res[name] = one_config(xs, sticky, tails)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def _median(v):
    s = sorted(x for x in v if x is not None)
    return s[len(s) // 2] if s else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lattice-thresh", type=float, default=0.05,
                    help="repeat_frac above this = lattice series")
    ap.add_argument("--out", default=os.path.join(_HERE, "sticky_ablation_results.jsonl"))
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
            print(f"[{i}/{len(picked)}] {res['sid'][:18]:18s} "
                  f"rep={res['repeat_frac']:.2f} {res['seconds']:5.1f}s", flush=True)
    ofh.close()

    lat = [r for r in results if r["repeat_frac"] > args.lattice_thresh]
    con = [r for r in results if r["repeat_frac"] <= args.lattice_thresh]
    print(f"\n=== sticky ablation (n={len(results)}: "
          f"{len(lat)} lattice, {len(con)} continuous) ===")
    for label, grp in (("LATTICE (atoms fire)", lat), ("CONTINUOUS (control)", con)):
        if not grp:
            continue
        print(f"\n-- {label}, n={len(grp)} --")
        print(f"{'config':16s} {'ll':>8s}" + "".join(f"  fpr@{a:g}".rjust(12) for a in ALPHAS))
        for name, _s, _t in CONFIGS:
            ll = _median([r[name]["mean_ll"] for r in grp])
            row = f"{name:16s} {ll:8.4f}"
            for a in ALPHAS:
                row += f"{_median([r[name]['fpr'][str(a)] if str(a) in r[name]['fpr'] else r[name]['fpr'][a] for r in grp]):12.2e}"
            print(row)
        # paired sticky-on minus off at 1e-3, gpd tails
        d = [r["sticky_gpd"]["fpr"].get("0.001", r["sticky_gpd"]["fpr"].get(1e-3))
             - r["nosticky_gpd"]["fpr"].get("0.001", r["nosticky_gpd"]["fpr"].get(1e-3))
             for r in grp
             if r["sticky_gpd"]["fpr"] and r["nosticky_gpd"]["fpr"]]
        worse = sum(1 for x in d if x > 0)
        print(f"  paired @1e-3 gpd: sticky-on FPR minus off, median {_median(d):+.2e}; "
              f"sticky-on worse on {worse}/{len(d)}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
