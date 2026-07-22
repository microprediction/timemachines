"""Detection delay at fixed alarm budget — the other axis of the Wald tradeoff.

The calibration panel measures false-alarm rate vs nominal alpha on clean
stretches; this harness measures how FAST each method raises its first alarm
after a real onset, at a matched alarm budget. FRED injection backgrounds
(same deterministic seeds as fred_anomaly.py), burst and shift injections
only — spikes are instant-or-never and carry no delay information.

Two threshold regimes per method and alpha in {1e-2, 1e-3}:
  * stated:  methods that emit a p-value (mah, z1 via erfc) alarm at
             p < alpha directly — the deployable, no-oracle regime.
  * matched: every method (including score-only ones: mz, dspot, rrcf)
             alarms above the empirical (1 - alpha) quantile of its own
             scores on the calibration region [0.2n, 0.4n). This hands
             score methods free calibration (RESEARCH.md pitfall), stated
             plainly — it is the fair axis for ranking delay.

Per series and method: first-alarm delay after onset (censored at n),
pre-onset false alarms per 1k ticks (the realized budget). Resumable: one
jsonl row per series.

Usage:
    python benchmarks/detection_delay.py --limit 150 --workers 2
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
from fred_anomaly import inject, UNIVERSE, MIN_LEN  # noqa: E402

ALPHAS = (1e-2, 1e-3)


def quantile(xs, q):
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(q * len(s))))
    return s[i]


def delay_stats(scores, pvals, n, calib_lo, score_from, tau):
    """Per method-channel: {regime_alpha: {delay, fa_per_1k}}."""
    out = {}
    for alpha in ALPHAS:
        atag = f"{alpha:g}"
        if pvals is not None:
            alarms = [t for t in range(score_from, n) if pvals[t] is not None
                      and pvals[t] < alpha]
            out[f"stated_{atag}"] = _summ(alarms, tau, score_from, n)
        thr = quantile(scores[calib_lo:score_from], 1.0 - alpha)
        alarms = [t for t in range(score_from, n) if scores[t] > thr]
        out[f"matched_{atag}"] = _summ(alarms, tau, score_from, n)
    return out


def _summ(alarms, tau, score_from, n):
    pre = [t for t in alarms if t < tau]
    post = [t for t in alarms if t >= tau]
    pre_ticks = tau - score_from
    return {"delay": (post[0] - tau) if post else None,
            "fa_per_1k": round(1000.0 * len(pre) / max(pre_ticks, 1), 3)}


def run_one(args):
    sid, = args
    levels = _load_levels(sid)
    xs = _to_changes(levels) if levels else []
    if len(xs) < MIN_LEN:
        return None
    xs, kind, (a_s, _a_e) = inject(xs, sid)
    if kind == "spike":
        return None                      # no delay information
    n = len(xs)
    calib_lo, score_from, tau = int(0.2 * n), int(0.4 * n), a_s
    t0 = time.time()

    from timemachines import laplace, mahalanobis

    f = mahalanobis(laplace(3), k=3)
    state = None
    mah_p = [None] * n
    z1_p = [None] * n
    mah_s = [0.0] * n
    z1_s = [0.0] * n
    mz = [0.0] * n
    zs = [0.0] * n
    mz_m, mz_v, mz_n = 0.0, 0.0, 0
    for t, y in enumerate(xs):
        _, state = f(y, state)
        z = state["base"]["z"][0]
        zs[t] = z if z is not None else 0.0
        if z is not None:
            z1_s[t] = abs(z)
            z1_p[t] = math.erfc(abs(z) / math.sqrt(2.0))
        p = state["pvalue"]
        if p is not None:
            mah_p[t] = p
            mah_s[t] = -math.log10(max(p, 1e-300))
        mz_n += 1
        a = max(0.02, 1.0 / mz_n)
        if mz_n > 3 and mz_v > 0:
            # denormal-tiny variance can overflow the division to inf
            mz[t] = min(abs(y - mz_m) / math.sqrt(mz_v), 1e12)
        d = y - mz_m
        mz_m += a * d
        mz_v = (1 - a) * mz_v + a * d * (y - mz_m)

    channels = {"mah": (mah_s, mah_p), "z1": (z1_s, z1_p), "mz": (mz, None)}
    for cond, series in (("raw", xs), ("z", zs)):
        d = DSpot(list(series[:score_from]))
        ds = [0.0] * n
        for t in range(score_from, n):
            ds[t] = d.score(series[t])
        channels[f"dspot_{cond}"] = (ds, None)
        channels[f"rrcf_{cond}"] = (rrcf_scores(series), None)

    res = {"sid": sid, "n": n, "kind": kind, "tau": tau}
    for m, (s, p) in channels.items():
        res[m] = delay_stats(s, p, n, calib_lo, score_from, tau)
    res["seconds"] = round(time.time() - t0, 1)
    return res


KEYS = ("mah", "z1", "mz", "dspot_raw", "dspot_z", "rrcf_raw", "rrcf_z")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(
        _HERE, "detection_delay_results.jsonl"))
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
    print(f"\n=== detection delay, burst+shift injections (n={n}) ===")
    for regime in [f"{r}_{a:g}" for r in ("stated", "matched") for a in ALPHAS]:
        print(f"\n-- {regime} --")
        print(f"{'method':10s} {'detect':>7s} {'med delay':>10s} {'fa/1k':>7s}")
        for m in KEYS:
            rows = [r[m][regime] for r in results if regime in r[m]]
            if not rows:
                continue
            dl = sorted(r["delay"] for r in rows if r["delay"] is not None)
            det = len(dl) / len(rows)
            med = dl[len(dl) // 2] if dl else float("nan")
            fa = sorted(r["fa_per_1k"] for r in rows)[len(rows) // 2]
            print(f"{m:10s} {det:7.3f} {med:10.1f} {fa:7.2f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
