"""Front-end experiment: do SOTA streaming detectors work better on the
laplace-transformed series (the parade 1-step z) than on the raw series?

If yes, skaters is not a competitor to these detectors but a normalising
front-end for them: the forecaster removes trend/season/scale so the detector
hunts in a stationarised, unit-scale stream.

Detectors:
    rrcf:  Robust Random Cut Forest (Guha et al. 2016; pip rrcf), canonical
           streaming setup — 40 trees, shingle 4, tree size 256, CoDisp score.
    dspot: DSPOT (Siffer et al. KDD 2017) re-implemented here (no maintained
           PyPI package): EWMA drift removal, POT with method-of-moments GPD
           tails, two-sided; score is -log tail probability.

Conditions: raw y   vs   z (parade 1-step surprise from laplace(3)).
Protocol: UCR argmax with tolerance max(100, L), scored region t >= trainLen,
prefix tail (4000) fed for warm-up only — identical to ucr_run.py.

Usage:
    python benchmarks/anomaly/frontend_run.py --limit 60 --workers 2
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from collections import deque
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from ucr_run import DATA, _NAME, parse_name, load_series  # noqa: E402

WARMUP_TAIL = 4000


# ---------------------------------------------------------------------------
# DSPOT (streaming EVT thresholding), two-sided, method-of-moments GPD.
# ---------------------------------------------------------------------------

class _Tail:
    """One tail's POT state: initial threshold t, excesses, GPD fit."""

    def __init__(self, calib, level):
        s = sorted(calib)
        self.t = s[min(int(level * len(s)), len(s) - 1)]
        self.exc = [x - self.t for x in calib if x > self.t]
        self.n = len(calib)

    def _gpd(self):
        m = sum(self.exc) / len(self.exc)
        v = sum((e - m) ** 2 for e in self.exc) / max(len(self.exc) - 1, 1)
        if v <= 0:
            return 0.0, max(m, 1e-12)
        gamma = 0.5 * (1.0 - m * m / v)
        sigma = 0.5 * m * (m * m / v + 1.0)
        return gamma, max(sigma, 1e-12)

    def logp(self, x):
        """-log tail probability of x (0 when below the POT threshold)."""
        self.n += 1
        if x <= self.t or not self.exc:
            return 0.0
        gamma, sigma = self._gpd()
        zeta = len(self.exc) / self.n            # P(X > t)
        e = x - self.t
        if abs(gamma) < 1e-9:
            lt = -e / sigma
        else:
            arg = 1.0 + gamma * e / sigma
            lt = (-1.0 / gamma) * math.log(arg) if arg > 0 else -745.0
        # log P(X > x) = log zeta + log survival of GPD at e
        self.exc.append(e)
        if len(self.exc) > 500:                  # cap refit cost
            self.exc.pop(0)
        return -(math.log(max(zeta, 1e-300)) + lt)


class DSpot:
    def __init__(self, calib, level=0.98, drift_alpha=0.02):
        self.a = drift_alpha
        self.ma = sum(calib) / len(calib)
        resid = [x - self.ma for x in calib]
        self.up = _Tail(resid, level)
        self.lo = _Tail([-r for r in resid], level)

    def score(self, x):
        r = x - self.ma
        s = max(self.up.logp(r), self.lo.logp(-r))
        self.ma += self.a * r                    # EWMA drift (post-score)
        return s


# ---------------------------------------------------------------------------
# RRCF (canonical streaming CoDisp)
# ---------------------------------------------------------------------------

def rrcf_scores(xs, num_trees=40, shingle=4, tree_size=256, seed=0):
    import random as _random
    import rrcf as _rrcf
    _random.seed(seed)
    forest = [_rrcf.RCTree(random_state=seed + i) for i in range(num_trees)]
    out = [0.0] * len(xs)
    window = deque(maxlen=shingle)
    for i, x in enumerate(xs):
        window.append(x)
        if len(window) < shingle:
            continue
        point = tuple(window)
        total = 0.0
        for tree in forest:
            if len(tree.leaves) > tree_size:
                tree.forget_point(i - tree_size)
            tree.insert_point(point, index=i)
            total += tree.codisp(i)
        out[i] = total / num_trees               # score at shingle END
    return out


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------

def z_series(ys, start):
    """Parade 1-step z from laplace(3), 0.0 where unavailable."""
    from timemachines import laplace
    f = laplace(3)
    state = None
    zs = [0.0] * len(ys)
    for t in range(start, len(ys)):
        _, state = f(ys[t], state)
        z = state["z"][0]
        zs[t] = z if z is not None else 0.0
    return zs


def run_one(args):
    fname, = args
    sid, name, train_len, a_start, a_end = parse_name(fname)
    ys = load_series(os.path.join(DATA, fname))
    n = len(ys)
    start = max(0, train_len - WARMUP_TAIL)
    t0 = time.time()

    zs = z_series(ys, start)
    tol = max(100, a_end - a_start)
    lo, hi = a_start - tol, a_end + tol
    res = {"sid": sid, "name": name, "fname": fname, "n": n}

    for cond, xs in (("raw", ys), ("z", zs)):
        calib = xs[start:train_len]
        # DSPOT
        d = DSpot(list(calib))
        best, loc = -1.0, -1
        for t in range(train_len, n):
            s = d.score(xs[t])
            if s > best:
                best, loc = s, t
        res[f"dspot_{cond}"] = {"loc": loc, "hit": bool(lo <= loc <= hi)}
        # RRCF (feed from `start` so trees warm on the prefix tail)
        scores = rrcf_scores(xs[start:n])
        best, loc = -1.0, -1
        for t in range(train_len, n):
            s = scores[t - start]
            if s > best:
                best, loc = s, t
        res[f"rrcf_{cond}"] = {"loc": loc, "hit": bool(lo <= loc <= hi)}

    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(DATA) if _NAME.match(f))
    files.sort(key=lambda f: os.path.getsize(os.path.join(DATA, f)))
    if args.limit:
        files = files[:args.limit]

    keys = ["dspot_raw", "dspot_z", "rrcf_raw", "rrcf_z"]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "frontend_results.jsonl")

    results = []
    if os.path.exists(out):
        with open(out) as fh:
            results = [json.loads(line) for line in fh if line.strip()]
        done = {r.get("fname") for r in results}
        files = [f for f in files if f not in done]
        print(f"resuming: {len(results)} done in {out}, "
              f"{len(files)} to go", flush=True)
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(
                run_one, [(f,) for f in files])):
            results.append(res)
            with open(out, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            hits = {m: sum(r[m]["hit"] for r in results) for m in keys}
            print(f"[{i+1}/{len(files)}] {res['sid']:03d} "
                  f"{res['name'][:26]:26s} {res['seconds']:6.1f}s  "
                  + "  ".join(f"{m}={'HIT' if res[m]['hit'] else '.'}"
                              for m in keys)
                  + f"   running: " + " ".join(f"{m}:{hits[m]}" for m in keys),
                  flush=True)

    n = len(results)
    print("\n=== UCR accuracy: raw vs laplace-transformed (z) ===")
    for m in keys:
        h = sum(r[m]["hit"] for r in results)
        print(f"{m:12s} {h}/{n} = {h/n:.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
