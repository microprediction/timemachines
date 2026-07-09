"""The sufficiency study, corrected: a competent learner this time.

The first pass (sufficiency_study.py) returned a verdict that turned out
to measure the learners, not the information: river's SGD linear model
diverges on this universe and a Hoeffding tree barely leaves its grace
period on two thousand points, so both left strong contemporaneous
sibling signal (median |corr| 0.45) on the table, while a plain RLS
exploits it on 115/126 pairs. This harness re-asks the question with
recursive least squares (forgetting 0.999, running standardization,
windup guard) as the one learner for every condition:

    body    the target's Laplace forecast alone
    five    mu_x, z_x, mu_y, z_y, mu_y_prev (the rollup, built directly
            from two laplace(1) bodies)
    ewma5   the same five shapes from EWMA forecasters
    base1   raw [x_t, y_{t-1}]
    ladder  raw [x_t, x lags 1..4, y lags 1..8]

Same non-price pairs, alignment, variants, metric and resume protocol as
sufficiency_study.py. Usage:

    TIMEMACHINES_FRED_DATA=~/github/skaters/benchmarks/data \\
        python benchmarks/sufficiency_rls.py --pairs 150 --workers 6
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
import zlib
import random as _random
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

from sufficiency_study import (  # noqa: E402
    build_pairs, aligned_changes, family, _EwmaRoll, ZCAP)

CONDS = ("body", "five", "ewma5", "base1", "ladder")
_EPS = 1e-12


class RLS:
    """Forgetting RLS over a feature dict, with running standardization of
    features and target (prequential; unit scale until warm)."""

    def __init__(self, keys, lam=0.999, p0=100.0):
        self.keys = list(keys)
        d = len(self.keys) + 1
        self.d, self.lam = d, lam
        self.w = [0.0] * d
        self.P = [[p0 if i == j else 0.0 for j in range(d)] for i in range(d)]
        self.stats = {k: [0, 0.0, 1.0] for k in self.keys + ["__y"]}

    def _std(self, k):
        n, _, v = self.stats[k]
        return math.sqrt(v) if (n >= 10 and v > 1e-24) else 1.0

    def _upd_stat(self, k, x):
        s = self.stats[k]
        s[0] += 1
        old = s[1]
        s[1] += (x - old) / s[0]
        s[2] += ((x - old) * (x - s[1]) - s[2]) / s[0]

    def _phi(self, x):
        out = [1.0]
        for k in self.keys:
            s = self.stats[k]
            out.append((x.get(k, 0.0) - s[1]) / self._std(k))
        return out

    def predict_one(self, x):
        phi = self._phi(x)
        ys = self.stats["__y"]
        return sum(w * p for w, p in zip(self.w, phi)) * self._std("__y") + ys[1]

    def learn_one(self, x, y):
        phi = self._phi(x)
        ys = self.stats["__y"]
        target = (y - ys[1]) / self._std("__y")
        err = target - sum(w * p for w, p in zip(self.w, phi))
        Pp = [sum(self.P[i][j] * phi[j] for j in range(self.d))
              for i in range(self.d)]
        den = self.lam + sum(p * q for p, q in zip(phi, Pp))
        g = [q / den for q in Pp]
        for i in range(self.d):
            self.w[i] += g[i] * err
        for i in range(self.d):
            for j in range(self.d):
                self.P[i][j] = (self.P[i][j] - g[i] * Pp[j]) / self.lam
        tr = sum(self.P[i][i] for i in range(self.d))
        if not math.isfinite(tr) or tr > 1e7:
            self.__init__(self.keys, self.lam)
        for k in self.keys:
            self._upd_stat(k, x.get(k, 0.0))
        self._upd_stat("__y", y)


def run_one(job):
    y_id, x_id, variant = job
    ys, xs = aligned_changes(y_id, x_id)
    if ys is None:
        return None
    from skaters import laplace
    from skaters.dist import Dist
    std_normal = Dist.gaussian(0.0, 1.0)

    n = len(ys)
    burn = max(100, n // 10)
    t0 = time.time()

    if variant == "spikes":
        rng = _random.Random(zlib.crc32(f"{y_id}:{x_id}".encode()))
        mu = sum(xs) / n
        sd = math.sqrt(sum((v - mu) ** 2 for v in xs) / n) or 1e-12
        xs = [v + ((6.0 + 4.0 * rng.random()) * sd * rng.choice((-1, 1))
                   if rng.random() < 0.02 else 0.0) for v in xs]

    models = {
        "five": RLS(["mu_x", "z_x", "mu_y", "z_y", "mu_y_prev"]),
        "ewma5": RLS(["mu_x", "z_x", "mu_y", "z_y", "mu_y_prev"]),
        "base1": RLS(["x", "y_1"]),
        "ladder": RLS(["x"] + [f"x_{l}" for l in range(1, 5)]
                      + [f"y_{l}" for l in range(1, 9)]),
    }
    roll = _EwmaRoll()
    ew_zy = 0.0

    f = laplace(1)
    st_y = st_x = None
    pend_y = pend_x = None
    prev_mu_y = 0.0
    zy = 0.0

    acc = {c: {"ae": 0.0, "n": 0} for c in CONDS}

    for t in range(n):
        # x body consumes the current x (known at prediction time)
        mu_x = pend_x.mean if pend_x is not None else xs[t]
        if pend_x is not None:
            u = min(max(pend_x.cdf(xs[t]), _EPS), 1.0 - _EPS)
            z_x = std_normal.quantile(u)
        else:
            z_x = 0.0
        dists_x, st_x = f(xs[t], st_x)

        mu_y = pend_y.mean if pend_y is not None else 0.0
        five = {"mu_x": mu_x, "z_x": z_x, "mu_y": mu_y,
                "z_y": zy, "mu_y_prev": prev_mu_y}

        emu_x, ez_x = roll.pair("x", xs[t], update=False)
        emu_y, _ = roll.pair("__y", ys[t], update=False)
        ew5 = {"mu_x": emu_x, "z_x": ez_x, "mu_y": emu_y,
               "z_y": ew_zy, "mu_y_prev": roll.mu_prev_y}

        base1 = {"x": xs[t], "y_1": ys[t - 1] if t >= 1 else 0.0}
        lad = {"x": xs[t]}
        for l in range(1, 5):
            lad[f"x_{l}"] = xs[t - l] if t >= l else 0.0
        for l in range(1, 9):
            lad[f"y_{l}"] = ys[t - l] if t >= l else 0.0

        feats = {"five": five, "ewma5": ew5, "base1": base1, "ladder": lad}
        for c in CONDS:
            p = mu_y if c == "body" else models[c].predict_one(feats[c])
            if t >= burn:
                acc[c]["ae"] += abs(p - ys[t])
                acc[c]["n"] += 1
        for c in ("five", "ewma5", "base1", "ladder"):
            models[c].learn_one(feats[c], ys[t])

        # advance the target body and the ewma roll
        prev_mu_y = mu_y if pend_y is not None else prev_mu_y
        if pend_y is not None:
            u = min(max(pend_y.cdf(ys[t]), _EPS), 1.0 - _EPS)
            zy = std_normal.quantile(u)
        dists_y, st_y = f(ys[t], st_y)
        pend_y = dists_y[0]
        pend_x = dists_x[0]
        roll.pair("x", xs[t], update=True)
        roll.mu_prev_y = roll.m.get("__y", 0.0)
        _, ew_zy = roll.pair("__y", ys[t], update=True)

    res = {"y": y_id, "x": x_id, "variant": variant, "n": n,
           "fam": family(y_id)}
    for c in CONDS:
        res[c] = acc[c]["ae"] / max(acc[c]["n"], 1)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def summarize(results):
    results = [r for r in results if r]
    print(f"\n=== sufficiency, RLS learner (n={len(results)}) ===")
    for variant in ("asis", "spikes"):
        rows = [r for r in results if r["variant"] == variant]
        if not rows:
            continue
        best = [min(r["base1"], r["ladder"]) for r in rows]
        ratios = sorted(r["five"] / max(b, 1e-15) for r, b in zip(rows, best))
        med = ratios[len(ratios) // 2]
        suff = sum(1 for x in ratios if x <= 1.05) / len(ratios)
        fams = {}
        for r, b in zip(rows, best):
            fams.setdefault(r["fam"], []).append(r["five"] <= 1.05 * b)
        fw = sum(sum(v) / len(v) for v in fams.values()) / len(fams)
        w1 = sum(1 for r in rows if r["five"] < r["base1"])
        wl = sum(1 for r in rows if r["five"] < r["ladder"])
        we = sum(1 for r in rows if r["five"] < r["ewma5"])
        wb = sum(1 for r in rows if r["five"] < r["body"])
        rb = sum(1 for r, b in zip(rows, best) if b < r["body"])
        print(f"  {variant} ({len(rows)} pairs): median five/best-raw {med:.3f}; "
              f"sufficient {100*suff:.0f}% / {100*fw:.0f}% fam-weighted; "
              f"five wins vs base1 {w1}, ladder {wl}, ewma5 {we}, body {wb}; "
              f"best-raw beats body on {rb}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=150)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, "sufficiency_rls_results.jsonl"))
    args = ap.parse_args()

    pairs = build_pairs(args.pairs)
    print(f"{len(pairs)} non-price pairs", flush=True)
    jobs = [(y, x, v) for (y, x) in pairs for v in ("asis", "spikes")]

    results = []
    if os.path.exists(args.out):
        with open(args.out) as fh:
            results = [json.loads(line) for line in fh if line.strip()]
        done = {(r["y"], r["x"], r["variant"]) for r in results}
        jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
        print(f"resuming: {len(results)} done, {len(jobs)} to go", flush=True)

    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, jobs)):
            if res is None:
                continue
            results.append(res)
            with open(args.out, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            print(f"[{i + 1}/{len(jobs)}] {res['y'][:14]:14s}~{res['x'][:14]:14s} "
                  f"{res['variant']:6s} {res['seconds']:6.1f}s", flush=True)

    summarize(results)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
