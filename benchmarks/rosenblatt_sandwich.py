"""The Rosenblatt sandwich as an actual skater: laplace -> inner -> laplace^-1.

The fronted log-lik studies scored the sandwich implicitly (change of
variables is exact for the density at the realized point). What they cannot
score is everything needing the pushforward PREDICTIVE as an object: CRPS,
intervals, point forecasts. This builds it, one-step (where the composition
is exact) and measures CRPS + log-lik on price series.

Construction, per tick:
    z_t   = Phi^-1(F_t(y_t))          laplace's parade (the forward map)
    Q_z   = inner(z_t)                a z-space predictive (garch_leaf: an
                                      online GARCH(1,1)-with-t-ish-tails)
    Q_y   = (F_t^-1 o Phi)_* Q_z      pushforward by quantile mapping: a
                                      probability grid through the z-space
                                      quantiles, then Phi, then laplace's
                                      inverse cdf; refit as a mixture.

Rows: laplace alone | sandwich | garch_leaf directly on y (reference).
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

from fred import _load_levels, _to_changes  # noqa: E402

BURN = 500
N_MAX = 3000
# probability grid for the pushforward (dense in the tails, where CRPS lives)
_PGRID = ([0.001, 0.005, 0.01, 0.025, 0.05]
          + [0.05 + 0.9 * i / 30 for i in range(1, 30)]
          + [0.95, 0.975, 0.99, 0.995, 0.999])


def _grid_to_dist(qs, ps):
    """Quantile grid -> Dist: mass between quantiles as narrow components."""
    from skaters.dist import Dist
    comps = []
    for i in range(len(qs) - 1):
        w = ps[i + 1] - ps[i]
        if w <= 0:
            continue
        m = 0.5 * (qs[i] + qs[i + 1])
        s = max((qs[i + 1] - qs[i]) / 3.5, 1e-12)
        comps.append((w, m, s))
    return Dist(comps)


def sandwich(k1_inner_factory):
    """laplace -> inner (z-space) -> pushforward, as a skater (k=1, exact)."""
    from skaters import laplace
    from skaters.dist import Dist
    std_normal = Dist.gaussian(0.0, 1.0)
    eng = laplace(1)
    inner = k1_inner_factory(k=1)

    def _skater(y, state):
        if state is None:
            state = {"eng": None, "in": None}
        dists_y, state["eng"] = eng(y, state["eng"])
        z = state["eng"]["z"][0]
        zdists, state["in"] = inner(z if z is not None else 0.0, state["in"])
        F = dists_y[0]                      # tomorrow's y-space cdf (laplace)
        zd = zdists[0]                      # tomorrow's z-space predictive
        qs = []
        for p in _PGRID:
            u = std_normal.cdf(zd.quantile(p))
            u = min(max(u, 1e-9), 1.0 - 1e-9)
            qs.append(F.quantile(u))
        return [_grid_to_dist(qs, _PGRID)], state

    return _skater


def run_one(args):
    sid, = args
    levels = _load_levels(sid)
    ys = _to_changes(levels) if levels else []
    if len(ys) < BURN + 500:
        return None
    ys = ys[-N_MAX:]
    n = len(ys)
    t0 = time.time()

    from skaters import laplace
    from skaters.leaf import garch_leaf

    models = {
        "laplace": laplace(1),
        "sandwich": sandwich(garch_leaf),
        "garch_leaf_y": garch_leaf(k=1),
    }
    res = {"sid": sid, "n": n}
    for name, f in models.items():
        state, pend = None, None
        ll, crps, cnt = 0.0, 0.0, 0
        for t, y in enumerate(ys):
            if pend is not None and t >= BURN:
                lp = pend.logpdf(y)
                ll += max(min(lp, 20.0), -20.0)
                crps += pend.crps(y)
                cnt += 1
            dists, state = f(y, state)
            pend = dists[0]
        res[name] = {"ll": ll / cnt, "crps": crps / cnt}
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="SP500,DJIA,NASDAQCOM,DEXUSEU,DEXJPUS,"
                    "DEXUSUK,DEXCAUS,DEXSZUS,DCOILWTICO,DCOILBRENTEU,"
                    "DHHNGSP,VIXCLS")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    ids = args.ids.split(",")

    results = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(run_one, [(s,) for s in ids]):
            if r is None:
                continue
            results.append(r)
            print(f"{r['sid'][:14]:14s} {r['seconds']:6.1f}s  "
                  + "  ".join(f"{m}: ll={r[m]['ll']:+.3f} crps={r[m]['crps']:.5f}"
                              for m in ("laplace", "sandwich")), flush=True)

    out = os.path.join(_HERE, f"rosenblatt_sandwich_n{len(results)}.jsonl")
    with open(out, "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    n = len(results)
    print(f"\n=== one-step, y-space, n={n} price series ===")
    print(f"{'model':14s} {'log-lik':>9s} {'CRPS':>10s} {'ll wins vs laplace':>20s}")
    for m in ("laplace", "sandwich", "garch_leaf_y"):
        ll = sum(r[m]["ll"] for r in results) / n
        cr = sum(r[m]["crps"] for r in results) / n
        wins = sum(r[m]["ll"] > r["laplace"]["ll"] for r in results)
        tag = "" if m == "laplace" else f"{wins}/{n}"
        print(f"{m:14s} {ll:9.3f} {cr:10.6f} {tag:>20s}")
    cwins = sum(r["sandwich"]["crps"] < r["laplace"]["crps"] for r in results)
    print(f"\nsandwich CRPS wins vs laplace: {cwins}/{n}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
