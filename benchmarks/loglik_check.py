"""Quick check: does search() lift one-step log-likelihood over laplace on
UCR-style series? Detection quality is bounded by forecast quality; if search
doesn't lift log-lik on these series, its periodicity discovery isn't landing
and there is nothing for the detector to inherit.

Uses the first 20k points of a handful of representative series, scores mean
one-step logpdf over the second half (first half = warm-up).
"""
import os
import sys
import math
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from ucr_run import DATA, load_series  # noqa: E402

PICKS = [  # one per family: temperature, bleeding, MARS, EPG, ECG, gait
    "115_UCR_Anomaly_CIMIS44AirTemperature3_4000_6520_6544.txt",
    "141_UCR_Anomaly_InternalBleeding5_4000_6200_6370.txt",
    "157_UCR_Anomaly_TkeepFirstMARS_3500_5365_5380.txt",
    "174_UCR_Anomaly_insectEPG2_3700_8000_8025.txt",
    "182_UCR_Anomaly_qtdbSel1005V_4000_12400_12800.txt",
    "249_UCR_Anomaly_weallwalk_2951_7290_7296.txt",
]
N_MAX = 20000


def run_one(fname):
    from skaters import search
    from timemachines import laplace
    ys = load_series(os.path.join(DATA, fname))[:N_MAX]
    half = len(ys) // 2
    out = {}
    for label, f in (("laplace", laplace(3)), ("search", search(k=3))):
        state = None
        pend = None
        tot, cnt = 0.0, 0
        for t, y in enumerate(ys):
            if pend is not None and t >= half:
                lp = max(pend.logpdf(y), -20.0)
                tot += lp
                cnt += 1
            dists, state = f(y, state)
            pend = dists[0]
        out[label] = tot / cnt
    short = fname.split("_")[3]
    return short, out["laplace"], out["search"]


if __name__ == "__main__":
    with Pool(3) as pool:
        rows = pool.map(run_one, [f for f in PICKS
                                  if os.path.exists(os.path.join(DATA, f))])
    print(f"{'series':32s} {'laplace':>9s} {'search':>9s} {'lift':>7s}")
    for name, ll, ls in rows:
        print(f"{name:32s} {ll:9.3f} {ls:9.3f} {ls - ll:+7.3f}")
