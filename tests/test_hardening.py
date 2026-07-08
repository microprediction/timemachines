"""Hardening: the machine must survive what real streams actually contain —
NaNs, infinities, dead channels, unit changes, checkpoint/restore — without
crashing, poisoning its state, or emitting an out-of-range p-value."""

import math
import pickle
import random
from timemachines import wald


def _quiet(f, state, n, scale=0.1, seed=None):
    if seed is not None:
        random.seed(seed)
    for _ in range(n):
        _, state = f(random.gauss(0, scale), state)
    return state


def test_nonfinite_inputs_do_not_poison_or_crash():
    """NaN/inf ticks are skipped (counted), forecasts held, and the detector
    still works afterwards — including firing on a real outlier."""
    f = wald(k=2)
    state = _quiet(f, None, 200, seed=3)
    for bad in (float("nan"), float("inf"), float("-inf")):
        dists, state = f(bad, state)
        assert state["pvalue"] is None
        assert all(math.isfinite(d.mean) and math.isfinite(d.std) for d in dists)
    assert state["skipped"] == 3
    # recovery: still calibrated and still fires
    state = _quiet(f, state, 50, seed=4)
    assert state["pvalue"] is not None and 0.0 <= state["pvalue"] <= 1.0
    _, state = f(30.0, state)
    assert state["pvalue"] < 1e-6


def test_nan_on_first_tick():
    f = wald(k=3)
    dists, state = f(float("nan"), None)
    assert len(dists) == 3
    assert state["skipped"] == 1
    _, state = f(0.1, state)          # and the stream proceeds normally
    assert state["pvalue"] is None or 0.0 <= state["pvalue"] <= 1.0


def test_pvalue_bounds_under_hostile_stream():
    """Fuzz with regime chaos: huge jumps, dead-constant stretches, tiny
    scales, exact repeats. Every emitted p-value in [0,1], every d2 finite,
    every forecast finite."""
    random.seed(11)
    f = wald(k=3)
    state = None
    stream = ([0.0] * 40 + [1e12, -1e12, 0.0, 5.0] + [3.3] * 60
              + [random.gauss(0, 1e-9) for _ in range(80)]
              + [random.gauss(1e6, 1e4) for _ in range(80)])
    for y in stream:
        dists, state = f(y, state)
        assert all(math.isfinite(d.mean) for d in dists)
        if state["pvalue"] is not None:
            assert 0.0 <= state["pvalue"] <= 1.0
            assert math.isfinite(state["d2"])


def test_state_pickles_and_resumes_identically():
    """Checkpoint/restore is how streaming systems deploy: a pickled state
    must resume to bit-identical scores."""
    f = wald(k=2)
    state = _quiet(f, None, 150, seed=7)
    blob = pickle.dumps(state)
    resumed = pickle.loads(blob)
    random.seed(99)
    future = [random.gauss(0, 0.1) for _ in range(60)] + [15.0]
    pa, pb = [], []
    sa, sb = state, resumed
    for y in future:
        _, sa = f(y, sa)
        _, sb = f(y, sb)
        pa.append(sa["pvalue"])
        pb.append(sb["pvalue"])
    assert pa == pb
    assert pa[-1] < 1e-6                 # and the outlier still fires


def test_determinism():
    """Same stream, two fresh machines: identical p-value sequences."""
    random.seed(21)
    ys = [random.gauss(0, 1) for _ in range(250)]
    outs = []
    for _ in range(2):
        f = wald(k=2)
        state = None
        ps = []
        for y in ys:
            _, state = f(y, state)
            ps.append(state["pvalue"])
        outs.append(ps)
    assert outs[0] == outs[1]


def test_scale_invariance_of_detection():
    """A unit change (x 1e6) must not change what counts as an anomaly."""
    for scale in (1.0, 1e6):
        random.seed(5)
        f = wald(k=2)
        state = None
        for _ in range(250):
            _, state = f(random.gauss(0, 0.1) * scale, state)
        _, state = f(30.0 * scale, state)
        assert state["pvalue"] < 1e-6, f"failed at scale {scale}"


def test_long_constant_stream():
    """A dead channel (constant forever) must not crash or emit alarms."""
    f = wald(k=2)
    state = None
    alarms = 0
    for _ in range(500):
        _, state = f(3.3, state)
        if state["pvalue"] is not None and state["pvalue"] < 1e-4:
            alarms += 1
    assert alarms <= 2                    # transient warmup at most
