"""Extreme-input hardening: finite values must never kill the machine.

The non-finite gate (NaN/inf skipped, forecasts held) predates these
tests; what they pin down is the *finite* attack surface: double
arithmetic inside any body dies long before the float range ends, so the
head winsorizes at |y| = 1e60 and a million predictive sigmas — both far
beyond the parade z saturation at |z| = 7.03, hence decision-invariant.
"""

import math

import pytest

from timemachines import laplace, mahalanobis, wald

BASE = [math.sin(0.1 * t) for t in range(200)]


def _feed(f, ys):
    st = None
    for y in ys:
        dists, st = f(y, st)
    return dists, st


@pytest.mark.parametrize("ys", [
    BASE + [1e300] + BASE,
    [1e300] * 100,
    [10.0 ** (t % 280) for t in range(300)],          # exponential ladder
    [10.0 ** -(t % 280) for t in range(300)],         # collapse to tiny
    BASE + [-1e300, 1e300] * 3 + BASE,
    [1e59] * 50 + [1e60] * 50,
    # the two cases that broke earlier clamp designs: a legitimate value
    # arriving after a degenerate-variance stretch is billions of sigmas
    # out and must NOT be clamped; ultra-predictable streams (collapsed
    # predictive variance) must keep their legitimate steps
    [0.0] * 100 + [35.0] + BASE,
    [736000.0 + t for t in range(300)],
], ids=["spike-1e300", "all-1e300", "ladder-up", "ladder-down",
        "alternating", "near-clamp", "recovery-from-zeros",
        "date-like-linear"])
def test_huge_finite_values_never_crash_wald(ys):
    dists, st = _feed(wald(3), ys)
    assert st["pvalue"] is None or 0.0 <= st["pvalue"] <= 1.0
    assert all(math.isfinite(d.mean) for d in dists)


def test_spike_still_reads_as_maximal_surprise():
    """The clamp must not blunt detection: a huge spike still saturates z."""
    f, st = wald(3), None
    for y in BASE:
        _, st = f(y, st)
    _, st = f(1e300, st)
    assert abs(st["base"]["z"][0]) > 6.9


def test_custom_base_with_non_finite_z_does_not_poison_null():
    from skaters.dist import Dist

    def bad_base(y, st):
        st = st or {"n": 0}
        st["n"] += 1
        st["z"] = [float("inf"), 2.0] if st["n"] == 50 else [0.1, 0.2]
        return [Dist.gaussian(0.0, 1.0)] * 2, st

    f, st = mahalanobis(bad_base, k=2), None
    for _ in range(120):
        _, st = f(0.0, st)
    assert st["pvalue"] is not None and 0.0 <= st["pvalue"] <= 1.0


@pytest.mark.parametrize("kwargs", [
    {"min_exc": 1}, {"guard_p": 1.0}, {"guard_p": 0.0},
    {"pot_level": 1.0}, {"pot_level": 0.0},
])
def test_bad_head_parameters_rejected_at_construction(kwargs):
    with pytest.raises(AssertionError):
        mahalanobis(laplace(2), k=2, **kwargs)


def test_cross_k_state_reuse_is_a_clear_error():
    f2, st = wald(2), None
    for y in BASE[:50]:
        _, st = f2(y, st)
    with pytest.raises(ValueError, match="k=2"):
        wald(3)(0.5, st)
