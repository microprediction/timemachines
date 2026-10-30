"""Release regression: run with the wheel venv's Python -I, not pytest.

The checkout's tests/conftest.py inserts src/ into sys.path. This standalone
check instead requires an installed module inside this interpreter's prefix.
It checks combination arithmetic, not empirical p-value calibration.
"""
import math
from importlib.metadata import version
from pathlib import Path
import sys

import timemachines
from timemachines import chi2_ppf, mahalanobis
from skaters.dist import Dist


def check_installed_wheel():
    module_path = Path(timemachines.__file__).resolve()
    assert Path(sys.prefix).resolve() in module_path.parents, (
        f"Expected an installed wheel under {sys.prefix}, got {module_path}"
    )
    print(f"timemachines {version('timemachines')} from {module_path}")
    z = math.sqrt(chi2_ppf(0.993, 1))

    def base(y, state):
        return [Dist.gaussian(0.0, 1.0)], {"z": [z]}

    def score(armed):
        f = mahalanobis(base, k=1, pot_level=.999, min_exc=30)
        _, state = f(0.0, None)
        state.update(
            mu=[0.0], S=[1.0], m2=1.0, v2=2.0,
            n_n=30 if armed else 0,
            n_exc=[1.0] * 30 if armed else [], nm=0.0, nv=1.0,
            pend1=Dist.gaussian(0.0, 1.0), run=0,
        )
        _, state = f(0.0, state)
        return state

    single, combined = score(False), score(True)
    p_single, p_combined = single["pvalue"], combined["pvalue"]
    # Fixed scripted-state counterexample from #167: v2.1.0 incorrectly
    # reports ~0.00703 in both cases; corrected code reports ~0.01406 armed.
    assert 0.005 < p_single < 0.01, p_single
    assert math.isclose(p_combined, 2.0 * p_single, rel_tol=1e-12), (
        p_single, p_combined
    )
    assert p_combined > 0.01, p_combined
    assert combined["p_nlp"] == 1.0
    assert combined["p_mahal"] == single["p_mahal"] == p_single
    assert single["p_nlp"] is None
    print(f"two-channel regression passed: {p_single=} {p_combined=}")


if __name__ == "__main__":
    check_installed_wheel()
