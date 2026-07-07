"""timemachines: temporal online machines.

Streaming decision layers — anomaly detection first — built on the
calibrated surprise streams of the ``skaters`` package. A *body* (a skaters
forecaster) turns any stream into forecasts plus standardized surprises; a
*head* (this package) turns surprises into decisions with controlled error
rates.

    from timemachines import wald          # the anomaly machine
    from timemachines import laplace       # the body, re-exported

v1 of this package was a zoo of forecasting wrappers, deprecated in favour
of skaters; v2 is its rebirth one layer up.
"""

from skaters import laplace, parade                     # bodies (re-exported)
from timemachines.api import wald                       # named machines
from timemachines.heads.mahalanobis import (            # composable heads
    mahalanobis, zbank, chi2_sf, chi2_ppf,
)

__all__ = [
    "wald",
    "laplace",
    "parade",
    "mahalanobis",
    "zbank",
    "chi2_sf",
    "chi2_ppf",
]
