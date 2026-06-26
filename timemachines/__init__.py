"""
timemachines is deprecated.

It has been superseded by `skaters` (https://github.com/microprediction/skaters),
a faster, lighter univariate forecasting package that runs in Pyodide.

This package now exists only as a thin compatibility shim: it re-exports
`laplace` from `skaters` so that existing `from timemachines import laplace`
imports keep working. Everything else has been removed.

Please migrate to `skaters`:

    pip install skaters
    from skaters import laplace
"""
import warnings

warnings.warn(
    "timemachines is deprecated and has been replaced by `skaters` "
    "(pip install skaters; https://github.com/microprediction/skaters). "
    "timemachines now only re-exports `laplace` from skaters. "
    "Please update your imports to `from skaters import laplace`.",
    DeprecationWarning,
    stacklevel=2,
)

from skaters import laplace

__all__ = ["laplace"]
