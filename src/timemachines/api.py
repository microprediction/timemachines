"""User-facing API.

One named machine so far, ``wald`` — sequential anomaly detection with
calibrated per-observation p-values. Bodies (forecasters) come from
``skaters`` and are re-exported for convenience; heads (decision layers)
live here and multiply here.

    from timemachines import wald

    f = wald(k=3)
    state = None
    for y in stream:
        dists, state = f(y, state)          # forecasts pass through
        if state["pvalue"] is not None and state["pvalue"] < 1e-4:
            alarm(y, state["pvalue"], state["run"])
"""

from __future__ import annotations
from skaters import laplace
from timemachines.heads.mahalanobis import mahalanobis


def wald(k: int = 3, scale_alpha: float = 0.01, det_alpha: float = 0.005,
         engine=None, **detector_kwargs):
    """Sequential anomaly detection: a skaters body plus a calibrated alarm.

    Runs :func:`skaters.laplace` underneath — the forecasts pass through
    unchanged — and scores each arriving observation's multi-horizon
    surprise vector (the parade z) with a robust Mahalanobis statistic
    against a streaming empirical null (:mod:`timemachines.heads.mahalanobis`):

        state["pvalue"]   calibrated per-observation p-value; alarming on
                          ``p < alpha`` yields a false-alarm rate of ~alpha
                          by construction — no threshold tuning;
        state["d2"]       the underlying Wald statistic z' Sigma^{-1} z;
        state["run"]      consecutive guarded ticks (an isolated spike reads
                          as a point outlier, a growing run as a break).

    Defaults differ from ``laplace``'s in one respect: memory. Detection
    wants a long-held notion of normal — a scale that re-adapts within an
    anomaly absorbs it — so the residual-scale EWMA and the detector's own
    estimators run slow (``scale_alpha=0.01``, ``det_alpha=0.005``; the
    UCR sigma-sweep evidence, see the benchmarks). Pass
    ``engine=laplace(k)`` — or any parade-wrapped skater — to detect on top
    of exactly the forecaster you already run.

    Named for Abraham Wald: the statistic is a Wald test, computed
    sequentially as Wald invented, with the detection-delay/false-alarm
    tradeoff he formalised; CUSUM descends from his SPRT, and sequential
    analysis was born doing streaming industrial inspection.
    """
    base = engine if engine is not None else laplace(k, scale_alpha=scale_alpha)
    f = mahalanobis(base, k=k, alpha=det_alpha, **detector_kwargs)
    f.__name__ = f"wald(k={k})"
    return f
