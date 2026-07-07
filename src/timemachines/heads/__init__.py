"""Heads: decision layers over calibrated surprise streams.

A head consumes a parade-wrapped skater (anything exposing ``state["z"]``)
and adds decision state on top, passing the forecasts through. Heads are
expected to multiply — Mahalanobis p-values today; CUSUM run-length
(``page``) and EVT/GPD tails (``pickands``) are the roadmap.
"""

from timemachines.heads.mahalanobis import mahalanobis, zbank

__all__ = ["mahalanobis", "zbank"]
