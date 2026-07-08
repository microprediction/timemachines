# Changelog

## 2.1.0 (2026-07-08)

The calibration release: the advertised false-alarm rate is now the real
one, measured. On 144 anomaly-free real-world prefixes (strictly
prequential), wald's empirical/nominal FPR ratio is 1.69 / 1.86 / 3.20 at
alpha = 1e-2 / 1e-3 / 1e-4 (median series at 1e-4: 1.00) — the best-
calibrated streaming detector of those measured at operating depth
(DSPOT: 0.98 / 1.95 / 9.34; RRCF threshold-transfer: 1.10 / 1.64 / 5.42).

- Streaming GPD tail (peaks-over-threshold) on the empirical null:
  threshold-relative excesses, probability-weighted-moments fit. The bulk
  Satterthwaite null alone ran 39x nominal at 1e-4.
- Deep-evidence channel: unbounded -logpdf beside the clamped Mahalanobis
  (the parade's |z| <= 7.03 clamp saturated d2, making a 250-sigma and a
  10-sigma event indistinguishable); its own POT tail, Bonferroni combine,
  silent in the bulk. Evidence now resolves at any depth.
- Both changes mirrored in the JS twin; calibration panel harness added
  (benchmarks/calibration_panel.py) plus GARCH-t front-end and Rosenblatt
  sandwich studies (benchmarks/RESULTS.md).

## 2.0.1 (2026-07-07)

Hardening release.

- Non-finite inputs (NaN/inf — real streams contain them) no longer crash
  or poison the machine: the head gates them, holds the last forecasts,
  counts them in `state["skipped"]`, and keeps going.
- Checkpoint/restore: machine state now pickles and resumes bit-identically
  (requires skaters>=0.12.1, whose state became pure data for exactly this).
- New hardening suite: hostile-stream fuzz (p-values always in [0,1]),
  determinism, scale invariance of detection, dead-channel streams.

## 2.0.0 (2026-07-07)

The rebirth. v1 (a zoo of forecasting wrappers) was deprecated in favour of
[skaters](https://github.com/microprediction/skaters); v2 is the decision
layer built on top of it: temporal online machines.

- `wald(k)` — streaming anomaly detection with calibrated per-observation
  p-values; alarm on `p < alpha` and the false-alarm rate is ~alpha by
  construction. Forecasts pass through.
- `heads.mahalanobis` — the detection head on any parade-wrapped skater
  (robust factor scatter with exact Woodbury inverse, winsorized empirical
  null, changepoint escape) — plus `zbank`, the multi-scale feature bank.
- `from timemachines import laplace` still works (bodies re-exported).
- Live demo: https://timemachines.microprediction.org
- Benchmarks: see `benchmarks/RESULTS.md` (detector front-end lift DSPOT
  5.2x / RRCF 1.8x on UCR-60; forecaster front-end lift ~+2 nats/point,
  149/150 series, ETS/AutoARIMA/GARCH/Prophet).

## 1.0.0 (2024)

Deprecation shim over skaters (re-exported `laplace` with a warning).
