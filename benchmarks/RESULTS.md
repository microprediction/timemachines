# The Rosenblatt front-end: results log

Working results for the anomaly/front-end study (issues #88, PR branch
`proto/anomaly-mahalanobis`). Thesis under test:

> `laplace` defines a causal bijection on paths (the Rosenblatt transform
> z_t = Phi^-1(F_t(y_t))). In its coordinates, other people's detectors and
> forecasters get better — and our own detection head emits calibrated
> p-values no score-based method can.

All runs: strictly causal scores (no whole-series normalisation), defaults
unless stated, one pass. Protocols in `RESEARCH.md`; harnesses in this
directory. Updated 2026-07-07.

## 1. Detector front-end — FINAL (UCR-60, argmax protocol)

Same detector, same series; only the input changes (raw y vs parade z from
`laplace(3)`). `frontend_run.py`, n = 60 shortest UCR series.

| detector | raw | laplace-fronted | lift |
|---|---|---|---|
| DSPOT (EVT thresholding, KDD 2017) | 6/60 = 0.100 | **31/60 = 0.517** | **5.2x** |
| RRCF (random cut forest, ICML 2016) | 15/60 = 0.250 | **27/60 = 0.450** | **1.8x** |

Reading: DSPOT = weak normaliser + excellent GPD tail head; its stationarity
premise starves on raw waveforms and feasts on z. The front-end gives its
theorem the stream it assumes.

## 2. Forecaster front-end — FINAL (FRED-30, exact change of variables)

One-step log-likelihood in y-space; z-space scores mapped back through the
exact Jacobian log f_t(y_t) - log phi(z_t). Per-tick losses clamped to
[-20, 20] for every method symmetrically (bounded loss; one degenerate
predictive must not own a series mean). `frontend_loglik.py`, n = 30 FRED
series, rolling refit every 200 on trailing 1000.

| opponent | raw | fronted | mean lift | median | wins |
|---|---|---|---|---|---|
| ETS (statsmodels Holt, damped) | 1.642 | 3.680 | **+2.04** | +0.79 | **30/30** |
| AutoARIMA (statsforecast) | 1.646 | 3.719 | **+2.07** | +0.76 | **30/30** |
| GARCH(1,1) (arch) | 1.732 | 3.703 | **+1.97** | +0.77 | **30/30** |
| Prophet (real calendar dates) | 1.636 | 3.702 | **+2.07** | **+0.85** | **30/30** |
| EWMA-Gauss (control head) | 2.243 | 3.795 | +1.55 | +0.51 | 29/30 |
| *laplace alone (reference)* | | *3.674* | | | |

Bijection prediction verified: fronted opponents converge to laplace + eps
(AutoARIMA adds +0.045 nats over laplace alone — laplace had already
extracted ~98% of the structure ARIMA can model). The control head on z
reproduces laplace's own score, confirming the Jacobian accounting.

Prophet is the sharpest row: its calendar machinery (weekday/yearly from
real dates — structure laplace's integer-lag seasonal block cannot
represent) earns the largest median lift when fronted, yet adds only
+0.03 nats over laplace alone: even the calendar model finds almost
nothing left in the residuals. FINAL: five opponents, 149/150 wins.

## 3. Own head on UCR — honest and secondary

The headline claims of this study are the front-end lifts above (sections 1-2);
the own head's UCR standing is reported for honesty, not marketing. Full 250,
default config: best scorer (|z1|) 0.276 vs trivial 0.216. The diagnostic
split: **0.500 on the 84 series <= 20k points; 0.169 on the 166 longer ones**
— two-thirds of the archive is 20k-900k-point periodic physiology whose
waveform periods (~50-400 samples) the laplace body cannot represent (its
seasonal grid is calendar {7,12,24}), so every cycle reads as fresh surprise.
That is a *body* weakness observed through the head — filed upstream as
skaters#91 — and matrix-profile methods own that terrain by construction.
Slow-memory config (sa 0.01/da 0.005) improves the multivariate scan (~0.32
full-archive, and 0.675 on the 40 shortest, beating per-horizon |z1| at
scale); the zbank sigma-grid sits between the configs without tuning.

Findings the study produced regardless of scores: a horizon-misalignment bug
in skaters' search() (third instance of the pattern; fixed, parity green);
the effective-rank collapse of the z scatter -> empirical null + factor
scatter with exact Woodbury; masking through the null's second moment ->
winsorised updates; the sigma-memory axis; and (from the hardening suite)
skaters' state-purity fix enabling checkpoint/restore (0.12.1).

## 4. FRED injection (argmax) — protocol needs v2

`fred_anomaly.py`, n=100 real FRED change series, planted spike/burst/shift.
Interim (~73/100): everyone weak (~0.14 best), orderings inside binomial
noise. Diagnosis: real backgrounds contain genuine unlabeled anomalies
(2008, COVID); argmax scores "which real crisis did you prefer", a
confounded measure. v2: score the planted window's rank percentile in the
full score ordering (robust to dominant natural events), and/or mask known
crisis windows. Keep argmax row for reference.

## 5. Still to come

- slow-alpha full-250 (running); zbank-60 and default-250 (running,
  detached).
- FRED v2 with rank-percentile scoring.
- The calibration panel — empirical false-alarm rate vs nominal alpha,
  prequential protocol, detection delay; the verified literature gap
  (RESEARCH.md section 2) and the method's actual differentiator.
- GPD/EVT tail for the detector's extreme p-values (steal DSPOT's tail
  theorem for our head); unclamped -logpdf surprise channel (the z-clamp
  saturates at |z|=7.03, erasing 20-sigma vs 100-sigma distinctions).
- TSB-AD-U leaderboard run (VUS-PR; top is 0.42, simple methods lead).

## Reproduce

```
python benchmarks/anomaly/ucr_run.py --limit 60 --workers 8 [--scale-alpha 0.01 --det-alpha 0.005 | --base zbank]
python benchmarks/anomaly/frontend_run.py --limit 60 --workers 4
python benchmarks/anomaly/frontend_loglik.py --limit 30 --workers 3
python benchmarks/anomaly/fred_anomaly.py --limit 100 --workers 4
```
UCR data: see RESEARCH.md (download + extract into `data/UCR_Anomaly_FullData`).
FRED data: cached under `benchmarks/data/` (see `benchmarks/fred.py`).
