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

## 3. Own head on UCR (argmax, their home turf)

`ucr_run.py`. Scorers: mah (Mahalanobis p-value, single tick), mahS (scan,
windowed mean of -log10 p, w in {8,64}), z1 (per-horizon 1-step |z|,
ablation), zU (union min-p), mz (trivial EWMA z-score control).

| config | subset | best scorer | accuracy |
|---|---|---|---|
| defaults (sa 0.03 / da 0.02) | n=40 shortest | z1 | 0.650 |
| slow memory (sa 0.01 / da 0.005) | n=40 shortest | **mahS** | **0.675** |
| slow memory | n=60 shortest | mahS | 0.583 |
| slow memory | full 250 | mahS | RUNNING (~0.41 at 158/250) |
| search() engine (adaptive periods) | n=100 | z1 | 0.390 — worse; dropped |
| zbank sigma-grid | n=60 | | RUNNING |
| trivial control (mz) | n=60 | | 0.28-0.30 (matches its published band) |

Context bands (full 250, published): trivial 0.30-0.40; good single classical
methods 0.50-0.60; DAMP/matrix-profile 0.65-0.75; 2021 contest ensembles
0.70-0.80. Long ECG/periodic blocks are where we bleed; slow memory makes the
multivariate scan clearly beat the per-horizon rule at scale (65 vs 52 at
158/250). UCR is the credibility row, not the differentiator: argmax scoring
is blind to calibration by construction.

Findings that came out of this study regardless of scores:
- horizon-misalignment bug in `search()` candidate scoring (third instance
  of the pattern; fixed, parity regenerated);
- effective-rank collapse of the z scatter -> empirical null (Satterthwaite)
  + factor scatter with exact Sherman-Morrison/Woodbury inverse;
- masking through the null's second moment -> winsorised null updates;
- the sigma-memory axis (user hypothesis) lifts mahS by +6/40 on the subset.

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
