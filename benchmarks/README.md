# Benchmarks

`RESEARCH.md` — verified SOTA survey (datasets, metrics, baselines) for
streaming univariate anomaly detection, 2026-07.
`RESULTS.md` — the Rosenblatt front-end study results log.

## Harnesses

All strictly causal (no whole-series normalisation), one pass, argmax or
log-likelihood protocols as documented in each file's docstring.

| script | study |
|---|---|
| `ucr_run.py` | UCR Anomaly Archive: Wald scorers vs ablations/trivial |
| `frontend_run.py` | detector front-end lift: DSPOT/RRCF on raw vs parade z |
| `frontend_loglik.py` | forecaster front-end lift: ETS/ARIMA/GARCH/Prophet, exact change of variables |
| `fred_anomaly.py` | FRED injection benchmark (spike/burst/shift on real backgrounds) |
| `regression_frontend.py` | online regression front-end lift: one RLS learner, coordinates varied (raw/affine/robust/Laplace-z), contaminated simulation |
| `river_frontend.py` | same question, river's learners: Laplace front-end vs river's StandardScaler/TargetStandardScaler pipeline (needs `pip install river`) |
| `river_data_frontend.py` | the ticket gate: same comparison on river's own regression datasets, as-is and feature-spiked, with a body-alone attribution column |
| `ablation_frontend.py` | mu-vs-z ablation on the simulation: which of the two scalars carries the value |
| `ablation_river.py` | same ablation on river's datasets, dogfooding the published ice-skaters package (needs `pip install ice-skaters`) |
| `ablation_lag.py` | the fairness audit: lag-equipped baselines (raw y lags 1..8) vs the wrapper, alone and combined |
| `ablation_ewma.py` | the cheap-rollup control: EWMA (mu, z) pairs vs the Laplace ones |
| `sufficiency_study.py` | 126 non-price FRED cross-series pairs, river default learners (the pass that measured the learners; see RESULTS) |
| `sufficiency_rls.py` / `sufficiency_blr.py` | the corrected sufficiency passes: RLS and river's BayesianLinearRegression; verdict and mechanism in RESULTS |
| `additive_study.py` | the adopter's framing: forecasts added on top of raw lags (zero additive alpha; see RESULTS) |
| `granger_study.py` | the calibrated Granger test: size and power vs classical F, HAC, and the focused F1 |
| `granger2_study.py` | round two, the harsher grid: variance-causality trap, vol regimes, t(3)+GARCH, wrong lags, and a Cheung-Ng-style opponent |

Third-party baselines need the extra: `pip install "timemachines[benchmarks]"`.

## Data

Not shipped. Point the harnesses at your copies via environment variables:

- `TIMEMACHINES_UCR_DATA` — directory of UCR `*.txt` series
  (download: see RESEARCH.md section 1; extract `UCR_Anomaly_FullData`).
- `TIMEMACHINES_FRED_DATA` — FRED csv cache + `universe_daily.json`
  (`fred.py` builds it given a `FRED_API_KEY`; the universe rule is
  documented in the skaters repo's `benchmarks/fred_universe.py`).

Defaults fall back to `benchmarks/data/` in this repo.

## Fresh machine: reproduce and scale

Everything below is self-contained given the two data acquisitions.

```bash
git clone git@github.com:microprediction/timemachines.git && cd timemachines
pip install -e ".[benchmarks]"          # pulls skaters>=0.12.1 from PyPI
python -m pytest tests/ -q              # 23 tests, ~30s

# UCR Anomaly Archive (~184MB zip, nested; no password despite convention)
curl -L -o /tmp/ucr.zip "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip"
unzip -q /tmp/ucr.zip -d /tmp/ucr && unzip -q /tmp/ucr/AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021.zip -d /tmp/ucr/inner
export TIMEMACHINES_UCR_DATA=/tmp/ucr/inner/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData

# FRED cache: either rsync benchmarks/data/ from an existing machine
# (skaters repo: benchmarks/data/, ~2,600 csvs + universe_daily.json), or
# rebuild: set FRED_API_KEY, run skaters' benchmarks/fred_universe.py to
# resolve the universe, then any harness fetches+caches on demand.
export TIMEMACHINES_FRED_DATA=/path/to/fred/data
```

### Run matrix (observed single-machine costs; all embarrassingly parallel per series)

| harness | scope | cost |
|---|---|---|
| `calibration_panel.py --limit 0 --rrcf` | 144 prefixes | ~7 CPU-h (RRCF dominates) |
| `ucr_run.py` (full 250, per config) | argmax accuracy | ~14 CPU-h |
| `frontend_run.py --limit 60` | detector lift | ~4 CPU-h (RRCF) |
| `frontend_loglik.py --limit 30` | forecaster lift | ~1 CPU-h (ARIMA/Prophet refits) |
| `fred_anomaly.py --limit 100` | FRED injection | ~4 CPU-h |
| `rosenblatt_sandwich.py` | 12 price series | ~10 CPU-min |
| `regression_frontend.py --seeds 30` | 240 simulated runs | ~7 CPU-min |
| `river_frontend.py --seeds 30` | 240 simulated runs | ~20 CPU-min |
| `river_data_frontend.py` | 4 river datasets, 44 runs | ~15 CPU-min |
| `ablation_frontend.py --seeds 30` | 240 simulated runs | ~5 CPU-min |
| `ablation_river.py` | 4 river datasets, 44 runs | ~45 CPU-min (12 models/row) |
| `ablation_lag.py` | 4 river datasets, 44 runs | ~25 CPU-min |
| `sufficiency_study.py --pairs 150` | 126 FRED pairs, 252 runs | ~2 CPU-h |
| `sufficiency_rls.py` / `sufficiency_blr.py` | 126 FRED pairs, 252 runs each | ~1 CPU-h each |
| `ablation_ewma.py` | 4 river datasets, 44 runs | ~5 CPU-min |

### Bigger-machine targets, in value order

1. `frontend_run.py --limit 250` — the detector-lift headline at full-archive scale.
2. `calibration_panel.py` with a 1e-5 column (edit `ALPHAS`) and `CAP=50000`
   — deeper tail, more clean points per series.
3. `ucr_run.py --base zbank` full 250 (feature bank, ~6x Laplace cost).
4. `frontend_loglik.py --limit 200` — forecaster lift on the whole universe.
5. FRED-injection v2 (rank-percentile scoring — see RESULTS.md section 5)
   before spending compute on the argmax version.

Results land as `*.jsonl` beside the harnesses (gitignored); the aggregate
tables print at completion and belong in RESULTS.md.

### Checkpointing

Every long harness appends one json line per completed series and flushes
immediately, so a killed run loses at most the series in flight. Re-running
the same command resumes: series already present in the output file are
skipped (`resuming: N done, M to go`). Output filenames are stable per
configuration; pass `--out` to redirect. To restart a study from scratch,
delete its jsonl first.
