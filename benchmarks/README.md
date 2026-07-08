# Benchmarks

`RESEARCH.md` — verified SOTA survey (datasets, metrics, baselines) for
streaming univariate anomaly detection, 2026-07.
`RESULTS.md` — the Rosenblatt front-end study results log.

## Harnesses

All strictly causal (no whole-series normalisation), one pass, argmax or
log-likelihood protocols as documented in each file's docstring.

| script | study |
|---|---|
| `ucr_run.py` | UCR Anomaly Archive: wald scorers vs ablations/trivial |
| `frontend_run.py` | detector front-end lift: DSPOT/RRCF on raw vs parade z |
| `frontend_loglik.py` | forecaster front-end lift: ETS/ARIMA/GARCH/Prophet, exact change of variables |
| `fred_anomaly.py` | FRED injection benchmark (spike/burst/shift on real backgrounds) |

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

### Bigger-machine targets, in value order

1. `frontend_run.py --limit 250` — the detector-lift headline at full-archive scale.
2. `calibration_panel.py` with a 1e-5 column (edit `ALPHAS`) and `CAP=50000`
   — deeper tail, more clean points per series.
3. `ucr_run.py --base zbank` full 250 (feature bank, ~6x laplace cost).
4. `frontend_loglik.py --limit 200` — forecaster lift on the whole universe.
5. FRED-injection v2 (rank-percentile scoring — see RESULTS.md section 5)
   before spending compute on the argmax version.

Results land as `*.jsonl` beside the harnesses (gitignored); the aggregate
tables print at completion and belong in RESULTS.md.
