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
