# timemachines

**Temporal online machines**: streaming decision layers — anomaly detection
first — built on the calibrated surprise streams of
[skaters](https://github.com/microprediction/skaters).

![skating](https://i.imgur.com/elu5muO.png)

```python
from timemachines import wald

f = wald(k=3)
state = None
for y in stream:
    dists, state = f(y, state)              # skaters forecasts pass through
    if state["pvalue"] is not None and state["pvalue"] < 1e-4:
        alarm(y, state["pvalue"], state["run"])
```

That `1e-4` is a **false-alarm rate, not a tuned threshold**: `wald` emits a
calibrated per-observation p-value, so alarming on `p < alpha` yields a
false-alarm rate of approximately alpha. Measured, not asserted — on 144
anomaly-free real-world prefixes (strictly prequential), empirical/nominal:

| method | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|
| `wald` | 1.69 | **1.86** | **3.20** (median series **1.00**) |
| DSPOT | **0.98** | 1.95 | 9.34 |
| RRCF (pre-committed threshold) | 1.10 | 1.64 | 5.42 |

At operating depth, the best-calibrated streaming detector measured; no
other family even carries nominal semantics to be wrong about.

## v2: what happened here

v1 of timemachines was a zoo of forecasting wrappers. It was deprecated in
favour of [skaters](https://github.com/microprediction/skaters), which does
the forecasting job properly: one pass, constant memory, stdlib only,
distributional outputs, and — crucially for us — the **prediction parade**:
alongside each forecast, the standardized surprise `z` of every arriving
point under the predictions previously made *for it*.

v2 is the rebirth one layer up. A skaters forecaster (a **body**) turns any
stream into forecasts plus calibrated surprises; this package's **heads**
turn surprises into decisions with controlled error rates. Bodies are few
and stable; heads multiply — that is why they get their own package.
(`from timemachines import laplace` worked in the v1 shim and still works.)

## Why a forecaster-first detector

The hard part of anomaly detection is not the detector — it is the *null*.
A calibrated online forecaster is the best null model there is: under it,
the surprise stream is approximately iid N(0,1), which is exactly the
homogeneous input every classical detection method assumes and raw data
never provides. Two measured consequences (protocols and full tables in
`benchmarks/`):

**Other people's detectors get better in these coordinates.** Same detector,
same series (UCR anomaly archive, 60 series), only the input changed:

| detector | raw series | laplace-transformed | lift |
|---|---|---|---|
| DSPOT (EVT thresholding, KDD 2017) | 0.100 | **0.517** | **5.2x** |
| RRCF (random cut forest, ICML 2016) | 0.250 | **0.450** | **1.8x** |

**Other people's forecasters get better too.** One-step log-likelihood on 30
FRED series, exact change of variables through the bijection
`z_t = Phi^-1(F_t(y_t))`:

| opponent | lift (nats/point) | wins |
|---|---|---|
| ETS | +2.04 | 30/30 |
| AutoARIMA | +2.07 | 30/30 |
| GARCH(1,1) | +1.97 | 30/30 |
| Prophet | +2.07 | 30/30 |

Fronted opponents converge to the skaters forecaster plus a few hundredths
of a nat: the body had already extracted nearly everything they know how to
model.

## The machines

| name | job | state it adds |
|---|---|---|
| `wald(k)` | sequential anomaly detection | `pvalue` (calibrated), `d2` (the Wald statistic z' Sigma^-1 z), `run` (spike vs break) |

Named machines are curated recipes — a body, a head, and settings that
earned their defaults on benchmarks. The composable parts underneath:

- `mahalanobis(base, k, ...)` — the detection head on any parade-wrapped
  skater: robust streaming location/scatter of the surprise vector (factor
  model + exact Woodbury inverse), an empirical null (two-moment
  Satterthwaite, winsorized against masking), Huberised updates with a
  changepoint escape.
- `zbank(k, sigmas, strides)` — a feature bank of bodies across memory and
  clock scales, concatenated surprises, for multi-scale detection.
- `laplace`, `parade` — re-exported from skaters for convenience.

Roadmap heads, each named for its theorem: `page` (CUSUM run-length
changepoints), `pickands` (EVT/GPD tails for extreme p-values).

## Design rules

- **Bodies live in skaters, heads live here.** The API boundary is the
  parade state contract (`state["z"]`, `state["pit"]`).
- Dependency arrows point one way: this package depends on skaters, never
  the reverse. The core is pure stdlib on top of it; third-party detectors
  and benchmark tooling sit behind the `benchmarks` extra.
- Everything is one-pass, constant-memory, strictly causal: scores at time
  t use only observations up to t. No whole-series normalisation, ever.

## Install

```
pip install timemachines        # v2: requires skaters
```

v1 (`<2.0`) remains on PyPI for pinned users; it is unmaintained.

MIT licensed.
