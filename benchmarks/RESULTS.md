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

| detector | raw | Laplace-fronted | lift |
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
| *Laplace alone (reference)* | | *3.674* | | | |

Bijection prediction verified: fronted opponents converge to Laplace + eps
(AutoARIMA adds +0.045 nats over Laplace alone — Laplace had already
extracted ~98% of the structure ARIMA can model). The control head on z
reproduces Laplace's own score, confirming the Jacobian accounting.

The hardest case — GARCH-t on its home turf (12 price/return series:
equities, FX majors, oil, gas, VIX; Student-t density scored exactly):

| opponent | raw | fronted | mean lift | wins |
|---|---|---|---|---|
| GARCH-t | 2.630 | **2.708** | **+0.078** | **12/12** |
| GARCH(1,1) Gaussian | 2.496 | 2.693 | +0.20 | 12/12 |
| *Laplace alone* | | *2.700* | | |

Even the model that wins on price series gains in the transformed
coordinates, and the composite (Laplace front-end + GARCH-t on z) is the
best cell measured — the composition `laplace(leaf=garch_leaf)` already
ships in skaters. Caveat: raw GARCH-t trailing Laplace alone here differs
from the skaters paper's price-table verdict; this protocol (rolling
200-refit, 1000-window, 3000-point cap) is not that one.

Prophet is the sharpest row: its calendar machinery (weekday/yearly from
real dates — structure Laplace's integer-lag seasonal block cannot
represent) earns the largest median lift when fronted, yet adds only
+0.03 nats over Laplace alone: even the calendar model finds almost
nothing left in the residuals. FINAL: five opponents, 149/150 wins.

### The sandwich as an object (`rosenblatt_sandwich.py`)

Change of variables scores Laplace -> inner -> Laplace^-1 exactly for
log-lik (the tables above). Built as an actual predictive object (quantile-
grid pushforward, k=1), it currently pays ~0.06 nats of representation tax
— more than the inner's true add over Laplace on prices (~+0.01), so the
object loses 0/12 there while `garch_leaf` directly on y wins 9/12 (the
no-free-lunch caveat at object level). The object route matters for
delivering CRPS/intervals of fronted models; route and open items in
skaters#92 (self-hosting conjugation).

## 3. Own head on UCR — secondary

The headline claims of this study are the front-end lifts above (sections 1-2).
The own head's UCR standing, full 250,
default config is: best scorer (|z1|) 0.276 vs trivial 0.216. The diagnostic
split: **0.500 on the 84 series <= 20k points; 0.169 on the 166 longer ones**
— two-thirds of the archive is 20k-900k-point periodic physiology whose
waveform periods (~50-400 samples) the Laplace body cannot represent (its
seasonal grid is calendar {7,12,24}), so every cycle reads as fresh surprise.
That is a *body* weakness observed through the head — filed upstream as
skaters#91 — and matrix-profile methods own that terrain by construction.
Slow-memory config (sa 0.01/da 0.005), full archive (n=247 of 250; the run
was stopped three 300k-point series short): mahS 0.308 > |z1| 0.291 > mah
0.283 > trivial 0.219. Slow memory is worth ~+3 points over default and
flips the ordering — the multivariate scan beats the per-horizon rule at
scale, which default memory never managed (and 0.675 on the 40 shortest).
The zbank sigma-grid sits between the configs without tuning.

Findings the study produced regardless of scores: a horizon-misalignment bug
in skaters' search() (third instance of the pattern; fixed, parity green);
the effective-rank collapse of the z scatter -> empirical null + factor
scatter with exact Woodbury; masking through the null's second moment ->
winsorised updates; the sigma-memory axis; and (from the hardening suite)
skaters' state-purity fix enabling checkpoint/restore (0.12.1).

## 4. The calibration panel — the differentiator, measured

`calibration_panel.py`: 144 UCR anomaly-free prefixes (clean by
construction; every flag a false alarm), strictly prequential — thresholds
and nulls never see the data they are scored on. Wald flags at p < alpha;
DSPOT at its own EVT risk (alpha/2 per tail); RRCF has no nominal
semantics, so it is deployed the only way an uncalibrated score can be:
threshold pre-committed on the first half, measured on the second.

| empirical FPR / nominal (1.0 = calibrated) | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|
| Wald (Mahalanobis + deep-evidence channels) | 1.69 | **1.86** | **3.20 (median series 1.00)** |
| DSPOT (KDD 2017) | **0.98** | 1.95 | 9.34 |
| RRCF threshold transfer | 1.10 | 1.64 | 5.42 |

At the depth where alarms operate, Wald is the best-calibrated of the
three and its median series is exact; the aggregate 3.2 is a tail of
waveform-burst prefixes. Getting here required two head upgrades this
panel itself forced: the bulk Satterthwaite null alone ran 39x at 1e-4
(a two-moment fit extrapolated into the deep tail); a streaming GPD over
threshold-relative excesses (PWM-fitted — MOM cannot exceed shape 1/2 and
the excess tails run ~0.7) brought it to ~5x; and the z-clamp saturation
(a 250-sigma and a 10-sigma event both read |z|=7.03) was broken by an
unbounded -logpdf channel with its own POT tail, restoring evidence at any
depth (2e-11 where the clamp capped at 3e-5) at a small bulk cost.

## 5. FRED injection (argmax) — protocol needs v2

`fred_anomaly.py`, n=100 real FRED change series, planted spike/burst/shift.
Interim (~73/100): everyone weak (~0.14 best), orderings inside binomial
noise. Diagnosis: real backgrounds contain genuine unlabeled anomalies
(2008, COVID); argmax scores "which real crisis did you prefer", a
confounded measure. v2: score the planted window's rank percentile in the
full score ordering (robust to dominant natural events), and/or mask known
crisis windows. Keep argmax row for reference.

## 6. Regression front-end — contaminated simulation (240 runs)

`regression_frontend.py`: the front-end thesis pointed at online regression.
One fixed dumb learner (RLS, forgetting 0.999) predicts y one step ahead on
a linear Gaussian pair (x AR(0.8) driving y, beta=1); only the coordinate
system varies. The headline condition is **zin**: each stream is replaced
by two scalars from its own `laplace(1)` body — the one-step predictive
mean and the parade surprise z — and the target stays raw. 8 scenarios
x 30 seeds; excess MSE vs the clean conditional mean (oracle = 0), median
over seeds. Same 33-node pinball CRPS estimator for every condition.
Output-fixing conditions ran too; they are a footnote (end of section).

| scenario | Laplace | raw | zscore | robust | huber | zin |
|---|---|---|---|---|---|---|
| clean | 1.14 | **0.001** | 0.036 | 0.108 | 0.001 | 0.057 |
| spikes_x | 1.15 | 1.08 | 0.98 | 0.91 | 1.38 | **0.72** |
| spikes_y | 15.6 | 3.78 | 5.66 | 4.42 | 3.75 | **1.84** |
| spikes_both | 16.0 | 9.05 | 9.78 | 7.66 | 9.65 | **6.33** |
| heavy (t2) | 9.54 | 0.008 | 2.05 | 2.17 | **0.006** | 0.95 |
| drift | 3.59 | **0.003** | 0.059 | 0.232 | 0.003 | 0.059 |
| distort | 1.17 | 1.05 | 0.71 | 0.53 | 1.48 | **0.51** |
| shift | 1.22 | **0.001** | 0.105 | 0.159 | 0.001 | 0.089 |

(robust = median/MAD + winsorize at 4; huber = raw coordinates, clipped
RLS update; zscore = EWMA-standardized, the RevIN-style affine case.)

Reading, in order of importance:

- **Fixing the inputs is nearly free insurance.** zin beats raw 30/30 on
  every contaminated scenario — x-spikes, y-spikes, both, distortion — and
  beats the robust affine baseline on all of them too (it halves raw's MSE
  under y-spikes). Its clean toll is 0.057 excess MSE (CRPS 0.32 vs 0.28).
- **Heavy tails are not contamination.** With genuine t(2) driving noise
  and a correctly specified linear model, the extreme points are the most
  informative ones; raw and huber sit at the oracle and every coordinate
  fix pays dearly for taming the signal. The practitioner's question
  "are my tails noise or signal?" decides the coordinates.
- **Monotone distortion is home turf** (the nonparanormal generative
  model): zin 0.51 vs raw 1.05, 30/30 — here the coordinates *are* the
  model.
- **The loss is the wrong place to fix inputs**: huber tracks raw
  everywhere except x-spikes, where clipping the residual cannot repair a
  corrupted feature and it loses to raw (1.38 vs 1.08).
- Caveat on the shift row: a level shift in x passes straight through a
  correctly specified linear map, so raw recovers instantly; that row
  tests specification, not adaptation — a target-intercept shift is the
  v2 scenario.

### The same question with river's learners (`river_frontend.py`)

Third-party learners, third-party baseline: river 0.25's LinearRegression
(SGD) and HoeffdingTreeRegressor under river's own recommended pipeline —
`StandardScaler` on features, `TargetStandardScaler` on the target — vs
the Laplace front-end composed WITH that same pipeline (features replaced
by body means + surprises). Same generator, scenarios, seeds and metric as
above. Median excess MSE, and lapin-vs-std wins out of 30 (sign test:
30/30 is p = 1.9e-9, 26/30 is p = 5e-5):

| scenario | lin std | lin lapin | wins | tree std | tree lapin | wins |
|---|---|---|---|---|---|---|
| clean | **0.010** | 0.080 | 0/30 | **0.011** | 0.086 | 0/30 |
| spikes_x | 1.26 | **0.94** | **30/30** | 1.30 | **0.96** | **30/30** |
| spikes_y | 5.04 | **3.12** | **30/30** | 4.48 | **3.45** | **27/30** |
| spikes_both | 11.1 | **9.2** | **27/30** | 7.9 | 8.1 | 16/30 |
| heavy (t2) | **0.21** | 1.43 | 7/30 | **0.21** | 1.29 | 4/30 |
| distort | 1.10 | **0.81** | **26/30** | 1.11 | **0.88** | 19/30 |

Reading: the section-6 pattern survives the transplant. Under measurement
contamination and monotone distortion the front-end lifts river's
canonical pipeline significantly on its two model-based learners; on
clean/drift/shift data it pays the same small toll, and the informative-
tails caveat (heavy) transfers unchanged. Bare unscaled SGD diverges
(excess MSE up to 1e25), confirming the std pipeline as the only fair
baseline. Boundaries disclosed: KNNRegressor is the counterexample —
neighbour averaging is already spike-robust and extra surprise dimensions
degrade its distance metric, so the front-end hurts it nearly everywhere.
Before any upstream ticket: repeat on river's own real datasets with
progressive validation.

### River's own data (`river_data_frontend.py`) — the ticket gate

Four river regression datasets, progressive validation, MAE, numeric
features only (dropped symmetrically), Bikes capped at 20k of 182k rows.
`body` = the target's Laplace predictive mean alone, no regression, no
features — the attribution control. Untouched data (asis), tree learner:

| dataset | std | lapin | body |
|---|---|---|---|
| TrumpApproval | 0.334 | 0.381 | **0.150** |
| ChickWeights | **23.8** | 24.7 | 25.5 |
| AirlinePassengers | 41.9 | **26.6** | 29.4 |
| Bikes (20k) | 5.07 | 5.29 | **4.94** |

With 2% feature spikes (10 seeds), lapin beats std 10/10 on TrumpApproval,
AirlinePassengers and Bikes for the tree learner (9/10, 10/10, 0/10 for
linear), and 1-3/10 on ChickWeights.

Reading: on history-dominated streams (three of the four datasets) the
univariate body ALONE already beats river's full feature pipeline — on
river's own flagship example by 2.2x — the section-2 finding transplanted
to regression: the features add almost nothing the calibrated forecaster
had not already extracted.
Where the features genuinely carry entity identity that an interleaved
single-stream body cannot represent (ChickWeights: 50 chick growth curves,
per-entity bodies are the obvious v2), the pipeline keeps its edge and the
front-end loses, and the table says so. The insurance claim transfers to real
backgrounds: under feature contamination the front-end wins 10/10 wherever
it was competitive as-is. Disclosed instability: river's default SGD
LinearRegression degrades on Bikes in every coordinate system (14.4 asis
vs the tree's 5.07) and diverges outright on the lapin features (MAE
2.5e10); the tree learner is the fair carrier of the comparison there.

The river ticket case, stated plainly: (1) a calibrated forecaster's
state — predictive mean plus standardized surprise — is a strong feature
set and baseline that river has no native transformer for; (2) it makes
model-based pipelines significantly more robust to feature contamination.
Boundaries to state in the ticket: distance-based learners (KNN), entity-
interleaved streams without per-entity bodies, and genuinely informative
heavy tails.

The integration ships as its own micro-package, deep-river style, so
river core is never asked to carry a dependency and timemachines stays
one layer up: [ice-skaters](https://github.com/microprediction/ice-skaters)
(skaters on a river) provides `LaplaceFeatures` (a river Transformer;
each numeric stream becomes the (mean, z) pair, non-numerics pass
through, NaN is forecast-imputed) and `LaplaceTarget` (a regressor
wrapper adding the target's own pair, in the style of
TargetStandardScaler). Both pickle and deep-copy; predict and learn
paths see identical features despite Pipeline's learn-then-transform
ordering. Runnable demo in that repo, `examples/trump_approval.py`
(TrumpApproval: baseline 0.328 clean / 0.597 under 2% corrupted
pollster readings; fronted 0.382 / 0.407). The ticket should say plainly
that we are unsure a skaters dependency is desirable — the package is
young — and note the alternative: skaters is stdlib-only pure Python
(no numpy, runs in Pyodide), so a frozen version could be vendored into
river in river-idiomatic form if the maintainers ever wanted it native;
the generic shape would be a ForecasterFeatures transformer wrapping any
of river's own time_series models, with Laplace as one plug.

### The ablation: which scalar carries it, and what it costs

`ablation_frontend.py` decomposes the pair on the section-6 simulation
(same generator, seeds, RLS learner; excess MSE, medians):

| scenario | raw | mu only | z only | the pair | pair, raw y lag |
|---|---|---|---|---|---|
| clean | **0.001** | 0.177 | 16.9 | 0.057 | 0.016 |
| spikes_y | 3.78 | 3.21 | 17.6 | **1.84** | 2.38 |
| heavy (t2) | 0.008 | 1.36 | 146 | 0.95 | 0.18 |
| distort | 1.05 | 1.06 | 18.0 | **0.51** | 0.67 |

z alone is useless: surprises carry no level, and a level cannot be
regressed from pure news. mu alone pays a large clean toll and misses
distortion. The pair is genuinely a unit — the learner recombines level
and news into a conditional mean neither scalar supports alone. The
target's own pair is insurance for the target specifically: swapping it
for the raw y lag is better on clean/heavy/drift/shift and worse under
target spikes and distortion.

`ablation_river.py` repeats on river's datasets, dogfooding the published
ice-skaters package, and adds the condition the earlier study missed:
`lt` = LaplaceTarget alone, raw features untouched. MAE, tree learner,
untouched data:

| dataset | std | lt | full recipe |
|---|---|---|---|
| TrumpApproval | 0.334 | **0.301** | 0.387 |
| ChickWeights | **23.8** | 24.1 | 24.5 |
| AirlinePassengers | 41.9 | **26.6** | 29.5 |
| Bikes (20k) | 5.07 | **5.01** | 5.51 |

Under 2% feature spikes lt beats std 10/10 on TrumpApproval,
AirlinePassengers and Bikes (tree; 6/10 on ChickWeights), despite its
features being the raw, spiked ones — the target pair anchors the
prediction and reduces reliance on corrupted features. **Revised
recommendation, in order:** add the target pair always (one wrapper,
helps clean and contaminated, ties the pipeline even on the
counterexample dataset); replace features with their (mu, z) pairs only
when you distrust the features; never use z alone.

Cost, measured (`ice-skaters`, 3 numeric streams): LaplaceFeatures ~390
microseconds per stream per sample vs ~0.4 for StandardScaler, a ~900x
premium, i.e. ~2,500 samples/s per stream single-threaded. Right for
polls, sensors and market bars; wrong inside a hot path at hundreds of
thousands of ticks per second.

### Footnote: the output sandwich (dropped from the recommendation)

All three harnesses also ran output-fixing conditions: zout (raw
features, the target's parade z as the learn target, mapped back through
the body's predictive CDF) and the full sandwich (z on both sides,
33-node quadrature in the RLS harness, plug-in for river's point
learners). Verdict: everywhere the sandwich won, input fixing was at its
heels (distort 0.50 vs zin's 0.51; spikes_x 0.71 vs 0.72), and on
river's real data it was body plus epsilon (TrumpApproval 0.158 vs the
body's 0.150; Bikes 4.90 vs 4.94). Its one distinctive behaviour is the
failure mode: under target spikes the default-memory body chases the
contamination and the map-back amplifies it — zout 226 vs raw's 3.8,
sandwich 38.5, clean toll 0.170. Input fixing has bounded influence by
construction; output fixing is only as sound as the body's own
robustness (a slow-memory body, `scale_alpha=0.01`, is the predicted but
untested fix). The construction stays in the harnesses and in the theory
— the log-loss isometry and the KL decomposition live on the output
bijection, and it is the only route from point learners to
distributional outputs (CRPS, intervals) — but it exits the practical
recommendation: fix the inputs, keep the target raw. Self-hosting route:
skaters#92. Theoretical resolution: the regret-transfer theorem
(ice-skaters, `papers/regret-transfer.md`) proves the sandwich as a
*density* can never lose more than O(d log T) nats to the body on any
data sequence; the pathology above is a property of extracting the
pushforward mean, not of the density.

## 7. Still to come

- slow-alpha full-250 (running); zbank-60 and default-250 (running,
  detached).
- FRED v2 with rank-percentile scoring.
- Detection delay at fixed alpha (the panel covers FPR; delay is the
  other axis of the Wald tradeoff).
- GPD/EVT tail for the detector's extreme p-values (steal DSPOT's tail
  theorem for our head); unclamped -logpdf surprise channel (the z-clamp
  saturates at |z|=7.03, erasing 20-sigma vs 100-sigma distinctions).
- TSB-AD-U leaderboard run (VUS-PR; top is 0.42, simple methods lead).
- Regression front-end v2: slow-memory body for the zout/spikes_y cell;
  target-intercept shift scenario; per-entity bodies (the ChickWeights
  fix); k=3 multi-horizon surprise features for the drift/shift rows;
  concept-drift classification (Elec2/Insects); then the FRED
  cross-series prong and the academic prong (RevIN/DAIN/Dish-TS on ETT
  with the same dumb learner). Done since the first pass: the mu-vs-z
  ablation, the LaplaceTarget-only condition, per-tick cost, and the
  regret-transfer theorem with its numerical check (both in
  ice-skaters).

## Reproduce

```
python benchmarks/anomaly/ucr_run.py --limit 60 --workers 8 [--scale-alpha 0.01 --det-alpha 0.005 | --base zbank]
python benchmarks/anomaly/frontend_run.py --limit 60 --workers 4
python benchmarks/anomaly/frontend_loglik.py --limit 30 --workers 3
python benchmarks/anomaly/fred_anomaly.py --limit 100 --workers 4
python benchmarks/regression_frontend.py --seeds 30 --workers 6
python benchmarks/river_frontend.py --seeds 30 --workers 6   # pip install river
python benchmarks/river_data_frontend.py --workers 6
python benchmarks/ablation_frontend.py --seeds 30 --workers 6
python benchmarks/ablation_river.py --workers 6   # pip install ice-skaters
```
UCR data: see RESEARCH.md (download + extract into `data/UCR_Anomaly_FullData`).
FRED data: cached under `benchmarks/data/` (see `benchmarks/fred.py`).
