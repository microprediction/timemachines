# Changelog

All notable changes to the `skaters` npm package (the JavaScript port). The port
tracks the Python [`skaters`](https://pypi.org/project/skaters/) package and is
kept numerically identical to it within `1e-6`, enforced by the parity checker on
every release.

## 0.12.0

Better default forecaster. `laplace`'s terminal-leaf `scaleAlpha` (the residual-variance
EWMA rate — how fast the predictive scale tracks changing volatility) now defaults to
**0.03** instead of 0.01. On the continuous FRED universe this beats the old default on
held-out **log-likelihood** (~+0.02 nats, ~79% of series) **and CRPS** (~80% of series), on
both non-price and price series — validated out-of-sample (see
`benchmarks/leaf_experiments/`). New optional `scaleAlpha` argument on `laplace`; pass
`scaleAlpha: 0.01` to reproduce the previous default. Minor bump because default predictions
change; Python/JS parity re-verified to 1e-6.

## 0.11.4

Docs: lead the README with the live in-browser race demo
(<https://skaters.microprediction.org/demos/race.html>) — the JS port racing
`arima`, `@bsull/augurs` (ETS and MSTL), and Prophet (Stan compiled to WASM) on
real FRED series, scored on held-out log-likelihood. No code changes.

## 0.11.3

Fix: `Dist.logpdf` returned `-Infinity` for finite inputs when the scale had
collapsed near a Dirac (e.g. on a near-constant stream), because `log(sum(w *
pdf))` underflowed each `exp(-z**2/2)` to `0`. A Gaussian mixture is strictly
positive everywhere, so this is now computed as a log-sum-exp over the
per-component log densities and is always finite for finite `x`. Numerically
identical to the old path wherever it did not underflow (parity vectors
unchanged); only the former `-Infinity` cases now return the correct large
finite value.

## 0.11.2

Fix: RLS covariance windup in the `ar` and `grouped_ar` transforms. Under
low-excitation input the RLS `P` matrix inflated by `1/lam` each step and
eventually overflowed to `Inf` (~74k steps), giving `NaN` coefficients and a
non-finite forecast. `P` is now reset if it grows implausibly large or turns
non-finite (keeping the coefficients); it never triggers on well-excited data, so
results and Python↔JS parity are unchanged.

## 0.11.1

Fix: the CRPS leaf's exponentiated-gradient weight update could overflow `exp()`
to `Inf` on some streams, normalizing to `NaN` weights and throwing in the `Dist`
constructor. The step is now stabilized by subtracting the max exponent
(mathematically invariant, so results and Python↔JS parity are unchanged), with a
guard that keeps the current weights on any degenerate update.

## 0.11.0

Initial npm release of the JavaScript port. Zero-dependency ES modules for Node
and the browser. Exports `laplace` and `buildCandidates`, the `Dist` object,
transforms (`difference`, `standardize`, `garch`, `ar`, `holtLinear`,
`powerTransform`, …), ensembles, leaves (`scaleMixtureLeaf`, `crpsLeaf`,
`garchLeaf`), `multiscale`, `sticky`, periodicity and covariance helpers, and the
spec (de)serialisers. Version chosen to align with the Python package.
