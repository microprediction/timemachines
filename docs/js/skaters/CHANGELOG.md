# Changelog

All notable changes to the `skaters` npm package (the JavaScript port). The port
tracks the Python [`skaters`](https://pypi.org/project/skaters/) package and is
kept numerically identical to it within `1e-6`, enforced by the parity checker on
every release.

## Unreleased

State round-trips through JSON bit-exactly — a skater is a fold. New exports
`stateToJSON(state)`, `stateFromJSON(obj)` and `assertPlainState(state)`
(`state.mjs`): the first turns every `Dist` (and the GPD-spliced predictive)
into its `toDict()` form and throws, naming the path, on anything JSON would
silently damage (a function, `Map`, `Set`, typed array, foreign class,
`undefined`, or non-finite number); the second rebuilds the `Dist`s without
renormalising. `Dist.fromDict` now goes through `Dist.trusted`, a constructor path that
keeps weights that already sum to one bit-for-bit (the old path divided by a
total that was one only up to rounding and moved the last bit on a fraction of
mixtures). Two skaters held non-plain state and are fixed: `sticky` kept its
frequency table in a `Map` (now an array of `[value, weight]` pairs in
first-seen order, so the tie-break still matches Python's dict), and `search`
kept the pool's skater closures in state (now rebuilt from each entry's
recipe by the wrapper). A new release gate, `parity/roundtrip.mjs`, checkpoints
every exported skater every 100 steps over a 620-step series, restores it
through `JSON.parse(JSON.stringify(...))`, and requires the restored copy to
match the original byte-for-byte in both predictives and state for the next
25 steps, plus a cold-start check: a fresh process that imports `state.mjs`
and nothing else must still decode a spliced dist. The spliced decoder is now
registered by a named `registerSplicedDist()` call rather than a bare
`import "./tails.mjs"`, which a bundler drops because the package declares
`sideEffects: false` (esbuild reports `ignored-bare-import`); a consumer that
only rehydrates checkpoints, never running a skater, previously got a bundle
with no spliced support and every restore threw. The gate also fails on any
bare side-effect import the package does not declare. Removed a duplicate
`get length()` in `dist.mjs`, where the second definition silently won.
The parity checker also runs every scenario a second time across
a JSON restore at the burn-in step, against vectors the Python generator
produced across a pickle restore. Numerics are unchanged: parity is
byte-identical.

## 0.13.0

GPD tails by default — the conditional tail fit. Every `laplace` predictive now
splices censored-ML generalized-Pareto tails over the body's own tail region
(`tails.mjs`; pass `tails = "gaussian"` to opt out). The body's matured PIT
defines frozen warm-up-quantile thresholds per horizon; exceedances fit a GPD
per side (Grimshaw profile); the predictive keeps the body density in the
interior and the GPD beyond, so `logpdf`, `cdf`, `quantile`, `crps` and the
parade's `state.z` all read the corrected tail. Measured on non-price FRED:
**+0.027 nats/tick** median held-out log-likelihood (84/85 series) and the
parade-z alarm rate at nominal 1e-3 drops from ~8x over budget to ~1.4x
(consistent with the genuine-anomaly base rate); at k=3 the horizon-3
predictive gains the same (+0.027 nats, z3 rate at nominal 1e-3:
9.0e-3 -> 1.4e-3). CRPS, the default leaf objective, is unchanged to
within noise. The exceedance
rate is an EWMA (`rateAlpha = 0.002`) so the splice re-calibrates after
regime shifts; intake is winsorised at the fitted 1-in-1000 excess with a
changepoint escape (masking resistance); spliced predictives round-trip
through `Dist.fromDict`. ~5% runtime. Parity: 105,658 values, 54 scenarios
(new `gpd_tails` scenario exercises the splice directly).

## 0.12.1

- Skater state is now pure data (picklable/serialisable for checkpoint-restore
  deployments): the terminal ensemble's per-horizon leaf closures moved out of
  the state dict into the wrapper, mirroring how multiscale holds its
  sub-skaters. No numerical change (parity: 104,678 values, 53 scenarios).

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
