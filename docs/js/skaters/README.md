# skaters

[![npm version](https://img.shields.io/npm/v/skaters)](https://www.npmjs.com/package/skaters)
[![npm downloads](https://img.shields.io/npm/dm/skaters)](https://www.npmjs.com/package/skaters)
[![license: MIT](https://img.shields.io/npm/l/skaters)](https://github.com/microprediction/skaters/blob/main/LICENSE)

### ▶ [Watch it race live in your browser](https://skaters.microprediction.org/demos/race.html)

The actual JavaScript port racing `arima`, `@bsull/augurs` (ETS and MSTL), and
Prophet (Stan compiled to WASM) on 150 real FRED series — every method scored on
held-out log-likelihood, rolling one-step-ahead, no server. `laplace` keeps up
with a moving average while the refit baselines grind. **[Start here →](https://skaters.microprediction.org/demos/race.html)**

---

Fast univariate **online distributional** time-series forecasting. Zero
dependencies, runs in Node or the browser. This is the JavaScript port of the
Python [`skaters`](https://pypi.org/project/skaters/) package, numerically
identical to it within `1e-6` (enforced on every release by a parity checker
against the Python reference).

Every prediction is a full predictive **distribution** (a `Dist`), so it can be
scored on log-likelihood — not just a point or an interval.

## Install

```
npm install skaters
```

## Quickstart

```js
import { laplace } from "skaters";

const f = laplace(1);          // online, O(1) per step, the general-purpose default
let state = null;
for (const y of stream) {
  let dists;
  [dists, state] = f(y, state);
  const d = dists[0];
  d.mean;             // point forecast
  d.std;              // uncertainty
  d.quantile(0.975);  // 97.5th percentile
  d.logpdf(y);        // a real density — scorable on log-likelihood
  d.crps(y);          // ...and on CRPS
}
```

In the browser, import the hosted module directly — no build step:

```html
<script type="module">
  import { laplace } from "https://skaters.microprediction.org/js/skaters/index.mjs";
</script>
```

## What's exported

`laplace` and `buildCandidates`; the `Dist` object; transforms (`difference`,
`standardize`, `garch`, `ar`, `holtLinear`, `powerTransform`, …); ensembles
(`precisionWeightedEnsemble`, `bayesianEnsemble`, `terminalLeafEnsemble`); leaves
(`scaleMixtureLeaf`, `crpsLeaf`, `garchLeaf`); `multiscale`, `sticky`, periodicity
and covariance helpers; and spec (de)serialisers. See `index.mjs` for the full
surface.

## Notes

- `laplace` is the general-purpose default; its signature is
  `laplace(k = 1, objective = "crps", sticky = true, scales = null)`. On
  price/return series with volatility clustering there is no free lunch — a
  dedicated GARCH-t model is the right tool there.
- The port is a faithful mirror of the Python package; the two agree to `1e-6`.

## Links

- Docs & benchmarks: <https://skaters.microprediction.org>
- Paper: *Transforms All the Way Down — Automatic Online Distributional Forecasting by Conjugation*
- Python package: <https://pypi.org/project/skaters/>
- Source: <https://github.com/microprediction/skaters>

MIT © Peter Cotton
