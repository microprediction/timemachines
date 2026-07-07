// The prediction parade — JS port of skaters/parade.py.
//
// Resolve each arriving observation against the predictions previously made
// for it: state.pit[m-1] is the PIT of y under the m-step-ahead predictive
// issued m steps ago (roughly Uniform(0,1) when calibrated); state.z[m-1] is
// the same through the standard-normal quantile (roughly N(0,1)). Entries are
// null until horizon m has matured. Pass-through for the forecasts themselves.
// Named for the prediction parade of the timemachines package.

import { Dist } from "./dist.mjs";

const STD_NORMAL = Dist.gaussian(0.0, 1.0);
// Clamp the PIT away from {0,1}: |z| is then bounded by ~7.03 and the
// bisection in Dist.quantile stays inside its +-8 sigma bracket. No input can
// produce an infinite z. Non-finite CDF values leave the entry null.
const EPS = 1e-12;

export function parade(base, k) {
  return function skater(y, state) {
    if (state === null || state === undefined) {
      state = { base: null, pending: [], pit: new Array(k).fill(null), z: new Array(k).fill(null) };
    }
    const pend = state.pending;
    const n = pend.length;
    const pit = new Array(k).fill(null);
    const z = new Array(k).fill(null);
    for (let m = 1; m <= k; m++) {
      if (m <= n) {
        const d = pend[n - m][m - 1];      // issued m steps ago, horizon m
        let u = d.cdf(y);
        if (!Number.isFinite(u)) continue; // degenerate predictive or bad y
        u = Math.min(Math.max(u, EPS), 1.0 - EPS);
        pit[m - 1] = u;
        z[m - 1] = STD_NORMAL.quantile(u);
      }
    }
    const [dists, st] = base(y, state.base);
    state.base = st;
    pend.push(dists.slice());
    if (pend.length > k) pend.shift();
    state.pit = pit;
    state.z = z;
    return [dists, state];
  };
}
