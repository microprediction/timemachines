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
// A leading missing tick has no prior forecast to age and no observation has
// fixed the series' scale yet, so emit a deliberately wide zero-centred
// Gaussian rather than a confident guess. The tree is left untouched, so the
// first finite value initializes it exactly as a true first observation would.
const NO_INFO_STD = 1e6;

export function parade(base, k) {
  return function skater(y, state) {
    if (state === null || state === undefined) {
      state = { base: null, pending: [], pit: new Array(k).fill(null), z: new Array(k).fill(null), skipped: 0 };
    }
    const pend = state.pending;
    const n = pend.length;
    // Missing observation — port of the Python parade's missing-tick rule. A
    // non-finite y must never reach the tree: an EWMA fed NaN is poisoned
    // permanently (mu + alpha*(nan - mu) = nan) and no later clean data
    // recovers it. Treat the tick as "no observation": time advanced,
    // information did not. Leave the base state untouched and SHIFT the fan,
    // so the previous tick's horizon h+1 becomes this tick's horizon h and the
    // forecast ages by one step. After k consecutive gaps h=k is held.
    if (!Number.isFinite(y)) {
      state.skipped = (state.skipped ?? 0) + 1;
      state.pit = new Array(k).fill(null);
      state.z = new Array(k).fill(null);
      if (!n) {
        const wide = [];
        for (let h = 0; h < k; h++) wide.push(Dist.gaussian(0.0, NO_INFO_STD));
        return [wide, state];
      }
      const prev = pend[n - 1];
      const shifted = [];
      for (let h = 1; h <= k; h++) shifted.push(prev[Math.min(h, prev.length - 1)]);
      pend.push(shifted);
      if (pend.length > k) pend.shift();
      return [shifted, state];
    }
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
    // Port of the Python parade's extreme-input gate: double arithmetic in
    // the transforms dies long before the float range ends, so clamp the
    // observation before the tree consumes it. Magnitude-relative window
    // (NOT sigma-relative); PIT/z above are computed on the raw y; identity
    // on any stream doubles represent comfortably. A tail-spliced
    // predictive's exact moments are numeric grids, so read the body's
    // closed forms: a location/scale proxy is all the gate needs.
    let yFed = y;
    if (Number.isFinite(yFed)) {
      yFed = Math.min(Math.max(yFed, -1e60), 1e60);
      if (n) {
        let d1 = pend[n - 1][0];              // the 1-step predictive for y
        d1 = d1.body ?? d1;
        const mp = d1.mean, sp = d1.std;
        if (Number.isFinite(mp) && Number.isFinite(sp)) {
          const w = 1e12 * (1.0 + Math.abs(mp) + sp);
          yFed = Math.min(Math.max(yFed, mp - w), mp + w);
        }
      }
    }
    const [dists, st] = base(yFed, state.base);
    state.base = st;
    pend.push(dists.slice());
    if (pend.length > k) pend.shift();
    state.pit = pit;
    state.z = z;
    return [dists, state];
  };
}
