// Multi-scale ensemble — JS port of skaters/multiscale.py.
//
// Combine forecasters running on decimated clocks: one base instance per
// stride s (with s phase-shifted copies so the freshest coarse forecast is
// always anchored at the current tick; exactly one copy steps per tick), and
// per fine horizon h mix each eligible scale's (s <= h) predictive Dist under
// likelihood softmax weights. Granularity is the one candidate axis that
// cannot live inside the transform pool (decimation is many-ticks-to-one and
// a coarse model emits no fine one-step predictive to weight), hence a
// wrapper. With scales == [1] (e.g. k == 1) the base is returned unwrapped.

import { Dist } from "./dist.mjs";

const LOGPDF_FLOOR = -20.0;

export function multiscale(base, k, { scales = null, forget = 0.99, maxComponents = 20 } = {}) {
  if (scales === null) scales = [1, Math.ceil(Math.sqrt(k)), k];
  scales = [...new Set(scales.map((s) => Math.trunc(s)).filter((s) => s >= 1 && s <= k))]
    .sort((a, b) => a - b);
  if (!(scales.length && scales[0] === 1)) throw new Error("scales must include 1");
  if (scales.length === 1) return base(k);
  const subs = {};
  for (const s of scales) subs[s] = base(Math.max(1, Math.ceil(k / s)));

  return function skater(y, state) {
    if (state === null || state === undefined) {
      state = { t: 0, phase: {}, pending: {}, latest: {}, score: {} };
      for (const s of scales) {
        state.phase[s] = new Array(s).fill(null);
        state.pending[s] = new Array(s).fill(null);
        state.score[s] = null;
      }
    }
    const t = state.t;
    for (const s of scales) {
      const ph = t % s;
      const prev = state.pending[s][ph];
      if (prev !== null) {
        const lp = Math.max(prev.logpdf(y), LOGPDF_FLOOR);
        const m = state.score[s];
        state.score[s] = m === null ? lp : forget * m + (1.0 - forget) * lp;
      }
      const [dists, st] = subs[s](y, state.phase[s][ph]);
      state.phase[s][ph] = st;
      state.pending[s][ph] = dists[0];
      state.latest[s] = dists;
    }
    state.t = t + 1;

    const ms = scales.map((s) => state.score[s]).filter((m) => m !== null);
    const top = ms.length ? Math.max(...ms) : 0.0;
    const out = [];
    for (let h = 1; h <= k; h++) {
      const fcs = [], wts = [];
      for (const s of scales) {
        if (s > h || !(s in state.latest)) continue;
        const j = Math.max(1, Math.floor(h / s + 0.5));   // half-up, matches Python
        const dists = state.latest[s];
        if (j - 1 >= dists.length) continue;
        fcs.push(dists[j - 1]);
        const m = state.score[s];
        wts.push(Math.exp((m === null ? top : m) - top));
      }
      // One eligible scale (e.g. h < every coarse stride): pass its Dist
      // through untouched — no renormalise, no prune.
      out.push(fcs.length === 1 ? fcs[0] : Dist.combine(fcs, wts).prune(maxComponents));
    }
    return [out, state];
  };
}
