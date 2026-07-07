// Sticky / lattice projection — JS port of skaters/sticky.py.
// Recency-weighted frequency table of exact values; atoms fire at the values
// revisited above a noise floor (top-k by frequency), mean-preserving. The
// single-spike behaviour is the max_atoms=1 case. Uses a Map so numeric-key
// insertion order (hence the sort tie-break) matches Python's dict.

import { fsum, Dist } from "./dist.mjs";

export function sticky(base, k = 1, propensityAlpha = 0.05, spikeFrac = 0.005,
                       threshMult = 1.8, maxAtoms = 6, pruneEps = 1e-6) {
  function _skater(y, state) {
    if (state === null || state === undefined) {
      state = { base: null, counts: new Map() };
    }
    const [dists, baseState] = base(y, state.base);
    state.base = baseState;

    // recency-weighted frequency table of exact values.
    const c = state.counts;
    const drop = [];
    for (const key of c.keys()) {
      const v = c.get(key) * (1.0 - propensityAlpha);
      if (v < pruneEps) drop.push(key);
      else c.set(key, v);
    }
    for (const key of drop) c.delete(key);
    c.set(y, (c.get(y) || 0.0) + propensityAlpha);

    // lattice atoms = revisited values above the floor, top-k by weight.
    const thr = threshMult * propensityAlpha;
    let atoms = [];
    for (const [v, w] of c) if (w > thr) atoms.push([v, w]);
    // stable sort by descending weight; ties keep insertion order (matches Python)
    atoms = atoms.map((a, i) => [a, i])
      .sort((A, B) => (B[0][1] - A[0][1]) || (A[1] - B[1]))
      .map((p) => p[0])
      .slice(0, maxAtoms);

    const out = [];
    for (const d of dists) {
      if (atoms.length === 0) {
        out.push(d);
        continue;
      }
      const sw = fsum(atoms.map((a) => a[1]));
      const P = Math.min(sw, 0.999);
      const pc = 1.0 - P;
      const atomMean = fsum(atoms.map((a) => a[1] * a[0])) / sw;
      const spikeStd = Math.max(spikeFrac * d.std, 1e-9);
      if (pc <= 1e-9) {
        out.push(new Dist(atoms.map(([v, w]) => [w / sw, v, spikeStd])));
        continue;
      }
      const mu = d.mean;
      const delta = (P * (mu - atomMean)) / pc;
      const comps = atoms.map(([v, w]) => [P * (w / sw), v, spikeStd]);
      for (const [w, m, s] of d.components) comps.push([pc * w, m + delta, s]);
      out.push(new Dist(comps));
    }
    return [out, state];
  }
  _skater.skaterName = `sticky(${base.skaterName || "?"})`;
  return _skater;
}
