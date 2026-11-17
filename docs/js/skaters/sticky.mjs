// Sticky / lattice projection — JS port of skaters/sticky.py.
// Recency-weighted frequency table of exact values; atoms fire at the values
// revisited above a noise floor (top-k by frequency), mean-preserving. The
// single-spike behaviour is the max_atoms=1 case. The table is an array of
// [value, weight] pairs in first-seen order, so the sort tie-break matches
// Python's dict insertion order AND the state survives JSON: a Map flattens
// to {}, and a plain object would reorder integer-like keys.

import { fsum, Dist } from "./dist.mjs";

export function sticky(base, k = 1, propensityAlpha = 0.05, spikeFrac = 0.005,
                       threshMult = 1.8, maxAtoms = 6, pruneEps = 1e-6) {
  function _skater(y, state) {
    if (state === null || state === undefined) {
      state = { base: null, counts: [] };
    }
    const [dists, baseState] = base(y, state.base);
    state.base = baseState;

    // recency-weighted frequency table of exact values: decay, prune in place
    // (order preserved), then credit y — an existing entry keeps its slot, a
    // new or just-pruned value goes to the end, exactly as a Map/dict would.
    const c = state.counts;
    let j = 0;
    for (let i = 0; i < c.length; i++) {
      const v = c[i][1] * (1.0 - propensityAlpha);
      if (v < pruneEps) continue;
      c[i][1] = v;
      c[j++] = c[i];
    }
    c.length = j;
    let hit = -1;
    for (let i = 0; i < c.length; i++) {
      const v = c[i][0];
      if (v === y || (v !== v && y !== y)) { hit = i; break; }   // SameValueZero, as Map.get
    }
    if (hit >= 0) c[hit][1] += propensityAlpha;
    else c.push([y, propensityAlpha]);

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
