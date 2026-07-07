// Terminal-leaf ensemble — JS port of skaters/terminal.py.
// Mix the sub-models for the MEAN; model the combined residual with one
// terminal leaf (default the scale-mixture leaf), so its shape reaches the
// output undiluted.

import { fsum, Dist } from "./dist.mjs";
import { crpsLeaf } from "./leaf.mjs";

export function terminalLeafEnsemble(skaters, {
  leafFn = crpsLeaf,
  k = 1,
  learningRate = 0.5,
  complexityPenalty = 0.0,
  depths = null,
  priorLogWeights = null,
  maxComponents = 20,
  forget = 1.0,
} = {}) {
  const n = skaters.length;
  const d = depths === null ? new Array(n).fill(0) : depths;
  const prior = priorLogWeights === null ? new Array(n).fill(0.0) : priorLogWeights;

  function _skater(y, state) {
    if (state === null || state === undefined) {
      state = {
        sub: new Array(n).fill(null),
        qdist: Array.from({ length: n }, () => []),
        log_w: prior.slice(),
        tleaf: Array.from({ length: k }, () => leafFn(1)),
        leafState: new Array(k).fill(null),
        leafPred: new Array(k).fill(null),
        meanQ: Array.from({ length: k }, () => []),
      };
    }

    const allDists = [];
    for (let i = 0; i < n; i++) {
      const [di, s] = skaters[i](y, state.sub[i]);
      state.sub[i] = s;
      allDists.push(di);
    }

    for (let i = 0; i < n; i++) {
      const q = state.qdist[i];
      if (q.length) {
        // Bounded loss (mixability): clamp to a finite band so neither a -inf
        // (y far from every component) nor a +inf (an exact hit on a Dirac atom,
        // e.g. the sticky lattice path) can dominate or NaN-poison log_w. The
        // `!(lp >= -20)` arm also catches NaN.
        let lp = q.shift().logpdf(y);
        if (lp > 20.0) lp = 20.0;
        else if (!(lp >= -20.0)) lp = -20.0;
        state.log_w[i] = forget * state.log_w[i] + learningRate * lp - complexityPenalty * d[i];
      }
      q.push(allDists[i][0]);
    }

    const maxLw = Math.max(...state.log_w);
    const w = state.log_w.map((lw) => Math.exp(lw - maxLw));
    const tot = fsum(w);

    const combined = [];
    for (let h = 0; h < k; h++) {
      const muH = fsum(w.map((wi, i) => wi * allDists[i][h].mean)) / tot;

      const mq = state.meanQ[h];
      if (mq.length >= h + 1) {
        const r = y - mq.shift();
        const [ld, ls] = state.tleaf[h](r, state.leafState[h]);
        state.leafState[h] = ls;
        state.leafPred[h] = ld[0];
      }

      let pred;
      if (state.leafPred[h] !== null) {
        pred = state.leafPred[h].shift(muH);
      } else {
        pred = Dist.combine(allDists.map((di) => di[h]), w);
        if (pred.components.length > maxComponents) pred = pred.prune(maxComponents);
      }
      combined.push(pred);
      mq.push(muH);
    }
    return [combined, state];
  }
  _skater.skaterName = `terminal_leaf_ensemble(n=${n}, k=${k})`;
  return _skater;
}
