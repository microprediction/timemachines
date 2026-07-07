// Bayesian model-averaging ensemble — JS port of skaters/bayesian.py.
//
// Weights models by cumulative shrunk log-likelihood minus a per-depth
// complexity penalty, combining predictions via Dist.combine with softmax
// weights (XGBoost-inspired regularization).

import { Dist } from "./dist.mjs";

export function bayesianEnsemble(
  skaters,
  {
    k = 1,
    learningRate = 0.5,
    complexityPenalty = 0.0,
    depths = null,
    priorLogWeights = null,
    maxComponents = 20,
  } = {}
) {
  const n = skaters.length;
  if (n <= 0) throw new Error("ensemble needs at least one skater");
  if (!(learningRate > 0 && learningRate <= 1)) throw new Error("learningRate in (0,1]");
  if (complexityPenalty < 0) throw new Error("complexityPenalty >= 0");

  const d = depths === null ? new Array(n).fill(0) : depths;
  const prior = priorLogWeights === null ? new Array(n).fill(0.0) : priorLogWeights;

  function _skater(y, state) {
    if (state === null || state === undefined) {
      state = {
        sub: new Array(n).fill(null),
        queues: Array.from({ length: n }, () => Array.from({ length: k }, () => [])),
        log_w: Array.from({ length: n }, (_, i) => new Array(k).fill(prior[i])),
        n_obs: 0,
      };
    }
    state.n_obs += 1;

    const allDists = [];
    for (let i = 0; i < n; i++) {
      const [distsI, sub] = skaters[i](y, state.sub[i]);
      state.sub[i] = sub;
      allDists.push(distsI);
    }

    // Enqueue current predictions, then resolve the ones that have matured.
    // Horizon h (0-based) is (h+1)-step-ahead, so the Dist issued h+1 steps ago
    // is the one that targeted the current y: buffer h+1 predictions before
    // scoring. (At h=0 this is the ordinary one-step lag.)
    for (let i = 0; i < n; i++) {
      for (let h = 0; h < k; h++) {
        const q = state.queues[i][h];
        q.push(allDists[i][h]);
        if (q.length > h + 1) {
          const pastDist = q.shift();
          // Bounded loss (mixability): clamp to a finite band so neither a -inf
          // nor a +inf (exact hit on a Dirac atom) can dominate or NaN-poison
          // log_w. The `!(lp >= -20)` arm also catches NaN.
          let lp = pastDist.logpdf(y);
          if (lp > 20.0) lp = 20.0;
          else if (!(lp >= -20.0)) lp = -20.0;
          state.log_w[i][h] += learningRate * lp - complexityPenalty * d[i];
        }
      }
    }

    // Softmax weights per horizon, then combine.
    const combined = [];
    for (let h = 0; h < k; h++) {
      const logWs = [];
      for (let i = 0; i < n; i++) logWs.push(state.log_w[i][h]);
      const maxLw = Math.max(...logWs);
      let weights;
      if (Number.isFinite(maxLw)) weights = logWs.map((lw) => Math.exp(lw - maxLw));
      else weights = new Array(n).fill(1.0);

      const horizonDists = [];
      for (let i = 0; i < n; i++) horizonDists.push(allDists[i][h]);
      let dist = Dist.combine(horizonDists, weights);
      if (dist.length > maxComponents) dist = dist.prune(maxComponents);
      combined.push(dist);
    }
    return [combined, state];
  }
  _skater.skaterName = `bayesian_ensemble(n=${n}, k=${k}, eta=${learningRate})`;
  return _skater;
}
