// Adaptive search over the transform tree — JS port of skaters/search.py.

import { Dist } from "./dist.mjs";
import { leaf } from "./leaf.mjs";
import { conjugate } from "./conjugate.mjs";
import {
  difference, fractionalDifference, standardize, emaTransform, garch,
  seasonalDifference, powerTransform, ar, groupedAr, drift, holtLinear, theta,
} from "./transform.mjs";
import { periodDetector, topPeriods } from "./periodicity.mjs";

// Grammar: [name, factory, costPerObs]. Order matters for parity.
export const TRANSFORMS = [
  ["ema_t(0.05)", () => emaTransform(0.05), 1],
  ["ema_t(0.1)", () => emaTransform(0.1), 1],
  ["ema_t(0.3)", () => emaTransform(0.3), 1],
  ["diff", () => difference(), 1],
  ["std(0.05)", () => standardize(0.05), 1],
  ["frac(0.3)", () => fractionalDifference(0.3, 30), 3],
  ["garch", () => garch(), 1],
  ["pow(0.5)", () => powerTransform(0.5), 1],
  ["ar(2)", () => ar(2), 2],
  ["ar(5)", () => ar(5, 0.99, 1.0, 1), 3],
  ["gar(16)", () => groupedAr(16), 2],
  ["theta(0.1)", () => theta(0.1), 1],
  ["theta(0.3)", () => theta(0.3), 1],
  ["drift", () => drift(), 1],
  ["drift(0.01)", () => drift(0.01, 0.002), 1],
  ["holt(0.1,0.05)", () => holtLinear(0.1, 0.05), 1],
  ["holt(0.3,0.1)", () => holtLinear(0.3, 0.1), 1],
  ["seas(7)", () => seasonalDifference(7), 1],
  ["seas(12)", () => seasonalDifference(12), 1],
  ["seas(24)", () => seasonalDifference(24), 1],
];

function avgW(entry, k) {
  let s = 0.0;
  for (const w of entry.log_w) s += w;
  return s / k;
}

function makeEntry(skaterFn, depth, recipe, k, cost = 0.0) {
  return {
    f: skaterFn,
    s: null,
    depth,
    recipe,
    cost,
    age: 0,
    warmed: false,
    log_w: new Array(k).fill(0.0),
    queues: Array.from({ length: k }, () => []),
    dists: null,
  };
}

function warmup(entry, buffer, k) {
  for (const y of buffer) {
    const [dists, s] = entry.f(y, entry.s);
    entry.s = s;
    entry.dists = dists;
    entry.age += 1;
  }
  if (entry.dists !== null) {
    for (let h = 0; h < k; h++) {
      entry.queues[h].length = 0;
      entry.queues[h].push(entry.dists[h]);
    }
  }
  entry.warmed = true;
}

function initPool(k, costBudget) {
  const pool = [];
  const leafEntry = makeEntry(leaf(k), 0, [], k, 1.0);
  leafEntry.warmed = true;
  pool.push(leafEntry);
  for (const [tName, tFactory, tCost] of TRANSFORMS) {
    const candidateCost = 1.0 + tCost;
    if (candidateCost > costBudget) continue;
    const f = conjugate(leaf(k), tFactory(), k);
    f.skaterName = `${tName}|leaf`;
    const entry = makeEntry(f, 1, [tName], k, candidateCost);
    entry.warmed = true;
    pool.push(entry);
  }
  return pool;
}

function buildFromRecipe(recipe, k, transforms) {
  const lookup = new Map(transforms.map(([name, factory]) => [name, factory]));
  let f = leaf(k);
  for (const tName of recipe) f = conjugate(f, lookup.get(tName)(), k);
  return f;
}

function expand(pool, k, topN, maxDepth, transforms, costBudget) {
  const scored = pool.map((e, i) => ({ avg: avgW(e, k), i, e }));
  scored.sort((a, b) => (b.avg - a.avg) || (b.i - a.i));

  const existing = new Set(pool.map((e) => e.recipe.join("|")));
  const children = [];
  for (const { e: parent } of scored.slice(0, topN)) {
    if (parent.depth >= maxDepth) continue;
    for (const [tName, , tCost] of transforms) {
      const childCost = parent.cost + tCost;
      if (childCost > costBudget) continue;
      if (parent.recipe.length && parent.recipe[parent.recipe.length - 1] === tName) continue;
      const newRecipe = parent.recipe.concat([tName]);
      const key = newRecipe.join("|");
      if (existing.has(key)) continue;
      existing.add(key);
      const childFn = buildFromRecipe(newRecipe, k, transforms);
      childFn.skaterName = newRecipe.join("|") + "|leaf";
      children.push(makeEntry(childFn, newRecipe.length, newRecipe, k, childCost));
    }
  }
  return children;
}

function prune(pool, threshold, maxPool, k) {
  if (pool.length <= 1) return;
  let best = -Infinity;
  for (const e of pool) best = Math.max(best, avgW(e, k));

  let i = 0;
  while (i < pool.length) {
    if (avgW(pool[i], k) < best + threshold && pool.length > 1) pool.splice(i, 1);
    else i += 1;
  }
  while (pool.length > maxPool) {
    let worstIdx = 0;
    let worst = Infinity;
    for (let j = 0; j < pool.length; j++) {
      const a = avgW(pool[j], k);
      if (a < worst) {
        worst = a;
        worstIdx = j;
      }
    }
    pool.splice(worstIdx, 1);
  }
}

export function search({
  k = 1,
  learningRate = 0.5,
  complexityPenalty = 0.02,
  maxPool = 30,
  expandInterval = 100,
  expandTopN = 3,
  maxDepth = 3,
  replayBuffer = 500,
  pruneThreshold = -50.0,
  maxComponents = 20,
  costBudget = Infinity,
} = {}) {
  const pdFunc = periodDetector();

  function _search(y, state) {
    if (state === null || state === undefined) {
      state = {
        pool: initPool(k, costBudget),
        n_obs: 0,
        buffer: [],
        pd_state: null,
        detected_periods: new Set(),
        transforms: TRANSFORMS.slice(),
      };
    }

    state.n_obs += 1;
    state.buffer.push(y);
    if (state.buffer.length > replayBuffer) state.buffer.shift();

    const pool = state.pool;

    for (const entry of pool) {
      const [dists, s] = entry.f(y, entry.s);
      entry.s = s;
      entry.dists = dists;
      entry.age += 1;
    }

    // Queue current predictions, then resolve the ones that have matured.
    // Horizon h (0-based) is (h+1)-step-ahead, so the Dist issued h+1 steps
    // ago is the one that targeted the current y: buffer h+1 predictions
    // before scoring. (At h=0 this is the ordinary one-step lag.)
    for (const entry of pool) {
      for (let h = 0; h < k; h++) {
        const q = entry.queues[h];
        q.push(entry.dists[h]);
        if (q.length > h + 1) {
          const pastDist = q.shift();
          if (entry.warmed) {
            // Bounded loss: clamp both tails so neither -inf nor a +inf
            // (exact hit on a Dirac atom) can dominate or NaN-poison the
            // log-weight; `!(lp >= -20)` also catches NaN.
            let lp = pastDist.logpdf(y);
            if (lp > 20.0) lp = 20.0;
            else if (!(lp >= -20.0)) lp = -20.0;
            entry.log_w[h] += learningRate * lp - complexityPenalty * entry.depth;
          }
        }
      }
    }

    const [scores, pdState] = pdFunc(y, state.pd_state);
    state.pd_state = pdState;

    if (state.n_obs % expandInterval === 0 && state.n_obs > 10) {
      const detected = topPeriods(scores, 0.3, 3);
      for (const period of detected) {
        if (!state.detected_periods.has(period)) {
          state.detected_periods.add(period);
          const tName = `seas(${period})`;
          state.transforms.push([tName, () => seasonalDifference(period), 2]);
        }
      }
      const newChildren = expand(pool, k, expandTopN, maxDepth, state.transforms, costBudget);
      for (const child of newChildren) warmup(child, state.buffer, k);
      for (const child of newChildren) pool.push(child);
      prune(pool, pruneThreshold, maxPool, k);
    }

    const combined = [];
    for (let h = 0; h < k; h++) {
      const logWs = pool.map((e) => e.log_w[h]);
      const maxLw = Math.max(...logWs);
      let weights;
      if (Number.isFinite(maxLw)) weights = logWs.map((lw) => Math.exp(lw - maxLw));
      else weights = new Array(pool.length).fill(1.0);
      const horizonDists = pool.map((e) => e.dists[h]);
      let dist = Dist.combine(horizonDists, weights);
      if (dist.length > maxComponents) dist = dist.prune(maxComponents);
      combined.push(dist);
    }
    return [combined, state];
  }
  _search.skaterName = `search(k=${k})`;
  return _search;
}
