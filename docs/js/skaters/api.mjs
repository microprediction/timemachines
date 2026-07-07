// User-facing API — JS port of skaters/api.py (named search policies).
//
// One forecaster: laplace (general). Volatility/mean-reversion are composable transforms.

import { leaf, scaleMixtureLeaf, crpsLeaf } from "./leaf.mjs";
import { conjugate } from "./conjugate.mjs";
import { terminalLeafEnsemble } from "./terminal.mjs";
import { bayesianEnsemble } from "./bayesian.mjs";
import { sticky as project } from "./sticky.mjs";
import { multiscale } from "./multiscale.mjs";
import { parade } from "./parade.mjs";
import {
  difference, fractionalDifference, standardize, emaTransform, ouTransform, drift,
  holtLinear, ar, theta, seasonalDifference, garch, powerTransform, yeoJohnson,
} from "./transform.mjs";

// ---------------------------------------------------------------------------
// Shared candidate population
// ---------------------------------------------------------------------------

export function buildCandidates(k) {
  const candidates = [];
  const depths = [];
  const groups = {};
  const families = [];                 // human family label per candidate (for introspection)
  const push = (c, d, fam) => {
    candidates.push(c);
    depths.push(d);
    families.push(fam);
    return candidates.length - 1;
  };

  // Depth 0: baseline noise
  push(leaf(k), 0, "baseline");

  // Depth 1: single EMA at various speeds
  for (const alpha of [0.01, 0.05, 0.1, 0.3]) push(conjugate(leaf(k), emaTransform(alpha), k), 1, "EMA");

  // Depth 1: differencing (pure random walk)
  groups.diff = [];
  groups.diff.push(push(conjugate(leaf(k), difference(), k), 1, "random walk"));

  // Depth 1: drift (random walk with adaptive drift)
  groups.drift = [];
  for (const [a, s] of [[0.05, 0.01], [0.01, 0.002], [0.002, 0.001], [0.0005, 0.0002]]) {
    groups.drift.push(push(conjugate(leaf(k), drift(a, s), k), 1, "drift"));
  }

  // Depth 1: Theta
  for (const a of [0.05, 0.1, 0.3]) push(conjugate(leaf(k), theta(a), k), 1, "Theta");

  // Depth 1: AR
  push(conjugate(leaf(k), ar(1), k), 1, "AR");
  push(conjugate(leaf(k), ar(2, 0.99, 1.0, 1), k), 1, "AR");

  // Depth 1: Holt linear
  groups.holt = [];
  for (const [a, b] of [[0.1, 0.02], [0.1, 0.05], [0.3, 0.1]]) {
    groups.holt.push(push(conjugate(leaf(k), holtLinear(a, b), k), 1, "Holt trend"));
  }

  // Depth 1: Seasonal differencing
  for (const period of [7, 12, 24]) push(conjugate(leaf(k), seasonalDifference(period), k), 1, "seasonal");

  // Depth 2: Seasonal differencing + EMA
  for (const period of [7, 12, 24]) {
    for (const alpha of [0.05, 0.1]) {
      push(conjugate(conjugate(leaf(k), emaTransform(alpha), k), seasonalDifference(period), k), 2, "seasonal → EMA");
    }
  }

  // Depth 2: differencing + EMA
  for (const alpha of [0.05, 0.1, 0.3]) {
    groups.diff.push(push(conjugate(conjugate(leaf(k), emaTransform(alpha), k), difference(), k), 2, "random walk → EMA"));
  }

  // Depth 2: standardize + EMA
  for (const alpha of [0.05, 0.1]) {
    push(conjugate(conjugate(leaf(k), emaTransform(alpha), k), standardize(), k), 2, "standardize → EMA");
  }

  // Depth 2: fractional diff + EMA
  groups.frac = [];
  for (const d of [0.2, 0.4]) {
    groups.frac.push(push(conjugate(conjugate(leaf(k), emaTransform(0.1), k), fractionalDifference(d, 30), k), 2, "frac-diff → EMA"));
  }

  // Depth 2: drift + EMA
  for (const [aDrift, sDrift] of [[0.002, 0.001], [0.0005, 0.0002]]) {
    for (const aEma of [0.05, 0.1]) {
      groups.drift.push(push(conjugate(conjugate(leaf(k), emaTransform(aEma), k), drift(aDrift, sDrift), k), 2, "drift → EMA"));
    }
  }

  // Depth 2: drift + Holt linear
  {
    const idx = push(conjugate(conjugate(leaf(k), holtLinear(0.1, 0.05), k), drift(0.001, 0.0005), k), 2, "drift → Holt");
    groups.drift.push(idx);
    groups.holt.push(idx);
  }

  // Depth 2: GARCH + EMA
  push(conjugate(conjugate(leaf(k), emaTransform(0.1), k), garch(), k), 2, "GARCH vol → EMA");

  // Depth 2: power transform + EMA
  push(conjugate(conjugate(leaf(k), emaTransform(0.1), k), powerTransform(0.5), k), 2, "power → EMA");

  // Depth 2: thinking fast and slow (fast tracker outside, slow scale inside)
  const fastTrackers = () => [
    emaTransform(0.3), emaTransform(0.5), holtLinear(0.4, 0.2), ar(1), drift(0.05, 0.01), difference(),
  ];
  groups.fast_slow = [];
  for (const scaleAlpha of [0.02, 0.05]) {
    for (const tracker of fastTrackers()) {
      groups.fast_slow.push(push(conjugate(conjugate(leaf(k), standardize(scaleAlpha), k), tracker, k), 2, "fast/slow"));
    }
  }

  // Coordinate prior (Yeo-Johnson): learn the coordinate the series is simple in.
  groups.coordinate = [];
  for (const L of [0.0, 0.5]) {
    for (const innerTx of [difference(), emaTransform(0.1)]) {
      groups.coordinate.push(push(conjugate(conjugate(leaf(k), innerTx, k), yeoJohnson(L), k), 2, "coordinate (Yeo-Johnson)"));
    }
  }

  // Mean-reversion prior (Ornstein-Uhlenbeck), MULTI-STEP ONLY: redundant with
  // the ema/random-walk mix at one step, so gated on k > 1 (see api.py).
  groups.mean_revert = [];
  if (k > 1) {
    for (const L of [0.0, 0.5]) {
      for (const kappa of [0.03, 0.1, 0.3]) {
        groups.mean_revert.push(push(
          conjugate(conjugate(leaf(k), ouTransform(kappa, 0.02), k), yeoJohnson(L), k), 2, "mean reversion (OU)"));
      }
    }
  }

  return [candidates, depths, groups, families];
}

// ---------------------------------------------------------------------------
// Policies
// ---------------------------------------------------------------------------

function objectiveLeaf(objective, scaleAlpha) {
  if (objective === "crps") return (k) => crpsLeaf(k, undefined, scaleAlpha);
  if (objective === "likelihood") return (k) => scaleMixtureLeaf(k, undefined, scaleAlpha);
  throw new Error(`objective must be 'crps' or 'likelihood', got ${objective}`);
}

function laplaceSingleScale(k, objective, sticky, scaleAlpha) {
  const [candidates, depths] = buildCandidates(k);
  let f = terminalLeafEnsemble(candidates, {
    k, leafFn: objectiveLeaf(objective, scaleAlpha), learningRate: 0.8, complexityPenalty: 0.005, depths, maxComponents: 20,
    forget: 0.99,
  });
  if (sticky) f = project(f, k);
  return f;
}

// Multi-scale by default at k > 1: one instance per decimation stride
// (default {1, ceil(sqrt(k)), k}), horizons mix eligible scales by likelihood.
// Pass scales = [1] for the single-scale (native fan-out) variant.
//
// scaleAlpha is the terminal leaf's residual-variance EWMA rate (how fast the
// predictive scale tracks volatility). Default 0.03 beats the older 0.01 on
// held-out log-likelihood AND CRPS across the continuous FRED universe; pass
// scaleAlpha = 0.01 to reproduce the earlier default.
export function laplace(k = 1, objective = "crps", sticky = true, scales = null, scaleAlpha = 0.03) {
  // parade adds state.pit / state.z calibration diagnostics (see parade.mjs)
  const f = parade(multiscale((kk) => laplaceSingleScale(kk, objective, sticky, scaleAlpha), k, { scales }), k);
  f.skaterName = `laplace(k=${k})`;
  return f;
}
