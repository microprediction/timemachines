// granger: laplace with trained parameters, mixed with laplace so it cannot do
// much worse. Port of src/skaters/granger.py (horizon 1); parity is checked by
// parity/check_granger.mjs against the Python reference.
//
// After Clive Granger, whose work with Bates (1969) founded forecast
// combination. The issued predictive is the Aggregating-Algorithm mixture of
// laplace and its trained twin under log loss, so its total log score is never
// more than log(1/piLap) behind laplace's (log 4 at the shipped 0.25), up to
// floating-point rounding.
//
// Parameters: the JSON files in src/skaters/granger_params (also copied next
// to this module). granger(params) takes the parsed object, so it runs in a
// browser; loadGrangerParams(frequency) reads a shipped file under Node.

import { laplace } from "./api.mjs";
import { conjugate } from "./conjugate.mjs";
import { leaf as plainLeaf, crpsLeaf } from "./leaf.mjs";
import { parade } from "./parade.mjs";
import { sticky as project } from "./sticky.mjs";
import { gpdtails } from "./tails.mjs";
import { terminalLeafEnsemble } from "./terminal.mjs";
import {
  difference, fractionalDifference, standardize, emaTransform, theta, drift, holtLinear,
  garch, seasonalDifference, seasonalAnchor, powerTransform, yeoJohnson, ar,
} from "./transform.mjs";

const NONFINITE_LP = -1e6;
const INF_LP = 1e6;

// a component log density as it enters the mixture weights: +inf (an exact hit
// on a lattice atom) counts as +1e6, NaN or -inf as -1e6
function score(lp) {
  if (lp === Infinity) return INF_LP;
  return Number.isFinite(lp) ? lp : NONFINITE_LP;
}

function noExact(kw, name) {
  if (kw.exact) throw new Error(`granger: ${name}(exact=True) is not ported`);
}

const TRANSFORMS = {
  ema_transform: (kw) => emaTransform(kw.alpha),
  difference: () => difference(),
  drift: (kw) => drift(kw.alpha, kw.shrinkage),
  theta: (kw) => theta(kw.alpha),
  ar: (kw) => ar(kw.order, kw.lam, kw.ridge, kw.decay),
  holt_linear: (kw) => holtLinear(kw.alpha, kw.beta),
  seasonal_difference: (kw) => seasonalDifference(kw.period),
  seasonal_anchor: (kw) => seasonalAnchor(kw.period, kw.alpha, kw.weight),
  standardize: (kw) => standardize(kw.alpha, kw.eps),
  fractional_difference: (kw) => fractionalDifference(kw.d, kw.window),
  garch: (kw) => garch(kw.omega, kw.alpha, kw.beta, kw.mean_alpha, kw.eps),
  power_transform: (kw) => { noExact(kw, "power_transform"); return powerTransform(kw.p); },
  yeo_johnson: (kw) => { noExact(kw, "yeo_johnson"); return yeoJohnson(kw.lmbda); },
};

const REQUIRED = {
  ema_transform: ["alpha"], difference: [], drift: ["alpha", "shrinkage"], theta: ["alpha"],
  ar: ["order", "lam", "ridge", "decay"], holt_linear: ["alpha", "beta"],
  seasonal_difference: ["period"], seasonal_anchor: ["period", "alpha", "weight"],
  standardize: ["alpha", "eps"], fractional_difference: ["d", "window"],
  garch: ["omega", "alpha", "beta", "mean_alpha", "eps"], power_transform: ["p"], yeo_johnson: ["lmbda"],
};
const INT_RANGES = { order: [1, 64], period: [1, 10000], window: [1, 10000] };

// Throw unless P is a well-formed granger parameter set.
export function validateGrangerParams(P) {
  if ((P.granger_format ?? 1) !== 1) throw new Error(`unsupported granger parameter format ${P.granger_format}`);
  if (!(typeof P.pi_lap === "number" && P.pi_lap > 0 && P.pi_lap < 1)) throw new Error(`pi_lap must lie in (0, 1), got ${P.pi_lap}`);
  const n = (P.candidates || []).length;
  if (n === 0 || P.trunk.prior_log_weights.length !== n) throw new Error("prior_log_weights must have one entry per candidate");
  for (const c of P.candidates) {
    for (const [name, kw] of c.chain) {
      if (!(name in REQUIRED)) throw new Error(`unknown transform ${name}`);
      for (const key of REQUIRED[name]) {
        if (typeof kw[key] !== "number") throw new Error(`${name}: missing or non-numeric ${key}`);
      }
      for (const [key, [lo, hi]] of Object.entries(INT_RANGES)) {
        if (key in kw && !(Number.isInteger(kw[key]) && kw[key] >= lo && kw[key] <= hi)) {
          throw new Error(`${name}: ${key} must be an integer in [${lo}, ${hi}]`);
        }
      }
      if (kw.exact) throw new Error(`${name}(exact=True) is not ported`);
    }
  }
  const G = P.gate;
  const hidden = G.l1_weight.length;
  if (hidden === 0 || G.l1_bias.length !== hidden || G.l1_weight.some((r) => r.length !== n + 3)
      || G.l2_weight.length !== n || G.l2_bias.length !== n || G.l2_weight.some((r) => r.length !== hidden)) {
    throw new Error(`gate dimensions do not match ${n} candidates (inputs must be ${n + 3})`);
  }
}

function candidate(chain, k) {
  let f = plainLeaf(k);
  for (let j = chain.length - 1; j >= 0; j--) {
    const [name, kw] = chain[j];
    const make = TRANSFORMS[name];
    if (!make) throw new Error(`granger: unknown transform ${name}`);
    f = conjugate(f, make(kw), k);
  }
  return f;
}

function expert(P, k, gateHook) {
  const cands = P.candidates.map((c) => candidate(c.chain, k));
  const depths = P.candidates.map((c) => c.depth);
  const tl = P.tleaf;
  let f = terminalLeafEnsemble(cands, {
    k,
    leafFn: (kk = 1) => crpsLeaf(kk, tl.eta, tl.scale_alpha),
    learningRate: P.trunk.learning_rate,
    complexityPenalty: P.trunk.complexity_penalty,
    depths,
    priorLogWeights: P.trunk.prior_log_weights.slice(),
    maxComponents: 20,
    forget: P.trunk.forget,
    gateHook,
  });
  f = project(f, k);
  f = gpdtails(f, k);
  f = parade(f, k);
  return f;
}

function trunkState(s) {
  for (let i = 0; i < 3; i++) {
    if (s === null || s === undefined) return null;
    s = s.base;
  }
  return s === undefined ? null : s;
}

function gate(G, feats) {
  const hidden = G.l1_weight.map((row, r) => {
    let acc = 0.0;
    for (let j = 0; j < row.length; j++) acc += row[j] * feats[j];
    return Math.tanh(G.l1_bias[r] + acc);
  });
  return G.l2_weight.map((row, r) => {
    let acc = 0.0;
    for (let j = 0; j < row.length; j++) acc += row[j] * hidden[j];
    return G.l2_bias[r] + acc;
  });
}

function features(state, n, k) {
  const z = state.z || new Array(k).fill(null);
  const zs = [];
  for (let i = 0; i < k; i++) {
    const v = z[i];
    zs.push(v === null || v === undefined ? 0.0 : Math.max(-8.0, Math.min(8.0, v)));
  }
  const desc = [state.repeat_frac, state.abs_z_mean];
  const ts = trunkState(state.learned);
  let lw;
  if (ts !== null && ts.log_w && ts.log_w.length === n) {
    const m = Math.max(...ts.log_w);
    lw = ts.log_w.map((v) => Math.max(v - m, -20.0) / 20.0);
  } else {
    lw = new Array(n).fill(0.0);
  }
  return zs.concat(desc, lw);
}

export class MixtureDist {
  constructor(dists, logW) {
    this.dists = dists;
    const m = Math.max(...logW);
    const s = Math.log(logW.reduce((a, v) => a + Math.exp(v - m), 0.0)) + m;
    this.logW = logW.map((v) => v - s);
    this.w = this.logW.map((v) => Math.exp(v));
  }

  logpdf(x) {
    const t = this.dists.map((d, i) => {
      const v = d.logpdf(x);
      return Number.isNaN(v) ? -Infinity : this.logW[i] + v;   // a NaN component contributes nothing
    });
    const m = Math.max(...t);
    if (m === Infinity) return Infinity;
    if (m === -Infinity) return -Infinity;
    return m + Math.log(t.reduce((a, v) => a + Math.exp(v - m), 0.0));
  }

  pdf(x) { return Math.exp(this.logpdf(x)); }

  cdf(x) { return this.dists.reduce((a, d, i) => a + this.w[i] * d.cdf(x), 0.0); }

  get mean() { return this.dists.reduce((a, d, i) => a + this.w[i] * d.mean, 0.0); }

  get var() {
    const mu = this.mean;
    return this.dists.reduce((a, d, i) => a + this.w[i] * (d.var + (d.mean - mu) ** 2), 0.0);
  }

  get std() { return Math.sqrt(Math.max(this.var, 0.0)); }

  quantile(p, tol = 1e-9, maxIter = 200) {
    if (!(p > 0 && p < 1)) throw new Error(`quantile needs 0 < p < 1, got ${p}`);
    const qs = this.dists.map((d) => d.quantile(p));
    let lo = Math.min(...qs);
    let hi = Math.max(...qs);
    // absolute tol at scale >= 1, relative below it so small-scale series keep their digits
    tol *= Math.min(1.0, Math.max(Math.abs(lo), Math.abs(hi)));
    if (hi - lo <= tol) return 0.5 * (lo + hi);
    for (let i = 0; i < maxIter; i++) {
      const mid = 0.5 * (lo + hi);
      if (this.cdf(mid) < p) lo = mid; else hi = mid;
      if (hi - lo <= tol) break;
    }
    return 0.5 * (lo + hi);
  }

  crps(x, n = 65) {
    const q = [];
    for (let i = 0; i < n; i++) q.push(this.quantile((i + 0.5) / n));
    const t1 = q.reduce((a, v) => a + Math.abs(v - x), 0.0) / n;
    const t2 = 2.0 * q.reduce((a, v, i) => a + v * (2.0 * (i + 0.5) / n - 1.0), 0.0) / n;
    return t1 - 0.5 * t2;
  }
}

// granger(params) -> f(y, state) -> [[MixtureDist], state]; horizon 1.
export function granger(params, k = 1) {
  if (k !== 1) throw new Error("granger is trained at horizon 1 only");
  const P = params;
  validateGrangerParams(P);
  const logPrior = [Math.log(P.pi_lap), Math.log(1.0 - P.pi_lap)];
  const ref = laplace(k);
  const current = { offset: null };   // the gate's offset for the current call only
  const ex = expert(P, k, () => current.offset);
  const n = P.candidates.length;

  function f(y, state) {
    if (state === null || state === undefined) {
      state = {
        lap: null, learned: null, pend: [],
        cum: [[[0.0, 0.0]]],
        t: 0, z: new Array(k).fill(null), repeat_frac: 0.0, abs_z_mean: 0.0,
        n: 0, last_y: null,
      };
    }
    const yv = y === null || y === undefined ? NaN : Number(y);
    const finite = Number.isFinite(yv);
    const t = state.t;
    if (finite && state.pend.length > 0) {
      const [lapD, lrnD] = state.pend[state.pend.length - 1];
      const c = state.cum[0][0];
      [lapD[0], lrnD[0]].forEach((d, j) => {
        c[j] += score(d.logpdf(yv));
      });
      // re-centre: the weights depend only on the difference
      const mx = Math.max(c[0], c[1]);
      c[0] -= mx;
      c[1] -= mx;
    }
    current.offset = gate(P.gate, features(state, n, k));
    let lrn;
    try {
      [lrn, state.learned] = ex(yv, state.learned);
    } finally {
      current.offset = null;
    }
    let lap;
    [lap, state.lap] = ref(yv, state.lap);
    state.pend = [[lap, lrn]];
    const pz = state.learned && state.learned.z ? state.learned.z : null;
    state.z = pz !== null ? pz.slice() : new Array(k).fill(null);
    if (finite) {
      const nn = state.n + 1;
      const rep = state.last_y !== null && yv === state.last_y ? 1.0 : 0.0;
      state.repeat_frac += (rep - state.repeat_frac) / nn;
      const z0 = state.z.length ? state.z[0] : null;
      if (z0 !== null && z0 !== undefined && Number.isFinite(z0)) {
        state.abs_z_mean += (Math.min(Math.abs(z0), 8.0) - state.abs_z_mean) / nn;
      }
      state.n = nn;
      state.last_y = yv;
    }
    const c = state.cum[0][0];
    const out = [new MixtureDist([lap[0], lrn[0]], [logPrior[0] + c[0], logPrior[1] + c[1]])];
    state.t = t + 1;
    return [out, state];
  }
  f.skaterName = `granger(k=${k})`;
  return f;
}

// Node only: read a shipped parameter file ("monthly", "weekly", "hourly").
export async function loadGrangerParams(frequency = "monthly") {
  const { readFile } = await import("node:fs/promises");
  const url = new URL(`./granger_params/${frequency}.json`, import.meta.url);
  return JSON.parse(await readFile(url, "utf8"));
}
