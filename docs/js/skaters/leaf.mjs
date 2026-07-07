// Residual distribution estimator (the leaf) — JS port of skaters/leaf.py.

import { fsum, Dist, erf } from "./dist.mjs";
import { runningVarInit, runningVarUpdate, runningVarGet } from "./runstats.mjs";

// A skater is a function (y, state) -> [dists, state].
// state is null on the first call.

export function leaf(k = 1) {
  function _leaf(y, state) {
    if (state === null || state === undefined) {
      state = { var: runningVarInit() };
    }
    state.var = runningVarUpdate(state.var, y);
    const [, varr] = runningVarGet(state.var);

    let std;
    if (Number.isFinite(varr) && varr > 0) {
      std = Math.sqrt(varr);
    } else {
      // Bootstrap: use |y| as a rough scale until we have data.
      std = Math.max(Math.abs(y), 1e-8);
    }

    const d = Dist.gaussian(0.0, std);
    const dists = new Array(k).fill(d);
    return [dists, state];
  }
  _leaf.skaterName = `leaf(k=${k})`;
  return _leaf;
}

const SCALE_BASIS = [0.7, 1.0, 1.6, 3.0, 6.0];

// Fixed Gaussian scale mixture, online-EM weights + EWMA scale — the
// "discrepancy from N(0,1)" leaf. JS port of scale_mixture_leaf.
export function scaleMixtureLeaf(k = 1, gamma = 0.02, scaleAlpha = 0.01, scales = SCALE_BASIS) {
  const C = scales.slice();
  const K = C.length;
  let oneIdx = 0;
  let best = Infinity;
  for (let i = 0; i < K; i++) {
    const dd = Math.abs(C[i] - 1.0);
    if (dd < best) { best = dd; oneIdx = i; }
  }

  function _leaf(y, state) {
    if (state === null || state === undefined) {
      const w = new Array(K).fill(1e-6);
      w[oneIdx] = 1.0;
      state = { v: 0.0, w, n: 0 };
    }
    state.n += 1;
    const a = scaleAlpha > 1.0 / state.n ? scaleAlpha : 1.0 / state.n;
    state.v = (1 - a) * state.v + a * y * y;
    const varr = state.v;
    const sigma = Number.isFinite(varr) && varr > 0 ? Math.sqrt(varr) : Math.max(Math.abs(y), 1e-8);
    const z = y / sigma;
    const w = state.w;
    const dens = new Array(K);
    for (let i = 0; i < K; i++) {
      dens[i] = w[i] * Math.exp(-0.5 * z * z / (C[i] * C[i])) / C[i];
    }
    const total = fsum(dens);
    if (total > 0) {
      const g = gamma > 1.0 / state.n ? gamma : 1.0 / state.n;
      state.w = w.map((wi, i) => (1 - g) * wi + g * dens[i] / total);
    }
    const comps = C.map((c, i) => [state.w[i], 0.0, c * sigma]);
    const d = new Dist(comps);
    return [new Array(k).fill(d), state];
  }
  _leaf.skaterName = `scale_mixture_leaf(k=${k})`;
  return _leaf;
}

// CRPS leaf — JS port of skaters/leaf.py:crps_leaf. Same scale-mixture form,
// weights fit by online CRPS-gradient (exponentiated gradient on the simplex).
const S2 = Math.sqrt(2.0);
const INV = 1.0 / Math.sqrt(2.0 * Math.PI);
const A0 = 2.0 * INV;
const FINE = [0.4, 0.512, 0.6554, 0.8389, 1.0737, 1.3744, 1.7592, 2.2518, 2.8823, 3.6893, 4.7224, 6.0446, 7.7371, 9.9035, 12.6765];

function phiStd(x) { return Math.exp(-0.5 * x * x) * INV; }
function PhiStd(x) { return 0.5 * (1.0 + erf(x / S2)); }
function absNormal(m, s) {
  if (s <= 0) return Math.abs(m);
  const z = m / s;
  return m * (2.0 * PhiStd(z) - 1.0) + 2.0 * s * phiStd(z);
}

export function crpsLeaf(k = 1, eta = 1.0, scaleAlpha = 0.01, scales = FINE) {
  const C = scales.slice();
  const K = C.length;
  const B = C.map((ca) => C.map((cb) => Math.sqrt(ca * ca + cb * cb) * A0));
  let oneIdx = 0, best = Infinity;
  for (let i = 0; i < K; i++) { const d = Math.abs(C[i] - 1.0); if (d < best) { best = d; oneIdx = i; } }

  function _leaf(y, state) {
    if (state === null || state === undefined) {
      const w = new Array(K).fill(1e-6); w[oneIdx] = 1.0;
      state = { v: 0.0, w, n: 0 };
    }
    state.n += 1;
    const a = scaleAlpha > 1.0 / state.n ? scaleAlpha : 1.0 / state.n;
    state.v = (1 - a) * state.v + a * y * y;
    const sig = Number.isFinite(state.v) && state.v > 0 ? Math.sqrt(state.v) : Math.max(Math.abs(y), 1e-8);
    const z = y / sig;
    const w = state.w;
    const g = new Array(K);
    for (let c = 0; c < K; c++) {
      let dot = 0.0;
      dot = fsum(w.map((wj, j) => wj * B[c][j]));
      g[c] = absNormal(-z, C[c]) - dot;
    }
    const gm = fsum(g) / K;
    // Exponentiated-gradient step, stabilized by subtracting the max exponent so
    // exp() cannot overflow to Inf (which would normalize to NaN weights and
    // crash Dist). Subtracting a constant from every exponent leaves the
    // normalized weights unchanged.
    const e = new Array(K);
    for (let c = 0; c < K; c++) e[c] = -eta * (g[c] - gm);
    let emax = -Infinity;
    for (let c = 0; c < K; c++) if (e[c] > emax) emax = e[c];
    const nw = new Array(K);
    for (let c = 0; c < K; c++) nw[c] = w[c] * Math.exp(e[c] - emax);
    let Z = fsum(nw);
    if (!(Z > 0.0) || !Number.isFinite(Z)) {
      for (let c = 0; c < K; c++) nw[c] = w[c];  // degenerate update: keep weights
      Z = fsum(nw);
    }
    state.w = nw.map((x) => x / Z);
    const comps = C.map((c, i) => [state.w[i], 0.0, c * sig]);
    return [new Array(k).fill(new Dist(comps)), state];
  }
  _leaf.skaterName = `crps_leaf(k=${k}, eta=${eta})`;
  return _leaf;
}


// GARCH(1,1)-t leaf — JS port of skaters/leaf.garch_leaf. Genuine GARCH
// conditional variance (variance-targeted QMLE refit over a fixed grid) + the
// same Gaussian-scale-mixture (Student-t) tails.
const GARCH_AB_GRID = [];
for (const a of [0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.20]) {
  for (const b of [0.72, 0.78, 0.84, 0.88, 0.92, 0.95, 0.97]) {
    if (a + b < 0.999) GARCH_AB_GRID.push([a, b]);
  }
}
const GARCH_OMEGA_MULT = [0.5, 0.7, 1.0, 1.4, 2.0];

export function garchLeaf(k = 1, gamma = 0.02, refitEvery = 40, minObs = 80,
                          window = 400, scales = SCALE_BASIS) {
  const C = scales.slice();
  const K = C.length;
  let oneIdx = 0, best = Infinity;
  for (let i = 0; i < K; i++) { const dd = Math.abs(C[i] - 1.0); if (dd < best) { best = dd; oneIdx = i; } }

  function _leaf(y, state) {
    if (state === null || state === undefined) {
      const w = new Array(K).fill(1e-6); w[oneIdx] = 1.0;
      state = { h: 0.0, s2: 0.0, n: 0, omega: 0.0, alpha: 0.05, beta: 0.90, buf: [], w, last_r2: 0.0 };
    }
    const s = state;
    s.n += 1;
    const a0 = 0.02 > 1.0 / s.n ? 0.02 : 1.0 / s.n;
    s.s2 = (1 - a0) * s.s2 + a0 * y * y;
    if (s.s2 <= 0) s.s2 = Math.max(y * y, 1e-12);

    let h;
    if (s.n === 1) h = s.s2;
    else h = s.omega + s.alpha * s.last_r2 + s.beta * s.h;
    if (h <= 1e-300) h = s.s2;
    s.h = h;
    s.last_r2 = y * y;
    s.buf.push(y);
    if (s.buf.length > window) s.buf.shift();

    if (s.n >= minObs && s.n % refitEvery === 0 && s.buf.length >= minObs) {
      const resid = s.buf;
      const s2 = fsum(resid.map((r) => r * r)) / resid.length;
      if (s2 > 0) {
        // grid over (alpha, beta) AND a free omega multiplier (issue #25)
        let bestV = Infinity, bom = s.omega, bal = s.alpha, bbe = s.beta;
        for (let gi = 0; gi < GARCH_AB_GRID.length; gi++) {
          const al = GARCH_AB_GRID[gi][0], be = GARCH_AB_GRID[gi][1];
          const base = (1.0 - al - be) * s2;
          for (let ci = 0; ci < GARCH_OMEGA_MULT.length; ci++) {
            let om = base * GARCH_OMEGA_MULT[ci];
            if (om <= 1e-12) om = 1e-12;
            let hh = om / (1.0 - al - be);          // unconditional-variance init
            let v = 0.0;
            for (let i = 0; i < resid.length; i++) {
              const r = resid[i];
              hh = om + al * (r * r) + be * hh;
              if (hh <= 1e-300) hh = 1e-300;
              v += Math.log(hh) + (r * r) / hh;
            }
            if (v < bestV) { bestV = v; bom = om; bal = al; bbe = be; }   // first-wins on tie
          }
        }
        s.alpha = bal; s.beta = bbe; s.omega = bom;
      }
    }

    const sigma = (Number.isFinite(h) && h > 0) ? Math.sqrt(h) : Math.max(Math.abs(y), 1e-8);
    const z = y / sigma;
    const w = s.w;
    const dens = new Array(K);
    let total = 0.0;
    for (let i = 0; i < K; i++) dens[i] = w[i] * Math.exp(-0.5 * z * z / (C[i] * C[i])) / C[i];
    total = fsum(dens);
    if (total > 0) {
      const g = gamma > 1.0 / s.n ? gamma : 1.0 / s.n;
      s.w = w.map((wi, i) => (1 - g) * wi + g * dens[i] / total);
    }
    const comps = C.map((c, i) => [s.w[i], 0.0, c * sigma]);
    return [new Array(k).fill(new Dist(comps)), s];
  }
  _leaf.skaterName = `garch_leaf(k=${k})`;
  return _leaf;
}
