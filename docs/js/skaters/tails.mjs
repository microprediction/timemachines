// GPD tail splice — JS port of skaters/tails.py (the conditional tail fit).
//
// The body defines a predictable tail region (its own matured PIT pushed
// through the standard normal, beyond thresholds frozen at warm-up
// quantiles); a trailing tail model fits a generalized Pareto to the
// exceedances per side by censored ML (Grimshaw profile over a fixed tau
// grid); every issued predictive is spliced: body density in the interior
// (rescaled so mass is exact), GPD beyond. Information flows
// body -> region -> tail -> output, never backwards.
//
// Constants and operation order mirror tails.py for 1e-6 parity.

import { erf, Dist, registerDistDecoder, registerDistClass } from "./dist.mjs";

const EPS = 1e-12;
const LOG_SQRT2PI = 0.5 * Math.log(2.0 * Math.PI);
const REFIT_EVERY = 25;
const SQRT2 = Math.sqrt(2.0);

function phi(z) {
  return 0.5 * (1.0 + erf(z / SQRT2));
}

function phiLogpdf(z) {
  return -0.5 * z * z - LOG_SQRT2PI;
}

const ACK_A = [-3.969683028665376e+01, 2.209460984245205e+02,
               -2.759285104469687e+02, 1.383577518672690e+02,
               -3.066479806614716e+01, 2.506628277459239e+00];
const ACK_B = [-5.447609879822406e+01, 1.615858368580409e+02,
               -1.556989798598866e+02, 6.680131188771972e+01,
               -1.328068155288572e+01];
const ACK_C = [-7.784894002430293e-03, -3.223964580411365e-01,
               -2.400758277161838e+00, -2.549732539343734e+00,
               4.374664141464968e+00, 2.938163982698783e+00];
const ACK_D = [7.784695709041462e-03, 3.224671290700398e-01,
               2.445134137142996e+00, 3.754408661907416e+00];

export function phiInv(p) {
  // Acklam's rational approximation, polished with one Halley step.
  p = Math.min(Math.max(p, EPS), 1.0 - EPS);
  let x;
  if (p < 0.02425) {
    const q = Math.sqrt(-2.0 * Math.log(p));
    x = ((((((ACK_C[0] * q + ACK_C[1]) * q + ACK_C[2]) * q
            + ACK_C[3]) * q + ACK_C[4]) * q + ACK_C[5])
         / ((((ACK_D[0] * q + ACK_D[1]) * q + ACK_D[2]) * q
             + ACK_D[3]) * q + 1.0));
  } else if (p <= 0.97575) {
    const q = p - 0.5;
    const r = q * q;
    x = ((((((ACK_A[0] * r + ACK_A[1]) * r + ACK_A[2]) * r
            + ACK_A[3]) * r + ACK_A[4]) * r + ACK_A[5]) * q
         / (((((ACK_B[0] * r + ACK_B[1]) * r + ACK_B[2]) * r
              + ACK_B[3]) * r + ACK_B[4]) * r + 1.0));
  } else {
    const q = Math.sqrt(-2.0 * Math.log(1.0 - p));
    x = -((((((ACK_C[0] * q + ACK_C[1]) * q + ACK_C[2]) * q
             + ACK_C[3]) * q + ACK_C[4]) * q + ACK_C[5])
          / ((((ACK_D[0] * q + ACK_D[1]) * q + ACK_D[2]) * q
              + ACK_D[3]) * q + 1.0));
  }
  const e = phi(x) - p;
  const u = e * Math.sqrt(2.0 * Math.PI) * Math.exp(0.5 * x * x);
  return x - u / (1.0 + 0.5 * x * u);
}

function gpdLogpdf(e, gamma, sigma) {
  if (Math.abs(gamma) < 1e-9) return -Math.log(sigma) - e / sigma;
  const arg = 1.0 + gamma * e / sigma;
  if (arg <= 0.0) return -745.0;
  return -Math.log(sigma) - (1.0 / gamma + 1.0) * Math.log(arg);
}

function gpdSf(e, gamma, sigma) {
  if (e <= 0.0) return 1.0;
  if (Math.abs(gamma) < 1e-9) return Math.exp(-e / sigma);
  const arg = 1.0 + gamma * e / sigma;
  if (arg <= 0.0) return 0.0;
  return Math.pow(arg, -1.0 / gamma);
}

function gpdIsf(p, gamma, sigma) {
  p = Math.min(Math.max(p, 1e-300), 1.0);
  if (Math.abs(gamma) < 1e-9) return -sigma * Math.log(p);
  return sigma / gamma * (Math.pow(p, -gamma) - 1.0);
}

const TAU_GRID = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0,
                  5.0, 8.0];

function fitMl(exc, s1) {
  // Censored-ML GPD fit (Grimshaw profile over a fixed tau grid).
  const n = exc.length;
  const emean = s1 / n;
  if (n < 20 || emean <= 0.0) return [0.0, Math.max(emean, 1e-12)];
  const emax = Math.max(...exc);
  let bestG = 0.0, bestS = Math.max(emean, 1e-12), bestLl = -1e300;
  const taus = TAU_GRID.map((t) => t / emean);
  taus.push(-0.5 / emax, -0.25 / emax, -0.1 / emax);
  for (const tau of taus) {
    if (tau <= -1.0 / emax || Math.abs(tau) < 1e-12) continue;
    let g = 0.0;
    for (const e of exc) g += Math.log1p(tau * e);
    g /= n;
    if (g <= 1e-9) continue;
    const sigma = g / tau;
    if (sigma <= 0.0) continue;
    const ll = -n * Math.log(sigma) - (1.0 + 1.0 / g) * n * g;
    if (ll > bestLl) { bestLl = ll; bestG = g; bestS = sigma; }
  }
  return [bestG, bestS];
}

const GRID_N = 65;

export class SplicedDist {
  // A body Dist with GPD tails spliced in beyond frozen z-thresholds.
  constructor(body, tLo, tUp, zetaLo, zetaUp, gLo, sLo, gUp, sUp) {
    this.body = body;
    this.tLo = tLo; this.tUp = tUp;
    this.zetaLo = zetaLo; this.zetaUp = zetaUp;
    this.gLo = gLo; this.sLo = sLo;
    this.gUp = gUp; this.sUp = sUp;
    this._plo = phi(tLo); this._pup = phi(tUp);
    const interior = Math.max(this._pup - this._plo, 1e-12);
    this._c = Math.max(1.0 - zetaLo - zetaUp, 1e-12) / interior;
    this._grid = null;
  }

  _z(x) {
    const u = Math.min(Math.max(this.body.cdf(x), EPS), 1.0 - EPS);
    return phiInv(u);
  }

  cdf(x) {
    const z = this._z(x);
    if (z < this.tLo) return this.zetaLo * gpdSf(this.tLo - z, this.gLo, this.sLo);
    if (z > this.tUp) return 1.0 - this.zetaUp * gpdSf(z - this.tUp, this.gUp, this.sUp);
    return this.zetaLo + this._c * (phi(z) - this._plo);
  }

  logpdf(x) {
    const base = this.body.logpdf(x);
    if (!Number.isFinite(base)) return base;
    const z = this._z(x);
    let corr;
    if (z < this.tLo) {
      corr = Math.log(Math.max(this.zetaLo, 1e-300))
           + gpdLogpdf(this.tLo - z, this.gLo, this.sLo) - phiLogpdf(z);
    } else if (z > this.tUp) {
      corr = Math.log(Math.max(this.zetaUp, 1e-300))
           + gpdLogpdf(z - this.tUp, this.gUp, this.sUp) - phiLogpdf(z);
    } else {
      corr = Math.log(this._c);
    }
    return base + corr;
  }

  pdf(x) {
    const lp = this.logpdf(x);
    return lp < 700.0 ? Math.exp(lp) : Infinity;
  }

  quantile(p, tol = 1e-9, maxIter = 100) {
    if (!(p > 0 && p < 1)) throw new Error("quantile p must be in (0, 1)");
    let z;
    if (p < this.zetaLo) {
      z = this.tLo - gpdIsf(p / this.zetaLo, this.gLo, this.sLo);
    } else if (p > 1.0 - this.zetaUp) {
      z = this.tUp + gpdIsf((1.0 - p) / this.zetaUp, this.gUp, this.sUp);
    } else {
      const u = this._plo + (p - this.zetaLo) / this._c;
      z = phiInv(Math.min(Math.max(u, EPS), 1.0 - EPS));
    }
    const ub = Math.min(Math.max(phi(z), EPS), 1.0 - EPS);
    return this.body.quantile(ub, tol, maxIter);
  }

  _qgrid() {
    if (this._grid === null) {          // one grid serves mean/var/crps
      const out = new Array(GRID_N);
      for (let i = 0; i < GRID_N; i++) out[i] = this.quantile((i + 0.5) / GRID_N);
      this._grid = out;
    }
    return this._grid;
  }

  get mean() {
    const q = this._qgrid();
    return q.reduce((a, b) => a + b, 0.0) / q.length;
  }

  get var() {
    const q = this._qgrid();
    const m = q.reduce((a, b) => a + b, 0.0) / q.length;
    return q.reduce((a, x) => a + (x - m) * (x - m), 0.0) / q.length;
  }

  get std() {
    return Math.sqrt(this.var);
  }

  crps(x) {
    // CRPS = E|X - x| - 0.5 E|X - X'| via the quantile representation.
    const q = this._qgrid();
    const n = q.length;
    let t1 = 0.0;
    for (const v of q) t1 += Math.abs(v - x);
    t1 /= n;
    let t2 = 0.0;
    for (let i = 0; i < n; i++) t2 += q[i] * (2.0 * (i + 0.5) / n - 1.0);
    t2 = 2.0 * t2 / n;
    return t1 - 0.5 * t2;
  }

  static fromDict(d) {
    return new SplicedDist(Dist.fromDict(d.body), d.t_lo, d.t_up,
                           d.zeta_lo, d.zeta_up, d.g_lo, d.s_lo, d.g_up, d.s_up);
  }

  toDict() {
    return { spliced: true, body: this.body.toDict(),
             t_lo: this.tLo, t_up: this.tUp,
             zeta_lo: this.zetaLo, zeta_up: this.zetaUp,
             g_lo: this.gLo, s_lo: this.sLo,
             g_up: this.gUp, s_up: this.sUp };
  }
}

const GUARD_SF = 1e-3;   // winsorize intake beyond the fitted 1-in-1000 excess
const ADAPT_AFTER = 10;  // consecutive guarded ticks => changepoint, stop capping

function tailNew() {
  return { t: null, exc: [], s1: 0.0, nx: 0, r: 0.0, g: 0.0, s: 1.0, since: 0, run: 0 };
}

function tailAdd(tail, e, nexc) {
  // Contamination guard with changepoint escape (mirrors tails.py).
  if (tail.exc.length >= 20) {
    const cap = gpdIsf(GUARD_SF, tail.g, tail.s);
    if (e > cap) {
      if (tail.run < ADAPT_AFTER) e = cap;
      tail.run += 1;                 // stays escaped while it lasts
    } else {
      tail.run = 0;
    }
  }
  tail.exc.push(e);
  tail.s1 += e;
  tail.nx += 1;
  if (tail.exc.length > nexc) tail.s1 -= tail.exc.shift();
  tail.since += 1;
  // refit every add while the window is small (cheap), then every 25th
  if (tail.since >= REFIT_EVERY || tail.exc.length <= 25) {
    const [g, s] = fitMl(tail.exc, tail.s1);
    tail.g = g; tail.s = s;
    tail.since = 0;
  }
}

export function gpdtails(base, k, level = 0.98, nexc = 500, warmup = 500, rateAlpha = 0.002) {
  // Wrap a k-horizon skater so every issued predictive carries GPD tails.
  return function skater(y, state) {
    if (state === null || state === undefined) {
      const tails = [];
      for (let i = 0; i < k; i++) tails.push({ up: tailNew(), lo: tailNew(), warm: [], n: 0 });
      state = { base: null, pending: [], tails };
    }
    const pend = state.pending;
    const n = pend.length;
    for (let m = 1; m <= k; m++) {
      if (m > n) continue;
      const d = pend[n - m][m - 1];
      const u = d.cdf(y);
      if (!Number.isFinite(u)) continue;
      const z = phiInv(Math.min(Math.max(u, EPS), 1.0 - EPS));
      const th = state.tails[m - 1];
      const up = th.up, lo = th.lo;
      if (up.t === null) {
        th.warm.push(z);
        if (th.warm.length >= warmup) {
          const w = th.warm.slice().sort((a, b) => a - b);
          const iu = Math.min(Math.floor(level * w.length), w.length - 1);
          up.t = w[iu];
          lo.t = w[w.length - 1 - iu];
          for (const x of w) {
            if (x > up.t) tailAdd(up, x - up.t, nexc);
            else if (x < lo.t) tailAdd(lo, lo.t - x, nexc);
          }
          th.n = w.length;
          up.r = up.nx / w.length;   // seed the EWMA rate
          lo.r = lo.nx / w.length;
          th.warm = [];
        }
      } else {
        th.n += 1;
        // the exceedance rate forgets, like everything else
        up.r += rateAlpha * ((z > up.t ? 1.0 : 0.0) - up.r);
        lo.r += rateAlpha * ((z < lo.t ? 1.0 : 0.0) - lo.r);
        if (z > up.t) tailAdd(up, z - up.t, nexc);
        else if (z < lo.t) tailAdd(lo, lo.t - z, nexc);
      }
    }

    const [dists, st] = base(y, state.base);
    state.base = st;
    pend.push(dists.slice());
    if (pend.length > k) pend.shift();

    const out = [];
    for (let m = 1; m <= dists.length; m++) {
      const d = dists[m - 1];
      const th = state.tails[m - 1];
      const up = th.up, lo = th.lo;
      if (up.t === null || up.exc.length < 8 || lo.exc.length < 8 || th.n <= 0) {
        out.push(d);
        continue;
      }
      out.push(new SplicedDist(d, lo.t, up.t, lo.r, up.r,
                               lo.g, lo.s, up.g, up.s));
    }
    return [out, state];
  };
}

// Teach Dist.fromDict and the state walker about SplicedDist. Exported as a
// function, not left as a bare module side effect: the package declares
// sideEffects:false, so a bundler drops `import "./tails.mjs"` written purely
// for its registration (esbuild warns ignored-bare-import and the spliced
// decoder silently goes missing). Callers that need the decoder without
// running a skater import this by name and call it; state.mjs does. Idempotent.
export function registerSplicedDist() {
  registerDistDecoder("spliced", (d) => SplicedDist.fromDict(d));
  registerDistClass(SplicedDist);
}

registerSplicedDist();
