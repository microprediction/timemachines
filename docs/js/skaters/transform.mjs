// Invertible online transforms — JS port of skaters/transform.py.
//
// A transform is { forward, inverseK }:
//   [yPrime, state] = forward(y, state)     // scalar in, scalar out
//   dists            = inverseK(dists, state) // map k Dists back
//
// forward runs on every observation; inverseK maps k Dist predictions in
// the transformed space back to the original space.

import { fsum, Dist } from "./dist.mjs";

// --- small helpers ---

function copysign(a, b) {
  const m = Math.abs(a);
  return b < 0 || Object.is(b, -0) ? -m : m;
}

function matVec(M, v, n) {
  const out = new Array(n).fill(0.0);
  for (let i = 0; i < n; i++) {
    let s = 0.0;
    for (let j = 0; j < n; j++) s += M[i * n + j] * v[j];
    out[i] = s;
  }
  return out;
}

function dot(a, b, n) {
  const terms = new Array(n);
  for (let i = 0; i < n; i++) terms[i] = a[i] * b[i];
  return fsum(terms);
}

function eye(n, scale = 1.0) {
  const m = new Array(n * n).fill(0.0);
  for (let i = 0; i < n; i++) m[i * n + i] = scale;
  return m;
}

// ---------------------------------------------------------------------------
// Differencing:  y'_t = y_t - y_{t-1}
// ---------------------------------------------------------------------------

export function difference() {
  function forward(y, state) {
    if (state === null || state === undefined) return [0.0, { last: y }];
    return [y - state.last, { last: y }];
  }
  function inverseK(dists, state) {
    const anchor = state.last;
    const result = [];
    let cumsumMean = 0.0;
    let cumsumVar = 0.0;
    for (const d of dists) {
      cumsumMean += d.mean;
      cumsumVar += d.var;
      const std = cumsumVar > 0 ? Math.sqrt(cumsumVar) : Math.max(d.std, 1e-12);
      result.push(Dist.gaussian(anchor + cumsumMean, std));
    }
    return result;
  }
  return { forward, inverseK };
}

// ---------------------------------------------------------------------------
// Fractional differencing:  (1 - B)^d
// ---------------------------------------------------------------------------

function fracDiffWeights(d, window) {
  const w = [1.0];
  for (let i = 1; i < window; i++) w.push((-w[w.length - 1] * (d - i + 1)) / i);
  return w;
}

export function fractionalDifference(d = 0.4, window = 50) {
  const wFwd = fracDiffWeights(d, window);

  function forward(y, state) {
    if (state === null || state === undefined) state = { buffer: [] };
    const buf = state.buffer;
    buf.push(y);
    if (buf.length > window) buf.shift();
    const n = buf.length;
    const yPrime = fsum(Array.from({ length: n }, (_, j) => wFwd[j] * buf[n - 1 - j]));
    return [yPrime, state];
  }

  function inverseK(dists, state) {
    // Recovered horizons feed back through the same weights, so each carries a row
    // of responses to the future innovations and Var = sum_i c_i^2 var_i (issue #255).
    const H = dists.length;
    const buf = state.buffer.slice();
    const cbuf = buf.map(() => null);
    const result = [];
    for (let h = 0; h < H; h++) {
      const dIn = dists[h];
      buf.push(0.0);
      cbuf.push(null);
      const n = buf.length;
      let shift = 0.0;
      const coef = new Array(H).fill(0.0);
      coef[h] = 1.0;
      const upper = Math.min(n, window);
      for (let j = 1; j < upper; j++) {
        shift -= wFwd[j] * buf[n - 1 - j];
        const cj = cbuf[n - 1 - j];
        if (cj !== null) for (let i = 0; i < h; i++) coef[i] -= wFwd[j] * cj[i];
      }
      const recoveredMean = dIn.mean + shift;
      buf[n - 1] = recoveredMean;
      cbuf[n - 1] = coef;
      let totalVar = 0.0;
      for (let i = 0; i <= h; i++) totalVar += coef[i] * coef[i] * dists[i].var;
      const std = totalVar > 0 ? Math.sqrt(totalVar) : Math.max(dIn.std, 1e-12);
      result.push(Dist.gaussian(recoveredMean, std));
      if (buf.length > window) { buf.shift(); cbuf.shift(); }
    }
    return result;
  }
  return { forward, inverseK };
}

// ---------------------------------------------------------------------------
// Running standardization
// ---------------------------------------------------------------------------

export function standardize(alpha = 0.05, eps = 1e-8) {
  function forward(y, state) {
    if (state === null || state === undefined) return [0.0, { mu: y, var: 0.0 }];
    const mu = state.mu;
    const varr = state.var;
    const diff = y - mu;
    // Emit against the PRIOR state, so the forward map is the affine change
    // of coordinates z = (y - mu) / sigma that the inverse applies. Scaling
    // by the post-update std makes the emission self-normalized (bounded by
    // 1/sqrt(alpha)) and the affine inverse is then not its inverse.
    // Cold start: before the variance is informative, scale by the first
    // nonzero residual, so the first informative emission is +-1.
    let sigma;
    if (varr > eps * eps) {
      sigma = Math.sqrt(varr);
    } else {
      sigma = Math.abs(diff) > eps ? Math.abs(diff) : eps;
    }
    const yPrime = diff / sigma;
    const muNew = mu + alpha * diff;
    const varNew = (1 - alpha) * varr + alpha * diff * diff;
    return [yPrime, { mu: muNew, var: varNew }];
  }
  function inverseK(dists, state) {
    const varr = state.var;
    const sigma = varr > 1e-16 ? Math.sqrt(varr) : 1e-8;
    return dists.map((d) => d.affine(sigma, state.mu));
  }
  return { forward, inverseK };
}

// ---------------------------------------------------------------------------
// EMA as a transform: subtract the running level
// ---------------------------------------------------------------------------

export function emaTransform(alpha = 0.05) {
  function forward(y, state) {
    if (state === null || state === undefined) return [0.0, { level: y }];
    // Residual is the one-step forecast error (against the PRIOR level); using the
    // post-update level would shrink it by (1-alpha) -> overconfident intervals.
    const residual = y - state.level;
    const level = state.level + alpha * residual;
    return [residual, { level }];
  }
  function inverseK(dists, state) {
    const level = state.level;
    const out = [];
    let add = 0.0; // alpha^2 * sum_{i<h} var_i: the level's accumulated response (ETS(A,N,N), issue #254)
    for (const d of dists) {
      if (add > 0.0) out.push(new Dist(d.components.map(([w, m, s]) => [w, m + level, Math.sqrt(s * s + add)])));
      else out.push(d.shift(level));
      add += alpha * alpha * d.var;
    }
    return out;
  }
  return { forward, inverseK };
}

// Ornstein-Uhlenbeck mean-reversion (JS port of transform.ou_transform).
// Reverts toward a running mean at speed kappa; phi = 1 - kappa. Multi-step
// inverse uses the exact OU moments (phi^h mean decay, saturating variance).
export function ouTransform(kappa = 0.1, alpha = 0.02) {
  const phi = 1.0 - kappa;
  function forward(y, state) {
    if (state === null || state === undefined || !Number.isFinite(y)) {
      const y0 = Number.isFinite(y) ? y : 0.0;
      return [0.0, { m: y0, fc: y0, y: y0 }];
    }
    let resid = y - state.fc;
    if (!Number.isFinite(resid)) resid = 0.0;
    const m = state.m + alpha * (y - state.m);
    const fc = m + phi * (y - m);
    return [resid, { m, fc, y }];
  }
  function inverseK(dists, state) {
    const m = state.m;
    const ylast = state.y;
    return dists.map((d, i) => {
      const h = i + 1;
      const center = m + Math.pow(phi, h) * (ylast - m);
      // Var = sum_{k<=i} phi^{2(i-k)} var_k, as a scale on this horizon's Dist (issue #256);
      // equals the old sqrt(sum_j phi^2j) for equal variances.
      const vh = d.var;
      let g;
      if (vh > 0.0) {
        let tot = 0.0;
        for (let k = 0; k <= i; k++) tot += Math.pow(phi, 2 * (i - k)) * dists[k].var;
        g = Math.sqrt(tot / vh);
      } else {
        g = phi < 1.0 - 1e-9
          ? Math.sqrt((1.0 - Math.pow(phi, 2 * h)) / (1.0 - phi * phi))
          : Math.sqrt(h);
      }
      return d.scale(g).shift(center);
    });
  }
  return { forward, inverseK };
}

// ---------------------------------------------------------------------------
// Theta method (Assimakopoulos & Nikolopoulos, 2000)
// ---------------------------------------------------------------------------

export function theta(alpha = 0.1) {
  function forward(y, state) {
    if (state === null || state === undefined) {
      return [0.0, { ses: y, t: 1, sum_t: 1.0, sum_t2: 1.0, sum_y: y, sum_ty: y }];
    }
    const s = state;
    s.t += 1;
    const t = s.t;
    // Forecast from the PRIOR state, then update (forecasting after folding y into
    // ses/slope would leak y into its own residual -> overconfident interval).
    const forecast = s.ses + (s.slope === undefined ? 0.0 : s.slope) / 2;
    const residual = y - forecast;
    s.ses = alpha * y + (1 - alpha) * s.ses;
    s.sum_t += t;
    s.sum_t2 += t * t;
    s.sum_y += y;
    s.sum_ty += t * y;
    const n = t;
    const denom = n * s.sum_t2 - s.sum_t * s.sum_t;
    const slope = Math.abs(denom) > 1e-12 ? (n * s.sum_ty - s.sum_t * s.sum_y) / denom : 0.0;
    s.slope = slope;
    return [residual, s];
  }
  function inverseK(dists, state) {
    const ses = state.ses;
    const slope = state.slope === undefined ? 0.0 : state.slope;
    const result = [];
    for (let h = 0; h < dists.length; h++) {
      const d = dists[h];
      const forecast = ses + ((h + 1) * slope) / 2 + d.mean;
      // SES: c_0 = 1, c_j = alpha; Var = sum_j c_j^2 var_{h-j} (issue #253).
      let totalVar = 0.0;
      for (let j = 0; j <= h; j++) {
        const c = j === 0 ? 1.0 : alpha;
        totalVar += c * c * dists[h - j].var;
      }
      const std = totalVar > 0 ? Math.sqrt(totalVar) : Math.max(d.std, 1e-12);
      result.push(Dist.gaussian(forecast, std));
    }
    return result;
  }
  return { forward, inverseK };
}

// ---------------------------------------------------------------------------
// Random walk with drift
// ---------------------------------------------------------------------------

export function drift(alpha = 0.002, shrinkage = 0.001) {
  const decay = 1 - alpha - shrinkage;
  function forward(y, state) {
    if (state === null || state === undefined) return [0.0, { last: y, mu: 0.0 }];
    const dy = y - state.last;
    // Increment minus the PRIOR drift (one-step forecast error); the post-update
    // drift would leak dy into its own residual -> overconfident interval.
    const residual = dy - state.mu;
    const mu = decay * state.mu + alpha * dy;
    return [residual, { last: y, mu }];
  }
  function inverseK(dists, state) {
    const anchor = state.last;
    const mu = state.mu;
    const rho = decay + alpha; // 1 - shrinkage: persistence of the drift estimate
    const result = [];
    let cumsumMean = 0.0;
    for (let h = 0; h < dists.length; h++) {
      const d = dists[h];
      cumsumMean += d.mean;
      const totalMean = anchor + (h + 1) * mu + cumsumMean;
      // c = 1 + alpha * sum_{m<h-i} rho^m: direct plus through the drift estimate (issue #257).
      let totalVar = 0.0;
      for (let i = 0; i <= h; i++) {
        let acc = 0.0, f = 1.0;
        for (let m = 0; m < h - i; m++) { acc += f; f *= rho; }
        const c = 1.0 + alpha * acc;
        totalVar += c * c * dists[i].var;
      }
      const totalStd = totalVar > 0 ? Math.sqrt(totalVar) : Math.max(d.std, 1e-12);
      result.push(Dist.gaussian(totalMean, totalStd));
    }
    return result;
  }
  return { forward, inverseK };
}

// ---------------------------------------------------------------------------
// Holt linear (level + trend)
// ---------------------------------------------------------------------------

export function holtLinear(alpha = 0.1, beta = 0.05) {
  function forward(y, state) {
    if (state === null || state === undefined) return [0.0, { level: y, trend: 0.0 }];
    const lPrev = state.level;
    const bPrev = state.trend;
    const lNew = alpha * y + (1 - alpha) * (lPrev + bPrev);
    const bNew = beta * (lNew - lPrev) + (1 - beta) * bPrev;
    return [y - (lPrev + bPrev), { level: lNew, trend: bNew }];
  }
  function inverseK(dists, state) {
    const level = state.level;
    const trend = state.trend;
    const result = [];
    for (let h = 0; h < dists.length; h++) {
      const d = dists[h];
      const forecast = level + (h + 1) * trend + d.mean;
      // ETS(A,A,N): c_0 = 1, c_j = alpha + alpha*beta*j; Var = sum_j c_j^2 var_{h-j} (issue #242).
      let totalVar = 0.0;
      for (let j = 0; j <= h; j++) {
        const c = j === 0 ? 1.0 : alpha + alpha * beta * j;
        totalVar += c * c * dists[h - j].var;
      }
      const std = totalVar > 0 ? Math.sqrt(totalVar) : Math.max(d.std, 1e-12);
      result.push(Dist.gaussian(forecast, std));
    }
    return result;
  }
  return { forward, inverseK };
}

// ---------------------------------------------------------------------------
// GARCH(1,1) volatility scaling
// ---------------------------------------------------------------------------

export function garch(omega = 0.01, alpha = 0.1, beta = 0.85, meanAlpha = 0.05, eps = 1e-8) {
  function forward(y, state) {
    if (state === null || state === undefined) {
      const persist = alpha + beta;
      const var0 = persist < 1 ? omega / (1 - persist) : omega / eps;
      return [0.0, { var: var0, last_dev: 0.0, mu: y }];
    }
    // GARCH conditional variance of the DEVIATION from a running mean, not of
    // the raw value: alpha*y^2 treats y as a mean-zero return, so on a level
    // series the "volatility" is of order |y| and the inverse re-inflates it
    // (unstable). Tracking a mean makes garch shift-invariant; on mean-zero
    // returns mu stays ~0 and this reduces to the original y/sigma.
    const mu = state.mu;
    const dev = y - mu;
    const varr = omega + alpha * state.last_dev * state.last_dev + beta * state.var;
    const sigma = varr > eps * eps ? Math.sqrt(varr) : eps;
    return [dev / sigma, { var: varr, last_dev: dev, mu: mu + meanAlpha * dev }];
  }
  function inverseK(dists, state) {
    const sigma = state.var > 1e-16 ? Math.sqrt(state.var) : 1e-8;
    return dists.map((d) => d.affine(sigma, state.mu));
  }
  return { forward, inverseK };
}

// ---------------------------------------------------------------------------
// Seasonal differencing:  y'_t = y_t - y_{t-s}
// ---------------------------------------------------------------------------

export function seasonalDifference(period = 12) {
  function forward(y, state) {
    if (state === null || state === undefined) return [0.0, { buffer: [y] }];
    const buf = state.buffer;
    let yPrime;
    if (buf.length >= period) yPrime = y - buf[buf.length - period];
    else yPrime = 0.0;
    buf.push(y);
    if (buf.length > 2 * period) buf.shift();
    return [yPrime, state];
  }
  function inverseK(dists, state) {
    const buf = state.buffer.slice();
    const recoveredMeans = [];
    const recoveredVars = [];
    const result = [];
    for (let h = 0; h < dists.length; h++) {
      const lagIdx = h - period;
      let anchorMean, anchorVar;
      if (lagIdx < 0) {
        // Observed buffer value: deterministic, only shifts the location.
        const bufIdx = buf.length - period + h;
        anchorMean = bufIdx >= 0 && bufIdx < buf.length ? buf[bufIdx] : 0.0;
        anchorVar = 0.0;
      } else {
        // Value recovered earlier in this call: it carries its own variance,
        // which must be added along the seasonal integration chain (mirrors the
        // difference / grouped_ar inverses). Dropping it understates uncertainty
        // at every horizon h >= period.
        anchorMean = recoveredMeans[lagIdx];
        anchorVar = recoveredVars[lagIdx];
      }
      recoveredMeans.push(dists[h].mean + anchorMean);
      recoveredVars.push(dists[h].var + anchorVar);
      if (anchorVar > 0.0) {
        // Convolve the mixture with N(0, anchorVar): shift means, inflate stds.
        const comps = dists[h].components.map(
          ([w, m, s]) => [w, m + anchorMean, Math.sqrt(s * s + anchorVar)]
        );
        result.push(new Dist(comps));
      } else {
        result.push(dists[h].shift(anchorMean));
      }
    }
    return result;
  }
  return { forward, inverseK };
}

// ---------------------------------------------------------------------------
// Seasonal anchor: hedged seasonal location (phase-EMA blended with naive)
// ---------------------------------------------------------------------------

export function seasonalAnchor(period, alpha = 0.2, weight = 0.5) {
  const anchorOf = (ema, snaive) =>
    ema !== null ? weight * ema + (1.0 - weight) * snaive : snaive;

  function forward(y, state) {
    if (state === null || state === undefined) {
      return [0.0, { ema: new Array(period).fill(null), buffer: [y], n: 1 }];
    }
    const buf = state.buffer;
    const p = state.n % period;
    const snaive = buf.length >= period ? buf[buf.length - period] : buf[buf.length - 1];
    const yPrime = y - anchorOf(state.ema[p], snaive);
    const e = state.ema[p];
    state.ema[p] = e === null ? y : e + alpha * (y - e);
    buf.push(y);
    if (buf.length > 2 * period) buf.shift();
    state.n += 1;
    return [yPrime, state];
  }

  function inverseK(dists, state) {
    const buf = state.buffer;
    const H = dists.length;
    const recoveredMeans = [];
    // Innovation-response rows of recovered values and of each phase's EMA: the
    // forward updates the EMA from recovered values too, so an anchor at h >= period
    // responds to the value one period back through (1 - weight) and weight*alpha (issue #258).
    const recResp = [];
    const emaResp = [];
    for (let q = 0; q < period; q++) emaResp.push(new Array(H).fill(0.0));
    const result = [];
    for (let h = 0; h < H; h++) {
      const p = (state.n + h) % period;
      const lagIdx = h - period;
      let snaive, snResp;
      if (lagIdx < 0) {
        const bufIdx = buf.length - period + h;
        snaive = bufIdx >= 0 && bufIdx < buf.length ? buf[bufIdx] : buf[buf.length - 1];
        snResp = null;
      } else {
        snaive = recoveredMeans[lagIdx];
        snResp = recResp[lagIdx];
      }
      const e = state.ema[p];
      const aMean = anchorOf(e, snaive);
      const aResp = new Array(H).fill(0.0);
      for (let i = 0; i < h; i++) {
        const sn = snResp !== null ? snResp[i] : 0.0;
        aResp[i] = (e !== null && e !== undefined) ? weight * emaResp[p][i] + (1.0 - weight) * sn : sn;
      }
      let aVar = 0.0;
      for (let i = 0; i < h; i++) aVar += aResp[i] * aResp[i] * dists[i].var;
      const r = aResp.slice();
      r[h] = 1.0;
      recoveredMeans.push(dists[h].mean + aMean);
      recResp.push(r);
      if (e === null || e === undefined) emaResp[p] = r.slice();
      else emaResp[p] = emaResp[p].map((v, i) => (1.0 - alpha) * v + alpha * r[i]);
      if (aVar > 0.0) {
        const comps = dists[h].components.map(
          ([w, m, s]) => [w, m + aMean, Math.sqrt(s * s + aVar)]
        );
        result.push(new Dist(comps));
      } else {
        result.push(dists[h].shift(aMean));
      }
    }
    return result;
  }
  return { forward, inverseK };
}

// ---------------------------------------------------------------------------
// Signed power transform
// ---------------------------------------------------------------------------

export function powerTransform(p = 0.5) {
  const invP = 1.0 / p;
  const fwd = (y) => copysign(Math.pow(Math.abs(y), p), y);
  const inv = (yp) => copysign(Math.pow(Math.abs(yp), invP), yp);

  function forward(y, state) {
    return [fwd(y), state || {}];
  }
  function inverseK(dists, state) {
    const result = [];
    for (const d of dists) {
      const components = [];
      for (const [w, mu, sigma] of d.components) {
        const origMean = inv(mu);
        const absMu = Math.abs(mu);
        const deriv = absMu > 1e-12 ? invP * Math.pow(absMu, invP - 1) : invP;
        const origStd = Math.max(sigma * deriv, 1e-12);
        components.push([w, origMean, origStd]);
      }
      result.push(new Dist(components));
    }
    return result;
  }
  return { forward, inverseK };
}

// Yeo-Johnson coordinate transform — signed Box-Cox family. JS port of
// skaters/transform.py:yeo_johnson. Learns the coordinate the series is simple
// in (lambda=0 -> log1p / non-negative, 1 -> identity, 0.5 -> compression).
export function yeoJohnson(lmbda = 0.0) {
  const L = lmbda;
  const fwd = (y) => {
    if (y >= 0.0) return L === 0.0 ? Math.log1p(y) : (Math.pow(y + 1.0, L) - 1.0) / L;
    if (L === 2.0) return -Math.log1p(-y);
    return -((Math.pow(-y + 1.0, 2.0 - L) - 1.0) / (2.0 - L));
  };
  // clamp exp args so squaring the result in var() stays finite (exp(350)~1e152).
  const inv = (yp) => {
    if (yp >= 0.0) {
      if (L === 0.0) return Math.expm1(Math.min(yp, 350.0));
      return Math.pow(Math.max(L * yp + 1.0, 1e-12), 1.0 / L) - 1.0;
    }
    if (L === 2.0) return 1.0 - Math.exp(Math.min(-yp, 350.0));
    return 1.0 - Math.pow(Math.max(-(2.0 - L) * yp + 1.0, 1e-12), 1.0 / (2.0 - L));
  };
  const dinv = (yp) => {
    if (yp >= 0.0) {
      if (L === 0.0) return Math.exp(Math.min(yp, 350.0));
      return Math.pow(Math.max(L * yp + 1.0, 1e-12), 1.0 / L - 1.0);
    }
    if (L === 2.0) return Math.exp(Math.min(-yp, 350.0));
    return Math.pow(Math.max(-(2.0 - L) * yp + 1.0, 1e-12), 1.0 / (2.0 - L) - 1.0);
  };

  function forward(y, state) {
    return [fwd(y), state || {}];
  }
  function inverseK(dists, state) {
    const result = [];
    for (const d of dists) {
      const components = [];
      for (const [w, mu, sigma] of d.components) {
        components.push([w, inv(mu), Math.max(sigma * dinv(mu), 1e-12)]);
      }
      result.push(new Dist(components));
    }
    return result;
  }
  return { forward, inverseK };
}

// ---------------------------------------------------------------------------
// AR(p) with online recursive least squares
// ---------------------------------------------------------------------------

// Spectral radius of the AR companion matrix (largest |characteristic root|).
// The h-step forecast converges iff this is < 1. Closed form for p<=2, power
// iteration otherwise.
function arSpectralRadius(phi) {
  const p = phi.length;
  if (p === 1) return Math.abs(phi[0]);
  if (p === 2) {
    const a = phi[0], b = phi[1];
    const disc = a * a + 4.0 * b;
    if (disc >= 0.0) {
      const r = Math.sqrt(disc);
      return Math.max(Math.abs((a + r) / 2.0), Math.abs((a - r) / 2.0));
    }
    return Math.hypot(a / 2.0, Math.sqrt(-disc) / 2.0);
  }
  // Power iteration from a single seed can silently miss the true dominant
  // eigenvalue if that seed happens to be (near-)orthogonal to it. Two seeds
  // unlikely to BOTH align with a non-dominant eigenvector; take the max.
  let best = 0.0;
  for (const seed of [new Array(p).fill(1.0), Array.from({ length: p }, (_, i) => (i % 2 === 0 ? 1.0 : -1.0))]) {
    let v = seed.slice(), rho = 0.0;
    for (let it = 0; it < 60; it++) {
      let s = 0.0;
      for (let j = 0; j < p; j++) s += phi[j] * v[j];
      const nv = new Array(p);
      nv[0] = s;
      for (let i = 1; i < p; i++) nv[i] = v[i - 1];
      let m = 0.0;
      for (const x of nv) m = Math.max(m, Math.abs(x));
      if (m === 0.0) m = 1.0;
      v = nv.map((x) => x / m);
      rho = m;
    }
    best = Math.max(best, rho);
  }
  return best;
}

// Damp AR coefficients into the stationary region for forecasting. Scaling
// phi_j by gamma^(j+1) scales every companion eigenvalue by gamma; a no-op when
// already stationary. Constrained forecasting, not a magnitude clip.
function arStationary(phi, margin = 0.999) {
  const rho = arSpectralRadius(phi);
  if (rho <= margin) return phi;
  const g = margin / rho;
  return phi.map((c, j) => c * Math.pow(g, j + 1));
}

export function ar(order = 2, lam = 0.99, ridge = 1.0, decay = 0.0) {
  const p = order;

  function initP() {
    const P = new Array(p * p).fill(0.0);
    for (let j = 0; j < p; j++) {
      P[j * p + j] = decay > 0 ? ridge / Math.pow(j + 1, decay) : ridge;
    }
    return P;
  }

  function forward(y, state) {
    if (state === null || state === undefined) {
      state = { buffer: [], phi: new Array(p).fill(0.0), phiUsed: new Array(p).fill(0.0), P: initP(), n: 0 };
    }
    const buf = state.buffer;
    const phi = state.phi;
    if (state.phiUsed === undefined) state.phiUsed = arStationary(phi.slice());   // pre-#235 checkpoint
    state.n += 1;
    let residual;
    if (buf.length >= p) {
      const x = new Array(p);
      for (let i = 0; i < p; i++) x[i] = buf[buf.length - 1 - i];

      // Emitted residual uses `phiUsed` -- the stationarity-projected
      // coefficients as of the end of the previous tick -- not the raw RLS
      // estimate, so forward's emission and inverseK's reconstruction always
      // agree on which predictor produced the residual (skaters#232).
      const phiUsed = state.phiUsed;
      const prediction = fsum(Array.from({ length: p }, (_, i) => phiUsed[i] * x[i]));
      residual = y - prediction;

      // RLS update keeps fitting the RAW (unconstrained) least-squares
      // problem with its own prediction error.
      const rawPrediction = fsum(Array.from({ length: p }, (_, i) => phi[i] * x[i]));
      const rawResidual = y - rawPrediction;
      const P = state.P;
      const Px = matVec(P, x, p);
      const denom = lam + dot(x, Px, p);
      if (Math.abs(denom) > 1e-15) {
        const K = Px.map((px) => px / denom);
        for (let i = 0; i < p; i++) phi[i] += K[i] * rawResidual;
        for (let i = 0; i < p; i++) {
          for (let j = 0; j < p; j++) {
            P[i * p + j] = (P[i * p + j] - K[i] * Px[j]) / lam;
          }
        }
        // Guard against RLS covariance windup (see the Python note): under
        // low-excitation input P inflates by 1/lam each step and overflows to
        // Inf, giving nan coefficients. Reset the covariance (keeping phi) if it
        // grows implausibly large or non-finite. No effect on well-excited data.
        let bad = false, pmax = 0.0;
        for (let m = 0; m < P.length; m++) {
          const v = P[m];
          if (!Number.isFinite(v)) { bad = true; break; }
          const av = Math.abs(v);
          if (av > pmax) pmax = av;
        }
        if (bad || pmax > 1e10) { const np = initP(); for (let m = 0; m < P.length; m++) P[m] = np[m]; }
      }
      // Re-project after every update so forward's next emission and
      // inverseK always agree on the same constrained coefficients.
      state.phiUsed = arStationary(phi.slice());
    } else {
      residual = y;
    }
    buf.push(y);
    if (buf.length > 2 * p + 10) buf.shift();
    return [residual, state];
  }

  function inverseK(dists, state) {
    const buf = state.buffer.slice();
    // Same coefficients forward() already emitted residuals under
    // (skaters#232) -- maintained incrementally there, not re-derived here,
    // except for a pre-#235 checkpoint that has no cache yet.
    const phi = state.phiUsed !== undefined ? state.phiUsed : arStationary(state.phi.slice());
    const H = dists.length;
    // Impulse responses psi_0..psi_{H-1}: psi_0 = 1, psi_i = sum_j phi_j psi_{i-j}.
    const psi = [1.0];
    for (let i = 1; i < H; i++) {
      let s = 0.0;
      for (let j = 0; j < p; j++) if (i - 1 - j >= 0) s += phi[j] * psi[i - 1 - j];
      psi.push(s);
    }
    const recoveredMeans = [];
    const result = [];
    for (let h = 0; h < H; h++) {
      let arMean = 0.0;
      for (let j = 0; j < p; j++) {
        const lagH = h - j - 1;
        if (lagH < 0) {
          const bufIdx = buf.length + lagH;
          if (bufIdx >= 0 && bufIdx < buf.length) arMean += phi[j] * buf[bufIdx];
        } else if (lagH < recoveredMeans.length) {
          arMean += phi[j] * recoveredMeans[lagH];
        }
      }
      const totalMean = dists[h].mean + arMean;
      // Var(y_{t+h+1}) = sum_i psi_{h-i}^2 var_i (issue #245); same order as Python/Rust.
      let totalVar = 0.0;
      for (let i = 0; i <= h; i++) totalVar += psi[h - i] * psi[h - i] * dists[i].var;
      const totalStd = totalVar > 0 ? Math.sqrt(totalVar) : Math.max(dists[h].std, 1e-12);
      recoveredMeans.push(totalMean);
      result.push(Dist.gaussian(totalMean, totalStd));
    }
    return result;
  }
  return { forward, inverseK };
}

// ---------------------------------------------------------------------------
// Grouped AR with geometrically grouped coefficients
// ---------------------------------------------------------------------------

function buildGroups(maxLag) {
  const groups = [];
  let g = 0;
  let size = 1;
  let assigned = 0;
  while (assigned < maxLag) {
    for (let i = 0; i < size; i++) {
      if (assigned >= maxLag) break;
      groups.push(g);
      assigned += 1;
    }
    g += 1;
    size *= 2;
  }
  return groups;
}

function groupRegressor(buf, groups, nGroups, maxLag) {
  const x = new Array(nGroups).fill(0.0);
  for (let j = 0; j < maxLag; j++) x[groups[j]] += buf[buf.length - 1 - j];
  return x;
}

export function groupedAr(maxLag = 16, lam = 0.99, ridge = 1.0) {
  const groups = buildGroups(maxLag);
  const nGroups = Math.max(...groups) + 1;

  function forward(y, state) {
    if (state === null || state === undefined) {
      state = { buffer: [], theta: new Array(nGroups).fill(0.0), phiUsed: new Array(maxLag).fill(0.0), P: eye(nGroups, ridge), n: 0 };
    }
    const buf = state.buffer;
    const th = state.theta;
    if (state.phiUsed === undefined) {   // pre-#235 checkpoint
      state.phiUsed = arStationary(Array.from({ length: maxLag }, (_, j) => th[groups[j]]));
    }
    state.n += 1;
    let residual;
    if (buf.length >= maxLag) {
      const x = groupRegressor(buf, groups, nGroups, maxLag);

      // Emitted residual uses the previous tick's stationarity-projected,
      // per-lag coefficients -- same fix as ar() (skaters#232).
      const phiUsed = state.phiUsed;
      const lagX = new Array(maxLag);
      for (let j = 0; j < maxLag; j++) lagX[j] = buf[buf.length - 1 - j];
      const prediction = fsum(Array.from({ length: maxLag }, (_, j) => phiUsed[j] * lagX[j]));
      residual = y - prediction;

      // RLS update keeps fitting the RAW group-level least-squares problem
      // with its own prediction error.
      const rawPrediction = fsum(Array.from({ length: nGroups }, (_, g) => th[g] * x[g]));
      const rawResidual = y - rawPrediction;
      const P = state.P;
      const Px = matVec(P, x, nGroups);
      const denom = lam + dot(x, Px, nGroups);
      if (Math.abs(denom) > 1e-15) {
        const K = Px.map((px) => px / denom);
        for (let g = 0; g < nGroups; g++) th[g] += K[g] * rawResidual;
        for (let i = 0; i < nGroups; i++) {
          for (let j = 0; j < nGroups; j++) {
            P[i * nGroups + j] = (P[i * nGroups + j] - K[i] * Px[j]) / lam;
          }
        }
        // RLS covariance windup guard (see ar()): reset P if it inflates
        // implausibly or turns non-finite; keeps theta. No effect on well-excited data.
        let bad = false, pmax = 0.0;
        for (let m = 0; m < P.length; m++) {
          const v = P[m];
          if (!Number.isFinite(v)) { bad = true; break; }
          const av = Math.abs(v);
          if (av > pmax) pmax = av;
        }
        if (bad || pmax > 1e10) { const np = eye(nGroups, ridge); for (let m = 0; m < P.length; m++) P[m] = np[m]; }
      }
      const rawPhi = [];
      for (let j = 0; j < maxLag; j++) rawPhi.push(th[groups[j]]);
      state.phiUsed = arStationary(rawPhi);
    } else {
      residual = y;
    }
    buf.push(y);
    if (buf.length > maxLag + 10) buf.shift();
    return [residual, state];
  }

  function inverseK(dists, state) {
    const buf = state.buffer.slice();
    // Same coefficients forward() already emitted residuals under
    // (skaters#232) -- maintained incrementally there, not re-derived here,
    // except for a pre-#235 checkpoint that has no cache yet.
    const phi = state.phiUsed !== undefined
      ? state.phiUsed
      : arStationary(Array.from({ length: maxLag }, (_, j) => state.theta[groups[j]]));
    const H = dists.length;
    const psi = [1.0];
    for (let i = 1; i < H; i++) {
      let s = 0.0;
      for (let j = 0; j < maxLag; j++) if (i - 1 - j >= 0) s += phi[j] * psi[i - 1 - j];
      psi.push(s);
    }
    const recoveredMeans = [];
    const result = [];
    for (let h = 0; h < H; h++) {
      let arMean = 0.0;
      for (let j = 0; j < maxLag; j++) {
        const lagH = h - j - 1;
        if (lagH < 0) {
          const bufIdx = buf.length + lagH;
          if (bufIdx >= 0 && bufIdx < buf.length) arMean += phi[j] * buf[bufIdx];
        } else if (lagH < recoveredMeans.length) {
          arMean += phi[j] * recoveredMeans[lagH];
        }
      }
      const totalMean = dists[h].mean + arMean;
      // Var(y_{t+h+1}) = sum_i psi_{h-i}^2 var_i (issue #245); same order as Python/Rust.
      let totalVar = 0.0;
      for (let i = 0; i <= h; i++) totalVar += psi[h - i] * psi[h - i] * dists[i].var;
      const totalStd = totalVar > 0 ? Math.sqrt(totalVar) : Math.max(dists[h].std, 1e-12);
      recoveredMeans.push(totalMean);
      result.push(Dist.gaussian(totalMean, totalStd));
    }
    return result;
  }
  return { forward, inverseK };
}
