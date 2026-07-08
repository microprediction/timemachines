// Streaming Mahalanobis anomaly head — JS twin of timemachines/heads/mahalanobis.py.
// Consumes any parade-wrapped skater (state.z), emits calibrated p-values.

const EPS = 3e-12;
const ITMAX = 300;

// --- chi-square tail via regularized incomplete gamma ---

function gser(a, x) {
  if (x <= 0) return 0.0;
  let ap = a, total = 1.0 / a, term = total;
  for (let i = 0; i < ITMAX; i++) {
    ap += 1.0;
    term *= x / ap;
    total += term;
    if (Math.abs(term) < Math.abs(total) * EPS) break;
  }
  return total * Math.exp(-x + a * Math.log(x) - lgamma(a));
}

function gcf(a, x) {
  const tiny = 1e-300;
  let b = x + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
  for (let i = 1; i <= ITMAX; i++) {
    const an = -i * (i - a);
    b += 2.0;
    d = an * d + b;
    if (Math.abs(d) < tiny) d = tiny;
    c = b + an / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1.0 / d;
    const de = d * c;
    h *= de;
    if (Math.abs(de - 1.0) < EPS) break;
  }
  return h * Math.exp(-x + a * Math.log(x) - lgamma(a));
}

// Lanczos lgamma (Math.lgamma does not exist in JS)
const LG = [76.18009172947146, -86.50532032941677, 24.01409824083091,
  -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
function lgamma(x) {
  let y = x;
  const tmp0 = x + 5.5 - (x + 0.5) * Math.log(x + 5.5);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) ser += LG[j] / ++y;
  return -tmp0 + Math.log(2.5066282746310005 * ser / x);
}

export function chi2Sf(x, dof) {
  if (x <= 0) return 1.0;
  const a = 0.5 * dof, xx = 0.5 * x;
  return xx < a + 1.0 ? 1.0 - gser(a, xx) : gcf(a, xx);
}

export function chi2Ppf(p, dof, tol = 1e-10) {
  let lo = 0.0, hi = dof + 40.0 * Math.sqrt(2.0 * dof) + 100.0;
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    if (1.0 - chi2Sf(mid, dof) < p) lo = mid; else hi = mid;
    if (hi - lo < tol) break;
  }
  return 0.5 * (lo + hi);
}

function gpdFit(exc) {
  // Probability-weighted moments (Hosking & Wallis 1987): valid for shape < 1;
  // method of moments cannot exceed 1/2 and the excess tails run ~0.7.
  const x = exc.slice().sort((a, b) => a - b);
  const n = x.length;
  let a0 = 0.0, a1 = 0.0;
  for (let i = 0; i < n; i++) { a0 += x[i]; a1 += ((n - 1 - i) / (n - 1)) * x[i]; }
  a0 /= n; a1 /= n;
  const denom = a0 - 2.0 * a1;
  if (denom <= 1e-12) return [0.95, Math.max(a0, 1e-12)];
  const gamma = Math.min(Math.max(2.0 - a0 / denom, -0.5), 0.95);
  return [gamma, Math.max(2.0 * a0 * a1 / denom, 1e-12)];
}

function gpdSf(e, gamma, sigma) {
  if (Math.abs(gamma) < 1e-9) return Math.exp(-Math.min(e / sigma, 700.0));
  const arg = 1.0 + gamma * e / sigma;
  return arg <= 0.0 ? 0.0 : Math.pow(arg, -1.0 / gamma);
}

// --- small dense linear algebra on flat k x k matrices ---

function cholesky(A, n, jitter = 1e-12) {
  const L = new Array(n * n).fill(0.0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * n + j];
      for (let t = 0; t < j; t++) s -= L[i * n + t] * L[j * n + t];
      if (i === j) L[i * n + i] = Math.sqrt(s > jitter ? s : jitter);
      else L[i * n + j] = s / L[j * n + j];
    }
  }
  return L;
}

function mahal2(L, v, n) {
  const w = new Array(n).fill(0.0);
  let d2 = 0.0;
  for (let i = 0; i < n; i++) {
    let s = v[i];
    for (let t = 0; t < i; t++) s -= L[i * n + t] * w[t];
    const wi = s / L[i * n + i];
    w[i] = wi;
    d2 += wi * wi;
  }
  return d2;
}

function topEig(S, n, iters = 60) {
  let v = Array.from({ length: n }, (_, i) => 1.0 + 1e-3 * i);
  let norm = Math.sqrt(v.reduce((a, x) => a + x * x, 0));
  v = v.map((x) => x / norm);
  let lam = 0.0;
  for (let it = 0; it < iters; it++) {
    const w = new Array(n).fill(0.0);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) w[i] += S[i * n + j] * v[j];
    norm = Math.sqrt(w.reduce((a, x) => a + x * x, 0));
    if (norm <= 0) return [0.0, v];
    for (let i = 0; i < n; i++) w[i] /= norm;
    lam = 0.0;
    for (let i = 0; i < n; i++) {
      let si = 0.0;
      for (let j = 0; j < n; j++) si += S[i * n + j] * w[j];
      lam += w[i] * si;
    }
    v = w;
  }
  return [Math.max(lam, 0.0), v];
}

function topFactors(S, n, r) {
  const work = S.slice();
  let meanDiag = 0.0;
  for (let i = 0; i < n; i++) meanDiag += S[i * n + i];
  meanDiag /= n;
  const out = [];
  for (let f = 0; f < r; f++) {
    const [lam, v] = topEig(work, n);
    if (lam <= 0.01 * meanDiag) break;
    out.push([lam, v]);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) work[i * n + j] -= lam * v[i] * v[j];
  }
  return out;
}

function solveSym(A, b, n) {
  const L = cholesky(A, n);
  const y = new Array(n).fill(0.0);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let t = 0; t < i; t++) s -= L[i * n + t] * y[t];
    y[i] = s / L[i * n + i];
  }
  const x = new Array(n).fill(0.0);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let t = i + 1; t < n; t++) s -= L[t * n + i] * x[t];
    x[i] = s / L[i * n + i];
  }
  return x;
}

// --- the head ---

export function mahalanobis(base, k, {
  alpha = 0.02, scatter = "factor", factors = 1, shrink = 0.05,
  dfloor = 1e-3, guardP = 0.99, adaptAfter = 10,
  potLevel = 0.98, minExc = 30,
} = {}) {
  function _skater(y, state) {
    if (state === null || state === undefined) {
      const S = new Array(k * k).fill(0.0);
      for (let i = 0; i < k; i++) S[i * k + i] = 1.0;   // calibrated prior
      state = { base: null, mu: new Array(k).fill(0.0), S,
                m2: k, v2: 2 * k, exc: [], zeta: 1.0 - potLevel,
                pend1: null, nm: 0.0, nv: 1.0, n_exc: [],
                n_zeta: 1.0 - potLevel, n_n: 0,
                run: 0, d2: null, pvalue: null,
                skipped: 0, last_dists: null };
    }
    // Harden the gate: a non-finite tick must not reach the body.
    if (typeof y !== "number" || !Number.isFinite(y)) {
      state.skipped += 1;
      state.d2 = null; state.pvalue = null;
      return [state.last_dists || new Array(k).fill(null), state];
    }
    // Deep-evidence channel: -logpdf under last tick's 1-step predictive.
    // The parade clamps |z| ~7.03 so d2 saturates; nlp is unbounded.
    let nlp = null;
    if (state.pend1 !== null) {
      const lp = state.pend1.logpdf(y);
      nlp = Number.isFinite(lp) ? -lp : 1e6;
    }
    const [dists, bs] = base(y, state.base);
    state.last_dists = dists;
    state.pend1 = dists[0];
    state.base = bs;
    const z = bs && bs.z ? bs.z : null;
    if (!z || z.length !== k) throw new Error("mahalanobis needs a parade-wrapped skater (state.z)");
    if (z.some((v) => v === null || v === undefined)) {
      state.d2 = null; state.pvalue = null;
      return [dists, state];
    }
    const mu = state.mu, S = state.S;

    // Score BEFORE updating — a point must not defend itself.
    const v = z.map((zi, i) => zi - mu[i]);
    let d2;
    if (scatter === "factor") {
      const fac = topFactors(S, k, factors);
      let meanDiag = 0.0;
      for (let i = 0; i < k; i++) meanDiag += S[i * k + i];
      meanDiag /= k;
      const floor = Math.max(dfloor * meanDiag, 1e-12);
      const D = new Array(k);
      for (let i = 0; i < k; i++) {
        let resid = S[i * k + i];
        for (const [lam, w] of fac) resid -= lam * w[i] * w[i];
        D[i] = Math.max(resid, floor);
      }
      let q1 = 0.0;
      for (let i = 0; i < k; i++) q1 += v[i] * v[i] / D[i];
      if (!fac.length) d2 = q1;
      else {
        const r = fac.length;
        const b = fac.map(([, w]) => {
          let s = 0.0;
          for (let i = 0; i < k; i++) s += w[i] * v[i] / D[i];
          return s;
        });
        const B = new Array(r * r).fill(0.0);
        for (let a_ = 0; a_ < r; a_++) {
          B[a_ * r + a_] = 1.0 / fac[a_][0];
          for (let c_ = a_; c_ < r; c_++) {
            let g = 0.0;
            for (let i = 0; i < k; i++) g += fac[a_][1][i] * fac[c_][1][i] / D[i];
            B[a_ * r + c_] += g;
            if (c_ !== a_) B[c_ * r + a_] += g;
          }
        }
        const x = solveSym(B, b, r);
        let corr = 0.0;
        for (let j = 0; j < r; j++) corr += b[j] * x[j];
        d2 = q1 - corr;
      }
    } else {
      const Ssh = new Array(k * k);
      for (let i = 0; i < k; i++)
        for (let j = 0; j < k; j++)
          Ssh[i * k + j] = (1 - shrink) * S[i * k + j] + (i === j ? shrink : 0);
      d2 = mahal2(cholesky(Ssh, k), v, k);
    }

    // Empirical null (two-moment Satterthwaite), seeded at chi2_k moments.
    const m2 = Math.max(state.m2, 1e-9), v2 = Math.max(state.v2, 1e-9);
    const c = Math.max(v2 / (2.0 * m2), 1e-9);
    const nu = Math.min(Math.max(2.0 * m2 * m2 / v2, 0.5), 1000.0);
    state.d2 = d2;
    // Bulk null (Satterthwaite) below the POT threshold; streaming GPD above
    // it — the bulk fit understates tail p by ~an order of magnitude
    // (measured on the calibration panel), and the GPD is authoritative there.
    const tPot = c * chi2Ppf(potLevel, nu);
    const tScale = Math.max(tPot, 1e-9);
    if (d2 > tPot && state.exc.length >= minExc) {
      // Threshold-relative excesses: the null drifts, absolute excesses
      // pooled across its history read as spurious tail weight.
      const [gamma, sigma] = gpdFit(state.exc);
      state.pvalue = Math.min(
        Math.max(state.zeta, 1e-12) * gpdSf((d2 - tPot) / tScale, gamma, sigma), 1.0);
    } else {
      state.pvalue = chi2Sf(d2 / c, nu);
    }
    // nlp channel speaks only in its own POT tail (Bonferroni combine).
    if (nlp !== null && state.n_n >= minExc) {
      const ns = Math.sqrt(Math.max(state.nv, 1e-12));
      const tN = state.nm + 2.33 * ns;
      if (nlp > tN && state.n_exc.length >= minExc) {
        const [g2, s2] = gpdFit(state.n_exc);
        const pN = Math.max(state.n_zeta, 1e-12)
          * gpdSf((nlp - tN) / Math.max(tN - state.nm, 1e-9), g2, s2);
        state.pvalue = Math.min(state.pvalue, 2.0 * pN);
      }
    }

    // Huberised update with a changepoint escape.
    const qGuard = c * chi2Ppf(guardP, nu);
    let w;
    if (d2 > qGuard) {
      state.run += 1;
      w = state.run > adaptAfter ? 1.0 : qGuard / d2;
    } else {
      state.run = 0;
      w = 1.0;
    }
    const a = alpha * w;
    // Null moments by WINSORIZATION (the Huber weight bounds linear influence
    // but not the quadratic variance update).
    const d2n = w === 1.0 ? d2 : Math.min(d2, qGuard);
    const dm = d2n - state.m2;
    state.m2 += alpha * dm;
    state.v2 = (1 - alpha) * state.v2 + alpha * dm * (d2n - state.m2);
    const aw = alpha * w;
    state.zeta = (1 - aw) * state.zeta + aw * (d2 > tPot ? 1.0 : 0.0);
    if (d2 > tPot) {
      state.exc.push(Math.min((d2 - tPot) / tScale, 50.0));
      if (state.exc.length > 250) state.exc.shift();
    }
    if (nlp !== null) {
      state.n_n += 1;
      const ns = Math.sqrt(Math.max(state.nv, 1e-12));
      const tN = state.nm + 2.33 * ns;
      const nw = Math.min(nlp, state.nm + 6.0 * ns);
      const dn = nw - state.nm;
      state.nm += alpha * dn;
      state.nv = (1 - alpha) * state.nv + alpha * dn * (nw - state.nm);
      state.n_zeta = (1 - aw) * state.n_zeta + aw * (nlp > tN ? 1.0 : 0.0);
      if (nlp > tN) {
        state.n_exc.push(Math.min((nlp - tN) / Math.max(tN - state.nm, 1e-9), 50.0));
        if (state.n_exc.length > 250) state.n_exc.shift();
      }
    }
    const delta = v;
    for (let i = 0; i < k; i++) mu[i] += a * delta[i];
    const delta2 = z.map((zi, i) => zi - mu[i]);
    for (let i = 0; i < k; i++)
      for (let j = 0; j < k; j++)
        S[i * k + j] = (1 - a) * S[i * k + j] + a * delta[i] * delta2[j];

    return [dists, state];
  }
  _skater.skaterName = `mahalanobis(k=${k})`;
  return _skater;
}

// --- the named machine ---

export function wald(k = 3, { scaleAlpha = 0.01, detAlpha = 0.005,
                              engine = null, ...rest } = {}, laplaceFn = null) {
  // laplaceFn injection avoids a hard import path; pass skaters' laplace.
  if (!engine && !laplaceFn) throw new Error("wald needs engine or laplaceFn");
  const base = engine || laplaceFn(k, "crps", true, null, scaleAlpha);
  const f = mahalanobis(base, k, { alpha: detAlpha, ...rest });
  f.skaterName = `wald(k=${k})`;
  return f;
}
