// Gaussian mixture distribution — JS port of skaters/dist.py.
//
// A Dist is a weighted mixture of Gaussians: components = [[w, m, s], ...].
// Weights are positive and normalized to sum to 1; std is always > 0.
// Pure JS, no dependencies. JS lacks Math.erf, so we provide a
// high-accuracy erf (regularized incomplete gamma) to match Python's
// math.erf closely enough for CDF/quantile parity.

const SQRT2 = Math.SQRT2;
const SQRT2PI = Math.sqrt(2.0 * Math.PI);
const LOG_SQRT2PI = 0.5 * Math.log(2.0 * Math.PI);
const LGAMMA_HALF = 0.5723649429247001; // ln(Gamma(1/2)) = ln(sqrt(pi))

// --- erf via the regularized lower incomplete gamma P(1/2, x^2) ---

function _gser(a, x) {
  // Series expansion for P(a, x), valid for x < a + 1.
  let ap = a;
  let sum = 1.0 / a;
  let del = sum;
  for (let i = 0; i < 300; i++) {
    ap += 1.0;
    del *= x / ap;
    sum += del;
    if (Math.abs(del) < Math.abs(sum) * 1e-17) break;
  }
  return sum * Math.exp(-x + a * Math.log(x) - LGAMMA_HALF);
}

function _gcf(a, x) {
  // Continued fraction for Q(a, x) via the modified Lentz algorithm.
  const TINY = 1e-300;
  let b = x + 1.0 - a;
  let c = 1.0 / TINY;
  let d = 1.0 / b;
  let h = d;
  for (let i = 1; i <= 300; i++) {
    const an = -i * (i - a);
    b += 2.0;
    d = an * d + b;
    if (Math.abs(d) < TINY) d = TINY;
    c = b + an / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1.0 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1.0) < 1e-17) break;
  }
  return Math.exp(-x + a * Math.log(x) - LGAMMA_HALF) * h;
}

function _gammp(a, x) {
  if (x <= 0.0) return 0.0;
  if (x < a + 1.0) return _gser(a, x);
  return 1.0 - _gcf(a, x);
}

export function erf(x) {
  if (x === 0.0) return 0.0;
  const r = _gammp(0.5, x * x);
  return x < 0.0 ? -r : r;
}

// --- pure-JS Gaussian math (mirrors dist.py) ---

function gaussianPdf(x, mean, std) {
  if (std <= 0) return x === mean ? Infinity : 0.0;
  const z = (x - mean) / std;
  return Math.exp(-0.5 * z * z) / (std * SQRT2PI);
}

function gaussianCdf(x, mean, std) {
  if (std <= 0) return x >= mean ? 1.0 : 0.0;
  return 0.5 * (1.0 + erf((x - mean) / (std * SQRT2)));
}

function absExpectation(m, s) {
  // E|N(m, s^2)| = m(2Φ(m/s) - 1) + 2s·φ(m/s).
  if (s <= 0) return Math.abs(m);
  const z = m / s;
  return m * (2.0 * gaussianCdf(z, 0.0, 1.0) - 1.0) + 2.0 * s * gaussianPdf(z, 0.0, 1.0);
}

// Neumaier-compensated sum, matching CPython 3.12+'s built-in sum() on floats
// (the Python reference uses sum() for Dist normalisation and moments; the port
// must reproduce it bit-for-bit or pruning tie-breaks diverge).
export function fsum(values) {
  let s = 0.0, c = 0.0;
  for (const x of values) {
    const t = s + x;
    if (Math.abs(s) >= Math.abs(x)) c += (s - t) + x;
    else c += (x - t) + s;
    s = t;
  }
  return s + c;
}

export class Dist {
  // components: array of [weight, mean, std]
  constructor(components) {
    if (!components || components.length === 0) {
      throw new Error("Dist requires at least one component");
    }
    const wTotal = fsum(components.map((c) => c[0]));
    if (!(wTotal > 0)) throw new Error("Dist weights must sum to > 0");
    this.components = components.map(([w, m, s]) => [w / wTotal, m, s]);
  }

  // --- constructors ---

  static gaussian(mean = 0.0, std = 1.0) {
    return new Dist([[1.0, mean, std]]);
  }

  static combine(dists, weights = null) {
    const n = dists.length;
    if (weights === null) weights = new Array(n).fill(1.0 / n);
    const wTotal = fsum(weights);
    const components = [];
    for (let i = 0; i < n; i++) {
      const wOuter = weights[i];
      for (const [wInner, m, s] of dists[i].components) {
        components.push([(wOuter / wTotal) * wInner, m, s]);
      }
    }
    return new Dist(components);
  }

  // --- queries ---

  pdf(x) {
    let total = 0.0;
    for (const [w, m, s] of this.components) total += w * gaussianPdf(x, m, s);
    return total;
  }

  logpdf(x) {
    // Log density accumulated in log-space (log-sum-exp). A Gaussian mixture is
    // strictly positive everywhere, so for finite x this is always finite;
    // log(sum(w * pdf)) would underflow to -Infinity once x sits many standard
    // deviations from every component (e.g. after the scale collapses on a
    // near-constant stream), because each exp(-z*z/2) rounds to 0.
    let best = -Infinity;
    const terms = [];
    for (const [w, m, s] of this.components) {
      if (w <= 0.0) continue;
      if (s <= 0.0) {
        if (x === m) return Infinity;
        continue;
      }
      const z = (x - m) / s;
      const t = Math.log(w) - 0.5 * z * z - Math.log(s) - LOG_SQRT2PI;
      terms.push(t);
      if (t > best) best = t;
    }
    if (best === -Infinity) return -Infinity;
    let acc = 0.0;
    for (const t of terms) acc += Math.exp(t - best);
    return best + Math.log(acc);
  }

  cdf(x) {
    let total = 0.0;
    for (const [w, m, s] of this.components) total += w * gaussianCdf(x, m, s);
    return total;
  }

  crps(x) {
    // Closed-form CRPS for a Gaussian mixture (Grimit et al. 2006).
    const comps = this.components;
    let t1 = 0.0;
    for (const [w, m, s] of comps) t1 += w * absExpectation(m - x, s);
    let t2 = 0.0;
    for (const [wi, mi, si] of comps) {
      for (const [wj, mj, sj] of comps) {
        t2 += wi * wj * absExpectation(mi - mj, Math.sqrt(si * si + sj * sj));
      }
    }
    return t1 - 0.5 * t2;
  }

  quantile(p, tol = 1e-9, maxIter = 100) {
    if (!(p > 0 && p < 1)) throw new Error("quantile p must be in (0, 1)");
    const mu = this.mean;
    const sigma = Math.sqrt(this.var);
    let lo = mu - 8 * sigma;
    let hi = mu + 8 * sigma;
    for (let i = 0; i < maxIter; i++) {
      const mid = 0.5 * (lo + hi);
      if (this.cdf(mid) < p) lo = mid;
      else hi = mid;
      if (hi - lo < tol) break;
    }
    return 0.5 * (lo + hi);
  }

  get mean() {
    return fsum(this.components.map(([w, m]) => w * m));
  }

  get var() {
    // centered form: avoids catastrophic cancellation when means are large
    // relative to spreads (which would otherwise yield var=0, std=0).
    const mu = this.mean;
    return fsum(this.components.map(([w, m, s]) => w * (s * s + (m - mu) * (m - mu))));
  }

  get std() {
    const v = this.var;
    return v > 0 ? Math.sqrt(v) : 0.0;
  }

  // --- transform support ---

  shift(delta) {
    return new Dist(this.components.map(([w, m, s]) => [w, m + delta, s]));
  }

  scale(factor) {
    if (factor === 0) throw new Error("scale factor must be nonzero");
    const f = Math.abs(factor);
    return new Dist(this.components.map(([w, m, s]) => [w, m * factor, s * f]));
  }

  affine(a, b) {
    if (a === 0) throw new Error("affine a must be nonzero");
    return new Dist(this.components.map(([w, m, s]) => [w, a * m + b, Math.abs(a) * s]));
  }

  // Number of mixture components (mirrors Python's __len__).
  get length() {
    return this.components.length;
  }

  // --- pruning ---

  prune(maxComponents = 20) {
    maxComponents = Math.max(1, maxComponents);   // a Dist needs >=1 component
    if (this.components.length <= maxComponents) return this;
    // Sort first so the merge path (and hence the result) is independent of
    // component order — mixtures are order-free but the closest-pair scan
    // is not, and ties at equal means are common (lattice atoms).
    let comps = this.components.map((c) => c.slice())
      .sort((a, b) => (a[1] - b[1]) || (a[2] - b[2]) || (a[0] - b[0]));
    // Pair selection tolerates last-ulp noise in the means: pick the FIRST
    // pair within a hair of the true minimum distance, so platforms that
    // disagree at the ulp level (e.g. libm erf vs a polynomial) still merge
    // the same pairs in the same order. Exact argmin would amplify ulp
    // noise into macroscopically different mixtures.
    const scale = Math.abs(comps[0][1]) + Math.abs(comps[comps.length - 1][1]) + 1e-12;
    while (comps.length > maxComponents) {
      let bestDist = Infinity;
      for (let i = 0; i < comps.length; i++)
        for (let j = i + 1; j < comps.length; j++) {
          const d = Math.abs(comps[i][1] - comps[j][1]);
          if (d < bestDist) bestDist = d;
        }
      const thresh = bestDist + 1e-9 * scale;
      let bestI = null, bestJ = null;
      outer:
      for (let i = 0; i < comps.length; i++)
        for (let j = i + 1; j < comps.length; j++)
          if (Math.abs(comps[i][1] - comps[j][1]) <= thresh) { bestI = i; bestJ = j; break outer; }
      const [wi, mi, si] = comps[bestI];
      const [wj, mj, sj] = comps[bestJ];
      const wNew = wi + wj;
      let mNew, sNew;
      if (wNew < 1e-300) {
        mNew = 0.5 * (mi + mj);
        sNew = Math.max(si, sj, 1e-12);
      } else {
        mNew = (wi * mi + wj * mj) / wNew;
        const vNew =
          (wi * (si * si + (mi - mNew) * (mi - mNew))
            + wj * (sj * sj + (mj - mNew) * (mj - mNew))) / wNew;
        sNew = Math.sqrt(Math.max(vNew, 0.0));
      }
      comps[bestI] = [wNew, mNew, sNew];
      comps.splice(bestJ, 1);
    }
    return new Dist(comps);
  }

  // --- serialization ---

  toDict() {
    return { components: this.components.map((c) => c.slice()) };
  }

  static fromDict(d) {
    return new Dist(d.components.map((c) => c.slice()));
  }

  get length() {
    return this.components.length;
  }
}
