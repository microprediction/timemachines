// Online covariance estimators — JS port of skaters/cov/*.py.
//
// API: [mean, cov, state] = f(y, state)
// y is an array of floats; cov is a flat row-major n*n array.

export function runningCov(y, state) {
  const n = y.length;
  if (state === null || state === undefined) {
    state = { n: 0, mean: new Array(n).fill(0.0), C: new Array(n * n).fill(0.0) };
  }
  state.n += 1;
  const k = state.n;
  const mean = state.mean;
  const C = state.C;

  const delta = new Array(n);
  for (let i = 0; i < n; i++) delta[i] = y[i] - mean[i];
  for (let i = 0; i < n; i++) mean[i] += delta[i] / k;
  const delta2 = new Array(n);
  for (let i = 0; i < n; i++) delta2[i] = y[i] - mean[i];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) C[i * n + j] += delta[i] * delta2[j];
  }

  let cov;
  if (k < 2) cov = new Array(n * n).fill(0.0);
  else cov = C.map((c) => c / (k - 1));
  return [mean.slice(), cov, state];
}

export function emaCov(y, state, alpha = 0.05) {
  const n = y.length;
  if (state === null || state === undefined) {
    state = { mean: y.slice(), cov: new Array(n * n).fill(0.0), n: 1 };
    return [y.slice(), new Array(n * n).fill(0.0), state];
  }
  const mean = state.mean;
  const cov = state.cov;
  state.n += 1;

  const delta = new Array(n);
  for (let i = 0; i < n; i++) delta[i] = y[i] - mean[i];
  for (let i = 0; i < n; i++) mean[i] += alpha * delta[i];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      cov[i * n + j] = (1 - alpha) * (cov[i * n + j] + alpha * delta[i] * delta[j]);
    }
  }
  return [mean.slice(), cov.slice(), state];
}

export function ledoitWolfCov(y, state, alpha = 0.05, shrinkage = 0.5) {
  const n = y.length;
  if (state === null || state === undefined) {
    const corr = new Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) corr[i * n + j] = i === j ? 1.0 : 0.0;
    state = { mean: y.slice(), var: new Array(n).fill(0.0), corr, n: 1 };
    return [y.slice(), new Array(n * n).fill(0.0), state];
  }
  const mean = state.mean;
  const varr = state.var;
  const corr = state.corr;
  state.n += 1;

  const delta = new Array(n);
  for (let i = 0; i < n; i++) delta[i] = y[i] - mean[i];
  for (let i = 0; i < n; i++) mean[i] += alpha * delta[i];
  const delta2 = new Array(n);
  for (let i = 0; i < n; i++) delta2[i] = y[i] - mean[i];
  for (let i = 0; i < n; i++) varr[i] = (1 - alpha) * varr[i] + alpha * delta[i] * delta2[i];

  for (let i = 0; i < n; i++) {
    const si = varr[i] > 1e-16 ? Math.sqrt(varr[i]) : 1e-8;
    for (let j = i + 1; j < n; j++) {
      const sj = varr[j] > 1e-16 ? Math.sqrt(varr[j]) : 1e-8;
      const zCross = (delta[i] / si) * (delta[j] / sj);
      const idx = i * n + j;
      let c = (1 - alpha) * corr[idx] + alpha * zCross;
      c = Math.max(-1.0, Math.min(1.0, c));
      corr[idx] = c;
      corr[j * n + i] = c;
    }
  }

  const shrunkCov = new Array(n * n).fill(0.0);
  for (let i = 0; i < n; i++) {
    const si = varr[i] > 1e-16 ? Math.sqrt(varr[i]) : 1e-8;
    for (let j = 0; j < n; j++) {
      const sj = varr[j] > 1e-16 ? Math.sqrt(varr[j]) : 1e-8;
      if (i === j) shrunkCov[i * n + j] = varr[i];
      else shrunkCov[i * n + j] = (1 - shrinkage) * corr[i * n + j] * si * sj;
    }
  }
  return [mean.slice(), shrunkCov, state];
}
