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

// skaters#224: delegate to emaCov (a genuine rank-one EMA update, PSD by
// construction) and shrink THAT coherent covariance toward its diagonal --
// (1-rho)*C + rho*diag(C) stays PSD because C and diag(C) are both PSD and
// PSD matrices are convex. The prior version updated each pairwise
// correlation independently and clamped it to [-1, 1] on its own; pairwise-
// valid correlations do not imply the resulting MATRIX is PSD.
export function ledoitWolfCov(y, state, alpha = 0.05, shrinkage = 0.5) {
  const n = y.length;
  const [mean, cov, innerState] = emaCov(y, state ? state.inner : null, alpha);
  const newState = { inner: innerState };

  const shrunkCov = new Array(n * n).fill(0.0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      shrunkCov[i * n + j] = i === j ? cov[i * n + j] : (1 - shrinkage) * cov[i * n + j];
    }
  }
  return [mean, shrunkCov, newState];
}
