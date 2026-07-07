// Lightweight online running statistics — JS port of skaters/runstats.py.
// Welford's algorithm for mean/variance. Pure JS, no dependencies.

export function runningVarInit() {
  return { n: 0, mean: 0.0, m2: 0.0 };
}

export function runningVarUpdate(state, x) {
  const n = state.n + 1;
  const delta = x - state.mean;
  const mean = state.mean + delta / n;
  const delta2 = x - mean;
  const m2 = state.m2 + delta * delta2;
  return { n, mean, m2 };
}

export function runningVarGet(state) {
  // Returns [mean, variance]. Variance is +Infinity until n >= 2.
  if (state.n < 2) return [state.mean, Infinity];
  return [state.mean, state.m2 / (state.n - 1)];
}

export function runningStdGet(state) {
  const [, v] = runningVarGet(state);
  return Number.isFinite(v) ? Math.sqrt(v) : Infinity;
}

export function runningMseGet(state) {
  if (state.n < 1) return Infinity;
  const [mean, v] = runningVarGet(state);
  if (!Number.isFinite(v)) return Infinity;
  return mean * mean + v;
}
