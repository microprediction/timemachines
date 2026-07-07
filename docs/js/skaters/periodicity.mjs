// Online periodicity detection — JS port of skaters/periodicity.py.

export const DEFAULT_LAGS = [2, 3, 4, 5, 6, 7, 12, 14, 24, 28, 30, 52, 60, 90, 168, 365];

export function periodDetector(lags = null, alpha = 0.01, minObservations = 50) {
  if (lags === null) lags = DEFAULT_LAGS.slice();
  const maxLag = Math.max(...lags);

  function _detect(y, state) {
    if (state === null || state === undefined) {
      const cross = {};
      for (const L of lags) cross[L] = 0.0;
      state = { buffer: [], n: 0, mean: 0.0, var: 0.0, cross };
    }
    const buf = state.buffer;
    buf.push(y);
    state.n += 1;

    const diff = y - state.mean;
    state.mean += alpha * diff;
    state.var = (1 - alpha) * (state.var + alpha * diff * diff);

    const mu = state.mean;
    const varr = state.var;

    for (const L of lags) {
      if (buf.length > L) {
        const yLagged = buf[buf.length - 1 - L];
        const cross = (y - mu) * (yLagged - mu);
        state.cross[L] = (1 - alpha) * state.cross[L] + alpha * cross;
      }
    }

    if (buf.length > maxLag + 1) buf.shift();

    if (state.n < minObservations || varr < 1e-12) return [[], state];

    const scores = [];
    for (let li = 0; li < lags.length; li++) {
      const L = lags[li];
      if (state.n > L) {
        const acf = varr > 0 ? state.cross[L] / varr : 0.0;
        scores.push([L, acf, li]); // li carried to keep the stable-sort order
      }
    }
    // Sort by |acf| descending, ties keep original (lags) order.
    scores.sort((a, b) => {
      const d = Math.abs(b[1]) - Math.abs(a[1]);
      return d !== 0 ? d : a[2] - b[2];
    });
    return [scores.map(([L, acf]) => [L, acf]), state];
  }
  return _detect;
}

export function topPeriods(scores, threshold = 0.3, maxPeriods = 3) {
  return scores.slice(0, maxPeriods).filter(([, acf]) => Math.abs(acf) >= threshold).map(([lag]) => lag);
}
