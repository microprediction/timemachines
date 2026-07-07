// EMA skater — JS port of skaters/ema.py.
// ema(alpha, k) = conjugate(leaf(k), emaTransform(alpha)).

import { leaf } from "./leaf.mjs";
import { emaTransform } from "./transform.mjs";
import { conjugate } from "./conjugate.mjs";

export function ema(alpha = 0.05, k = 1) {
  const f = conjugate(leaf(k), emaTransform(alpha), k);
  f.skaterName = `ema(alpha=${alpha}, k=${k})`;
  return f;
}
