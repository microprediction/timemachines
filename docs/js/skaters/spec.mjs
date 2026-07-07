// Symbolic spec for skater pipelines — JS port of skaters/spec.py.

import { leaf } from "./leaf.mjs";
import { ema } from "./ema.mjs";
import { precisionWeightedEnsemble } from "./ensemble.mjs";
import { conjugate } from "./conjugate.mjs";
import { difference, fractionalDifference, standardize, emaTransform } from "./transform.mjs";

export function build(spec) {
  const op = spec.op;
  let f;
  if (op === "leaf") {
    f = leaf(spec.k);
  } else if (op === "ema") {
    f = ema(spec.alpha, spec.k);
  } else if (op === "ensemble") {
    const subs = spec.skaters.map(build);
    f = precisionWeightedEnsemble(subs, spec.k, spec.floor === undefined ? 1e-6 : spec.floor);
  } else if (op === "conjugate") {
    const inner = build(spec.skater);
    const t = buildTransform(spec.transform);
    f = conjugate(inner, t, inferK(spec.skater));
  } else {
    throw new Error(`Unknown op: ${op}`);
  }
  f.skaterName = name(spec);
  return f;
}

function buildTransform(spec) {
  const op = spec.op;
  if (op === "diff") return difference();
  if (op === "frac") return fractionalDifference(spec.d, spec.window === undefined ? 50 : spec.window);
  if (op === "std") return standardize(spec.alpha === undefined ? 0.05 : spec.alpha);
  if (op === "ema_t") return emaTransform(spec.alpha);
  throw new Error(`Unknown transform op: ${op}`);
}

function inferK(spec) {
  if ("k" in spec) return spec.k;
  if ("skater" in spec) return inferK(spec.skater);
  if ("skaters" in spec) return inferK(spec.skaters[0]);
  throw new Error("Cannot infer k from spec");
}

// --- canonical name ---

function fmt(x) {
  // Mimic Python's "%.6g" with integer collapse.
  if (x === Math.trunc(x)) return String(Math.trunc(x));
  let s = x.toPrecision(6);
  if (s.indexOf(".") >= 0 && s.indexOf("e") < 0 && s.indexOf("E") < 0) {
    s = s.replace(/0+$/, "").replace(/\.$/, "");
  }
  return s;
}

export function name(spec) {
  const op = spec.op;
  if (op === "leaf") return "leaf";
  if (op === "ema") return `ema(${fmt(spec.alpha)})`;
  if (op === "ensemble") return `ensemble(${spec.skaters.map(name).join(",")})`;
  if (op === "conjugate") return `${transformName(spec.transform)}|${name(spec.skater)}`;
  throw new Error(`Unknown op: ${op}`);
}

function transformName(spec) {
  const op = spec.op;
  if (op === "diff") return "diff";
  if (op === "frac") {
    const w = spec.window === undefined ? 50 : spec.window;
    return w === 50 ? `frac(${fmt(spec.d)})` : `frac(${fmt(spec.d)},w=${w})`;
  }
  if (op === "std") return `std(${fmt(spec.alpha === undefined ? 0.05 : spec.alpha)})`;
  if (op === "ema_t") return `ema_t(${fmt(spec.alpha)})`;
  throw new Error(`Unknown transform op: ${op}`);
}

// --- serialization ---

export function toJson(spec) {
  return JSON.stringify(spec);
}

export function fromJson(s) {
  return JSON.parse(s);
}

// --- spec constructors ---

export const leafSpec = (k = 1) => ({ op: "leaf", k });
export const emaSpec = (alpha = 0.05, k = 1) => ({ op: "ema", alpha, k });
export const ensembleSpec = (skaterSpecs, k = 1) => ({ op: "ensemble", k, skaters: skaterSpecs });
export const conjugateSpec = (skaterSpec, transformSpec) => ({ op: "conjugate", skater: skaterSpec, transform: transformSpec });
export const diffSpec = () => ({ op: "diff" });
export const fracSpec = (d = 0.4, window = 50) => ({ op: "frac", d, window });
export const stdSpec = (alpha = 0.05) => ({ op: "std", alpha });
export const emaTSpec = (alpha = 0.05) => ({ op: "ema_t", alpha });
