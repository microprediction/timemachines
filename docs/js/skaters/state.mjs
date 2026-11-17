// Checkpoint and restore for skater state — the fold contract.
//
// A skater is a fold: state must be plain data that round-trips through JSON
// bit-exactly, so a stream can be stopped, stored, and resumed anywhere with
// the same numbers. Skater state holds only null, booleans, finite numbers,
// strings, arrays, plain objects, and Dist objects (including registered
// extensions such as the GPD-spliced predictive). Closures are configuration
// and live in the skater's wrapper, never in state.
//
//   const plain = stateToJSON(state);            // Dist -> {components}, else as is
//   const text  = JSON.stringify(plain);
//   const back  = stateFromJSON(JSON.parse(text)); // {components} -> Dist, no renormalising
//
// Both walkers throw, naming the offending path, on anything JSON would
// silently damage: functions, Map, Set, typed arrays, class instances other
// than a Dist, undefined, and non-finite numbers. parity/roundtrip.mjs holds
// every exported skater to this contract.

import { Dist, isDistInstance } from "./dist.mjs";
import { registerSplicedDist } from "./tails.mjs";

// Restoring a checkpoint must not depend on import order, or on whether the
// consumer also runs a skater. A reader that only rehydrates states imports
// this module and nothing else, so the spliced decoder is registered here, by
// a named call a bundler cannot drop. (A bare `import "./tails.mjs"` is
// dropped: the package declares sideEffects:false.)
registerSplicedDist();

function isPlainObject(v) {
  const p = Object.getPrototypeOf(v);
  return p === Object.prototype || p === null;
}

function offend(path, what) {
  throw new Error(`skater state is not plain data at ${path || "<root>"}: ${what}`);
}

function walk(v, path, build) {
  if (v === null) return v;
  const t = typeof v;
  if (t === "number") {
    if (!Number.isFinite(v)) offend(path, `non-finite number ${v} (JSON would write null)`);
    return v;
  }
  if (t === "string" || t === "boolean") return v;
  if (t === "undefined") offend(path, "undefined (JSON drops it or writes null)");
  if (t === "function") offend(path, "a function (closures belong in the wrapper, not in state)");
  if (t !== "object") offend(path, `a ${t}`);
  if (isDistInstance(v)) return build ? v.toDict() : v;
  if (Array.isArray(v)) {
    if (!build) { for (let i = 0; i < v.length; i++) walk(v[i], `${path}[${i}]`, false); return v; }
    const out = new Array(v.length);
    for (let i = 0; i < v.length; i++) out[i] = walk(v[i], `${path}[${i}]`, true);
    return out;
  }
  if (v instanceof Map) offend(path, "a Map (JSON flattens it to {})");
  if (v instanceof Set) offend(path, "a Set (JSON flattens it to {})");
  if (ArrayBuffer.isView(v)) offend(path, "a typed array (JSON writes an object of indices)");
  if (!isPlainObject(v)) {
    const name = (v.constructor && v.constructor.name) || "unknown";
    offend(path, `an instance of ${name} (methods do not survive JSON)`);
  }
  if (!build) { for (const k of Object.keys(v)) walk(v[k], `${path}.${k}`, false); return v; }
  const out = {};
  for (const k of Object.keys(v)) out[k] = walk(v[k], `${path}.${k}`, true);
  return out;
}

// Throws if `state` holds anything that would not survive JSON.stringify/parse.
export function assertPlainState(state) {
  walk(state, "", false);
  return state;
}

// Plain-data image of `state`: every Dist becomes its toDict() form. Throws
// on anything else non-plain. Also accepts a list of Dists (a skater output).
export function stateToJSON(state) {
  return walk(state, "", true);
}

function isDistDict(v) {
  if (v.spliced === true) return true;
  const keys = Object.keys(v);
  return keys.length === 1 && keys[0] === "components" && Array.isArray(v.components);
}

// Inverse of stateToJSON. A {components: [[w, m, s], ...]} object becomes a
// Dist without renormalising (Dist.trusted), a {spliced: true, ...}
// object becomes a SplicedDist; everything else is copied as is.
export function stateFromJSON(obj) {
  if (obj === null || typeof obj !== "object") return obj;
  if (Array.isArray(obj)) return obj.map(stateFromJSON);
  if (isDistDict(obj)) return Dist.fromDict(obj);
  const out = {};
  for (const k of Object.keys(obj)) out[k] = stateFromJSON(obj[k]);
  return out;
}
