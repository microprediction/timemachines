"""TSB-AD-U metrics — phase 2 of the leaderboard run (see tsb_ad_run.py).

Reads the per-series score arrays cached by ``tsb_ad_run.py`` and computes
the official metric bundle (VUS-PR headline, plus VUS-ROC / AUC-PR / AUC-ROC
and the rest of what ``get_metrics`` returns) with the official TSB-AD
package. Run it in the dedicated metrics venv (.venv-tsb) — TSB-AD pins
numpy<2 and torch, and must not share the detection venv:

    .venv-tsb/bin/python benchmarks/tsb_ad_eval.py --split tuning

Resumable: one jsonl row per (series, method); existing keys are skipped.
Windowed scan variants (mahS<w>) are derived here from the cached ``mah``
array — sweeping w never re-runs detection. The sliding window for VUS is
estimated from the data with the package's own ``find_length_rank``,
mirroring the TSB-AD benchmark scripts.
"""

from __future__ import annotations
import argparse
import csv
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("TIMEMACHINES_TSB_DATA",
                      os.path.join(_HERE, "data", "TSB-AD-U"))
SCORES = os.environ.get("TIMEMACHINES_TSB_SCORES",
                        os.path.join(_HERE, "data", "tsb_scores"))

try:
    from TSB_AD.evaluation.metrics import get_metrics
    from TSB_AD.utils.slidingWindows import find_length_rank
except ImportError as e:
    raise SystemExit(
        f"TSB-AD package not importable ({e}); run this under .venv-tsb "
        "(uv pip install --python .venv-tsb/bin/python TSB-AD)")


def find_csv(fname: str) -> str:
    direct = os.path.join(DATA, fname)
    if os.path.exists(direct):
        return direct
    for root, _dirs, files in os.walk(DATA):
        if fname in files:
            return os.path.join(root, fname)
    raise FileNotFoundError(fname)


def load_series(path: str):
    with open(path) as f:
        rows = list(csv.reader(f))
    li = rows[0].index("Label")
    vi = 0 if li != 0 else 1
    ys = np.array([float(r[vi]) for r in rows[1:]], dtype=np.float64)
    labels = np.array([int(float(r[li])) for r in rows[1:]], dtype=np.int64)
    return ys, labels


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Causal rolling mean, right-aligned; prefix uses the partial window."""
    c = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    out = np.empty_like(x, dtype=np.float64)
    n = len(x)
    idx = np.arange(n)
    lo = np.maximum(idx - w + 1, 0)
    out = (c[idx + 1] - c[lo]) / (idx - lo + 1)
    return out


def methods_from(npz) -> dict:
    out = {k: npz[k].astype(np.float64) for k in npz.files}
    for w in (8, 64):
        out[f"mahS{w}"] = rolling_mean(out["mah"], w)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("tuning", "eval"), default="tuning")
    ap.add_argument("--tag", default="laplace_k3_sa0.01_da0.005")
    args = ap.parse_args()

    score_dir = os.path.join(SCORES, args.tag)
    manifest = os.path.join(score_dir, f"manifest_{args.split}.jsonl")
    if not os.path.exists(manifest):
        raise SystemExit(f"no manifest {manifest} — run tsb_ad_run.py first")
    fnames = sorted({json.loads(l)["fname"] for l in open(manifest) if l.strip()})

    out_path = os.path.join(score_dir, f"metrics_{args.split}.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {(r["fname"], r["method"])
                for r in map(json.loads, open(out_path)) if r}
    print(f"[tsb_ad_eval] {len(fnames)} series scored; "
          f"{len(done)} (series, method) rows cached", flush=True)

    with open(out_path, "a") as out:
        for i, fname in enumerate(fnames, 1):
            npz_path = os.path.join(score_dir, fname.replace(".csv", ".npz"))
            if not os.path.exists(npz_path):
                continue
            ys, labels = load_series(find_csv(fname))
            scores = methods_from(np.load(npz_path))
            sw = find_length_rank(ys.reshape(-1, 1), rank=1)
            for method, s in scores.items():
                if (fname, method) in done:
                    continue
                if len(s) != len(labels):
                    print(f"  SKIP {fname}/{method}: len {len(s)} != {len(labels)}",
                          flush=True)
                    continue
                m = get_metrics(s, labels, slidingWindow=sw)
                row = {"fname": fname, "method": method, "sliding_window": int(sw)}
                row.update({k: (round(float(v), 6) if isinstance(v, (int, float, np.floating)) else v)
                            for k, v in m.items()})
                out.write(json.dumps(row) + "\n")
            out.flush()
            if i % 10 == 0:
                print(f"  [{i}/{len(fnames)}] {fname}", flush=True)

    # summary: mean per method, VUS-PR headline
    rows = [json.loads(l) for l in open(out_path) if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["method"], []).append(r)
    keys = [k for k in ("VUS-PR", "VUS-ROC", "AUC-PR", "AUC-ROC")
            if rows and k in rows[0]]
    print(f"\n=== TSB-AD-U {args.split} split, tag {args.tag} "
          f"({len(fnames)} series) ===")
    for method, rs in sorted(by.items(),
                             key=lambda kv: -np.mean([r.get("VUS-PR", 0) for r in kv[1]])):
        line = "  ".join(f"{k} {np.mean([r.get(k, 0.0) for r in rs]):.4f}"
                         for k in keys)
        print(f"{method:8s} n={len(rs):3d}  {line}")


if __name__ == "__main__":
    main()
