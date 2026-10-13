"""TSB-AD-U scorer — phase 1 of the leaderboard run (see RESEARCH.md).

Computes strictly-prequential per-tick anomaly scores for every series in the
TSB-AD-U tuning or eval split and caches them as one .npz per series. This is
the expensive phase; it is resumable (existing .npz files are skipped) and
deliberately cheap on the machine (default --workers 2, run it under nice).

Metrics (VUS-PR etc.) are computed in phase 2 by ``tsb_ad_eval.py`` from the
cached arrays, using the official TSB-AD package in its own venv — that
package pins numpy<2/torch and must not share this venv. Windowed scan
variants (mahS w) are derived at eval time from the cached ``mah`` array, so
window sweeps never re-run detection.

Protocol notes:
  * Scores are emitted for EVERY tick (t=0 onward, score-before-update), so
    the arrays align 1:1 with the Label column; the eval phase can restrict
    to the post-``tr_`` region if wanted. Nothing here peeks ahead: no
    whole-series normalisation, no threshold, one pass.
  * Tune on the 48-series tuning split ONLY (--split tuning); the eval split
    is run once per frozen config.

Data: benchmarks/data/TSB-AD-U/ (see RESEARCH.md for the zip URL); file lists
are fetched once from the TSB-AD GitHub repo and cached alongside the data.

Usage:
    python benchmarks/tsb_ad_run.py --split tuning --workers 2
    python benchmarks/tsb_ad_run.py --split eval --workers 2   # long run
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.request
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get(
    "TIMEMACHINES_TSB_DATA",
    os.path.join(_HERE, "data", "TSB-AD-U"))
SCORES = os.environ.get(
    "TIMEMACHINES_TSB_SCORES",
    os.path.join(_HERE, "data", "tsb_scores"))
_LIST_URL = ("https://raw.githubusercontent.com/TheDatumOrg/TSB-AD/main/"
             "Datasets/File_List/TSB-AD-U-{split}.csv")
_SPLIT_FILE = {"tuning": "Tuning", "eval": "Eva"}


def file_list(split: str) -> list:
    """Cached official split lists (tuning: 48 series, eval: 350)."""
    os.makedirs(DATA, exist_ok=True)
    cache = os.path.join(DATA, f"file_list_{split}.csv")
    if not os.path.exists(cache):
        url = _LIST_URL.format(split=_SPLIT_FILE[split])
        with urllib.request.urlopen(url) as r:
            body = r.read().decode()
        with open(cache, "w") as f:
            f.write(body)
    names = [ln.strip() for ln in open(cache) if ln.strip()]
    assert names[0] == "file_name", f"unexpected list header: {names[0]}"
    return names[1:]


def find_csv(fname: str) -> str:
    """The zip layout has varied (flat vs nested); search once, cheaply."""
    direct = os.path.join(DATA, fname)
    if os.path.exists(direct):
        return direct
    for root, _dirs, files in os.walk(DATA):
        if fname in files:
            return os.path.join(root, fname)
    raise FileNotFoundError(f"{fname} not under {DATA} — dataset not extracted?")


def load_series(path: str):
    """Return (values, labels) from a TSB-AD csv (value column(s) + Label)."""
    with open(path) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    li = header.index("Label")
    vi = 0 if li != 0 else 1
    ys = np.array([float(r[vi]) for r in rows[1:]], dtype=np.float64)
    labels = np.array([int(float(r[li])) for r in rows[1:]], dtype=np.int8)
    return ys, labels


def parse_train_len(fname: str) -> int:
    for tok in fname.replace(".csv", "").split("_"):
        pass
    parts = fname.replace(".csv", "").split("_")
    return int(parts[parts.index("tr") + 1]) if "tr" in parts else 0


def run_one(job):
    fname, k, scale_alpha, det_alpha, out_dir = job
    out_path = os.path.join(out_dir, fname.replace(".csv", ".npz"))
    ys, labels = load_series(find_csv(fname))
    n = len(ys)

    from timemachines import laplace, mahalanobis
    import skaters as _sk

    f = mahalanobis(laplace(k, scale_alpha=scale_alpha), k=k, alpha=det_alpha)
    state = None
    mah = np.zeros(n, dtype=np.float32)   # -log10 p (Mahalanobis)
    z1 = np.zeros(n, dtype=np.float32)    # |z_1| one-step parade surprise
    zU = np.zeros(n, dtype=np.float32)    # union min-p channel
    mz = np.zeros(n, dtype=np.float32)    # trivial EWMA z-score of raw y
    mz_m, mz_v, mz_n = 0.0, 0.0, 0
    ALPHA = 0.02

    t0 = time.time()
    for t in range(n):
        y = float(ys[t])
        _, state = f(y, state)
        mz_n += 1
        a = max(ALPHA, 1.0 / mz_n)
        if mz_n > 3 and mz_v > 0:
            # denormal-tiny variance can overflow the division to inf
            mz[t] = min(abs(y - mz_m) / math.sqrt(mz_v), 1e12)
        d = y - mz_m
        mz_m += a * d
        mz_v = (1 - a) * mz_v + a * d * (y - mz_m)

        p = state["pvalue"]
        if p is not None:
            mah[t] = -math.log10(max(p, 1e-300))
        zvec = state["base"]["z"] if isinstance(state["base"], dict) else None
        if zvec and zvec[0] is not None:
            z1[t] = abs(zvec[0])
        if p is not None and zvec is not None and all(v is not None for v in zvec):
            su = mah[t]
            for zm in zvec:
                pm = math.erfc(abs(zm) / math.sqrt(2.0))
                su = max(su, -math.log10(max(pm, 1e-300)))
            zU[t] = su

    np.savez_compressed(out_path, mah=mah, z1=z1, zU=zU, mz=mz)
    return {"fname": fname, "n": n, "n_anom": int(labels.sum()),
            "train_len": parse_train_len(fname),
            "seconds": round(time.time() - t0, 1),
            "skaters": getattr(_sk, "__version__", "?"),
            "k": k, "scale_alpha": scale_alpha, "det_alpha": det_alpha}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("tuning", "eval"), default="tuning")
    ap.add_argument("--limit", type=int, default=0,
                    help="run only the N shortest series (0 = whole split)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2,
                    help="keep low; another study may share the machine")
    ap.add_argument("--scale-alpha", type=float, default=0.01,
                    help="0.01 = slow memory (UCR winner); 0.03 = default")
    ap.add_argument("--det-alpha", type=float, default=0.005)
    args = ap.parse_args()

    tag = f"laplace_k{args.k}_sa{args.scale_alpha:g}_da{args.det_alpha:g}"
    out_dir = os.path.join(SCORES, tag)
    os.makedirs(out_dir, exist_ok=True)
    manifest = os.path.join(out_dir, f"manifest_{args.split}.jsonl")

    names = file_list(args.split)
    # shortest first: fast series land early, long stragglers resume cleanly
    names.sort(key=lambda f: os.path.getsize(find_csv(f)))
    if args.limit:
        names = names[:args.limit]
    done = set()
    if os.path.exists(manifest):
        done = {json.loads(l)["fname"] for l in open(manifest) if l.strip()}
    todo = [f for f in names
            if f not in done
            or not os.path.exists(os.path.join(out_dir, f.replace(".csv", ".npz")))]
    print(f"[tsb_ad_run] split={args.split} tag={tag}: "
          f"{len(names)} series, {len(names) - len(todo)} cached, {len(todo)} to go",
          flush=True)
    if not todo:
        return

    jobs = [(f, args.k, args.scale_alpha, args.det_alpha, out_dir) for f in todo]
    t0 = time.time()
    with Pool(args.workers) as pool, open(manifest, "a") as mf:
        for i, res in enumerate(pool.imap_unordered(run_one, jobs), 1):
            mf.write(json.dumps(res) + "\n")
            mf.flush()
            print(f"  [{i}/{len(jobs)}] {res['fname']} n={res['n']} "
                  f"{res['seconds']}s (elapsed {round(time.time()-t0)}s)",
                  flush=True)
    print(f"[tsb_ad_run] done: scores in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
