"""FRED data loader for the benchmark — stdlib only, with local caching.

Set FRED_API_KEY (env var, a gitignored skaters/.env, ./.env, or ~/.env). Series
are fetched once and cached under benchmarks/data/, so later runs work offline.

We forecast the one-step *change* of each series (first difference, or
log-difference for strictly-positive levels) — that is the stationary-ish
innovation stream where heavy tails and regime shifts actually live, and it
keeps log-likelihood comparable across series.
"""

from __future__ import annotations
import json
import math
import os
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = os.environ.get("TIMEMACHINES_FRED_DATA",
                        os.path.join(_HERE, "data"))

# Macro/financial series with the heavy tails and regime shifts that break
# exchangeability-based conformal — the point of the race.
SERIES = {
    "DGS10": "10Y Treasury yield (daily)",
    "DFF": "Effective fed funds rate (daily)",
    "VIXCLS": "VIX (daily) — fat-tailed",
    "DCOILWTICO": "WTI crude oil (daily) — fat-tailed",
    "DEXUSEU": "USD/EUR exchange rate (daily)",
    "T10Y2Y": "10Y-2Y term spread (daily)",
    "BAMLH0A0HYM2": "High-yield credit spread (daily)",
    "DEXJPUS": "JPY/USD exchange rate (daily)",
}


def _api_key():
    key = os.environ.get("FRED_API_KEY", "")
    for path in (os.path.join(_HERE, "..", ".env"),   # skaters/.env (gitignored)
                 "./.env", os.path.expanduser("~/.env")):
        if not key and os.path.exists(path):
            for line in open(path):
                if line.strip().startswith("FRED_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    return key


def _fetch(series_id, start="2005-01-01", retries=4):
    key = _api_key()
    if not key:
        return None
    url = (f"https://api.stlouisfed.org/fred/series/observations?"
           f"series_id={series_id}&api_key={key}&file_type=json&observation_start={start}")
    for attempt in range(retries):
        try:
            resp = urllib.request.urlopen(url, timeout=20)
            obs = json.loads(resp.read()).get("observations", [])
            out = []
            for o in obs:
                try:
                    out.append((o["date"], float(o["value"])))
                except (ValueError, KeyError):
                    continue
            return out
        except Exception as e:  # noqa: BLE001 — network is best-effort
            print(f"  FRED error for {series_id}: {e}")
            time.sleep(2 * (attempt + 1))
    return None


def _load_levels(series_id):
    """Return [(date, level)] for a series, from cache or the API."""
    os.makedirs(_CACHE, exist_ok=True)
    cache = os.path.join(_CACHE, f"{series_id}.csv")
    if os.path.exists(cache):
        rows = []
        for line in open(cache):
            d, v = line.rstrip("\n").split(",")
            rows.append((d, float(v)))
        return rows
    rows = _fetch(series_id)
    if rows:
        with open(cache, "w") as f:
            for d, v in rows:
                f.write(f"{d},{v}\n")
    return rows


def _to_changes(levels):
    """First difference, or log-difference if strictly positive."""
    vals = [v for _, v in levels]
    if len(vals) < 10:
        return []
    positive = all(v > 0 for v in vals)
    out = []
    for a, b in zip(vals[:-1], vals[1:]):
        out.append(math.log(b) - math.log(a) if positive else b - a)
    return out


def load_fred(min_len=400):
    """Return {series_id: change_series}. Empty if no key and no cache."""
    data = {}
    for sid in SERIES:
        levels = _load_levels(sid)
        if not levels:
            continue
        ch = _to_changes(levels)
        if len(ch) >= min_len:
            data[sid] = ch
    return data


if __name__ == "__main__":
    d = load_fred()
    if not d:
        print("No FRED data (set FRED_API_KEY). Cache dir:", _CACHE)
    for sid, ch in d.items():
        print(f"{sid:14s} n={len(ch)}  {SERIES[sid]}")
