"""Benchmark-only simulation identity and fail-closed resume checks."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
import granger_overnight as g


def test_simulations_ignore_python_hash_seed():
    code = f'''import sys,json,hashlib
sys.path.insert(0, {str(BENCHMARKS)!r})
import granger_overnight as g
print(hashlib.sha256(json.dumps([g.simulate(sc,T,17) for sc,T,_ in g.CELLS]).encode()).hexdigest())
'''
    values = [subprocess.check_output([sys.executable, "-c", code],
              env={**os.environ, "PYTHONHASHSEED": str(seed)}, text=True)
              for seed in [0, 1, 2]]
    assert len(set(values)) == 1


def test_seed_mapping_is_versioned_and_distinguishes_keys():
    import hashlib
    expected = int(hashlib.sha256(b'["sha256-json-v1","n_volc",300,17]').hexdigest(), 16)
    assert g.resolved_seed("n_volc", 300, 17) == expected
    assert len({expected, g.resolved_seed("n_volc", 300, 18),
                g.resolved_seed("n_volc", 301, 17),
                g.resolved_seed("n_iid", 300, 17)}) == 4


def row():
    return dict(scenario="n_volc", T=300, seed=17,
                seed_scheme=g.SEED_SCHEME, rng_seed=g.resolved_seed("n_volc", 300, 17))


def test_resume_accepts_current_and_missing(tmp_path):
    path = tmp_path / "results.jsonl"
    assert g.load_results(path) == []
    path.write_text("\n" + json.dumps(row()) + "\n")
    assert g.load_results(path) == [row()]


@pytest.mark.parametrize("change", ["legacy", "scheme", "seed", "mixed"])
def test_resume_rejects_incompatible_without_writing(tmp_path, change):
    bad = row()
    if change == "legacy":
        del bad["seed_scheme"]
        del bad["rng_seed"]
    elif change == "scheme":
        bad["seed_scheme"] = "future-v2"
    else:
        bad["rng_seed"] += 1
    contents = (json.dumps(row()) + "\n" if change == "mixed" else "") + json.dumps(bad) + "\n"
    path = tmp_path / "results.jsonl"
    path.write_text(contents)
    with pytest.raises(ValueError, match="fresh --out"):
        g.load_results(path)
    assert path.read_text() == contents


def test_run_one_records_actual_seed(monkeypatch):
    monkeypatch.setattr(g, "_ols_tests", lambda y,x: (1,1,1,[]))
    monkeypatch.setattr(g, "_ar_resid", lambda x: x)
    monkeypatch.setattr(g, "_ewma_std", lambda x: x)
    monkeypatch.setattr(g, "_garch_fit_std", lambda x: x)
    monkeypatch.setattr(g, "_ccf_pvals", lambda x,y: [1]*4)
    monkeypatch.setattr(g, "_z2_pvals", lambda x,y: [1]*4)
    result = g.run_one(("n_volc",300,17))
    assert all(result[key] == value for key,value in row().items())


def test_cli_rejects_legacy_before_pool(tmp_path):
    path = tmp_path / "legacy.jsonl"
    contents = json.dumps(dict(scenario="n_volc", T=300, seed=17)) + "\n"
    path.write_text(contents)
    result = subprocess.run([sys.executable, str(BENCHMARKS / "granger_overnight.py"),
                             "--workers", "0", "--out", str(path)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "Incompatible seed provenance" in result.stderr
    assert "Number of processes" not in result.stderr
    assert path.read_text() == contents
