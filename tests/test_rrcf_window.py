"""Regression coverage for the benchmark RRCF rolling population (#165)."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def frontend(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    spec = importlib.util.spec_from_file_location(
        "rrcf_frontend_under_test", root / "benchmarks" / "frontend_run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("shingle,tree_size,length", [
    (1, 1, 9), (1, 4, 20), (4, 1, 12), (4, 4, 24),
    (4, 256, 2000), (8, 3, 30), (4, 256, 3), (4, 256, 4),
])
def test_population_is_exactly_latest_complete_shingles(
    frontend, monkeypatch, shingle, tree_size, length
):
    trees = []

    class RecordingTree:
        def __init__(self, random_state):
            self.leaves = {}
            self.snapshots = []
            trees.append(self)

        def forget_point(self, index):
            del self.leaves[index]

        def insert_point(self, point, index):
            self.leaves[index] = point

        def codisp(self, index):
            self.snapshots.append((index, set(self.leaves)))
            return float(len(self.leaves))

    monkeypatch.setitem(sys.modules, "rrcf", SimpleNamespace(RCTree=RecordingTree))
    scores = frontend.rrcf_scores(
        list(range(length)), num_trees=3, shingle=shingle, tree_size=tree_size
    )
    assert len(trees) == 3
    assert len(scores) == length
    assert scores[:min(length, shingle - 1)] == [0.0] * min(length, shingle - 1)
    for tree in trees:
        assert len(tree.snapshots) == max(0, length - shingle + 1)
        for i, retained in tree.snapshots:
            assert retained == set(range(max(shingle - 1, i - tree_size + 1), i + 1))
            assert scores[i] == float(len(retained))


@pytest.mark.parametrize("constant", [False, True])
def test_real_rrcf_rolls_indices_even_for_duplicate_points(frontend, monkeypatch, constant):
    rrcf = pytest.importorskip("rrcf")
    original = rrcf.RCTree
    snapshots = []

    class RecordingTree(original):
        def codisp(self, index):
            snapshots.append((index, set(self.leaves)))
            return super().codisp(index)

    monkeypatch.setattr(rrcf, "RCTree", RecordingTree)
    xs = [1.0] * 50 if constant else [float(i) for i in range(50)]
    scores = frontend.rrcf_scores(xs, num_trees=2, shingle=4, tree_size=8)
    assert len(scores) == 50
    assert len(snapshots) == 2 * 47
    for i, retained in snapshots:
        assert retained == set(range(max(3, i - 7), i + 1))
