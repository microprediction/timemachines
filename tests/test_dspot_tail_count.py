"""Benchmark POT frequency must not depend on the capped GPD fit buffer."""
import math
from pathlib import Path
import pickle
import sys

import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / 'benchmarks'
sys.path.insert(0, str(BENCHMARKS))
from frontend_run import _Tail, DSpot


def test_frequency_after_fit_buffer_saturates():
    tail = _Tail([0.0] * 981 + [1.0] * 19, .98)
    count = 19
    for tick in range(1, 50051):
        x = float(tick % 50 == 0)
        expected = 1 - math.log(count / (1000 + tick)) if x else 0.0
        assert tail.logp(x) == pytest.approx(expected, rel=1e-14)
        count += int(x)
        assert len(tail.exc) == min(count, 500)
    assert tail.n_exc_total == 1020
    assert tail.n_exc_total / tail.n == pytest.approx(1020 / 51050)


def test_count_tracks_only_strict_exceedances_and_scores_before_update():
    tail = _Tail([0.] * 98 + [1.] * 2, .97)
    for x in (-2., 0., 0., 1., 0., 2.):
        old_count, old_n = tail.n_exc_total, tail.n
        if x > tail.t:
            gamma, sigma = tail._gpd()
            e = x - tail.t
            lt = -e / sigma if abs(gamma) < 1e-9 else -math.log(1 + gamma * e / sigma) / gamma
            assert tail.logp(x) == pytest.approx(-math.log(old_count / (old_n + 1)) - lt)
        else:
            assert tail.logp(x) == 0.
        assert tail.n_exc_total == old_count + int(x > tail.t)
        assert tail.n == old_n + 1


def test_two_sided_counts_and_new_checkpoint_continuation():
    detector = DSpot([-1., 0., 1.] * 500, level=.5, drift_alpha=0.)
    for x in [-1., 0., 1.] * 600:
        detector.score(x)
    assert detector.up.n_exc_total == detector.lo.n_exc_total == 1100
    assert len(detector.up.exc) == len(detector.lo.exc) == 500
    restored = pickle.loads(pickle.dumps(detector))
    assert [detector.score(x) for x in [0., 1., -1., 2., -2.]] == [restored.score(x) for x in [0., 1., -1., 2., -2.]]


def test_empty_excess_behavior_is_unchanged():
    tail = _Tail([0.] * 100, .98)
    for x in [0., 1., -1.]:
        assert tail.logp(x) == 0.
    assert tail.n_exc_total == 0
    assert tail.exc == []
