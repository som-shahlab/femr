import numpy as np
import pytest

from femr.featurizers.utils import OnlineStatistics


def _assert_correct_stats(stat: OnlineStatistics, values: list):
    TOLERANCE = 1e-6  # Allow for some floating point error
    true_mean = np.mean(values)
    true_sample_variance = np.var(values, ddof=1)
    true_m2 = true_sample_variance * (len(values) - 1)
    assert stat.current_count == len(values), f"{stat.current_count} != {len(values)}"
    assert np.isclose(stat.mean(), true_mean), f"{stat.mean()} != {true_mean}"
    assert np.isclose(
        stat.variance(), true_sample_variance, atol=TOLERANCE
    ), f"{stat.variance()} != {true_sample_variance}"
    assert np.isclose(stat.current_M2, true_m2, atol=TOLERANCE), f"{stat.current_M2} != {true_m2}"


def test_add():
    # Test adding things to the statistics
    def _run_test(values):
        stat = OnlineStatistics()
        for i in values:
            stat.add(i)
        _assert_correct_stats(stat, values)

    # Positive integers
    _run_test(range(51))
    _run_test(range(10, 10000, 3))
    # Negative integers
    _run_test(range(-400, -300))
    # Positive/negative integers
    _run_test(list(range(4, 900, 2)) + list(range(-1000, -300, 7)))
    _run_test(list(range(-100, 100, 7)) + list(range(-100, 100, 2)))
    # Decimals
    _run_test(np.linspace(0, 1, 100))
    _run_test(np.logspace(-100, 3, 100))
    # Small lists
    _run_test([0, 1])
    _run_test([-1, 1])


def test_constructor():
    # Test default
    stat = OnlineStatistics()
    assert stat.current_count == 0
    assert stat.current_mean == stat.mean() == 0
    assert stat.current_M2 == 0

    # Test explicitly setting args
    stat = OnlineStatistics(current_count=1, current_mean=2, current_variance=3)
    assert stat.current_count == 1
    assert stat.current_mean == stat.mean() == 2
    assert stat.current_M2 == 0

    # Test M2
    stat = OnlineStatistics(current_count=10, current_mean=20, current_variance=30)
    assert stat.current_count == 10
    assert stat.current_mean == 20
    assert stat.current_M2 == 30 * (10 - 1)

    # Test getters/setters
    stat = OnlineStatistics(current_count=10, current_mean=20, current_variance=30)
    assert stat.mean() == 20
    assert stat.variance() == 30
    assert stat.standard_deviation() == np.sqrt(30)

    # Test fail cases
    with pytest.raises(ValueError) as _:
        # Negative count
        stat = OnlineStatistics(current_count=-1, current_mean=2, current_variance=3)
    with pytest.raises(ValueError) as _:
        # Negative variance
        stat = OnlineStatistics(current_count=1, current_mean=2, current_variance=-3)
    with pytest.raises(ValueError) as _:
        # Positive variance with 0 count
        stat = OnlineStatistics(current_count=0, current_mean=2, current_variance=1)
    with pytest.raises(ValueError) as _:
        # Can only compute variance with >1 observation
        stat = OnlineStatistics()
        stat.add(1)
        stat.variance()


def test_merge_pair():
    # Simulate two statistics being merged via `merge_pair``
    stat1 = OnlineStatistics()
    values1 = list(range(-300, 300, 4)) + list(range(400, 450))
    for i in values1:
        stat1.add(i)
    stat2 = OnlineStatistics()
    values2 = list(range(100, 150))
    for i in values2:
        stat2.add(i)
    merged_stat = OnlineStatistics.merge_pair(stat1, stat2)
    merged_stat_values = values1 + values2
    _assert_correct_stats(merged_stat, merged_stat_values)


def test_merge():
    # Simulate parallel statistics being merged via `merge`
    stats = []
    values = [
        np.linspace(-100, 100, 50),
        np.linspace(100, 200, 50),
        np.linspace(100, 150, 100),
        np.linspace(-10, 100, 100),
        np.linspace(10, 200, 3),
    ]
    for i in range(len(values)):
        stat = OnlineStatistics()
        for v in values[i]:
            stat.add(v)
        stats.append(stat)
    merged_stat = OnlineStatistics.merge(stats)
    _assert_correct_stats(merged_stat, np.concatenate(values))
