from __future__ import annotations

import piton.metrics
import numpy as np
import scipy
import sklearn.metrics


def test_c_statistic():
    N = 200
    B = 8

    END = 40

    np.random.seed(12313)

    time_bins = np.linspace(0, END * 0.8, num=B).astype(np.float64)

    times = np.random.randint(0, END, size=(N,)).astype(np.float64)
    is_censor = np.random.binomial(1, 0.7, size=(N,)).astype(bool)
    hazards = np.random.normal(size=(N, B)).astype(np.float64)

    for i in range(N):
        if times[i] > 10 and not is_censor[i]:
            hazards[i, :] += 1
        if times[i] > 50 and not is_censor[i]:
            hazards[i, :] += 3

    total_auroc = 0

    surv = 1

    total_weight = 0

    for time in sorted(list(set(times))):
        mask = (
            ((times > time) & (is_censor != 0))
            | ((times >= time) & (is_censor == 0))
        ).astype(bool)

        for i, val in enumerate(time_bins):
            if time < val:
                i -= 1
                break

        risks = hazards[:, i]

        correct = (times == time) & (is_censor == 0)

        f = correct.sum() / mask.sum()
        factor = surv * f
        surv *= 1 - f

        weight = factor * surv

        if correct.sum() == mask.sum() or correct.sum() == 0:
            assert weight == 0
        else:
            current_risks = risks[mask]
            current_correct = correct[mask]

            auroc = sklearn.metrics.roc_auc_score(
                current_correct, current_risks
            )

            total_auroc += auroc * weight
            total_weight += weight

    expected = total_auroc / total_weight

    actual = piton.metrics.compute_c_statistic(
        times, is_censor, time_bins, hazards
    )

    assert actual == expected


def test_calibration():
    np.random.seed(341231)

    event_h = 5
    censor_h = 8

    size = 100000

    T = scipy.stats.expon.rvs(scale=1 / event_h, size=size)
    C = scipy.stats.expon.rvs(scale=1 / censor_h, size=size)

    t = np.minimum(T, C)
    c = C < T

    dummy_probs = scipy.stats.expon.cdf(T, scale=1 / event_h)
    valid_probs = scipy.stats.expon.cdf(t, scale=1 / event_h)
    invalid_probs = scipy.stats.expon.cdf(t, scale=1 / (2 * event_h))

    B = 10

    expected = np.ones(shape=(B,)) / B

    dummy = piton.metrics.compute_calibration(
        dummy_probs, np.zeros_like(dummy_probs), B
    )

    assert np.allclose(expected, dummy, atol=0.01)

    valid = piton.metrics.compute_calibration(valid_probs, c, B)

    assert np.allclose(expected, valid, atol=0.01)

    invalid = piton.metrics.compute_calibration(invalid_probs, c, B)

    assert not np.allclose(expected, invalid, atol=0.01)
