from __future__ import annotations

import numpy as np
import scipy
import sklearn.metrics

import piton.metrics

def baseline_c_statistic(times, is_censor, time_bins, hazards):
    total_auroc = 0

    surv = 1

    total_weight = 0

    for time in sorted(list(set(times))):
        mask = (((times > time) & (is_censor != 0)) | ((times >= time) & (is_censor == 0))).astype(bool)

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

            auroc = sklearn.metrics.roc_auc_score(current_correct, current_risks)

            total_auroc += auroc * weight
            total_weight += weight

    expected = total_auroc / total_weight
    return expected

def baseline_c_statistic_weighted(times, is_censor, time_bins, hazards, weights):
    total_auroc = 0

    surv = 1

    total_weight = 0

    for time in sorted(list(set(times))):
        mask = (((times > time) & (is_censor != 0)) | ((times >= time) & (is_censor == 0))).astype(bool)

        for i, val in enumerate(time_bins):
            if time < val:
                i -= 1
                break

        risks = hazards[:, i]

        correct = (times == time) & (is_censor == 0)

        f = weights[correct].sum() / weights[mask].sum()
        factor = surv * f
        surv *= 1 - f

        weight = factor * surv

        if correct.sum() == mask.sum() or correct.sum() == 0:
            assert weight == 0
        else:
            current_risks = risks[mask]
            current_correct = correct[mask]

            auroc = sklearn.metrics.roc_auc_score(current_correct, current_risks, sample_weight=weights[mask])

            total_auroc += auroc * weight
            total_weight += weight

    expected = total_auroc / total_weight
    return expected

def test_c_statistic_weighted():
    N = 200
    B = 8

    END = 40

    np.random.seed(12313)

    time_bins = np.linspace(0, END * 0.8, num=B).astype(np.float64)

    times = np.random.randint(0, END, size=(N,)).astype(np.float64)
    is_censor = np.random.binomial(1, 0.7, size=(N,)).astype(bool)
    hazards = np.random.normal(size=(N, B)).astype(np.float64)

    weights = np.random.randint(1, 5, size=(N,))

    for i in range(N):
        if times[i] > 10 and not is_censor[i]:
            hazards[i, :] += 1
        if times[i] > 50 and not is_censor[i]:
            hazards[i, :] += 3

    flattened = np.sum(weights)
    flat_times = np.zeros(shape=(flattened,), dtype=np.float64)
    flat_is_censor = np.zeros(shape=(flattened,), dtype=bool)
    flat_hazards = np.zeros(shape=(flattened, B), dtype=np.float64)
    current_index = 0
    for i in range(N):
        end_index = current_index + weights[i]
        flat_times[current_index:end_index] = times[i]
        flat_is_censor[current_index:end_index] = is_censor[i]
        flat_hazards[current_index:end_index, :] = hazards[i, :]
        current_index = end_index

    bad_one = baseline_c_statistic(times, is_censor, time_bins, hazards)
    expected = baseline_c_statistic(flat_times, flat_is_censor, time_bins, flat_hazards)
    expected2 = baseline_c_statistic_weighted(times, is_censor, time_bins, hazards, weights)
    assert bad_one != expected
    assert expected == expected2
    actual = piton.metrics.compute_c_statistic(times, is_censor, time_bins, hazards, weights)[0]

    assert actual == expected

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


    expected = baseline_c_statistic(times, is_censor, time_bins, hazards)
    expected2 = baseline_c_statistic_weighted(times, is_censor, time_bins, hazards, np.ones_like(times))
    assert expected == expected2
    actual = piton.metrics.compute_c_statistic(times, is_censor, time_bins, hazards)[0]

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

    dummy = piton.metrics.compute_calibration(dummy_probs, np.zeros_like(dummy_probs), B)

    assert np.allclose(expected, dummy, atol=0.01)

    valid = piton.metrics.compute_calibration(valid_probs, c, B)

    assert np.allclose(expected, valid, atol=0.01)

    invalid = piton.metrics.compute_calibration(invalid_probs, c, B)

    assert not np.allclose(expected, invalid, atol=0.01)


def test_breslow():
    np.random.seed(341231)

    event_h = 5
    censor_h = 8

    size = 100000

    T = scipy.stats.expon.rvs(scale=1 / event_h, size=size)
    C = scipy.stats.expon.rvs(scale=1 / censor_h, size=size)

    t = np.minimum(T, C)
    c = C < T

    hazard = np.ones(shape=(size, 2))
    hazard[:, 0] = 5
    hazard[:, 0] = 7

    times = [0, 1]

    breslow = piton.metrics.estimate_breslow(t[: size // 10], c[: size // 10], times, hazard[: size // 10, :])
    cdf = piton.metrics.apply_breslow(t, times, hazard, breslow)

    valid_probs = scipy.stats.expon.cdf(t, scale=1 / event_h)

    assert np.allclose(cdf, valid_probs, atol=0.03)
