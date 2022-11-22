from __future__ import annotations

import piton.extension.metrics
import numpy as np
import sklearn.metrics


def test_c_statistic():
    N = 200
    B = 8

    END = 40

    np.random.seed(1231)

    time_bins = np.linspace(0, END * 0.8, num=B).astype(np.float64)

    times = np.random.randint(0, END, size=(N,)).astype(np.float64)
    is_censor = np.random.binomial(1, 0.7, size=(N,)).astype(bool)
    hazards = np.random.normal(size=(N, B)).astype(np.float64)

    for i in range(N):
        if times[i] > 10 and not is_censor[i]:
            hazards[i, :] += 1
        if times[i] > 50 and not is_censor[i]:
            hazards[i, :] += 3

    total = 0
    total_auroc = 0

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
        if correct.sum() == 0:
            continue
        current_risks = risks[mask]
        current_correct = correct[mask]

        if correct.sum() == mask.sum():
            auroc = 1
        else:
            auroc = sklearn.metrics.roc_auc_score(
                current_correct, current_risks
            )
        if False:
            print(
                time,
                current_risks,
                current_correct,
                mask.sum() - current_correct.sum(),
                current_correct.sum(),
                auroc * current_correct.sum(),
            )
        if False:
            print(
                time,
                auroc,
                current_correct.sum(),
                mask.sum() - current_correct.sum(),
            )
            if time == 0:
                print(risks[correct][0])
                print((current_risks < risks[correct][0]).sum())
                print((current_risks > risks[correct][0]).sum())
        total += current_correct.sum()
        total_auroc += auroc * current_correct.sum()

    expected = total_auroc / total

    actual = piton.extension.metrics.compute_c_statistic(
        times, is_censor, time_bins, hazards
    )

    assert actual == expected
