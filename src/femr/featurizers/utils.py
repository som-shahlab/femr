from __future__ import annotations

import math
import copy
from typing import List

class OnlineStatistics:
    """
    A class for computing online statistics such as mean and variance.
    Uses Welford's online algorithm.
    From https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm.

    NOTE: The variance we calculate is the sample variance, not the population variance.
    """

    def __init__(
        self,
        current_count: int = 0,
        current_mean: float = 0,
        current_variance: float = 0,
    ):
        """
        Initialize online statistics.
            `mean` accumulates the mean of the entire dataset
            `count` aggregates the number of samples seen so far
            `current_M2` aggregates the squared distances from the mean
        """
        if not (current_count >= 0 and current_variance >= 0):
            raise ValueError(
                "Must set `current_count` and `current_variance` to be non-negative."
                f"You specified `current_count` = {current_count} and `current_variance` = {current_variance}."
            )
        self.current_count: int = current_count
        self.current_mean: float = current_mean
        if current_count == 0 and current_variance == 0:
            self.current_M2 = 0.0
        elif current_count > 0:
            self.current_M2 = current_variance * (current_count - 1)
        else:
            raise ValueError(
                "Cannot specify `current_variance` with a value > 0"
                "without specifying `current_count` with a value > 0. "
                f"You specified `current_count` = {current_count} and `current_variance` = {current_variance}."
            )

    def add(self, newValue: float) -> None:
        """
        Add an observation to the calculation using Welford's online algorithm.

        Taken from: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        """
        self.current_count += 1
        delta: float = newValue - self.current_mean
        self.current_mean += delta / self.current_count
        delta2: float = newValue - self.current_mean
        self.current_M2 += delta * delta2

    def mean(self) -> float:
        """
        Return the current mean.
        """
        return self.current_mean

    def variance(self) -> float:
        """
        Return the current sample variance.
        """
        if self.current_count < 2:
            raise ValueError(f"Cannot compute variance with only {self.current_count} observations.")

        return self.current_M2 / (self.current_count - 1)

    def standard_deviation(self) -> float:
        """
        Return the current standard devation.
        """
        return math.sqrt(self.variance())

    @classmethod
    def merge_pair(cls, stats1: OnlineStatistics, stats2: OnlineStatistics) -> OnlineStatistics:
        """
        Merge two sets of online statistics using Chan's parallel algorithm.

        Taken from: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        if stats1.current_count == 0:
            return stats2
        elif stats2.current_count == 0:
            return stats1

        count: int = stats1.current_count + stats2.current_count
        delta: float = stats2.current_mean - stats1.current_mean
        mean: float = stats1.current_mean + delta * stats2.current_count / count
        M2 = stats1.current_M2 + stats2.current_M2 + delta**2 * stats1.current_count * stats2.current_count / count
        return OnlineStatistics(count, mean, M2 / (count - 1))

    @classmethod
    def merge(cls, stats_list: List[OnlineStatistics]) -> OnlineStatistics:
        """
        Merge a list of online statistics.
        """
        if len(stats_list) == 0:
            raise ValueError("Cannot merge an empty list of statistics.")
        unmerged_stats: List[OnlineStatistics] = copy.deepcopy(stats_list)
        # Run tree reduction to merge together all pairs of statistics
        # in a numerically stable way
        #   Example: 1 2 3 4 5 -> 3 7 5 -> 10 5 -> 15
        while len(unmerged_stats) > 1:
            merged_stats: List[OnlineStatistics] = []
            for i in range(0, len(unmerged_stats), 2):
                if i + 1 < len(unmerged_stats):
                    # If there's another stat after this one, merge them
                    merged_stats.append(cls.merge_pair(unmerged_stats[i], unmerged_stats[i + 1]))
                else:
                    # We've reached the end of our list, so just add the last stat back
                    merged_stats.append(unmerged_stats[i])
            unmerged_stats = merged_stats
        assert len(unmerged_stats) == 1, f"Should only have one stat left after merging, not ({len(unmerged_stats)})."
        return unmerged_stats[0]
