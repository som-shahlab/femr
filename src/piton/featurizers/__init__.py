from __future__ import annotations

import copy
import math
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

T = TypeVar("T")


class Dictionary(Generic[T]):
    mapper: Dict[T, int]
    reverse_mapper: Dict[int, T]

    def __init__(self, old_dict: Optional[Dict[str, Any]] = None):
        """
        Create a dictionary which is used for mapping words back and forth to integers.
        """
        if old_dict is not None:
            self.from_dict(old_dict)
        else:
            self.mapper = {}
            self.reverse_mapper = {}

    def __contains__(self, item: T) -> bool:
        return item in self.mapper

    def __len__(self) -> int:
        return len(self.mapper)

    def add(self, word: T) -> int:
        """
        Add a word to the dictionary.
        Args:
            word (str): The word to add
        Returns:
            The integer index for that word
        """
        result = self.mapper.get(word)

        if result is None:
            next_index = len(self.mapper)
            self.mapper[word] = next_index
            self.reverse_mapper[next_index] = word
            return next_index
        else:
            return result

    def transform(self, word: T) -> int:
        """
        Transform a word to an integer.
        Args:
            word (str): The word to transform
        Returns:
            The integer index for that word
        """

        return self.mapper[word]

    def transform_all(self, words: List[T]) -> List[int]:
        return [self.transform(word) for word in words if word in self]

    def get_word(self, index: int) -> Optional[T]:
        """
        Transforms an integer into the corresponding word.
        Args:
            index (int): The index to map
        Returns:
            The string word for that index
        """
        if index in self.reverse_mapper:
            return self.reverse_mapper[index]
        else:
            return None

    def get_words(self) -> List[T]:
        return list(self.mapper.keys())

    def get_items(self) -> List[Tuple[T, int]]:
        return list(self.mapper.items())

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the dictionary to a Python dict for serialization purposes
        """
        return {"values": list(self.mapper.items())}

    def from_dict(self, old_dict: Dict[str, Any]) -> None:
        """
        Load the dictionary from a dict obtained with to_dict
        """

        self.mapper = {}
        self.reverse_mapper = {}

        for word, idx in old_dict["values"]:
            self.mapper[word] = idx
            self.reverse_mapper[idx] = word


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
        assert (
            current_count >= 0 and current_variance >= 0
        ), "Cannot specify negative values for `current_count` or `current_variance`."
        self.current_count: int = current_count
        self.current_mean: float = current_mean
        if current_count == 0 and current_variance == 0:
            self.current_M2: float = 0
        elif current_count > 0:
            self.current_M2: float = current_variance * (current_count - 1)
        else:
            raise ValueError(
                "Cannot specify `current_variance` with a value > 0 without specifying `current_count` with a value > 0."
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
            raise ValueError(
                f"Cannot compute variance with only {self.current_count} observations."
            )

        return self.current_M2 / (self.current_count - 1)

    def standard_deviation(self) -> float:
        """
        Return the current standard devation.
        """
        return math.sqrt(self.variance())

    @classmethod
    def merge_pair(
        cls, stats1: OnlineStatistics, stats2: OnlineStatistics
    ) -> OnlineStatistics:
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
        M2 = (
            stats1.current_M2
            + stats2.current_M2
            + delta**2 * stats1.current_count * stats2.current_count / count
        )
        return OnlineStatistics(count, mean, M2 / (count - 1))

    @classmethod
    def merge(cls, stats_list: List[OnlineStatistics]) -> OnlineStatistics:
        """
        Merge a list of online statistics.
        """
        if len(stats_list) == 0:
            raise ValueError("Cannot merge an empty list of statistics.")
        unmerged_stats = copy.deepcopy(stats_list)
        stats1: OnlineStatistics = unmerged_stats.pop(0)
        while len(unmerged_stats) > 0:
            stats2: OnlineStatistics = unmerged_stats.pop(0)
            stats1 = cls.merge_pair(stats1, stats2)
        return stats1
