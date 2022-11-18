from __future__ import annotations

import math
import os
import pickle
import torch
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
    From https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance.
    """

    count: int
    current_mean: float
    variance: float

    def __init__(self, old_data: Optional[Dict[str, Any]] = None):
        """
        Initialize online statistics. Optionally takes the results of self.to_dict() to initialize from old data.
        """
        if old_data is None:
            old_data = {"count": 0, "current_mean": 0, "variance": 0}

        self.count = old_data["count"]
        self.current_mean = old_data["current_mean"]
        self.variance = old_data["variance"]

    def add(self, newValue: float) -> None:
        """
        Add an observation to the calculation.
        """
        self.count += 1
        delta = newValue - self.current_mean
        self.current_mean += delta / self.count
        delta2 = newValue - self.current_mean
        self.variance += delta * delta2

    def mean(self) -> float:
        """
        Return the current mean.
        """
        return self.current_mean

    def standard_deviation(self) -> float:
        """
        Return the current standard devation.
        """
        if self.count == 1:
            return math.sqrt(self.variance)

        return math.sqrt(self.variance / (self.count - 1))

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the data to a dictionary that can be encoded with JSON.
        Feed the resulting dictionary back into the constructor of this class to extract information from it.
        """
        return {
            "count": self.count,
            "current_mean": self.current_mean,
            "variance": self.variance,
        }


def get_gpus_with_minimum_free_memory(min_mem: float = 5, num_gpus: int = 8) -> List[int]:
    """Return a list of GPU devices with at least `min_mem` free memory is in GB."""
    devices = []
    for i in range(num_gpus):
        free, __ = torch.cuda.mem_get_info(i)
        if free >= min_mem * 1e9:
            devices.append(i)
    return devices


def save_to_file(object_to_save, path_to_file: str):
    """Save object to Pickle file."""
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "wb") as fd:
        pickle.dump(object_to_save, fd)

def load_from_file(path_to_file: str):
    """Load object from Pickle file."""
    with open(path_to_file, "rb") as fd:
        result = pickle.load(fd)
    return result
