from __future__ import annotations

import logging
import math
import os
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


def set_up_logging(filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    logFormatter = logging.Formatter("%(asctime)s %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(filename, mode="w")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.INFO)

def inspect_patient_collection() -> None:
    parser = argparse.ArgumentParser(
        description="A tool for inspecting a piton patient_collection"
    )

    parser.add_argument(
        "extract_dir",
        type=str,
        help="Path of the folder to the ehr_ml extraction",
    )

    parser.add_argument(
        "patient_id", type=int, help="The patient id to inspect",
    )

    args = parser.parse_args()

    source_file = args.extract_dir
    timelines = TimelineReader(source_file)

    if args.patient_id is not None:
        patient_id = int(args.patient_id)
    else:
        patient_id = timelines.get_patient_ids()[0]

    patient = timelines.get_patient(patient_id)

    print(f"Patient: {patient.patient_id}")

    def value_to_str(value: Value) -> str:
        if value.type == ValueType.NONE:
            return ""
        elif value.type == ValueType.NUMERIC:
            return str(value.numeric_value)
        elif value.type == ValueType.TEXT:
            return value.text_value

    for i, event in enumerate(patient.events):
        print(f"--- Event {i}----")
        print(event.start_age)
        print(event.code + " " + value_to_str(event.value))