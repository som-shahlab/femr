from __future__ import annotations

import csv
import dataclasses
import hashlib
import struct
from typing import Callable, List

import datasets

import femr.index


@dataclasses.dataclass
class PatientSplit:
    train_patient_ids: List[int]
    test_patient_ids: List[int]

    def save_to_csv(self, fname: str):
        with open(fname, "w") as f:
            writer = csv.DictWriter(f, ("patient_id", "split_name"))
            writer.writeheader()
            for train in self.train_patient_ids:
                writer.writerow({"patient_id": train, "split_name": "train"})
            for test in self.test_patient_ids:
                writer.writerow({"patient_id": test, "split_name": "test"})

    @classmethod
    def load_from_csv(cls, fname: str):
        train_patient_ids: List[int] = []
        test_patient_ids: List[int] = []
        with open(fname, "r") as f:
            for row in csv.DictReader(f):
                if row["split_name"] == "train":
                    train_patient_ids.append(int(row["patient_id"]))
                else:
                    test_patient_ids.append(int(row["patient_id"]))

        return PatientSplit(train_patient_ids=train_patient_ids, test_patient_ids=test_patient_ids)

    def split_dataset(self, dataset: datasets.Dataset, index: femr.index.PatientIndex) -> datasets.DatasetDict:
        train_indices = [index.get_index(patient_id) for patient_id in self.train_patient_ids]
        test_indices = [index.get_index(patient_id) for patient_id in self.test_patient_ids]
        return datasets.DatasetDict(
            {
                "train": dataset.select(train_indices),
                "test": dataset.select(test_indices),
            }
        )


def generate_hash_split(patient_ids: List[int], seed: int, frac_test: float = 0.15) -> PatientSplit:
    train_patient_ids = []
    test_patient_ids = []

    for patient_id in patient_ids:
        # Convert the integer to bytes
        value_bytes = struct.pack(">q", seed) + struct.pack(">q", patient_id)

        # Calculate SHA-256 hash
        sha256_hash = hashlib.sha256(value_bytes).hexdigest()

        # Convert the hexadecimal hash to an integer
        hash_int = int(sha256_hash, 16)

        # Take the modulus
        result = hash_int % (2**16)
        if result <= frac_test * (2**16):
            test_patient_ids.append(patient_id)
        else:
            train_patient_ids.append(patient_id)

    return PatientSplit(train_patient_ids=train_patient_ids, test_patient_ids=test_patient_ids)


def generate_split(patient_ids: List[int], is_test_set_fn: Callable[[int], bool]) -> PatientSplit:
    """Generates a patient split based on a user-defined function.

    This function categorizes each patient ID as either 'test' or 'train' based on
    the user-defined function's return value.

    Args:
        patient_ids (List[int]): A list of patient IDs.
        is_test_set_fn (Callable[[int], bool]): A function that takes a patient ID
            and returns True if it belongs to the test set, otherwise False.

    Returns:
        PatientSplit: A dataclass instance containing lists of train and test patient IDs.

    """
    train_patient_ids: List[int] = []
    test_patient_ids: List[int] = []

    for patient_id in patient_ids:
        if is_test_set_fn(patient_id):
            test_patient_ids.append(patient_id)
        else:
            train_patient_ids.append(patient_id)

    return PatientSplit(train_patient_ids=train_patient_ids, test_patient_ids=test_patient_ids)
