from __future__ import annotations

import csv
import dataclasses
import hashlib
import struct
from typing import List


@dataclasses.dataclass
class SubjectSplit:
    train_subject_ids: List[int]
    test_subject_ids: List[int]

    def save_to_csv(self, fname: str):
        with open(fname, "w") as f:
            writer = csv.DictWriter(f, ("subject_id", "split_name"))
            writer.writeheader()
            for train in self.train_subject_ids:
                writer.writerow({"subject_id": train, "split_name": "train"})
            for test in self.test_subject_ids:
                writer.writerow({"subject_id": test, "split_name": "test"})

    @classmethod
    def load_from_csv(cls, fname: str):
        train_subject_ids: List[int] = []
        test_subject_ids: List[int] = []
        with open(fname, "r") as f:
            for row in csv.DictReader(f):
                if row["split_name"] == "train":
                    train_subject_ids.append(int(row["subject_id"]))
                else:
                    test_subject_ids.append(int(row["subject_id"]))

        return SubjectSplit(train_subject_ids=train_subject_ids, test_subject_ids=test_subject_ids)


def generate_hash_split(subject_ids: List[int], seed: int, frac_test: float = 0.15) -> SubjectSplit:
    train_subject_ids = []
    test_subject_ids = []

    for subject_id in subject_ids:
        # Convert the integer to bytes
        value_bytes = struct.pack(">q", seed) + struct.pack(">q", subject_id)

        # Calculate SHA-256 hash
        sha256_hash = hashlib.sha256(value_bytes).hexdigest()

        # Convert the hexadecimal hash to an integer
        hash_int = int(sha256_hash, 16)

        # Take the modulus
        result = hash_int % (2**16)
        if result <= frac_test * (2**16):
            test_subject_ids.append(subject_id)
        else:
            train_subject_ids.append(subject_id)

    return SubjectSplit(train_subject_ids=train_subject_ids, test_subject_ids=test_subject_ids)
