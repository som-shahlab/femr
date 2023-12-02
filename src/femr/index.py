from __future__ import annotations

import datasets
import msgpack

import femr.hf_utils


def agg_index(index1, index2):
    index1.extend(index2)
    return index1


def map_index(batch, indices):
    return list(zip(batch["patient_id"], indices))


class PatientIndex:
    def __init__(self, index_map):
        self.index_map = index_map

    def get_index(self, patient_id):
        return self.index_map[patient_id]

    @classmethod
    def load(cls, path: str) -> PatientIndex:
        with open(path, "rb") as f:
            data = msgpack.load(f)
            return PatientIndex({a: b for a, b in data})

    @classmethod
    def create_from_dataset(cls, dataset: datasets.Dataset, num_proc: int = 1) -> PatientIndex:
        data = femr.hf_utils.aggregate_over_dataset(
            dataset, map_index, agg_index, num_proc=num_proc, batch_size=1_000, with_indices=True
        )
        return PatientIndex({a: b for a, b in data})

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            data = list(self.index_map.items())
            msgpack.dump(data, f)
