from __future__ import annotations

import datasets

import femr.hf_utils


def map_index(batch, indices):
    return list(zip(batch["patient_id"], indices))


def agg_index(index1, index2):
    index1.extend(index2)
    return index1


class PatientIndex:
    def __init__(self, dataset: datasets.Dataset, num_proc: int = 1):
        data = femr.hf_utils.aggregate_over_dataset(
            dataset, map_index, agg_index, num_proc=num_proc, batch_size=1_000, with_indices=True
        )
        self.index_map = dict(data)

    def get_patient_ids(self):
        return self.index_map.keys()

    def get_index(self, patient_id):
        return self.index_map[patient_id]

    def filter_dataset(self, dataset, patient_ids):
        return dataset.select([self.get_index(patient_id) for patient_id in patient_ids])
