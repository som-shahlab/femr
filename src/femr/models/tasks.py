from __future__ import annotations

import abc
import collections
import datetime
from typing import Any, List, Mapping, Sequence, Tuple

import datasets
import meds
import numpy as np

import femr.index
import femr.models.transformer


class Task(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_task_config(self) -> femr.models.transformer.FEMRTaskConfig:
        ...

    @abc.abstractmethod
    def start_batch(self) -> None:
        ...

    @abc.abstractmethod
    def start_patient(self, patient: meds.Patient) -> None:
        ...

    @abc.abstractmethod
    def needs_exact(self) -> bool:
        ...

    @abc.abstractmethod
    def add_event(
        self, current_date: datetime.datetime, next_date: datetime.datetime, next_features: Sequence[int]
    ) -> int:
        ...

    @abc.abstractmethod
    def get_batch_data(self) -> Mapping[str, np.ndarray]:
        ...


class LabeledPatientTask(Task):
    def __init__(self, labels: Sequence[meds.Label]):
        super().__init__()

        self.label_map: Mapping[int, Any] = collections.defaultdict(list)
        for label in labels:
            row_without_patient_id = dict(label)
            del row_without_patient_id["patient_id"]
            self.label_map[label["patient_id"]].append(row_without_patient_id)

        for k, v in self.label_map.items():
            v.sort(key=lambda a: a["prediction_time"])

    def get_task_config(self) -> femr.models.transformer.FEMRTaskConfig:
        return femr.models.transformer.FEMRTaskConfig(task_type="labeled_patients")

    def filter_dataset(self, dataset: datasets.Dataset, index: femr.index.PatientIndex) -> datasets.Dataset:
        indices = [index.get_index(patient_id) for patient_id in self.label_map]
        return dataset.select(indices)

    def start_patient(self, patient: meds.Patient) -> None:
        self.current_labels = self.label_map[patient["patient_id"]]
        self.current_label_index = 0
        self.patient_id = patient["patient_id"]

    def needs_exact(self) -> bool:
        return True

    def start_batch(self) -> None:
        self.patient_ids: List[int] = []
        self.prediction_timestamps: List[float] = []

    def add_event(
        self, current_date: datetime.datetime, next_date: datetime.datetime, next_features: Sequence[int]
    ) -> int:
        num_added = 0
        while True:
            if self.current_label_index == len(self.current_labels):
                return num_added

            current_label = self.current_labels[self.current_label_index]

            is_valid = current_date <= current_label["prediction_time"]
            next_valid = next_date is not None and next_date <= current_label["prediction_time"]

            assert is_valid, (
                "We have labels that appear to be before birth? "
                + f"{self.patient_id} {current_label} {current_date} {next_date}"
            )

            if next_valid:
                # Next one is valid, so break early to give it a chance next time
                return num_added
            else:
                self.patient_ids.append(self.patient_id)
                self.prediction_timestamps.append(current_label["prediction_time"].timestamp())
                num_added += 1
                self.current_label_index += 1

        assert False, "Should never reach end"

    def get_batch_data(self) -> Mapping[str, np.ndarray]:
        return {
            "patient_ids": np.array(self.patient_ids, dtype=np.int64),
            "prediction_timestamps": np.array(self.prediction_timestamps, dtype=np.int64),
        }


class CLMBRTask(Task):
    def __init__(self, clmbr_vocab_size: int):
        self.clmbr_vocab_size = clmbr_vocab_size

    def get_task_config(self) -> femr.models.transformer.FEMRTaskConfig:
        return femr.models.transformer.FEMRTaskConfig(task_type="clmbr", clmbr_vocab_size=self.clmbr_vocab_size)

    def start_patient(self, patient: meds.Patient) -> None:
        pass

    def needs_exact(self) -> bool:
        return False

    def start_batch(self) -> None:
        self.batch_labels: List[int] = []

    def add_event(
        self, current_date: datetime.datetime, next_date: datetime.datetime, next_features: Sequence[int]
    ) -> int:
        if len(next_features) == 0:
            return 0

        if len(next_features) != 1:
            raise RuntimeError("Only supports one for right now")

        next_feature = next_features[0]

        if next_feature >= self.clmbr_vocab_size:
            return 0

        self.batch_labels.append(next_feature)

        return 1

    def get_batch_data(self) -> Mapping[str, np.ndarray]:
        return {"labels": np.array(self.batch_labels, dtype=np.int32)}


class MOTORTask(Task):
    @classmethod
    def train_pretraining_task_info(cls, dataset: datasets.Dataset, num_tasks: int, num_bins: int) -> None:


    def __init__(self, pretraining_task_info: List[Tuple[str, float]]):
        self.pretraining_task_info = pretraining_task_info

    def get_task_config(self) -> femr.models.transformer.FEMREncoderLayer:
        return femr.models.transformer.FEMRTaskConfig(task_type="motor")

    def start_patient(self, patient: meds.Patient) -> None:
        pass

    def needs_exact(self) -> bool:
        return False

    def start_batch(self) -> None:
        self.batch_labels: List[int] = []

    def add_event(
        self, current_date: datetime.datetime, next_date: datetime.datetime, next_features: Sequence[int]
    ) -> int:
        if len(next_features) == 0:
            return 0

        if len(next_features) != 1:
            raise RuntimeError("Only supports one for right now")

        next_feature = next_features[0]

        self.batch_labels.append(next_feature)

        return 1

    def get_batch_data(self) -> Mapping[str, np.ndarray]:
        return {"labels": np.array(self.batch_labels, dtype=np.int32)}
