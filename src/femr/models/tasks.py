from __future__ import annotations

import abc
import collections
import datetime
import functools
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import datasets
import meds
import numpy as np
import scipy.sparse
import torch

import femr.index
import femr.models.config
import femr.pat_utils
import femr.stat_utils


class Task(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_task_config(self) -> femr.models.config.FEMRTaskConfig: ...

    @abc.abstractmethod
    def start_batch(self) -> None: ...

    @abc.abstractmethod
    def start_patient(self, patient: meds.Patient, ontology: Optional[femr.ontology.Ontology]) -> None: ...

    @abc.abstractmethod
    def add_patient_labels(self, patient_label_offsets: List[int]) -> None: ...

    @abc.abstractmethod
    def needs_exact(self) -> bool: ...

    @abc.abstractmethod
    def add_event(
        self,
        current_date: datetime.datetime,
        next_date: Optional[datetime.datetime],
        next_features: Optional[Sequence[int]],
    ) -> int: ...

    @abc.abstractmethod
    def get_batch_data(self) -> Mapping[str, np.ndarray]: ...

    def cleanup(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        return batch


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

    def get_task_config(self) -> femr.models.config.FEMRTaskConfig:
        return femr.models.config.FEMRTaskConfig(task_type="labeled_patients")

    def filter_dataset(self, dataset: datasets.Dataset, index: femr.index.PatientIndex) -> datasets.Dataset:
        indices = [index.get_index(patient_id) for patient_id in self.label_map]
        return dataset.select(indices)

    def start_patient(self, patient: meds.Patient, _ontology: Optional[femr.ontology.Ontology]) -> None:
        self.current_labels = self.label_map[patient["patient_id"]]
        self.current_label_index = 0

    def needs_exact(self) -> bool:
        return True

    def start_batch(self) -> None:
        """LabeledPatientTask currently has no per label state."""
        pass

    def add_patient_labels(self, _patient_label_offsets: List[int]) -> None:
        """As there is no per label state, this is ignored"""
        pass

    def add_event(
        self,
        current_date: datetime.datetime,
        next_date: Optional[datetime.datetime],
        next_features: Optional[Sequence[int]] = None,
    ) -> int:
        has_label = False

        while True:
            if self.current_label_index == len(self.current_labels):
                break

            current_label = self.current_labels[self.current_label_index]

            is_valid = current_date <= current_label["prediction_time"]
            next_valid = next_date is not None and next_date <= current_label["prediction_time"]

            if next_valid:
                # Next one is valid, so break eary to give it a chance next time
                break

            if is_valid:
                has_label = True
                self.current_label_index += 1
            else:
                # The next label isn't valid, so we have to break here
                break

        if has_label:
            return 1
        else:
            return 0

    def get_batch_data(self) -> Mapping[str, np.ndarray]:
        return {}


class CLMBRTask(Task):
    def __init__(self, clmbr_vocab_size: int):
        self.clmbr_vocab_size = clmbr_vocab_size

    def get_task_config(self) -> femr.models.config.FEMRTaskConfig:
        return femr.models.config.FEMRTaskConfig(
            task_type="clmbr", task_kwargs=dict(clmbr_vocab_size=self.clmbr_vocab_size)
        )

    def start_patient(self, _patient: meds.Patient, _ontology: Optional[femr.ontology.Ontology]) -> None:
        self.per_patient_batch_labels: List[int] = []

    def needs_exact(self) -> bool:
        return False

    def start_batch(self) -> None:
        self.batch_labels: List[int] = []

    def add_patient_labels(self, patient_label_offsets: List[int]) -> None:
        self.batch_labels.extend([self.per_patient_batch_labels[i] for i in patient_label_offsets])

    def add_event(
        self,
        current_date: datetime.datetime,
        next_date: Optional[datetime.datetime],
        next_features: Optional[Sequence[int]] = None,
    ) -> int:
        if next_features is None:
            return 0

        if len(next_features) != 1:
            raise RuntimeError("Only supports one for right now")

        next_feature = next_features[0]

        if next_feature >= self.clmbr_vocab_size:
            return 0

        self.per_patient_batch_labels.append(next_feature)

        return 1

    def get_batch_data(self) -> Mapping[str, np.ndarray]:
        return {"labels": np.array(self.batch_labels, dtype=np.int32)}


def should_make_survival_prediction(current_date: datetime.datetime, next_date: Optional[datetime.datetime]):
    if next_date is None:
        return False

    if current_date == next_date or current_date.date() == next_date.date():
        return False

    return True


class SurvivalCalculator:
    def __init__(
        self, ontology: femr.ontology.Ontology, patient: meds.Patient, code_whitelist: Optional[Set[str]] = None
    ):
        self.survival_events = []
        self.final_date = patient["events"][-1]["time"]
        self.future_times = collections.defaultdict(list)

        for event in patient["events"]:
            codes = set()
            for measurement in event["measurements"]:
                for parent in ontology.get_all_parents(measurement["code"]):
                    if code_whitelist is None or parent in code_whitelist:
                        codes.add(parent)

            for code in codes:
                self.future_times[code].append(event["time"])
                self.survival_events.append((code, event["time"]))

        for v in self.future_times.values():
            v.reverse()

        self.survival_events.reverse()

    def get_future_events_for_time(
        self, time: datetime.datetime
    ) -> Tuple[datetime.timedelta, Mapping[str, datetime.timedelta]]:
        while len(self.survival_events) > 0 and self.survival_events[-1][1] <= time:
            code = self.survival_events[-1][0]
            vals = self.future_times[code]
            vals.pop()
            if len(vals) == 0:
                del self.future_times[code]

            self.survival_events.pop()

        delta = self.final_date - time
        return (delta, {k: v[-1] - time for k, v in self.future_times.items()})


def _prefit_motor_map(batch, *, tasks: List[str], ontology: femr.ontology.Ontology) -> Any:
    task_time_stats: List[Any] = [[0, 0, femr.stat_utils.OnlineStatistics()] for _ in range(len(tasks))]
    event_times = femr.stat_utils.ReservoirSampler(100_000)
    task_set = set(tasks)

    for patient_id, events in zip(batch["patient_id"], batch["events"]):
        patient = {"patient_id": patient_id, "events": events}

        calculator = SurvivalCalculator(ontology, patient, task_set)

        birth = femr.pat_utils.get_patient_birthdate(patient)

        for event, next_event in zip(patient["events"], patient["events"][1:]):
            if (event["time"] - birth).days <= 1:
                continue
            if should_make_survival_prediction(event["time"], next_event["time"]):
                censor_time, tte = calculator.get_future_events_for_time(event["time"])

                for i, task in enumerate(tasks):
                    if task in tte:
                        time = tte[task]
                        is_censored = False
                    else:
                        time = censor_time
                        is_censored = True

                    if is_censored:
                        task_time_stats[i][0] += 1
                    else:
                        event_times.add(time.total_seconds(), 1)
                        task_time_stats[i][1] += 1
                    task_time_stats[i][2].add(1, time.total_seconds())

    return (event_times, task_time_stats)


def _prefit_motor_agg(first: Any, second: Any) -> Any:
    for a, b in zip(first[1], second[1]):
        a[0] += b[0]
        a[1] += b[1]
        a[2].combine(b[2])
    first[0].combine(second[0])
    return first


class MOTORTask(Task):
    @classmethod
    def fit_pretraining_task_info(
        cls,
        dataset: datasets.Dataset,
        tokenizer: femr.models.tokenizer.FEMRTokenizer,
        num_tasks: int,
        num_bins: int,
        final_layer_size: int,
        num_proc: int = 1,
    ) -> MOTORTask:
        tasks = []
        for dict_entry in tokenizer.dictionary["vocab"]:
            if dict_entry["type"] == "code":
                tasks.append(dict_entry["code_string"])
                if len(tasks) == num_tasks:
                    break

        assert len(tasks) == num_tasks, "Could not find enough tasks in the provided tokenizer"

        length_samples, stats = femr.hf_utils.aggregate_over_dataset(
            dataset,
            functools.partial(_prefit_motor_map, tasks=tasks, ontology=tokenizer.ontology),
            _prefit_motor_agg,
            1_000,
            num_proc=num_proc,
        )

        time_bins = np.percentile(length_samples.samples, np.linspace(0, 100, num_bins + 1))
        time_bins[0] = 0
        time_bins[-1] = float("inf")
        time_bins = list(time_bins)

        task_data = []

        for task, task_stats in zip(tasks, stats):
            frac_events = task_stats[1] / (task_stats[0] + task_stats[1])
            rate = frac_events / task_stats[2].mean()
            task_data.append((task, rate))

        return MOTORTask(task_data, time_bins, final_layer_size)

    def __init__(self, pretraining_task_info: List[Tuple[str, float]], time_bins: List[float], final_layer_size: int):
        self.pretraining_task_info = pretraining_task_info
        self.time_bins = time_bins
        self.final_layer_size = final_layer_size

        self.pretraining_task_codes = set()
        self.task_to_index_map = {}
        for i, task in enumerate(self.pretraining_task_info):
            self.pretraining_task_codes.add(task[0])
            self.task_to_index_map[task[0]] = i

    def get_task_config(self) -> femr.models.config.FEMRTaskConfig:
        return femr.models.config.FEMRTaskConfig(
            task_type="motor",
            task_kwargs=dict(
                pretraining_task_info=self.pretraining_task_info,
                time_bins=self.time_bins,
                final_layer_size=self.final_layer_size,
            ),
        )

    def start_patient(self, patient: meds.Patient, ontology: Optional[femr.ontology.Ontology]) -> None:
        assert ontology
        self.calculator = SurvivalCalculator(ontology, patient, self.pretraining_task_codes)

        self.per_patient_censor_time: List[float] = []
        self.per_patient_time_sparse: Dict[str, List[float]] = {
            "data": [],
            "indices": [],
            "indptr": [0],
        }

    def needs_exact(self) -> bool:
        return False

    def start_batch(self) -> None:
        self.censor_time: List[float] = []

        self.time_sparse: Dict[str, List[float]] = {
            "data": [],
            "indices": [],
            "indptr": [0],
        }

    def add_patient_labels(self, patient_label_offsets: List[int]) -> None:
        """Add per-patient labels to the global task labels."""
        self.censor_time.extend([self.per_patient_censor_time[i] for i in patient_label_offsets])

        for index in patient_label_offsets:
            start = int(self.per_patient_time_sparse["indptr"][index])
            end = int(self.per_patient_time_sparse["indptr"][index + 1])

            self.time_sparse["data"].extend(self.per_patient_time_sparse["data"][start:end])
            self.time_sparse["indices"].extend(self.per_patient_time_sparse["indices"][start:end])
            self.time_sparse["indptr"].append(len(self.time_sparse["indices"]))

    def add_event(
        self,
        current_date: datetime.datetime,
        next_date: Optional[datetime.datetime],
        next_features: Optional[Sequence[int]] = None,
    ) -> int:
        if not should_make_survival_prediction(current_date, next_date):
            return 0

        censor_time, tte = self.calculator.get_future_events_for_time(current_date)

        if len(tte) == 0:
            return 0

        censor_seconds = censor_time.total_seconds()
        self.per_patient_censor_time.append(censor_seconds)

        for event_name, time in tte.items():
            j = self.task_to_index_map[event_name]
            seconds = time.total_seconds()

            self.per_patient_time_sparse["data"].append(seconds)
            self.per_patient_time_sparse["indices"].append(j)

        self.per_patient_time_sparse["indptr"].append(len(self.per_patient_time_sparse["data"]))

        return 1

    def get_batch_data(self) -> Mapping[str, np.ndarray]:
        def h(a, dtype):
            return {
                "data": np.array(a["data"], dtype=dtype),
                "indices": np.array(a["indices"], dtype=np.int32),
                "indptr": np.array(a["indptr"], dtype=np.int32),
            }

        return {
            "censor_time": np.array(self.censor_time, dtype=np.float32),
            "time_sparse": h(self.time_sparse, dtype=np.float32),
        }

    def cleanup(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        num_time_bins = len(self.time_bins) - 1
        num_tasks = len(self.pretraining_task_info)
        num_indices = len(batch["censor_time"])

        def h(a):
            shape = (num_indices, num_tasks)
            a = {k: v.numpy() for k, v in batch[a].items()}
            s = scipy.sparse.csr_array((a["data"], a["indices"], a["indptr"]), shape=shape)
            return torch.from_numpy(s.toarray())

        time = h("time_sparse")

        log_time = torch.zeros(size=(num_time_bins, num_indices, num_tasks), dtype=torch.float16)
        is_event = torch.zeros(size=(num_time_bins, num_indices, num_tasks), dtype=torch.bool)

        is_event_global = time != 0

        def inf_log(arr):
            mask = arr != 0
            arr[mask] = torch.log2(arr[mask])
            arr[~mask] = -torch.inf

            return arr

        for i, (start, end) in enumerate(zip(self.time_bins, self.time_bins[1:])):
            censor_time_in_bin = torch.unsqueeze(torch.clip(batch["censor_time"] - start, 0, float(end - start)), -1)
            event_time_in_bin = torch.clip(time - start, 0, float(end - start))
            time_in_bin = torch.where(is_event_global, event_time_in_bin, censor_time_in_bin)
            log_time[i, :] = inf_log(time_in_bin)
            is_event[i, :] = is_event_global & (start <= time) & (time < end)

        log_time = torch.transpose(log_time, 0, 1).contiguous()
        is_event = torch.transpose(is_event, 0, 1).contiguous()

        return {"is_event": is_event, "log_time": log_time}
