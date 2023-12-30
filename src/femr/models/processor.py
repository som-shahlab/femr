from __future__ import annotations

import datetime
import functools
import random
from typing import List, Tuple

import datasets
import numpy as np

import femr.hf_utils
import femr.models.tokenizer
import femr.pat_utils


def map_length_stats(batch, indices, *, processor, max_length):
    lengths = []

    for patient_index, patient_id, events in zip(indices, batch["patient_id"], batch["events"]):
        total_events = 0
        for event in events:
            for measurement in event["measurements"]:
                total_events += 1

        patient = {
            "patient_id": patient_id,
            "events": events,
        }
        data = processor.convert_patient(patient)
        if data["transformer"]["label_indices"].shape[0] == 0:
            continue
        if data["needs_exact"]:
            current_start = 0
            current_end = 0
            for label_index in data["transformer"]["label_indices"]:
                if (label_index - current_start + 1) >= max_length:
                    lengths.append((patient_index, current_start, current_end - current_start + 1))
                    current_start = label_index - max_length + 1
                    current_end = label_index
                else:
                    current_end = label_index

            lengths.append((patient_index, current_start, current_end - current_start + 1))
        else:
            last_index = data["transformer"]["label_indices"][-1]
            length = min(max_length, last_index + 1)
            lengths.append((patient_index, last_index + 1 - length, length))
    return lengths


def agg_length_stats(lengths1, lengths2):
    lengths1.extend(lengths2)

    return lengths1


class BatchCreator:
    def __init__(self, tokenizer, task=None):
        self.tokenizer = tokenizer
        self.task = task

    def start_batch(self):
        self.patient_ids = []
        self.offsets = []
        self.patient_lengths = []

        if not self.tokenizer.is_hierarchical:
            self.tokens = []
        else:
            self.hierarchical_tokens = []
            self.hierarchical_weights = []
            self.token_indices = [0]

        self.valid_tokens = []

        self.ages = []
        self.normalized_ages = []
        self.timestamps = []

        self.label_indices = []

        if self.task is not None:
            self.task.start_batch()

    def add_patient(self, patient, offset, max_patient_length=None):
        self.patient_ids.append(patient["patient_id"])
        self.offsets.append(offset)

        current_date = None
        last_time = None

        if self.task is not None:
            self.task.start_patient(patient, self.tokenizer.ontology)

        start_index = len(self.ages)
        patient_length_index = 0

        needs_exact = True
        if self.task and not self.task.needs_exact():
            needs_exact = False

        birth = femr.pat_utils.get_patient_birthdate(patient)

        for event in patient["events"]:
            if event["time"].date() != current_date:
                current_date = event["time"].date()
                codes_seen_today = set()

            for measurement in event["measurements"]:
                features, weights = self.tokenizer.get_feature_codes(measurement)
                if len(features) == 0:
                    continue
                if all(feature in codes_seen_today for feature in features):
                    continue

                if patient_length_index < offset:
                    patient_length_index += 1
                    continue

                codes_seen_today |= set(features)

                if (
                    (self.task is not None)
                    and (last_time is not None)
                    and (needs_exact or (last_time - birth).days > 1)
                ):
                    num_added = self.task.add_event(last_time, event["time"], features)
                    for i in range(num_added):
                        self.label_indices.append(len(self.ages) - 1)

                if max_patient_length is not None and (patient_length_index - offset >= max_patient_length):
                    break

                if not self.tokenizer.is_hierarchical:
                    assert len(features) == 1
                    self.tokens.append(features[0])
                else:
                    self.hierarchical_tokens.extend(features)
                    self.hierarchical_weights.extend(weights)
                    self.token_indices.append(len(self.hierarchical_tokens))

                self.valid_tokens.append(True)
                self.ages.append((event["time"] - birth) / datetime.timedelta(days=1))
                self.normalized_ages.append(self.tokenizer.normalize_age(event["time"] - birth))
                self.timestamps.append(event["time"].timestamp())

                patient_length_index += 1

                last_time = event["time"]

        if self.task is not None:
            num_added = self.task.add_event(last_time, None, [])
            for i in range(num_added):
                self.label_indices.append(len(self.ages) - 1)

        self.patient_lengths.append(len(self.ages) - start_index)

    def get_batch_data(self):
        transformer = {
            "valid_tokens": np.array(self.valid_tokens),
            "ages": np.array(self.ages, dtype=np.float32),
            "normalized_ages": np.array(self.normalized_ages, dtype=np.float32),
            "timestamps": np.array(self.timestamps, dtype=np.float64),
            "patient_lengths": np.array(self.patient_lengths, dtype=np.int32),
            "label_indices": np.array(self.label_indices, dtype=np.int32),
        }

        if not self.tokenizer.is_hierarchical:
            transformer["tokens"] = np.array(self.tokens, dtype=np.int32)
        else:
            transformer["hierarchical_tokens"] = np.array(self.hierarchical_tokens, dtype=np.int32)
            transformer["hierarchical_weights"] = np.array(self.hierarchical_weights, dtype=np.float16)
            transformer["token_indices"] = np.array(self.token_indices, dtype=np.int32)

        final = {
            "num_patients": len(self.patient_ids),
            "num_indices": len(self.label_indices),
            "patient_ids": np.array(self.patient_ids, dtype=np.int64),
            "offsets": np.array(self.offsets, dtype=np.int32),
            "transformer": transformer,
        }

        if self.task is not None and transformer["label_indices"].shape[0] > 0:
            final["task"] = self.task.get_batch_data()
            final["needs_exact"] = self.task.needs_exact()
        return final


class FEMRBatchProcessor:
    def __init__(self, tokenizer, task=None):
        self.creator = BatchCreator(tokenizer, task)

    def convert_patient(self, patient, tensor_type=None, **formatter_kwargs):
        total_events = 0
        for event in patient["events"]:
            for measurement in event["measurements"]:
                total_events += 1
        self.creator.start_batch()
        self.creator.add_patient(patient, 0)
        batch_data = self.creator.get_batch_data()
        if tensor_type is not None:
            formatter = datasets.formatting.get_formatter(tensor_type, **formatter_kwargs)
            batch_data = formatter.recursive_tensorize(batch_data)
        return batch_data

    def convert_dataset(self, dataset, tokens_per_batch: int, min_samples_per_batch: int = 4, num_proc: int = 1):
        if isinstance(dataset, datasets.DatasetDict):
            return datasets.DatasetDict(
                {
                    k: self.convert_dataset(v, tokens_per_batch, min_samples_per_batch, num_proc)
                    for k, v in dataset.items()
                }
            )

        print("Processing", len(dataset))
        max_length = tokens_per_batch // min_samples_per_batch
        lengths = femr.hf_utils.aggregate_over_dataset(
            dataset,
            functools.partial(map_length_stats, processor=self, max_length=max_length),
            agg_length_stats,
            num_proc=num_proc,
            batch_size=1_000,
            with_indices=True,
        )

        random.shuffle(lengths)

        batches: List[List[Tuple[int, int, int]]] = []
        current_batch: List[Tuple[int, int, int]] = []
        current_batch_length = 0

        for patient_id, offset, length in lengths:
            if current_batch_length + length > tokens_per_batch:
                batches.append(current_batch)
                current_batch = []
                current_batch_length = 0

            current_batch_length += length
            current_batch.append((patient_id, offset, length))

        batches.append(current_batch)

        def batch_generator(dataset, batches):
            for batch in batches:
                self.creator.start_batch()
                for patient_index, offset, length in batch:
                    self.creator.add_patient(dataset[patient_index], offset, length)

                yield self.creator.get_batch_data()

        batch_dataset = datasets.Dataset.from_generator(
            batch_generator,
            gen_kwargs={
                "dataset": dataset,
                "batches": batches,
            },
            num_proc=num_proc,
        )

        return batch_dataset
