from __future__ import annotations

import collections
import datetime
import functools
import random
from typing import List

import datasets
import numpy as np

import femr.hf_utils
import femr.models.tokenizer


def map_length_stats(batch, indices, *, processor, max_length):
    length_map = collections.defaultdict(list)

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
                    length_map[(current_end - current_start + 1)].append((patient_index, current_start))
                    current_start = label_index - max_length + 1
                    current_end = label_index
                else:
                    current_end = label_index

            length_map[(current_end - current_start + 1)].append((patient_index, current_start))
        else:
            last_index = data["transformer"]["label_indices"][-1]
            length = min(max_length, last_index + 1)
            length_map[length].append((patient_index, last_index + 1 - length))
    return length_map


def agg_length_stats(length_stats1, length_stats2):
    for k, v in length_stats2.items():
        length_stats1[k].extend(v)

    return length_stats1


class BatchCreator:
    def __init__(self, tokenizer, task=None):
        self.tokenizer = tokenizer
        self.task = task

    def start_batch(self, num_patients, max_length):
        self.patient_ids = np.zeros(num_patients, dtype=np.int64)
        self.offsets = np.zeros(num_patients, dtype=np.int32)

        self.tokens = np.zeros((num_patients, max_length), dtype=np.int32)
        self.valid_tokens = np.zeros((num_patients, max_length), dtype=np.bool_)

        self.ages = np.zeros((num_patients, max_length), dtype=np.float32)
        self.integer_ages = np.zeros((num_patients, max_length), dtype=np.int32)
        self.normalized_ages = np.zeros((num_patients, max_length), dtype=np.float32)
        self.timestamps = np.zeros((num_patients, max_length), dtype=np.int64)

        self.label_indices = []

        self.patient_index = 0

        self.num_patients = num_patients
        self.max_length = max_length

        if self.task is not None:
            self.task.start_batch()

    def add_patient(self, patient, offset):
        self.patient_ids[self.patient_index] = patient["patient_id"]
        self.offsets[self.patient_index] = offset

        current_date = None
        last_time = None

        if self.task is not None:
            self.task.start_patient(patient)

        self.length_index = 0

        birth = patient["events"][0]["time"]

        for event in patient["events"]:
            if event["time"].date() != current_date:
                current_date = event["time"].date()
                codes_seen_today = set()

            for measurement in event["measurements"]:
                features = self.tokenizer.get_feature_codes(measurement)
                if len(features) == 0:
                    continue
                if all(feature in codes_seen_today for feature in features):
                    continue

                if self.length_index < offset:
                    self.length_index += 1
                    continue

                codes_seen_today |= set(features)

                if self.task is not None and last_time is not None:
                    num_added = self.task.add_event(last_time, event["time"], features)
                    for i in range(num_added):
                        self.label_indices.append(self.patient_index * self.max_length + self.length_index - offset - 1)

                if self.length_index - offset >= self.max_length:
                    break

                if self.tokenizer.is_hierarchical:
                    assert False  # TODO: Implement this
                else:
                    self.tokens[self.patient_index, self.length_index - offset] = features[0]

                self.valid_tokens[self.patient_index, self.length_index - offset] = True
                self.ages[self.patient_index, self.length_index - offset] = (
                    event["time"] - birth
                ) / datetime.timedelta(days=1)
                self.integer_ages[self.patient_index, self.length_index - offset] = (
                    event["time"] - birth
                ) / datetime.timedelta(minutes=1)
                self.normalized_ages[self.patient_index, self.length_index - offset] = self.tokenizer.normalize_age(
                    self.integer_ages[self.patient_index, self.length_index - offset]
                )
                self.timestamps[self.patient_index, self.length_index - offset] = event["time"].timestamp()

                self.length_index += 1

                last_time = event["time"]

        if self.task is not None:
            num_added = self.task.add_event(last_time, None, None)
            for i in range(num_added):
                self.label_indices.append(self.patient_index * self.max_length + self.length_index - offset - 1)

        self.patient_index += 1

    def get_batch_data(self):
        transformer = {
            "length": self.max_length,
            "tokens": self.tokens,
            "valid_tokens": self.valid_tokens,
            "ages": self.ages,
            "integer_ages": self.integer_ages,
            "normalized_ages": self.normalized_ages,
            "label_indices": np.array(self.label_indices, dtype=np.int32),
            "timestamps": self.timestamps,
        }

        final = {
            "num_patients": len(self.patient_ids),
            "num_indices": len(self.label_indices),
            "patient_ids": self.patient_ids,
            "offsets": self.offsets,
            "transformer": transformer,
        }

        if self.task is not None:
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
        self.creator.start_batch(1, total_events)
        self.creator.add_patient(patient, 0)
        batch_data = self.creator.get_batch_data()
        if tensor_type is not None:
            formatter = datasets.formatting.get_formatter(tensor_type, **formatter_kwargs)
            batch_data = formatter.recursive_tensorize(batch_data)
        return batch_data

    def convert_dataset(self, dataset, tokens_per_batch: int, min_samples_per_batch: int = 4, num_proc: int = 16):
        if isinstance(dataset, datasets.DatasetDict):
            return datasets.DatasetDict(
                {
                    k: self.convert_dataset(v, tokens_per_batch, min_samples_per_batch, num_proc)
                    for k, v in dataset.items()
                }
            )

        max_length = tokens_per_batch // min_samples_per_batch
        length_stats = femr.hf_utils.aggregate_over_dataset(
            dataset,
            functools.partial(map_length_stats, processor=self, max_length=max_length),
            agg_length_stats,
            num_proc=num_proc,
            batch_size=1_000,
            with_indices=True,
        )

        size_and_samples = sorted(list(length_stats.items()), reverse=True)

        batches = []

        current_batch_size = size_and_samples[0][0]
        current_batch: List[int] = []
        for size, samples in size_and_samples:
            random.shuffle(samples)
            for sample in samples:
                if (1 + len(current_batch)) * current_batch_size >= tokens_per_batch:
                    batches.append((current_batch_size, current_batch))

                    current_batch_size = size
                    current_batch = []

                current_batch.append(sample)

        batches.append((current_batch_size, current_batch))

        def batch_generator(dataset, batches):
            for batch_size, batch_items in batches:
                self.creator.start_batch(len(batch_items), batch_size)
                for patient_index, offset in batch_items:
                    self.creator.add_patient(dataset[patient_index], offset)

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
