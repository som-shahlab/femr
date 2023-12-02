import collections
import datetime
import functools
import random

import datasets
import numpy as np

import femr.hf_utils
import femr.models.dictionary


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
                next_end = label_index
                if label_index - current_start >= max_length:
                    length_map[(current_end - current_start + 1)].append((patient_id, current_start))
                    current_start = current_end - max_length + 1
                    current_end = current_end
                else:
                    current_end = next_end
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
    def __init__(self, dictionary, task=None):
        self.feature_lookup = femr.models.dictionary.FeatureLookup(dictionary)

        self.task = task
        self.dictionary = dictionary

    def start_batch(self, num_patients, max_length):
        self.patient_ids = np.zeros(num_patients, dtype=np.int64)
        self.offsets = np.zeros(num_patients, dtype=np.int32)

        self.tokens = np.zeros((num_patients, max_length), dtype=np.int32)
        self.valid_tokens = np.zeros((num_patients, max_length), dtype=np.bool_)

        self.ages = np.zeros((num_patients, max_length), dtype=np.float32)
        self.integer_ages = np.zeros((num_patients, max_length), dtype=np.int32)
        self.normalized_ages = np.zeros((num_patients, max_length), dtype=np.float32)

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
                features = self.feature_lookup.get_feature_codes(measurement)
                if len(features) == 0:
                    continue
                if all(feature in codes_seen_today for feature in features):
                    continue

                if self.length_index < offset:
                    self.length_index += 1
                    continue

                codes_seen_today |= set(features)

                if self.task is not None:
                    if self.length_index == 0:
                        # We cannot create tasks before birth so this needs special handling
                        # If we are at birth and the task needs an exact match, we are forced to try to add a task even if it is invalid
                        if not self.task.needs_exact():
                            # Don't need exact task mapping, so we can safely ignore this edge case
                            pass
                        else:
                            # Hope that this returns False so we can ignore this
                            added_task_event = self.task.add_event(last_time, event["time"], features)
                            assert not added_task_event, f"Cannot create labels before birth {patient['patient_id']}"
                    else:
                        added_task_event = self.task.add_event(last_time, event["time"], features)
                        if added_task_event:
                            self.label_indices.append(
                                self.patient_index * self.max_length + self.length_index - offset - 1
                            )

                if self.length_index - offset >= self.max_length:
                    break

                if self.dictionary["is_hierarchical"]:
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
                self.normalized_ages[
                    self.patient_index, self.length_index - offset
                ] = self.feature_lookup.normalize_age(self.integer_ages[self.patient_index, self.length_index - offset])

                self.length_index += 1

            last_time = event["time"]

        self.patient_index += 1

    def get_batch_data(self):
        transformer = {
            "length": self.max_length,
            "tokens": self.tokens,
            "valid_tokens": self.valid_tokens,
            "ages": self.ages,
            "normalized_ages": self.normalized_ages,
            "label_indices": np.array(self.label_indices, dtype=np.int32),
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
    def __init__(self, dictionary, task=None):
        self.creator = BatchCreator(dictionary, task)

    def convert_patient(self, patient):
        total_events = 0
        for event in patient["events"]:
            for measurement in event["measurements"]:
                total_events += 1
        self.creator.start_batch(1, total_events)
        self.creator.add_patient(patient, 0)
        return self.creator.get_batch_data()

    def convert_dataset(self, dataset, tokens_per_batch, min_samples_per_batch=4, num_proc=16):
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
        current_batch = []
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
            num_proc=16,
        )

        return batch_dataset
