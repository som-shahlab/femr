from __future__ import annotations

import collections
import datetime
import functools
from typing import Any, Dict, List, Mapping, Tuple, Optional

import datasets
import numpy as np
import torch.utils.data

import femr.hf_utils
import femr.models.tokenizer
import femr.models.tasks
import femr.pat_utils

import meds


def map_length_stats(batch, indices, *, processor: FEMRBatchProcessor, max_length):
    """
    construct a set of "lengths" for each patient in a batch. Each length is a tuple (patient_index, start_index, length), with:
    - patient_index: the index of the patient in the dataset
    - start_index: ...
    - length: ...

    if data["needs_exact"], then every label needs to have a prediction. otherwise...

    <to be continued>
    """
    lengths = []

    for patient_index, patient_id, events in zip(indices, batch["patient_id"], batch["events"]):
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
    if len(lengths) > 0:
        return [np.array(lengths, dtype=np.int64)]
    else:
        return []


def agg_length_stats(lengths1, lengths2):
    lengths1.extend(lengths2)

    return lengths1


class BatchCreator:
    """
    this object processes batches of patient data for a specific tokenizer and task.
    batches are used for training and inference.

    Usage:

    after initializing the BatchCreator object you need to call start_batch() to initialize required fields.

    then call add_patient(...) to add patients to the batch.

    each patient in a batch is represented by three integers:
    - patient_id: patient ID from MEDS dataset
    - offset: ...
    - length: ...

    <to be continued>
    """

    def __init__(self, tokenizer: femr.models.tokenizer.FEMRTokenizer, task: femr.models.tasks.Task = None):
        self.tokenizer = tokenizer
        self.task = task

    def start_batch(self):
        """
        TODO: need to define each of the fields below, which are initialized as [] or [0]
        - patient_ids: a list of patient ids in the batch?
        - offsets: ...
        - patient_lengths: ...
        - tokens: list of 
        """
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

    def add_patient(self, patient: meds.Patient, offset, max_patient_length=None):
        """
        TODO: this needs documentation. specifically:
        - describe logical flow
        - describe return value (and, when is return value None?)

        TODO: Q: what is offset?
        """
        self.patient_ids.append(patient["patient_id"])
        self.offsets.append(offset)

        def process_patient_events():
            """
            this needs documentation. 
            what is the logical flow? 
            what is the intended return value?
            why does this need to be a function?
            """
            
            # what is current_date?
            current_date = None

            # it seems like "last_time" is the time of the previous event in the loop: for event in patient["events"]...
            last_time = None

            if self.task is not None:
                self.task.start_patient(patient, self.tokenizer.ontology)

            patient_length_index = 0

            birth = femr.pat_utils.get_patient_birthdate(patient)
            self.tokenizer.start_patient()  # <-- why???
            
            # for all events in this patient's timeline...
            for event in patient["events"]:

                # if we haven't processed events for this day, start collecting a new set of codes that have occured on this day
                if event["time"].date() != current_date:
                    current_date = event["time"].date()
                    
                    # all codes we have seen on this day 
                    codes_seen_today = set()

                for measurement in event["measurements"]:

                    # "features" is different depending on the type of code.
                    # I'm not really sure what "features" contains.. sometimes it contains an empty list.
                    # the argument event["time"] is also never used by this function...
                    # wtf guys.
                    features, weights = self.tokenizer.get_feature_codes(event["time"], measurement)
                    if len(features) == 0:
                        continue
                    if all(feature in codes_seen_today for feature in features):
                        continue
                    
                    # add all features to codes_seen_today.
                    # why use "|="? very exotic syntax
                    codes_seen_today |= set(features)

                    # not sure what offset is. ¯\_(ツ)_/¯
                    if patient_length_index < offset:
                        patient_length_index += 1
                        continue

                    # it looks like last_time is the time of the previous event visited by this loop
                    # so... last_time is not None when this loop has passed the first event in the patient timeline
                    if (self.task is not None) and (last_time is not None):
                        # this will return 0 in some cases, meaning that nothing will be added to label_indices
                        # these cases are...
                        # 1 - next_date is None, or the date is the same for current_date and next_date
                        # 2 - something about the "calculator" and function get_future_events_for_time
                        num_added = self.task.add_event(
                            current_date=last_time, 
                            next_date=event["time"], 
                            next_features=features,  # <-- this is not used...
                        )
                        for _ in range(num_added):
                            self.label_indices.append(len(self.ages) - 1)

                    if max_patient_length is not None and (patient_length_index - offset >= max_patient_length):
                        return None

                    if not self.tokenizer.is_hierarchical:
                        assert len(features) == 1
                        self.tokens.append(features[0])
                    else:
                        self.hierarchical_tokens.extend(features)
                        self.hierarchical_weights.extend(weights)
                        self.token_indices.append(len(self.hierarchical_tokens))

                    self.patient_ids.append(patient["patient_id"])
                    self.valid_tokens.append(True)
                    self.ages.append((event["time"] - birth) / datetime.timedelta(days=1))
                    self.normalized_ages.append(self.tokenizer.normalize_age(event["time"] - birth))
                    self.timestamps.append(event["time"].timestamp())

                    patient_length_index += 1

                    last_time = event["time"]

            return last_time

        start_index = len(self.ages)
        final_time = process_patient_events()

        if self.task is not None and final_time is not None:
            num_added = self.task.add_event(final_time, None, [])
            for _ in range(num_added):
                self.label_indices.append(len(self.ages) - 1)

        self.patient_lengths.append(len(self.ages) - start_index)

    def get_batch_data(self):
        if self.tokenizer.vocab_size <= 2**15:
            token_dtype = np.int16
        else:
            token_dtype = np.int32

        transformer = {
            "valid_tokens": np.array(self.valid_tokens),
            "ages": np.array(self.ages, dtype=np.float32),
            "normalized_ages": np.array(self.normalized_ages, dtype=np.float16),
            "timestamps": np.array(self.timestamps, dtype=np.int64),
            "patient_lengths": np.array(self.patient_lengths, dtype=np.int32),
            "label_indices": np.array(self.label_indices, dtype=np.int32),
        }

        if not self.tokenizer.is_hierarchical:
            transformer["tokens"] = np.array(self.tokens, dtype=token_dtype)
        else:
            transformer["hierarchical_tokens"] = np.array(self.hierarchical_tokens, dtype=token_dtype)
            transformer["hierarchical_weights"] = np.array(self.hierarchical_weights, dtype=np.float16)
            transformer["token_indices"] = np.array(self.token_indices, dtype=np.int32)

        final = {
            "num_patients": len(self.patient_lengths),
            "num_indices": len(self.label_indices),
            "patient_ids": np.array(self.patient_ids, dtype=np.int64),
            "offsets": np.array(self.offsets, dtype=np.int32),
            "transformer": transformer,
        }

        # BUG: possible bug - if task is not none, why would we not populate task?
        # Q: where does label_indices come from?
        # this is created in BatchCreator.start_batch and updated in .add_patient
        if self.task is not None and transformer["label_indices"].shape[0] > 0:
            final["task"] = self.task.get_batch_data()
            final["needs_exact"] = self.task.needs_exact()
        return final

    def cleanup_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Clean a batch, applying final processing.

        This is necessary as some tasks use sparse matrices that need to be postprocessed."""

        # BUG: batch was on gpu already, so this was causing problems during inference? maybe during training this would not be an issue and we need this code
        # batch["transformer"]["patient_lengths"] = np.array(batch["transformer"]["patient_lengths"])  # BUG?
        # assert isinstance(batch["transformer"]["patient_lengths"], np.ndarray)

        # BUG: possible issue: is "task" not getting passed here, even if self.task is not None?
        if self.task is not None and "task" in batch:
            batch["task"] = self.task.cleanup(batch["task"])

        return batch


def _batch_generator(batch_data: Tuple[np.ndarray, np.ndarray], *, creator: BatchCreator, dataset: datasets.Dataset):
    for lengths, offsets in batch_data:
        offsets = list(offsets)
        for start, end in zip(offsets, offsets[1:]):
            creator.start_batch()
            for patient_index, offset, length in lengths[start:end, :]:
                creator.add_patient(dataset[patient_index.item()], offset, length)

            result = creator.get_batch_data()
            assert "task" in result, f"No task present in {lengths[start:end,:]}"

            yield result


def _add_dimension(data: Any) -> Any:
    if isinstance(data, collections.abc.Mapping):
        return {k: _add_dimension(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.unsqueeze(dim=0)
    elif isinstance(data, np.ndarray):
        return np.expand_dims(data, axis=0)
    elif isinstance(data, (int, float, np.number, np.bool_)):
        return data
    else:
        raise RuntimeError("Could not convert item of type " + str(type(data)))


class FEMRBatchProcessor:
    """
    TODO: needs documentation.

    FEMRBatchProcessor object includes functions for creating batches from a single patient (convert_patient) or a dataset (convert_dataset). 
    
    The batching process is determined by a FEMRTokenizer and Task.

    It looks like task is only passed to BatchCreator
    """
    def __init__(self, tokenizer: femr.models.tokenizer.FEMRTokenizer, task: femr.models.tasks.Task = None):
        self.creator = BatchCreator(tokenizer, task)

    def convert_patient(self, patient, offset=0, max_patient_length=None, tensor_type=None, **formatter_kwargs):
        """convert a single patient to a batch? why is this needed?"""
        self.creator.start_batch()
        self.creator.add_patient(patient, offset=offset, max_patient_length=max_patient_length)
        batch_data = self.creator.get_batch_data()
        if tensor_type is not None:
            formatter = datasets.formatting.get_formatter(tensor_type, **formatter_kwargs)
            batch_data = formatter.recursive_tensorize(batch_data)
        return batch_data

    def collate(self, batches: List[Mapping[str, Any]]) -> Mapping[str, Any]:
        assert len(batches) == 1, "Can only have one batch when collating"
        return {"batch": _add_dimension(self.creator.cleanup_batch(batches[0]))}

    def convert_dataset(self, dataset, tokens_per_batch: int, min_samples_per_batch: int = 4, num_proc: int = 1, batch_size: int = 200):
        if isinstance(dataset, datasets.DatasetDict):
            return datasets.DatasetDict(
                {
                    k: self.convert_dataset(v, tokens_per_batch, min_samples_per_batch, num_proc)
                    for k, v in dataset.items()
                }
            )

        max_length = tokens_per_batch // min_samples_per_batch
        lengths = femr.hf_utils.aggregate_over_dataset(
            dataset,
            functools.partial(map_length_stats, processor=self, max_length=max_length),
            agg_length_stats,
            num_proc=num_proc,
            batch_size=batch_size,
            with_indices=True,
        )

        lengths = np.concatenate(lengths)

        # this should have a seed
        rng = np.random.default_rng(seed=0)
        rng.shuffle(lengths)

        current_batch_length = 0

        batch_offsets = [0]

        for i, length in enumerate(lengths[:, 2]):
            if current_batch_length + length > tokens_per_batch:
                batch_offsets.append(i)
                current_batch_length = 0

            current_batch_length += length

        batch_offsets.append(len(lengths))

        batches = list(zip(batch_offsets, batch_offsets[1:]))

        split_batches = np.array_split(batches, num_proc)

        final_batch_data = []

        for batch_part in split_batches:
            if len(batch_part) == 0:
                continue
            start = batch_part[0][0]
            end = batch_part[-1][-1]
            lengths_part = lengths[start:end, :]
            offsets = [0] + [b - start for _, b in batch_part]

            final_batch_data.append(
                (
                    lengths_part,
                    np.array(offsets, dtype=np.int32),
                )
            )

        print("Creating batches", len(batches))

        batch_func = functools.partial(
            _batch_generator,
            creator=self.creator,
            dataset=dataset,
        )

        batch_dataset = datasets.Dataset.from_generator(
            batch_func,
            gen_kwargs={
                "batch_data": final_batch_data,
            },
            num_proc=num_proc,
            writer_batch_size=8,
        )

        return batch_dataset
