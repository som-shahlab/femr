from __future__ import annotations

import collections
import datetime
import functools
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import datasets
import meds
import meds_reader
import numpy as np
import torch.utils.data

import femr.models.tokenizer
import femr.pat_utils


def map_preliminary_batch_stats(
    patients: Iterable[meds_reader.Patient], *, processor: FEMRBatchProcessor, max_length: int
):
    """
    This function creates preliminary batch statistics, to be used for final batching.

    The overall problem this is trying to solve is that Patient records can be very long, so
    when we featurize patients we actually featurize subsequences of patients.

    AKA every patient in a batch is actually (patient_id, start_index, length), which
    specifies a subsequence of the patient.

    The trickiness becomes when a particular patient has multiple labels, some
    of which will require multiple subsequences.

    The goal of this function is to compute the list [(patient_id, start_index, length)] such
    that every label is covered by at least one batch. Note that some labels will be covered multiple times.

    Note that there is a special setting for tasks that don't need exact labeling
    (needs_exact in tasks.py returns False).

    For these patients we only generate one tuple for them, and drop some labels.

    Later code will then take [(patient_id, start_index, length)], and create actual batches.
    """
    lengths = []

    for patient in patients:
        data = processor.convert_patient(patient)

        # There are no labels for this patient
        if data["transformer"]["label_indices"].shape[0] == 0:
            continue

        # We need exact batching, so we need an algorithm that precisely covers every label in the batch
        if data["needs_exact"]:
            current_start = 0
            current_end = 0
            for label_index in data["transformer"]["label_indices"]:
                if (label_index - current_start + 1) >= max_length:
                    if current_start != current_end:
                        lengths.append((patient.patient_id, current_start, current_end - current_start + 1))
                    current_start = label_index - max_length + 1
                    current_end = label_index
                else:
                    current_end = label_index

            lengths.append((patient.patient_id, current_start, current_end - current_start + 1))
        else:
            last_index = data["transformer"]["label_indices"][-1]
            length = min(max_length, last_index + 1)
            lengths.append((patient.patient_id, last_index + 1 - length, length))
    if len(lengths) > 0:
        return np.array(lengths, dtype=np.int64)
    else:
        return np.zeros(shape=(0, 3), dtype=np.int64)


class BatchCreator:
    """The BatchCreator is designed to generate batches from patient data."""

    def __init__(self, tokenizer: femr.models.tokenizer.FEMRTokenizer, task: Optional[femr.models.tasks.Task] = None):
        """Initialize a BatchCreator, with a tokenizer, and optionally a task."""
        self.tokenizer = tokenizer
        self.task = task

    def start_batch(self):
        """Start a batch."""
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

    def add_patient(self, patient: meds_reader.Patient, offset: int = 0, max_length: Optional[int] = None):
        """Add a patient to the current batch.

        Note that the two optional parameters are used to add a subset of a patient to a batch.

        It is generally recommended to never manually use offset or max_length as
        you should rely on FEMRBatchProcessor.convert_dataset.

        Arguments:
            patient: The patient to add.
            offset: The offset into the patient to featurize.
            max_length: The maximum length of the batch sequence. There is no max when left at None.

        """
        current_date = None
        last_time = None

        # The overall algorithm here is a bit complex
        # First we featurize the entire patient
        # Then we slice the patient indices according to offset and max_length

        # These are the indices of the labels into the patient vectors
        per_patient_label_indices = []

        # The ages at each index for the patient
        per_patient_ages = []

        # The normalized age at index for the patient
        per_patient_normalized_ages = []

        # The timestamps at each index for the patient
        per_patient_timestamps = []

        # For a regular tokenizer, we just have tokens
        per_patient_tokens = []

        # For a hierarchical tokenizer, we have a more complex setup
        # These are designed to match the inputs required for an EmbeddingBag.
        # See PyTorch's EmbeddingBag documentation to understand what these mean.
        per_patient_hierarchical_tokens: List[int] = []
        per_patient_hierarchical_weights: List[float] = []
        per_patient_token_indices: List[int] = [0]

        if self.task is not None:
            self.task.start_patient(patient, self.tokenizer.ontology)

        birth = femr.pat_utils.get_patient_birthdate(patient)
        self.tokenizer.start_patient()

        for event in patient.events:
            # We want to avoid duplicate codes in the same day, so we maintain codes_seen_today
            if event.time.date() != current_date:
                current_date = event.time.date()
                codes_seen_today = set()

            # Get features and weights for the current event
            features, weights = self.tokenizer.get_feature_codes(event)

            # Ignore events with no features
            if len(features) == 0:
                continue

            # Ignore events where all features have already occurred
            if all(feature in codes_seen_today for feature in features):
                continue

            codes_seen_today |= set(features)

            if (self.task is not None) and (last_time is not None):
                # Now we have to consider whether or not to have labels for this time step
                # The add_event function returns how many labels to assign for this time
                num_added = self.task.add_event(last_time, event.time, features)
                for _ in range(num_added):
                    per_patient_label_indices.append(len(per_patient_ages) - 1)

            if not self.tokenizer.is_hierarchical:
                assert len(features) == 1
                per_patient_tokens.append(features[0])
            else:
                assert weights is not None
                per_patient_hierarchical_tokens.extend(features)
                per_patient_hierarchical_weights.extend(weights)
                per_patient_token_indices.append(len(per_patient_hierarchical_tokens))

            per_patient_ages.append((event.time - birth) / datetime.timedelta(days=1))
            per_patient_normalized_ages.append(self.tokenizer.normalize_age(event.time - birth))
            per_patient_timestamps.append(event.time.replace(tzinfo=datetime.timezone.utc).timestamp())

            last_time = event.time

        if self.task is not None and last_time is not None:
            num_added = self.task.add_event(last_time, None, None)
            for _ in range(num_added):
                per_patient_label_indices.append(len(per_patient_ages) - 1)

        # Now we want to actually add the patient data to the batch.
        # This will involve some clever slicing.

        # First, let's get the length we are adding
        length_found = len(per_patient_ages)
        if max_length is not None:
            length_to_add = min(length_found - offset, max_length)
        else:
            length_to_add = length_found - offset

        start_index = len(self.ages)

        # Let's add the constants first
        self.valid_tokens.extend([True] * length_to_add)
        self.patient_ids.extend([patient.patient_id] * length_to_add)
        self.offsets.append(offset)
        self.patient_lengths.append(length_to_add)

        # Ages, normalized ages and timestamps are also easy to add
        self.ages.extend(per_patient_ages[offset : offset + length_to_add])
        self.normalized_ages.extend(per_patient_normalized_ages[offset : offset + length_to_add])
        self.timestamps.extend(per_patient_timestamps[offset : offset + length_to_add])

        if not self.tokenizer.is_hierarchical:
            # Easy for simple tokenizer
            self.tokens.extend(per_patient_tokens[offset : offset + length_to_add])
        else:
            # Hierarchical tokenizer is more complex since we have to shift the indices as well
            # Remember, these arrays are all designed for PyTorch EmbeddingBag

            # We need to get the start and end at a particular offset
            internal_start = per_patient_token_indices[offset]
            internal_end = per_patient_token_indices[offset + length_to_add]

            # We need to offset the token indices to account for the existing tokens
            self.token_indices.extend(
                [
                    len(self.hierarchical_tokens) - internal_start + value
                    for value in per_patient_token_indices[offset + 1 : offset + length_to_add + 1]
                ]
            )

            self.hierarchical_tokens.extend(per_patient_hierarchical_tokens[internal_start:internal_end])
            self.hierarchical_weights.extend(per_patient_hierarchical_weights[internal_start:internal_end])

        # The label indices are also a bit tricky as they have to be offset accordingly.
        # We also need to collect good labels that should be sent to the final numpy arrays.
        labels_to_add = []
        for i, label_index in enumerate(per_patient_label_indices):
            corrected_label = label_index - offset

            if 0 <= corrected_label < length_to_add:
                labels_to_add.append(i)
                self.label_indices.append(start_index + corrected_label)

        if self.task is not None:
            self.task.add_patient_labels(labels_to_add)

    def get_batch_data(self):
        """Convert the batch to numpy arrays. The data structure is defined inline in this function."""
        if self.tokenizer.vocab_size <= 2**15:
            token_dtype = np.int16
        else:
            token_dtype = np.int32

        transformer = {
            # Whether or not the token is valid at this index
            "valid_tokens": np.array(self.valid_tokens),
            # The age of the patient in days at this index
            "ages": np.array(self.ages, dtype=np.float32),
            # The normalized ages at this index
            "normalized_ages": np.array(self.normalized_ages, dtype=np.float16),
            # The timestamp (in seconds) at this index
            "timestamps": np.array(self.timestamps, dtype=np.int64),
            # The length of the patient
            "patient_lengths": np.array(self.patient_lengths, dtype=np.int32),
            # The indices of the labels
            "label_indices": np.array(self.label_indices, dtype=np.int32),
        }

        if not self.tokenizer.is_hierarchical:
            # For a single tokenizer, these are simple the token indices
            transformer["tokens"] = np.array(self.tokens, dtype=token_dtype)
        else:
            # See PyTorch's EmbeddingBag for what these numpy arrays mean.
            transformer["hierarchical_tokens"] = np.array(self.hierarchical_tokens, dtype=token_dtype)
            transformer["hierarchical_weights"] = np.array(self.hierarchical_weights, dtype=np.float16)
            transformer["token_indices"] = np.array(self.token_indices, dtype=np.int32)

        # Some general metadata
        final = {
            "num_patients": len(self.patient_lengths),
            "num_indices": len(self.label_indices),
            "patient_ids": np.array(self.patient_ids, dtype=np.int64),
            "offsets": np.array(self.offsets, dtype=np.int32),
            "transformer": transformer,
        }

        # Add the task data
        if self.task is not None and transformer["label_indices"].shape[0] > 0:
            final["task"] = self.task.get_batch_data()
            final["needs_exact"] = self.task.needs_exact()
        return final

    def cleanup_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Clean a batch, applying final processing.

        This is necessary as some tasks use sparse matrices that need to be postprocessed."""

        batch["transformer"]["patient_lengths"] = np.array(batch["transformer"]["patient_lengths"])
        assert isinstance(batch["transformer"]["patient_lengths"], np.ndarray)

        if self.task is not None and "task" in batch:
            batch["task"] = self.task.cleanup(batch["task"])

        return batch


def _batch_generator(batch_data: Tuple[np.ndarray, np.ndarray], *, creator: BatchCreator, path_to_database: str):
    with meds_reader.PatientDatabase(path_to_database) as database:
        for lengths, offsets in batch_data:
            offsets = list(offsets)
            for start, end in zip(offsets, offsets[1:]):
                creator.start_batch()
                for patient_index, offset, length in lengths[start:end, :]:
                    creator.add_patient(database[patient_index.item()], offset, length)

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
    """The FEMR Batch processor creates batches for processing by a transformer."""

    def __init__(self, tokenizer, task=None):
        self.creator = BatchCreator(tokenizer, task)

    def convert_patient(
        self,
        patient: meds_reader.Patient,
        offset: int = 0,
        max_length: Optional[int] = None,
        tensor_type=None,
        **formatter_kwargs,
    ):
        """Convert a single patient to a batch.

        Note that this can also convert parts of a patient to a batch using the offset and max_length parameters.
        This is useful for processing long patients.

        NOTE: This function is primarily for debugging purposes. It is
        recommended to use convert_dataset for maximum correctness and efficiency.

        Arguments:
            patient: The patient to convert
            offset: The integer offset into the patient to convert
            max_length: The maximum length to convert
            tensor_type: The dataset to return
            formatter_kwargs: Arguments for a datasets formatter when converting datatypes

        Returns:
            A batch, ready to be fed into a FEMR transformer model
        """
        self.creator.start_batch()
        self.creator.add_patient(patient, offset=offset, max_length=max_length)
        batch_data = self.creator.get_batch_data()
        if tensor_type is not None:
            formatter = datasets.formatting.get_formatter(tensor_type, **formatter_kwargs)
            batch_data = formatter.recursive_tensorize(batch_data)
        return batch_data

    def collate(self, batches: List[Mapping[str, Any]]) -> Mapping[str, Any]:
        """A collate function that prepares batches for being fed into a dataloader."""
        assert len(batches) == 1, "Can only have one batch when collating"
        return {"batch": _add_dimension(self.creator.cleanup_batch(batches[0]))}

    def convert_dataset(
        self, db: meds_reader.PatientDatabase, tokens_per_batch: int, min_patients_per_batch: int = 4, num_proc: int = 1
    ):
        """Convert an entire dataset to batches.

        Arguments:
            dataset: A huggingface dataset containing MEDS patients
            tokens_per_batch: The number of tokens allowed per batch
            min_patients_per_batch: The minimum number of patients per batch
            num_proc: The number of processers to use when converting

        Returns:
            A huggingface dataset object containing batches
        """

        max_length = tokens_per_batch // min_patients_per_batch

        length_chunks = tuple(
            db.map(
                functools.partial(map_preliminary_batch_stats, processor=self, max_length=max_length),
            )
        )

        lengths = np.concatenate(length_chunks)

        rng = np.random.default_rng()
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
        print("Got batches", len(batches))

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

        batch_func = functools.partial(
            _batch_generator,
            creator=self.creator,
            path_to_database=db.path_to_database,
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
