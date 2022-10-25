from __future__ import annotations

import pprint
import collections
import datetime
import json
import pickle
import os
from collections.abc import MutableMapping
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    DefaultDict,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

from .. import Event, Patient
from ..extension import datasets as extension_datasets

import numpy as np
import scipy.sparse

ColumnValue = namedtuple("ColumnValue", ["column", "value"])
"""A value for a particular column
.. py:attribute:: column
    The index for the column
.. py:attribute:: value
    The value for that column
"""

class FeaturizerList:
    """
        Featurizer list consists of a list of featurizers that will be used (in sequence) to featurize data.
        It enables training of featurizers, featurization, column name extraction, and serialization/deserialization.
    """

    def __init__(self, featurizers: List[Featurizer]):
        """Create a :class:`FeaturizerList` from a sequence of featurizers.
        
        Args:
            featurizers (List[Featurizer]): The featurizers to use for featurizeing patients.
        """
        self.featurizers = featurizers

    def train_featurizers(
        self,
        timelines: timeline.TimelineReader,
        labeler: labeler.Labeler,
        end_date: Optional[datetime.date] = None,
    ) -> None:
        """Train a list of featurizers on the provided patients using the given labeler.
        
        Args:
            timelines (:class:`stride_ml.timeline.TimelineReader`): The timelines to read from.
            labeler (:class:`stride_ml.labeler.Labeler`): The labeler to train with.
            end_date (datetime.date): An optional date used to filter data off the end of the timeline.
        """

        any_needs_training = any(
            featurizer.needs_training() for featurizer in self.featurizers
        )

        if not any_needs_training:
            return

        all_patients = labeler.get_all_patient_ids()

        for patient_id in timelines.get_patient_ids():
            if all_patients is not None and patient_id not in all_patients:
                continue

            patient = timelines.get_patient(patient_id, end_date=end_date)

            labels = labeler.label(patient)

            if len(labels) == 0:
                continue

            label_indices = {label.day_index for label in labels}

            for featurizer in self.featurizers:
                if featurizer.needs_training():
                    featurizer.train(patient, label_indices)

        for featurizer in self.featurizers:
            featurizer.finalize_training()

    def featurize(
        self,
        timelines: timeline.TimelineReader,
        labeler: labeler.Labeler,
        end_date: Optional[datetime.date] = None,
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Apply a list of featurizers to obtain a feature matrix and label vector for the given patients.
        Args:
            timelines (:class:`stride_ml.timeline.TimelineReader`): The timelines to read from.
            labeler (:class:`stride_ml.labeler.Labeler`): The labeler to compute labels with.
            end_date (datetime.date): An optional date used to filter data off the end of the timeline.
        Returns:
            This returns a tuple (data_matrix, labels, patient_ids, patient_day_indices).
            data_matrix is a sparse matrix of all the features of all the featurizers.
            labels is a list of boolean values representing the labels for each row in the matrix.
            patient_ids is a list of the patient ids for each row.
            patient_day_indices is a list of the day indices for each row.
        """
        data = []
        indices: List[int] = []
        indptr = []

        result_labels = []
        patient_ids = []
        patient_day_indices = []

        all_patients = labeler.get_all_patient_ids()

        for patient_id in timelines.get_patient_ids():
            if all_patients is not None and patient_id not in all_patients:
                continue

            patient = timelines.get_patient(patient_id, end_date=end_date)

            labels = labeler.label(patient)

            if len(labels) == 0:
                continue

            label_indices = set()
            for label in labels:
                if label.day_index in label_indices:
                    raise ValueError(
                        "The provided labeler is invalid as it contains multiple labels "
                        f"for patient {patient.patient_id} at day index {label.day_index}"
                    )
                label_indices.add(label.day_index)

            columns_by_featurizer = []

            for featurizer in self.featurizers:
                columns = featurizer.featurize(patient, label_indices)
                assert len(columns) == len(label_indices), (
                    f"The featurizer {featurizer} didn't provide enough rows for {labeler}"
                    " on patient {patient_id} ({len(columns)} != {len(label_indices)})"
                )
                columns_by_featurizer.append(columns)

            for i, label in enumerate(labels):
                indptr.append(len(indices))
                result_labels.append(label.is_positive)
                patient_ids.append(patient.patient_id)
                patient_day_indices.append(label.day_index)

                column_offset = 0
                for j, feature_columns in enumerate(columns_by_featurizer):
                    for column, value in feature_columns[i]:
                        assert (
                            0 <= column < self.featurizers[j].num_columns()
                        ), (
                            f"The featurizer {self.featurizers[j]} provided an out of bounds column for "
                            f"{labeler} on patient {patient.patient_id} ({column} should be between 0 and "
                            f"{self.featurizers[j].num_columns()})"
                        )
                        indices.append(column_offset + column)
                        data.append(value)

                    column_offset += self.featurizers[j].num_columns()

        total_columns = sum(
            featurizer.num_columns() for featurizer in self.featurizers
        )

        indptr.append(len(indices))

        data = np.array(data, dtype=np.float32)
        indices = np.array(indices, dtype=np.int32)
        indptr = np.array(indptr, dtype=np.int32)

        data_matrix = scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=(len(result_labels), total_columns)
        )

        return (
            data_matrix,
            np.array(result_labels, dtype=np.float32),
            np.array(patient_ids, dtype=np.int32),
            np.array(patient_day_indices, dtype=np.int32),
        )

    def get_column_name(self, column_index: int) -> str:
        offset = 0

        for featurizer in self.featurizers:
            if offset <= column_index < (offset + featurizer.num_columns()):
                return f"Featurizer {featurizer}, {featurizer.get_column_name(column_index - offset)}"

            offset += featurizer.num_columns()

        assert False, "This should never happen"

    def save(self, fp: TextIO) -> None:
        json.dump([featurizer.to_dict() for featurizer in self.featurizers], fp)

    def load(self, fp: TextIO) -> None:
        for data_for_featurizer, featurizer in zip(
            json.load(fp), self.featurizers
        ):
            featurizer.from_dict(data_for_featurizer)

class Featurizer(ABC):
    """A Featurizer takes a Patient and a list of Labels, then returns a row for each timepoint. 
    Featurizers must be trained before they are used to compute normalization statistics. 
    A sparse representation named ColumnValue is used to represent the values returned by a Featurizer.
    """
    
    def train(self, patient: Patient, labels: List[Label]) -> None:
        """Train the featurizer on the given patient and label indices.
        This should do nothing if `needs_training()` returns FALSE, i.e. the featurizer doesn't need training.
        
        Args:
            patient (Patient): A patient to train on.
            labels (List[Label]): The list of labels of this patient to train on.
        """
        pass

    def finalize_training(self) -> None:
        """Finish the featurizer at the end of training. This is not needed for every
        featurizer, but does become necessary for things like verifying counts, etc.
        """
        pass

    @abstractmethod
    def num_columns(self) -> int:
        """Returns the number of columns that this featurizer creates.
        """
        pass

    @abstractmethod
    def featurize(self, patient: Patient, labels: List[Label]) -> List[List[ColumnValue]]:
        """Featurizes the patient into a series of rows using the specified timepoints.
        
        Args:
            patient (Patient): A patient to featurize.
            labels (List[Label]): We will generate features for each Label in `labels`.

        Returns:
             List[List[ColumnValue]]: A list of features (where features is a list itself) for each Label.
                The length of this list == length of `labels`
                    [idx] = corresponds to the Label at `labels[idx]`
                    [value] = List of :class:`ColumnValues<ColumnValue>` which contain the features for this label
        """

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the featurizer to a JSON compatible dict
        
        Returns: 
            A JSON compatible dict.
        """
        return {}

    def from_dict(self, data: Mapping[str, Any]):
        """Restore the state of the featurizer from a JSON compatible dict.
        
        Args:
            data (Mapping[str, Any): A JSON compatible dict from `to_dict()`
        """
        pass

    def get_column_name(self, column_index: int) -> str:
        """An optional method that enables the user to get the name of a column by its index
        
        Args:
            column_index (int): The index of the column
        """
        return "no name"

    def needs_training(self) -> bool:
        """Returns TRUE if you must run `train()`. If FALSE, then `train()` should do nothing.
        """        
        return False