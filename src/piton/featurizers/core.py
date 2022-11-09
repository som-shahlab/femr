from __future__ import annotations

import datetime
import json
import os
import collections
from collections import namedtuple
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from .. import Event, Patient
from ..extension import datasets as extension_datasets
from ..labelers.core import LabelingFunction

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
        It enables preprocessing of featurizers, featurization, column name extraction, and serialization/deserialization.
    """

    def __init__(self, featurizers: List[Featurizer]):
        """Create a :class:`FeaturizerList` from a sequence of featurizers.
        
        Args:
            featurizers (List[Featurizer]): The featurizers to use for featurizeing patients.
        """
        self.featurizers = featurizers

    def preprocess_featurizers(
        self,
        patients: Sequence[Patient],
        labeling_function: LabelingFunction,
    ) -> None:
        """preprocess a list of featurizers on the provided patients using the given labeler.
        
        Args:
            timelines (:class:`stride_ml.timeline.TimelineReader`): The timelines to read from.
            labeler (:class:`stride_ml.labeler.Labeler`): The labeler to preprocess with.
            end_date (datetime.date): An optional date used to filter data off the end of the timeline.
        """

        any_needs_preprocessing = any(
            featurizer.needs_preprocessing() for featurizer in self.featurizers
        )

        if not any_needs_preprocessing:
            return
        
        for patient in patients:
            labels = labeling_function.label(patient)

            if len(labels) == 0:
                continue

            for featurizer in self.featurizers:
                if featurizer.needs_preprocessing():
                    featurizer.preprocess(patient, labels)

        for featurizer in self.featurizers:
            featurizer.finalize_preprocessing()

    def featurize(
        self,
        patients: Sequence[Patient],
        labeling_function: LabelingFunction,
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
        labeling_time = []

        for patient in patients:
            labels = labeling_function.label(patient)



            if len(labels) == 0:
                continue

            columns_by_featurizer = []

            for featurizer in self.featurizers:
                columns = featurizer.featurize(patient, labels)
                assert len(columns) == len(labels), (
                    f"The featurizer {featurizer} didn't provide enough rows for {labeling_function} on patient {patient.patient_id} ({len(columns)} != {len(labels)})"
                )
                columns_by_featurizer.append(columns)

            for i, label in enumerate(labels):
                indptr.append(len(indices))
                assert label.value == label.value
                result_labels.append(label.value)
                patient_ids.append(patient.patient_id)
                labeling_time.append(label.time)

                column_offset = 0
                for j, feature_columns in enumerate(columns_by_featurizer):
                    for column, value in feature_columns[i]:
                        assert (
                            0 <= column < self.featurizers[j].num_columns()
                        ), (
                            f"The featurizer {self.featurizers[j]} provided an out of bounds column for "
                            f"{labeling_function} on patient {patient.patient_id} ({column} should be between 0 and "
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
            np.array(labeling_time, dtype=np.datetime64)
        )

    def get_column_name(self, column_index: int) -> str:
        offset = 0

        for featurizer in self.featurizers:
            if offset <= column_index < (offset + featurizer.num_columns()):
                return f"Featurizer {featurizer}, {featurizer.get_column_name(column_index - offset)}"

            offset += featurizer.num_columns()

        assert False, "This should never happen"


class Featurizer(ABC):
    """A Featurizer takes a Patient and a list of Labels, then returns a row for each timepoint. 
    Featurizers must be preprocessed before they are used to compute normalization statistics. 
    A sparse representation named ColumnValue is used to represent the values returned by a Featurizer.
    """
    
    def preprocess(self, patient: Patient, labels: List[Label]) -> None:
        """preprocess the featurizer on the given patient and label indices.
        This should do nothing if `needs_preprocessing()` returns FALSE, i.e. the featurizer doesn't need preprocessing.
        
        Args:
            patient (Patient): A patient to preprocess on.
            labels (List[Label]): The list of labels of this patient to preprocess on.
        """
        pass

    def finalize_preprocessing(self) -> None:
        """Finish the featurizer at the end of preprocessing. This is not needed for every
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


    def get_column_name(self, column_index: int) -> str:
        """An optional method that enables the user to get the name of a column by its index
        
        Args:
            column_index (int): The index of the column
        """
        return "no name"

    def needs_preprocessing(self) -> bool:
        """Returns TRUE if you must run `preprocess()`. If FALSE, then `preprocess()` should do nothing.
        """        
        return False