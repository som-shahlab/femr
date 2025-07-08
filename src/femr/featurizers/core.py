"""Core featurizer functionality, shared across Featurizers."""

from __future__ import annotations

import collections
import datetime
import functools
from abc import ABC, abstractmethod
from typing import Any, Iterator, List, Mapping, NamedTuple, Sequence, Tuple, TypeVar

import meds_reader
import numpy as np
import pandas as pd
import scipy.sparse

import femr.labelers


class ColumnValue(NamedTuple):
    """A value for a particular column
    `column` is the index for the column
    `value` is the value for that column. Values must be numeric
    """

    column: int
    value: float | int


def _preprocess_map_func(
    subjects_and_labels: Iterator[Tuple[meds_reader.Subject, Sequence[femr.labelers.Label]]],
    featurizers: List[Featurizer],
) -> List[List[Any]]:
    initial_data = [featurizer.get_initial_preprocess_data() for featurizer in featurizers]
    for subject, labels in subjects_and_labels:
        for data, featurizer in zip(initial_data, featurizers):
            featurizer.add_preprocess_data(data, subject, labels)

    return initial_data


def _features_map_func(
    subjects_and_labels: Iterator[Tuple[meds_reader.Subject, Sequence[femr.labelers.Label]]],
    *,
    featurizers: List[Featurizer],
) -> Mapping[str, Any]:
    # Construct CSR sparse matrix
    #   non-zero entries in sparse matrix
    data_and_indices = np.zeros((1024, 2), np.float64)
    data_and_indices_arrays: List[np.ndarray] = []

    current_index = 0

    #   maps each element in `data` and `indices` to the rows of the sparse matrix
    indptr: List[int] = []

    subject_ids: List[int] = []
    feature_times: List[datetime.datetime] = []

    for subject, labels in subjects_and_labels:
        assert len(labels) != 0, "Must have at least one label per subject processed"

        for label in labels:
            subject_ids.append(subject.subject_id)
            feature_times.append(label.prediction_time)

        # For each Featurizer, apply it to this Subject...
        features_per_label: List[List[List[ColumnValue]]] = [[] for _ in range(len(labels))]
        for featurizer in featurizers:
            features: List[List[ColumnValue]] = featurizer.featurize(subject, labels)
            assert len(features) == len(labels), (
                f"The featurizer `{featurizer}` didn't generate a set of features for "
                f"every label for subject {subject.subject_id} ({len(features)} != {len(labels)})"
            )
            for a, b in zip(features_per_label, features):
                a.append(b)

        for features in features_per_label:
            indptr.append(current_index + len(data_and_indices_arrays) * 1024)

            # Keep track of starting column for each successive featurizer as we
            # combine their features into one large matrix
            column_offset: int = 0
            for featurizer, feature_columns in zip(featurizers, features):
                for column, value in feature_columns:
                    assert 0 <= column < featurizer.get_num_columns(), (
                        f"The featurizer {featurizer} provided an out of bounds column for "
                        f"{column} on subject {subject.subject_id} ({column} must be between 0 and "
                        f"{featurizer.get_num_columns()})"
                    )
                    data_and_indices[current_index, 0] = value
                    data_and_indices[current_index, 1] = column_offset + column

                    current_index += 1

                    if current_index == 1024:
                        current_index = 0
                        data_and_indices_arrays.append(data_and_indices.copy())

                # Record what the starting column should be for the next featurizer
                column_offset += featurizer.get_num_columns()

    # Need one last `indptr` for end of last row in CSR sparse matrix
    indptr.append(current_index + len(data_and_indices_arrays) * 1024)

    # n_rows = number of Labels across all Subjects
    total_rows: int = len(indptr) - 1
    # n_cols = sum of number of columns output by each Featurizer
    total_columns: int = sum(x.get_num_columns() for x in featurizers)

    # Explanation of CSR Matrix: https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr
    data_and_indices_arrays.append(data_and_indices[:current_index, :])

    np_data_and_indices: np.ndarray = np.concatenate(data_and_indices_arrays)

    np_data = np_data_and_indices[:, 0].astype(np.float32)
    np_indices = np_data_and_indices[:, 1].astype(np.int64)

    np_indptr: np.ndarray = np.array(indptr, dtype=np.int64)

    assert (
        np_indptr.shape[0] == total_rows + 1
    ), f"`indptr` length should be equal to '{total_rows + 1}', but instead is '{np_indptr.shape[0]}"
    assert (
        np_data.shape == np_indices.shape
    ), f"`data` should have equal shape as `indices`, but instead have {np_data.shape} != {np_indices.shape}"
    data_matrix = scipy.sparse.csr_matrix((np_data, np_indices, np_indptr), shape=(total_rows, total_columns))

    np_subject_ids: np.ndarray = np.array(subject_ids, dtype=np.int64)
    np_feature_times: np.ndarray = np.array(feature_times, dtype="datetime64[us]")

    return {"subject_ids": np_subject_ids, "feature_times": np_feature_times, "features": data_matrix}


class Featurizer(ABC):
    """A Featurizer takes a Subject and a list of Labels, then returns a row for each timepoint.
    Featurizers must be preprocessed before they are used to compute normalization statistics.
    A sparse representation named ColumnValue is used to represent the values returned by a Featurizer.
    """

    def get_initial_preprocess_data(self) -> Any:
        """
        Get the initial preprocess data
        """
        pass

    def add_preprocess_data(self, data: Any, subject: meds_reader.Subject, labels: Sequence[femr.labelers.Label]):
        """
        Some featurizers need to do some preprocessing in order to prepare for featurization.
        This function performs that preprocessing on the given subjects and labels, and returns some state.
        That state is concatinated across the entire database,
            and then passed to encorperate_preprocessed_data.

        Note that this function shouldn't mutate the Featurizer as it will be sharded.
        """
        pass

    def encorperate_prepreprocessed_data(self, data_elements: List[Any]) -> None:
        """
        This encorperates data generated from generate_preprocess_data across the entire dataset into the featurizer

        This should mutate the Featurizer.
        """
        pass

    @abstractmethod
    def get_num_columns(self) -> int:
        """Return the number of columns that this featurizer creates."""
        pass

    @abstractmethod
    def featurize(
        self,
        subject: meds_reader.Subject,
        labels: Sequence[femr.labelers.Label],
    ) -> List[List[ColumnValue]]:
        """Featurize the subject such that each label in `labels` has an associated list of features.

        Example:
            return [
                [ ColumnValue(0, 10), ColumnValue(3, 12), ], # features for label 0
                [ ColumnValue(1, 0) ], # features for label 1
                [ ColumnValue(2, 2), ColumnValue(1, 3), ColumnValue(10, 3), ], # features for label 2
                ...
                [ ColumnValue(8, 1.3), ColumnValue(9, 5), ], # features for label n
            ]

        Where each ColumnValue is of the form: (idx of column for this feature, feature value).

        Thus, the List[List[ColumnValue]] represents a 2D sparse matrix, where each row is a distinct
        Label and each (sparse) column is a feature

        Args:
            subject (Subject): A subject to featurize.
            labels (List[Label]): We will generate features for each Label in `labels`.
            ontology (Optional[Ontology]): Ontology for Event codes.

        Returns:
             List[List[ColumnValue]]: A list of 'features' (where 'features' is a list itself) for
             each Label.
                The length of this list of lists == length of `labels`
                    [idx] = corresponds to the Label at `labels[idx]`
                    [value] = List of :class:`ColumnValues<ColumnValue>` which contain the features
                    for this label
        """
        pass

    def get_column_name(self, column_idx: int) -> str:
        """Enable the user to get the name of a column by its index

        Args:
            column_idx (int): The index of the column
        """
        return "no name"

    def is_needs_preprocessing(self) -> bool:
        """Return TRUE if you must run `preprocess()`. If FALSE, then `preprocess()`
        should do nothing.
        """
        return False


FeaturizerType = TypeVar("FeaturizerType", bound=Featurizer)


class FeaturizerList:
    """
    FeaturizerList consists of a list of Featurizers that will be used to (sequentially)
    featurize Subjects based on their Labels.

    It enables preprocessing of featurizers, featurization, and column name extraction.
    """

    def __init__(self, featurizers: List[Featurizer]):
        """Create a `FeaturizerList` from a sequence of featurizers.

        Args:
            featurizers (List[Featurizer]): The featurizers to use for featurizing subjects.
        """
        self.featurizers: List[Featurizer] = featurizers

    def preprocess_featurizers(
        self,
        db: meds_reader.SubjectDatabase,
        labels: pd.DataFrame,
    ) -> None:
        """Preprocess `self.featurizers` on the provided set of labels."""

        # Check if any featurizers need preprocessing. If not, return early.
        any_needs_preprocessing: bool = any(featurizer.is_needs_preprocessing() for featurizer in self.featurizers)
        if not any_needs_preprocessing:
            return

        # Split subjects across multiple threads
        featurize_stats: List[List[Any]] = [[] for _ in self.featurizers]

        for chunk_stats in db.map_with_data(
            functools.partial(_preprocess_map_func, featurizers=self.featurizers),
            labels,
            assume_sorted=True,
        ):
            for a, b in zip(featurize_stats, chunk_stats):
                a.append(b)

        # Aggregate featurizers
        for featurizer, featurizer_stat in zip(self.featurizers, featurize_stats):
            # Merge all featurizers of the same class as `featurizer`
            featurizer.encorperate_prepreprocessed_data(featurizer_stat)

    def featurize(
        self,
        db: meds_reader.SubjectDatabase,
        labels: pd.DataFrame,
    ) -> Mapping[str, np.ndarray]:
        """
        Apply a list of Featurizers (in sequence) to obtain a feature matrix for each Label for each subject.

        Args:
            database_path (str): Path to `SubjectDatabase` on disk

        Returns:
            This returns a tuple (data_matrix, labels, subject_ids, labeling_time).
                data_matrix is a sparse matrix of all the features of all the featurizers.
                label_pids is a list of the subject ids for each row.
                label_values is a list of boolean values representing the labels for each row in the matrix.
                labeling_time is a list of labeling/prediction time for each row.
        """
        features = collections.defaultdict(list)

        for feat_chunk in db.map_with_data(
            functools.partial(_features_map_func, featurizers=self.featurizers),
            labels,
            assume_sorted=True,
        ):
            for k, v in feat_chunk.items():
                features[k].append(v)

        result = {k: np.concatenate(features[k]) for k in ("subject_ids", "feature_times")}

        result["features"] = scipy.sparse.vstack(features["features"])

        return result

    def get_column_name(self, column_idx: int) -> str:
        column_offset: int = 0
        for featurizer in self.featurizers:
            if column_offset <= column_idx < (column_offset + featurizer.get_num_columns()):
                return f"Featurizer {featurizer}, {featurizer.get_column_name(column_idx - column_offset)}"
            column_offset += featurizer.get_num_columns()
        raise IndexError(f"Column index '{column_idx}' out of bounds for this FeaturizerList")


def join_labels(features: Mapping[str, np.ndarray], labels: pd.DataFrame) -> Mapping[str, np.ndarray]:
    indices = []
    label_values = []
    label_times = []

    order = np.lexsort((features["feature_times"], features["subject_ids"]))

    feature_index = 0

    for label in labels.itertuples(index=False):
        while ((feature_index + 1) < len(order)):
            next_key = (features['subject_ids'][order[feature_index + 1]], features["feature_times"][order[feature_index + 1]])
            if next_key <= (label.subject_id, label.prediction_time):
                feature_index += 1
            else:
                break

        is_valid = (
            (feature_index < len(order))
            and (features["subject_ids"][order[feature_index]] == label.subject_id)
            and (features["feature_times"][order[feature_index]] <= label.prediction_time)
        )
            
        assert is_valid, (
            f'{feature_index} {label} {features["subject_ids"][order[feature_index]]} '
            + f'{features["feature_times"][order[feature_index]]} {len(order)} {next_key}'
        )
        indices.append(order[feature_index])
        label_values.append(label.boolean_value)
        label_times.append(label.prediction_time)

    return {
        "boolean_values": np.array(label_values),
        "subject_ids": features["subject_ids"][indices],
        "times": np.array(label_times),
        "features": features["features"][indices, :],
    }
