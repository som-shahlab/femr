"""Core featurizer functionality, shared across Featurizers."""

from __future__ import annotations

import multiprocessing
from abc import ABC, abstractmethod
from typing import Any, List, Literal, NamedTuple, Optional, Tuple, TypeVar

import numpy as np
import scipy.sparse
from nptyping import NDArray

from femr import Patient
from femr.extension import datasets as extension_datasets
from femr.labelers import Label, LabeledPatients

PatientDatabase = extension_datasets.PatientDatabase
Ontology = extension_datasets.Ontology


class ColumnValue(NamedTuple):
    """A value for a particular column
    `column` is the index for the column
    `value` is the value for that column. Values must be numeric
    """

    column: int
    value: float | int


def _run_featurizer(args: Tuple[str, List[int], LabeledPatients, List[Featurizer]]) -> Tuple[Any, Any, Any, Any]:
    """Apply featurization to the set of patients included in `patient_ids`.
    Gets called as a parallelized subprocess of the .featurize() method of `FeaturizerList`.
    """
    database_path: str = args[0]
    patient_ids: List[int] = args[1]
    labeled_patients: LabeledPatients = args[2]
    featurizers: List[Featurizer] = args[3]

    # Load patients + ontology
    database: PatientDatabase = PatientDatabase(database_path)
    ontology: Ontology = database.get_ontology()

    # Construct CSR sparse matrix
    #   non-zero entries in sparse matrix
    data: List[Any] = []
    #   maps each element in `data`` to its column in the sparse matrix
    indices: List[int] = []
    #   maps each element in `data` and `indices` to the rows of the sparse matrix
    indptr: List[int] = []
    #   tracks Labels
    label_data: List[Tuple] = []

    # For each Patient...
    for patient_id in patient_ids:
        patient: Patient = database[patient_id]  # type: ignore
        labels: List[Label] = labeled_patients.get_labels_from_patient_idx(patient_id)

        if len(labels) == 0:
            continue

        # For each Featurizer, apply it to this Patient...
        columns_by_featurizer: List[List[List[ColumnValue]]] = []
        for featurizer in featurizers:
            # `features` can be thought of as a 2D array (i.e. list of lists),
            # where rows correspond to `labels` and columns to `ColumnValue` (i.e. features)
            features: List[List[ColumnValue]] = featurizer.featurize(patient, labels, ontology)
            assert len(features) == len(labels), (
                f"The featurizer `{featurizer}` didn't generate a set of features for "
                f"every label for patient {patient_id} ({len(features)} != {len(labels)})"
            )
            columns_by_featurizer.append(features)

        for i, label in enumerate(labels):
            indptr.append(len(indices))
            label_data.append(
                (
                    patient_id,  # patient_ids
                    label.value,  # result_labels
                    label.time,  # labeling_time
                )
            )

            # Keep track of starting column for each successive featurizer as we
            # combine their features into one large matrix
            column_offset: int = 0
            for j, feature_columns in enumerate(columns_by_featurizer):
                for column, value in feature_columns[i]:
                    assert 0 <= column < featurizers[j].get_num_columns(), (
                        f"The featurizer {featurizers[j]} provided an out of bounds column for "
                        f"{column} on patient {patient_id} ({column} must be between 0 and "
                        f"{featurizers[j].get_num_columns()})"
                    )
                    indices.append(column_offset + column)
                    data.append(value)

                # Record what the starting column should be for the next featurizer
                column_offset += featurizers[j].get_num_columns()
    # Need one last `indptr` for end of last row in CSR sparse matrix
    indptr.append(len(indices))

    # n_rows = number of Labels across all Patients
    total_rows: int = len(label_data)
    # n_cols = sum of number of columns output by each Featurizer
    total_columns: int = sum(x.get_num_columns() for x in featurizers)

    # Explanation of CSR Matrix: https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr
    np_data: NDArray[Literal["n_total_features, 1"], np.float32] = np.array(data, dtype=np.float32)
    np_indices: NDArray[Literal["n_total_features, 1"], np.int64] = np.array(indices, dtype=np.int64)
    np_indptr: NDArray[Literal["n_labels + 1, 1"], np.int64] = np.array(indptr, dtype=np.int64)

    assert (
        np_indptr.shape[0] == total_rows + 1
    ), f"`indptr` length should be equal to '{total_rows + 1}', but instead is '{np_indptr.shape[0]}"
    assert (
        np_data.shape == np_indices.shape
    ), f"`data` should have equal shape as `indices`, but instead have {np_data.shape} != {np_indices.shape}"
    data_matrix = scipy.sparse.csr_matrix((np_data, np_indices, np_indptr), shape=(total_rows, total_columns))

    label_pids: NDArray[Literal["n_labels, 1"], np.int64] = np.array([x[0] for x in label_data], dtype=np.int64)
    label_values: NDArray[Literal["n_labels, 1"], Any] = np.array([x[1] for x in label_data])
    label_times: NDArray[Literal["n_labels, 1"], np.datetime64] = np.array(
        [x[2] for x in label_data], dtype=np.datetime64
    )
    assert (
        label_pids.shape == label_values.shape == label_times.shape
    ), f"These should all be equal: {label_pids.shape} | {label_values.shape} | {label_times.shape}"

    return data_matrix, label_pids, label_values, label_times


def _run_preprocess_featurizers(args: Tuple[str, List[int], LabeledPatients, List[Featurizer]]) -> List[Featurizer]:
    """Apply preprocessing of featurizers to the set of patients included in `patient_ids`.
    Gets called as a parallelized subprocess of the .preprocess_featurizers() method of `FeaturizerList`.
    """
    database_path: str = args[0]
    patient_ids: List[int] = args[1]
    labeled_patients: LabeledPatients = args[2]
    featurizers: List[Featurizer] = args[3]

    # Load patients
    database: PatientDatabase = PatientDatabase(database_path)

    # Preprocess featurizers on all Labels for each Patient...
    for patient_id in patient_ids:
        patient: Patient = database[patient_id]  # type: ignore
        labels: List[Label] = labeled_patients.get_labels_from_patient_idx(patient_id)

        if len(labels) == 0:
            continue

        # Preprocess featurizers
        for featurizer in featurizers:
            if featurizer.is_needs_preprocessing():
                featurizer.preprocess(patient, labels, database.get_ontology())

    return featurizers


class Featurizer(ABC):
    """A Featurizer takes a Patient and a list of Labels, then returns a row for each timepoint.
    Featurizers must be preprocessed before they are used to compute normalization statistics.
    A sparse representation named ColumnValue is used to represent the values returned by a Featurizer.
    """

    def preprocess(self, patient: Patient, labels: List[Label], ontology: Ontology):
        """Preprocess the featurizer on the given patient and label indices.
        This should do nothing if `is_needs_preprocessing()` returns FALSE,
        i.e. the featurizer doesn't need preprocessing.

        Args:
            patient (Patient): A patient to preprocess on.
            labels (List[Label]): The list of labels of this patient to preprocess on.
        """
        pass

    @classmethod
    def aggregate_preprocessed_featurizers(cls, featurizers: List[FeaturizerType]) -> FeaturizerType:
        """After preprocessing featurizer using multiprocessing, this method aggregates all
        those featurizers into one.

        NOTE: This function only needs to be provided if you are using multiprocessing to
        distribute featurization across multiple processes.

        Note: This runs BEFORE `aggregate_preprocessed_featurizers()`.

        Args:
            featurizers (List[self]): A list of preprocessed featurizers
        """
        return featurizers[0]

    def finalize_preprocessing(self):
        """Finish the featurizer at the end of preprocessing. This is not needed for every
        featurizer, but does become necessary for things like verifying counts, etc.

        Note: This runs AFTER `aggregate_preprocessed_featurizers()`.
        """
        pass

    @abstractmethod
    def get_num_columns(self) -> int:
        """Return the number of columns that this featurizer creates."""
        pass

    @abstractmethod
    def featurize(
        self,
        patient: Patient,
        labels: List[Label],
        ontology: Optional[Ontology],
    ) -> List[List[ColumnValue]]:
        """Featurize the patient such that each label in `labels` has an associated list of features.

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
            patient (Patient): A patient to featurize.
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
    featurize Patients based on their Labels.

    It enables preprocessing of featurizers, featurization, and column name extraction.
    """

    def __init__(self, featurizers: List[Featurizer]):
        """Create a `FeaturizerList` from a sequence of featurizers.

        Args:
            featurizers (List[Featurizer]): The featurizers to use for featurizing patients.
        """
        self.featurizers: List[Featurizer] = featurizers

    def preprocess_featurizers(
        self,
        database_path: str,
        labeled_patients: LabeledPatients,
        num_threads: int = 1,
    ):
        """Preprocess `self.featurizers` on the provided set of `labeled_patients`."""

        # Check if any featurizers need preprocessing. If not, return early.
        any_needs_preprocessing: bool = any(featurizer.is_needs_preprocessing() for featurizer in self.featurizers)
        if not any_needs_preprocessing:
            return

        # Split patients across multiple threads
        patient_ids: List[int] = labeled_patients.get_all_patient_ids()
        patient_ids_per_thread: List[NDArray[np.int64]] = np.array_split(patient_ids, num_threads * 10)
        tasks = [
            (database_path, patient_ids, labeled_patients, self.featurizers) for patient_ids in patient_ids_per_thread
        ]

        # Preprocess in parallel
        with multiprocessing.Pool(num_threads) as pool:
            preprocessed_featurizers: List[Featurizer] = [
                y for x in pool.imap(_run_preprocess_featurizers, tasks) for y in x
            ]

        # Aggregate featurizers
        for idx, featurizer in enumerate(self.featurizers):
            # Merge all featurizers of the same class as `featurizer`
            self.featurizers[idx] = featurizer.aggregate_preprocessed_featurizers(
                [f for f in preprocessed_featurizers if f.__class__.__name__ == featurizer.__class__.__name__]
            )

        # Finalize preprocessing
        for featurizer in self.featurizers:
            featurizer.finalize_preprocessing()

    def featurize(
        self,
        database_path: str,
        labeled_patients: LabeledPatients,
        num_threads: int = 1,
    ) -> Tuple[
        Any,
        NDArray[Literal["n_labels, 1"], np.int64],
        NDArray[Literal["n_labels, 1"], Any],
        NDArray[Literal["n_labels, 1"], np.datetime64],
    ]:
        """
        Apply a list of Featurizers (in sequence) to obtain a feature matrix for each Label for each patient.

        Args:
            database_path (str): Path to `PatientDatabase` on disk

        Returns:
            This returns a tuple (data_matrix, labels, patient_ids, labeling_time).
                data_matrix is a sparse matrix of all the features of all the featurizers.
                label_pids is a list of the patient ids for each row.
                label_values is a list of boolean values representing the labels for each row in the matrix.
                labeling_time is a list of labeling/prediction time for each row.
        """

        patient_ids: List[int] = labeled_patients.get_all_patient_ids()
        patient_ids_per_thread: List[NDArray[np.int64]] = np.array_split(patient_ids, num_threads * 10)
        tasks = [
            (database_path, patient_ids, labeled_patients, self.featurizers)
            for patient_ids in patient_ids_per_thread
            if len(patient_ids) > 0
        ]

        # Run featurizers in parallel
        with multiprocessing.Pool(num_threads) as pool:
            results: List[Tuple[Any, Any, Any, Any]] = list(pool.imap(_run_featurizer, tasks))

        results = [res for res in results if res[2].shape[0] > 0]

        # Join results
        data_matrix = scipy.sparse.vstack([x[0] for x in results])
        label_pids: NDArray[Literal["n_labels, 1"], np.int64] = np.concatenate([x[1] for x in results])
        label_values: NDArray[Literal["n_labels, 1"], Any] = np.concatenate([x[2] for x in results])
        label_times: NDArray[Literal["n_labels, 1"], np.datetime64] = np.concatenate([x[3] for x in results])

        return data_matrix, label_pids, label_values, label_times

    def get_column_name(self, column_idx: int) -> str:
        column_offset: int = 0
        for featurizer in self.featurizers:
            if column_offset <= column_idx < (column_offset + featurizer.get_num_columns()):
                return f"Featurizer {featurizer}, {featurizer.get_column_name(column_idx - column_offset)}"
            column_offset += featurizer.get_num_columns()
        raise IndexError(f"Column index '{column_idx}' out of bounds for this FeaturizerList")
