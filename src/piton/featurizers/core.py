''' RAHUL '''
from __future__ import annotations

from abc import ABC, abstractmethod
import collections
from typing import Any, List, Optional, Tuple, Dict, Mapping
import multiprocessing

from nptyping import NDArray, Shape

import numpy as np
import scipy.sparse

from .. import Patient
from ..labelers.core import Label, LabeledPatients
import itertools
from piton.extension import datasets as extension_datasets
PatientDatabase = extension_datasets.PatientDatabase
Ontology = extension_datasets.Ontology


ColumnValue = collections.namedtuple("ColumnValue", ["column", "value"])
"""A value for a particular column
.. py:attribute:: column
    The index for the column
.. py:attribute:: value
    The value for that column
"""

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
    data: List[Any] = [] # non-zero entries in sparse matrix
    indices: List[int] = [] # maps each element in `data`` to its column in the sparse matrix
    indptr: List[int] = [] # maps each element in `data` and `indices` to the rows of the sparse matrix
    label_data: List[Tuple] = []
    
    # For each Patient...
    for patient_id in patient_ids:
        patient: Patient = database[patient_id] # type: ignore
        labels: List[Label] = labeled_patients.get_labels_from_patient_idx(patient_id)

        if len(labels) == 0:
            continue

        # Keep track of starting column for each successive featurizer as we combine their features
        # into one large matrix
        column_offset: int = 0 
        # For each Featurizer, apply it to this Patient...
        for featurizer in featurizers:
            # `features` can be thought of as a 2D array (i.e. list of lists),
            # where rows correspond to `labels` and columns to `ColumnValue` (i.e. features)
            features: List[List[ColumnValue]] = featurizer.featurize(patient, labels, ontology)
            assert len(features) == len(labels), (
                f"The featurizer `{featurizer}` didn't generate a set of features for every label for patient {patient_id} ({len(features)} != {len(labels)})"
            )
            
            # For each Label, add a row for its features in our CSR sparse matrix...
            for (label, label_features) in zip(labels, features):
                # `data[indptr[i]:indptr[i+1]]` is the data corresponding to row `i` in CSR sparse matrix
                # `indices[indptr[i]:indptr[i+1]]` is the columns corresponding to row `i` in CSR sparse matrix
                indptr.append(len(indices))
                label_data.append((
                    patient_id, # patient_ids
                    label.value, # result_labels
                    label.time, # labeling_time
                ))
                
                for (column, value) in label_features:
                    assert 0 <= column < featurizer.num_columns(), (
                        f"The featurizer {featurizer} provided an out of bounds column for "
                        f"{column} on patient {patient_id} ({column} must be between 0 and "
                        f"{featurizer.num_columns()})"
                    )
                    indices.append(column_offset + column)
                    data.append(value)
            
            # Record what the starting column should be for the next featurizer
            column_offset += featurizer.num_columns()
    indptr.append(len(indices)) # Need one last `indptr` for end of last row in CSR sparse matrix

    # Explanation of CSR Matrix: https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr
    np_data: NDArray[Shape["n_total_features, 1"], np.float] = np.array(data, dtype=np.float)
    np_indices: NDArray[Shape["n_total_features, 1"], np.int] = np.array(indices, dtype=np.int)
    np_indptr: NDArray[Shape["n_labels + 1, 1"], np.int] = np.array([ x[0] for x in label_data ], dtype=np.int)
    # n_rows = number of Labels across all Patients
    total_rows: int = len(label_data)
    # n_cols = sum of number of columns output by each Featurizer
    total_columns: int = sum(x.num_columns() for x in featurizers)
    assert np_indptr.shape[0] == total_rows + 1, f"`indptr` length should be equal to '{total_rows + 1}', but instead is '{np_indptr.shape[0]}"
    assert np_data.shape == np_indices.shape, f"`data` should have equal shape as `indices`, but instead have {np_data.shape} != {np_indices.shape}"
    data_matrix = scipy.sparse.csr_matrix(
        (np_data, np_indices, np_indptr), shape=(total_rows, total_columns)
    )

    label_pids: NDArray[Shape["n_labels, 1"], np.int] = np.array([ x[1] for x in label_data ], dtype=np.int)
    label_values: NDArray[Shape["n_labels, 1"], Any] = np.array([ x[2] for x in label_data ])
    label_times: NDArray[Shape["n_labels, 1"], np.datetime64]  = np.array([ x[3] for x in label_data ], dtype=np.datetime64)
    assert label_pids.shape == label_values.shape == label_times.shape, f"These should all be equal: {label_pids.shape} | {label_values.shape} | {label_times.shape}"
    
    data_matrix.check_format() # remove when we think its works
    
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
        patient: Patient = database[patient_id] # type: ignore
        labels: List[Label] = labeled_patients.get_labels_from_patient_idx(patient_id)

        if len(labels) == 0:
            continue
        
        # Preprocess featurizers
        for featurizer in featurizers:
            if featurizer.is_needs_preprocessing():
                featurizer.preprocess(patient, labels)
        
    return featurizers

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

    # TODO - restructure code so that multiprocessing happens within the featurizer
    # (and thus no need to do a bespoke merge of results here)
    def preprocess_featurizers(
        self,
        database_path: str,
        labeled_patients: LabeledPatients,
        num_threads: int = 1,
    ):
        """Preprocess `self.featurizers` on the provided set of `labeled_patients`."""

        any_needs_preprocessing: bool = any(
            featurizer.is_needs_preprocessing() for featurizer in self.featurizers
        )
        if not any_needs_preprocessing:
            return

        patient_ids: List[int] = labeled_patients.get_all_patient_ids()
        patient_ids_per_thread: List[NDArray[np.int]] = np.array_split(patient_ids, num_threads)
        tasks = [ (database_path, patient_ids, labeled_patients, self.featurizers) 
                 for patient_ids in patient_ids_per_thread ]

        # Preprocess in parallel
        ctx = multiprocessing.get_context('forkserver')
        with ctx.Pool(num_threads) as pool:
            trained_featurizers: List[Featurizer] = [ y for x in pool.imap(_run_preprocess_featurizers, tasks) for y in x ]
        grouped_featurizers: collections.defaultdict = collections.defaultdict(list)
        for featurizer in trained_featurizers:
            grouped_featurizers[featurizer.get_name()].append(featurizer)
    
        # Aggregating age featurizers
        age_featurizer_idx: int = [ idx for idx, f in enumerate(self.featurizers) if f.get_name() == 'AgeFeaturizer' ][0]
        for featurizer in grouped_featurizers.get("AgeFeaturizer", []):
            if featurizer.to_dict()["age_statistics"]["current_mean"] != 0:
                self.featurizers[age_featurizer_idx].from_dict(featurizer.to_dict())
                break
        
        # Aggregating count featurizers
        count_featurizer_idx: int = [ idx for idx, f in enumerate(self.featurizers) if f.get_name() == 'CountFeaturizer' ][0]
        if "CountFeaturizer" in grouped_featurizers:
            patient_codes_dict_list = [
                f.to_dict()["patient_codes"]["values"] for f in grouped_featurizers["CountFeaturizer"]
            ]
            patient_codes = list(itertools.chain.from_iterable(patient_codes_dict_list))
            self.featurizers[count_featurizer_idx].from_dict({
                "patient_codes": { "values": patient_codes }
            })

        for featurizer in self.featurizers:
            featurizer.finalize_preprocessing()

    def featurize(
        self,
        database_path: str,
        labeled_patients: LabeledPatients,
        num_threads: int = 1,
    ) -> Tuple[Any, Any, Any, Any]:
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
        patient_ids_per_thread: List[NDArray[np.int]] = np.array_split(patient_ids, num_threads)
        tasks = [ (database_path, patient_ids, labeled_patients, self.featurizers) 
                 for patient_ids in patient_ids_per_thread ]

        # Run featurizers in parallel
        ctx = multiprocessing.get_context('forkserver')
        with ctx.Pool(num_threads) as pool:
            results: List[Tuple[Any, Any, Any, Any]] = list(pool.imap(_run_featurizer, tasks))
        
        # Join results
        data_matrix = scipy.sparse.vstack([ x[0] for x in results ])
        label_pids: NDArray[Shape["n_labels, 1"], np.int] = np.concatenate([ x[1] for x in results ])
        label_values: NDArray[Shape["n_labels, 1"], Any] = np.concatenate([ x[2] for x in results ])
        label_times: NDArray[Shape["n_labels, 1"], np.datetime64] = np.concatenate([ x[3] for x in results ])
        
        return data_matrix, label_pids, label_values, label_times

    def get_column_name(self, column_idx: int) -> str:
        column_offset: int = 0
        for featurizer in self.featurizers:
            if column_offset <= column_idx < (column_offset + featurizer.get_num_columns()):
                return f"Featurizer {featurizer}, {featurizer.get_column_name(column_idx - column_offset)}"
            column_offset += featurizer.get_num_columns()
        raise IndexError(f"Column index '{column_idx}' out of bounds for this FeaturizerList")

class Featurizer(ABC):
    """A Featurizer takes a Patient and a list of Labels, then returns a row for each timepoint.
    Featurizers must be preprocessed before they are used to compute normalization statistics.
    A sparse representation named ColumnValue is used to represent the values returned by a Featurizer.
    """

    # TODO - rename to 'train' ??
    def preprocess(self, patient: Patient, labels: List[Label]):
        """Preprocess the featurizer on the given patient and label indices.
        This should do nothing if `is_needs_preprocessing()` returns FALSE, i.e. the featurizer doesn't need preprocessing.

        Args:
            patient (Patient): A patient to preprocess on.
            labels (List[Label]): The list of labels of this patient to preprocess on.
        """
        pass

    def finalize_preprocessing(self):
        """Finish the featurizer at the end of preprocessing. This is not needed for every
        featurizer, but does become necessary for things like verifying counts, etc.
        """
        pass

    @abstractmethod
    def get_num_columns(self) -> int:
        """Return the number of columns that this featurizer creates."""
        pass

    @abstractmethod
    def featurize(
        self, patient: Patient, labels: List[Label], ontology: Optional[Ontology],
    ) -> List[List[ColumnValue]]:
        """Featurize the patient such that each label in `labels` has an associated list of features.
       
        Example:
            return [
                [ ColumnValue(0, 10), ColumnValue(3, 12), ], # features for label 0
                [ ColumnValue(1, 'hi') ], # features for label 1
                [ ColumnValue(2, 2), ColumnValue(1, 3), ColumnValue(10, 3), ], # features for label 2
                ...
                [ ColumnValue(8, True), ColumnValue(9, False), ], # features for label n
            ]

        Where each ColumnValue is of the form: (idx of column for this feature, feature value).
        
        Thus, the List[List[ColumnValue]] represents a 2D sparse matrix, where each row is a distinct Label
        and each (sparse) column is a feature

        Args:
            patient (Patient): A patient to featurize.
            labels (List[Label]): We will generate features for each Label in `labels`.
            ontology (Optional[Ontology]): Ontology for Event codes.

        Returns:
             List[List[ColumnValue]]: A list of 'features' (where 'features' is a list itself) for each Label.
                The length of this list of lists == length of `labels`
                    [idx] = corresponds to the Label at `labels[idx]`
                    [value] = List of :class:`ColumnValues<ColumnValue>` which contain the features for this label
        """
        pass

    def get_column_name(self, column_idx: int) -> str:
        """Enable the user to get the name of a column by its index

        Args:
            column_idx (int): The index of the column
        """
        return "no name"

    def is_needs_preprocessing(self) -> bool:
        """Return TRUE if you must run `preprocess()`. If FALSE, then `preprocess()` should do nothing."""
        return False

    def get_name(self) -> str:
        """Return a unique identifier for this Featurizer."""
        return "no name"

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary representation."""
        return {}

    def from_dict(self, data: Mapping[str, Any]):
        """Convert dictionary representation of Featurizer to object."""
        pass




