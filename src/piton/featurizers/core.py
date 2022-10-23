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

import numpy as np
import scipy.sparse

from . import labeler, ontology, timeline, utils

ColumnValue = namedtuple("ColumnValue", ["column", "value"])
"""A value for a particular column
.. py:attribute:: column
    The index for the column
.. py:attribute:: value
    The value for that column
"""

class FeaturizerList:
    """
        Featurizer list consists of a list of featurizers to be used to featurize data.
        It enables training, featurization, column name extraction and serialization/deserialization.
    """

    def __init__(self, featurizers: List[Featurizer]):
        """Create the FeaturizerList from a sequence of featurizers.
        Args:
            featurizers (list of :class:`Featurizer`): The featurizers to use for
                transforming the patients.
        """
        self.featurizers = featurizers

    def train_featurizers(
        self,
        timelines: timeline.TimelineReader,
        labeler: labeler.Labeler,
        end_date: Optional[datetime.date] = None,
    ) -> None:
        """
        Train a list of featurizers on the provided patients using the given labeler.
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
                columns = featurizer.transform(patient, label_indices)
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
    def train(self, patient: timeline.Patient, label_indices: Set[int]) -> None:
        """
        Train the featurizer on the given patients and label indices.
        This should do nothing if the featurizer doesn't need training.
        Args:
            patient: A patient to train on.
            label_indices (:obj:set: of int): The set of indices for that patient.
        """
        pass

    def finalize_training(self) -> None:
        """
        Finish the featurizer at the end of training. This is not needed for every
        featurizer, but does become necessary for things like verifying counts, etc.
        """
        pass  # The default version does nothing

    @abstractmethod
    def num_columns(self) -> int:
        """
        Returns: The number of columns that this featurizer creates.
        """

    @abstractmethod
    def transform(
        self, patient: timeline.Patient, label_indices: Set[int]
    ) -> List[List[ColumnValue]]:
        """
        Transform a patient into a series of rows using the specified timepoints.
        Args:
            patient: The patient to train on.
            label_indices (:obj:set of int): The indices which will be labeled.
        Returns:
            A list of rows. Each row in turn is a list of :class:`ColumnValues<ColumnValue>` for the
            values in each column.
        """

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the featurizer to a JSON compatible dict
        Returns:
            A JSON compatible dict.
        """
        return {}

    def from_dict(self, data: Mapping[str, Any]) -> None:
        """
        Restore the state of the featurizer from a JSON compatible dict.
        Args:
            data: A JSON compatible dict from to_dict
        """
        pass

    def get_column_name(self, column_index: int) -> str:
        """
        An optional method that enables the user to get the name of a column.
        Args:
            column_index: The index of the column
        """
        return "no name"

    def needs_training(self) -> bool:
        return False


###########################################
# Useful featurizers
###########################################


class AgeFeaturizer(Featurizer):
    """
    Produces the (possibly normalized) age at the prediction timepoint.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.age_statistics = utils.OnlineStatistics()

    def train(self, patient: timeline.Patient, label_indices: Set[int]) -> None:
        if self.normalize:
            for i, day in enumerate(patient.days):
                if i in label_indices:
                    self.age_statistics.add(day.age)

    def num_columns(self) -> int:
        return 1

    def transform(
        self, patient: timeline.Patient, label_indices: Set[int]
    ) -> List[List[ColumnValue]]:
        all_columns = []

        for i, day in enumerate(patient.days):
            if i in label_indices:
                if self.normalize:
                    standardized_age = (
                        day.age - self.age_statistics.mean()
                    ) / self.age_statistics.standard_deviation()
                    all_columns.append([ColumnValue(0, standardized_age)])
                else:
                    all_columns.append([ColumnValue(0, day.age)])

        return all_columns

    def to_dict(self) -> Dict[str, Any]:
        return {"age_statistics": self.age_statistics.to_dict()}

    def from_dict(self, data: Mapping[str, Any]) -> None:
        self.age_statistics = utils.OnlineStatistics(data["age_statistics"])

    def needs_training(self) -> bool:
        return self.normalize


class IsIcd10Era(Featurizer):
    """
    Produces the (possibly normalized) age at the prediction timepoint.
    """

    def num_columns(self) -> int:
        return 1

    def transform(
        self, patient: timeline.Patient, label_indices: Set[int]
    ) -> List[List[ColumnValue]]:
        all_columns = []

        for i, day in enumerate(patient.days):
            if i in label_indices:
                all_columns.append([ColumnValue(0, day.date.year >= 2016)])

        return all_columns


class CountFeaturizer(Featurizer):
    """
    Produces one column per each diagnosis code, procedure code or prescription code.
    The value in each column is the count of how many times that code appears in the patient record
    up until the prediction time.
    Note: time_bins should be a list optionally ending with None
    Each integer in time_bins represents the end point for a particular bin. A final bin with None represents
    a final bin which enables codes from any point in history.
    """

    def __init__(
        self,
        timelines: timeline.TimelineReader,
        ontologies: ontology.OntologyReader,
        rollup: bool = False,
        exclusion_codes: List[int] = [],
        time_bins: Optional[List[Optional[int]]] = None,
    ):
        self.patient_codes: utils.Dictionary[int] = utils.Dictionary()
        self.recorded_date_codes = set(ontologies.get_recorded_date_codes())
        self.exclusion_codes = set(exclusion_codes)
        self.time_bins = time_bins
        self.ontologies = ontologies
        self.rollup = rollup

    def get_codes(self, day: timeline.PatientDay) -> Iterator[int]:
        for code in day.observations:
            if (code in self.recorded_date_codes) and (
                code not in self.exclusion_codes
            ):
                if self.rollup:
                    for subcode in self.ontologies.get_subwords(code):
                        yield subcode
                else:
                    yield code

    def train(self, patient: timeline.Patient, label_indices: Set[int]) -> None:
        for day in patient.days:
            for code in self.get_codes(day):
                self.patient_codes.add(code)

    def num_columns(self) -> int:
        if self.time_bins is None:
            return len(self.patient_codes)
        else:
            return len(self.time_bins) * len(self.patient_codes)

    def transform(
        self, patient: timeline.Patient, label_indices: Set[int]
    ) -> List[List[ColumnValue]]:
        all_columns = []

        if self.time_bins is None:
            current_codes: Dict[int, int] = defaultdict(int)

            for i, day in enumerate(patient.days):
                for code in self.get_codes(day):
                    if code in self.patient_codes:
                        current_codes[self.patient_codes.transform(code)] += 1

                if i in label_indices:
                    all_columns.append(
                        [
                            ColumnValue(column, count)
                            for column, count in current_codes.items()
                        ]
                    )
        else:
            codes_per_bin: Dict[int, Deque[Tuple[int, datetime.date]]] = {
                i: deque() for i in range(len(self.time_bins))
            }

            code_counts_per_bin: Dict[int, Dict[int, int]] = {
                i: defaultdict(int) for i in range(len(self.time_bins))
            }

            for day_index, day in enumerate(patient.days):
                python_date = datetime.date(
                    day.date.year, day.date.month, day.date.day
                )
                for code in self.get_codes(day):
                    if code in self.patient_codes:
                        codes_per_bin[0].append((code, python_date))
                        code_counts_per_bin[0][code] += 1

                for i, max_time in enumerate(self.time_bins):
                    if max_time is None:
                        # This means that this bin accepts everything
                        continue

                    while len(codes_per_bin[i]) > 0:
                        next_code, next_date = codes_per_bin[i][0]

                        if (python_date - next_date).days <= max_time:
                            break
                        else:
                            codes_per_bin[i + 1].append(
                                codes_per_bin[i].popleft()
                            )

                            code_counts_per_bin[i][next_code] -= 1
                            if code_counts_per_bin[i][next_code] == 0:
                                del code_counts_per_bin[i][next_code]

                            code_counts_per_bin[i + 1][next_code] += 1

                if day_index in label_indices:
                    all_columns.append(
                        [
                            ColumnValue(
                                self.patient_codes.transform(code)
                                + i * len(self.patient_codes),
                                count,
                            )
                            for i in range(len(self.time_bins))
                            for code, count in code_counts_per_bin[i].items()
                        ]
                    )

        return all_columns

    def to_dict(self) -> Dict[str, Any]:
        return {"patient_codes": self.patient_codes.to_dict()}

    def from_dict(self, data: Mapping[str, Any]) -> None:
        self.patient_codes = utils.Dictionary(data["patient_codes"])

    def needs_training(self) -> bool:
        return True


class BinaryFeaturizer(CountFeaturizer):
    """
        Behaves like CountFeaturizer except all non-zero counts receive a value of 1.
    """

    def transform(
        self, patient: timeline.Patient, label_indices: Set[int]
    ) -> List[List[ColumnValue]]:
        all_columns = []

        current_codes = defaultdict(int)

        for i, day in enumerate(patient.days):
            for code in self.get_codes(day):
                if code in self.patient_codes:
                    current_codes[self.patient_codes.transform(code)] = 1

            if i in label_indices:
                all_columns.append(
                    [
                        ColumnValue(column, count)
                        for column, count in current_codes.items()
                    ]
                )

        return all_columns


class LabelerDerivedFeaturizer(Featurizer):
    def __init__(self, label: labeler.Labeler):
        self.label = label

    def num_columns(self) -> int:
        return 1

    def transform(
        self, patient: timeline.Patient, label_indices: Set[int]
    ) -> List[List[ColumnValue]]:
        result = []

        my_labels = self.label.label(patient)

        label_dict = {
            my_label.day_index: my_label.is_positive for my_label in my_labels
        }

        for i, day in enumerate(patient.days):
            if i in label_indices:
                feature = label_dict[i]
                result.append([ColumnValue(0, feature)])

        return result


class ConstantValueFeaturizer(Featurizer):
    """
    This featurizer returns a constant value for each item.
    It has only one column.
    """

    def __init__(self, value: float):
        self.value = value

    def num_columns(self) -> int:
        return 1

    def transform(
        self, patient: timeline.Patient, label_indices: Set[int]
    ) -> List[List[ColumnValue]]:
        result = []

        for i, day in enumerate(patient.days):
            if i in label_indices:
                result.append([ColumnValue(0, self.value)])

        return result


class PreprocessedFeaturizer(Featurizer):
    def __init__(self, value_map: Mapping[Tuple[int, int], float]):
        self.value_map = value_map

    def num_columns(self) -> int:
        return 1

    def transform(
        self, patient: timeline.Patient, label_indices: Set[int]
    ) -> List[List[ColumnValue]]:
        result = []

        for i, day in enumerate(patient.days):
            if i in label_indices:
                value = self.value_map[(patient.patient_id, i)]
                result.append([ColumnValue(0, value)])

        return result


class NumericObservationWithValueFeaturizer(Featurizer):
    """
    This featurizer transforms numeric lab values into binned counts.
    The basic idea is that we do a pass over the training data to compute percentiles for the values and then
    we use those percentiles to create bins for each lab.
    """

    def __init__(
        self,
        timelines: timeline.TimelineReader,
        ontologies: ontology.OntologyReader,
        min_labs_per_bin: int = 1,
        num_bins: int = 10,
    ):
        self.recorded_date_codes = set(ontologies.get_recorded_date_codes())
        self.observedNumericValues: Dict[int, List[float]] = defaultdict(list)
        self.min_labs_per_bin = min_labs_per_bin
        self.num_bins = num_bins

    def train(self, patient: timeline.Patient, label_indices: Set[int]) -> None:
        for day in patient.days:
            for codeWithValue in day.observations_with_values:
                if codeWithValue.code in self.recorded_date_codes:
                    if not codeWithValue.is_text:
                        self.observedNumericValues[codeWithValue.code].append(
                            codeWithValue.numeric_value
                        )

    def needs_training(self) -> bool:
        return True

    def get_percentile(self, item: float, percentiles: List[float]) -> int:
        """Get the index for the given percentiles.
        Note: There is one bin for each value in percentiles that starts at that value
        """
        for i, p in enumerate(percentiles):
            if item < p:
                return i - 1

        return len(percentiles) - 1

    def finalize_training(self) -> None:
        self.code_numeric_dictionary = {}
        self.next_index = 0

        for code, values in self.observedNumericValues.items():
            values.sort()
            percentiles = [-float("inf")]

            for i in range(self.num_bins - 1):
                next_value = values[
                    min(
                        round((len(values) - 1) * (i + 1) / self.num_bins),
                        len(values) - 1,
                    )
                ]
                percentiles.append(next_value)

            counts = [0 for _ in range(len(percentiles))]

            for item in values:
                counts[self.get_percentile(item, percentiles)] += 1

            filtered_percentiles = []
            current_low: Optional[float] = None

            for i, p in enumerate(percentiles):
                if counts[i] >= self.min_labs_per_bin:
                    if current_low is not None:
                        filtered_percentiles.append(current_low)
                        current_low = None
                    else:
                        filtered_percentiles.append(p)
                elif counts[i] < self.min_labs_per_bin:
                    # We are skipping this one as there are too few counts
                    if current_low is None:
                        current_low = p
                    if (i + 1) < len(percentiles):
                        counts[i + 1] += counts[i]

            if len(filtered_percentiles) == 0:
                continue

            indices_for_percentiles = list(
                range(
                    self.next_index, self.next_index + len(filtered_percentiles)
                )
            )
            self.next_index += len(filtered_percentiles)

            self.code_numeric_dictionary[code] = (
                filtered_percentiles,
                indices_for_percentiles,
            )

    def num_columns(self) -> int:
        return self.next_index

    def transform(
        self, patient: timeline.Patient, label_indices: Set[int]
    ) -> List[List[ColumnValue]]:
        all_columns = []

        current_codes: Dict[int, int] = defaultdict(int)

        for i, day in enumerate(patient.days):
            for codeWithValue in day.observations_with_values:
                if codeWithValue.code in self.code_numeric_dictionary:
                    if not codeWithValue.is_text:
                        (
                            percentiles,
                            indices_for_percentiles,
                        ) = self.code_numeric_dictionary[codeWithValue.code]
                        offset = self.get_percentile(
                            codeWithValue.numeric_value, percentiles
                        )
                        current_codes[indices_for_percentiles[offset]] += 1

            if i in label_indices:
                all_columns.append(
                    [
                        ColumnValue(column, count)
                        for column, count in current_codes.items()
                    ]
                )

        return all_columns

    def to_dict(self) -> Dict[str, Any]:
        return {
            "next_index": self.next_index,
            "code_numeric_dictionary": list(
                self.code_numeric_dictionary.items()
            ),
        }

    def from_dict(self, data: Mapping[str, Any]) -> None:
        self.next_index = data["next_index"]

        self.code_numeric_dictionary = {
            code: values for code, values in data["code_numeric_dictionary"]
        }