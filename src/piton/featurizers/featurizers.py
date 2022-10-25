from __future__ import annotations

import datetime
from collections.abc import MutableMapping
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from .core import ColumnValue, Featurizer
from .. import Patient
from ..extension import datasets as extension_datasets

# TODO - replace this with a more flexible/less hacky way to allow the user to 
# manage patient attributes (like age)
def get_patient_birthdate(patient: Patient) -> datetime.datetime:
    return patient.events[0].time if len(patient.events) > 0 else None

class AgeFeaturizer(Featurizer):
    """
    Produces the (possibly normalized) age at each label timepoint.
    """

    def __init__(self, is_normalize: bool = True):
        self.is_normalize = is_normalize
        self.age_statistics = utils.OnlineStatistics()

    def train(self, patient: Patient, labels: List[Label]) -> None:
        if not self.needs_training(): 
            return
        for label in labels:
            self.age_statistics.add(label.time)

    def num_columns(self) -> int:
        return 1

    def featurize(self, patient: Patient, labels: List[Label]) -> List[List[ColumnValue]]:
        all_columns: List[List[ColumnValue]] = []
        # TODO - replace `get_patient_birthdate()`
        patient_birth_date: datetime = get_patient_birthdate(patient)
        for label in labels:
            if self.is_normalize:
                standardized_age = (
                    (label.time - patient_birth_date) - self.age_statistics.mean()
                ) / self.age_statistics.standard_deviation()
                all_columns.append([ColumnValue(0, standardized_age)])
            else:
                all_columns.append([ColumnValue(0, label.time - patient_birth_date)])

        return all_columns

    def to_dict(self) -> Dict[str, Any]:
        return {"age_statistics": self.age_statistics.to_dict()}

    def from_dict(self, data: Mapping[str, Any]) -> None:
        self.age_statistics = utils.OnlineStatistics(data["age_statistics"])

    def needs_training(self) -> bool:
        return self.is_normalize

class CountFeaturizer(Featurizer):
    """
    Produces one column per each diagnosis code, procedure code, and prescription code.
    The value in each column is the count of how many times that code appears in the patient record
    before the corresponding label.
    """

    def __init__(
        self,
        patients: extension_datasets.PatientDatabase,
        ontologies: extension_datasets.Ontology,
        rollup: bool = False,
        exclusion_codes: List[int] = [],
        time_bins: Optional[List[Optional[int]]] = None
    ):
        self.patient_codes: Dictionary = Dictionary()
        self.recorded_date_codes = set(ontologies.get_recorded_date_codes())
        self.exclusion_codes = set(exclusion_codes)
        self.time_bins = time_bins
        self.ontologies = ontologies
        self.rollup = rollup

    def get_codes(self, day: PatientDay) -> Iterator[int]:
        for code in day.observations:
            if (code in self.recorded_date_codes) and (
                code not in self.exclusion_codes
            ):
                if self.rollup:
                    for subcode in self.ontologies.get_subwords(code):
                        yield subcode
                else:
                    yield code
                    
    def train(self, patient: Patient, labels: List[Label]):
        """Adds every event code in this patient's timeline to `patient_codes`
        """        
        for event in patient.events:
            self.patient_codes.add(event.code)

    def num_columns(self) -> int:
        return len(self.patient_codes)

    def featurize(self, patient: Patient, labels: List[Label]) -> List[List[ColumnValue]]:
        all_columns: List[List[ColumnValue]] = []

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

    def featurize(self, patient: Patient, label_indices: Set[int]) -> List[List[ColumnValue]]:
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

    def featurize(
        self, patient: Patient, label_indices: Set[int]
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

class NumericObservationWithValueFeaturizer(Featurizer):
    """
    This featurizer transforms numeric lab values into binned counts.
    The basic idea is that we do a pass over the training data to compute 
    percentiles for the values and then we use those percentiles to 
    create bins for each lab.
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

    def train(self, patient: Patient, label_indices: Set[int]) -> None:
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

    def featurize(
        self, patient: Patient, label_indices: Set[int]
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

class ConstantValueFeaturizer(Featurizer):
    """
        This featurizer returns a constant value for each item.
        It has only one column.
    """

    def __init__(self, value: float):
        self.value = value

    def num_columns(self) -> int:
        return 1

    def featurize(self, patient: Patient, labels: List[Label]) -> List[List[ColumnValue]]:
        all_columns: List[List[ColumnValue]] = []
        for label in labels:
            all_columns.append([ColumnValue(0, self.value)])
        return all_columns
    
class PreprocessedFeaturizer(Featurizer):
    """
        Maps (patient ID, time) -> feature value, as defined in `value_map`
        It has only one column.
    """    
    def __init__(self, value_map: Mapping[Tuple[int, int], float]):
        self.value_map = value_map

    def num_columns(self) -> int:
        return 1

    def featurize(self, patient: Patient, labels: List[Label]) -> List[List[ColumnValue]]:
        all_columns: List[List[ColumnValue]] = []
        for label in labels:
            value = self.value_map[(patient.patient_id, label.time)]
            all_columns.append([ColumnValue(0, value)])
        return all_columns



class IsIcd10Era(Featurizer):
    """
        For each label, return if it occured in the ICD10 era (i.e. year >= 2016)
    """

    def num_columns(self) -> int:
        return 1

    def featurize(self, patient: Patient, labels: List[Label]) -> List[List[ColumnValue]]:
        all_columns: List[List[ColumnValue]] = []
        for label in labels:
            all_columns.append([ColumnValue(0, label.time.year >= 2016)])
        return all_columns