from __future__ import annotations

import datetime
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Iterator, List, Mapping, Optional, Tuple

from .. import Patient
from ..extension import datasets as extension_datasets
from ..labelers.core import Label
from . import Dictionary, OnlineStatistics
from .core import ColumnValue, Featurizer



# TODO - replace this with a more flexible/less hacky way to allow the user to
# manage patient attributes (like age)
def get_patient_birthdate(patient: Patient) -> datetime.datetime:
    return patient.events[0].start if len(patient.events) > 0 else None


class AgeFeaturizer(Featurizer):
    """
    Produces the (possibly normalized) age at each label timepoint.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.age_statistics = OnlineStatistics()

    def preprocess(self, patient: Patient, labels: List[Label]) -> None:
        if not self.needs_preprocessing():
            return

        patient_birth_date = get_patient_birthdate(patient)
        for label in labels:
            age = (label.time - patient_birth_date).days / 365
            self.age_statistics.add(age)

    def num_columns(self) -> int:
        return 1

    def featurize(
        self, patient: Patient, labels: List[Label], ontology: extension_datasets.Ontology,
    ) -> List[List[ColumnValue]]:
        all_columns: List[List[ColumnValue]] = []

        patient_birth_date = get_patient_birthdate(patient)
        for label in labels:
            age = (label.time - patient_birth_date).days / 365
            if self.normalize:
                standardized_age = (age - self.age_statistics.mean()) / (
                    self.age_statistics.standard_deviation()
                )
                all_columns.append([ColumnValue(0, standardized_age)])
            else:
                all_columns.append([ColumnValue(0, age)])

        return all_columns

    def to_dict(self) -> Dict[str, Any]:
        return {"age_statistics": self.age_statistics.to_dict()}

    def from_dict(self, data: Mapping[str, Any]) -> None:
        self.age_statistics = OnlineStatistics(data["age_statistics"])

    def needs_preprocessing(self) -> bool:
        return self.normalize


class CountFeaturizer(Featurizer):
    """
    Produces one column per each diagnosis code, procedure code, and prescription code.
    The value in each column is the count of how many times that code appears in the patient record
    before the corresponding label.
    """

    def __init__(
        self,
        # ontology: extension_datasets.Ontology,
        rollup: bool = False,
        exclusion_codes: List[int] = [],
        time_bins: Optional[
            List[Optional[int]]
        ] = None,  # [90, 180] refers to [0-90, 90-180]; [90, 180, math.inf] refers to [0-90, 90-180, 180-inf]
    ):
        self.patient_codes: Dictionary = Dictionary()
        self.exclusion_codes = set(exclusion_codes)
        self.time_bins = time_bins
        # self.ontology = ontology
        self.rollup = rollup

    def get_codes(self, code: int, ontology: extension_datasets.Ontology) -> Iterator[int]:
        if code not in self.exclusion_codes:
            if self.rollup:
                for subcode in ontology.get_all_parents(code):
                    yield subcode
            else:
                yield code

    def preprocess(self, patient: Patient, labels: List[Label]):
        """Adds every event code in this patient's timeline to `patient_codes`"""
        for event in patient.events:
            if event.value is None:
                self.patient_codes.add(event.code)

    def num_columns(self) -> int:
        if self.time_bins is None:
            return len(self.patient_codes)
        else:
            return len(self.time_bins) * len(self.patient_codes)

    def featurize(
        self, patient: Patient, labels: List[Label], ontology: extension_datasets.Ontology,
    ) -> List[List[ColumnValue]]:
        all_columns: List[List[ColumnValue]] = []

        if self.time_bins is None:
            current_codes: Dict[int, int] = defaultdict(int)

            label_idx = 0
            for event in patient.events:
                while event.start > labels[label_idx].time:
                    label_idx += 1
                    all_columns.append(
                        [
                            ColumnValue(column, count)
                            for column, count in current_codes.items()
                        ]
                    )

                    if label_idx >= len(labels):
                        return all_columns

                if event.value is not None:
                    continue

                for code in self.get_codes(event.code, ontology):
                    if code in self.patient_codes:
                        current_codes[self.patient_codes.transform(code)] += 1

            if label_idx < len(labels):
                for label in labels[label_idx:]:
                    all_columns.append(
                        [
                            ColumnValue(column, count)
                            for column, count in current_codes.items()
                        ]
                    )
                

                # if label_idx == len(labels) - 1:
                #     all_columns.append(
                #         [
                #             ColumnValue(column, count)
                #             for column, count in current_codes.items()
                #         ]
                #     )
                #     break

        else:
            codes_per_bin: Dict[int, Deque[Tuple[int, datetime.date]]] = {
                i: deque() for i in range(len(self.time_bins) + 1)
            }

            code_counts_per_bin: Dict[int, Dict[int, int]] = {
                i: defaultdict(int) for i in range(len(self.time_bins) + 1)
            }

            label_idx = 0
            for event in patient.events:
                code = event.code
                while (
                    label_idx < len(labels)
                    and event.start > labels[label_idx].time
                ):
                    label_idx += 1
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
                for code in self.get_codes(code, ontology):
                    if code in self.patient_codes:
                        codes_per_bin[0].append((code, event.start))
                        code_counts_per_bin[0][code] += 1

                for i, max_time in enumerate(self.time_bins):
                    # if i + 1 == len(self.time_bins):
                    #     continue

                    if max_time is None:
                        # This means that this bin accepts everything
                        continue

                    while len(codes_per_bin[i]) > 0:
                        next_code, next_date = codes_per_bin[i][0]

                        if (event.start - next_date).days <= max_time:
                            break
                        else:
                            codes_per_bin[i + 1].append(
                                codes_per_bin[i].popleft()
                            )

                            code_counts_per_bin[i][next_code] -= 1
                            if code_counts_per_bin[i][next_code] == 0:
                                del code_counts_per_bin[i][next_code]

                            code_counts_per_bin[i + 1][next_code] += 1

                # print(codes_per_bin, " | ", code_counts_per_bin)
                # print()
                if label_idx == len(labels) - 1:
                    all_columns.append(
                        [
                            ColumnValue(
                                self.patient_codes.transform(code)
                                + i * len(self.patient_codes),
                                count,
                            )
                            for i in range(len(self.time_bins) - 1)
                            for code, count in code_counts_per_bin[i].items()
                        ]
                    )
                    break

        return all_columns

    def to_dict(self) -> Dict[str, Any]:
        return {"patient_codes": self.patient_codes.to_dict()}

    def from_dict(self, data: Mapping[str, Any]) -> None:
        self.patient_codes = Dictionary(data["patient_codes"])

    def needs_preprocessing(self) -> bool:
        return True
