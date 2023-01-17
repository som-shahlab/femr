from __future__ import annotations

import datetime
import itertools
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Iterator, List, Mapping, Optional, Tuple

from .. import Patient
from ..extension import datasets as extension_datasets
from ..labelers.core import Label
from . import Dictionary, OnlineStatistics
from .core import ColumnValue, Featurizer


# TODO - replace this with a more flexible/less hacky way to allow the user to
# manage patient attributes (like age)
def get_patient_birthdate(patient: Patient) -> Optional[datetime.datetime]:
    return patient.events[0].start if len(patient.events) > 0 else None


class AgeFeaturizer(Featurizer):
    """
    Produces the (possibly normalized) age at each label timepoint.
    """

    def __init__(self, is_normalize: bool = True):
        """
        Args:
            is_normalize (bool, optional): If TRUE, then normalize a patient's age at each
            label across their ages at all labels. Defaults to True.
        """
        self.is_normalize = is_normalize
        self.age_statistics = OnlineStatistics()

    def get_num_columns(self) -> int:
        return 1

    def preprocess(self, patient: Patient, labels: List[Label]):
        """Save the age of this patient (in years) at each label, to use for normalization."""
        if not self.is_needs_preprocessing():
            return

        patient_birth_date: Optional[datetime.datetime] = get_patient_birthdate(
            patient
        )
        if not patient_birth_date:
            return

        for label in labels:
            age_in_yrs: float = (label.time - patient_birth_date).days / 365
            self.age_statistics.add(age_in_yrs)

    @classmethod
    def aggregate_featurizers(  # type: ignore[override]
        cls, featurizers: List[AgeFeaturizer]
    ) -> AgeFeaturizer:
        "After preprocessing featurizer using multiprocessing, this method aggregates all those featurizers into one."
        # Aggregating age featurizers
        for featurizer in featurizers:
            if featurizer.to_dict()["age_statistics"]["current_mean"] != 0:
                new_featurizer = featurizers[0]
                new_featurizer.from_dict(featurizer.to_dict())
                return new_featurizer
        return featurizers[0]

    def featurize(
        self,
        patient: Patient,
        labels: List[Label],
        ontology: Optional[extension_datasets.Ontology],
    ) -> List[List[ColumnValue]]:
        """Return the age of the patient at each label.
        If `is_normalize`, then normalize each label's age across all patient's ages across all their labels."""
        assert (
            ontology is not None
        ), "Ontology cannot be `None` for AgeFeaturizer"
        all_columns: List[List[ColumnValue]] = []
        # Outer list is per label
        # Inner list is the list of features for that label

        patient_birth_date: Optional[datetime.datetime] = get_patient_birthdate(
            patient
        )
        if not patient_birth_date:
            return all_columns

        for label in labels:
            age_in_yrs: float = (label.time - patient_birth_date).days / 365
            if self.is_normalize:
                # age = (age - mean(ages)) / std(ages)
                age_in_yrs = (age_in_yrs - self.age_statistics.mean()) / (
                    self.age_statistics.standard_deviation()
                )
            all_columns.append([ColumnValue(0, age_in_yrs)])

        return all_columns

    def to_dict(self) -> Dict[str, Any]:
        return {
            "age_statistics": self.age_statistics.to_dict(),
            "is_normalize": self.is_normalize,
        }

    def from_dict(self, data: Mapping[str, Any]):
        self.age_statistics = OnlineStatistics(data["age_statistics"])
        self.is_normalize = data["is_normalize"]

    def is_needs_preprocessing(self) -> bool:
        return self.is_normalize


class CountFeaturizer(Featurizer):
    """
    Produces one column per each diagnosis code, procedure code, and prescription code.
    The value in each column is the count of how many times that code appears in the patient record
    before the corresponding label.
    """

    def __init__(
        self,
        is_ontology_expansion: bool = False,
        exclusion_codes: List[int] = [],
        time_bins: Optional[List[float]] = None,
    ):
        """
        Args:
            is_ontology_expansion (bool, optional): _description_. Defaults to False.
            exclusion_codes (List[int], optional): _description_. Defaults to [].
            time_bins (Optional[List[float]], optional): [90, 180] refers to [0-90, 90-180];
                                                         [90, 180, math.inf] refers to [0-90, 90-180, 180-inf]
        """
        self.patient_codes: Dictionary = Dictionary()
        self.exclusion_codes = set(exclusion_codes)
        self.time_bins: Optional[List[float]] = time_bins
        self.is_ontology_expansion: bool = is_ontology_expansion

        if self.time_bins:
            assert len(set(self.time_bins)) == len(
                self.time_bins
            ), "Duplicate entires. Please make sure the entries are unique"
            assert (
                sorted(self.time_bins) == self.time_bins
            ), "Time_bins list must be sorted."

    def get_codes(
        self, code: int, ontology: extension_datasets.Ontology
    ) -> Iterator[int]:
        if code not in self.exclusion_codes:
            if self.is_ontology_expansion:
                for subcode in ontology.get_all_parents(code):
                    yield subcode
            else:
                yield code

    def preprocess(self, patient: Patient, labels: List[Label]):
        """Adds every event code in this patient's timeline to `patient_codes`"""
        for event in patient.events:
            if event.value is None:
                self.patient_codes.add(event.code)

    @classmethod
    def aggregate_featurizers(  # type: ignore[override]
        cls, featurizers: List[CountFeaturizer]
    ) -> CountFeaturizer:
        """After preprocessing featurizer using multiprocessing, this method aggregates all
        those featurizers into one.
        """

        # Aggregating count featurizers
        patient_codes_dict_list = [
            featurizer.to_dict()["patient_codes"]["values"]
            for featurizer in featurizers
        ]
        patient_codes = list(
            itertools.chain.from_iterable(patient_codes_dict_list)
        )

        featurizer_dict = featurizers[0].to_dict()
        featurizer_dict["patient_codes"] = {"values": patient_codes}

        new_featurizer = featurizers[0]
        new_featurizer.from_dict(featurizer_dict)

        return new_featurizer

    def get_num_columns(self) -> int:
        if self.time_bins is None:
            return len(self.patient_codes)
        else:
            return len(self.time_bins) * len(self.patient_codes)

    def featurize(
        self,
        patient: Patient,
        labels: List[Label],
        ontology: Optional[extension_datasets.Ontology],
    ) -> List[List[ColumnValue]]:
        assert (
            ontology is not None
        ), "Ontology cannot be `None` for CountFeaturizer"
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
        else:
            codes_per_bin: Dict[int, Deque[Tuple[int, datetime.datetime]]] = {
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
        return {
            "patient_codes": self.patient_codes.to_dict(),
            "exclusion_codes": self.exclusion_codes,
            "time_bins": self.time_bins,
            "is_ontology_expansion": self.is_ontology_expansion,
        }

    def from_dict(self, data: Mapping[str, Any]):
        self.patient_codes = Dictionary(data["patient_codes"])
        self.exclusion_codes = data.get("exclusion_codes", set())
        self.time_bins = data.get("time_bins", None)
        self.is_ontology_expansion = data.get("is_ontology_expansion", False)

    def is_needs_preprocessing(self) -> bool:
        return True
