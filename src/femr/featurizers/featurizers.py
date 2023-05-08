from __future__ import annotations

import collections
import datetime
import functools
import random
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import numpy as np

from .. import Event, Patient
from ..extension import datasets as extension_datasets
from ..labelers import Label
from . import OnlineStatistics
from .core import ColumnValue, Featurizer


# TODO - replace this with a more flexible/less hacky way to allow the user to
# manage patient attributes (like age)
def get_patient_birthdate(patient: Patient) -> datetime.datetime:
    if len(patient.events) > 0:
        return patient.events[0].start
    raise ValueError("Couldn't find patient birthdate -- Patient has no events")


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
        self.is_normalize: bool = is_normalize
        self.age_statistics: OnlineStatistics = OnlineStatistics()

    def get_num_columns(self) -> int:
        return 1

    def preprocess(self, patient: Patient, labels: List[Label], ontology: ontology_ext.Ontology):
        """Save the age of this patient (in years) at each label, to use for normalization."""
        if not self.is_needs_preprocessing():
            return

        patient_birth_date: Optional[datetime.datetime] = get_patient_birthdate(patient)
        if not patient_birth_date:
            return

        for label in labels:
            age_in_yrs: float = (label.time - patient_birth_date).days / 365
            self.age_statistics.add(age_in_yrs)

    @classmethod
    def aggregate_preprocessed_featurizers(  # type: ignore[override]
        cls, featurizers: List[AgeFeaturizer]
    ) -> AgeFeaturizer:
        """After preprocessing an AgeFeaturizer using multiprocessing (resulting in the list of featurizers
        contained in `featurizers`), this method aggregates all those featurizers into one AgeFeaturizer.

        We need to collect all the means and variances calculated by each individual featurizer,
        then recompute the mean and variance to get an overall mean/variance for the total patient population.
        This is handled by the OnlineStatistics.merge() method.
        """
        if len(featurizers) == 0:
            raise ValueError("You must pass in at least one featurizer to `aggregate_preprocessed_featurizers`")

        # Calculate merged mean/variance/count across each individual featurizer
        merged_stats: OnlineStatistics = OnlineStatistics.merge([f.age_statistics for f in featurizers])
        # Create new featurizer with merged mean/variance/count
        template_featurizer: AgeFeaturizer = featurizers[0]
        aggregated_featurizer: AgeFeaturizer = AgeFeaturizer(template_featurizer.is_normalize)
        aggregated_featurizer.age_statistics = merged_stats
        return aggregated_featurizer

    def featurize(
        self,
        patient: Patient,
        labels: List[Label],
        ontology: Optional[extension_datasets.Ontology],
    ) -> List[List[ColumnValue]]:
        """Return the age of the patient at each label.
        If `is_normalize`, then normalize each label's age across all patient's ages across all their labels."""
        assert ontology is not None, "Ontology cannot be `None` for AgeFeaturizer"
        all_columns: List[List[ColumnValue]] = []
        # Outer list is per label
        # Inner list is the list of features for that label

        patient_birth_date: Optional[datetime.datetime] = get_patient_birthdate(patient)
        if not patient_birth_date:
            return all_columns

        for label in labels:
            age_in_yrs: float = (label.time - patient_birth_date).days / 365
            if self.is_normalize:
                # age = (age - mean(ages)) / std(ages)
                age_in_yrs = (age_in_yrs - self.age_statistics.mean()) / (self.age_statistics.standard_deviation())
            all_columns.append([ColumnValue(0, age_in_yrs)])

        return all_columns

    def is_needs_preprocessing(self) -> bool:
        return self.is_normalize

    def __repr__(self):
        return (
            f"AgeFeaturizer(is_normalize={self.is_normalize}, count={self.age_statistics.current_count}"
            f" mean={self.age_statistics.mean()}, std={self.age_statistics.standard_deviation()})"
        )


def _reshuffle_count_time_bins(
    time_bins: List[datetime.timedelta],
    codes_per_bin: Dict[int, Deque[Tuple[str, datetime.datetime]]],
    code_counts_per_bin: Dict[int, Dict[str, int]],
    label: Label,
):
    # From closest bin to prediction time -> farthest bin
    for bin_idx, bin_end in enumerate(time_bins):
        while len(codes_per_bin[bin_idx]) > 0:
            # Get the least recently added event (i.e. farthest back in patient's timeline
            # from the currently processed label)
            oldest_event_code, oldest_event_start = codes_per_bin[bin_idx][0]

            if (label.time - oldest_event_start) <= bin_end:
                # The oldest event that we're tracking is still within the closest (i.e. smallest distance)
                # bin to our label's prediction time, so all events will be within this bin,
                # so we don't have to worry about shifting events into farther bins (as the code
                # in the `else` clause does)
                break
            else:
                # Goal: Readjust codes so that they fall under the proper time bin
                # Move (oldest_event_code, oldest_event_start) from entry @ `bin_idx`
                # to entry @ `bin_idx + 1`.
                # Basically, move this code from the bin that is closer to the prediction time (`bin_idx`)
                # to a bin that is further away from the prediction time (`bin_idx + 1`)
                codes_per_bin[bin_idx + 1].append(codes_per_bin[bin_idx].popleft())

                # Remove oldest_event_code from current (closer to prediction time) bin `bin_idx`
                code_counts_per_bin[bin_idx][oldest_event_code] -= 1
                # Add oldest_event_code to the (farther from prediction time) bin `bin_idx + 11
                code_counts_per_bin[bin_idx + 1][oldest_event_code] += 1

                # Clear out ColumnValues with a value of 0 to preserve sparsity of matrix
                if code_counts_per_bin[bin_idx][oldest_event_code] == 0:
                    del code_counts_per_bin[bin_idx][oldest_event_code]


class ReservoirSampler:
    def __init__(self, k, rng_seed):
        self.k = k
        self.total = 0
        self.values = []
        self.rng = random.Random(rng_seed)

    def add(self, value):
        if len(self.values) < self.k:
            self.values.append(value)
        else:
            r = self.rng.randint(0, self.total)
            if r < self.k:
                self.values[r] = value

        self.total += 1


def exclusion_helper(event, fallback_function, excluded_codes_set):
    if excluded_codes_set is not None:
        if event.code in excluded_codes_set:
            return True
    if fallback_function is not None:
        return fallback_function(event)

    return False


class CountFeaturizer(Featurizer):
    """
    Produces one column per each diagnosis code, procedure code, and prescription code.
    The value in each column is the count of how many times that code appears in the patient record
    before the corresponding label.
    """

    def __init__(
        self,
        is_ontology_expansion: bool = False,
        excluded_codes: Iterable[str] = [],
        excluded_event_filter: Optional[Callable[[Event], bool]] = None,
        time_bins: Optional[List[datetime.timedelta]] = None,
        numeric_value_decile: bool = False,
        string_value_combination: bool = False,
        characters_for_string_values: int = 100,
    ):
        """
        Args:
            is_ontology_expansion (bool, optional): If TRUE, then do ontology expansion when counting codes.

                Example:
                    If `is_ontology_expansion=True` and your ontology is:
                        Code A -> Code B -> Code C
                    Where "->" denotes "is a parent of" relationship (i.e. A is a parent of B, B is a parent of C).
                    Then if we see 2 occurrences of Code "C", we count 2 occurrences of Code "B" and Code "A".

            excluded_codes (List[str], optional): A list of femr codes that we will ignore. Defaults to [].

            time_bins (Optional[List[datetime.timedelta]], optional): Group counts into buckets.
                Starts from the label time, and works backwards according to each successive value in `time_bins`.

                These timedeltas should be positive values, and will be internally converted to negative values

                If last value is `None`, then the last bucket will be from the penultimate value in `time_bins` to the
                    start of the patient's first event.

                Examples:
                    `time_bins = [
                        datetime.timedelta(days=90),
                        datetime.timedelta(days=180)
                    ]`
                        will create the following buckets:
                            [label time, -90 days], [-90 days, -180 days];
                    `time_bins = [
                        datetime.timedelta(days=90),
                        datetime.timedelta(days=180),
                        datetime.timedelta(years=100)
                    ]`
                        will create the following buckets:
                            [label time, -90 days], [-90 days, -180 days], [-180 days, -100 years];]
        """
        self.is_ontology_expansion: bool = is_ontology_expansion
        self.excluded_event_filter = functools.partial(
            exclusion_helper, fallback_function=excluded_event_filter, excluded_codes_set=set(excluded_codes)
        )
        self.time_bins: Optional[List[datetime.timedelta]] = time_bins
        self.characters_for_string_values: int = characters_for_string_values

        self.numeric_value_decile = numeric_value_decile
        self.string_value_combination = string_value_combination

        if self.time_bins is not None:
            assert len(set(self.time_bins)) == len(
                self.time_bins
            ), f"You cannot have duplicate values in the `time_bins` argument. You passed in: {self.time_bins}"

        self.observed_codes = set()
        self.observed_string_value = collections.defaultdict(int)
        self.observed_numeric_value = collections.defaultdict(functools.partial(ReservoirSampler, 10000, 100))

        self.finalized = False

    def get_codes(self, code: str, ontology: extension_datasets.Ontology) -> Iterator[str]:
        if self.is_ontology_expansion:
            for subcode in ontology.get_all_parents(code):
                yield subcode
        else:
            yield code

    def get_columns(self, event, ontology: extension_datasets.Ontology) -> Iterator[int]:
        if event.value is None:
            for code in self.get_codes(event.code, ontology):
                # If we haven't seen this code before, then add it to our list of included codes
                if code in self.code_to_column_index:
                    yield self.code_to_column_index[code]
        elif type(event.value) is str:
            k = (event.code, event.value[: self.characters_for_string_values])
            if k in self.code_string_to_column_index:
                yield self.code_string_to_column_index[k]
        else:
            if event.code in self.code_value_to_column_index:
                column, quantiles = self.code_value_to_column_index[event.code]
                for i, (start, end) in enumerate(zip(quantiles, quantiles[1:])):
                    if start <= event.value < end:
                        yield i + column

    def preprocess(self, patient: Patient, labels: List[Label], ontology: extension_datasets.Ontology):
        """Add every event code in this patient's timeline to `codes`."""
        for event in patient.events:
            # Check for excluded events
            if self.excluded_event_filter is not None and self.excluded_event_filter(event):
                continue

            if event.value is None:
                for code in self.get_codes(event.code, ontology):
                    # If we haven't seen this code before, then add it to our list of included codes
                    self.observed_codes.add(code)
            elif type(event.value) is str:
                if self.string_value_combination:
                    self.observed_string_value[(event.code, event.value[: self.characters_for_string_values])] += 1
            else:
                if self.numeric_value_decile:
                    self.observed_numeric_value[event.code].add(event.value)

    @classmethod
    def aggregate_preprocessed_featurizers(  # type: ignore[override]
        cls, featurizers: List[CountFeaturizer]
    ) -> CountFeaturizer:
        """After preprocessing a CountFeaturizer using multiprocessing (resulting in the list of featurizers
        contained in `featurizers`), this method aggregates all those featurizers into one CountFeaturizer.

        We need to collect all the unique event codes identified by each featurizer, and then create a new
        featurizer that combines all these codes
        """
        if len(featurizers) == 0:
            raise ValueError("You must pass in at least one featurizer to `aggregate_preprocessed_featurizers`")

        template_featurizer: CountFeaturizer = featurizers[0]

        for featurizer in featurizers[1:]:
            template_featurizer.observed_codes |= featurizer.observed_codes
            for k, v in template_featurizer.observed_string_value.items():
                featurizer.observed_string_value[k] += v
            for k, v in template_featurizer.observed_numeric_value.items():
                featurizer.observed_numeric_value[k].values += v.values

        return template_featurizer

    def finalize(self):
        if self.finalized:
            return

        self.finalized = True
        self.code_to_column_index = {}
        self.code_string_to_column_index = {}
        self.code_value_to_column_index = {}

        self.num_columns = 0

        for code in self.observed_codes:
            self.code_to_column_index[code] = self.num_columns
            self.num_columns += 1

        for (code, val), count in self.observed_string_value.items():
            if count > 1:
                self.code_string_to_column_index[(code, val)] = self.num_columns
                self.num_columns += 1

        for code, values in self.observed_numeric_value.items():
            quantiles = sorted(list(set(np.quantile(values.values, np.linspace(0, 1, num=11)[1:-1]))))
            quantiles = [float("-inf")] + quantiles + [float("inf")]
            self.code_value_to_column_index[code] = (self.num_columns, quantiles)
            self.num_columns += len(quantiles) - 1

    def get_num_columns(self) -> int:
        self.finalize()

        if self.time_bins is None:
            return self.num_columns
        else:
            return self.num_columns * len(self.time_bins)

    def featurize(
        self,
        patient: Patient,
        labels: List[Label],
        ontology: Optional[extension_datasets.Ontology],
    ) -> List[List[ColumnValue]]:
        self.finalize()
        if ontology is None:
            raise ValueError("`ontology` can't be `None` for CountFeaturizer")

        all_columns: List[List[ColumnValue]] = []

        if self.time_bins is None:
            # Count the number of times each code appears in the patient's timeline
            # [key] = column idx
            # [value] = count of occurrences of events with that code (up to the label at `label_idx`)
            code_counter: Dict[int, int] = defaultdict(int)

            label_idx = 0
            for event in patient.events:
                if self.excluded_event_filter is not None and self.excluded_event_filter(event):
                    continue

                while event.start > labels[label_idx].time:
                    label_idx += 1
                    # Create all features for label at index `label_idx`
                    all_columns.append([ColumnValue(code, count) for code, count in code_counter.items()])
                    if label_idx >= len(labels):
                        # We've reached the end of the labels for this patient,
                        # so no point in continuing to count events past this point.
                        # Instead, we just return the counts of all events up to this point.
                        return all_columns

                for column_idx in self.get_columns(event, ontology):
                    code_counter[column_idx] += 1

            # For all labels that occur past the last event, add all
            # events' total counts as these labels' feature values (basically,
            # the featurization of these labels is the count of every single event)
            for _ in labels[label_idx:]:
                all_columns.append([ColumnValue(code, count) for code, count in code_counter.items()])

        else:
            # First, sort time bins in ascending order (i.e. [100 days, 90 days, 1 days] -> [1, 90, 100])
            time_bins: List[datetime.timedelta] = sorted([x for x in self.time_bins if x is not None])

            codes_per_bin: Dict[int, Deque[Tuple[str, datetime.datetime]]] = {
                i: deque() for i in range(len(self.time_bins) + 1)
            }

            code_counts_per_bin: Dict[int, Dict[str, int]] = {
                i: defaultdict(int) for i in range(len(self.time_bins) + 1)
            }

            label_idx = 0
            for event in patient.events:
                if self.excluded_event_filter is not None and self.excluded_event_filter(event):
                    continue
                while event.start > labels[label_idx].time:
                    _reshuffle_count_time_bins(
                        time_bins,
                        codes_per_bin,
                        code_counts_per_bin,
                        labels[label_idx],
                    )
                    label_idx += 1
                    # Create all features for label at index `label_idx`
                    all_columns.append(
                        [
                            ColumnValue(
                                code + i * self.num_columns,
                                count,
                            )
                            for i in range(len(self.time_bins))
                            for code, count in code_counts_per_bin[i].items()
                        ]
                    )

                    if label_idx >= len(labels):
                        # We've reached the end of the labels for this patient,
                        # so no point in continuing to count events past this point.
                        # Instead, we just return the counts of all events up to this point.
                        return all_columns

                for column_idx in self.get_columns(event, ontology):
                    codes_per_bin[0].append((column_idx, event.start))
                    code_counts_per_bin[0][column_idx] += 1

            for label in labels[label_idx:]:
                _reshuffle_count_time_bins(
                    time_bins,
                    codes_per_bin,
                    code_counts_per_bin,
                    label,
                )
                all_columns.append(
                    [
                        ColumnValue(
                            code + i * self.num_columns,
                            count,
                        )
                        for i in range(len(self.time_bins))
                        for code, count in code_counts_per_bin[i].items()
                    ]
                )

        return all_columns

    def is_needs_preprocessing(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"CountFeaturizer(number of included codes={self.self.num_columns})"

    def get_column_name(self, column_idx: int) -> str:
        def helper(actual_idx):
            for code, idx in self.code_to_column_index.items():
                if idx == actual_idx:
                    return code
            for (code, val), idx in self.code_string_to_column_index.items():
                if idx == actual_idx:
                    return f"{code} {val}"

            for code, (idx, quantiles) in self.code_value_to_column_index.items():
                offset = actual_idx - idx
                if 0 <= offset < len(quantiles) - 1:
                    return f"{code} [{quantiles[offset]}, {quantiles[offset+1]})"

            raise RuntimeError("Could not find name for " + str(actual_idx))

        if self.time_bins is None:
            return helper(column_idx)
        else:
            return helper(column_idx % self.num_columns) + f"_{self.time_bins[column_idx // self.num_columns]}"
