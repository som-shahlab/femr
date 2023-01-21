from __future__ import annotations

import datetime
from collections import defaultdict, deque
from typing import Deque, Dict, Iterator, List, Optional, Tuple, Union

from .. import Patient
from ..extension import datasets as extension_datasets
from ..labelers.core import Label
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
            raise ValueError(
                "You must pass in at least one featurizer to `aggregate_preprocessed_featurizers`"
            )

        # Calculate merged mean/variance/count across each individual featurizer
        merged_stats: OnlineStatistics = OnlineStatistics.merge(
            [f.age_statistics for f in featurizers]
        )
        # Create new featurizer with merged mean/variance/count
        template_featurizer: AgeFeaturizer = featurizers[0]
        aggregated_featurizer: AgeFeaturizer = AgeFeaturizer(
            template_featurizer.is_normalize
        )
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
        excluded_codes: Union[set, List[int]] = [],
        included_codes: Union[set, List[int]] = [],
        time_bins: Optional[List[datetime.timedelta]] = None,
        is_keep_only_none_valued_events: bool = True,
    ):
        """
        Args:
            is_ontology_expansion (bool, optional): If TRUE, then do ontology expansion when counting codes.

                Example:
                    If `is_ontology_expansion=True` and your ontology is:
                        Code A -> Code B -> Code C
                    Where "->" denotes "is a parent of" relationship (i.e. A is a parent of B, B is a parent of C).
                    Then if we see 2 occurrences of Code "C", we will count 2 occurrences of Code "B" and Code "A" as well.

            excluded_codes (List[int], optional): A list of Piton codes that we will ignore. Defaults to [].

            included_codes (List[int], optional): A list of all unique event codes that we will include in our count. Defaults to [].

            time_bins (Optional[List[datetime.timedelta]], optional): Group counts into buckets. Starts from the prediction time,
                and works backwards according to each successive value in `time_bins`.
                NOTE: These timedeltas should be positive values, and they will be converted to negative values internally.
                NOTE: If last value is `None`, then the last bucket will be from the penultimate value in `time_bins` to the
                    start of the patient's first event.

                Examples:
                    `time_bins = [datetime.timedelta(days=90), datetime.timedelta(days=180)]` will create the following buckets:
                        [label time, -90 days], [-90 days, -180 days];
                    `time_bins = [datetime.timedelta(days=90), datetime.timedelta(days=180), datetime.timedelta(years=100)]` will create the following buckets:
                        [label time, -90 days], [-90 days, -180 days], [-180 days, -100 years];]

            is_keep_only_none_valued_events (bool): If TRUE, then only keep events that have no value (i.e. `event.value is None`). Defaults to True.
                Setting this to FALSE will include all events, including those with values (e.g. lab values), which will make
                this run slower.
        """
        self.is_ontology_expansion: bool = is_ontology_expansion
        self.included_codes: set = (
            set(included_codes)
            if not isinstance(included_codes, set)
            else included_codes
        )
        self.excluded_codes: set = (
            set(excluded_codes)
            if not isinstance(excluded_codes, set)
            else excluded_codes
        )
        self.time_bins: Optional[List[datetime.timedelta]] = time_bins
        self.is_keep_only_none_valued_events: bool = (
            is_keep_only_none_valued_events
        )

        # Map code to its feature's corresponding column index
        # NOTE: Must be sorted to preserve set ordering across instantiations
        self.code_to_column_index: Dict[int, int] = {
            code: idx for idx, code in enumerate(sorted(self.included_codes))
        }

        if self.time_bins is not None:
            assert len(set(self.time_bins)) == len(
                self.time_bins
            ), f"You cannot have duplicate values in the `time_bins` argument. You passed in: {self.time_bins}"

    def get_codes(
        self, code: int, ontology: extension_datasets.Ontology
    ) -> Iterator[int]:
        if code not in self.excluded_codes:
            if self.is_ontology_expansion:
                for subcode in ontology.get_all_parents(code):
                    yield subcode
            else:
                yield code

    def preprocess(self, patient: Patient, labels: List[Label]):
        """Add every event code in this patient's timeline to `codes`.

        If `self.is_keep_only_none_valued_events` is TRUE, then only add events that have no value (i.e. `event.value is None`).
        """
        for event in patient.events:
            if self.is_keep_only_none_valued_events and event.value is not None:
                # If we only want to keep events with no value, then skip this event
                # because it has a non-None value
                continue
            # If we haven't seen this code before, then add it to our list of included codes
            if event.code not in self.included_codes:
                # NOTE: Ordering of below two lines is important if want column indexes to start at 0
                self.code_to_column_index[event.code] = len(
                    self.code_to_column_index
                )
                self.included_codes.add(event.code)

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
            raise ValueError(
                "You must pass in at least one featurizer to `aggregate_preprocessed_featurizers`"
            )

        # Aggregating count featurizers
        all_codes: List[int] = [
            c for f in featurizers for c in f.included_codes
        ]

        template_featurizer: CountFeaturizer = featurizers[0]
        new_featurizer: CountFeaturizer = CountFeaturizer(
            is_ontology_expansion=template_featurizer.is_ontology_expansion,
            excluded_codes=template_featurizer.excluded_codes,
            included_codes=all_codes,
            time_bins=template_featurizer.time_bins,
            is_keep_only_none_valued_events=template_featurizer.is_keep_only_none_valued_events,
        )
        return new_featurizer

    def get_num_columns(self) -> int:
        if self.time_bins is None:
            return len(self.included_codes)
        else:
            return len(self.time_bins) * len(self.included_codes)

    def featurize(
        self,
        patient: Patient,
        labels: List[Label],
        ontology: Optional[extension_datasets.Ontology],
    ) -> List[List[ColumnValue]]:

        if ontology is None:
            raise ValueError("`ontology` can't be `None` for CountFeaturizer")

        all_columns: List[List[ColumnValue]] = []

        if self.time_bins is None:
            # Count the number of times each code appears in the patient's timeline
            # [key] = column idx
            # [value] = count of occurrences of events with that code (up to the label at `label_idx`)
            code_counter: Dict[int, int] = defaultdict(int)

            label_idx: int = 0
            for event in patient.events:
                while event.start > labels[label_idx].time:
                    label_idx += 1
                    # Create all features for label at index `label_idx`
                    all_columns.append(
                        [
                            ColumnValue(self.code_to_column_index[code], count)
                            for code, count in code_counter.items()
                        ]
                    )
                    if label_idx >= len(labels):
                        # We've reached the end of the labels for this patient,
                        # so no point in continuing to count events past this point.
                        # Instead, we just return the counts of all events up to this point.
                        return all_columns

                if (
                    self.is_keep_only_none_valued_events
                    and event.value is not None
                ):
                    # If we only want to keep events with no value, then skip this event
                    # because it has a non-None value
                    continue

                for code in self.get_codes(event.code, ontology):
                    # Increment the count for this event's code (plus any parent codes
                    # if we are doing ontology expansion, as handled in `self.get_codes`)
                    if code in self.included_codes:
                        code_counter[code] += 1

            if label_idx < len(labels):
                # For all labels that occur past the last event, add all
                # events' total counts as these labels' feature values (basically,
                # the featurization of these labels is the count of every single event)
                for label in labels[label_idx:]:
                    all_columns.append(
                        [
                            ColumnValue(self.code_to_column_index[code], count)
                            for code, count in code_counter.items()
                        ]
                    )
        else:
            # First, sort time bins in ascending order (i.e. [100 days, 90 days, 1 days] -> [1, 90, 100])
            time_bins: List[datetime.timedelta] = sorted(
                [x for x in self.time_bins if x is not None]
            )

            codes_per_bin: Dict[int, Deque[Tuple[int, datetime.datetime]]] = {
                i: deque() for i in range(len(self.time_bins) + 1)
            }

            code_counts_per_bin: Dict[int, Dict[int, int]] = {
                i: defaultdict(int) for i in range(len(self.time_bins) + 1)
            }

            label_idx: int = 0
            for event in patient.events:
                code: int = event.code
                while event.start > labels[label_idx].time:
                    label_idx += 1
                    # Create all features for label at index `label_idx`
                    all_columns.append(
                        [
                            ColumnValue(
                                self.code_to_column_index[code]
                                + i * len(self.included_codes),
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

                for code in self.get_codes(code, ontology):
                    # Increment the count for this event's code (plus any parent codes
                    # if we are doing ontology expansion, as handled in `self.get_codes`)
                    if code in self.included_codes:
                        codes_per_bin[0].append((code, event.start))
                        code_counts_per_bin[0][code] += 1

                # From closest bin to prediction time -> farthest bin
                for bin_idx, bin_end in enumerate(time_bins):
                    # if i + 1 == len(self.time_bins):
                    #     continue

                    while len(codes_per_bin[bin_idx]) > 0:
                        # Get the least recently added event (i.e. farthest back in patient's timeline
                        # from the currently processed label)
                        oldest_event_code, oldest_event_start = codes_per_bin[
                            bin_idx
                        ][0]

                        if (
                            labels[label_idx].time - oldest_event_start
                        ) <= bin_end:
                            # The oldest event that we're tracking is still within the closest (i.e. smallest distance)
                            # bin to our label's prediction time, so all events will be within this bin, so we don't have to
                            # worry about shifting events into farther bins (as the code in the `else` clause does)
                            break
                        else:
                            # Goal: Readjust codes so that they fall under the proper time bin
                            # Move (oldest_event_code, oldest_event_start) from entry @ `bin_idx` to entry @ `bin_idx + 1`
                            # Basically, move this code from the bin that is closer to the prediction time (`bin_idx`)
                            # to a bin that is further away from the prediction time (`bin_idx + 1`)
                            codes_per_bin[bin_idx + 1].append(
                                codes_per_bin[bin_idx].popleft()
                            )

                            # Remove oldest_event_code from current (closer to prediction time) bin `bin_idx`
                            code_counts_per_bin[bin_idx][oldest_event_code] -= 1
                            # Add oldest_event_code to the (farther from prediction time) bin `bin_idx + 11
                            code_counts_per_bin[bin_idx + 1][
                                oldest_event_code
                            ] += 1

                            # Clear out ColumnValues with a value of 0 to preserve sparsity of matrix
                            if (
                                code_counts_per_bin[bin_idx][oldest_event_code]
                                == 0
                            ):
                                del code_counts_per_bin[bin_idx][
                                    oldest_event_code
                                ]

                if label_idx == len(labels) - 1:
                    all_columns.append(
                        [
                            ColumnValue(
                                self.code_to_column_index[code]
                                + i * len(self.included_codes),
                                count,
                            )
                            for i in range(len(self.time_bins))
                            for code, count in code_counts_per_bin[i].items()
                        ]
                    )
                    break

        return all_columns

    def is_needs_preprocessing(self) -> bool:
        return True
