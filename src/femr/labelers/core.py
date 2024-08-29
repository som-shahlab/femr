"""Core labeling functionality/schemas, shared across all labeling functions."""

from __future__ import annotations

import datetime
import functools
import hashlib
import itertools
import struct
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, NamedTuple, Optional, Tuple

import meds_reader
import pandas as pd


# A more efficient copy of the MEDS label type definition
class Label(NamedTuple):
    subject_id: int
    prediction_time: datetime.datetime

    boolean_value: bool


@dataclass(frozen=True)
class TimeHorizon:
    """An interval of time. Mandatory `start`, optional `end`."""

    start: datetime.timedelta
    end: datetime.timedelta | None  # If NONE, then infinite time horizon


def _label_map_func(subjects: Iterator[meds_reader.Subject], *, labeler: Labeler) -> pd.DataFrame:
    data = itertools.chain.from_iterable(labeler.label(subject) for subject in subjects)
    final = pd.DataFrame.from_records(data, columns=Label._fields)
    final["prediction_time"] = final["prediction_time"].astype("datetime64[us]")
    return final


class Labeler(ABC):
    """An interface for labeling functions.

    A labeling function applies a label to a specific datetime in a given subject's timeline.
    It can be thought of as generating the following list given a specific subject:
        [(subject ID, datetime_1, label_1), (subject ID, datetime_2, label_2), ... ]
    Usage:
    ```
        labeling_function: Labeler = Labeler(...)
        subjects: Sequence[Subject] = ...
        labels: LabeledSubject = labeling_function.apply(subjects)
    ```
    """

    @abstractmethod
    def label(self, subject: meds_reader.Subject) -> List[Label]:
        """Apply every label that is applicable to the provided subject.

        This is only called once per subject.

        Args:
            subject (Subject): A subject object

        Returns:
            List[Label]: A list of :class:`Label` containing every label for the given subject
        """
        pass

    def apply(
        self,
        db: meds_reader.SubjectDatabase,
    ) -> pd.DataFrame:
        """Apply the `label()` function one-by-one to each Subject in a sequence of Subjects.

        Args:
            dataset (datasets.Dataset): A HuggingFace Dataset with meds_reader.Subject objects to be labeled.
            num_proc (int, optional): Number of CPU threads to parallelize across. Defaults to 1.

        Returns:
            A list of labels
        """

        # TODO: Cast the schema properly
        result = pd.concat(db.map(functools.partial(_label_map_func, labeler=self)), ignore_index=True)
        result.sort_values(by=["subject_id", "prediction_time"], inplace=True)

        return result


##########################################################
# Specific Labeler Superclasses
##########################################################


class TimeHorizonEventLabeler(Labeler):
    """Label events that occur within a particular time horizon.
    This support both "finite" and "infinite" time horizons.

    The time horizon can be "fixed" (i.e. has both a start and end date), or "infinite" (i.e. only a start date)

    A TimeHorizonEventLabeler enables you to label events that occur within a particular
    time horizon (i.e. `TimeHorizon`). It is a boolean event that is TRUE if the event of interest
    occurs within that time horizon, and FALSE if it doesn't occur by the end of the time horizon.

    No labels are generated if the subject record is "censored" before the end of the horizon.

    You are required to implement three methods:
        get_outcome_times() for defining the datetimes of the event of interset
        get_prediction_times() for defining the datetimes at which we make our predictions
        get_time_horizon() for defining the length of time (i.e. `TimeHorizon`) to use for the time horizon
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_outcome_times(self, subject: meds_reader.Subject) -> List[datetime.datetime]:
        """Return a sorted list containing the datetimes that the event of interest "occurs".

        IMPORTANT: Must be sorted ascending (i.e. start -> end of timeline)

        Args:
            subject (Subject): A subject object

        Returns:
            List[datetime.datetime]: A list of datetimes, one corresponding to an occurrence of the outcome
        """
        pass

    @abstractmethod
    def get_time_horizon(self) -> TimeHorizon:
        """Return time horizon for making predictions with this labeling function.

        Return the (start offset, end offset) of the time horizon (from the prediction time)
        used for labeling whether an outcome occurred or not. These can be arbitrary timedeltas.

        If end offset is None, then the time horizon is infinite (i.e. only has a start offset).
        If end offset is not None, then the time horizon is finite (i.e. has both a start and end offset),
            and it must be true that end offset >= start offset.

        Example:
            X is the time that you're making a prediction (given by `get_prediction_times()`)
            (A,B) is your time horizon (given by `get_time_horizon()`)
            O is an outcome (given by `get_outcome_times()`)

            Then given a subject timeline:
                X-----(X+A)------(X+B)------


            This has a label of TRUE:
                X-----(X+A)--O---(X+B)------

            This has a label of TRUE:
                X-----(X+A)--O---(X+B)----O-

            This has a label of FALSE:
                X---O-(X+A)------(X+B)------

            This has a label of FALSE:
                X-----(X+A)------(X+B)--O---
        """
        pass

    @abstractmethod
    def get_prediction_times(self, subject: meds_reader.Subject) -> List[datetime.datetime]:
        """Return a sorted list containing the datetimes at which we'll make a prediction.

        IMPORTANT: Must be sorted ascending (i.e. start -> end of timeline)
        """
        pass

    def get_subject_start_end_times(self, subject: meds_reader.Subject) -> Tuple[datetime.datetime, datetime.datetime]:
        """Return the datetimes that we consider the (start, end) of this subject."""
        return (subject.events[0].time, subject.events[-1].time)

    def allow_same_time_labels(self) -> bool:
        """Whether or not to allow labels with events at the same time as prediction"""
        return True

    def label(self, subject: meds_reader.Subject) -> List[Label]:
        """Return a list of Labels for an individual subject.

        Assumes that events in `subject['events']` are already sorted in chronologically
        ascending order (i.e. start -> end).

        Args:
            subject (Subject): A subject object

        Returns:
            List[Label]: A list containing a label for each datetime returned by `get_prediction_times()`
        """
        if len(subject.events) == 0:
            return []

        __, end_time = self.get_subject_start_end_times(subject)
        outcome_times: List[datetime.datetime] = self.get_outcome_times(subject)
        prediction_times: List[datetime.datetime] = self.get_prediction_times(subject)
        time_horizon: TimeHorizon = self.get_time_horizon()

        # Get (start, end) of time horizon. If end is None, then it's infinite (set timedelta to max)
        time_horizon_start: datetime.timedelta = time_horizon.start
        time_horizon_end: Optional[datetime.timedelta] = time_horizon.end  # `None` if infinite time horizon

        # For each prediction time, check if there is an outcome which occurs within the (start, end)
        # of the time horizon
        results: List[Label] = []
        curr_outcome_idx: int = 0
        last_time = None

        for time in prediction_times:
            if last_time is not None:
                assert time > last_time, f"Must be ascending prediction times, instead got {last_time} <= {time}"

            last_time = time
            while curr_outcome_idx < len(outcome_times) and outcome_times[curr_outcome_idx] < time + time_horizon_start:
                # `curr_outcome_idx` is the idx in `outcome_times` that corresponds to the first
                # outcome EQUAL or AFTER the time horizon for this prediction time starts (if one exists)
                curr_outcome_idx += 1

            if curr_outcome_idx < len(outcome_times) and outcome_times[curr_outcome_idx] == time:
                if not self.allow_same_time_labels():
                    continue
                warnings.warn(
                    "You are making predictions at the same time as the target outcome."
                    "This frequently leads to label leakage."
                )

            # TRUE if an event occurs within the time horizon
            is_outcome_occurs_in_time_horizon: bool = (
                (
                    # ensure there is an outcome
                    # (needed in case there are 0 outcomes)
                    curr_outcome_idx
                    < len(outcome_times)
                )
                and (
                    # outcome occurs after time horizon starts
                    time + time_horizon_start
                    <= outcome_times[curr_outcome_idx]
                )
                and (
                    # outcome occurs before time horizon ends (if there is an end)
                    (time_horizon_end is None)
                    or outcome_times[curr_outcome_idx] <= time + time_horizon_end
                )
            )
            # TRUE if subject is censored (i.e. timeline ends BEFORE this time horizon ends,
            # so we don't know if the outcome happened after the subject timeline ends)
            # If infinite time horizon labeler, then assume no censoring
            is_censored: bool = end_time < time + time_horizon_end if (time_horizon_end is not None) else False

            if is_outcome_occurs_in_time_horizon:
                results.append(Label(subject_id=subject.subject_id, prediction_time=time, boolean_value=True))
            elif not is_censored:
                # Not censored + no outcome => FALSE
                results.append(Label(subject_id=subject.subject_id, prediction_time=time, boolean_value=False))
            elif is_censored:
                # Censored => None
                pass

        return results


class NLabelsPerSubjectLabeler(Labeler):
    """Restricts `self.labeler` to returning a max of `self.k` labels per subject."""

    def __init__(self, labeler: Labeler, num_labels: int = 1, seed: int = 1):
        self.labeler: Labeler = labeler
        self.num_labels: int = num_labels  # number of labels per subject
        self.seed: int = seed

    def label(self, subject: meds_reader.Subject) -> List[Label]:
        labels: List[Label] = self.labeler.label(subject)
        if len(labels) <= self.num_labels:
            return labels
        elif self.num_labels == -1:
            return labels
        hash_to_label_list: List[Tuple[int, int, Label]] = [
            (i, compute_random_num(self.seed, subject.subject_id, i), labels[i]) for i in range(len(labels))
        ]
        hash_to_label_list.sort(key=lambda a: a[1])
        n_hash_to_label_list: List[Tuple[int, int, Label]] = hash_to_label_list[: self.num_labels]
        n_hash_to_label_list.sort(key=lambda a: a[0])
        n_labels: List[Label] = [hash_to_label[2] for hash_to_label in n_hash_to_label_list]
        return n_labels


def compute_random_num(seed: int, num_1: int, num_2: int, modulus: int = 100):
    network_num_1 = struct.pack("!q", num_1)
    network_num_2 = struct.pack("!q", num_2)
    network_seed = struct.pack("!q", seed)

    to_hash = network_seed + network_num_1 + network_num_2

    hash_object = hashlib.sha256()
    hash_object.update(to_hash)
    hash_value = hash_object.digest()

    result = 0
    for i in range(len(hash_value)):
        result = (result * 256 + hash_value[i]) % modulus

    return result
