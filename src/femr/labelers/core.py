"""Core labeling functionality/schemas, shared across all labeling functions."""

from __future__ import annotations

import datetime
import functools
import hashlib
import struct
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import datasets
import meds

import femr.hf_utils
import femr.ontology

##########################################################
##########################################################
#
# Helper functions
#
##########################################################
##########################################################


def identity(x: Any) -> Any:
    return x


def get_death_concepts() -> List[str]:
    return [
        meds.death_code,
    ]


def move_datetime_to_end_of_day(date: datetime.datetime) -> datetime.datetime:
    return date.replace(hour=23, minute=59, second=0)


##########################################################
##########################################################
#
# Shared classes
#
##########################################################
##########################################################


@dataclass(frozen=True)
class TimeHorizon:
    """An interval of time. Mandatory `start`, optional `end`."""

    start: datetime.timedelta
    end: datetime.timedelta | None  # If NONE, then infinite time horizon


def _label_map_func(batch, *, labeler: Labeler) -> List[meds.Label]:
    result = []
    for patient_id, events in zip(batch["patient_id"], batch["events"]):
        result.extend(labeler.label({"patient_id": patient_id, "events": events}))
    return result


def _label_agg_func(first_labels: List[meds.Label], second_labels: List[meds.Label]):
    first_labels.extend(second_labels)

    return first_labels


class Labeler(ABC):
    """An interface for labeling functions.

    A labeling function applies a label to a specific datetime in a given patient's timeline.
    It can be thought of as generating the following list given a specific patient:
        [(patient ID, datetime_1, label_1), (patient ID, datetime_2, label_2), ... ]
    Usage:
    ```
        labeling_function: Labeler = Labeler(...)
        patients: Sequence[Patient] = ...
        labels: LabeledPatient = labeling_function.apply(patients)
    ```
    """

    @abstractmethod
    def label(self, patient: meds.Patient) -> List[meds.Label]:
        """Apply every label that is applicable to the provided patient.

        This is only called once per patient.

        Args:
            patient (Patient): A patient object

        Returns:
            List[Label]: A list of :class:`Label` containing every label for the given patient
        """
        pass

    def apply(
        self,
        dataset: datasets.Dataset,
        num_proc: int = 1,
        batch_size: int = 10_000,
    ) -> List[meds.Label]:
        """Apply the `label()` function one-by-one to each Patient in a sequence of Patients.

        Args:
            dataset (datasets.Dataset): A HuggingFace Dataset with meds.Patient objects to be labeled.
            num_proc (int, optional): Number of CPU threads to parallelize across. Defaults to 1.

        Returns:
            A list of labels
        """

        return femr.hf_utils.aggregate_over_dataset(
            dataset,
            functools.partial(_label_map_func, labeler=self),
            _label_agg_func,
            batch_size=batch_size,
            num_proc=num_proc,
        )

    def get_patient_start_end_times(patient):
        """Return the datetimes that we consider the (start, end) of this patient."""
        return (patient["events"][0]["time"], patient["events"][-1]["time"])


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

    No labels are generated if the patient record is "censored" before the end of the horizon.

    You are required to implement three methods:
        get_outcome_times() for defining the datetimes of the event of interset
        get_prediction_times() for defining the datetimes at which we make our predictions
        get_time_horizon() for defining the length of time (i.e. `TimeHorizon`) to use for the time horizon
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_outcome_times(self, patient: meds.Patient) -> List[datetime.datetime]:
        """Return a sorted list containing the datetimes that the event of interest "occurs".

        IMPORTANT: Must be sorted ascending (i.e. start -> end of timeline)

        Args:
            patient (Patient): A patient object

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

            Then given a patient timeline:
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
    def get_prediction_times(self, patient: meds.Patient) -> List[datetime.datetime]:
        """Return a sorted list containing the datetimes at which we'll make a prediction.

        IMPORTANT: Must be sorted ascending (i.e. start -> end of timeline)
        """
        pass

    def get_patient_start_end_times(self, patient: meds.Patient) -> Tuple[datetime.datetime, datetime.datetime]:
        """Return the datetimes that we consider the (start, end) of this patient."""
        return (patient["events"][0]["time"], patient["events"][-1]["time"])

    def allow_same_time_labels(self) -> bool:
        """Whether or not to allow labels with events at the same time as prediction"""
        return True

    def label(self, patient: meds.Patient) -> List[meds.Label]:
        """Return a list of Labels for an individual patient.

        Assumes that events in `patient['events']` are already sorted in chronologically
        ascending order (i.e. start -> end).

        Args:
            patient (Patient): A patient object

        Returns:
            List[Label]: A list containing a label for each datetime returned by `get_prediction_times()`
        """
        if len(patient["events"]) == 0:
            return []

        __, end_time = self.get_patient_start_end_times(patient)
        outcome_times: List[datetime.datetime] = self.get_outcome_times(patient)
        prediction_times: List[datetime.datetime] = self.get_prediction_times(patient)
        time_horizon: TimeHorizon = self.get_time_horizon()

        # Get (start, end) of time horizon. If end is None, then it's infinite (set timedelta to max)
        time_horizon_start: datetime.timedelta = time_horizon.start
        time_horizon_end: Optional[datetime.timedelta] = time_horizon.end  # `None` if infinite time horizon

        # For each prediction time, check if there is an outcome which occurs within the (start, end)
        # of the time horizon
        results: List[meds.Label] = []
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
            # TRUE if patient is censored (i.e. timeline ends BEFORE this time horizon ends,
            # so we don't know if the outcome happened after the patient timeline ends)
            # If infinite time horizon labeler, then assume no censoring
            is_censored: bool = end_time < time + time_horizon_end if (time_horizon_end is not None) else False

            if is_outcome_occurs_in_time_horizon:
                results.append(meds.Label(patient_id=patient["patient_id"], prediction_time=time, boolean_value=True))
            elif not is_censored:
                # Not censored + no outcome => FALSE
                results.append(meds.Label(patient_id=patient["patient_id"], prediction_time=time, boolean_value=False))
            elif is_censored:
                # Censored => None
                pass

        return results


class NLabelsPerPatientLabeler(Labeler):
    """Restricts `self.labeler` to returning a max of `self.k` labels per patient."""

    def __init__(self, labeler: Labeler, num_labels: int = 1, seed: int = 1):
        self.labeler: Labeler = labeler
        self.num_labels: int = num_labels  # number of labels per patient
        self.seed: int = seed

    def label(self, patient: meds.Patient) -> List[meds.Label]:
        labels: List[meds.Label] = self.labeler.label(patient)
        if len(labels) <= self.num_labels:
            return labels
        elif self.num_labels == -1:
            return labels
        hash_to_label_list: List[Tuple[int, int, meds.Label]] = [
            (i, compute_random_num(self.seed, patient["patient_id"], i), labels[i]) for i in range(len(labels))
        ]
        hash_to_label_list.sort(key=lambda a: a[1])
        n_hash_to_label_list: List[Tuple[int, int, meds.Label]] = hash_to_label_list[: self.num_labels]
        n_hash_to_label_list.sort(key=lambda a: a[0])
        n_labels: List[meds.Label] = [hash_to_label[2] for hash_to_label in n_hash_to_label_list]
        return n_labels


class CodeLabeler(TimeHorizonEventLabeler):
    """Apply a label based on 1+ outcome_codes' occurrence(s) over a fixed time horizon."""

    def __init__(
        self,
        outcome_codes: List[str],
        time_horizon: TimeHorizon,
        prediction_codes: Optional[List[str]] = None,
        prediction_time_adjustment_func: Optional[Callable] = None,
    ):
        """Create a CodeLabeler, which labels events whose index in your Ontology is in `self.outcome_codes`

        Args:
            prediction_codes (List[int]): Events that count as an occurrence of the outcome.
            time_horizon (TimeHorizon): An interval of time. If the event occurs during this time horizon, then
                the label is TRUE. Otherwise, FALSE.
            prediction_codes (Optional[List[int]]): If not None, limit events at which you make predictions to
                only events with an `event.code` in these codes.
            prediction_time_adjustment_func (Optional[Callable]). A function that takes in a `datetime.datetime`
                and returns a different `datetime.datetime`. Defaults to the identity function.
        """
        self.outcome_codes: List[str] = outcome_codes
        self.time_horizon: TimeHorizon = time_horizon
        self.prediction_codes: Optional[List[str]] = prediction_codes
        self.prediction_time_adjustment_func: Callable = (
            prediction_time_adjustment_func if prediction_time_adjustment_func is not None else identity  # type: ignore
        )

    def get_prediction_times(self, patient: meds.Patient) -> List[datetime.datetime]:
        """Return each event's start time (possibly modified by prediction_time_adjustment_func)
        as the time to make a prediction. Default to all events whose `code` is in `self.prediction_codes`."""
        times: List[datetime.datetime] = []
        last_time = None
        for e in patient["events"]:
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(e.start)
            if ((self.prediction_codes is None) or (e.code in self.prediction_codes)) and (
                last_time != prediction_time
            ):
                times.append(prediction_time)
                last_time = prediction_time
        return times

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon

    def get_outcome_times(self, patient: meds.Patient) -> List[datetime.datetime]:
        """Return the start times of this patient's events whose `code` is in `self.outcome_codes`."""
        times: List[datetime.datetime] = []
        for event in patient.events:
            if event.code in self.outcome_codes:
                times.append(event.start)
        return times

    def allow_same_time_labels(self) -> bool:
        # We cannot allow labels at the same time as the codes since they will generally be available as features ...
        return False


class OMOPConceptCodeLabeler(CodeLabeler):
    """Same as CodeLabeler, but add the extra step of mapping OMOP concept IDs
    (stored in `omop_concept_ids`) to femr codes (stored in `codes`)."""

    # parent OMOP concept codes, from which all the outcome
    # are derived (as children from our ontology)
    original_omop_concept_codes: List[str] = []

    def __init__(
        self,
        ontology: femr.ontology.Ontology,
        time_horizon: TimeHorizon,
        prediction_codes: Optional[List[str]] = None,
        prediction_time_adjustment_func: Optional[Callable] = None,
    ):
        outcome_codes: List[str] = ontology.get_all_children(self.original_omop_concept_codes)
        super().__init__(
            outcome_codes=outcome_codes,
            time_horizon=time_horizon,
            prediction_codes=prediction_codes,
            prediction_time_adjustment_func=(
                prediction_time_adjustment_func if prediction_time_adjustment_func else identity
            ),
        )


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
