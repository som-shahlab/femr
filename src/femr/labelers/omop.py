"""meds.Labeling functions for OMOP data."""

from __future__ import annotations

from abc import abstractmethod
import datetime
from typing import Any, Callable, List, Optional, Set, Tuple, Union

import meds

import femr.ontology

from .core import Labeler, TimeHorizon, TimeHorizonEventLabeler, get_death_concepts, identity

def does_exist_event_within_time_range(
    patient: meds.Patient, start: datetime.datetime, end: datetime.datetime, exclude_event_idxs: List[int] = []
) -> bool:
    """Return True if there is at least one event within the given time range for this patient.
    If `exclude_event_idxs` is provided, exclude events with those indexes in `patient.events` from the search."""
    excluded = set(exclude_event_idxs)
    for idx, e in enumerate(patient.events):
        if idx in excluded:
            continue
        if start <= e.start <= end:
            return True
    return False

##########################################################
##########################################################
# Abstract classes derived from TimeHorizonEventLabeler
##########################################################
##########################################################


class WithinVisitLabeler(Labeler):
    """
    The `WithinVisitLabeler` predicts whether or not a patient experiences a specific event
    (as returned by `self.get_outcome_times()`) within each visit.

    Very similar to `TimeHorizonLabeler`, except here we use visits themselves as our time horizon.

    Prediction Time: Start of each visit (adjusted by `self.prediction_adjustment_timedelta` if provided)
    Time horizon: By end of visit
    """

    def __init__(
        self,
        ontology: femr.ontology.Ontology,
        visit_start_adjust_func: Optional[Callable] = None,
        visit_end_adjust_func: Optional[Callable] = None,
    ):
        """The argument `visit_start_adjust_func` is a function that takes in a `datetime.datetime`
        and returns a different `datetime.datetime`."""
        self.ontology: femr.ontology.Ontology = ontology
        self.visit_start_adjust_func: Callable = (
            visit_start_adjust_func if visit_start_adjust_func is not None else identity  # type: ignore
        )
        self.visit_end_adjust_func: Callable = (
            visit_end_adjust_func if visit_end_adjust_func is not None else identity  # type: ignore
        )

    @abstractmethod
    def get_outcome_times(self, patient: meds.Patient) -> List[datetime.datetime]:
        """Return a list of all times when the patient experiences an outcome"""
        return []

    @abstractmethod
    def get_visit_measurements(self, patient: meds.Patient) -> List[Tuple[datetime.datetime, meds.Measurement]]:
        """Return a list of all visits we want to consider (useful for limiting to inpatient visits)."""
        return []

    def label(self, patient: meds.Patient) -> List[meds.Label]:
        """
        Label all visits returned by `self.get_visit_measurements()` with whether the patient
        experiences an outcome in `self.outcome_codes` during each visit.
        """
        visits: List[Tuple[datetime.datetime, meds.Measurement]] = self.get_visit_measurements(patient)
        prediction_start_times: List[datetime.datetime] = [
            self.visit_start_adjust_func(start) for start, visit in visits
        ]
        prediction_end_times: List[datetime.datetime] = [self.visit_end_adjust_func(visit['metadata']['end']) for start, visit in visits]
        outcome_times: List[datetime.datetime] = self.get_outcome_times(patient)

        # For each visit, check if there is an outcome which occurs within the (start, end) of the visit
        results: List[meds.Label] = []
        curr_outcome_idx: int = 0
        for prediction_idx, (prediction_start, prediction_end) in enumerate(
            zip(prediction_start_times, prediction_end_times)
        ):
            # Error checking
            if curr_outcome_idx < len(outcome_times) and outcome_times[curr_outcome_idx] is None:
                raise RuntimeError(
                    "Outcome times must be of type `datetime.datetime`, but value of `None`"
                    " provided for `self.get_outcome_times(patient)[{curr_outcome_idx}]"
                )
            if prediction_start is None:
                raise RuntimeError(
                    "Prediction start times must be of type `datetime.datetime`, but value of `None`"
                    " provided for `prediction_start_time`"
                )
            if prediction_end is None:
                raise RuntimeError(
                    "Prediction end times must be of type `datetime.datetime`, but value of `None`"
                    " provided for `prediction_end_time`"
                )
            if prediction_start > prediction_end:
                raise RuntimeError(
                    "Prediction start time must be before prediction end time, but `prediction_start_time`"
                    f" is `{prediction_start}` and `prediction_end_time` is `{prediction_end}`."
                    " Maybe you `visit_start_adjust_func()` or `visit_end_adjust_func()` in such a way that"
                    " the `start` time got pushed after the `end` time?"
                    " For reference, the original state time of this visit is"
                    f" `{visits[prediction_idx].start}` and the original end time is `{visits[prediction_idx].end}`."
                    f" This is for patient with patient_id `{patient['patient_id']}`."
                )
            # Find the first outcome that occurs after this visit starts
            # (this works b/c we assume visits are sorted by `start`)
            while curr_outcome_idx < len(outcome_times) and outcome_times[curr_outcome_idx] < prediction_start:
                # `curr_outcome_idx` is the idx in `outcome_times` that corresponds to the first
                # outcome EQUAL or AFTER the visit for this prediction time starts (if one exists)
                curr_outcome_idx += 1

            # TRUE if an event occurs within the visit
            is_outcome_occurs_in_time_horizon: bool = (
                (
                    # ensure there is an outcome
                    # (needed in case there are 0 outcomes)
                    curr_outcome_idx
                    < len(outcome_times)
                )
                and (
                    # outcome occurs after visit starts
                    prediction_start
                    <= outcome_times[curr_outcome_idx]
                )
                and (
                    # outcome occurs before visit ends
                    outcome_times[curr_outcome_idx]
                    <= prediction_end
                )
            )
            # Assume no censoring for visits
            is_censored: bool = False

            if is_outcome_occurs_in_time_horizon:
                results.append(meds.Label(patient_id=patient['patient_id'], prediction_time=prediction_start, boolean_value=True))
            elif not is_censored:
                # Not censored + no outcome => FALSE
                results.append(meds.Label(patient_id=patient['patient_id'], prediction_time=prediction_start, boolean_value=False))

        return results

class CodeLabeler(TimeHorizonEventLabeler):
    """Apply a label based on 1+ outcome_codes' occurrence(s) over a fixed time horizon."""

    def __init__(
        self,
        outcome_codes: List[str],
        time_horizon: TimeHorizon,
        prediction_codes: Optional[List[str]] = None,
        prediction_time_adjustment_func: Callable = identity,
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
        self.prediction_time_adjustment_func: Callable = prediction_time_adjustment_func

    def get_prediction_times(self, patient: meds.Patient) -> List[datetime.datetime]:
        """Return each event's start time (possibly modified by prediction_time_adjustment_func)
        as the time to make a prediction. Default to all events whose `code` is in `self.prediction_codes`."""
        times: List[datetime.datetime] = []
        last_time = None
        for e in patient["events"]:
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(e["time"])

            for m in e["measurements"]:
                if ((self.prediction_codes is None) or (m["code"] in self.prediction_codes)) and (
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
        for event in patient["events"]:
            for measurement in event["measurements"]:
                if measurement["code"] in self.outcome_codes:
                    times.append(event["time"])
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
        prediction_time_adjustment_func: Callable = identity,
    ):
        outcome_codes: List[str] = []
        for code in self.original_omop_concept_codes:
            outcome_codes.extend(ontology.get_all_children(code))
        super().__init__(
            outcome_codes=outcome_codes,
            time_horizon=time_horizon,
            prediction_codes=prediction_codes,
            prediction_time_adjustment_func=prediction_time_adjustment_func,
        )


##########################################################
##########################################################
# meds.Labeling functions derived from CodeLabeler
##########################################################
##########################################################


class MortalityCodeLabeler(CodeLabeler):
    """Apply a label for whether or not a patient dies within the `time_horizon`.
    Make prediction at admission time.
    """

    def __init__(
        self,
        ontology: femr.ontology.Ontology,
        time_horizon: TimeHorizon,
        prediction_codes: Optional[List[str]] = None,
        prediction_time_adjustment_func: Callable = identity,
    ):
        """Create a Mortality labeler."""
        outcome_codes: List[str] = []
        for code in get_death_concepts():
            outcome_codes.extend(ontology.get_all_children(code))

        super().__init__(
            outcome_codes=outcome_codes,
            time_horizon=time_horizon,
            prediction_codes=prediction_codes,
            prediction_time_adjustment_func=prediction_time_adjustment_func,
        )


class LupusCodeLabeler(OMOPConceptCodeLabeler):
    """
    meds.Label if patient is diagnosed with Lupus.
    """

    original_omop_concept_codes = ["SNOMED/55464009", "SNOMED/201436003"]