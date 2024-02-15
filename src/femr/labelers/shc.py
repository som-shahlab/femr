"""Labeling functions for SHC data."""
from __future__ import annotations

import datetime
from typing import Any, Callable, List, Optional

import meds

import femr.ontology

from .core import TimeHorizon, TimeHorizonEventLabeler

from femr.labelers.omop import CodeLabeler, identity


##########################################################
##########################################################
# Abstract classes derived from TimeHorizonEventLabeler
##########################################################
##########################################################


class MeasurementLabeler(TimeHorizonEventLabeler):
    """
    Apply a binary label which is 1 if any measurements occur within a given time horizon and 0 otherwise.


    """

    def __init__(
        self,
        outcome_measurements: List[meds.Measurement],
        time_horizon: TimeHorizon,
        prediction_measuremnets: Optional[List[meds.Measurement]] = None,
        prediction_time_adjustment_func: Callable = identity,
    ):
        """Create a CodeLabeler, which labels events whose index in your Ontology is in `self.outcome_codes`

        Args:
            prediction_measurements (List[meds.Measurement]): meds.Measurement instances that count as an occurrence of the outcome.
            time_horizon (TimeHorizon): An interval of time. If the event occurs during this time horizon, then
                the label is TRUE. Otherwise, FALSE.
            prediction_measurements (Optional[List[meds.Measurement]]): If not None, limit measurements at which you make predictions to
                only measurements in this list.
            prediction_time_adjustment_func (Optional[Callable]). A function that takes in a `datetime.datetime`
                and returns a different `datetime.datetime`. Defaults to the identity function.
        """
        self.outcome_measurements: List[str] = outcome_measurements
        self.time_horizon: TimeHorizon = time_horizon
        self.prediction_measuremnets: Optional[List[str]] = prediction_measuremnets
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


class SHCConceptCodeLabeler(CodeLabeler):
    """Identical to OMOPConceptCodeLabeler for now."""

    # codes to include
    original_concept_codes: List[str] = []

    def __init__(
        self,
        ontology: femr.ontology.Ontology,
        time_horizon: TimeHorizon,
        prediction_codes: Optional[List[str]] = None,
        prediction_time_adjustment_func: Callable = identity,
    ):
        outcome_codes: List[str] = []
        for code in self.original_concept_codes:
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


class ICUEventLabeler(CodeLabeler):
    """
    Apply a label for whether or not a patient has an ICU encounter within the timeline

    Make a prediction at every PCP outpatient appointment
    """

    def __init__(
        self,
        time_horizon: TimeHorizon,
        prediction_codes: Optional[List[str]] = None,
        prediction_time_adjustment_func: Callable = identity,
    ):
        # these are the codes we're trying to predict
        outcome_codes: List[str] = [
            "Anesthesia Event:Intensive Care:Inpatient",
            "Anesthesia:Intensive Care:Inpatient",
            "Anesthesia:Intensive Care:Emergency",
            "Anesthesia Event:Intensive Care:Emergency",
            "Hospital Encounter:Intensive Care:Inpatient",
        ]

        # we want to make predictions when these codes occur
        prediction_codes: List[str] = [
            "encounter:Appointment:Primary Care:Outpatient",
        ]

        super().__init__(
            outcome_codes=outcome_codes,
            time_horizon=time_horizon,
            prediction_codes=prediction_codes,
            prediction_time_adjustment_func=prediction_time_adjustment_func,
        )


class HospEncLabeler(CodeLabeler):
    """
    Apply a label for whether or not a patient is admitted within the `time_horizon`.

    Make a prediction at every hospital discharge event
    """

    def __init__(
        self,
        time_horizon: TimeHorizon,
        prediction_codes: Optional[List[str]] = None,
        prediction_time_adjustment_func: Callable = identity,
    ):
        outcome_codes: List[str] = [
            "Encounter/Hospital Encounter",
            "Encounter/Historical Ambulatory Encounter",
        ]

        super().__init__(
            outcome_codes=outcome_codes,
            time_horizon=time_horizon,
            prediction_codes=prediction_codes,
            prediction_time_adjustment_func=prediction_time_adjustment_func,
        )
