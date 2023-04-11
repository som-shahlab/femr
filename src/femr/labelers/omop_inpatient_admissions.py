"""Labeling functions for OMOP data based on lab values."""
from __future__ import annotations

import datetime
from abc import abstractmethod
from typing import Any, Callable, List, Set

from .. import Event, Patient
from ..extension import datasets as extension_datasets
from .core import Label, Labeler, LabelType, TimeHorizon, TimeHorizonEventLabeler
from .omop import (
    WithinVisitLabeler,
    get_death_concepts,
    get_inpatient_admission_discharge_times,
    get_inpatient_admission_events,
    map_omop_concept_codes_to_femr_codes,
    move_datetime_to_end_of_day,
)


def identity(x: Any) -> Any:
    return x


class WithinInpatientVisitLabeler(WithinVisitLabeler):
    """
    The `WithinInpatientVisitLabeler` predicts whether or not a patient experiences
    a specific event within each INPATIENT visit.

    The only difference from `WithinVisitLabeler` is that these visits are restricted to only INPATIENT visits.

    Prediction Time: Start of each INPATIENT visit (adjusted by `self.prediction_time_adjustment_func()` if provided)
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        visit_start_adjust_func: Callable = identity,
        visit_end_adjust_func: Callable = identity,
    ):
        """The argument `visit_start_adjust_func` is a function that takes in a `datetime.datetime`
        and returns a different `datetime.datetime`."""
        super().__init__(
            ontology=ontology,
            visit_start_adjust_func=visit_start_adjust_func,
            visit_end_adjust_func=visit_end_adjust_func,
        )

    @abstractmethod
    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        return []

    def get_visit_events(self, patient: Patient) -> List[Event]:
        return get_inpatient_admission_events(patient, self.ontology)


class DummyAdmissionDischargeLabeler(Labeler):
    """Generate a placeholder Label at every admission and discharge time for this patient."""

    def __init__(self, ontology: extension_datasets.Ontology, prediction_time_adjustment_func: Callable = identity):
        self.ontology: extension_datasets.Ontology = ontology
        self.prediction_time_adjustment_func: Callable = prediction_time_adjustment_func

    def label(self, patient: Patient) -> List[Label]:
        labels: List[Label] = []
        for admission_time, discharge_time in get_inpatient_admission_discharge_times(patient, self.ontology):
            labels.append(Label(time=self.prediction_time_adjustment_func(admission_time), value=True))
            labels.append(Label(time=self.prediction_time_adjustment_func(discharge_time), value=True))
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class InpatientReadmissionLabeler(TimeHorizonEventLabeler):
    """
    This labeler is designed to predict whether a patient will be readmitted within `time_horizon`
    It explicitly does not try to deal with categorizing admissions as "unexpected" or not and is thus
    not comparable to other work.

    Prediction time: At discharge from an inpatient admission. Defaults to shifting prediction time
                     to the end of the day.
    Time horizon: Interval of time after discharg of length `time_horizon`
    Label: TRUE if patient has an inpatient admission within `time_horizon`

    Defaults to 30-day readmission labeler,
        i.e. `time_horizon = TimeHorizon(1 second, 30 days)`
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        time_horizon: TimeHorizon = TimeHorizon(
            start=datetime.timedelta(seconds=1), end=datetime.timedelta(days=30)
        ),  # type: ignore
        prediction_time_adjustment_func: Callable = move_datetime_to_end_of_day,
    ):
        self.ontology: extension_datasets.Ontology = ontology
        self.time_horizon: TimeHorizon = time_horizon
        self.prediction_time_adjustment_func = prediction_time_adjustment_func

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return the start times of inpatient admissions."""
        times: List[datetime.datetime] = []
        for admission_time, __ in get_inpatient_admission_discharge_times(patient, self.ontology):
            times.append(admission_time)
        return times

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return end of admission as prediction timm."""
        times: List[datetime.datetime] = []
        for __, discharge_time in get_inpatient_admission_discharge_times(patient, self.ontology):
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(discharge_time)
            times.append(prediction_time)
        times = sorted(list(set(times)))
        return times

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon


class InpatientLongAdmissionLabeler(Labeler):
    """
    This labeler predicts whether or not a patient will be admitted for a long time (defined
    as `admission.end - admission.start >= self.long_time`) during an INPATIENT visit.

    Prediction time: At time of INPATIENT admission.
    Time horizon: Till the end of the visit
    Label: TRUE if visit length is >= `long_time` (i.e. `visit.end - visit.start >= long_time`)

    Defaults to 7-day long length-of-stay (LOS)
        i.e. `long_time = 7 days`
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        long_time: datetime.timedelta = datetime.timedelta(days=7),
        prediction_time_adjustment_func: Callable = move_datetime_to_end_of_day,
    ):
        self.ontology: extension_datasets.Ontology = ontology
        self.long_time: datetime.timedelta = long_time
        self.prediction_time_adjustment_func = prediction_time_adjustment_func

    def label(self, patient: Patient) -> List[Label]:
        """Label all admissions with admission length > `self.long_time`"""
        labels: List[Label] = []
        for admission_time, discharge_time in get_inpatient_admission_discharge_times(patient, self.ontology):
            is_long_admission: bool = (discharge_time - admission_time) >= self.long_time
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(admission_time)
            labels.append(Label(prediction_time, is_long_admission))
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class InpatientMortalityLabeler(WithinInpatientVisitLabeler):
    """
    The inpatient labeler predicts whether or not a patient will die within the current INPATIENT admission.

    Prediction time: Defaults to 11:59:59pm on the day of the INPATIENT admission.
    Time horizon: (1 second, end of admission) [note this time horizon varies by visit]
    Label: TRUE if patient dies within visit
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        visit_start_adjust_func: Callable = move_datetime_to_end_of_day,
        visit_end_adjust_func: Callable = identity,
    ):
        femr_codes: Set[int] = map_omop_concept_codes_to_femr_codes(ontology, get_death_concepts())
        self.outcome_codes: Set[int] = femr_codes
        super().__init__(
            ontology=ontology,
            visit_start_adjust_func=visit_start_adjust_func,
            visit_end_adjust_func=visit_end_adjust_func,
        )

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return time of any event with a code in `self.outcome_codes`."""
        times: List[datetime.datetime] = []
        for e in patient.events:
            if e.code in self.outcome_codes:
                times.append(e.start)
        return times
