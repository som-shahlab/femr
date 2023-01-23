"""Labeling functions for OMOP data based on lab values."""
from __future__ import annotations

import datetime
from typing import List

from .. import Event, Patient
from ..extension import datasets as extension_datasets
from .core import (
    Label,
    Labeler,
    LabelType,
    TimeHorizon,
    TimeHorizonEventLabeler,
)
from .omop import (
    WithinVisitLabeler,
    _get_all_children,
    get_death_concepts,
    get_inpatient_admission_concepts,
    get_inpatient_admission_events,
)


class AdmissionDischargePlaceholderLabeler(Labeler):
    """Generate a placeholder Label at every admission and discharge time for this patient."""

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        dictionary = ontology.get_dictionary()
        self.prediction_codes: List[int] = [
            dictionary.index(x) for x in get_inpatient_admission_concepts()
        ]

    def label(self, patient: Patient) -> List[Label]:
        labels: List[Label] = []
        for e in patient.events:
            if e.code in self.prediction_codes:
                # Record label at admission time
                labels.append(Label(time=e.start, value=True))
                # Record label at discharge time
                if e.end is None:
                    raise RuntimeError(
                        f"Event {e} cannot have `None` as its `end` attribute."
                    )
                labels.append(Label(time=e.end, value=True))
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class InpatientReadmissionLabeler(TimeHorizonEventLabeler):
    """
    This labeler is designed to predict whether a patient will be readmitted within `time_horizon`
    It explicitly does not try to deal with categorizing admissions as "unexpected" or not and is thus
    not comparable to other work.
    It predicts at the end of each admission.
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        time_horizon: TimeHorizon,
    ):
        self.ontology: extension_datasets.Ontology = ontology
        self.time_horizon: TimeHorizon = time_horizon

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return the start times of inpatient admissions."""
        admission_events: List[Event] = get_inpatient_admission_events(
            patient, self.ontology
        )
        times: List[datetime.datetime] = []
        for x in admission_events:
            if x.start is None:
                raise ValueError(
                    f"Admission {x} cannot have the value `None` as its start time"
                )
            times.append(x.start)
        return times

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return end of admission as prediction time."""
        admission_events: List[Event] = get_inpatient_admission_events(
            patient, self.ontology
        )
        times: List[datetime.datetime] = []
        for x in admission_events:
            if x.end is None:
                raise ValueError(
                    f"Admission {x} cannot have the value `None` as its end time"
                )
            times.append(x.end)
        return times

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon


class InpatientLongAdmissionLabeler(Labeler):
    """
    The inpatient labeler predicts whether or not a patient will be admitted for a long time (defined
    as `admission.end - admission.start >= self.long_time`).
    The prediction time is at the time of admission.
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        long_time: datetime.timedelta,
    ):
        dictionary = ontology.get_dictionary()
        self.admission_codes: List[int] = [
            dictionary.index(x) for x in get_inpatient_admission_concepts()
        ]
        self.long_time: datetime.timedelta = long_time

    def label(self, patient: Patient) -> List[Label]:
        """Label all admissions with admission length > `self.long_time`"""
        labels: List[Label] = []
        for event in patient.events:
            if event.code in self.admission_codes:
                if event.end is None:
                    raise RuntimeError(
                        f"Admission {event} cannot have the value `None` as its end time"
                    )
                is_long_admission: bool = (
                    event.end - event.start
                ) >= self.long_time
                labels.append(Label(event.start, is_long_admission))
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class InpatientMortalityLabeler(WithinVisitLabeler):
    """
    The inpatient labeler predicts whether or not a patient will die within the current admission.

    Prediction time: At time of inpatient admission.
    Time horizon: (1 second, end of admission [variable])
    Label: TRUE if patient dies within visit
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        dictionary = ontology.get_dictionary()
        piton_codes = set()
        for code in get_death_concepts():
            piton_codes |= _get_all_children(ontology, dictionary.index(code))
        super().__init__(ontology=ontology, outcome_codes=list(piton_codes))


##########################################################
##########################################################
# Specific instances of the above labelers
##########################################################
##########################################################


class _30DayReadmissionLabeler(InpatientReadmissionLabeler):
    """Predict if patient will be readmitted to hospital within 30 days of discharge.

    Prediction time: At discharge from an inpatient admission.
    Time horizon: (1 second, 30 days)
    Label: TRUE if patient has an inpatient admission within time horizon
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        time_horizon = TimeHorizon(datetime.timedelta(seconds=1), datetime.timedelta(days=30))  # type: ignore
        super().__init__(ontology=ontology, time_horizon=time_horizon)


class _1WeekLongLOSLabeler(InpatientLongAdmissionLabeler):
    """Predict if inpatient visit is >= 7 days.

    Prediction time: At time of inpatient admission.
    Time horizon: Till the end of the visit
    Label: TRUE if visit length is >= 7 days (i.e. `visit.end - visit.start >= 7 days`)
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        long_time: datetime.timedelta = datetime.timedelta(days=7)
        super().__init__(ontology=ontology, long_time=long_time)
