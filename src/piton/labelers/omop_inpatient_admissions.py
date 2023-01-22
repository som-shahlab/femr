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
    get_death_concepts,
    get_inpatient_admission_concepts,
    get_inpatient_admission_events,
)


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
        return [x.start for x in admission_events]

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return end of admission as prediction time."""
        admission_events: List[Event] = get_inpatient_admission_events(
            patient, self.ontology
        )
        return [x.end for x in admission_events]

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon


class InpatientMortalityLabeler(WithinVisitLabeler):
    """
    The inpatient labeler predicts whether or not a patient will die within the current admission.
    The prediction time is at the time of admission.
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        dictionary = ontology.get_dictionary()
        outcome_codes = [dictionary.index(x) for x in get_death_concepts()]

        super().__init__(ontology, outcome_codes)


class LongInpatientAdmissionLabeler(Labeler):
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
                is_long_admission: bool = (
                    event.end - event.start
                ) >= self.long_time
                labels.append(Label(event.start, is_long_admission))
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"
