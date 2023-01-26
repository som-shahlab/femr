"""Labeling functions for OMOP data based on lab values."""
from __future__ import annotations

import datetime
from typing import Callable, Dict, List, Optional

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
    get_inpatient_admission_discharge_times,
    group_inpatient_events_by_visit_id,
    map_omop_concept_codes_to_piton_codes,
    move_datetime_to_end_of_day,
)


class WithinInpatientVisitLabeler(WithinVisitLabeler):
    """
    The `WithinInpatientVisitLabeler` predicts whether or not a patient experiences
    a specific event (i.e. has a `code` within `self.outcome_codes`) within each INPATIENT visit.

    The only difference from `WithinVisitLabeler` is that these visits are
    restricted to only INPATIENT visits.

    Prediction Time: Start of each INPATIENT visit (adjusted by `self.prediction_time_adjustment_func()` if provided)

    IMPORTANT: This labeler assumes that every event has a `event.visit_id` property.
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        outcome_codes: List[int],
        prediction_time_adjustment_func: Optional[Callable] = None,
    ):
        super().__init__(
            ontology=ontology,
            outcome_codes=outcome_codes,
            prediction_time_adjustment_func=prediction_time_adjustment_func,
        )

    def label(self, patient: Patient) -> List[Label]:
        """Label all visits with whether the patient experiences outcomes
        in `self.outcome_codes` during each INPATIENT visit."""
        events_by_visit_id: Dict[
            int, List[Event]
        ] = group_inpatient_events_by_visit_id(patient, self.ontology)
        return self.label_each_visit(events_by_visit_id)


class DummyAdmissionDischargeLabeler(Labeler):
    """Generate a placeholder Label at every admission and discharge time for this patient."""

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        self.ontology = ontology

    def label(self, patient: Patient) -> List[Label]:
        labels: List[Label] = []
        for (
            admission_time,
            discharge_time,
        ) in get_inpatient_admission_discharge_times(patient, self.ontology):
            labels.append(Label(time=admission_time, value=True))
            labels.append(Label(time=discharge_time, value=True))
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class InpatientReadmissionLabeler(TimeHorizonEventLabeler):
    """
    This labeler is designed to predict whether a patient will be readmitted within `time_horizon`
    It explicitly does not try to deal with categorizing admissions as "unexpected" or not and is thus
    not comparable to other work.

    Prediction time: At 11:59:59pm on day of discharge from an inpatient admission.
    Time horizon: Interval of time after discharg of length `time_horizon`
    Label: TRUE if patient has an inpatient admission within `time_horizon`

    Defaults to 30-day readmission labeler,
        i.e. `time_horizon = TimeHorizon(1 second, 30 days)`
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        time_horizon: TimeHorizon = TimeHorizon(
            datetime.timedelta(seconds=1), datetime.timedelta(days=30)
        ),  # type: ignore
    ):
        self.ontology: extension_datasets.Ontology = ontology
        self.time_horizon: TimeHorizon = time_horizon

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return the start times of inpatient admissions."""
        times: List[datetime.datetime] = []
        for (
            admission_time,
            discharge_time,
        ) in get_inpatient_admission_discharge_times(patient, self.ontology):
            times.append(admission_time)
        return times

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return end of admission as prediction time, scaled to 11:59:59pm."""
        times: List[datetime.datetime] = []
        for (
            admission_time,
            discharge_time,
        ) in get_inpatient_admission_discharge_times(patient, self.ontology):
            times.append(move_datetime_to_end_of_day(discharge_time))
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
    ):
        self.ontology = ontology
        self.long_time: datetime.timedelta = long_time

    def label(self, patient: Patient) -> List[Label]:
        """Label all admissions with admission length > `self.long_time`"""
        labels: List[Label] = []
        for (
            admission_time,
            discharge_time,
        ) in get_inpatient_admission_discharge_times(patient, self.ontology):
            is_long_admission: bool = (
                discharge_time - admission_time
            ) >= self.long_time
            prediction_time: datetime.datetime = move_datetime_to_end_of_day(
                admission_time
            )
            labels.append(Label(prediction_time, is_long_admission))
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class InpatientMortalityLabeler(WithinInpatientVisitLabeler):
    """
    The inpatient labeler predicts whether or not a patient will die within the current INPATIENT admission.

    Prediction time: At 11:59:59pm on the day of the INPATIENT admission.
    Time horizon: (1 second, end of admission) [note this is a variable time horizon]
    Label: TRUE if patient dies within visit
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        piton_codes = map_omop_concept_codes_to_piton_codes(
            ontology, get_death_concepts()
        )
        super().__init__(
            ontology=ontology,
            outcome_codes=list(piton_codes),
            prediction_time_adjustment_func=move_datetime_to_end_of_day,
        )
