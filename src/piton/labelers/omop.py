"""Labeling functions for OMOP data."""
from __future__ import annotations

import datetime
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from .. import Event, Patient
from ..extension import datasets as extension_datasets
from .core import (
    Label,
    Labeler,
    LabelType,
    TimeHorizon,
    TimeHorizonEventLabeler,
)


def get_visit_concepts() -> List[str]:
    return ["Visit/IP"]


def get_inpatient_admission_concepts() -> List[str]:
    return ["Visit/IP"]


def get_death_concepts() -> List[str]:
    return [
        "Death Type/OMOP generated",
        "Condition Type/OMOP4822053",
    ]


def get_inpatient_admission_events(
    patient: Patient, ontology: extension_datasets.Ontology
) -> List[Event]:
    dictionary = ontology.get_dictionary()
    admission_codes: List[int] = [
        dictionary.index(x) for x in get_inpatient_admission_concepts()
    ]
    admissions: List[Event] = []
    for e in patient.events:
        if e.code in admission_codes:
            admissions.append(e)
    return admissions


def get_inpatient_admission_discharge_times(
    ontology: extension_datasets.Ontology, patient: Patient
) -> Tuple[List[datetime.datetime], List[datetime.datetime]]:
    """Return a list of all admission/discharge times for this patient."""
    dictionary = ontology.get_dictionary()
    admission_codes: List[int] = [
        dictionary.index(x) for x in get_inpatient_admission_concepts()
    ]
    admission_times: List[datetime.datetime] = []
    discharge_times: List[datetime.datetime] = []
    for e in patient.events:
        if e.code in admission_codes:
            # Record label at admission time
            admission_times.append(e.start)
            # Record label at discharge time
            if e.end is None:
                raise RuntimeError(
                    f"Event {e} cannot have `None` as its `end` attribute."
                )
            discharge_times.append(e.end)
    return admission_times, discharge_times


def map_omop_concept_ids_to_piton_codes(
    ontology: extension_datasets.Ontology, omop_concept_ids: List[int]
) -> List[int]:
    codes: List[int] = []
    for omop_concept_id in omop_concept_ids:
        piton_code: Optional[
            int
        ] = ontology.get_code_from_concept_id(  # type:ignore
            omop_concept_id
        )
        if piton_code is None:
            print(f"code {omop_concept_id} not found")
        else:
            codes.append(piton_code)
    return list(set(codes))


# TODO - move this into the ontology class
def _get_all_children(
    ontology: extension_datasets.Ontology, code: int
) -> Set[int]:
    children_code_set = set([code])
    parent_deque = deque([code])

    while len(parent_deque) > 0:
        temp_parent_code: int = parent_deque.popleft()
        for temp_child_code in ontology.get_children(temp_parent_code):
            children_code_set.add(temp_child_code)
            parent_deque.append(temp_child_code)

    return children_code_set


##########################################################
##########################################################
# Abstract classes derived from Labeler
##########################################################
##########################################################


class WithinVisitLabeler(Labeler):
    """
    The `WithinVisitLabeler` predicts whether or not a patient experiences a specific event (i.e. has a `code` within
    `self.outcome_codes`) within each visit.

    Prediction time: Start of each visit (adjusted by `self.prediction_adjustment_timedelta` if provided)

    IMPORTANT: This labeler assumes that every event has a `event.visit_id` property.
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        outcome_codes: List[int],
        prediction_adjustment_timedelta: Optional[datetime.timedelta] = None,
    ):
        dictionary = ontology.get_dictionary()
        self.visit_codes: List[int] = [
            dictionary.index(x) for x in get_visit_concepts()
        ]
        self.outcome_codes: List[int] = outcome_codes
        self.prediction_adjustment_timedelta: Optional[
            datetime.timedelta
        ] = prediction_adjustment_timedelta

    def label(self, patient: Patient) -> List[Label]:
        """Label all visits with whether the patient experiences outcomes
        in `self.outcome_codes` during each visit."""
        # Loop through all visits in patient, check if outcome, if so, mark
        # that it occurred in `visit_to_outcome_count`.
        # NOTE: `visit_to_outcome_count` and `visits` are kept in sync with each other
        # Contains all events whose `event.visit_id`` is referenced in `visit_to_outcome_count`
        visit_to_outcome_count: Dict[int, int] = {}
        visits: List[Event] = []
        for event in patient.events:
            if event.visit_id is None:
                # Ignore events without a `visit_id`
                continue
                # raise RuntimeError(
                #     f"Event with code={event.code} at time={event.start} for patient id={patient.patient_id}"
                #     f" does not have a `visit_id`"
                # )
            if (
                event.visit_id not in visit_to_outcome_count
                and event.code in self.visit_codes
            ):
                visit_to_outcome_count[event.visit_id] = 0
                visits.append(event)
            elif event.code in self.outcome_codes:
                if event.visit_id not in visit_to_outcome_count:
                    raise RuntimeError(
                        f"Outcome event with code={event.code} at time={event.start}"
                        f" for patient id={patient.patient_id} occurred before its"
                        f" corresponding admission event with visit_id {event.visit_id}"
                    )
                visit_to_outcome_count[event.visit_id] += 1

        # Generate labels
        labels: List[Label] = []
        for event in visits:
            is_outcome_occurs: bool = visit_to_outcome_count[event.visit_id] > 0
            prediction_time: datetime.datetime = (
                (event.start + self.prediction_adjustment_timedelta)
                if self.prediction_adjustment_timedelta is not None
                else event.start
            )
            labels.append(Label(prediction_time, is_outcome_occurs))
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class WithinInpatientVisitLabeler(WithinVisitLabeler):
    """
    The `WithinInpatientVisitLabeler` predicts whether or not a patient experiences
    a specific event (i.e. has a `code` within `self.outcome_codes`) within each INPATIENT visit.

    The only difference from `WithinVisitLabeler` is that these visits are
    restricted to only INPATIENT visits.

    Prediction time: Start of each INPATIENT visit (adjusted by `self.prediction_adjustment_timedelta` if provided)

    IMPORTANT: This labeler assumes that every event has a `event.visit_id` property.
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        outcome_codes: List[int],
        prediction_adjustment_timedelta: Optional[datetime.timedelta] = None,
    ):
        super().__init__(
            ontology=ontology,
            outcome_codes=outcome_codes,
            prediction_adjustment_timedelta=prediction_adjustment_timedelta,
        )
        dictionary = ontology.get_dictionary()
        self.visit_codes: List[int] = [
            dictionary.index(x) for x in get_inpatient_admission_concepts()
        ]


##########################################################
##########################################################
# Abstract classes derived from TimeHorizonEventLabeler
##########################################################
##########################################################


class CodeLabeler(TimeHorizonEventLabeler):
    """Apply a label based on 1+ outcome_codes' occurrence(s) over a fixed time horizon."""

    def __init__(
        self,
        outcome_codes: List[int],
        time_horizon: TimeHorizon,
        prediction_codes: Optional[List[int]] = None,
    ):
        """Create a CodeLabeler, which labels events whose index in your Ontology is in `self.outcome_codes`

        Args:
            prediction_codes (List[int]): Events that count as an occurrence of the outcome.
            time_horizon (TimeHorizon): An interval of time. If the event occurs during this time horizon, then
                the label is TRUE. Otherwise, FALSE.
            prediction_codes (Optional[List[int]]): If not None, limit events at which you make predictions to
                only events with an `event.code` in these codes.

        Raises:
            ValueError: Raised if there are multiple unique codes that map to the death code
        """
        self.outcome_codes: List[int] = outcome_codes
        self.time_horizon: TimeHorizon = time_horizon
        self.prediction_codes: Optional[List[int]] = prediction_codes

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return each event's start time as the time to make a prediction.
        Default to all events whose `code` is in `self.prediction_codes`."""
        return [
            # datetime.datetime.strptime(
            #     # TODO - Why add 23:59:00?
            #     str(e.start)[:10] + " 23:59:00", "%Y-%m-%d %H:%M:%S"
            # )
            e.start
            for e in patient.events
            if (self.prediction_codes is None)
            or (e.code in self.prediction_codes)
        ]

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return the start times of this patient's events whose `code` is in `self.outcome_codes`."""
        times: List[datetime.datetime] = []
        for event in patient.events:
            if event.code in self.outcome_codes:
                times.append(event.start)
        return times


class OMOPConceptCodeLabeler(CodeLabeler):
    """Same as CodeLabeler, but add the extra step of mapping OMOP concept IDs
    (stored in `omop_concept_ids`) to Piton codes (stored in `codes`)."""

    # parent OMOP concept codes, from which all the outcome
    # are derived (as children from our ontology)
    original_omop_concept_codes: List[str] = []

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        time_horizon: TimeHorizon,
        prediction_codes: Optional[List[int]] = None,
    ):
        outcome_codes: List[int] = []

        # We need to traverse through the ontology ourselves using
        # OMOP Concept Codes (e.g. "LOINC/123") instead of pre-specified
        # OMOP Concept IDs (e.g. 3939430) to get all revelant children
        for omop_concept_code in self.original_omop_concept_codes:
            try:
                piton_code = ontology.get_dictionary().index(omop_concept_code)
            except ValueError:
                raise ValueError(
                    f"OMOP Concept Code {omop_concept_code} not found in ontology."
                )
            all_children: Set[int] = _get_all_children(ontology, piton_code)
            outcome_codes += list(all_children)
        outcome_codes = list(set(outcome_codes))

        super().__init__(
            outcome_codes=outcome_codes,
            time_horizon=time_horizon,
            prediction_codes=prediction_codes,
        )


##########################################################
##########################################################
# Labeling functions derived from CodeLabeler
##########################################################
##########################################################


class MortalityCodeLabeler(CodeLabeler):
    """Apply a label for whether or not a patient dies within the `time_horizon`.
    Make prediction at admission time.
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        time_horizon: TimeHorizon,
        prediction_codes: Optional[List[int]] = None,
    ):
        """Create a Mortality labeler."""
        dictionary = ontology.get_dictionary()
        outcome_codes = [dictionary.index(x) for x in get_death_concepts()]

        super().__init__(
            outcome_codes=outcome_codes,
            time_horizon=time_horizon,
            prediction_codes=prediction_codes,
        )


class LupusCodeLabeler(CodeLabeler):
    """
    Label if patient is diagnosed with Lupus.
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        time_horizon: TimeHorizon,
        prediction_codes: Optional[List[int]] = None,
    ):
        dictionary = ontology.get_dictionary()
        codes = set()
        snomed_codes: List[str] = ["55464009", "201436003"]
        for code in snomed_codes:
            codes |= _get_all_children(
                ontology, dictionary.index("SNOMED/" + code)
            )

        super().__init__(
            outcome_codes=list(codes),
            time_horizon=time_horizon,
            prediction_codes=prediction_codes,
        )


class HighHbA1cCodeLabeler(Labeler):
    """
    The high HbA1c labeler tries to predict whether a non-diabetic patient will test as diabetic.
    Note: This labeler will only trigger at most once every 6 months.
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        last_trigger_timedelta: datetime.timedelta = datetime.timedelta(
            days=180
        ),
    ):
        """Create a High HbA1c (i.e. diabetes) labeler."""
        self.last_trigger_timedelta = last_trigger_timedelta

        HbA1c_str: str = "LOINC/4548-4"
        self.hba1c_lab_code = ontology.get_dictionary().index(HbA1c_str)

        diabetes_str: str = "SNOMED/44054006"
        diabetes_code = ontology.get_dictionary().index(diabetes_str)
        self.diabetes_codes = _get_all_children(ontology, diabetes_code)

    def label(self, patient: Patient) -> List[Label]:
        if len(patient.events) == 0:
            return []

        high_cutoff_threshold: float = 6.5
        labels: List[Label] = []
        last_trigger: Optional[datetime.datetime] = None

        first_diabetes_code_date = None
        for event in patient.events:
            if event.code in self.diabetes_codes:
                first_diabetes_code_date = event.start
                break

        for event in patient.events:

            if (
                first_diabetes_code_date is not None
                and event.start > first_diabetes_code_date
            ):
                break

            if event.value is None or type(event.value) is memoryview:
                continue

            if event.code == self.hba1c_lab_code:
                is_diabetes = float(event.value) > high_cutoff_threshold
                if last_trigger is None or (
                    event.start - last_trigger > self.last_trigger_timedelta
                ):
                    labels.append(
                        Label(
                            time=event.start - datetime.timedelta(minutes=1),
                            value=is_diabetes,
                        )
                    )
                    last_trigger = event.start

                if is_diabetes:
                    break

        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"


##########################################################
##########################################################
# Labeling functions derived from OMOPConceptCodeLabeler
##########################################################
##########################################################


class HypoglycemiaCodeLabeler(OMOPConceptCodeLabeler):
    """Apply a label for whether a patient has at 1+ explicitly
    coded occurrence(s) of Hypoglycemia in `time_horizon`."""

    # fmt: off
    original_omop_concept_codes = [
        'SNOMED/267384006', 'SNOMED/421725003', 'SNOMED/719216001',
        'SNOMED/302866003', 'SNOMED/237633009', 'SNOMED/120731000119103',
        'SNOMED/190448007', 'SNOMED/230796005', 'SNOMED/421437000',
        'SNOMED/52767006', 'SNOMED/237637005', 'SNOMED/84371000119108'
    ]
    # fmt: on


class AKICodeLabeler(OMOPConceptCodeLabeler):
    """Apply a label for whether a patient has at 1+ explicitly
    coded occurrence(s) of AKI in `time_horizon`."""

    # fmt: off
    original_omop_concept_codes = [
        'SNOMED/14669001', 'SNOMED/298015003', 'SNOMED/35455006',
    ]
    # fmt: on


class AnemiaCodeLabeler(OMOPConceptCodeLabeler):
    """Apply a label for whether a patient has at 1+ explicitly
    coded occurrence(s) of Anemia in `time_horizon`."""

    # fmt: off
    original_omop_concept_codes = [
        'SNOMED/271737000', 'SNOMED/713496008', 'SNOMED/713349004', 'SNOMED/767657005',
        'SNOMED/111570005', 'SNOMED/691401000119104', 'SNOMED/691411000119101',
    ]
    # fmt: on


class HyperkalemiaCodeLabeler(OMOPConceptCodeLabeler):
    """Apply a label for whether a patient has at 1+ explicitly
    coded occurrence(s) of Hyperkalemia in `time_horizon`."""

    # fmt: off
    original_omop_concept_codes = [
        'SNOMED/14140009',
    ]
    # fmt: on


class HyponatremiaCodeLabeler(OMOPConceptCodeLabeler):
    """Apply a label for whether a patient has at 1+ explicitly
    coded occurrence(s) of Hyponatremia in `time_horizon`."""

    # fmt: off
    original_omop_concept_codes = [
        'SNOMED/267447008', 'SNOMED/89627008'
    ]
    # fmt: on


class ThrombocytopeniaCodeLabeler(OMOPConceptCodeLabeler):
    """Apply a label for whether a patient has at 1+ explicitly
    coded occurrence(s) of Thrombocytopenia in `time_horizon`."""

    # fmt: off
    original_omop_concept_codes = [
        'SNOMED/267447008', 'SNOMED/89627008',
    ]
    # fmt: on


class NeutropeniaCodeLabeler(OMOPConceptCodeLabeler):
    """Apply a label for whether a patient has at 1+ explicitly
    coded occurrence(s) of Neutkropenia in `time_horizon`."""

    # fmt: off
    original_omop_concept_codes = [
        'SNOMED/165517008',
    ]
    # fmt: on


##########################################################
##########################################################
# Other labeling functions
##########################################################
##########################################################


class OpioidOverdoseLabeler(TimeHorizonEventLabeler):
    """
    TODO - check
    The opioid overdose labeler predicts whether or not an opioid overdose will occur in the time horizon
    after being prescribed opioids.
    It is conditioned on the patient being prescribed opioids.
    """

    def __init__(
        self, ontology: extension_datasets.Ontology, time_horizon: TimeHorizon
    ):
        self.time_horizon: TimeHorizon = time_horizon

        dictionary = ontology.get_dictionary()
        icd9_codes: List[str] = [
            "E850.0",
            "E850.1",
            "E850.2",
            "965.00",
            "965.01",
            "965.02",
            "965.09",
        ]
        icd10_codes: List[str] = ["T40.0", "T40.1", "T40.2", "T40.3", "T40.4"]

        self.overdose_codes: Set[int] = set()
        for code in icd9_codes:
            self.overdose_codes |= _get_all_children(
                ontology, dictionary.index("ICD9CM/" + code)
            )
        for code in icd10_codes:
            self.overdose_codes |= _get_all_children(
                ontology, dictionary.index("ICD10CM/" + code)
            )

        self.opioid_codes = _get_all_children(
            ontology, dictionary.index("ATC/N02A")
        )

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return the start times of this patient's events whose `code` is in `self.overdose_codes`."""
        times: List[datetime.datetime] = []
        for event in patient.events:
            if event.code in self.overdose_codes:
                times.append(event.start)
        return times

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a sorted list containing the datetimes at which we'll make a prediction."""
        times: List[datetime.datetime] = []
        for event in patient.events:
            if event.code in self.opioid_codes:
                times.append(event.start)
        return times

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon

    def get_labeler_type(self) -> LabelType:
        return "boolean"


class IsMaleLabeler(Labeler):
    """Apply a label for whether or not a patient is male or not.

    The prediction time is on admission.

    This is primarily intended as a "debugging" labeler that should be "trivial" and get 1.0 AUROC.

    """

    def __init__(self, ontology: extension_datasets.Ontology):
        self.male_code: int = ontology.get_dictionary().index("Gender/M")

    def label(self, patient: Patient) -> List[Label]:
        """Label this patient as Male (TRUE) or not (FALSE)."""
        # Determine if patient is male
        is_male: bool = self.male_code in [e.code for e in patient.events]

        # Apply `is_male` label to every admission
        labels: List[Label] = []
        for event in patient.events:
            if event.code in get_inpatient_admission_concepts():
                labels.append(Label(time=event.start, value=is_male))
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"


if __name__ == "__main__":
    pass
