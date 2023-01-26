"""Labeling functions for OMOP data."""
from __future__ import annotations

import datetime
from collections import deque, defaultdict
from typing import Dict, List, Optional, Set, Tuple, Callable

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

def move_datetime_to_end_of_day(date: datetime.datetime) -> datetime.datetime:
    return date.replace(hour=23, minute=59, second=59)

def get_visit_codes(ontology: extension_datasets.Ontology) -> List[int]:
    return [ y for x in get_visit_concepts() for y in _get_all_children(ontology, ontology.get_dictionary().index(x)) ]

def get_inpatient_admission_codes(ontology: extension_datasets.Ontology) -> List[int]:
    return [ y for x in get_inpatient_admission_concepts() for y in _get_all_children(ontology, ontology.get_dictionary().index(x)) ]

def group_events_by_visit_id(patient: Patient) -> Dict[int, List[Event]]:
    """Return a dict mapping `visit_id` to a list of events in that visit."""
    visit_id_to_events: Dict[int, List[Event]] = defaultdict(list)
    for e in patient.events:
        if e.visit_id is None:
            # Ignore events that don't have a `visit_id`
            continue
        visit_id_to_events[e.visit_id].append(e)
    return dict(visit_id_to_events)

# TODO - Move this visit transformation logic to ETL pipeline
def group_inpatient_events_by_visit_id(patient: Patient, ontology: extension_datasets.Ontology) -> Dict[int, List[Event]]:
    """Return a dict mapping `visit_id` to a list of events in that INPATIENT visit (i.e. discard all non-INPATIENT visits), 
    based on combining the `visit_occurrence` and `visit_detail` OMOP table.
    
    If multiple `visit_detail` events have the same `visit_id`, this only keeps the first one.
    
    If multiple `visit_occurrence` events have the same `visit_id`, this prioritizes whichever one has an "inpatient" label.
    """
    events_by_visit_id: Dict[int, List[Event]] = group_events_by_visit_id(patient)
    admission_codes: List[int] = get_inpatient_admission_codes(ontology)
    # In the below code, we combine the best of both the `visit_occurrence` and `visit_detail` tables
    # to get more accurate inpatient visit times.
    #
    # Loop through all patient's events, keeping track of each unique `visit_id` we see
    # belonging to an event from the `visit_occurrence` table. If we see a corresponding event
    # from the `visit_occurrence` table with the same `visit_id` as an event from the
    # `visit_detail` table, then assign the `visit_occurrence` event's code to the `visit_detail` event.
    #
    # This is necessary b/c the `visit_detail` table keeps accurate track of (start, end) times but doesn't
    # distinguish inpatient from outpatient visits, while the `visit_occurrence` table does explicitly label
    # inpatient visits but has inaccurate (start, end) event times
    inpatient_events_by_visit_id = {}
    for visit_id, events in events_by_visit_id.items():
        is_inpatient: bool = False
        event: Optional[Event] = None
        for e in events:
            if e.omop_table == 'visit_occurrence':
                # Track inpatient status
                # If multiple `visit_occurrence` events for same visit, then the 'inpatient' label overrides the others
                is_inpatient = is_inpatient or (e.code in admission_codes)
            elif e.omop_table == 'visit_detail':
                # Track (start, end) status
                # We use the `visit_detail` event as the canonical event for this visit
                if event is None:
                    # If multiple `visit_detail` events with the same `visit_id`, then we only keep the first one
                    event = e
        if is_inpatient and event:
            inpatient_events_by_visit_id[visit_id] = events
    return inpatient_events_by_visit_id

def get_inpatient_admission_discharge_times(
    patient: Patient, ontology: extension_datasets.Ontology
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """Return a list of (admission, discharge) times for all inpatient visits."""
    inpatient_events_by_visit_id = group_inpatient_events_by_visit_id(patient, ontology)
    admission_discharge_times: List[Tuple[datetime.datetime, datetime.datetime]] = []
    for visit_id, events in inpatient_events_by_visit_id.items():
        for e in events:
            if e.omop_table == 'visit_detail':
                if e.start is None:
                    raise RuntimeError(f"Every `visit_detail` event must have a start time, but none found for {e}")
                if e.end is None:
                    raise RuntimeError(f"Every `visit_detail` event must have a end time, but none found for {e}")
                admission_discharge_times.append((e.start, e.end))
                # Only keep the first (admission, discharge) time that we see for this visit, so
                # break after we find it
                break
    return admission_discharge_times

def map_omop_concept_ids_to_piton_codes(
    ontology: extension_datasets.Ontology, omop_concept_ids: List[int], is_ontology_expansion: bool = True,
) -> Set[int]:
    """Maps OMOP concept IDs (e.g. 3939430) => Piton codes (e.g. 123).
        If `is_ontology_expansion` is True, then this function will also return all children of the given codes.
    """
    codes: Set[int] = set()
    for omop_concept_id in omop_concept_ids:
        # returns `None` if `omop_concept_id` is not found in the ontology
        piton_code: Optional[int] = ontology.get_code_from_concept_id(omop_concept_id)
        if piton_code is None:
            print(f"OMOP Concept ID {omop_concept_id} not found in ontology")
        else:
            codes.update(_get_all_children(ontology, piton_code) if is_ontology_expansion else { piton_code })
    return codes

def map_omop_concept_codes_to_piton_codes(
    ontology: extension_datasets.Ontology, omop_concept_codes: List[str], is_ontology_expansion: bool = True,
) -> Set[int]:
    """Maps OMOP codes (e.g. "LOINC/123") => Piton codes (e.g. 123).
        If `is_ontology_expansion` is True, then this function will also return all children of the given codes.
    """
    codes: Set[int] = set()
    for omop_concept_code in omop_concept_codes:
        try:
            piton_code = ontology.get_dictionary().index(omop_concept_code)
        except ValueError:
            raise ValueError(
                f"OMOP Concept Code {omop_concept_code} not found in ontology."
            )
        codes.update(_get_all_children(ontology, piton_code) if is_ontology_expansion else { piton_code })
    return codes

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

    Prediction Time: Start of each visit (adjusted by `self.prediction_adjustment_timedelta` if provided)

    IMPORTANT: This labeler assumes that every event has a `event.visit_id` property.
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        outcome_codes: List[int],
        prediction_time_adjustment_func: Optional[Callable] = None,
    ):
        """`prediction_time_adjustment_func` is a function that takes in a `datetime.datetime` and returns a different `datetime.datetime`."""
        self.ontology = ontology
        self.outcome_codes: List[int] = outcome_codes
        self.prediction_time_adjustment_func: Callable = prediction_time_adjustment_func if prediction_time_adjustment_func is not None else lambda x: x

    def label(self, patient: Patient) -> List[Label]:
        """Label all visits with whether the patient experiences outcomes
        in `self.outcome_codes` during each visit."""
        events_by_visit_id: Dict[int, List[Event]] = group_events_by_visit_id(patient)
        return self.label_each_visit(events_by_visit_id)
    
    def label_each_visit(self, events_by_visit_id: Dict[int, List[Event]]):
        """Given a set of events grouped by `visit_id`, label each visit with whether the patient experiences outcomes.
        
        This is separated from the `label()` function so that it can be used by other labelers that extend this class with
        different filtering criteria applied to visits (i.e. just Inpatient visits, just Outpatient visits, etc.)
        """
        labels: List[Label] = []
        # Loop through all visits in patient, check if outcome occurs during visit
        # and if so, mark that it occurred in `labels`.
        for visit_id, events in events_by_visit_id.items():
            is_outcome_occurs: bool = False
            visit_event: Optional[Event] = None
            for e in events:
                if e.omop_table == 'visit_detail':
                    # Track the (start, end) of this visit_id from its `visit_detail` event
                    visit_event = e
                if e.code in self.outcome_codes:
                    # This is an outcome event
                    is_outcome_occurs: bool = True
            if visit_event is None:
                raise RuntimeError(f"Every `visit_id` must have an event from the `visit_detail` OMOP table associated with it, but no `visit_detail` events were found with `visit_id={visit_id}`.")
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(visit_event.start)
            labels.append(Label(prediction_time, is_outcome_occurs))
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"


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
                and returns a different `datetime.datetime`.
        """
        self.outcome_codes: List[int] = outcome_codes
        self.time_horizon: TimeHorizon = time_horizon
        self.prediction_codes: Optional[List[int]] = prediction_codes
        self.prediction_time_adjustment_func: Callable = prediction_time_adjustment_func if prediction_time_adjustment_func is not None else lambda x: x

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return each event's start time (possibly modified by prediction_time_adjustment_func) 
        as the time to make a prediction. Default to all events whose `code` is in `self.prediction_codes`."""
        times: List[datetime.datetime] = []
        for e in patient.events:
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(e.start)
            if (self.prediction_codes is None) or (e.code in self.prediction_codes):
                times.append(prediction_time)
        return times
    
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
        prediction_time_adjustment_func: Optional[Callable] = None,
    ):
        outcome_codes: List[int] = list(map_omop_concept_codes_to_piton_codes(ontology, self.original_omop_concept_codes, is_ontology_expansion=True))
        super().__init__(
            outcome_codes=outcome_codes,
            time_horizon=time_horizon,
            prediction_codes=prediction_codes,
            prediction_time_adjustment_func=prediction_time_adjustment_func,
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
        prediction_time_adjustment_func: Optional[Callable] = None,
    ):
        """Create a Mortality labeler."""
        outcome_codes = list(map_omop_concept_codes_to_piton_codes(ontology, get_death_concepts(), is_ontology_expansion=True))

        super().__init__(
            outcome_codes=outcome_codes,
            time_horizon=time_horizon,
            prediction_codes=prediction_codes,
            prediction_time_adjustment_func=prediction_time_adjustment_func,
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
        prediction_time_adjustment_func: Optional[Callable] = None,
    ):
        concept_codes: List[str] = ["SNOMED/55464009", "SNOMED/201436003"]
        outcome_codes = list(map_omop_concept_codes_to_piton_codes(ontology, concept_codes, is_ontology_expansion=True))
        super().__init__(
            outcome_codes=outcome_codes,
            time_horizon=time_horizon,
            prediction_codes=prediction_codes,
            prediction_time_adjustment_func=prediction_time_adjustment_func,
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
