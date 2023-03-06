"""Labeling functions for OMOP data."""
from __future__ import annotations

import datetime
from abc import abstractmethod
from collections import deque
from typing import Callable, List, Union, Set, Tuple, Optional

from .. import Event, Patient
from ..extension import datasets as extension_datasets
from .core import Label, Labeler, LabelType, TimeHorizon, TimeHorizonEventLabeler


def get_visit_concepts() -> List[str]:
    return ["Visit/IP"]


def get_inpatient_admission_concepts() -> List[str]:
    return ["Visit/IP"]


def get_death_concepts() -> List[str]:
    return [
        "Death Type/OMOP generated",
        "Condition Type/OMOP4822053",
    ]

def get_icu_visit_detail_concepts() -> List[str]:
    return [
        'CARE_SITE/7928450', 
        'CARE_SITE/7930385', 
        'CARE_SITE/7930600', 
        'CARE_SITE/7928852', 
        'CARE_SITE/7928619', 
        'CARE_SITE/7929727', 
        'CARE_SITE/7928675', 
        'CARE_SITE/7930225', 
        'CARE_SITE/7928759', 
        'CARE_SITE/7928227', 
        'CARE_SITE/7928810', 
        'CARE_SITE/7929179', 
        'CARE_SITE/7928650', 
        'CARE_SITE/7929351', 
        'CARE_SITE/7928457', 
        'CARE_SITE/7928195', 
        'CARE_SITE/7930681', 
        'CARE_SITE/7930670', 
        'CARE_SITE/7930176', 
        'CARE_SITE/7931420', 
        'CARE_SITE/7929149', 
        'CARE_SITE/7930857', 
        'CARE_SITE/7931186', 
        'CARE_SITE/7930934', 
        'CARE_SITE/7930924',
    ]

def move_datetime_to_end_of_day(date: datetime.datetime) -> datetime.datetime:
    return date.replace(hour=23, minute=59, second=59)


def does_exist_event_within_time_range(patient: Patient, start: datetime.datetime, end: datetime.datetime, exclude_event_idxs: List[int] = []) -> bool:
    """Return True if there is at least one event within the given time range for this patient.
    If `exclude_event_idxs` is provided, exclude events with those indexes in `patient.events` from the search."""
    excluded = set(exclude_event_idxs)
    for idx, e in enumerate(patient.events):
        if idx in excluded:
            continue
        if start <= e.start <= end:
            return True
    return False

def map_omop_concepts_to_femr_codes(ontology: extension_datasets.Ontology, 
                                    concepts: List[str],
                                    is_get_children: bool = True, 
                                    is_silent_not_found_error: bool = True) -> Set[int]:
    """Converts OMOP concept names (strings from OHDSI) to FEMR event codes (integers internal to FEMR).

    Args:
        ontology (extension_datasets.Ontology): FEMR ontology
        concepts (List[str]): List of OMOP concept names
        is_get_children (bool, optional): If TRUE, then return all children of a concept, in addition to that concept. Defaults to True.
        is_silent_not_found_error (bool, optional): If TRUE, then if a concept cannot be found in `ontology` an error is NOT raised. Defaults to True.

    Returns:
        Set[int]: Set of FEMR event codes
    """
    codes = set()
    for x in concepts:
        try:
            femr_code: int = ontology.get_dictionary().index(x)
            codes.add(femr_code)
            if is_get_children:
                for y in _get_all_children(ontology, femr_code):
                    codes.add(y)
        except:
            if not is_silent_not_found_error:
                raise ValueError(f"Concept {x} not found in `ontology`.")
    return codes

def get_visit_codes(ontology: extension_datasets.Ontology) -> Set[int]:
    return map_omop_concepts_to_femr_codes(ontology, get_visit_concepts(), is_get_children=True, is_silent_not_found_error=True)

def get_icu_visit_detail_codes(ontology: extension_datasets.Ontology) -> Set[int]:
    return map_omop_concepts_to_femr_codes(ontology, get_icu_visit_detail_concepts(), is_get_children=True, is_silent_not_found_error=True)

def get_inpatient_admission_codes(ontology: extension_datasets.Ontology) -> Set[int]:
    # Don't get children here b/c it adds noise (i.e. "Medicare Specialty/AO")
    return map_omop_concepts_to_femr_codes(ontology, get_inpatient_admission_concepts(), is_get_children=False, is_silent_not_found_error=True)

def get_icu_events(patient: Patient, ontology: extension_datasets.Ontology, is_return_idx: bool = False) -> Union[List[Event], List[Tuple[int, Event]]]:
    """Return all ICU events for this patient.
    If `is_return_idx` is True, then return a list of tuples (event, idx) where `idx` is the index of the event in `patient.events`."""
    icu_visit_detail_codes: Set[int] = get_icu_visit_detail_codes(ontology)
    events: Union[List[Event], List[Tuple[int, Event]]] = []
    for idx, e in enumerate(patient.events):
        if e.code in icu_visit_detail_codes and e.omop_table == "visit_detail":
            # Error checking
            if e.start is None or e.end is None:
                raise RuntimeError(f"Event {e} for patient {patient.patient_id} cannot have `None` as its `start` or `end` attribute.")
            elif e.start > e.end:
                raise RuntimeError(f"Event {e} for patient {patient.patient_id} cannot have `start` after `end`.")
            if is_return_idx:
                events.append((idx, e)) # type: ignore
            else:
                events.append(e)
    return events

def get_inpatient_admission_events(patient: Patient, ontology: extension_datasets.Ontology) -> List[Event]:
    admission_codes: Set[int] = get_inpatient_admission_codes(ontology)
    events: List[Event] = []
    for e in patient.events:
        if e.code in admission_codes and e.omop_table == "visit_occurrence":
            # Error checking
            if e.start is None or e.end is None:
                raise RuntimeError(f"Event {e} cannot have `None` as its `start` or `end` attribute.")
            elif e.start > e.end:
                raise RuntimeError(f"Event {e} cannot have `start` after `end`.")
            # Drop single point in time events
            if e.start == e.end:
                continue
            events.append(e)
    return events


def get_inpatient_admission_discharge_times(
    patient: Patient, ontology: extension_datasets.Ontology
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """Return a list of all admission/discharge times for this patient."""
    events: List[Event] = get_inpatient_admission_events(patient, ontology)
    times: List[Tuple[datetime.datetime, datetime.datetime]] = []
    for e in events:
        if e.end is None:
            raise RuntimeError(f"Event {e} cannot have `None` as its `end` attribute.")
        if e.start > e.end:
            raise RuntimeError(f"Event {e} cannot have `start` after `end`.")
        times.append((e.start, e.end))
    return times


# TODO - move this into the ontology class
def map_omop_concept_ids_to_femr_codes(
    ontology: extension_datasets.Ontology,
    omop_concept_ids: List[int],
    is_ontology_expansion: bool = True,
) -> Set[int]:
    """Maps OMOP concept IDs (e.g. 3939430) => FEMR codes (e.g. 123).
    If `is_ontology_expansion` is True, then this function will also return all children of the given codes.
    """
    codes: Set[int] = set()
    for omop_concept_id in omop_concept_ids:
        # returns `None` if `omop_concept_id` is not found in the ontology
        femr_code: Optional[int] = ontology.get_code_from_concept_id(omop_concept_id)  # type: ignore
        if femr_code is None:
            print(f"OMOP Concept ID {omop_concept_id} not found in ontology")
        else:
            codes.update(_get_all_children(ontology, femr_code) if is_ontology_expansion else {femr_code})
    return codes


# TODO - move this into the ontology class
def map_omop_concept_codes_to_femr_codes(
    ontology: extension_datasets.Ontology,
    omop_concept_codes: List[str],
    is_ontology_expansion: bool = True,
    is_silent_not_found_error: bool = True,
) -> Set[int]:
    """Maps OMOP codes (e.g. "LOINC/123") => FEMR codes (e.g. 123).
    If `is_ontology_expansion` is True, then this function will also return all children of the given codes.
    If `is_silent_not_found_error` is True, then this function will NOT raise an error if a given OMOP concept ID is not found in the ontology.
    """
    codes: Set[int] = set()
    for omop_concept_code in omop_concept_codes:
        try:
            femr_code: int = ontology.get_dictionary().index(omop_concept_code)
            codes.update(_get_all_children(ontology, femr_code) if is_ontology_expansion else {femr_code})
        except ValueError:
            if not is_silent_not_found_error:
                raise ValueError(f"OMOP Concept Code {omop_concept_code} not found in ontology.")
    return codes


# TODO - move this into the ontology class
def _get_all_children(ontology: extension_datasets.Ontology, code: int) -> Set[int]:
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
    The `WithinVisitLabeler` predicts whether or not a patient experiences a specific event
    (as returned by `self.get_outcome_times()`) within each visit.

    Very similar to `TimeHorizonLabeler`, except here we use visits themselves as our time horizon.

    Prediction Time: Start of each visit (adjusted by `self.prediction_adjustment_timedelta` if provided)
    Time horizon: By end of visit
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        visit_start_adjust_func: Callable = lambda x: x,
        visit_end_adjust_func: Callable = lambda x: x,
    ):
        """The argument `visit_start_adjust_func` is a function that takes in a `datetime.datetime`
        and returns a different `datetime.datetime`."""
        self.ontology: extension_datasets.Ontology = ontology
        self.visit_start_adjust_func: Callable = visit_start_adjust_func
        self.visit_end_adjust_func: Callable = visit_end_adjust_func

    @abstractmethod
    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a list of all times when the patient experiences an outcome"""
        return []

    @abstractmethod
    def get_visit_events(self, patient: Patient) -> List[Event]:
        """Return a list of all visits we want to consider (useful for limiting to inpatient visits)."""
        return []

    def label(self, patient: Patient) -> List[Label]:
        """
        Label all visits returned by `self.get_visit_events()`with whether the patient
        experiences an outcome in `self.outcome_codes` during each visit.
        """
        visits: List[Event] = self.get_visit_events(patient)
        prediction_start_times: List[datetime.datetime] = [
            self.visit_start_adjust_func(visit.start) for visit in visits
        ]
        prediction_end_times: List[datetime.datetime] = [self.visit_end_adjust_func(visit.end) for visit in visits]
        outcome_times: List[datetime.datetime] = self.get_outcome_times(patient)

        # For each visit, check if there is an outcome which occurs within the (start, end) of the visit
        results: List[Label] = []
        curr_outcome_idx: int = 0
        for prediction_idx, (prediction_start, prediction_end) in enumerate(zip(prediction_start_times, prediction_end_times)):
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
                    f" This is for patient with patient_id `{patient.patient_id}`."
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
                results.append(Label(time=prediction_start, value=True))
            elif not is_censored:
                # Not censored + no outcome => FALSE
                results.append(Label(time=prediction_start, value=False))

        return results

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
        prediction_time_adjustment_func: Callable = lambda x: x,
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
        self.outcome_codes: List[int] = outcome_codes
        self.time_horizon: TimeHorizon = time_horizon
        self.prediction_codes: Optional[List[int]] = prediction_codes
        self.prediction_time_adjustment_func: Callable = prediction_time_adjustment_func

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return each event's start time (possibly modified by prediction_time_adjustment_func)
        as the time to make a prediction. Default to all events whose `code` is in `self.prediction_codes`."""
        times: List[datetime.datetime] = []
        last_time = None
        for e in patient.events:
            prediction_time: datetime.datetime = self.prediction_time_adjustment_func(e.start)
            if ((self.prediction_codes is None) or (e.code in self.prediction_codes)) and (
                last_time != prediction_time
            ):
                times.append(prediction_time)
                last_time = prediction_time
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
        prediction_time_adjustment_func: Callable = lambda x: x,
    ):
        outcome_codes: List[int] = list(
            map_omop_concept_codes_to_femr_codes(
                ontology,
                self.original_omop_concept_codes,
                is_ontology_expansion=True,
            )
        )
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
        prediction_time_adjustment_func: Callable = lambda x: x,
    ):
        """Create a Mortality labeler."""
        outcome_codes = list(
            map_omop_concept_codes_to_femr_codes(ontology, get_death_concepts(), is_ontology_expansion=True)
        )

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
        prediction_time_adjustment_func: Callable = lambda x: x,
    ):
        concept_codes: List[str] = ["SNOMED/55464009", "SNOMED/201436003"]
        outcome_codes = list(map_omop_concept_codes_to_femr_codes(ontology, concept_codes, is_ontology_expansion=True))
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
        last_trigger_timedelta: datetime.timedelta = datetime.timedelta(days=180),
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
            if first_diabetes_code_date is not None and event.start > first_diabetes_code_date:
                break

            if event.value is None or type(event.value) is memoryview:
                continue

            if event.code == self.hba1c_lab_code:
                is_diabetes = float(event.value) > high_cutoff_threshold
                if last_trigger is None or (event.start - last_trigger > self.last_trigger_timedelta):
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

    def __init__(self, ontology: extension_datasets.Ontology, time_horizon: TimeHorizon):
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
            self.overdose_codes |= _get_all_children(ontology, dictionary.index("ICD9CM/" + code))
        for code in icd10_codes:
            self.overdose_codes |= _get_all_children(ontology, dictionary.index("ICD10CM/" + code))

        self.opioid_codes = _get_all_children(ontology, dictionary.index("ATC/N02A"))

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



##########################################################
##########################################################
# MIMIC-III Benchmark Tasks
# See: https://www.nature.com/articles/s41597-019-0103-9/figures/7 for
# details on how this was reproduced.
#
# Citation: Harutyunyan, H., Khachatrian, H., Kale, D.C. et al. 
# Multitask learning and benchmarking with clinical time series data. 
# Sci Data 6, 96 (2019). https://doi.org/10.1038/s41597-019-0103-9
##########################################################
##########################################################


class Harutyunyan_DecompensationLabeler(CodeLabeler):
    """Decompensation prediction task from Harutyunyan et al. 2019.
    
    Hourly binary prediction task on whether the patient dies in the next 24 hours.
    Make prediction every 60 minutes after ICU admission, starting at hour 4.
    
    Excludes:
        - ICU admissions with no length-of-stay (i.e. `event.end is None` )
        - ICU admissions < 4 hours
        - ICU admissions with no events
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        # Next 24 hours
        time_horizon = TimeHorizon(datetime.timedelta(hours=0), datetime.timedelta(hours=24))
        # Death events
        outcome_codes = list(
            map_omop_concept_codes_to_femr_codes(ontology, get_death_concepts(), is_ontology_expansion=True)
        )
        # Save ontology for `get_prediction_times()`
        self.ontology = ontology

        super().__init__(
            outcome_codes=outcome_codes,
            time_horizon=time_horizon,
        )
    
    def is_apply_censoring(self) -> bool:
        """Consider censored patients to be alive."""
        return False

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a list of every hour after every ICU visit. 
        Note that this requires creating an artificial event for each hour since there will only be one true
        event per ICU admission, but we'll need to create many subevents (at each hour) within this event.
        Also note that these events may not align with :00 minutes if the ICU visit does not start exactly "on the hour".
    
        Excludes:
            - ICU admissions with no length-of-stay (i.e. `event.end is None` )
            - ICU admissions < 4 hours
            - ICU admissions with no events
        """
        times: List[datetime.datetime] = []
        icu_events: List[Tuple[int, Event]] = get_icu_events(patient, self.ontology, is_return_idx=True) # type: ignore
        icu_event_idxs = [idx for idx, __ in icu_events]
        for __, e in icu_events:
            if (
                e.end is not None
                and e.end - e.start >= datetime.timedelta(hours=4)
                and does_exist_event_within_time_range(patient, 
                                                       e.start, 
                                                       e.end, 
                                                       exclude_event_idxs=icu_event_idxs)
            ):
                # Record every hour after admission (i.e. every hour between `e.start` and `e.end`),
                # but only after 4 hours have passed (i.e. start at `e.start + 4 hours`)
                event_time = e.start + datetime.timedelta(hours=4)
                while event_time < e.end:
                    times.append(event_time)
                    event_time += datetime.timedelta(hours=1)
        return times


class Harutyunyan_MortalityLabeler(WithinVisitLabeler):
    """In-hospital mortality prediction task from Harutyunyan et al. 2019.
    Single binary prediction task of whether patient dies within ICU admission 48 hours after admission.
    Make prediction 48 hours into ICU admission. 
    
    Excludes:
        - ICU admissions with no length-of-stay (i.e. `event.end is None` )
        - ICU admissions < 48 hours
        - ICU admissions with no events before 48 hours
    """
    
    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        visit_start_adjust_func: Callable = lambda x: x,
        visit_end_adjust_func: Callable = lambda x: x,
    ):
        visit_start_adjust_func = lambda x: x + datetime.timedelta(hours=48) # Make prediction 48 hours into ICU admission
        super().__init__(ontology, visit_start_adjust_func, visit_end_adjust_func)
    
    def is_apply_censoring(self) -> bool:
        """Consider censored patients to be alive."""
        return False

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a list of all times when the patient experiences an outcome"""
        outcome_codes = list(
            map_omop_concept_codes_to_femr_codes(self.ontology, get_death_concepts(), is_ontology_expansion=True)
        )
        times: List[datetime.datetime] = []
        for e in patient.events:
            if e.code in outcome_codes:
                times.append(e.start)
        return times

    def get_visit_events(self, patient: Patient) -> List[Event]:
        """Return a list of all ICU visits > 48 hours.
        
        Excludes:
            - ICU admissions with no length-of-stay (i.e. `event.end is None` )
            - ICU admissions < 48 hours
            - ICU admissions with no events before 48 hours
        """
        icu_events: List[Tuple[int, Event]] = get_icu_events(patient, self.ontology, is_return_idx=True) # type: ignore
        icu_event_idxs = [idx for idx, __ in icu_events]
        valid_events: List[Event] = []
        for __, e in icu_events:
            if (
                e.end is not None
                and e.end - e.start >= datetime.timedelta(hours=48)
                and does_exist_event_within_time_range(patient, 
                                                       e.start, 
                                                       e.start + datetime.timedelta(hours=48), 
                                                       exclude_event_idxs=icu_event_idxs)
            ):
                valid_events.append(e)
        return valid_events


class Harutyunyan_LengthOfStayLabeler(CodeLabeler):
    # TODO - need to create multiclass labeler
    """LOS prediciton task from Harutyunyan et al. 2019.
    
    Hourly multiclass prediction task on the patient's remaining length-of-stay in the ICU.
    Make prediction every 60 minutes after ICU admission, starting at hour 4.
    
    Classification buckets:
        1. < 24 hours
        2-8. Day-long buckets for each day of the first week
        9. >1 week but <=2 weeks
        10. >2 weeks
    
    Excludes:
        - ICU admissions with no length-of-stay (i.e. `event.end is None` )
        - ICU admissions < 4 hours
        - ICU admissions with no events
    """

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        # TODO
        # Next 24 hours
        time_horizon = TimeHorizon(datetime.timedelta(hours=0), datetime.timedelta(hours=24))
        # Death events
        outcome_codes = list(
            map_omop_concept_codes_to_femr_codes(ontology, get_death_concepts(), is_ontology_expansion=True)
        )
        # Save ontology for `get_prediction_times()`
        self.ontology = ontology

        super().__init__(
            outcome_codes=outcome_codes,
            time_horizon=time_horizon,
        )
    
    def is_apply_censoring(self) -> bool:
        """Consider censored patients to be alive."""
        return False

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return a list of every hour after every ICU visit. 
        Note that this requires creating an artificial event for each hour since there will only be one true
        event per ICU admission, but we'll need to create many subevents (at each hour) within this event.
        Also note that these events may not align with :00 minutes if the ICU visit does not start exactly "on the hour".
        
        This is the same as `get_prediction_times()` for `Harutyunyan_MortalityLabeler`.
        
        Excludes:
            - ICU admissions with no length-of-stay (i.e. `event.end is None` )
            - ICU admissions < 4 hours
            - ICU admissions with no events
        """
        times: List[datetime.datetime] = []
        icu_events: List[Tuple[int, Event]] = get_icu_events(patient, self.ontology, is_return_idx=True) # type: ignore
        icu_event_idxs = [idx for idx, __ in icu_events]
        for __, e in icu_events:
            if (
                e.end is not None
                and e.end - e.start >= datetime.timedelta(hours=4)
                and does_exist_event_within_time_range(patient, 
                                                       e.start, 
                                                       e.end, 
                                                       exclude_event_idxs=icu_event_idxs)
            ):
                # Record every hour after admission (i.e. every hour between `e.start` and `e.end`),
                # but only after 4 hours have passed (i.e. start at `e.start + 4 hours`)
                event_time = e.start + datetime.timedelta(hours=4)
                while event_time < e.end:
                    times.append(event_time)
                    event_time += datetime.timedelta(hours=1)
        return times

if __name__ == "__main__":
    pass
