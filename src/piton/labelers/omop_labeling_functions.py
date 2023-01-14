"""Labeling functions for OMOP data."""
from __future__ import annotations

import datetime
from typing import List, Set, Tuple, Optional
from collections import deque

from .. import Event, Patient
from ..extension import datasets as extension_datasets
from .core import (
    FixedTimeHorizonEventLF,
    InfiniteTimeHorizonEventLF,
    InfiniteTimeHorizon,
    Label,
    LabelingFunction,
    LabelType,
    TimeHorizon,
)

def _get_all_children(ontology: extension_datasets.Ontology, code:int) -> Set[int]:
    children_code_set = set([code])
    parent_deque = deque([code])
    
    print(parent_deque)

    while len(parent_deque) > 0:
        temp_parent_code = parent_deque.popleft()
        for temp_child_code in ontology.get_children(temp_parent_code):
            children_code_set.add(temp_child_code)
            parent_deque.append(temp_child_code)
        
    return children_code_set

##########################################################
# Labeling functions derived from FixedTimeHorizonEventLF
##########################################################

class CodeLF(FixedTimeHorizonEventLF):
    """Apply a label based on a 1+ codes' occurrence(s) over a fixed time horizon."""

    def __init__(
        self, codes: List[int], time_horizon: TimeHorizon, prediction_codes: List[int] 
    ):
        """Label events whose index in your Ontology is in `self.codes`."""
        self.codes: List[int] = codes
        self.time_horizon: TimeHorizon = time_horizon
        self.prediction_codes: List[int] = prediction_codes

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return each event's start time as the time to make a prediction.
            Default to all events whose `code` is in `self.prediction_codes`."""
        return [
            datetime.datetime.strptime(
                # TODO - Why add 23:59:00?
                str(e.start)[:10] + " 23:59:00", "%Y-%m-%d %H:%M:%S"
            )
            for e in patient.events
            if e.code in self.prediction_codes
        ]

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return the start times of this patient's events whose `code` is in `self.codes`."""
        times: List[datetime.datetime] = []
        for event in patient.events:
            if event.code in self.codes:
                times.append(event.start)
        return times

##########################################################
# Labeling functions derived from CodeLF
##########################################################

class MortalityLF(CodeLF):
    """Apply a label for whether or not a patient dies within the `time_horizon`."""

    def __init__(
        self, ontology: extension_datasets.Ontology, time_horizon: TimeHorizon
    ):
        """Create a Mortality labeler.

        Args:
            ontology (extension_datasets.Ontology): Maps code IDs to concept names
            time_horizon (TimeHorizon): An interval of time. If the event occurs during this time horizon, then
                the label is TRUE. Otherwise, FALSE.

        Raises:
            ValueError: Raised if there are multiple unique codes that map to the death code
        """
        CODE_DEATH = "Condition Type/OMOP4822053"
        INPATIENT_VISIT_CODE = "Visit/IP"
        admission_code = ontology.get_dictionary().index(INPATIENT_VISIT_CODE)

        death_codes: Set[Tuple[str, int]] = set()
        for code, code_str in enumerate(ontology.get_dictionary()):
            code_str = bytes(code_str).decode("utf-8")
            if code_str == CODE_DEATH:
                death_codes.add((code_str, code))

        if len(death_codes) != 1:
            raise ValueError(
                f"Could not find exactly one death code -- instead found {len(death_codes)} codes: {str(death_codes)}"
            )
        else:
            death_code: int = list(death_codes)[0][1]
            super().__init__(
                codes=[death_code],
                time_horizon=time_horizon,
                prediction_codes=[admission_code],
            )


class DiabetesLF(CodeLF):
    """Apply a label for whether or not a patient has diabetes within the `time_horizon`."""

    def __init__(
        self, ontology: extension_datasets.Ontology, time_horizon: TimeHorizon
    ):
        """Create a Diabetes labeler.

        Args:
            ontology (extension_datasets.Ontology): Maps code IDs to concept names
            time_horizon (TimeHorizon): An interval of time. If the event occurs during this time horizon, then
                the label is TRUE. Otherwise, FALSE.

        Raises:
            ValueError: Raised if there are multiple unique codes that map to the death code
        """
        DIABETES_CODE = "SNOMED/44054006"
        INPATIENT_VISIT_CODE = "Visit/IP"
        admission_code = ontology.get_dictionary().index(INPATIENT_VISIT_CODE)

        diabetes_codes: Set[Tuple[str, int]] = set()
        for code, code_str in enumerate(ontology.get_dictionary()):
            code_str = bytes(code_str).decode("utf-8")
            if code_str == DIABETES_CODE:
                diabetes_codes.add((code_str, code))

        if len(diabetes_codes) != 1:
            raise ValueError(
                "Could not find exactly one death code -- instead found "
                f"{len(diabetes_codes)} codes: {str(diabetes_codes)}"
            )
        else:
            diabetes_code: int = list(diabetes_codes)[0][1]
            super().__init__(
                codes=[diabetes_code],
                time_horizon=time_horizon,
                prediction_codes=[admission_code],
            )


class HighHbA1cLF(LabelingFunction):
    """
    The high HbA1c labeler tries to predict whether a non-diabetic patient will test as diabetic.
    Note: This labeler will only trigger at most once every 6 months.
    """

    def __init__(
        self, 
        ontology: extension_datasets.Ontology, 
        last_trigger_days: int = 180,
    ):
        self.last_trigger_days = last_trigger_days

        HbA1c_str: str = "LOINC/4548-4"
        self.hba1c_lab_code = ontology.get_dictionary().index(HbA1c_str)

        diabetes_str: str = "SNOMED/44054006"
        diabetes_code = ontology.get_dictionary().index(diabetes_str)
        self.diabetes_codes = _get_all_children(ontology, diabetes_code)

    def label(self, patient: Patient) -> List[Label]:

        if len(patient.events) == 0:
            return []

        labels: List[Label] = []
        last_trigger: Optional[int] = None

        first_diabetes_code_date = None

        for event in patient.events:
            if event.code in self.diabetes_codes:
                first_diabetes_code_date = event.start
                break

        for event in patient.events:

            if (first_diabetes_code_date is not None and 
                event.start > first_diabetes_code_date):
                break
            
            if event.value is None or type(event.value) is memoryview:
                continue

            if event.code == self.hba1c_lab_code:
                is_diabetes = float(event.value) > 6.5
                # TODO - Type mismatch between `last_trigger` (int) and `event.start` (datetime)
                # TODO - maybe change `last_trigger` to be a datetime.timedelta(days=last_trigger) ??
                if last_trigger is None or (event.start - last_trigger).days > self.last_trigger_days:
                    labels.append(Label(time=event.start - datetime.timedelta(minutes=1), 
                                        value=is_diabetes))
                    last_trigger = event.start 
                
                if is_diabetes:
                    break
        
        return labels
    
    def get_labeler_type(self) -> LabelType:
        return "boolean"


class IsMaleLF(LabelingFunction):
    """Apply a label for whether or not a patient is male or not.

    The prediction time is on admission.

    This is primarily intended as a "debugging" labeler that should be "trivial" and get 1.0 AUROC.

    """

    def __init__(self, ontology: extension_datasets.Ontology):
        """Construct a Male labeler.

        Args:
            ontology (extension_datasets.Ontology): Maps code IDs to code names.

        Raises:
            ValueError: Raised if there is no code corresponding to inpatient visit.
        """
        INPATIENT_VISIT_CODE = "Visit/IP"
        self.male_code: int = ontology.get_dictionary().index("Gender/M")
        admission_code = ontology.get_dictionary().index(INPATIENT_VISIT_CODE)
        if admission_code is None:
            raise ValueError(
                f"Could not find inpatient visit code for: {INPATIENT_VISIT_CODE}"
            )
        else:
            self.admission_code = admission_code

    def is_inpatient_admission(self, event: Event) -> bool:
        """Return TRUE if this event is an admission."""
        return event.code == self.admission_code

    def label(self, patient: Patient) -> List[Label]:
        """Label this patient as Male (TRUE) or not (FALSE)."""
        if len(patient.events) == 0:
            return []

        labels: List[Label] = []

        is_male: bool = self.male_code in [
            event.code for event in patient.events
        ]

        for event in patient.events:
            if self.is_inpatient_admission(event):
                labels.append(
                    Label(time=event.start, value=is_male)
                )
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"

class OpioidOverdoseLabeler(FixedTimeHorizonEventLF):
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
        icd9_codes: List[str] = [ "E850.0", "E850.1", "E850.2", "965.00", "965.01", "965.02", "965.09", ]
        icd10_codes: List[str] = ["T40.0", "T40.1", "T40.2", "T40.3", "T40.4"]

        self.overdose_codes: Set[int] = set()
        for code in icd9_codes:
            self.overdose_codes |= _get_all_children(ontology, dictionary.index("ICD9CM/" + code))
        for code in icd10_codes:
            self.overdose_codes |= _get_all_children(ontology, dictionary.index("ICD10CM/" + code))

        self.opioid_codes = _get_all_children(ontology, dictionary.index("ATC/N02A"))

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return the start times of this patient's events whose `code` is in `self.codes`."""
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
        return "binary"


class LupusDiseaseLabeler(InfiniteTimeHorizonEventLF):
    """
    Label if patient is ever diagnosed with Lupus at any time in the future.
    """

    def __init__(
        self, ontology: extension_datasets.Ontology
    ):
        dictionary = ontology.get_dictionary()
        
        icd9_codes: List[str] = ["710.0"]
        icd10_codes: List[str] = ["M32"]
        self.codes = set()
        for code in icd9_codes:
            self.codes |= _get_all_children(ontology, dictionary.index("ICD9CM/" + code))
        for code in icd10_codes:
            self.codes |= _get_all_children(ontology, dictionary.index("ICD10CM/" + code))

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return the start times of this patient's events whose `code` is in `self.codes`."""
        times: List[datetime.datetime] = []
        for event in patient.events:
            if event.code in self.codes:
                times.append(event.start)
        return times

    def get_time_horizon(self) -> InfiniteTimeHorizon:
        return InfiniteTimeHorizon(start=datetime.timedelta(0)) # type: ignore

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        # TODO - Return a sorted list containing the datetimes at which we'll make a prediction.
        return []

    def get_labeler_type(self) -> LabelType:
        return "binary"


class CeliacTestLabeler(LabelingFunction):
    """
        The Celiac test labeler predicts whether or not a celiac test will be positive or negative.
        The prediction time is 24 hours before the lab results come in.
        Note: This labeler excludes patients who either already had a celiac test or were previously diagnosed.
    """

    def __init__(
        self, ontology: extension_datasets.Ontology, time_horizon: TimeHorizon
    ):
        dictionary = ontology.get_dictionary()
        self.lab_codes = _get_all_children(ontology, dictionary.index("LNC/31017-7"))
        self.celiac_codes = _get_all_children(ontology, dictionary.index("ICD9CM/579.0")) | \
                                _get_all_children(ontology, dictionary.index("ICD10CM/K90.0"))
        
        self.pos_value = 'Positive'
        self.neg_value = 'Negative'

    def label(self, patient: Patient) -> List[Label]:
        if len(patient.events) == 0:
            return []
        
        for event in patient.events:
            if event.code in self.celiac_codes:
                # This patient has Celiacs
                return []
            if (event.code in self.lab_codes
                and event.value in [ self.pos_value, self.neg_value ]
            ):
                # This patient got a Celiac lab test back
                # We'll return the Label 24 hours prior
                return [
                    Label(event.start - datetime.timedelta(hours=24), 
                          event.value == self.pos_value)
                ]
        return []

    def get_labeler_type(self) -> LabelType:
        return "binary"