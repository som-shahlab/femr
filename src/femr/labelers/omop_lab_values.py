"""Labeling functions for OMOP data based on lab values."""
from __future__ import annotations

import datetime
from abc import abstractmethod
from typing import Any, Callable, List, Optional, Set

from femr import Event, Patient
from femr.labelers import Label, Labeler, LabelType, TimeHorizon
from femr.labelers.omop import (
    OMOPConceptCodeLabeler,
    _get_all_children,
    get_inpatient_admission_events,
    map_omop_concept_codes_to_femr_codes,
)
from femr.labelers.omop_inpatient_admissions import WithinInpatientVisitLabeler

from ..extension import datasets as extension_datasets


def identity(x: Any) -> Any:
    return x


##########################################################
##########################################################
# Labelers based on Lab Values.
#
# The difference between these Labelers and the ones based on codes
# is that these Labelers are based on lab values, not coded
# diagnoses. Thus, they may catch more cases of a given
# condition due to under-coding, but they are also more
# likely to be noisy.
##########################################################
##########################################################


class InstantLabValueLabeler(Labeler):
    """Apply a multi-class label for the outcome of a lab test.

    Prediction Time: Immediately before lab result is returned (i.e. 1 minute before)
    Time Horizon: The next immediate result for this lab test
    Label: Severity level of lab

    Excludes:
        - Labels that occur at the same exact time as the very first event in a patient's history
    """

    # parent OMOP concept codes, from which all the outcomes are derived (as children in our ontology)
    original_omop_concept_codes: List[str] = []

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
    ):
        self.ontology = ontology
        self.outcome_codes: Set[int] = map_omop_concept_codes_to_femr_codes(
            ontology,
            self.original_omop_concept_codes,
            is_ontology_expansion=True,
        )

    def label(self, patient: Patient, is_show_warnings: bool = False) -> List[Label]:
        labels: List[Label] = []
        for e in patient.events:
            if patient.events[0].start == e.start:
                # Ignore events that occur at the same time as the first event in the patient's history
                continue
            if e.code in self.outcome_codes:
                # This is an outcome event
                if e.value is not None:
                    try:
                        # `e.unit` is string of form "mg/dL", "ounces", etc.
                        label: int = self.label_to_int(self.value_to_label(str(e.value), str(e.unit)))
                        prediction_time: datetime.datetime = e.start - datetime.timedelta(minutes=1)
                        labels.append(Label(prediction_time, label))
                    except Exception as exception:
                        if is_show_warnings:
                            print(
                                f"Warning: Error parsing value='{e.value}' with unit='{e.unit}'"
                                f" for code='{e.code}' @ {e.start} for patient_id='{patient.patient_id}'"
                                f" | Exception: {exception}"
                            )
        return labels

    def get_labeler_type(self) -> LabelType:
        return "categorical"

    def label_to_int(self, label: str) -> int:
        if label == "normal":
            return 0
        elif label == "mild":
            return 1
        elif label == "moderate":
            return 2
        elif label == "severe":
            return 3
        raise ValueError(f"Invalid label without a corresponding int: {label}")

    @abstractmethod
    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        """Convert `value` to a string label: "mild", "moderate", "severe", or "normal".
        NOTE: Some units have the form 'mg/dL (See scan or EMR data for detail)', so you
        need to use `.startswith()` to check for the unit you want.
        """
        return "normal"


class TomasevLabValueLabeler(Labeler):
    # TODO
    """Every 6 hours after admission, predict the maximum value for a lab value within the next 48 hours.
    Inspired by Tomasev et al. 2021

        Prediction Time: Every 360 minutes after admission
        Time Horizon: Next 48 hours
        Label: Max value of lab over next 48 hours
    """

    # parent OMOP concept codes, from which all the outcomes are derived (as children in our ontology)
    original_omop_concept_codes: List[str] = []

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        severity: str,
        visit_start_adjust_func: Callable = identity,
        visit_end_adjust_func: Callable = identity,
    ):
        self.ontology = ontology
        self.outcome_codes: Set[int] = map_omop_concept_codes_to_femr_codes(
            ontology,
            self.original_omop_concept_codes,
            is_ontology_expansion=True,
        )

    def label(self, patient: Patient) -> List[Label]:
        labels: List[Label] = []
        for e in patient.events:
            if e.code in self.outcome_codes:
                # This is an outcome event
                if e.value is not None:
                    try:
                        # `e.unit` is string of form "mg/dL", "ounces", etc.
                        label: int = self.label_to_int(self.value_to_label(str(e.value), str(e.unit)))
                        prediction_time: datetime.datetime = e.start - datetime.timedelta(milliseconds=1)
                        labels.append(Label(prediction_time, label))
                    except Exception as exception:
                        print(
                            f"Warning: Error parsing value='{e.value}' with unit='{e.unit}'"
                            f" for code='{e.code}' @ {e.start} for patient_id='{patient.patient_id}'"
                            f" | Exception: {exception}"
                        )
        return labels

    def get_labeler_type(self) -> LabelType:
        return "categorical"

    def label_to_int(self, label: str) -> int:
        if label == "normal":
            return 0
        elif label == "mild":
            return 1
        elif label == "moderate":
            return 2
        elif label == "severe":
            return 3
        raise ValueError(f"Invalid label without a corresponding int: {label}")

    @abstractmethod
    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        """Convert `value` to a string label: "mild", "moderate", "severe", or "normal".
        NOTE: Some units have the form 'mg/dL (See scan or EMR data for detail)', so you
        need to use `.startswith()` to check for the unit you want.
        """
        return "normal"


class ThrombocytopeniaInstantLabValueLabeler(InstantLabValueLabeler):
    """lab-based definition for thrombocytopenia based on platelet count (10^9/L).
    Thresholds: mild (<150), moderate(<100), severe(<50), and reference range."""

    original_omop_concept_codes = [
        "LOINC/LP393218-5",
        "LOINC/LG32892-8",
        "LOINC/777-3",
    ]

    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        if raw_value.lower() in ["normal", "adequate"]:
            return "normal"
        value = float(raw_value)
        if value < 50:
            return "severe"
        elif value < 100:
            return "moderate"
        elif value < 150:
            return "mild"
        return "normal"


class HyperkalemiaInstantLabValueLabeler(InstantLabValueLabeler):
    """lab-based definition for hyperkalemia using blood potassium concentration (mmol/L).
    Thresholds: mild(>5.5),moderate(>6),severe(>7), and abnormal range."""

    original_omop_concept_codes = [
        "LOINC/LG7931-1",
        "LOINC/LP386618-5",
        "LOINC/LG10990-6",
        "LOINC/6298-4",
        "LOINC/2823-3",
    ]

    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        if raw_value.lower() in ["normal", "adequate"]:
            return "normal"
        value = float(raw_value)
        if unit is not None:
            unit = unit.lower()
            if unit.startswith("mmol/l"):
                # mmol/L
                # Original OMOP concept ID: 8753
                value = value
            elif unit.startswith("meq/l"):
                # mEq/L (1-to-1 -> mmol/L)
                # Original OMOP concept ID: 9557
                value = value
            elif unit.startswith("mg/dl"):
                # mg / dL (divide by 18 to get mmol/L)
                # Original OMOP concept ID: 8840
                value = value / 18.0
            else:
                raise ValueError(f"Unknown unit: {unit}")
        else:
            raise ValueError(f"Unknown unit: {unit}")
        if value > 7:
            return "severe"
        elif value > 6.0:
            return "moderate"
        elif value > 5.5:
            return "mild"
        return "normal"


class HypoglycemiaInstantLabValueLabeler(InstantLabValueLabeler):
    """lab-based definition for hypoglycemia using blood glucose concentration (mmol/L).
    Thresholds: mild(<3), moderate(<3.5), severe(<=3.9), and abnormal range."""

    original_omop_concept_codes = [
        "SNOMED/33747003",
        "LOINC/LP416145-3",
        "LOINC/14749-6",
        "LOINC/15074-8",
    ]

    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        if raw_value.lower() in ["normal", "adequate"]:
            return "normal"
        value = float(raw_value)
        if unit is not None:
            unit = unit.lower()
            if unit.startswith("mg/dl"):
                # mg / dL
                # Original OMOP concept ID: 8840, 9028
                value = value / 18
            elif unit.startswith("mmol/l"):
                # mmol / L (x 18 to get mg/dl)
                # Original OMOP concept ID: 8753
                value = value
            else:
                raise ValueError(f"Unknown unit: {unit}")
        else:
            raise ValueError(f"Unknown unit: {unit}")
        if value < 3:
            return "severe"
        elif value < 3.5:
            return "moderate"
        elif value <= 3.9:
            return "mild"
        return "normal"


class HyponatremiaInstantLabValueLabeler(InstantLabValueLabeler):
    """lab-based definition for hyponatremia based on blood sodium concentration (mmol/L).
    Thresholds: mild (<=135),moderate(<130),severe(<125), and abnormal range."""

    original_omop_concept_codes = ["LOINC/LG11363-5", "LOINC/2951-2", "LOINC/2947-0"]

    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        if raw_value.lower() in ["normal", "adequate"]:
            return "normal"
        value = float(raw_value)
        if value < 125:
            return "severe"
        elif value < 130:
            return "moderate"
        elif value <= 135:
            return "mild"
        return "normal"


class AnemiaInstantLabValueLabeler(InstantLabValueLabeler):
    """lab-based definition for anemia based on hemoglobin levels (g/L).
    Thresholds: mild(<120),moderate(<110),severe(<70), and reference range"""

    original_omop_concept_codes = [
        "LOINC/LP392452-1",
    ]

    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        if raw_value.lower() in ["normal", "adequate"]:
            return "normal"
        value = float(raw_value)
        if unit is not None:
            unit = unit.lower()
            if unit.startswith("g/dl"):
                # g / dL
                # Original OMOP concept ID: 8713
                # NOTE: This weird *10 / 100 is how Lawrence did it
                value = value * 10
            elif unit.startswith("mg/dl"):
                # mg / dL (divide by 1000 to get g/dL)
                # Original OMOP concept ID: 8840
                # NOTE: This weird *10 / 100 is how Lawrence did it
                value = value / 100
            elif unit.startswith("g/l"):
                value = value
            else:
                raise ValueError(f"Unknown unit: {unit}")
        else:
            raise ValueError(f"Unknown unit: {unit}")
        if value < 70:
            return "severe"
        elif value < 110:
            return "moderate"
        elif value < 120:
            return "mild"
        return "normal"


##########################################################
##########################################################
# Labelers based on codes
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


class InpatientLabValueLabeler(WithinInpatientVisitLabeler):
    """Apply a label based on 1+ ocurrences of an outcome defined by a lab value during an ICU visit

    Hourly binary prediction task on whether the patient dies in the next 24 hours.
    Make prediction every 60 minutes after ICU admission, starting at hour 4.

    Excludes:
        - ICU admissions with no length-of-stay (i.e. `event.end is None` )
        - ICU admissions < 4 hours
        - ICU admissions with no events
    """

    """Apply a label based on 1+ occurrence(s) of an outcome defined by a lab value during each INPATIENT visit
        where that lab test has a result recorded (thus, we're conditioning on ordering the lab test).

    Prediction Time: 4 hours into
    Time Horizon: Within an INPATIENT visit
    Label: TRUE if any lab value comes back with severity level == `self.severity` during the visit, 0 otherwise.
    """

    # parent OMOP concept codes, from which all the outcomes are derived (as children in our ontology)
    original_omop_concept_codes: List[str] = []

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        severity: str,
        visit_start_adjust_func: Callable = lambda x: x,
        visit_end_adjust_func: Callable = lambda x: x,
    ):
        """Matches lab test on any Piton code that maps to one of the `omop_concept_ids`.
        Specify `severity` as one of "mild", "moderate", "severe", or "normal" to determine binary label."""
        self.severity: str = severity
        self.outcome_codes: Set[int] = map_omop_concept_codes_to_femr_codes(
            ontology,
            self.original_omop_concept_codes,
            is_ontology_expansion=True,
        )
        super().__init__(
            ontology=ontology,
            visit_start_adjust_func=visit_start_adjust_func,
            visit_end_adjust_func=visit_end_adjust_func,
        )

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        times: List[datetime.datetime] = []
        for e in patient.events:
            if e.code in self.outcome_codes:
                # This is an outcome event
                if e.value is not None:
                    label: Optional[str] = None
                    try:
                        # `e.unit` is string of form "mg/dL", "ounces", etc.
                        label = self.value_to_label(str(e.value), str(e.unit))
                        if label == self.severity:
                            times.append(e.start)
                    except Exception as exception:
                        print(
                            f"Warning: Error parsing value='{e.value}' with unit='{e.unit}'"
                            f" for code='{e.code}' @ {e.start} for patient_id='{patient.patient_id}'"
                            f" | Exception: {exception}"
                        )
        return times

    def get_visit_events(self, patient: Patient) -> List[Event]:
        """Only keep inpatient visits where a lab test result is returned."""
        # Get list of all times when lab test result was returned
        valid_times: List[datetime.datetime] = []
        for e in patient.events:
            if e.code in self.outcome_codes:
                # This is an outcome event
                if e.value is not None:
                    try:
                        # A valid lab value was returned
                        _ = self.value_to_label(str(e.value), str(e.unit))
                        # record this visit as valid
                        valid_times.append(e.start)
                    except Exception:
                        # ignore this visit b/c a valid lab value was not returned
                        pass
        if len(valid_times) == 0:
            # Note: this is a necessary check, otherwise the `while` loop below will trip up on its first iteration
            return []
        # Filter inpatient events to only those where a valid lab test result was returned
        visits: List[Event] = get_inpatient_admission_events(patient, self.ontology)
        valid_visits: List[Event] = []
        curr_valid_time_idx: int = 0
        for e in visits:
            while valid_times[curr_valid_time_idx] < e.start:
                # Increment valid_times until we get one that occurs after this visit starts
                curr_valid_time_idx += 1
                if curr_valid_time_idx >= len(valid_times):
                    # We've passed through all valid_times, and none occur after this visit starts,
                    # so we can safely break and return the valid visits we've already found
                    return valid_visits
            if e.start <= valid_times[curr_valid_time_idx] <= e.end:
                # If this valid_times falls within this visit, record that this visit is valid
                valid_visits.append(e)
        return valid_visits

    def get_labeler_type(self) -> LabelType:
        return "boolean"

    @abstractmethod
    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        """Convert `value` to a string label: "mild", "moderate", "severe", or "normal".
        NOTE: Some units have the form 'mg/dL (See scan or EMR data for detail)', so you
        need to use `.startswith()` to check for the unit you want.
        """
        return "normal"
