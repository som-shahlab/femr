"""Labeling functions for OMOP data based on lab values."""
from __future__ import annotations

import datetime
from abc import abstractmethod
from typing import Any, List, Optional, Set, Dict

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
    _get_all_children,
    move_datetime_to_end_of_day,
    map_omop_concept_codes_to_piton_codes,
    group_inpatient_events_by_visit_id
)

##########################################################
##########################################################
# Labelers based on Lab Values.
#
# The difference between these Labelers and the ones in `omop.py`
# is that these Labelers are based on lab values, not coded
# diagnoses. Thus, they may catch more cases of a given
# condition due to under-coding, but they are also more
# likely to be noisy.
##########################################################
##########################################################


class OMOPConceptOutcomeFromLabValueLabeler(Labeler):
    """Apply a label based on 1+ occurrence(s) of an outcome defined by a lab value during each INPATIENT visit 
        where that lab test has a result recorded (thus, we're conditioning on ordering the lab test).
    
    Prediction Time: At admission for every INPATIENT visit where a lab test result is returned,
                    where the admission time is manually adjusted to 11:59:59pm
    Time Horizon: Within an INPATIENT visit
    Label: TRUE if any lab value comes back with severity level == `self.severity` during the visit, 0 otherwise.
    """

    # parent OMOP concept codes, from which all the outcome
    # are derived (as children from our ontology)
    original_omop_concept_codes: List[str] = []

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        severity: str,
    ):
        """Matches lab test on any Piton code that maps to one of the `omop_concept_ids`.
        Specify `severity` as one of "mild", "moderate", "severe", or "normal" to determine binary label."""
        self.ontology: extension_datasets.Ontology = ontology
        self.severity: str = severity
        self.outcome_codes: Set[int] = map_omop_concept_codes_to_piton_codes(ontology, self.original_omop_concept_codes, is_ontology_expansion=True)

    def label(self, patient: Patient) -> List[Label]:
        """Label all INPATIENT visits in which the patient experiences an outcome
        in `self.outcome_codes` with whether that outcome was severe or not."""
        events_by_visit_id: Dict[int, List[Event]] = group_inpatient_events_by_visit_id(patient, self.ontology)
        labels: List[Label] = []
        # Loop through all visits in patient, check if outcome occurs during visit
        # and if so, mark that it occurred in `labels`.
        for visit_id, events in events_by_visit_id.items():
            # if `is_outcome_occurs` is None, then no outcome events were found (so we ignore this visit)
            # if `is_outcome_occurs` is True, then an outcome event was found and it was `self.severity`
            # if `is_outcome_occurs` is False, then an outcome event was found, but it was not `self.severity`
            is_outcome_occurs: Optional[bool] = None
            visit_event: Optional[Event] = None
            for e in events:
                if e.omop_table == 'visit_detail':
                    # Track the (start, end) of this visit_id from its `visit_detail` event
                    visit_event = e
                if e.code in self.outcome_codes:
                    # This is an outcome event
                    if e.value is not None:
                        label: Optional[str] = None
                        try:
                            # `e.unit` is string of form "mg/dL", "ounces", etc.
                            label = self.value_to_label(str(e.value), str(e.unit))
                        except Exception as exception:
                            print(
                                f"Warning: Error parsing value='{e.value}' with unit='{e.unit}'"
                                f" for code='{e.code}' @ {e.start} for patient_id='{patient.patient_id}'"
                                f" | Exception: {exception}"
                            )
                        is_outcome_occurs = label == self.severity
            if visit_event is None:
                raise RuntimeError(f"Every `visit_id` must have an event from the `visit_detail` OMOP table associated with it, but no `visit_detail` events were found with `visit_id={visit_id}`.")
            if is_outcome_occurs is not None:
                # If lab test result was reported, then record whether the outcome was TRUE or FALSE.
                # Otherwise, if no lab test occurred in this visit, we ignore it.
                prediction_time: datetime.datetime = move_datetime_to_end_of_day(visit_event.start)
                labels.append(Label(prediction_time, is_outcome_occurs))
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"
    
    @abstractmethod
    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        """Convert `value` to a string label: "mild", "moderate", "severe", or "normal".
        NOTE: Some units have the form 'mg/dL (See scan or EMR data for detail)', so you
        need to use `.startswith()` to check for the unit you want.
        """
        return "normal"


class ThrombocytopeniaLabValueLabeler(OMOPConceptOutcomeFromLabValueLabeler):
    """lab-based definition for thrombocytopenia based on platelet count (10^9/L).
    Thresholds: mild (<150), moderate(<100), severe(<50), and reference range."""

    original_omop_concept_codes = [
        "LOINC/LP393218-5",
        "LOINC/LG32892-8",
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


class HyperkalemiaLabValueLabeler(OMOPConceptOutcomeFromLabValueLabeler):
    """lab-based definition for hyperkalemia using blood potassium concentration (mmol/L).
    Thresholds: mild(>5.5),moderate(>6),severe(>7), and abnormal range."""

    original_omop_concept_codes = [
        "LOINC/LG7931-1",
        "LOINC/LP386618-5",
        "LOINC/LG10990-6",
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


class HypoglycemiaLabValueLabeler(OMOPConceptOutcomeFromLabValueLabeler):
    """lab-based definition for hypoglycemia using blood glucose concentration (mmol/L).
    Thresholds: mild(<3), moderate(<3.5), severe(<=3.9), and abnormal range."""

    original_omop_concept_codes = [
        "SNOMED/33747003",
        "LOINC/LP416145-3",
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


class HyponatremiaLabValueLabeler(OMOPConceptOutcomeFromLabValueLabeler):
    """lab-based definition for hyponatremia based on blood sodium concentration (mmol/L).
    Thresholds: mild (<=135),moderate(<130),severe(<125), and abnormal range."""

    original_omop_concept_codes = [
        "LOINC/LG11363-5",
    ]

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


class AnemiaLabValueLabeler(OMOPConceptOutcomeFromLabValueLabeler):
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


class NeutropeniaLabValueLabeler(OMOPConceptOutcomeFromLabValueLabeler):
    """lab-based definition for neutropenia based on neutrophils count (thousands/uL).
    Thresholds: mild(<1.5), moderate(<1), severe(<0.5)"""

    # TODO

    original_wbc_concept_ids = [
        3000905,
        4298431,
        3010813,
    ]
    wbc_concept_ids = original_wbc_concept_ids
    original_band_concept_ids = [
        3035839,
        3018199,
    ]
    band_concept_ids = original_band_concept_ids
    original_neutrophil_concept_ids = [37045722, 37049637]
    neutrophil_concept_ids = [
        37045722,
        37049637,
        3017501,
        3018010,
        3027368,
        3008342,
        3013650,
        3017732,
    ]


class AcuteKidneyInjuryLabValueLabeler(OMOPConceptOutcomeFromLabValueLabeler):
    # TODO - very complicated
    """lab-based definition for acute kidney injury based on blood creatinine levels (umol/L)
    according to KDIGO (stages 1,2, and 3), and abnormal range."""
    original_expanded_omop_concept_ids = [
        43055236,
        3020564,
        3035090,
        3022243,
        3019397,
        3040495,
        3016723,
    ]


##########################################################
##########################################################
# Other lab value related labelers
##########################################################
##########################################################


class CeliacTestLabeler(Labeler):
    # TODO - check
    """
    The Celiac test labeler predicts whether or not a celiac test will be positive or negative.
    The prediction time is 24 hours before the lab results come in.
    Note: This labeler excludes patients who either already had a celiac test or were previously diagnosed.
    """

    def __init__(
        self, ontology: extension_datasets.Ontology, time_horizon: TimeHorizon
    ):
        dictionary = ontology.get_dictionary()
        self.lab_codes = _get_all_children(
            ontology, dictionary.index("LNC/31017-7")
        )
        self.celiac_codes = _get_all_children(
            ontology, dictionary.index("ICD9CM/579.0")
        ) | _get_all_children(ontology, dictionary.index("ICD10CM/K90.0"))

        self.pos_value = "Positive"
        self.neg_value = "Negative"

    def label(self, patient: Patient) -> List[Label]:
        if len(patient.events) == 0:
            return []

        for event in patient.events:
            if event.code in self.celiac_codes:
                # This patient already has Celiacs
                return []
            if event.code in self.lab_codes and event.value in [
                self.pos_value,
                self.neg_value,
            ]:
                # This patient got a Celiac lab test result
                # We'll return the Label 24 hours prior
                return [
                    Label(
                        event.start - datetime.timedelta(hours=24),
                        event.value == self.pos_value,
                    )
                ]
        return []

    def get_labeler_type(self) -> LabelType:
        return "boolean"