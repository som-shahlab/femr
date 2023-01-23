"""Labeling functions for OMOP data based on lab values."""
from __future__ import annotations

import datetime
from abc import abstractmethod
from typing import List, Optional, Set

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
    get_inpatient_admission_events,
    map_omop_concept_ids_to_piton_codes,
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


class OMOPConceptOutcomeFromLabValueLabeler(TimeHorizonEventLabeler):
    """Apply a label based on 1+ occurrence(s) of an outcome defined by a lab value over a time horizon."""

    # parent OMOP concept IDs, from which `omop_concept_ids` are derived
    original_omop_concept_ids: List[int] = []
    # names for OMOP concept IDs
    original_omop_concept_codes: List[str] = []
    # parent OMOP concept IDs + all their children
    omop_concept_ids: List[int] = []

    def __init__(
        self,
        ontology: extension_datasets.Ontology,
        time_horizon: TimeHorizon,
        severity: str,
    ):
        """Matches lab test on any Piton code that maps to one of the `omop_concept_ids`.
        Specify `severity` as one of "mild", "moderate", "severe", or "normal" to determine binary label."""
        self.ontology: extension_datasets.Ontology = ontology
        self.time_horizon: TimeHorizon = time_horizon
        self.severity: str = severity
        self.outcome_codes: List[int] = []

        if hasattr(self, "original_omop_concept_codes"):
            # We need to traverse through the ontology ourselves using
            # OMOP Concept Codes (e.g. "LOINC/123") instead of pre-specified
            # OMOP Concept IDs (e.g. 3939430) to get all revelant children
            for omop_concept_code in self.original_omop_concept_codes:
                piton_code = ontology.get_dictionary().index(omop_concept_code)
                all_children: Set[int] = _get_all_children(ontology, piton_code)
                self.outcome_codes += [code for code in all_children]
        else:
            # This Labeler explicitly specifies the list of OMOP Concept IDs that
            # corresopnd to this label, so use those directly.
            # This relies on the `ontology` class having a `get_code_from_concept_id()`
            # method implemented.
            self.outcome_codes = map_omop_concept_ids_to_piton_codes(
                ontology, self.omop_concept_ids
            )
        self.outcome_codes = list(set(self.outcome_codes))

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return the start times of this patient's events which correspond to a lab test result
        at severity level `self.severity`."""
        times: List[datetime.datetime] = []
        for event in patient.events:
            if event.code in self.outcome_codes:
                if event.value is not None:
                    # `unit` is string of form "mg/dL", "ounces", etc.
                    unit: Optional[str] = event.unit
                    value: float = self.normalize_value_with_units(
                        float(event.value), unit
                    )
                    if self.value_to_label(value) == self.severity:
                        times.append(event.start)
        return times

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Default to making prediction at admission time"""
        admission_events: List[Event] = get_inpatient_admission_events(
            patient, self.ontology
        )
        return [x.start for x in admission_events]

    @abstractmethod
    def value_to_label(self, value: float) -> str:
        """Convert `value` to a string label: "mild", "moderate", "severe", or "normal"."""
        return "normal"

    @abstractmethod
    def normalize_value_with_units(
        self, value: float, unit: Optional[str]
    ) -> float:
        """Convert `value` to a float in the same units as the thresholds in `self.value_to_label`.

        NOTE: Some units have the form 'mg/dL (See scan or EMR data for detail)', so you
        need to use `.startswith()` to check for the unit you want."""
        return value


class ThrombocytopeniaLabValueLabeler(OMOPConceptOutcomeFromLabValueLabeler):
    """lab-based definition for thrombocytopenia based on platelet count (10^9/L).
    Thresholds: mild (<150), moderate(<100), severe(<50), and reference range."""

    original_omop_concept_ids = [
        37037425,
        40654106,
    ]
    original_omop_concept_codes = [
        "LOINC/LP393218-5",
        "LOINC/LG32892-8",
    ]
    omop_concept_ids = [
        37037425,
        40654106,
        3031586,
        3033641,
        3007461,
        3010834,
        3024929,
        21492791,
    ]

    def value_to_label(self, value: float) -> str:
        if value < 50:
            return "severe"
        elif value < 100:
            return "moderate"
        elif value < 150:
            return "mild"
        return "normal"

    def normalize_value_with_units(
        self, value: float, unit: Optional[str]
    ) -> float:
        return value


class HyperkalemiaLabValueLabeler(OMOPConceptOutcomeFromLabValueLabeler):
    """lab-based definition for hyperkalemia using blood potassium concentration (mmol/L).
    Thresholds: mild(>5.5),moderate(>6),severe(>7), and abnormal range."""

    original_omop_concept_ids = [
        40653595,
        37074594,
        40653596,
    ]
    original_omop_concept_codes = [
        "LOINC/LG7931-1",
        "LOINC/LP386618-5",
        "LOINC/40653596",
    ]
    omop_concept_ids = [
        40653596,
        40653595,
        37074594,
        3024920,
        46235078,
        3041354,
        3023103,
        3015066,
        3031219,
        3024380,
        3043409,
        3005456,
        3039651,
        21490733,
        3040893,
    ]

    def value_to_label(self, value: float) -> str:
        if value > 7:
            return "severe"
        elif value > 6.0:
            return "moderate"
        elif value > 5.5:
            return "mild"
        return "normal"

    def normalize_value_with_units(
        self, value: float, unit: Optional[str]
    ) -> float:
        if unit is not None:
            if unit.startswith("mmol/L"):
                # mmol/L
                # Original OMOP concept ID: 8753
                return value
            elif unit.startswith("mEq/L"):
                # mEq/L (1-to-1 -> mmol/L)
                # Original OMOP concept ID: 9557
                return value
            elif unit.startswith("mg/dL"):
                # mg / dL (divide by 18 to get mmol/L)
                # Original OMOP concept ID: 8840
                return value / 18
        raise ValueError(f"Unknown unit: {unit}")


class HypoglycemiaLabValueLabeler(OMOPConceptOutcomeFromLabValueLabeler):
    """lab-based definition for hypoglycemia using blood glucose concentration (mmol/L).
    Thresholds: mild(<3), moderate(<3.5), severe(<=3.9), and abnormal range."""

    original_omop_concept_ids = [
        4144235,
        1002597,
    ]
    original_omop_concept_codes = [
        "SNOMED/33747003",
        "LOINC/LP416145-3",
    ]
    concept_ids = [
        3009397,
        3040694,
        3016567,
        3048282,
    ]

    def value_to_label(self, value: float) -> str:
        if value < 3:
            return "severe"
        elif value < 3.5:
            return "moderate"
        elif value <= 3.9:
            return "mild"
        return "normal"

    def normalize_value_with_units(
        self, value: float, unit: Optional[str]
    ) -> float:
        if unit is not None:
            if unit.startswith("mg/dL"):
                # mg / dL
                # Original OMOP concept ID: 8840, 9028
                return value / 18
            elif unit.startswith("mmol/L"):
                # mmol / L (x 18 to get mg/dl)
                # Original OMOP concept ID: 8753
                return value
        raise ValueError(f"Unknown unit: {unit}")


class HyponatremiaLabValueLabeler(OMOPConceptOutcomeFromLabValueLabeler):
    """lab-based definition for hyponatremia based on blood sodium concentration (mmol/L).
    Thresholds: mild (<=135),moderate(<130),severe(<125), and abnormal range."""

    original_omop_concept_ids = [40653762]
    original_omop_concept_codes = [
        "LOINC/LG11363-5",
    ]
    concept_ids = [
        40653762,
        3038702,
        3031579,
        3000285,
        3041473,
        3043706,
        46235784,
        3019550,
    ]

    def value_to_label(self, value: float) -> str:
        if value < 125:
            return "severe"
        elif value < 130:
            return "moderate"
        elif value <= 135:
            return "mild"
        return "normal"

    def normalize_value_with_units(
        self, value: float, unit: Optional[str]
    ) -> float:
        return value


class AnemiaLabValueLabeler(OMOPConceptOutcomeFromLabValueLabeler):
    """lab-based definition for anemia based on hemoglobin levels (g/L).
    Thresholds: mild(<120),moderate(<110),severe(<70), and reference range"""

    original_omop_concept_ids = [37072252]
    original_omop_concept_codes = [
        "LOINC/LP392452-1",
    ]
    concept_ids = [
        37072252,
        3048275,
        40758903,
        3005872,
        3027484,
        3000963,
        40762351,
    ]

    def value_to_label(self, value: float) -> str:
        if value < 70:
            return "severe"
        elif value < 110:
            return "moderate"
        elif value < 120:
            return "mild"
        return "normal"

    def normalize_value_with_units(
        self, value: float, unit: Optional[str]
    ) -> float:
        if unit is not None:
            if unit.startswith("g/dL"):
                # g / dL
                # Original OMOP concept ID: 8713
                # NOTE: This weird *10 / 100 is how Lawrence did it
                return value * 10
            elif unit.startswith("mg/dL"):
                # mg / dL (divide by 1000 to get g/dL)
                # Original OMOP concept ID: 8840
                # NOTE: This weird *10 / 100 is how Lawrence did it
                return value / 100
        raise ValueError(f"Unknown unit: {unit}")


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
    concept_ids = [
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
