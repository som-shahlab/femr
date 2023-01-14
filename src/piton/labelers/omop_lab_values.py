"""Labeling functions for OMOP data based on lab values."""
from __future__ import annotations

from abc import abstractmethod
import datetime
from typing import List, Set, Tuple, Deque, Dict

from .. import Event, Patient
from ..extension import datasets as extension_datasets
from .omop import (
    CodeLF
)
from .core import (
    FixedTimeHorizonEventLF,
    Label,
    LabelingFunction,
    LabelType,
    TimeHorizon,
)

##########################################################
# Labeling functions based on Lab Values.
#
# The difference between these LFs and the ones in `omop.py` 
# is that these LFs are based on lab values, not coded 
# diagnoses. Thus, they may catch more cases of a given
# condition due to under-coding, but they are also more 
# likely to be noisy.
##########################################################

class OutcomeFromLabValue(FixedTimeHorizonEventLF):
    """Apply a label based on 1+ occurrence(s) of an outcome defined by a lab value over a fixed time horizon."""
    original_omop_concept_ids: List[int] = []
    omop_concept_ids: List[int] = []
    
    def __init__(
        self, ontology: extension_datasets.Ontology, time_horizon: TimeHorizon, severity: str,
    ):
        """Matches lab test on any Piton code that maps to one of the `omop_concept_ids`.
        Specify `severity` as one of "mild", "moderate", "severe", or "normal" to determine binary label."""
        self.time_horizon: TimeHorizon = time_horizon
        self.severity: str = severity
        self.codes = []
        for omop_concept_id in self.omop_concept_ids:
            piton_code: int = ontology.get_code_from_concept_id(omop_concept_id)
            self.codes.append(piton_code)

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return the start times of this patient's events which correspond to a lab test result 
            at severity level `self.severity`."""
        times: List[datetime.datetime] = []
        for event in patient.events:
            if event.value is not None:
                value: float = self.normalize_value_with_units(float(event.value), event.unit_concept_id)
                if (event.code in self.codes 
                    and self.value_to_label(value) == self.severity
                ):
                    times.append(event.start)
        return times

    @abstractmethod
    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return each event's start time as the time to make a prediction.
            Default to all events whose `code` is in `self.prediction_codes`."""
        return []

    @abstractmethod
    def value_to_label(self, value: float) -> str:
        """Convert `value` to a string label: "mild", "moderate", "severe", or "normal"."""
        return 'normal'

    @abstractmethod
    def normalize_value_with_units(self, value: float, unit_concept_id: int) -> float:
        """Convert `value` to a float in the same units as the thresholds in `self.value_to_label`."""
        return value

class ThrombocytopeniaLabValue(OutcomeFromLabValue):
    """lab-based definition for thrombocytopenia based on platelet count (10^9/L). Thresholds: mild (<150), moderate(<100), severe(<50), and reference range."""
    original_omop_concept_ids = [37037425,40654106] # parent OMOP concept IDs, from which `omop_concept_ids`` are derived
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

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        return []

    def value_to_label(self, value: float) -> str:
        if value < 150:
            return 'mild'
        elif value < 100:
            return 'moderate'
        elif value < 50:
            return 'severe'
        return 'normal'

    def normalize_value_with_units(self, value: float, unit_concept_id: int) -> float:
        return value

class HyperkalemiaLabValue(OutcomeFromLabValue):
    """lab-based definition for hyperkalemia using blood potassium concentration (mmol/L). Thresholds: mild(>5.5),moderate(>6),severe(>7), and abnormal range."""
    original_omop_concept_ids = [40653595, 37074594, 40653596,] # parent OMOP concept IDs, from which `omop_concept_ids`` are derived
    omop_concept_ids = [
        40653596, 40653595, 37074594, 3024920, 46235078,
        3041354, 3023103, 3015066, 3031219, 3024380,
        3043409, 3005456, 3039651, 21490733,3040893,
    ]

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        return []

    def value_to_label(self, value: float) -> str:
        if value > 5.5:
            return 'mild'
        elif value > 6.0:
            return 'moderate'
        elif value > 7:
            return 'severe'
        return 'normal'

    def normalize_value_with_units(self, value: float, unit_concept_id: int) -> float:
        if unit_concept_id == 8753:
            # mmol/L
            return value
        elif unit_concept_id == 9557:
            # mEq/L (1-to-1 -> mmol/L)
            return value
        elif unit_concept_id == 8840:
            # mg / dL (divide by 18 to get mmol/L)
            return value / 18
        raise ValueError(f"Unknown unit_concept_id: {unit_concept_id}")

class HypoglycemiaLabValue(OutcomeFromLabValue):
    """lab-based definition for hypoglycemia using blood glucose concentration (mmol/L). Thresholds: mild(<3), moderate(<3.5), severe(<=3.9), and abnormal range."""
    original_omop_concept_ids = [4144235, 1002597] # parent OMOP IDs, `from`` which omop_
    concept_ids = [
        3009397,
        3040694,
        3016567,
        3048282,
    ]

    def value_to_label(self, value: float) -> str:
        if value <= 3.9:
            return 'mild'
        elif value < 3.5:
            return 'moderate'
        elif value < 3:
            return 'severe'
        return 'normal'

    def normalize_value_with_units(self, value: float, unit_concept_id: int) -> float:
        if unit_concept_id == 8840:
            # mg / dL
            return value / 18
        elif unit_concept_id == 9028:
            # mg / dL calculated
            return value / 18
        elif unit_concept_id == 8753:
            # mmol / L (x 18 to get mg/dl)
            return value  
        raise ValueError(f"Unknown unit_concept_id: {unit_concept_id}")
    
class HyponatremiaLabValue(OutcomeFromLabValue):
    """lab-based definition for hyponatremia based on blood sodium concentration (mmol/L). Thresholds: mild (<=135),moderate(<130),severe(<125), and abnormal range."""
    original_omop_concept_ids = [40653762] # parent OMOP IDs, `from`` which omop_
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
        if value <= 135:
            return 'mild'
        elif value < 130:
            return 'moderate'
        elif value < 125:
            return 'severe'    
        return 'normal'

    def normalize_value_with_units(self, value: float, unit_concept_id: int) -> float:
        return value
    
class AnemiaLabValue(OutcomeFromLabValue):
    """lab-based definition for anemia based on hemoglobin levels (g/L). Thresholds: mild(<120),moderate(<110),severe(<70), and reference range"""
    original_omop_concept_ids = [37072252] # parent OMOP IDs, `from`` which omop_
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
        if value < 120:
            return 'mild'
        elif value < 110:
            return 'moderate'
        elif value < 70:
            return 'severe'
        return 'normal'

    def normalize_value_with_units(self, value: float, unit_concept_id: int) -> float:
        if unit_concept_id == 8713:
            # g / dL
            return value * 10 # NOTE: This weird *10 / 100 is how Lawrence did it
        elif unit_concept_id == 8840:
            # mg / dL (divide by 1000 to get g/dL)
            return value / 100 # NOTE: This weird *10 / 100 is how Lawrence did it
        raise ValueError(f"Unknown unit_concept_id: {unit_concept_id}")

class ThrombocytopeniaLabValue(OutcomeFromLabValue):
    """lab-based definition for thrombocytopenia based on platelet count (10^9/L). Thresholds: mild (<150), moderate(<100), severe(<50), and reference range."""
    original_omop_concept_ids = [37037425,40654106] # parent OMOP IDs, `from`` which omop_
    concept_ids = [
        40654106,
        37037425,
        3031586,
        3033641,
        3007461,
        3010834,
        3024929,
        21492791,
    ]
    def value_to_label(self, value: float) -> str:
        if value < 150:
            return 'mild'
        elif value < 100:
            return 'moderate'
        elif value < 50:
            return 'severe'
        return 'normal'

    def normalize_value_with_units(self, value: float, unit_concept_id: int) -> float:
        return value

class NeutropeniaLabValue(OutcomeFromLabValue):
    """lab-based definition for neutropenia based on neutrophils count (thousands/uL). Thresholds: mild(<1.5), moderate(<1), severe(<0.5)"""
    
    wbc_concept_ids = [
        3000905, 
        4298431, 
        3010813,
    ]
    band_concept_ids = [
        3035839, 
        3018199,
    ]
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
    
    def get_label_wrapper(self, neutrophil: float, bands: float, wbc: float):
        if neutrophil is not None or bands is not None:
            return self.get_label((neutrophil or 0) + (bands or 0))
        elif wbc is not None:
            return self.get_label(wbc)
        assert False, 'No valid values found'

    def value_to_label(self, value: float) -> str:
        if value < 1.5:
            return 'mild'
        elif value < 1:
            return 'moderate'
        elif value < 0.5:
            return 'severe'
        return 'normal'
    
    def wbc_normalize_value_with_units(self, value: float, unit_concept_id: int) -> float:
        if unit_concept_id == 8848:
            # 1000/uL
            return value
        elif unit_concept_id == 8961:
            # 1000/mm^3, equivalent to 8848
            return value
        elif unit_concept_id == 8647:
            # /uL - divide by 1000 to convert to 1000/uL
            return value / 1000
    
    def band_convert(self, unit_concept_id: int, value: float, measurement_concept_id: int, wbc: float):
        if measurement_concept_id == 3035839 and value <= 100:
            # band form /100 leukocytes (%)
            return value / 100 * wbc
        elif measurement_concept_id == 3018199 and unit_concept_id == 8784:
            # band form neutrophils in blood (count) 
            return value / 1000

    def neutrophil_convert(self, unit_concept_id: int, value: float, measurement_concept_id: int, wbc: float):
        if unit_concept_id == 8554 and value <= 100:
            # neutrophils /100 leukocytes
            return value / 100 * wbc
        elif unit_concept_id == 8554 and value > 100:
            return None
        elif unit_concept_id == 8784:
            return value / 1000
                        
class AcuteKidneyInjuryLabValue(OutcomeFromLabValue):
    # TODO - very complicated
    """lab-based definition for acute kidney injury based on blood creatinine levels (umol/L) according to KDIGO (stages 1,2, and 3), and abnormal range."""
    concept_ids = [
        43055236,
        3020564,
        3035090,
        3022243,
        3019397,
        3040495,
        3016723,
    ]