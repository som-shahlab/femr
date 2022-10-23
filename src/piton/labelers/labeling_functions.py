from __future__ import annotations

import collections
import datetime
import json
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    DefaultDict,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    TextIO,
    Tuple,
    Union,
)

from .core import TimeHorizon, LabelingFunction, FixedTimeHorizonEventLF
from .. import Event, Patient
from ..extension import datasets as extension_datasets

import numpy as np

class CodeLF(FixedTimeHorizonEventLF):
    """
        Applies a label based on a single code's occurrence over a fixed time horizon
    """

    def __init__(self, code: int, time_horizon: TimeHorizon):
        """Label the code whose index in your Ontology is equal to `code`
        """
        self.code = code
        self.time_horizon = time_horizon
    
    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Returns a list of datetimes corresponding to the start time of the Events
            in this patient's timeline which have the exact same `code` as `self.code`
        """
        times: List[datetime.datetime] = []
        for event in patient.events:
            if event.code == self.code:
                times.append(event.start)
        return times

class MortalityLF(CodeLF):
    """
        The mortality task is defined as predicting whether or not a
        patient will die within the next `time_horizon` time.
    """

    def __init__(self, 
                 ontology: extension_datasets.Ontology, 
                 time_horizon: TimeHorizon):
        CODE_DEATH_PREFIX = "Death Type/"

        death_codes: Set[Tuple[str, int]] = set()
        for code, code_str in enumerate(ontology.get_dictionary()):
            if code_str.startswith(CODE_DEATH_PREFIX):
                death_codes.add((code_str, code))

        if len(death_codes) != 1:
            raise ValueError(
                f"Could not find exactly one death code -- instead found {len(death_codes)} codes: {str(death_codes)}"
            )
        else:
            death_code: int = list(death_codes)[0][1]
            super().__init__(code=death_code,
                             time_horizon=time_horizon)

class IsMaleLF(LabelingFunction):
    """
        This labeler tries to predict whether or not a patient is male or not.
        The prediction time is on admission.

        This is primarily intended as a "debugging" labeler that should be "trivial" and get 1.0 AUROC.
    """

    def __init__(self, ontology: extension_datasets.Ontology):
        self.male_code: int = ontology.get_dictionary().index("demographics/gender/Male")
        self.admission_helper = InpatientAdmissionHelper(timelines)

    def label(self, patient: Patient) -> List[Label]:
        if len(patient.events) == 0:
            return []

        current_admission_index = 0
        admissions = self.admission_helper.get_inpatient_admissions(patient)

        labels = []

        is_male = self.male_code in patient.days[0].observations

        for i, day in enumerate(patient.days):
            if current_admission_index >= len(admissions):
                continue
            current_admission = admissions[current_admission_index]
            assert day.age <= current_admission.start_age

            if day.age == current_admission.start_age:
                current_admission_index += 1
                labels.append(Label(event_index=i, boolean_value=is_male))
                
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"