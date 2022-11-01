from __future__ import annotations

import datetime
from typing import List, Set, Tuple

from .. import Event, Patient
from ..extension import datasets as extension_datasets
from .core import (
    FixedTimeHorizonEventLF,
    Label,
    LabelingFunction,
    LabelType,
    TimeHorizon,
)

##########################################################
# Labeling functions derived from FixedTimeHorizonEventLF
##########################################################


class CodeLF(FixedTimeHorizonEventLF):
    """
    TODO - Test on real data
    Applies a label based on a single code's occurrence over a fixed time horizon
    """

    def __init__(self, code: int, time_horizon: TimeHorizon):
        """Label the code whose index in your Ontology is equal to `code`"""
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
    TODO - Test on real data
    The mortality task is defined as predicting whether or not a
    patient will die within the next `time_horizon` time.
    """

    def __init__(
        self, ontology: extension_datasets.Ontology, time_horizon: TimeHorizon
    ):
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
            super().__init__(code=death_code, time_horizon=time_horizon)


##########################################################
# Other
##########################################################


class IsMaleLF(LabelingFunction):
    """
    TODO - Test on real data
    This labeler tries to predict whether or not a patient is male or not.
    The prediction time is on admission.

    This is primarily intended as a "debugging" labeler that should be "trivial" and get 1.0 AUROC.
    """

    def __init__(self, ontology: extension_datasets.Ontology):
        INPATIENT_VISIT_CODE = "Visit/IP"
        self.male_code: int = ontology.get_dictionary().index(
            "demographics/gender/Male"
        )
        admission_code = ontology.get_dictionary().map(INPATIENT_VISIT_CODE)
        if admission_code is None:
            raise ValueError(
                f"Could not find inpatient visit code for: {INPATIENT_VISIT_CODE}"
            )
        else:
            self.admission_code = admission_code

    def is_inpatient_admission(self, event: Event) -> bool:
        return event.code == self.admission_code

    def label(self, patient: Patient) -> List[Label]:
        if len(patient.events) == 0:
            return []

        labels: List[Label] = []
        is_male: bool = self.male_code in [
            event.code for event in patient.events
        ]

        for event in patient.events:
            if self.is_inpatient_admission(event):
                labels.append(
                    Label(time=event.time, value=is_male, label_type="boolean")
                )
        return labels

    def get_labeler_type(self) -> LabelType:
        return "boolean"
