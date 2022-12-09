"""Labeling functions for OMOP data."""
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
    """Apply a label based on a single code's occurrence over a fixed time horizon.

    TODO - Test on real data
    """

    def __init__(self, code: int, time_horizon: TimeHorizon):
        """Label the code whose index in your Ontology is equal to `code`."""
        self.code = code
        self.time_horizon = time_horizon

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return each event's start time as the time to make a prediction."""
        return [e.start for e in patient.events]

    def get_time_horizon(self) -> TimeHorizon:
        """Return time horizon."""
        return self.time_horizon

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return the start times of this patient's events which have the exact same `code` as `self.code`."""
        times: List[datetime.datetime] = []
        for event in patient.events:
            if event.code == self.code:
                times.append(event.start)
        return times


class MortalityLF(CodeLF):
    """Apply a label for whether or not a patient dies within the `time_horizon`.

    TODO - Test on real data
    """

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
    """Apply a label for whether or not a patient is male or not.

    The prediction time is on admission.

    This is primarily intended as a "debugging" labeler that should be "trivial" and get 1.0 AUROC.

    TODO - Test on real data
    """

    def __init__(self, ontology: extension_datasets.Ontology):
        """Construct a Male labeler.

        Args:
            ontology (extension_datasets.Ontology): Maps code IDs to code names.

        Raises:
            ValueError: Raised if there is no code corresponding to inpatient visit.
        """
        INPATIENT_VISIT_CODE = "Visit/IP"
        self.male_code: int = ontology.get_dictionary().index(
            "demographics/gender/Male"
        )
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
                labels.append(Label(time=event.start, value=is_male))
        return labels

    def get_labeler_type(self) -> LabelType:
        """Return that these labels are booleans."""
        return "boolean"
