"""Labeling functions for OMOP data."""
from __future__ import annotations

import datetime
from collections import deque
from typing import List, Optional, Set, Tuple

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
    """Apply a label based on a single code's occurrence over a fixed time horizon."""

    def __init__(
        self, admission_code: int, code: int, time_horizon: TimeHorizon
    ):
        """Label the code whose index in your Ontology is equal to `code`."""
        self.admission_code = admission_code
        self.code = code
        self.time_horizon = time_horizon

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        """Return each event's start time as the time to make a prediction."""
        return [
            datetime.datetime.strptime(
                str(e.start)[:10] + " 23:59:00", "%Y-%m-%d %H:%M:%S"
            )
            for e in patient.events
            if e.code == self.admission_code
        ]

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
        CODE_DEATH_PREFIX = "Condition Type/OMOP4822053"
        INPATIENT_VISIT_CODE = "Visit/IP"
        admission_code = ontology.get_dictionary().index(INPATIENT_VISIT_CODE)

        death_codes: Set[Tuple[str, int]] = set()
        for code, code_str in enumerate(ontology.get_dictionary()):
            # code_str = bytes(code_str).decode("utf-8")
            if code_str == CODE_DEATH_PREFIX:
                death_codes.add((code_str, code))
            # if code_str.startswith(CODE_DEATH_PREFIX):
            #     death_codes.add((code_str, code))

        if len(death_codes) != 1:
            raise ValueError(
                f"Could not find exactly one death code -- instead found {len(death_codes)} codes: {str(death_codes)}"
            )
        else:
            death_code: int = list(death_codes)[0][1]
            super().__init__(
                admission_code=admission_code,
                code=death_code,
                time_horizon=time_horizon,
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
            # code_str = bytes(code_str).decode("utf-8")
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
                admission_code=admission_code,
                code=diabetes_code,
                time_horizon=time_horizon,
            )


def _get_all_children(
    ontology: extension_datasets.Ontology, code: int
) -> Set[int]:

    children_code_set = set([code])
    parent_deque = deque([code])

    while len(parent_deque) > 0:
        temp_parent_code = parent_deque.popleft()
        for temp_child_code in ontology.get_children(temp_parent_code):
            children_code_set.add(temp_child_code)
            parent_deque.append(temp_child_code)

    return children_code_set


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

        DIABETES_STR = "SNOMED/44054006"
        HbA1c_STR = "LOINC/4548-4"

        self.last_trigger_days = last_trigger_days
        self.hba1c_lab_code = ontology.get_dictionary().index(HbA1c_STR)

        diabetes_code = ontology.get_dictionary().index(DIABETES_STR)
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

            if (
                first_diabetes_code_date is not None
                and event.start > first_diabetes_code_date
            ):
                break

            if event.value is None or type(event.value) is str:
                continue

            if event.code == self.hba1c_lab_code:

                is_diabetes = event.value > 6.5

                if (
                    last_trigger is None
                    or (event.start - last_trigger).days
                    > self.last_trigger_days
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
        """Return that these labels are booleans."""
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
                labels.append(Label(time=event.start, value=is_male))
        return labels

    def get_labeler_type(self) -> LabelType:
        """Return that these labels are booleans."""
        return "boolean"
