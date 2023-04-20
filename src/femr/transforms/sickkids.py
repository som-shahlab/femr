"""Transforms that are unique to SK OMOP."""

import datetime
from typing import Optional

from femr.datasets import RawPatient
from femr.extractors.omop import OMOP_BIRTH


def replace_categorical_measurement_results(patient: RawPatient) -> RawPatient:
    """Replace value_as_number of 9999999 with None. 9999999 is the default
    Clarity assignment when the lab result is a categorical value.
    """
    for event in patient.events:
        if event.omop_table == "measurement" and event.value == 9999999:
            event.value = None

    return patient


def replace_default_birthdate(patient: RawPatient) -> Optional[RawPatient]:
    """Replace default birthdate in SickKids OMOP (1-1-1) with (1900-1-1)."""
    for event in patient.events:
        if event.code == OMOP_BIRTH and event.start == datetime.datetime(1, 1, 1):
            event.start = datetime.datetime(1900, 1, 1)

    patient.resort()

    return patient
