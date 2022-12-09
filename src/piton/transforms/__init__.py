"""A collection of general use transforms."""
import datetime
from typing import Any, Dict, Optional, Set, Tuple

from piton import Patient


def remove_short_patients(
    patient: Patient, min_num_dates: int = 3
) -> Optional[Patient]:
    """Remove patients with too few timepoints."""
    if (
        len(set(event.start.date() for event in patient.events))
        <= min_num_dates
    ):
        return None
    else:
        return patient


def remove_nones(patient: Patient) -> Patient:
    """Remove duplicate codes w/in same day if duplicate code has None value.

    There is no point having a NONE value in a timeline when we have an actual value within the same day.

    This removes those unnecessary NONE values.
    """
    has_value: Set[Tuple[int, datetime.date]] = set()

    for event in patient.events:
        if event.value is not None:
            has_value.add((event.code, event.start.date()))

    new_events = []
    for event in patient.events:
        if (
            event.value is None
            and (event.code, event.start.date()) in has_value
        ):
            continue
        new_events.append(event)

    patient.events = new_events

    patient.resort()

    return patient


def delta_encode(patient: Patient) -> Patient:
    """Delta encodes the patient.

    The idea behind delta encoding is that if we get duplicate values within a short amount of time
    (1 day for this code), there is not much point retaining the duplicate.

    This code removes all *sequential* duplicates within the same day.
    """
    last_value: Dict[Tuple[int, datetime.date], Any] = {}

    new_events = []
    for event in patient.events:
        key = (event.code, event.start.date())
        if key in last_value and last_value[key] == event.value:
            continue
        last_value[key] = event.value
        new_events.append(event)

    patient.events = new_events

    patient.resort()

    return patient
