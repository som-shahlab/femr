"""A collection of general use transforms."""
import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import meds


def remove_nones(
    patient: meds.Patient,
    do_not_apply_to_filter: Optional[Callable[[meds.Measurement], bool]] = None,
) -> meds.Patient:
    """Remove duplicate codes w/in same day if duplicate code has None value.

    There is no point having a NONE value in a timeline when we have an actual value within the same day.

    This removes those unnecessary NONE values.
    """
    do_not_apply_to_filter = do_not_apply_to_filter or (lambda _: False)
    has_value: Set[Tuple[str, datetime.date]] = set()

    for event in patient["events"]:
        for measurement in event["measurements"]:
            value = (measurement["numeric_value"], measurement["text_value"], measurement["datetime_value"])
            if any(v is not None for v in value):
                has_value.add((measurement["code"], event["time"].date()))

    new_events: List[meds.Event] = []
    for event in patient["events"]:
        new_measurements: List[meds.Measurement] = []

        for measurement in event["measurements"]:
            value = (measurement["numeric_value"], measurement["text_value"], measurement["datetime_value"])
            if (
                all(v is None for v in value)
                and (measurement["code"], event["time"].date()) in has_value
                and not do_not_apply_to_filter(measurement)
            ):
                # Skip this event as already in there
                continue

            new_measurements.append(measurement)

        if len(new_measurements) > 0:
            new_events.append({"time": event["time"], "measurements": new_measurements})

    patient["events"] = new_events
    patient["events"].sort(key=lambda a: a["time"])

    return patient


def delta_encode(
    patient: meds.Patient,
    do_not_apply_to_filter: Optional[Callable[[meds.Measurement], bool]] = None,
) -> meds.Patient:
    """Delta encodes the patient.

    The idea behind delta encoding is that if we get duplicate values within a short amount of time
    (1 day for this code), there is not much point retaining the duplicate.

    This code removes all *sequential* duplicates within the same day.
    """
    do_not_apply_to_filter = do_not_apply_to_filter or (lambda _: False)

    last_value: Dict[Tuple[str, datetime.date], Any] = {}

    new_events: List[meds.Event] = []
    for event in patient["events"]:
        new_measurements: List[meds.Measurement] = []
        for measurement in event["measurements"]:
            key = (measurement["code"], event["time"].date())
            value = (measurement["numeric_value"], measurement["text_value"], measurement["datetime_value"])
            if key in last_value and last_value[key] == value and not do_not_apply_to_filter(measurement):
                continue
            last_value[key] = value
            new_measurements.append(measurement)

        if len(new_measurements) > 0:
            new_events.append({"time": event["time"], "measurements": new_measurements})

    patient["events"] = new_events
    patient["events"].sort(key=lambda a: a["time"])

    return patient


def fix_events(patient: meds.Patient) -> meds.Patient:
    """After a series of transformations, sometimes the patient structure gets a bit messed up.
    The usual issues are either duplicate event times or missorted events.

    This does a final cleanup pass to meet the MEDS requirements.
    """
    patient["events"].sort(key=lambda a: a["time"])

    if len(patient["events"]) == 0:
        return patient

    new_events = []
    new_event: meds.Event = {"time": patient["events"][0]["time"], "measurements": []}
    for event in patient["events"]:
        if new_event["time"] != event["time"]:
            new_events.append(new_event)
            new_event = {"time": event["time"], "measurements": []}
        new_event["measurements"].extend(event["measurements"])
    new_events.append(new_event)
    patient["events"] = new_events

    return patient
