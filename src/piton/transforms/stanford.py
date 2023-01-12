"""Transforms that are unique to STARR OMOP."""

import datetime
from typing import Dict, Optional, Tuple

from piton import Patient
from piton.extractors.omop import OMOP_BIRTH


def _move_date_to_end(
    d: datetime.datetime,
) -> datetime.datetime:
    if d.time() == datetime.time.min:
        return d + datetime.timedelta(days=1) - datetime.timedelta(minutes=1)
    else:
        return d


def move_visit_start_to_day_start(patient: Patient) -> Patient:
    """Assign visit start times of 12:00 AM to the start of the day (12:01 AM)

    This avoids visits being pushed to the end of the day by e.g., functions that map
    all events with midnight start times to the end of the day, such as `move_to_day_end`
    """
    for event in patient.events:
        if (
            event.start.hour == 0
            and event.start.minute == 0
            and event.start.second == 0
            and event.omop_table == "visit"
        ):
            event.start = event.start + datetime.timedelta(minutes=1)

            if event.end is not None:
                event.end = max(event.start, event.end)

    patient.resort()

    return patient


def move_visit_start_to_first_event_start(patient: Patient) -> Patient:
    """Assign visit start times to be just before the first event

    This function assigns the start time associated with each visit to be
    either (1) one second before the start time of the first event on the 
    day of the associated with the visit, or (2) 11:58:59 PM if the visit
    has no associated events or all events associated with the visit have 
    a start time of '00:00:00'. Events that occur on days prior to the 
    visit do not affect the visit start time.

    NOTE: This function assumes that all non-visit events with the same
    start time as the visit event (e.g., events with a start time at midnight
    such as billing codes, in the case where visit events also have a midnight
    start time) are moved to day end, otherwise there may be non-visit events 
    with start times before the visit start time.

    The reason for assigning a start time of 11:58:59 PM is that it makes
    visit start times fall one second before the 11:59:00 PM start times
    to which non-visit events with start times of `00:00:00` (such as 
    billing and diagnosis codes) are assigned via `move_to_day_end`.
    
    Our design choice assumes that temporal granularity in the raw data is
    at the minute level so that if the first non-visit event was at 12:01 AM
    then there will be no other events between 12:00 AM and 12:01 AM.
    
    Note that not all visit start times are set to 12:00 AM in the raw data.
    STARR-OMOP currently uses the first available value out of (1) hospital 
    admission time, (2) effective date datetime, and (3) effective date, in 
    that order. In the OMOP DEID from 12/20/2022 about 10% of visits have 
    a time that is not  '00:00:00'.
    """
    first_event_starts: Dict[int, datetime.datetime] = {}
    visit_starts: Dict[int, datetime.datetime] = {}

    # Find the given start time for each visit
    for event in patient.events:
        if event.omop_table == "visit":
            visit_starts[event.visit_id] = event.start

    # Find the minimum start time over all non-visit events associated with each visit
    for event in patient.events:
        if event.visit_id is not None:
            # Ignore any events for which event start is before the associated visit start
            # Also ignore non-visit events starting same time as visit (i.e., at midnight)
            if event.start > visit_starts[event.visit_id]:
                if event.visit_id in first_event_starts:
                    first_event_starts[event.visit_id] = min(
                        event.start, first_event_starts[event.visit_id]
                    )
                else:
                    first_event_starts[event.visit_id] = event.start

    # Assign visit start times to be one second before the first event associated with that visit
    for event in patient.events:
        if event.omop_table == "visit":
            if event.visit_id in first_event_starts:
                first_event_time = first_event_starts[event.visit_id]
                event.start = first_event_time - datetime.timedelta(seconds=1)
            elif event.start.time() == datetime.time.min:
                event.start = event.start + datetime.timedelta(days=1) - datetime.timedelta(seconds=61)

            if event.end is not None:
                # Reset the visit end to be ≥ the visit start
                event.end = max(event.start, event.end)

    patient.resort()

    return patient


def move_to_day_end(patient: Patient) -> Patient:
    """We assume that everything coded at midnight should actually be moved to the end of the day."""
    for event in patient.events:
        if event.code == OMOP_BIRTH:
            continue

        event.start = _move_date_to_end(event.start)
        if event.end is not None:

            event.end = _move_date_to_end(event.end)
            event.end = max(event.end, event.start)

    patient.resort()

    return patient


def move_pre_birth(patient: Patient) -> Optional[Patient]:
    """Move all events to after the birth of a patient."""
    birth_date = None
    for event in patient.events:
        if event.code == OMOP_BIRTH:
            birth_date = event.start

    if birth_date is None:
        return None

    new_events = []
    for event in patient.events:
        if event.start < birth_date:
            delta = birth_date - event.start
            if delta > datetime.timedelta(days=30):
                continue

            event.start = birth_date

        if event.end is not None and event.end < birth_date:
            event.end = birth_date

        new_events.append(event)

    patient.events = new_events
    patient.resort()

    return patient


def move_billing_codes(patient: Patient) -> Patient:
    """Move billing codes to the end of each visit.

    One issue with our OMOP extract is that billing codes are incorrectly assigned at the start of the visit.
    This class fixes that by assigning them to the end of the visit.
    """
    end_visits: Dict[int, datetime.datetime] = {}
    lowest_visit: Dict[Tuple[datetime.datetime, int], int] = {}

    billing_codes = [
        "pat_enc_dx",
        "hsp_acct_dx_list",
        "arpb_transactions",
    ]

    all_billing_codes = {
        (prefix + "_" + billing_code)
        for billing_code in billing_codes
        for prefix in ["shc", "lpch"]
    }

    for event in patient.events:
        if (
            event.clarity_table in all_billing_codes
            and event.visit_id is not None
        ):
            key = (event.start, event.code)
            if key not in lowest_visit:
                lowest_visit[key] = event.visit_id
            else:
                lowest_visit[key] = min(lowest_visit[key], event.visit_id)

        if event.clarity_table in ("lpch_pat_enc", "shc_pat_enc"):
            if event.end is not None:
                if event.visit_id is None:
                    raise RuntimeError(
                        f"Expected visit id for visit? {patient.patient_id} {event}"
                    )
                if end_visits.get(event.visit_id, event.end) != event.end:
                    raise RuntimeError(
                        f"Multiple end visits? {end_visits.get(event.visit_id)} {event}"
                    )
                end_visits[event.visit_id] = event.end

    new_events = []
    for event in patient.events:
        if event.clarity_table in all_billing_codes:
            key = (event.start, event.code)
            if event.visit_id != lowest_visit.get(key, None):
                # Drop this event as we already have it, just with a different visit_id?
                continue

            if event.visit_id is None:
                # This is a bad code, but would rather keep it than get rid of it
                new_events.append(event)
                continue

            end_visit = end_visits.get(event.visit_id)
            if end_visit is None:
                raise RuntimeError(
                    f"Expected visit end for code {patient.patient_id} {event} {patient}"
                )
            event.start = max(event.start, end_visit)
            if event.end is not None:
                event.end = max(event.end, end_visit)
            new_events.append(event)
        else:
            new_events.append(event)

    patient.events = new_events

    patient.resort()

    return patient
