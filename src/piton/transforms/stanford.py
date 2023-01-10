"""Transforms that are unique to STARR OMOP."""

import datetime
from typing import Dict, Optional, Tuple

from piton import Patient
from piton.extractors.omop import OMOP_BIRTH


def move_visit_starts_to_day_start(patient: Patient) -> Patient:
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


def move_to_day_end(patient: Patient) -> Patient:
    """We assume that everything coded at midnight should actually be moved to the end of the day."""
    for event in patient.events:
        if (
            event.start.hour == 0
            and event.start.minute == 0
            and event.start.second == 0
            and event.code != OMOP_BIRTH
        ):
            event.start = (
                event.start
                + datetime.timedelta(days=1)
                - datetime.timedelta(minutes=1)
            )

            if event.end is not None:
                event.end = max(event.start, event.end)

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
