"""Transforms that are unique to STARR OMOP."""

import datetime
from typing import Dict, Optional, Tuple

from piton import Patient
from piton.extractors.omop import OMOP_BIRTH


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
    end_visits: Dict[
        int, datetime.datetime
    ] = {}  # Map from visit ID to visit end time
    lowest_visit: Dict[
        Tuple[datetime.datetime, int], int
    ] = {}  # Map from code/start time pairs to visit ID

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
        # For events that share the same code/start time, we find the lowest visit ID
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
                    # Every event with an end time should have a visit ID associated with it
                    raise RuntimeError(
                        f"Expected visit id for visit? {patient.patient_id} {event}"
                    )
                if end_visits.get(event.visit_id, event.end) != event.end:
                    # Every event associated with a visit should have an end time that matches the visit end time
                    # Also the end times of all events associated with a visit should have the same end time
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
                # We only keep the copy of the event associated with the lowest visit id
                # (Lowest visit id is arbitrary, no explicit connection to time)
                continue

            if event.visit_id is None:
                # This is a bad code (it has no associated visit_id), but
                # we would rather keep it than get rid of it
                new_events.append(event)
                continue

            end_visit = end_visits.get(event.visit_id)
            if end_visit is None:
                raise RuntimeError(
                    f"Expected visit end for code {patient.patient_id} {event} {patient}"
                )

            # The start time for an event should be no later than its associated visit end time
            event.start = max(event.start, end_visit)

            # The end time for an event should be no later than its associated visit end time
            if event.end is not None:
                event.end = max(event.end, end_visit)
            new_events.append(event)
        else:
            new_events.append(event)

    patient.events = new_events

    patient.resort()

    return patient
