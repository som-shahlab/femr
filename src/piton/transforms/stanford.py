"""Transforms that are unique to STARR OMOP."""

import dataclasses
import datetime
from typing import Dict, Optional, Tuple

from piton import Patient
from piton.extractors.omop import OMOP_BIRTH


def move_to_day_end(patient: Patient) -> Patient:
    """We assume that everything coded at midnight should actually be moved to the end of the day."""
    new_events = []
    for event in patient.events:
        if (
            event.start.hour == 0
            and event.start.minute == 0
            and event.start.second == 0
            and event.code != OMOP_BIRTH
        ):
            new_time = (
                event.start
                + datetime.timedelta(days=1)
                - datetime.timedelta(minutes=1)
            )

            old_end = event.get_datetime("end")

            new_metadata = event.metadata

            if old_end is not None:
                end = max(new_time, old_end)
                new_metadata |= {"end": end}

            new_events.append(
                dataclasses.replace(
                    event, start=new_time, metadata=new_metadata
                )
            )
        else:
            new_events.append(event)

    new_events.sort(key=lambda a: (a.start, a.code))

    return Patient(patient.patient_id, new_events)


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
        new_start = None
        new_end = None

        if event.start < birth_date:
            delta = birth_date - event.start
            if delta > datetime.timedelta(days=30):
                continue

            new_start = birth_date

        old_end = event.get_datetime("end")
        if old_end and old_end < birth_date:
            new_end = birth_date

        if new_start or new_end:
            new_events.append(
                dataclasses.replace(
                    event,
                    start=new_start or event.start,
                    metadata=event.metadata
                    | ({"end": new_end} if new_end else {}),
                )
            )
        else:
            new_events.append(event)

    return Patient(patient_id=patient.patient_id, events=new_events)


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
        event_visit_id = event.get_int("visit_id")
        if (
            event.get_str("clarity_table") in all_billing_codes
            and event_visit_id is not None
        ):
            key = (event.start, event.code)
            if key not in lowest_visit:
                lowest_visit[key] = event_visit_id
            else:
                lowest_visit[key] = min(lowest_visit[key], event_visit_id)

        if event.get_str("clarity_table") in ("lpch_pat_enc", "shc_pat_enc"):
            event_end_date = event.get_datetime("end")
            if event_end_date is not None:
                if event_visit_id is None:
                    raise RuntimeError(
                        f"Expected visit id for visit? {patient.patient_id} {event}"
                    )
                if (
                    end_visits.get(event_visit_id, event_end_date)
                    != event_end_date
                ):
                    raise RuntimeError(
                        f"Multiple end visits? {end_visits.get(event_visit_id)} {event}"
                    )
                end_visits[event_visit_id] = event_end_date

    new_events = []

    for event in patient.events:
        event_visit_id = event.get_int("visit_id")
        event_end_date = event.get_datetime("end")
        if event.get_str("clarity_table") in all_billing_codes:
            key = (event.start, event.code)
            if event_visit_id != lowest_visit.get(key, None):
                continue

            if event_visit_id is None:
                # This is a bad code, but would rather keep it than get rid of it
                new_events.append(event)
                continue

            end_visit = end_visits.get(event_visit_id)
            if end_visit is None:
                raise RuntimeError(
                    f"Expected visit end for code {patient.patient_id} {event} {patient}"
                )
            new_metadata = event.metadata
            if event_end_date is not None:
                new_metadata |= {"end": max(end_visit, event_end_date)}
            new_events.append(
                dataclasses.replace(
                    event, start=end_visit, metadata=new_metadata
                )
            )
        else:
            new_events.append(event)

    new_events.sort(key=lambda a: (a.start, a.code))

    return Patient(patient_id=patient.patient_id, events=new_events)
