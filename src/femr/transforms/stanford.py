# mypy: disable-error-code="attr-defined"

"""Transforms that are unique to STARR OMOP."""

import datetime
from typing import Dict, List, Tuple

import meds
import meds_reader


def _move_date_to_end(
    d: datetime.datetime,
) -> datetime.datetime:
    if d.time() == datetime.time.min:
        return d + datetime.timedelta(days=1) - datetime.timedelta(minutes=1)
    else:
        return d


def move_visit_start_to_first_event_start(patient: meds_reader.Patient) -> meds_reader.Patient:
    """Assign visit start times to equal start time of first event in visit

    This function assigns the start time associated with each visit to be
    the start time of the first event that (1) is associated with the visit
    (i.e., shares the same visit ID as the visit event), (2) is a non-visit
    event, and (3) occurs on the same day as the visit event. If the visit
    has no non-visit events or all events associated with the visit have
    the same start time as the visit event (e.g., events with a start time
    of midnight such as billing codes, assuming visit events also have a
    midnight start time) then the visit start time remains unchanged.
    Events that occur on days prior to the visit do not affect the visit
    start time.

    Note that not all visit start times are set to 12:00 AM in the raw data.
    STARR-OMOP currently uses the first available value out of (1) hospital
    admission time, (2) effective date datetime, and (3) effective date, in
    that order. In the OMOP DEID from 12/20/2022 about 10% of visits have
    a time that is not  '00:00:00'.
    """
    first_event_starts: Dict[int, datetime.datetime] = {}
    visit_starts: Dict[int, datetime.datetime] = {}

    # Find the stated start time for each visit
    for event in patient.events:
        if event.table == "visit":
            if event.visit_id in visit_starts and visit_starts[event.visit_id] != event.time:
                raise RuntimeError(
                    f"Multiple visit events with visit ID {event.visit_id} " + f" for patient ID {patient.patient_id}"
                )
            visit_starts[event.visit_id] = event.time

    # Find the minimum start time over all non-visit events associated with each visit
    for event in patient.events:
        if event.visit_id is not None:
            # Only trigger for non-visit events with start time after associated visit start
            # Note: ignores non-visit events starting same time as visit (i.e., at midnight)
            if event.visit_id in visit_starts and event.time > visit_starts[event.visit_id]:
                first_event_starts[event.visit_id] = min(
                    event.time,
                    first_event_starts.get(event.visit_id, event.time),
                )

    # Assign visit start times to be same as first non-visit event with same visit ID
    for event in patient.events:
        if event.table == "visit":
            # Triggers if there is a non-visit event associated with the visit ID that has
            # start time strictly after the recorded visit start
            if event.visit_id in first_event_starts:
                event.time = first_event_starts[event.visit_id]

            if event.end is not None:
                # Reset the visit end to be ≥ the visit start
                event.end = max(event.time, event.end)

    patient.events.sort(key=lambda a: a.time)

    return patient


def move_to_day_end(patient: meds_reader.Patient) -> meds_reader.Patient:
    """We assume that everything coded at midnight should actually be moved to the end of the day."""
    for event in patient.events:
        event.time = _move_date_to_end(event.time)
        if event.end is not None:
            event.end = _move_date_to_end(event.end)
            event.end = max(event.end, event.time)

    patient.events.sort(key=lambda a: a.time)

    return patient


def switch_to_icd10cm(patient: meds_reader.Patient) -> meds_reader.Patient:
    """Switch from ICD10 to ICD10CM."""
    for event in patient.events:
        if event.code.startswith("ICD10/"):
            event.code = event.code.replace("ICD10/", "ICD10CM/", 1)

    return patient


def move_pre_birth(patient: meds_reader.Patient) -> meds_reader.Patient:
    """Move all events to after the birth of a patient."""
    birth_date = None
    for event in patient.events:
        if event.code == meds.birth_code:
            birth_date = event.time

    assert birth_date is not None

    new_events = []
    for event in patient.events:
        if event.time < birth_date:
            delta = birth_date - event.time
            if delta > datetime.timedelta(days=30):
                continue

            event.time = birth_date

            if event.end is not None and event.end < birth_date:
                event.end = birth_date

        new_events.append(event)

    patient.events = new_events
    patient.events.sort(key=lambda a: a.time)

    return patient


def move_billing_codes(patient: meds_reader.Patient) -> meds_reader.Patient:
    """Move billing codes to the end of each visit.

    One issue with our OMOP extract is that billing codes are incorrectly assigned at the start of the visit.
    This class fixes that by assigning them to the end of the visit.
    """
    end_visits: Dict[int, datetime.datetime] = {}  # Map from visit ID to visit end time
    lowest_visit: Dict[Tuple[datetime.datetime, str], int] = {}  # Map from code/start time pairs to visit ID

    # List of billing code tables based on the original Clarity queries used to form STRIDE
    billing_codes = [
        "pat_enc_dx",
        "hsp_acct_dx_list",
        "arpb_transactions",
    ]

    all_billing_codes = {(prefix + "_" + billing_code) for billing_code in billing_codes for prefix in ["shc", "lpch"]}

    for event in patient.events:
        # For events that share the same code/start time, we find the lowest visit ID
        if event.clarity_table in all_billing_codes and event.visit_id is not None:
            key = (event.time, event.code)
            if key not in lowest_visit:
                lowest_visit[key] = event.visit_id
            else:
                lowest_visit[key] = min(lowest_visit[key], event.visit_id)

        if event.clarity_table in ("lpch_pat_enc", "shc_pat_enc"):
            if event.end is not None:
                if event.visit_id is None:
                    # Every event with an end time should have a visit ID associated with it
                    raise RuntimeError(f"Expected visit id for visit? {patient.patient_id} {event}")
                if end_visits.get(event.visit_id, event.end) != event.end:
                    # Every event associated with a visit should have an end time that matches the visit end time
                    # Also the end times of all events associated with a visit should have the same end time
                    raise RuntimeError(f"Multiple end visits? {end_visits.get(event.visit_id)} {event}")
                end_visits[event.visit_id] = event.end

    for event in patient.events:
        if event.clarity_table in all_billing_codes:
            key = (event.time, event.code)
            if event.visit_id != lowest_visit.get(key, None):
                # Drop this event as we already have it, just with a different visit_id?
                # We only keep the copy of the event associated with the lowest visit id
                # (Lowest visit id is arbitrary, no explicit connection to time)
                continue

            if event.visit_id is None:
                # This is a bad code (it has no associated visit_id), but
                # we would rather keep it than get rid of it
                continue

            end_visit = end_visits.get(event.visit_id)
            if end_visit is None:
                raise RuntimeError(f"Expected visit end for code {patient.patient_id} {event} {patient}")

            # The end time for an event should be no later than its associated visit end time
            if event.end is not None:
                event.end = max(event.end, end_visit)

            # The start time for an event should be no later than its associated visit end time
            event.time = max(event.time, end_visit)

    patient.events.sort(key=lambda a: a.time)

    return patient
