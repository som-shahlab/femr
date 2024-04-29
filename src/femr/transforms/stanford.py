"""Transforms that are unique to STARR OMOP."""

import datetime
from typing import Dict, Optional, Tuple

from femr.datasets import RawPatient
from femr.extractors.omop import OMOP_BIRTH


def _move_date_to_end(
    d: datetime.datetime,
) -> datetime.datetime:
    if d.time() == datetime.time.min:
        return d + datetime.timedelta(days=1) - datetime.timedelta(minutes=1)
    else:
        return d


def move_visit_start_to_day_start(patient: RawPatient) -> RawPatient:
    """Assign visit start times of 12:00 AM to the start of the day (12:01 AM)

    This avoids visits being pushed to the end of the day by e.g., functions that map
    all events with midnight start times to the end of the day, such as `move_to_day_end`
    """
    for event in patient.events:
        if (
            event.start.hour == 0
            and event.start.minute == 0
            and event.start.second == 0
            and event.omop_table == "visit_occurrence"
        ):
            event.start = event.start + datetime.timedelta(minutes=1)

            if event.end is not None:
                event.end = max(event.start, event.end)

    patient.resort()

    return patient


def move_visit_start_to_first_event_start(patient: RawPatient) -> RawPatient:
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
        if event.omop_table == "visit_occurrence":
            if event.visit_id in visit_starts:
                raise RuntimeError(
                    f"Multiple visit events with visit ID {event.visit_id} for patient ID {patient.patient_id}"
                )
            visit_starts[event.visit_id] = event.start

    # Find the minimum start time over all non-visit events associated with each visit
    for event in patient.events:
        if event.visit_id is not None:
            # Only trigger for non-visit events with start time after associated visit start
            # Note: ignores non-visit events starting same time as visit (i.e., at midnight)
            if event.visit_id in visit_starts and event.start > visit_starts[event.visit_id]:
                first_event_starts[event.visit_id] = min(
                    event.start,
                    first_event_starts.get(event.visit_id, event.start),
                )

    # Assign visit start times to be same as first non-visit event with same visit ID
    for event in patient.events:
        if event.omop_table == "visit_occurrence":
            # Triggers if there is a non-visit event associated with the visit ID that has
            # start time strictly after the recorded visit start
            if event.visit_id in first_event_starts:
                event.start = first_event_starts[event.visit_id]

            if event.end is not None:
                # Reset the visit end to be ≥ the visit start
                event.end = max(event.start, event.end)

    patient.resort()

    return patient


def move_to_day_end(patient: RawPatient) -> RawPatient:
    """We assume that everything coded at midnight should actually be moved to the end of the day."""
    for event in patient.events:
        if event.concept_id == OMOP_BIRTH:
            continue

        event.start = _move_date_to_end(event.start)
        if event.end is not None:
            event.end = _move_date_to_end(event.end)
            event.end = max(event.end, event.start)

    patient.resort()

    return patient


def move_pre_birth(patient: RawPatient) -> Optional[RawPatient]:
    """Move all events to after the birth of a patient."""
    birth_date = None
    for event in patient.events:
        if event.concept_id == OMOP_BIRTH:
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


def move_billing_codes(patient: RawPatient) -> RawPatient:
    """Move billing codes to the end of each visit.

    One issue with our OMOP extract is that billing codes are incorrectly assigned at the start of the visit.
    This class fixes that by assigning them to the end of the visit.
    """
    end_visits: Dict[int, datetime.datetime] = {}  # Map from visit ID to visit end time
    lowest_visit: Dict[Tuple[datetime.datetime, int], int] = {}  # Map from code/start time pairs to visit ID

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
            key = (event.start, event.concept_id)
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

    new_events = []
    for event in patient.events:
        if event.clarity_table in all_billing_codes:
            key = (event.start, event.concept_id)
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
                raise RuntimeError(f"Expected visit end for code {patient.patient_id} {event} {patient}")

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

def select_concept_id(current_concept_id: int, new_concept_id: int) -> int:
    # Prefer "Visit/ERIP" > "Visit/IP" > "Visit/OP" > "Visit/" > everything else
    if current_concept_id == 262 or new_concept_id == 262:
        return 262 # "Visit/ERIP"
    elif current_concept_id == 9201 or new_concept_id == 9201:
        return 9201 # "Visit/IP"
    elif current_concept_id == 9202 or new_concept_id == 9202:
        return 9202 # "Visit/OP"
    else:
        return current_concept_id

def join_consecutive_day_visits(patient: RawPatient) -> RawPatient:
    """If two visits are on consecutive days, merge them into one visit"""
    current_visit_id: int = None
    current_visit_end: datetime.datetime = None
    current_visit_concept_id: str = None
    old_visit_id_2_new_visit_id: Dict[int, int] = {}
    for event in patient.events:
        if event.visit_id is not None and event.omop_table in ["visit", "visit_occurrence", "visit_detail"]:
            # Found visit measurement
            m_end = (
                datetime.datetime.fromisoformat(event.end)
                if isinstance(event.end, str)
                else event.end
            )
            if current_visit_id is None:
                # Start a new visit
                current_visit_id = event.visit_id
                current_visit_end = m_end
                current_visit_concept_id = event.concept_id
            elif event.visit_id == current_visit_id:
                # Same visit, so update its end time
                current_visit_end = max(m_end, current_visit_end)
                current_visit_concept_id = select_concept_id(current_visit_concept_id, event.concept_id)
            elif event.visit_id in old_visit_id_2_new_visit_id:
                # We have already merged this visit, so update its end time
                current_visit_end = max(m_end, current_visit_end)
                current_visit_concept_id = select_concept_id(current_visit_concept_id, event.concept_id)
            else:
                if (event.start - current_visit_end).days <= 1:
                    # Merge the two visits
                    current_visit_end = max(m_end, current_visit_end)
                    current_visit_concept_id = select_concept_id(current_visit_concept_id, event.concept_id)
                else:
                    # Start a new visit
                    current_visit_id = event.visit_id
                    current_visit_end = m_end
                    current_visit_concept_id = event.concept_id
            # NOTE: Need to update both this visit_id and the current_visit_id
            old_visit_id_2_new_visit_id[event.visit_id] = {
                "visit_id": current_visit_id,
                "end": current_visit_end,
                "concept_id": current_visit_concept_id,
            }
            old_visit_id_2_new_visit_id[current_visit_id] = {
                "visit_id": current_visit_id,
                "end": current_visit_end,
                "concept_id": current_visit_concept_id,
            }
    events = []
    for event in patient.events:
        if event.visit_id:
            if event.omop_table == "visit_occurrence":
                # If this is a visit event, update its end time and delete (if not original visit_id)
                event.end = old_visit_id_2_new_visit_id[event.visit_id]["end"]
                event.concept_id = old_visit_id_2_new_visit_id[event.visit_id]["concept_id"]
                if old_visit_id_2_new_visit_id[event.visit_id]["visit_id"] == event.visit_id:
                    events.append(event)
            else:
                # Update the visit_id
                event.visit_id = old_visit_id_2_new_visit_id[event.visit_id]["visit_id"]
                events.append(event)
        else:
            events.append(event)
    patient.events = events
    return patient