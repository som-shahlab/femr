"""Transforms that are unique to MIMIC OMOP."""
import datetime
from typing import Dict, List

from femr.datasets import RawPatient


def move_early_end_date_to_start_date(patient: RawPatient) -> RawPatient:
    """
    Some MIMIC-IV rows have an erroneous admission end datetime that occurs
    before the corresponding admission start datetime. In these cases move
    end datetime to end of day of admission.
    """

    for event in patient.events:
        if event.end is not None and event.omop_table == "visit_occurrence" and event.end < event.start:
            event.end = event.start

    return patient


def move_billing_codes(patient: RawPatient) -> RawPatient:
    """Move billing codes to the end of each visit.

    One issue with MIMIC is that billing codes are assigned at the start of the visit.
    This class fixes that by assigning them to the end of the visit.
    """
    visit_starts: Dict[int, datetime.datetime] = {}
    visit_ends: Dict[int, datetime.datetime] = {}
    tables_w_billing_codes: List[str] = ["condition_occurrence", "procedure_occurrence", "observation"]

    for event in patient.events:
        if event.omop_table == "visit_occurrence" and event.visit_id is not None:
            if event.start is None or event.end is None:
                raise RuntimeError(f"Missing visit start/end time for visit_occurrence_id {event.visit_id}")

            visit_starts[event.visit_id] = event.start
            visit_ends[event.visit_id] = event.end

    new_events = []
    for event in patient.events:
        if event.omop_table in tables_w_billing_codes and event.visit_id is not None:
            visit_start = visit_starts.get(event.visit_id)

            if event.start == visit_start:
                visit_end = visit_ends.get(event.visit_id)
                event.start = visit_end  # type: ignore

                if event.end is not None:
                    event.end = max(event.end, visit_end)

                new_events.append(event)

            else:
                new_events.append(event)

        else:
            new_events.append(event)

    patient.events = new_events
    patient.resort()

    return patient

def remove_very_old(patient: RawPatient) -> RawPatient:
    """Remove events that are after 125 years."""
    birth_date = patient.events[0].start
    new_events = []
    for event in patient.events:
        age = (event.start - birth_date)
        if age > datetime.timedelta(days=125 * 365): # Greater than 125 years old is not plausible
            continue
        new_events.append(event)
    patient.events = new_events
    return patient
