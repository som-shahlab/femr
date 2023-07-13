"""Transforms that are unique to MIMIC OMOP."""

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
