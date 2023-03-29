from __future__ import annotations

import datetime

import femr
from femr.extractors.omop import OMOP_BIRTH
from femr.transforms import delta_encode, remove_nones, remove_short_patients
from femr.transforms.stanford import (
    move_billing_codes,
    move_pre_birth,
    move_to_day_end,
    move_visit_start_to_day_start,
    move_visit_start_to_first_event_start,
)


def test_pre_birth() -> None:
    patient = femr.Patient(
        patient_id=123,
        events=[
            femr.Event(start=datetime.datetime(1999, 7, 2), code=1234),
            femr.Event(start=datetime.datetime(1999, 7, 9), code=OMOP_BIRTH),
            femr.Event(start=datetime.datetime(1999, 7, 11), code=12345),
        ],
    )

    expected = femr.Patient(
        patient_id=123,
        events=[
            femr.Event(start=datetime.datetime(1999, 7, 9), code=1234),
            femr.Event(start=datetime.datetime(1999, 7, 9), code=OMOP_BIRTH),
            femr.Event(start=datetime.datetime(1999, 7, 11), code=12345),
        ],
    )

    assert move_pre_birth(patient) == expected


def test_remove_small() -> None:
    patient = femr.Patient(
        patient_id=123,
        events=[
            femr.Event(start=datetime.datetime(1999, 7, 2), code=1234),
            femr.Event(start=datetime.datetime(1999, 7, 9), code=OMOP_BIRTH),
            femr.Event(start=datetime.datetime(1999, 7, 11), code=12345),
            femr.Event(start=datetime.datetime(1999, 7, 13), code=12345),
        ],
    )

    invalid = femr.Patient(
        patient_id=123,
        events=[
            femr.Event(start=datetime.datetime(1999, 7, 9), code=OMOP_BIRTH),
            femr.Event(start=datetime.datetime(1999, 7, 11), code=12345),
        ],
    )

    assert remove_short_patients(patient) == patient

    assert remove_short_patients(invalid) is None


def test_move_visit_start_to_day_start() -> None:
    patient = femr.Patient(
        patient_id=123,
        events=[
            femr.Event(
                start=datetime.datetime(1999, 7, 2),
                code=1234,
                omop_table="visit_occurrence",
            ),
            femr.Event(start=datetime.datetime(1999, 7, 2, 12), code=4321),
            femr.Event(start=datetime.datetime(1999, 7, 9), code=OMOP_BIRTH),
        ],
    )

    expected = femr.Patient(
        patient_id=123,
        events=[
            femr.Event(
                start=datetime.datetime(1999, 7, 2, 0, 1),
                code=1234,
                omop_table="visit_occurrence",
            ),
            femr.Event(start=datetime.datetime(1999, 7, 2, 12), code=4321),
            femr.Event(start=datetime.datetime(1999, 7, 9), code=OMOP_BIRTH),
        ],
    )

    assert move_visit_start_to_day_start(patient) == expected


def test_move_visit_start_ignores_other_visits() -> None:
    patient = femr.Patient(
        patient_id=123,
        events=[
            femr.Event(  # A non-visit event with no explicit start time
                start=datetime.datetime(1999, 7, 2), code=1234, visit_id=9999
            ),
            femr.Event(  # A visit event with just date specified
                start=datetime.datetime(1999, 7, 2),
                code=4567,
                omop_table="visit_occurrence",
                visit_id=9999,
            ),
            femr.Event(  # A non-visit event from a separate visit ID
                start=datetime.datetime(1999, 7, 2, 11),
                code=2345,
                visit_id=8888,
            ),
            femr.Event(  # First recorded non-visit event for visit ID 9999
                start=datetime.datetime(1999, 7, 2, 12),
                code=3456,
                visit_id=9999,
            ),
        ],
    )

    # Note that events are implicitly sorted first by start time, then by code:
    # https://github.com/som-shahlab/femr/blob/main/src/femr/__init__.py#L69
    expected = femr.Patient(
        patient_id=123,
        events=[
            femr.Event(  # A non-visit event with no explicit start time
                start=datetime.datetime(1999, 7, 2), code=1234, visit_id=9999
            ),
            femr.Event(  # A non-visit event from a separate visit ID
                start=datetime.datetime(1999, 7, 2, 11),
                code=2345,
                visit_id=8888,
            ),
            femr.Event(  # First recorded non-visit event for visit ID 9999
                start=datetime.datetime(1999, 7, 2, 12),
                code=3456,
                visit_id=9999,
            ),
            femr.Event(  # Now visit event has date and time specified
                start=datetime.datetime(1999, 7, 2, 12),
                code=4567,  # Comes after previous event b/c 4567 > 3456
                omop_table="visit_occurrence",
                visit_id=9999,
            ),
        ],
    )

    assert move_visit_start_to_first_event_start(patient) == expected


def test_move_visit_start_minute_after_midnight() -> None:
    patient = femr.Patient(
        patient_id=123,
        events=[
            femr.Event(start=datetime.datetime(1999, 7, 2), code=1234, visit_id=9999),
            femr.Event(
                start=datetime.datetime(1999, 7, 2, 0, 1),
                code=2345,
                visit_id=9999,
            ),
            femr.Event(
                start=datetime.datetime(1999, 7, 2, 12),
                code=4567,
                visit_id=9999,
            ),
            femr.Event(
                start=datetime.datetime(1999, 7, 2),
                code=3456,
                visit_id=9999,
                omop_table="visit_occurrence",
            ),
        ],
    )

    expected = femr.Patient(
        patient_id=123,
        events=[
            femr.Event(start=datetime.datetime(1999, 7, 2), code=1234, visit_id=9999),
            femr.Event(
                start=datetime.datetime(1999, 7, 2, 0, 1),
                code=2345,
                visit_id=9999,
            ),
            femr.Event(
                start=datetime.datetime(1999, 7, 2, 0, 1),
                code=3456,
                visit_id=9999,
                omop_table="visit_occurrence",
            ),
            femr.Event(
                start=datetime.datetime(1999, 7, 2, 12),
                code=4567,
                visit_id=9999,
            ),
        ],
    )

    assert move_visit_start_to_first_event_start(patient) == expected


def test_move_visit_start_doesnt_move_without_event() -> None:
    patient = femr.Patient(
        patient_id=123,
        events=[
            femr.Event(start=datetime.datetime(1999, 7, 2), code=1234, visit_id=9999),
            femr.Event(
                start=datetime.datetime(1999, 7, 2),
                code=3456,
                visit_id=9999,
                omop_table="visit_occurrence",
            ),
            femr.Event(
                start=datetime.datetime(1999, 7, 2, 0, 0),
                code=2345,
                visit_id=9999,
            ),
        ],
    )

    # None of the non-visit events have start time > '00:00:00' so visit event
    # start time is unchanged, though order changes based on code under resort.
    expected = femr.Patient(
        patient_id=123,
        events=[
            femr.Event(start=datetime.datetime(1999, 7, 2), code=1234, visit_id=9999),
            femr.Event(
                start=datetime.datetime(1999, 7, 2, 0, 0),
                code=2345,
                visit_id=9999,
            ),
            femr.Event(
                start=datetime.datetime(1999, 7, 2),
                code=3456,
                visit_id=9999,
                omop_table="visit_occurrence",
            ),
        ],
    )

    assert move_visit_start_to_first_event_start(patient) == expected


def test_move_to_day_end() -> None:
    patient = femr.Patient(
        patient_id=123,
        events=[
            femr.Event(start=datetime.datetime(1999, 7, 2), code=1234),
            femr.Event(start=datetime.datetime(1999, 7, 2, 12), code=4321),
            femr.Event(start=datetime.datetime(1999, 7, 9), code=OMOP_BIRTH),
        ],
    )

    expected = femr.Patient(
        patient_id=123,
        events=[
            femr.Event(start=datetime.datetime(1999, 7, 2, 12), code=4321),
            femr.Event(start=datetime.datetime(1999, 7, 2, 23, 59), code=1234),
            femr.Event(start=datetime.datetime(1999, 7, 9), code=OMOP_BIRTH),
        ],
    )

    assert move_to_day_end(patient) == expected


def test_remove_nones() -> None:
    patient = femr.Patient(
        patient_id=123,
        events=[
            femr.Event(start=datetime.datetime(1999, 7, 2), code=1234),  # No value, to be removed
            femr.Event(start=datetime.datetime(1999, 7, 2, 12), code=1234, value=3),
            femr.Event(start=datetime.datetime(1999, 7, 9), code=OMOP_BIRTH),
        ],
    )

    expected = femr.Patient(
        patient_id=123,
        events=[
            femr.Event(start=datetime.datetime(1999, 7, 2, 12), code=1234, value=3),
            femr.Event(start=datetime.datetime(1999, 7, 9), code=OMOP_BIRTH),
        ],
    )

    assert remove_nones(patient) == expected


def test_delta_encode() -> None:
    patient = femr.Patient(
        patient_id=123,
        events=[
            femr.Event(start=datetime.datetime(1999, 7, 2), code=1234),
            femr.Event(start=datetime.datetime(1999, 7, 2), code=1234),
            femr.Event(start=datetime.datetime(1999, 7, 2, 12), code=1234, value=3),
            femr.Event(start=datetime.datetime(1999, 7, 2, 14), code=1234, value=3),
            femr.Event(start=datetime.datetime(1999, 7, 2, 19), code=1234, value=5),
            femr.Event(start=datetime.datetime(1999, 7, 2, 20), code=1234, value=3),
        ],
    )

    expected = femr.Patient(
        patient_id=123,
        events=[
            femr.Event(start=datetime.datetime(1999, 7, 2), code=1234),
            femr.Event(start=datetime.datetime(1999, 7, 2, 12), code=1234, value=3),
            femr.Event(start=datetime.datetime(1999, 7, 2, 19), code=1234, value=5),
            femr.Event(start=datetime.datetime(1999, 7, 2, 20), code=1234, value=3),
        ],
    )

    assert delta_encode(patient) == expected


def test_move_billing_codes() -> None:
    patient = femr.Patient(
        patient_id=123,
        events=[
            femr.Event(
                start=datetime.datetime(1999, 7, 2),
                code=1234,
                end=datetime.datetime(1999, 7, 20),
                visit_id=10,
                clarity_table="lpch_pat_enc",
            ),
            femr.Event(
                start=datetime.datetime(1999, 7, 9),
                code=OMOP_BIRTH,
                visit_id=10,
                clarity_table="lpch_pat_enc_dx",
            ),
            femr.Event(
                start=datetime.datetime(1999, 7, 10),
                code=42165,
                visit_id=10,
                clarity_table="shc_pat_enc_dx",
            ),
            femr.Event(
                start=datetime.datetime(1999, 7, 11),
                code=12345,
                visit_id=10,
            ),
            femr.Event(
                start=datetime.datetime(1999, 7, 13),
                code=123,
                visit_id=11,
            ),
        ],
    )

    expected = femr.Patient(
        patient_id=123,
        events=[
            femr.Event(
                start=datetime.datetime(1999, 7, 2),
                code=1234,
                visit_id=10,
                end=datetime.datetime(1999, 7, 20),
                clarity_table="lpch_pat_enc",
            ),
            femr.Event(
                start=datetime.datetime(1999, 7, 11),
                code=12345,
                visit_id=10,
            ),
            femr.Event(
                start=datetime.datetime(1999, 7, 13),
                code=123,
                visit_id=11,
            ),
            femr.Event(
                start=datetime.datetime(1999, 7, 20),
                code=42165,
                visit_id=10,
                clarity_table="shc_pat_enc_dx",
            ),
            femr.Event(
                start=datetime.datetime(1999, 7, 20),
                code=OMOP_BIRTH,
                visit_id=10,
                clarity_table="lpch_pat_enc_dx",
            ),
        ],
    )

    assert move_billing_codes(patient) == expected
