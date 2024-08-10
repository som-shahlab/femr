from __future__ import annotations

import datetime

import meds
import meds_reader.transform
from femr_test_tools import DummyEvent, DummyPatient

from femr.transforms import delta_encode, remove_nones
from femr.transforms.stanford import (
    move_billing_codes,
    move_pre_birth,
    move_to_day_end,
    move_visit_start_to_first_event_start,
)


def test_pre_birth() -> None:
    patient = DummyPatient(
        patient_id=123,
        events=[
            DummyEvent(time=datetime.datetime(1999, 7, 2), code="1234"),
            DummyEvent(time=datetime.datetime(1999, 7, 9), code=meds.birth_code),
            DummyEvent(time=datetime.datetime(1999, 7, 11), code="12345"),
        ],
    )

    expected = DummyPatient(
        patient_id=123,
        events=[
            DummyEvent(time=datetime.datetime(1999, 7, 9), code="1234"),
            DummyEvent(time=datetime.datetime(1999, 7, 9), code=meds.birth_code),
            DummyEvent(time=datetime.datetime(1999, 7, 11), code="12345"),
        ],
    )

    assert move_pre_birth(patient) == expected


def test_move_visit_start_ignores_other_visits() -> None:
    patient = DummyPatient(
        patient_id=123,
        events=[
            # A non-visit event with no explicit start time
            DummyEvent(time=datetime.datetime(1999, 7, 2), code="1234", visit_id=9999),
            # A visit event with just date specified
            DummyEvent(
                time=datetime.datetime(1999, 7, 2),
                code="4567",
                visit_id=9999,
                table="visit",
            ),
            # A non-visit event from a separate visit ID
            DummyEvent(
                time=datetime.datetime(1999, 7, 2, 11),
                code="2345",
                visit_id=8888,
            ),
            # First recorded non-visit event for visit ID 9999
            DummyEvent(
                time=datetime.datetime(1999, 7, 2, 12),
                code="3456",
                visit_id=9999,
            ),
        ],
    )

    # Note that events are implicitly sorted first by start time, then by code:
    # https://github.com/som-shahlab/femr/blob/main/src/femr/__init__.py#L69
    expected = DummyPatient(
        patient_id=123,
        events=[
            # A non-visit event with no explicit start time
            DummyEvent(time=datetime.datetime(1999, 7, 2), code="1234", visit_id=9999),
            # A non-visit event from a separate visit ID
            DummyEvent(
                time=datetime.datetime(1999, 7, 2, 11),
                code="2345",
                visit_id=8888,
            ),
            # A visit event with just date specified
            DummyEvent(
                time=datetime.datetime(1999, 7, 2, 12),
                code="4567",
                visit_id=9999,
                table="visit",
            ),
            # First recorded non-visit event for visit ID 9999
            DummyEvent(
                time=datetime.datetime(1999, 7, 2, 12),
                code="3456",
                visit_id=9999,
            ),
        ],
    )

    assert move_visit_start_to_first_event_start(patient) == expected


def test_move_visit_start_minute_after_midnight() -> None:
    patient = DummyPatient(
        patient_id=123,
        events=[
            DummyEvent(time=datetime.datetime(1999, 7, 2), code="3456", visit_id=9999, table="visit"),
            DummyEvent(time=datetime.datetime(1999, 7, 2), code="1234", visit_id=9999),
            DummyEvent(time=datetime.datetime(1999, 7, 2, 0, 1), code="2345", visit_id=9999),
            DummyEvent(time=datetime.datetime(1999, 7, 2, 2, 12), code="4567", visit_id=9999),
        ],
    )

    expected = DummyPatient(
        patient_id=123,
        events=[
            DummyEvent(time=datetime.datetime(1999, 7, 2), code="1234", visit_id=9999),
            DummyEvent(time=datetime.datetime(1999, 7, 2, 0, 1), code="3456", visit_id=9999, table="visit"),
            DummyEvent(time=datetime.datetime(1999, 7, 2, 0, 1), code="2345", visit_id=9999),
            DummyEvent(time=datetime.datetime(1999, 7, 2, 2, 12), code="4567", visit_id=9999),
        ],
    )

    assert move_visit_start_to_first_event_start(patient) == expected


def test_move_visit_start_doesnt_move_without_event() -> None:
    patient = DummyPatient(
        patient_id=123,
        events=[
            DummyEvent(time=datetime.datetime(1999, 7, 2), code="1234", visit_id=9999),
            DummyEvent(time=datetime.datetime(1999, 7, 2), code="3456", visit_id=9999, table="visit"),
            DummyEvent(time=datetime.datetime(1999, 7, 2), code="2345", visit_id=9999),
        ],
    )

    # None of the non-visit events have start time > '00:00:00' so visit event
    # start time is unchanged, though order changes based on code under resort.

    assert move_visit_start_to_first_event_start(patient) == patient


def test_move_to_day_end() -> None:
    patient = meds_reader.transform.MutablePatient(
        patient_id=123,
        events=[
            meds_reader.transform.MutableEvent(time=datetime.datetime(1999, 7, 2), code="1234"),
            meds_reader.transform.MutableEvent(time=datetime.datetime(1999, 7, 2, 12), code="4321"),
            meds_reader.transform.MutableEvent(time=datetime.datetime(1999, 7, 9), code=meds.birth_code),
        ],
    )

    expected = meds_reader.transform.MutablePatient(
        patient_id=123,
        events=[
            meds_reader.transform.MutableEvent(time=datetime.datetime(1999, 7, 2, 12), code="4321"),
            meds_reader.transform.MutableEvent(time=datetime.datetime(1999, 7, 2, 23, 59), code="1234"),
            meds_reader.transform.MutableEvent(time=datetime.datetime(1999, 7, 9, 23, 59), code=meds.birth_code),
        ],
    )

    assert move_to_day_end(patient) == expected


def test_remove_nones() -> None:
    patient = DummyPatient(
        patient_id=123,
        events=[
            DummyEvent(time=datetime.datetime(1999, 7, 2), code="1234"),  # No value, to be removed
            DummyEvent(time=datetime.datetime(1999, 7, 2, 12), code="1234", numeric_value=3),
            DummyEvent(time=datetime.datetime(1999, 7, 9), code=meds.birth_code),
        ],
    )

    expected = DummyPatient(
        patient_id=123,
        events=[
            DummyEvent(time=datetime.datetime(1999, 7, 2, 12), code="1234", numeric_value=3),
            DummyEvent(time=datetime.datetime(1999, 7, 9), code=meds.birth_code),
        ],
    )

    assert remove_nones(patient) == expected


def test_delta_encode() -> None:
    patient = DummyPatient(
        patient_id=123,
        events=[
            DummyEvent(time=datetime.datetime(1999, 7, 2), code="1234"),
            DummyEvent(time=datetime.datetime(1999, 7, 2), code="1234"),
            DummyEvent(time=datetime.datetime(1999, 7, 2, 12), code="1234", numeric_value=3),
            DummyEvent(time=datetime.datetime(1999, 7, 2, 14), code="1234", numeric_value=3),
            DummyEvent(time=datetime.datetime(1999, 7, 2, 19), code="1234", numeric_value=5),
            DummyEvent(time=datetime.datetime(1999, 7, 2, 20), code="1234", numeric_value=3),
        ],
    )

    expected = DummyPatient(
        patient_id=123,
        events=[
            DummyEvent(time=datetime.datetime(1999, 7, 2), code="1234"),
            DummyEvent(time=datetime.datetime(1999, 7, 2, 12), code="1234", numeric_value=3),
            DummyEvent(time=datetime.datetime(1999, 7, 2, 19), code="1234", numeric_value=5),
            DummyEvent(time=datetime.datetime(1999, 7, 2, 20), code="1234", numeric_value=3),
        ],
    )

    assert delta_encode(patient) == expected


def test_move_billing_codes() -> None:
    patient = DummyPatient(
        patient_id=123,
        events=[
            DummyEvent(
                time=datetime.datetime(1999, 7, 2, 0, 0),
                code=1234,
                visit_id=10,
                clarity_table="lpch_pat_enc",
                end=datetime.datetime(1999, 7, 20),
            ),
            DummyEvent(
                time=datetime.datetime(1999, 7, 9, 0, 0),
                code="SNOMED/184099003",
                visit_id=10,
                clarity_table="lpch_pat_enc_dx",
            ),
            DummyEvent(
                time=datetime.datetime(1999, 7, 10, 0, 0), code=42165, visit_id=10, clarity_table="shc_pat_enc_dx"
            ),
            DummyEvent(time=datetime.datetime(1999, 7, 11, 0, 0), code=12345, visit_id=10, clarity_table=None),
            DummyEvent(time=datetime.datetime(1999, 7, 13, 0, 0), code=123, visit_id=11, clarity_table=None),
        ],
    )

    expected = DummyPatient(
        patient_id=123,
        events=[
            DummyEvent(
                time=datetime.datetime(1999, 7, 2, 0, 0),
                code=1234,
                visit_id=10,
                clarity_table="lpch_pat_enc",
                end=datetime.datetime(1999, 7, 20),
            ),
            DummyEvent(time=datetime.datetime(1999, 7, 11, 0, 0), code=12345, visit_id=10, clarity_table=None),
            DummyEvent(time=datetime.datetime(1999, 7, 13, 0, 0), code=123, visit_id=11, clarity_table=None),
            DummyEvent(
                time=datetime.datetime(1999, 7, 20, 0, 0),
                code="SNOMED/184099003",
                visit_id=10,
                clarity_table="lpch_pat_enc_dx",
            ),
            DummyEvent(
                time=datetime.datetime(1999, 7, 20, 0, 0), code=42165, visit_id=10, clarity_table="shc_pat_enc_dx"
            ),
        ],
    )

    assert move_billing_codes(patient) == expected
