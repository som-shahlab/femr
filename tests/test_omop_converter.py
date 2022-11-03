from __future__ import annotations

import datetime

import piton
import piton.extractors.omop_converter

OMOP_BIRTH = 4216316


def test_pre_birth() -> None:
    patient = piton.Patient(
        patient_id=123,
        events=[
            piton.Event(start=datetime.datetime(1999, 7, 2), code=1234),
            piton.Event(start=datetime.datetime(1999, 7, 9), code=OMOP_BIRTH),
            piton.Event(start=datetime.datetime(1999, 7, 11), code=12345),
        ],
    )

    expected = piton.Patient(
        patient_id=123,
        events=[
            piton.Event(start=datetime.datetime(1999, 7, 9), code=OMOP_BIRTH),
            piton.Event(start=datetime.datetime(1999, 7, 11), code=12345),
        ],
    )

    assert (
        piton.extractors.omop_converter._remove_pre_birth(patient) == expected
    )


def test_remove_small() -> None:
    patient = piton.Patient(
        patient_id=123,
        events=[
            piton.Event(start=datetime.datetime(1999, 7, 2), code=1234),
            piton.Event(start=datetime.datetime(1999, 7, 9), code=OMOP_BIRTH),
            piton.Event(start=datetime.datetime(1999, 7, 11), code=12345),
            piton.Event(start=datetime.datetime(1999, 7, 13), code=12345),
        ],
    )

    invalid = piton.Patient(
        patient_id=123,
        events=[
            piton.Event(start=datetime.datetime(1999, 7, 9), code=OMOP_BIRTH),
            piton.Event(start=datetime.datetime(1999, 7, 11), code=12345),
        ],
    )

    assert (
        piton.extractors.omop_converter._remove_short_patients(patient)
        == patient
    )

    assert (
        piton.extractors.omop_converter._remove_short_patients(invalid) is None
    )


def test_move_to_day_end() -> None:
    patient = piton.Patient(
        patient_id=123,
        events=[
            piton.Event(start=datetime.datetime(1999, 7, 2), code=1234),
            piton.Event(start=datetime.datetime(1999, 7, 2, 12), code=4321),
            piton.Event(start=datetime.datetime(1999, 7, 9), code=OMOP_BIRTH),
        ],
    )

    expected = piton.Patient(
        patient_id=123,
        events=[
            piton.Event(start=datetime.datetime(1999, 7, 2, 12), code=4321),
            piton.Event(
                start=datetime.datetime(1999, 7, 2, 23, 59, 59), code=1234
            ),
            piton.Event(
                start=datetime.datetime(1999, 7, 9, 23, 59, 59), code=OMOP_BIRTH
            ),
        ],
    )

    assert piton.extractors.omop_converter._move_to_day_end(patient) == expected


def test_remove_nones() -> None:
    patient = piton.Patient(
        patient_id=123,
        events=[
            piton.Event(
                start=datetime.datetime(1999, 7, 2), code=1234
            ),  # No value, to be removed
            piton.Event(
                start=datetime.datetime(1999, 7, 2, 12), code=1234, value=3
            ),
            piton.Event(start=datetime.datetime(1999, 7, 9), code=OMOP_BIRTH),
        ],
    )

    expected = piton.Patient(
        patient_id=123,
        events=[
            piton.Event(
                start=datetime.datetime(1999, 7, 2, 12), code=1234, value=3
            ),
            piton.Event(start=datetime.datetime(1999, 7, 9), code=OMOP_BIRTH),
        ],
    )

    assert piton.extractors.omop_converter._remove_nones(patient) == expected


def test_delta_encode() -> None:
    patient = piton.Patient(
        patient_id=123,
        events=[
            piton.Event(start=datetime.datetime(1999, 7, 2), code=1234),
            piton.Event(start=datetime.datetime(1999, 7, 2), code=1234),
            piton.Event(
                start=datetime.datetime(1999, 7, 2, 12), code=1234, value=3
            ),
            piton.Event(
                start=datetime.datetime(1999, 7, 2, 14), code=1234, value=3
            ),
            piton.Event(
                start=datetime.datetime(1999, 7, 2, 19), code=1234, value=5
            ),
            piton.Event(
                start=datetime.datetime(1999, 7, 2, 20), code=1234, value=3
            ),
        ],
    )

    expected = piton.Patient(
        patient_id=123,
        events=[
            piton.Event(start=datetime.datetime(1999, 7, 2), code=1234),
            piton.Event(
                start=datetime.datetime(1999, 7, 2, 12), code=1234, value=3
            ),
            piton.Event(
                start=datetime.datetime(1999, 7, 2, 19), code=1234, value=5
            ),
            piton.Event(
                start=datetime.datetime(1999, 7, 2, 20), code=1234, value=3
            ),
        ],
    )

    assert piton.extractors.omop_converter._delta_encode(patient) == expected


def test_move_billing_codes() -> None:
    patient = piton.Patient(
        patient_id=123,
        events=[
            piton.Event(
                start=datetime.datetime(1999, 7, 2),
                end=datetime.datetime(1999, 7, 20),
                code=1234,
                visit_id=10,
                event_type="lpch_pat_enc",
            ),
            piton.Event(
                start=datetime.datetime(1999, 7, 9),
                code=OMOP_BIRTH,
                visit_id=10,
                event_type="lpch_pat_enc_dx",
            ),
            piton.Event(
                start=datetime.datetime(1999, 7, 10),
                code=42165,
                visit_id=10,
                event_type="shc_pat_enc_dx",
            ),
            piton.Event(
                start=datetime.datetime(1999, 7, 11), code=12345, visit_id=10
            ),
            piton.Event(
                start=datetime.datetime(1999, 7, 13), code=123, visit_id=11
            ),
        ],
    )

    expected = piton.Patient(
        patient_id=123,
        events=[
            piton.Event(
                start=datetime.datetime(1999, 7, 2),
                end=datetime.datetime(1999, 7, 20),
                code=1234,
                visit_id=10,
                event_type="lpch_pat_enc",
            ),
            piton.Event(
                start=datetime.datetime(1999, 7, 11), code=12345, visit_id=10
            ),
            piton.Event(
                start=datetime.datetime(1999, 7, 13), code=123, visit_id=11
            ),
            piton.Event(
                start=datetime.datetime(1999, 7, 20),
                code=42165,
                visit_id=10,
                event_type="shc_pat_enc_dx",
            ),
            piton.Event(
                start=datetime.datetime(1999, 7, 20),
                code=OMOP_BIRTH,
                visit_id=10,
                event_type="lpch_pat_enc_dx",
            ),
        ],
    )

    assert (
        piton.extractors.omop_converter._move_billing_codes(patient) == expected
    )
