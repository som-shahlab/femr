from __future__ import annotations

import datetime

import meds

from femr.transforms import delta_encode, remove_nones
from femr.transforms.stanford import (
    move_billing_codes,
    move_pre_birth,
    move_to_day_end,
    move_visit_start_to_first_event_start,
)


def cleanup(patient):
    for event in patient["events"]:
        for measurement in event["measurements"]:
            if "metadata" not in measurement:
                measurement["metadata"] = {}

            for k in ("numeric_value", "text_value", "datetime_value"):
                if k not in measurement:
                    measurement[k] = None

            if "table" not in measurement["metadata"]:
                measurement["metadata"]["table"] = None


def test_pre_birth() -> None:
    patient = {
        "patient_id": 123,
        "events": [
            {"time": datetime.datetime(1999, 7, 2), "measurements": [{"code": 1234}]},
            {"time": datetime.datetime(1999, 7, 9), "measurements": [{"code": meds.birth_code}]},
            {"time": datetime.datetime(1999, 7, 11), "measurements": [{"code": 12345}]},
        ],
    }

    expected = {
        "patient_id": 123,
        "events": [
            {"time": datetime.datetime(1999, 7, 9), "measurements": [{"code": 1234}]},
            {"time": datetime.datetime(1999, 7, 9), "measurements": [{"code": meds.birth_code}]},
            {"time": datetime.datetime(1999, 7, 11), "measurements": [{"code": 12345}]},
        ],
    }

    cleanup(patient)
    cleanup(expected)

    assert move_pre_birth(patient) == expected


def test_move_visit_start_ignores_other_visits() -> None:
    patient = {
        "patient_id": 123,
        "events": [
            {  # A non-visit event with no explicit start time
                "time": datetime.datetime(1999, 7, 2),
                "measurements": [{"code": 1234, "metadata": {"visit_id": 9999}}],
            },
            {  # A visit event with just date specified
                "time": datetime.datetime(1999, 7, 2),
                "measurements": [
                    {
                        "code": 4567,
                        "metadata": {
                            "table": "visit",
                            "visit_id": 9999,
                        },
                    }
                ],
            },
            {  # A non-visit event from a separate visit ID
                "time": datetime.datetime(1999, 7, 2, 11),
                "measurements": [{"code": 2345, "metadata": {"visit_id": 8888}}],
            },
            {  # First recorded non-visit event for visit ID 9999
                "time": datetime.datetime(1999, 7, 2, 12),
                "measurements": [{"code": 3456, "metadata": {"visit_id": 9999}}],
            },
        ],
    }

    # Note that events are implicitly sorted first by start time, then by code:
    # https://github.com/som-shahlab/femr/blob/main/src/femr/__init__.py#L69
    expected = {
        "patient_id": 123,
        "events": [
            {  # A non-visit event with no explicit start time
                "time": datetime.datetime(1999, 7, 2),
                "measurements": [{"code": 1234, "metadata": {"visit_id": 9999}}],
            },
            {  # A non-visit event from a separate visit ID
                "time": datetime.datetime(1999, 7, 2, 11),
                "measurements": [{"code": 2345, "metadata": {"visit_id": 8888}}],
            },
            {  # Now visit event has date and time specified
                "time": datetime.datetime(1999, 7, 2, 12),
                "measurements": [
                    {
                        "code": 4567,
                        "metadata": {
                            "table": "visit",
                            "visit_id": 9999,
                        },
                    }
                ],
            },
            {  # First recorded non-visit event for visit ID 9999
                "time": datetime.datetime(1999, 7, 2, 12),
                "measurements": [{"code": 3456, "metadata": {"visit_id": 9999}}],
            },
        ],
    }

    cleanup(patient)
    cleanup(expected)

    assert move_visit_start_to_first_event_start(patient) == expected


def test_move_visit_start_minute_after_midnight() -> None:
    patient = {
        "patient_id": 123,
        "events": [
            {
                "time": datetime.datetime(1999, 7, 2),
                "measurements": [
                    {"code": 3456, "metadata": {"visit_id": 9999, "table": "visit"}},
                    {"code": 1234, "metadata": {"visit_id": 9999}},
                ],
            },
            {
                "time": datetime.datetime(1999, 7, 2, 0, 1),
                "measurements": [{"code": 2345, "metadata": {"visit_id": 9999}}],
            },
            {
                "time": datetime.datetime(1999, 7, 2, 12),
                "measurements": [{"code": 4567, "metadata": {"visit_id": 9999}}],
            },
        ],
    }

    expected = {
        "patient_id": 123,
        "events": [
            {
                "time": datetime.datetime(1999, 7, 2),
                "measurements": [{"code": 1234, "metadata": {"visit_id": 9999}}],
            },
            {
                "time": datetime.datetime(1999, 7, 2, 0, 1),
                "measurements": [{"code": 3456, "metadata": {"visit_id": 9999, "table": "visit"}}],
            },
            {
                "time": datetime.datetime(1999, 7, 2, 0, 1),
                "measurements": [{"code": 2345, "metadata": {"visit_id": 9999}}],
            },
            {
                "time": datetime.datetime(1999, 7, 2, 12),
                "measurements": [{"code": 4567, "metadata": {"visit_id": 9999}}],
            },
        ],
    }

    cleanup(patient)
    cleanup(expected)

    assert move_visit_start_to_first_event_start(patient) == expected


def test_move_visit_start_doesnt_move_without_event() -> None:
    patient = {
        "patient_id": 123,
        "events": [
            {
                "time": datetime.datetime(1999, 7, 2),
                "measurements": [
                    {"code": 1234, "metadata": {"visit_id": 9999}},
                    {"code": 3456, "metadata": {"visit_id": 9999, "table": "visit"}},
                    {"code": 2345, "metadata": {"visit_id": 9999}},
                ],
            },
        ],
    }

    # None of the non-visit events have start time > '00:00:00' so visit event
    # start time is unchanged, though order changes based on code under resort.
    expected = {
        "patient_id": 123,
        "events": [
            {
                "time": datetime.datetime(1999, 7, 2),
                "measurements": [
                    {"code": 1234, "metadata": {"visit_id": 9999}},
                    {"code": 3456, "metadata": {"visit_id": 9999, "table": "visit"}},
                    {"code": 2345, "metadata": {"visit_id": 9999}},
                ],
            }
        ],
    }

    cleanup(patient)
    cleanup(expected)

    assert move_visit_start_to_first_event_start(patient) == expected


def test_move_to_day_end() -> None:
    patient = {
        "patient_id": 123,
        "events": [
            {"time": datetime.datetime(1999, 7, 2), "measurements": [{"code": 1234}]},
            {"time": datetime.datetime(1999, 7, 2, 12), "measurements": [{"code": 4321}]},
            {"time": datetime.datetime(1999, 7, 9), "measurements": [{"code": meds.birth_code}]},
        ],
    }

    expected = {
        "patient_id": 123,
        "events": [
            {"time": datetime.datetime(1999, 7, 2, 12), "measurements": [{"code": 4321}]},
            {"time": datetime.datetime(1999, 7, 2, 23, 59), "measurements": [{"code": 1234}]},
            {"time": datetime.datetime(1999, 7, 9, 23, 59), "measurements": [{"code": meds.birth_code}]},
        ],
    }

    cleanup(patient)
    cleanup(expected)

    print(move_to_day_end(patient))
    print(expected)

    assert move_to_day_end(patient) == expected


def test_remove_nones() -> None:
    patient = {
        "patient_id": 123,
        "events": [
            {"time": datetime.datetime(1999, 7, 2), "measurements": [{"code": 1234}]},  # No value, to be removed
            {"time": datetime.datetime(1999, 7, 2, 12), "measurements": [{"code": 1234, "numeric_value": 3}]},
            {"time": datetime.datetime(1999, 7, 9), "measurements": [{"code": meds.birth_code}]},
        ],
    }

    expected = {
        "patient_id": 123,
        "events": [
            {"time": datetime.datetime(1999, 7, 2, 12), "measurements": [{"code": 1234, "numeric_value": 3}]},
            {"time": datetime.datetime(1999, 7, 9), "measurements": [{"code": meds.birth_code}]},
        ],
    }

    cleanup(patient)
    cleanup(expected)

    assert remove_nones(patient) == expected


def test_delta_encode() -> None:
    patient = {
        "patient_id": 123,
        "events": [
            {"time": datetime.datetime(1999, 7, 2), "measurements": [{"code": 1234}]},
            {"time": datetime.datetime(1999, 7, 2), "measurements": [{"code": 1234}]},
            {"time": datetime.datetime(1999, 7, 2, 12), "measurements": [{"code": 1234, "numeric_value": 3}]},
            {"time": datetime.datetime(1999, 7, 2, 14), "measurements": [{"code": 1234, "numeric_value": 3}]},
            {"time": datetime.datetime(1999, 7, 2, 19), "measurements": [{"code": 1234, "numeric_value": 5}]},
            {"time": datetime.datetime(1999, 7, 2, 20), "measurements": [{"code": 1234, "numeric_value": 3}]},
        ],
    }

    expected = {
        "patient_id": 123,
        "events": [
            {"time": datetime.datetime(1999, 7, 2), "measurements": [{"code": 1234}]},
            {"time": datetime.datetime(1999, 7, 2, 12), "measurements": [{"code": 1234, "numeric_value": 3}]},
            {"time": datetime.datetime(1999, 7, 2, 19), "measurements": [{"code": 1234, "numeric_value": 5}]},
            {"time": datetime.datetime(1999, 7, 2, 20), "measurements": [{"code": 1234, "numeric_value": 3}]},
        ],
    }

    cleanup(patient)
    cleanup(expected)

    assert delta_encode(patient) == expected


def test_move_billing_codes() -> None:
    patient = {
        "patient_id": 123,
        "events": [
            {
                "time": datetime.datetime(1999, 7, 2),
                "measurements": [
                    {
                        "code": 1234,
                        "metadata": {
                            "end": datetime.datetime(1999, 7, 20),
                            "visit_id": 10,
                            "clarity_table": "lpch_pat_enc",
                        },
                    }
                ],
            },
            {
                "time": datetime.datetime(1999, 7, 9),
                "measurements": [
                    {
                        "code": meds.birth_code,
                        "metadata": {
                            "visit_id": 10,
                            "clarity_table": "lpch_pat_enc_dx",
                        },
                    }
                ],
            },
            {
                "time": datetime.datetime(1999, 7, 10),
                "measurements": [
                    {
                        "code": 42165,
                        "metadata": {
                            "visit_id": 10,
                            "clarity_table": "shc_pat_enc_dx",
                        },
                    }
                ],
            },
            {
                "time": datetime.datetime(1999, 7, 11),
                "measurements": [
                    {
                        "code": 12345,
                        "metadata": {
                            "visit_id": 10,
                        },
                    }
                ],
            },
            {
                "time": datetime.datetime(1999, 7, 13),
                "measurements": [
                    {
                        "code": 123,
                        "metadata": {
                            "visit_id": 11,
                        },
                    }
                ],
            },
        ],
    }

    expected = {
        "patient_id": 123,
        "events": [
            {
                "time": datetime.datetime(1999, 7, 2),
                "measurements": [
                    {
                        "code": 1234,
                        "metadata": {
                            "end": datetime.datetime(1999, 7, 20),
                            "visit_id": 10,
                            "clarity_table": "lpch_pat_enc",
                        },
                    }
                ],
            },
            {
                "time": datetime.datetime(1999, 7, 11),
                "measurements": [
                    {
                        "code": 12345,
                        "metadata": {
                            "visit_id": 10,
                        },
                    }
                ],
            },
            {
                "time": datetime.datetime(1999, 7, 13),
                "measurements": [
                    {
                        "code": 123,
                        "metadata": {
                            "visit_id": 11,
                        },
                    }
                ],
            },
            {
                "time": datetime.datetime(1999, 7, 20),
                "measurements": [
                    {
                        "code": meds.birth_code,
                        "metadata": {
                            "visit_id": 10,
                            "clarity_table": "lpch_pat_enc_dx",
                        },
                    }
                ],
            },
            {
                "time": datetime.datetime(1999, 7, 20),
                "measurements": [
                    {
                        "code": 42165,
                        "metadata": {
                            "visit_id": 10,
                            "clarity_table": "shc_pat_enc_dx",
                        },
                    }
                ],
            },
        ],
    }

    cleanup(patient)
    cleanup(expected)

    print(move_billing_codes(patient))
    print(expected)

    assert move_billing_codes(patient) == expected
