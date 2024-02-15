import datetime
from typing import Set

import femr.models.tasks


class DummyOntology:
    def get_all_parents(self, code: str) -> Set[str]:
        if code == "2":
            return {"2", "2_parent"}
        else:
            return {code}


def test_calculator():
    patient = {
        "patient_id": 100,
        "events": [
            {
                "time": datetime.datetime(1990, 1, 10),
                "measurements": [
                    {"code": "1"},
                ],
            },
            {
                "time": datetime.datetime(1990, 1, 20),
                "measurements": [
                    {"code": "2"},
                ],
            },
            {
                "time": datetime.datetime(1990, 1, 25),
                "measurements": [
                    {"code": "3"},
                ],
            },
            {
                "time": datetime.datetime(1990, 1, 25),
                "measurements": [
                    {"code": "1"},
                ],
            },
        ],
    }

    calculator = femr.models.tasks.SurvivalCalculator(DummyOntology(), patient)

    assert calculator.get_future_events_for_time(datetime.datetime(1990, 1, 1)) == (
        datetime.timedelta(days=24),
        {
            "1": datetime.timedelta(days=9),
            "2": datetime.timedelta(days=19),
            "2_parent": datetime.timedelta(days=19),
            "3": datetime.timedelta(days=24),
        },
    )
    assert calculator.get_future_events_for_time(datetime.datetime(1990, 1, 10)) == (
        datetime.timedelta(days=15),
        {
            "1": datetime.timedelta(days=15),
            "2": datetime.timedelta(days=10),
            "2_parent": datetime.timedelta(days=10),
            "3": datetime.timedelta(days=15),
        },
    )
    assert calculator.get_future_events_for_time(datetime.datetime(1990, 1, 20)) == (
        datetime.timedelta(days=5),
        {"1": datetime.timedelta(days=5), "3": datetime.timedelta(days=5)},
    )
