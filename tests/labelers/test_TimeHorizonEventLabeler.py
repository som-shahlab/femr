# flake8: noqa: E402
import datetime
import os
import pathlib
import sys
from typing import List

from piton import Patient
from piton.labelers.core import TimeHorizon, TimeHorizonEventLabeler

# Needed to import `tools` for local testing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import EventsWithLabels, event, run_test_for_labeler, run_test_locally


class DummyLabeler(TimeHorizonEventLabeler):
    """Dummy labeler that returns True if the event's `code` is in `self.outcome_codes`."""

    def __init__(
        self,
        outcome_codes: List[int],
        time_horizon: TimeHorizon,
    ):
        self.outcome_codes: List[int] = outcome_codes
        self.time_horizon: TimeHorizon = time_horizon

    def get_prediction_times(self, patient: Patient) -> List[datetime.datetime]:
        return [e.start for e in patient.events]

    def get_time_horizon(self) -> TimeHorizon:
        return self.time_horizon

    def get_outcome_times(self, patient: Patient) -> List[datetime.datetime]:
        times: List[datetime.datetime] = []
        for e in patient.events:
            if e.code in self.outcome_codes:
                times.append(e.start)
        return times


def test_no_outcomes(tmp_path: pathlib.Path):
    # No outcomes occur in this patient's timeline
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))
    labeler = DummyLabeler([100], time_horizon)
    events_with_labels: EventsWithLabels = [
        # fmt: off
        (event((2015, 1, 3), 2, None), False),
        (event((2015, 1, 3), 1, None), False),
        (event((2015, 1, 3), 3, None), False),
        (event((2015, 10, 5), 1, None), False),
        (event((2018, 1, 3), 2, None), False),
        (event((2018, 3, 3), 1, None), False),
        (event((2018, 5, 3), 2, None), False),
        (event((2018, 5, 3, 11), 1, None), False),
        (event((2018, 5, 4), 1, None), False),
        (event((2018, 12, 4), 1, None), None),
        # fmt: on
    ]
    run_test_for_labeler(labeler, events_with_labels, help_text="test_no_outcomes")


def test_horizon_0_180_days(tmp_path: pathlib.Path):
    # (0, 180) days
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))
    labeler = DummyLabeler([2], time_horizon)
    events_with_labels: EventsWithLabels = [
        # fmt: off
        (event((2015, 1, 3), 2, None), True),
        (event((2015, 1, 3), 1, None), True),
        (event((2015, 1, 3), 3, None), True),
        (event((2015, 10, 5), 1, None), False),
        (event((2018, 1, 3), 2, None), True),
        (event((2018, 3, 3), 1, None), True),
        (event((2018, 5, 3), 2, None), True),
        (event((2018, 5, 3, 11), 1, None), False),
        (event((2018, 5, 4), 1, None), False),
        (event((2018, 12, 4), 1, None), None),
        # fmt: on
    ]
    run_test_for_labeler(labeler, events_with_labels, help_text="test_horizon_0_180_days")


def test_horizon_1_180_days(tmp_path: pathlib.Path):
    # (1, 180) days
    time_horizon = TimeHorizon(datetime.timedelta(days=1), datetime.timedelta(days=180))
    labeler = DummyLabeler([2], time_horizon)
    events_with_labels: EventsWithLabels = [
        # fmt: off
        (event((2015, 1, 3), 2, None), False),
        (event((2015, 1, 3), 1, None), False),
        (event((2015, 1, 3), 3, None), False),
        (event((2015, 10, 5), 1, None), False),
        (event((2018, 1, 3), 2, None), True),
        (event((2018, 3, 3), 1, None), True),
        (event((2018, 5, 3), 2, None), False),
        (event((2018, 5, 3, 11), 1, None), False),
        (event((2018, 5, 4), 1, None), False),
        (event((2018, 12, 4), 1, None), None),
        # fmt: on
    ]
    run_test_for_labeler(labeler, events_with_labels, help_text="test_horizon_1_180_days")


def test_horizon_180_365_days(tmp_path: pathlib.Path):
    # (180, 365) days
    time_horizon = TimeHorizon(datetime.timedelta(days=180), datetime.timedelta(days=365))
    labeler = DummyLabeler([2], time_horizon)
    events_with_labels: EventsWithLabels = [
        # fmt: off
        (event((2000, 1, 3), 2, None), True),
        (event((2000, 10, 5), 2, None), False),
        (event((2002, 1, 5), 2, None), True),
        (event((2002, 3, 1), 1, None), True),
        (event((2002, 4, 5), 3, None), True),
        (event((2002, 4, 12), 1, None), True),
        (event((2002, 12, 5), 2, None), False),
        (event((2002, 12, 10), 1, None), False),
        (event((2004, 1, 10), 2, None), False),
        (event((2008, 1, 10), 2, None), None),
        # fmt: on
    ]
    run_test_for_labeler(labeler, events_with_labels, help_text="test_horizon_180_365_days")


def test_horizon_0_0_days(tmp_path: pathlib.Path):
    # (0, 0) days
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=0))
    labeler = DummyLabeler([2], time_horizon)
    events_with_labels: EventsWithLabels = [
        # fmt: off
        (event((2015, 1, 3), 2, None), True),
        (event((2015, 1, 3), 1, None), True),
        (event((2015, 1, 4), 1, None), False),
        (event((2015, 1, 5), 2, None), True),
        (event((2015, 1, 5, 10), 1, None), False),
        (event((2015, 1, 6), 2, None), True),
        # fmt: on
    ]
    run_test_for_labeler(labeler, events_with_labels, help_text="test_horizon_0_0_days")


def test_horizon_10_10_days(tmp_path: pathlib.Path):
    # (10, 10) days
    time_horizon = TimeHorizon(datetime.timedelta(days=10), datetime.timedelta(days=10))
    labeler = DummyLabeler([2], time_horizon)
    events_with_labels: EventsWithLabels = [
        # fmt: off
        (event((2015, 1, 3), 2, None), False),
        (event((2015, 1, 13), 1, None), True),
        (event((2015, 1, 23), 2, None), True),
        (event((2015, 2, 2), 2, None), False),
        (event((2015, 3, 10), 1, None), True),
        (event((2015, 3, 20), 2, None), False),
        (event((2015, 3, 29), 2, None), None),
        (event((2015, 3, 30), 1, None), None),
        # fmt: on
    ]
    run_test_for_labeler(labeler, events_with_labels, help_text="test_horizon_10_10_days")


def test_horizon_0_1000000_days(tmp_path: pathlib.Path):
    # (0, 1000000) days
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=1000000))
    labeler = DummyLabeler([2], time_horizon)
    events_with_labels: EventsWithLabels = [
        # fmt: off
        (event((2000, 1, 3), 2, None), True),
        (event((2001, 10, 5), 1, None), True),
        (event((2020, 10, 5), 2, None), True),
        (event((2021, 10, 5), 1, None), True),
        (event((2050, 1, 10), 2, None), True),
        (event((2051, 1, 10), 1, None), False),
        (event((5000, 1, 10), 1, None), None),
        # fmt: on
    ]
    run_test_for_labeler(labeler, events_with_labels, help_text="test_horizon_0_1000000_days")


def test_horizon_5_10_hours(tmp_path: pathlib.Path):
    # (5 hours, 10.5 hours)
    time_horizon = TimeHorizon(datetime.timedelta(hours=5), datetime.timedelta(hours=10, minutes=30))
    labeler = DummyLabeler([2], time_horizon)
    events_with_labels: EventsWithLabels = [
        # fmt: off
        (event((2015, 1, 1, 0, 0), 1, None), True),
        (event((2015, 1, 1, 10, 29), 2, None), False),
        (event((2015, 1, 1, 10, 30), 1, None), False),
        (event((2015, 1, 1, 10, 31), 1, None), False),
        #
        (event((2016, 1, 1, 0, 0), 1, None), True),
        (event((2016, 1, 1, 10, 29), 1, None), False),
        (event((2016, 1, 1, 10, 30), 2, None), False),
        (event((2016, 1, 1, 10, 31), 1, None), False),
        #
        (event((2017, 1, 1, 0, 0), 1, None), False),
        (event((2017, 1, 1, 10, 29), 1, None), False),
        (event((2017, 1, 1, 10, 30), 1, None), False),
        (event((2017, 1, 1, 10, 31), 2, None), False),
        #
        (event((2018, 1, 1, 0, 0), 1, None), False),
        (event((2018, 1, 1, 4, 59, 59), 2, None), False),
        (event((2018, 1, 1, 5), 1, None), False),
        #
        (event((2019, 1, 1, 0, 0), 1, None), True),
        (event((2019, 1, 1, 4, 59, 59), 1, None), None),
        (event((2019, 1, 1, 5), 2, None), None),
        # fmt: on
    ]
    run_test_for_labeler(labeler, events_with_labels, help_text="test_horizon_5_10_hours")


def test_horizon_infinite(tmp_path: pathlib.Path):
    # Infinite horizon
    time_horizon = TimeHorizon(
        datetime.timedelta(days=10),
        None,
    )
    labeler = DummyLabeler([2], time_horizon)
    events_with_labels: EventsWithLabels = [
        # fmt: off
        (event((1950, 1, 3), 1, None), True),
        (event((2000, 1, 3), 1, None), True),
        (event((2001, 10, 5), 1, None), True),
        (event((2020, 10, 5), 1, None), True),
        (event((2021, 10, 5), 1, None), True),
        (event((2050, 1, 10), 2, None), True),
        (event((2050, 1, 20), 2, None), False),
        (event((2051, 1, 10), 1, None), False),
        (event((5000, 1, 10), 1, None), False),
        # fmt: on
    ]
    run_test_for_labeler(labeler, events_with_labels, help_text="test_horizon_infinite")


# Local testing
if __name__ == "__main__":
    run_test_locally("../ignore/test_labelers/", test_horizon_0_180_days)
    run_test_locally("../ignore/test_labelers/", test_horizon_1_180_days)
    run_test_locally("../ignore/test_labelers/", test_horizon_180_365_days)
    run_test_locally("../ignore/test_labelers/", test_horizon_0_0_days)
    run_test_locally("../ignore/test_labelers/", test_horizon_10_10_days)
    run_test_locally("../ignore/test_labelers/", test_horizon_0_1000000_days)
    run_test_locally("../ignore/test_labelers/", test_horizon_5_10_hours)
    run_test_locally("../ignore/test_labelers/", test_horizon_infinite)
    run_test_locally("../ignore/test_labelers/", test_no_outcomes)
