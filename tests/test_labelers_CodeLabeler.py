import datetime
from typing import List, Tuple, Optional, Union

import piton
import piton.datasets
from piton.labelers.core import Label, LabeledPatients, TimeHorizon
from piton.labelers.omop import CodeLabeler

from tools import event, run_test_locally, assert_labels_are_accurate, create_patients

# 2nd elem of tuple -- 'skip' means no label, None means censored
EventsWithLabels = List[Tuple[piton.Event, Optional[Union[bool, str]]]]

def _run_test(labeler: CodeLabeler, 
              events_with_labels: EventsWithLabels,
              help_text: str = "",) -> None:
    patients: List[piton.Patient] = create_patients(10, 
        [ x[0] for x in events_with_labels ]
    )
    true_labels: List[Optional[bool]] = [ 
        x[1] for x in events_with_labels if isinstance(x[1], bool) or (x[1] is None)
    ]
    labeled_patients: LabeledPatients = labeler.apply(patient_database=patients)
    
    # Check accuracy of Labels
    for i in range(len(patients)):
        assert_labels_are_accurate(labeled_patients, i, true_labels, help_text=help_text)
    
    # Check CodeLabeler's internal functions
    for p in patients:
        assert labeler.get_outcome_times(p) == [
            event.start for event in p.events if event.code in labeler.codes
        ]

def test_prediction_codes():
    # Specify specific event codes at which to make predictions
    time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=10)
    )
    labeler = CodeLabeler([2], time_horizon, prediction_codes=[4, 5])
    events_with_labels: EventsWithLabels = [
        (event((2015, 1, 3), 2, None), 'skip'),
        (event((2015, 1, 3), 4, None), True),
        (event((2015, 1, 3), 1, None), 'skip'),
        (event((2015, 1, 3), 3, None), 'skip'),
        (event((2015, 10, 5), 1, None), 'skip'),
        (event((2018, 1, 3), 2, None), 'skip'),
        (event((2018, 3, 1), 4, None), False),
        (event((2018, 3, 3), 1, None), 'skip'),
        (event((2018, 5, 2), 5, None), True),
        (event((2018, 5, 3), 2, None), 'skip'),
        (event((2018, 5, 4), 4, None), False),
        (event((2018, 5, 3, 11), 1, None), 'skip'),
        (event((2018, 5, 4), 1, None), 'skip'),
        (event((2018, 11, 1), 5, None), False),
        (event((2018, 12, 4), 1, None), 'skip'),
        (event((2018, 12, 30), 4, None), None),
    ]
    _run_test(labeler, events_with_labels, help_text="prediction_codes")

def test_horizon_0_180_days():
    # (0, 180) days
    time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=180)
    )
    labeler = CodeLabeler([2], time_horizon, prediction_codes=None)
    events_with_labels: EventsWithLabels = [
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
    ]
    _run_test(labeler, events_with_labels, help_text="test_horizon_0_180_days")

def test_horizon_1_180_days():
    # (1, 180) days
    time_horizon = TimeHorizon(
        datetime.timedelta(days=1), datetime.timedelta(days=180)
    )
    labeler = CodeLabeler([2], time_horizon, prediction_codes=None)
    events_with_labels: EventsWithLabels = [
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
    ]
    _run_test(labeler, events_with_labels, help_text="test_horizon_1_180_days")

def test_horizon_180_365_days():
    # (180, 365) days
    time_horizon = TimeHorizon(
        datetime.timedelta(days=180), datetime.timedelta(days=365)
    )
    labeler = CodeLabeler([2], time_horizon, prediction_codes=None)
    events_with_labels: EventsWithLabels = [
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
    ]
    _run_test(labeler, events_with_labels, help_text="test_horizon_180_365_days")

def test_horizon_0_0_days():
    # (0, 0) days
    time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=0)
    )
    labeler = CodeLabeler([2], time_horizon, prediction_codes=None)
    events_with_labels: EventsWithLabels = [
        (event((2015, 1, 3), 2, None), True),
        (event((2015, 1, 3), 1, None), True),
        (event((2015, 1, 4), 1, None), False),
        (event((2015, 1, 5), 2, None), True),
        (event((2015, 1, 5, 10), 1, None), False),
        (event((2015, 1, 6), 2, None), True),
    ]
    _run_test(labeler, events_with_labels, help_text="test_horizon_0_0_days")

def test_horizon_10_10_days():
    # (10, 10) days
    time_horizon = TimeHorizon(
        datetime.timedelta(days=10), datetime.timedelta(days=10)
    )
    labeler = CodeLabeler([2], time_horizon, prediction_codes=None)
    events_with_labels: EventsWithLabels = [
        (event((2015, 1, 3), 2, None), False),
        (event((2015, 1, 13), 1, None), True),
        (event((2015, 1, 23), 2, None), True),
        (event((2015, 2, 2), 2, None), False),
        (event((2015, 3, 10), 1, None), True),
        (event((2015, 3, 20), 2, None), False),
        (event((2015, 3, 29), 2, None), None),
        (event((2015, 3, 30), 1, None), None),
    ]
    _run_test(labeler, events_with_labels, help_text="test_horizon_10_10_days")

def test_horizon_0_1000000_days():
    # (0, 1000000) days
    time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=1000000)
    )
    labeler = CodeLabeler([2], time_horizon, prediction_codes=None)
    events_with_labels: EventsWithLabels = [
        (event((2000, 1, 3), 2, None), True),
        (event((2001, 10, 5), 1, None), True),
        (event((2020, 10, 5), 2, None), True),
        (event((2021, 10, 5), 1, None), True),
        (event((2050, 1, 10), 2, None), True),
        (event((2051, 1, 10), 1, None), False),
        (event((5000, 1, 10), 1, None), None),
    ]
    _run_test(labeler, events_with_labels, help_text="test_horizon_0_1000000_days")

def test_horizon_5_10_hours():
    # (5 hours, 10.5 hours)
    time_horizon = TimeHorizon(
        datetime.timedelta(hours=5), datetime.timedelta(hours=10, minutes=30)
    )
    labeler = CodeLabeler([2], time_horizon, prediction_codes=None)
    events_with_labels: EventsWithLabels = [
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
    ]
    _run_test(labeler, events_with_labels, help_text="test_horizon_5_10_hours")

def test_horizon_infinite():
    # Infinite horizon
    time_horizon = TimeHorizon(
        datetime.timedelta(days=10), None,
    )
    labeler = CodeLabeler([2], time_horizon, prediction_codes=None)
    events_with_labels: EventsWithLabels = [
        (event((1950, 1, 3), 1, None), True),
        (event((2000, 1, 3), 1, None), True),
        (event((2001, 10, 5), 1, None), True),
        (event((2020, 10, 5), 1, None), True),
        (event((2021, 10, 5), 1, None), True),
        (event((2050, 1, 10), 2, None), True),
        (event((2050, 1, 20), 2, None), False),
        (event((2051, 1, 10), 1, None), False),
        (event((5000, 1, 10), 1, None), False),
    ]
    _run_test(labeler, events_with_labels, help_text="test_horizon_infinite")

# Local testing
if __name__ == '__main__':
    run_test_locally('../ignore/test_labelers/', test_prediction_codes)
    run_test_locally('../ignore/test_labelers/', test_horizon_0_180_days)
    run_test_locally('../ignore/test_labelers/', test_horizon_1_180_days)
    run_test_locally('../ignore/test_labelers/', test_horizon_180_365_days)
    run_test_locally('../ignore/test_labelers/', test_horizon_0_0_days)
    run_test_locally('../ignore/test_labelers/', test_horizon_10_10_days)
    run_test_locally('../ignore/test_labelers/', test_horizon_0_1000000_days)
    run_test_locally('../ignore/test_labelers/', test_horizon_5_10_hours)
    run_test_locally('../ignore/test_labelers/', test_horizon_infinite)