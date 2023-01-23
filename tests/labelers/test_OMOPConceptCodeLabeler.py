"""TODO"""
import datetime

import piton.datasets
from piton.labelers.core import LabeledPatients, TimeHorizon
from piton.labelers.omop import CodeLabeler

# Needed to import `tools` for local testing
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import event, run_test_locally, EventsWithLabels, run_test_for_labeler

# TODO

def test_labeler():
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
    run_test_for_labeler(labeler, events_with_labels, help_text="test_horizon_infinite")

# Local testing
if __name__ == '__main__':
    run_test_locally('../ignore/test_labelers/', test_labeler)
