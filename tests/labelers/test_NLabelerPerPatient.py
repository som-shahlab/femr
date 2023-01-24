# flake8: noqa
"""TODO"""
import datetime
import os
import pathlib
import sys
from typing import List

from piton.labelers.core import NLabelsPerPatientLabeler

# Needed to import `tools` for local testing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import (
    EventsWithLabels,
    event,
    run_test_for_labeler,
    run_test_locally,
)


def test_n_labels_per_patient(tmp_path: pathlib.Path):
    # TODO
    pass


# Local testing
if __name__ == "__main__":
    run_test_locally("../ignore/test_labelers/", test_n_labels_per_patient)
