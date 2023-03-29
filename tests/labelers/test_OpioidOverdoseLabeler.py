# flake8: noqa
"""TODO"""
import datetime
import os
import sys
from typing import List, cast

import femr
import femr.datasets
from femr.labelers.core import TimeHorizon
from femr.labelers.omop import OpioidOverdoseLabeler

# Needed to import `tools` for local testing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import EventsWithLabels, event, run_test_for_labeler, run_test_locally

# TODO


def test_OpioidOverdoseLabeler() -> None:
    """Create a OpioidOverdoseLabeler for codes 3 and 6"""
    pass


# Local testing
if __name__ == "__main__":
    run_test_locally("../ignore/test_labelers/", test_OpioidOverdoseLabeler)
