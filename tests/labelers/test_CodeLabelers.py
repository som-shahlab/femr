import datetime
import pathlib
from typing import List

import piton.datasets
from piton.labelers.core import TimeHorizon
from piton.labelers.omop import CodeLabeler, MortalityCodeLabeler, LupusCodeLabeler, get_death_concepts

# Needed to import `tools` for local testing
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import (
    EventsWithLabels,
    event,
    run_test_for_labeler,
    run_test_locally,
)


#############################################
# Generic CodeLabeler
#############################################

def test_outcome_codes(tmp_path: pathlib.Path):
    # Specify specific outcomes (and no prediction codes)
    # at which to make predictions
    time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=10)
    )
    labeler = CodeLabeler([2], time_horizon)
    events_with_labels: EventsWithLabels = [
        (event((2015, 1, 3), 2, None), True),
        (event((2015, 1, 3), 4, None), True),
        (event((2015, 1, 3), 1, None), True),
        (event((2015, 1, 3), 3, None), True),
        (event((2015, 10, 5), 1, None), False),
        (event((2018, 1, 3), 2, None), True),
        (event((2018, 3, 1), 4, None), False),
        (event((2018, 3, 3), 1, None), False),
        (event((2018, 5, 2), 5, None), True),
        (event((2018, 5, 3), 2, None), True),
        (event((2018, 5, 3, 11), 1, None), False),
        (event((2018, 5, 4), 1, None), False),
        (event((2018, 5, 4), 4, None), False),
        (event((2018, 11, 1), 5, None), False),
        (event((2018, 12, 4), 1, None), False),
        (event((2018, 12, 30), 4, None), None),
    ]
    run_test_for_labeler(
        labeler, events_with_labels, help_text="test_outcome_codes"
    )

def test_prediction_codes(tmp_path: pathlib.Path):
    # Specify specific event codes at which to make predictions
    time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=10)
    )
    labeler = CodeLabeler([2], time_horizon, prediction_codes=[4, 5])
    events_with_labels: EventsWithLabels = [
        (event((2015, 1, 3), 2, None), "skip"),
        (event((2015, 1, 3), 4, None), True),
        (event((2015, 1, 3), 1, None), "skip"),
        (event((2015, 1, 3), 3, None), "skip"),
        (event((2015, 10, 5), 1, None), "skip"),
        (event((2018, 1, 3), 2, None), "skip"),
        (event((2018, 3, 1), 4, None), False),
        (event((2018, 3, 3), 1, None), "skip"),
        (event((2018, 5, 2), 5, None), True),
        (event((2018, 5, 3), 2, None), "skip"),
        (event((2018, 5, 3, 11), 1, None), "skip"),
        (event((2018, 5, 4), 4, None), False),
        (event((2018, 5, 4), 1, None), "skip"),
        (event((2018, 11, 1), 5, None), False),
        (event((2018, 12, 4), 1, None), "skip"),
        (event((2018, 12, 30), 4, None), None),
    ]
    run_test_for_labeler(
        labeler, events_with_labels, help_text="prediction_codes"
    )


#############################################
# MortalityCodeLabeler
#############################################

class DummyOntology_Mortality:
    def get_dictionary(self):
        return [
            x
            for x in [
                "zero",
                "one",
                "Visit/IP",
                "Condition Type/OMOP4822053",
                "four",
                "five",
                "Death Type/OMOP generated",
            ]
        ]

def test_death_concepts() -> None:
    expected_death_concepts = set(
        [
            "Death Type/OMOP generated",
            "Condition Type/OMOP4822053",
        ]
    )
    assert set(get_death_concepts()) == expected_death_concepts

def test_MortalityCodeLabeler() -> None:
    """Create a MortalityCodeLabeler for codes 3 and 6"""
    time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=180)
    )
    events_with_labels: EventsWithLabels = [
        (event((1995, 1, 3), 0, 34.5), False),
        (event((2000, 1, 1), 1, "test_value"), True),
        (event((2000, 1, 5), 2, 1), True),
        (event((2000, 6, 5), 3, True), True),
        (event((2005, 2, 5), 2, None), False),
        (event((2005, 7, 5), 2, None), False),
        (event((2010, 10, 5), 1, None), False),
        (event((2015, 2, 5, 0), 2, None), False),
        (event((2015, 7, 5, 0), 0, None), True),
        (event((2015, 11, 5, 10, 10), 2, None), True),
        (event((2015, 11, 15, 11), 6, None), True),
        (event((2020, 1, 1), 2, None), None),
        (event((2020, 3, 1, 10, 10, 10), 2, None), None),
    ]

    ontology = DummyOntology_Mortality()

    # Run labeler
    labeler = MortalityCodeLabeler(ontology, time_horizon) # type: ignore
    
    # Check that we selected the right codes
    assert set(labeler.outcome_codes) == { 3, 6 }
    
    run_test_for_labeler(
        labeler, events_with_labels, help_text="MortalityLabeler"
    )


#############################################
# LupusCodeLabeler
#############################################

class DummyOntology_Lupus:
    def get_dictionary(self):
        return [
            "zero",
            "one",
            "Visit/IP",
            "SNOMED/201436003",
            "four",
            "five",
            "SNOMED/55464009",
            "Lupus_child_seven",
            "eight",
            "Lupus_child_nine",
            "Lupus_child_ten",
        ]

    def get_children(self, parent_code: int) -> List[int]:
        if parent_code == 6:
            return [7, 9, 10]
        else:
            return []

def test_LupusCodeLabeler() -> None:
    """Create a LupusCodeLabeler for codes 3 and 6"""
    time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=180)
    )
    events_with_labels: EventsWithLabels = [
        (event((1995, 1, 3), 0, 34.5), False),
        (event((2000, 1, 1), 1, "test_value"), True),
        (event((2000, 1, 5), 2, 1), True),
        (event((2000, 5, 5), 3, None), True),
        (event((2005, 2, 5), 2, None), False),
        (event((2005, 7, 5), 2, None), False),
        (event((2010, 10, 5), 1, None), True),
        (event((2010, 10, 8), 7, None), True),
        (event((2015, 2, 5, 0), 2, None), False),
        (event((2015, 7, 5, 0), 0, None), True),
        (event((2015, 11, 5, 10, 10), 2, None), True),
        (event((2015, 11, 15, 11), 6, None), True),
        (event((2020, 1, 1), 10, None), True),
        (event((2020, 3, 1, 10, 10, 10), 2, None), None),
    ]

    ontology = DummyOntology_Lupus()
    labeler = LupusCodeLabeler(ontology, time_horizon) # type: ignore
    # Check that we selected the right codes
    assert set(labeler.outcome_codes) == set([3, 6, 7, 9, 10])
    
    run_test_for_labeler(
        labeler, events_with_labels, help_text="LupusCodeLabeler"
    )


# Local testing
if __name__ == "__main__":
    run_test_locally("../ignore/test_labelers/", test_prediction_codes)
    run_test_locally("../ignore/test_labelers/", test_outcome_codes)
    run_test_locally("../ignore/test_labelers/", test_MortalityCodeLabeler)
    run_test_locally("../ignore/test_labelers/", test_LupusCodeLabeler)
    run_test_locally("../ignore/test_labelers/", test_death_concepts)