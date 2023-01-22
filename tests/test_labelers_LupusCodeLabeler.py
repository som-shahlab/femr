import datetime
from typing import List, cast

import piton
import piton.datasets
from piton.labelers.core import TimeHorizon
from piton.labelers.omop import LupusCodeLabeler
from tools import (
    EventsWithLabels,
    event,
    run_test_for_labeler,
    run_test_locally,
)


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

    # Create `Ontology` stub
    class DummyOntology:
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

    ontology = cast(piton.datasets.Ontology, DummyOntology())

    # Run labeler
    labeler = LupusCodeLabeler(ontology, time_horizon)
    run_test_for_labeler(
        labeler, events_with_labels, help_text="LupusCodeLabeler"
    )

    # Check that we selected the right codes
    assert set(labeler.outcome_codes) == set([3, 6, 7, 9, 10])


# Local testing
if __name__ == "__main__":
    run_test_locally("../ignore/test_labelers/", test_LupusCodeLabeler)
