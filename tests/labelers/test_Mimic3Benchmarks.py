import datetime
import os
import sys
from typing import List, Optional, Tuple

import piton.datasets
import pytest

import piton
from piton.labelers.core import Label, LabeledPatients, TimeHorizon
from piton.labelers.omop import (
    Harutyunyan_DecompensationLabeler,
    Harutyunyan_LengthOfStayLabeler,
    Harutyunyan_MortalityLabeler,
)

# Needed to import `tools` for local testing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import EventsWithLabels, assert_labels_are_accurate, event, run_test_for_labeler, run_test_locally


class DummyOntology:
    def get_dictionary(self):
        return [
            "zero",
            "CARE_SITE/7928450",
            "CARE_SITE/7928759",
            "Condition Type/OMOP4822053",
        ]

    def get_children(self, *args) -> List[int]:
        return []


def test_Harutyunyan_DecompensationLabeler() -> None:
    """Hourly binary prediction task on whether the patient dies in the next 24 hours."""
    ontology = DummyOntology()
    labeler = Harutyunyan_DecompensationLabeler(ontology)  # type: ignore
    events_with_labels: EventsWithLabels = [
        # fmt: off
        # exclude ICU admissions < 4 hours
        (event((2001, 1, 1, 0), 2, end=datetime.datetime(2001, 1, 1, 3, 59, 59), omop_table='visit_detail'), 'skip'),
        (event((2001, 1, 1, 1), 3), 'skip'),
        # exclude ICU admissions with no events
        (event((2002, 1, 1, 0), 1, end=datetime.datetime(2002, 1, 3, 23), omop_table='visit_detail'), 'skip'),
        (event((2002, 1, 4), 3), 'skip'),
        # ICU admission - true for second half of ICU visit
        (event((2003, 1, 1, 0), 2, end=datetime.datetime(2003, 1, 2, 20), omop_table='visit_detail'), True),
        (event((2003, 1, 2, 20), 3), 'skip'),
        # ICU admission - false for entire ICU visit
        (event((2004, 1, 1, 0), 1, end=datetime.datetime(2004, 1, 2, 20), omop_table='visit_detail'), False),
        (event((2004, 1, 1, 20), 0), 'skip'),
        # fmt: on
    ]
    patient = piton.Patient(0, [x[0] for x in events_with_labels])
    true_labels: List[Tuple[datetime.datetime, bool]] = (
        [(events_with_labels[4][0].start + datetime.timedelta(hours=x), False) for x in range(4, 20)]
        + [(events_with_labels[4][0].start + datetime.timedelta(hours=x), True) for x in range(20, 44)]
        + [(events_with_labels[6][0].start + datetime.timedelta(hours=x), False) for x in range(4, 44)]
    )
    labeled_patients: LabeledPatients = labeler.apply(patients=[patient])

    # Check accuracy of Labels
    assert_labels_are_accurate(
        labeled_patients,
        patient.patient_id,
        true_labels,
        help_text="| test_Harutyunyan_DecompensationLabeler",
    )

    with pytest.raises(RuntimeError):
        events_with_labels = [
            # fmt: off
            # exclude ICU admissions with no length-of-stay (i.e. `event.end is None` )
            (event((2000, 1, 1), 1, end=None, omop_table='visit_detail'), 'skip'),
            (event((2000, 1, 1, 1), 3), 'skip'),
            # fmt: on
        ]
        patient = piton.Patient(0, [x[0] for x in events_with_labels])
        labeler.label(patient)


def test_Harutyunyan_MortalityLabeler() -> None:
    # TODO
    ontology = DummyOntology()
    labeler = Harutyunyan_MortalityLabeler(ontology)  # type: ignore
    events_with_labels: EventsWithLabels = [
        # fmt: off
        # exclude ICU admissions < 48 hours
        (event((2001, 1, 1, 0), 2, end=datetime.datetime(2001, 1, 2, 23, 59, 59), omop_table='visit_detail'), 'skip'),
        (event((2001, 1, 1, 1), 3), 'skip'),
        # exclude ICU admissions with no events before 48 hours
        (event((2002, 1, 1, 0), 1, end=datetime.datetime(2002, 1, 10, 23), omop_table='visit_detail'), 'skip'),
        (event((2002, 1, 6, 0), 3), 'skip'),
        (event((2002, 1, 7, 0), 3), 'skip'),
        # ICU admission - true
        (event((2003, 1, 1, 2), 2, end=datetime.datetime(2003, 1, 5, 20), omop_table='visit_detail'), True),
        (event((2003, 1, 3, 1), 0), 'skip'),
        (event((2003, 1, 5, 20), 3), 'skip'),
        # ICU admission - true
        (event((2004, 1, 1, 0), 1, end=datetime.datetime(2004, 3, 25, 0), omop_table='visit_detail'), False),
        (event((2004, 1, 1, 2), 0), 'skip'),
        (event((2004, 1, 2, 4), 3), 'skip'),
        # ICU admission - false
        (event((2005, 1, 1, 0), 1, end=datetime.datetime(2005, 1, 5, 20), omop_table='visit_detail'), False),
        (event((2005, 1, 1, 0), 0), 'skip'),
        # ICU admission - false (after visit)
        (event((2006, 1, 1, 0), 1, end=datetime.datetime(2006, 1, 5, 20), omop_table='visit_detail'), False),
        (event((2006, 1, 1, 0), 0), 'skip'),
        (event((2006, 1, 6, 0), 3), 'skip'),
        # fmt: on
    ]
    true_prediction_times: List[datetime.datetime] = [
        labeler.visit_start_adjust_func(x[0].start) for x in events_with_labels if isinstance(x[1], bool)
    ]
    true_outcome_times: List[datetime.datetime] = [x[0].start for x in events_with_labels if x[0].code == 3]
    run_test_for_labeler(
        labeler,
        events_with_labels,
        true_outcome_times,
        true_prediction_times,
        help_text="test_Harutyunyan_MortalityLabeler",
    )

    with pytest.raises(RuntimeError):
        events_with_labels = [
            # fmt: off
            # exclude ICU admissions with no length-of-stay (i.e. `event.end is None` )
            (event((2000, 1, 1), 1, end=None, omop_table='visit_detail'), 'skip'),
            (event((2000, 1, 1, 1), 3), 'skip'),
            # fmt: on
        ]
        patient = piton.Patient(0, [x[0] for x in events_with_labels])
        labeler.label(patient)


def test_Harutyunyan_LengthOfStayLabeler() -> None:
    ontology = DummyOntology()
    labeler = Harutyunyan_LengthOfStayLabeler(ontology)  # type: ignore
    events_with_labels: EventsWithLabels = [
        # fmt: off
        # exclude ICU admissions < 4 hours
        (event((2001, 1, 1, 0), 2, end=datetime.datetime(2001, 1, 1, 3, 59, 59), omop_table='visit_detail'), 'skip'),
        (event((2001, 1, 1, 1), 3), 'skip'),
        # exclude ICU admissions with no events
        (event((2002, 1, 1, 0), 1, end=datetime.datetime(2002, 1, 3, 23), omop_table='visit_detail'), 'skip'),
        (event((2002, 1, 4), 3), 'skip'),
        # ICU admission - true for second half of ICU visit
        (event((2003, 1, 1, 0), 2, end=datetime.datetime(2003, 1, 2, 20), omop_table='visit_detail'), True),
        (event((2003, 1, 2, 20), 3), 'skip'),
        # ICU admission - false for entire ICU visit
        (event((2004, 1, 1, 0), 1, end=datetime.datetime(2004, 1, 2, 20), omop_table='visit_detail'), False),
        (event((2004, 1, 1, 20), 0), 'skip'),
        # fmt: on
    ]
    patient = piton.Patient(0, [x[0] for x in events_with_labels])
    true_labels: List[Tuple[datetime.datetime, bool]] = (
        [(events_with_labels[4][0].start + datetime.timedelta(hours=x), False) for x in range(4, 20)]
        + [(events_with_labels[4][0].start + datetime.timedelta(hours=x), True) for x in range(20, 44)]
        + [(events_with_labels[6][0].start + datetime.timedelta(hours=x), False) for x in range(4, 44)]
    )
    labeled_patients: LabeledPatients = labeler.apply(patients=[patient])

    # Check accuracy of Labels
    assert_labels_are_accurate(
        labeled_patients,
        patient.patient_id,
        true_labels,
        help_text="| test_Harutyunyan_LengthOfStayLabeler",
    )

    with pytest.raises(RuntimeError):
        events_with_labels = [
            # fmt: off
            # exclude ICU admissions with no length-of-stay (i.e. `event.end is None` )
            (event((2000, 1, 1), 1, end=None, omop_table='visit_detail'), 'skip'),
            (event((2000, 1, 1, 1), 3), 'skip'),
            # fmt: on
        ]
        patient = piton.Patient(0, [x[0] for x in events_with_labels])
        labeler.label(patient)


# Local testing
if __name__ == "__main__":
    run_test_locally("../ignore/test_labelers/", test_Harutyunyan_DecompensationLabeler)
    run_test_locally("../ignore/test_labelers/", test_Harutyunyan_MortalityLabeler)
    # run_test_locally("../ignore/test_labelers/", test_Harutyunyan_LengthOfStayLabeler)
