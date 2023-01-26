# flake8: noqa: E402
import datetime
import os
import pathlib
import sys
from typing import List, Optional, Tuple

import pytest

import piton.datasets
from piton.labelers.core import LabeledPatients, TimeHorizon
from piton.labelers.omop import (
    get_inpatient_admission_discharge_times,
    move_datetime_to_end_of_day
)
from piton.labelers.omop_inpatient_admissions import (
    DummyAdmissionDischargeLabeler,
    InpatientLongAdmissionLabeler,
    InpatientMortalityLabeler,
    InpatientReadmissionLabeler,
)

# Needed to import `tools` for local testing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import (
    EventsWithLabels,
    assert_labels_are_accurate,
    create_patients_list,
    event,
    run_test_for_labeler,
    run_test_locally,
)


class DummyOntology_GetInpatients:
    def get_dictionary(self):
        return [
            "zero",
            "Visit/IP",
            "Visit/OP",
            "three",
            "four",
            "five",
        ]

    def get_children(self, *args) -> List[int]:
        return []

def test_get_inpatient_admission_discharge_times(tmp_path: pathlib.Path):
    ontology = DummyOntology_GetInpatients()
    events_with_labels: EventsWithLabels = [
        # fmt: off
        # visit detail <> occurence - IP
        (event((2000, 1, 1), 0, end=datetime.datetime(2000, 1, 2), visit_id=1, omop_table='visit_detail'), True),
        (event((2000, 1, 1), 1, end=datetime.datetime(2000, 1, 2), visit_id=1, omop_table='visit_occurrence'), False),
        # visit detail <> occurence - not IP
        (event((2001, 1, 31), 4, end=datetime.datetime(2001, 1, 31), visit_id=2, omop_table='visit_detail'), False),
        (event((2001, 1, 1), 2, end=datetime.datetime(2001, 1, 2), visit_id=2, omop_table='visit_occurrence'), False),
        # visit detail (no occurence)
        (event((2005, 1, 1), 5, end=datetime.datetime(2005, 1, 2), visit_id=3, omop_table='visit_detail'), False),
        # visit detail <> multiple occurences - IP + not IP
        (event((2010, 1, 1), 4, end=datetime.datetime(2010, 3, 1), visit_id=4, omop_table='visit_detail'), True),
        (event((2009, 1, 1), 1, end=datetime.datetime(2009, 1, 2), visit_id=4, omop_table='visit_occurrence'), False),
        (event((2009, 1, 1), 2, end=datetime.datetime(2009, 1, 2), visit_id=4, omop_table='visit_occurrence'), False),
        # (no visit detail) occurrence - not IP
        (event((2020, 1, 1), 2, end=datetime.datetime(2020, 1, 2), visit_id=5, omop_table='visit_occurrence'), False),
        # (no visit detail) occurrence - IP
        (event((2021, 1, 1), 1, end=datetime.datetime(2021, 1, 2), visit_id=6, omop_table='visit_occurrence'), False),
        # multiple visit details <> occurence - IP
        (event((2022, 3, 30, 23, 59), 0, end=datetime.datetime(2022, 4, 1), visit_id=7, omop_table='visit_detail'), True),
        (event((2023, 3, 30, 23, 59), 0, end=datetime.datetime(2023, 4, 1), visit_id=7, omop_table='visit_detail'), False),
        (event((2022, 1, 1), 1, end=datetime.datetime(2022, 1, 2), visit_id=7, omop_table='visit_occurrence'), False),
        # multiple visit details <> multiple occurences - IP + not IP
        (event((2030, 3, 30, 23, 59), 0, end=datetime.datetime(2030, 4, 1), visit_id=8, omop_table='visit_detail'), True),
        (event((2031, 3, 30, 23, 59), 0, end=datetime.datetime(2031, 4, 1), visit_id=8, omop_table='visit_detail'), False),
        (event((2030, 1, 1), 1, end=datetime.datetime(2030, 1, 2), visit_id=8, omop_table='visit_occurrence'), False),
        (event((2030, 1, 1), 0, end=datetime.datetime(2030, 1, 2), visit_id=8, omop_table='visit_occurrence'), False),
        # fmt: on
    ]
    patient = piton.Patient(0, [ x[0] for x in events_with_labels ])
    results: List[piton.Event] = get_inpatient_admission_discharge_times(patient, ontology)
    assert results == list(zip([ x[0].start for x in events_with_labels if x[1] == True ],
                          [ x[0].end for x in events_with_labels if x[1] == True])), \
        f"Results: {results} | test_get_inpatient_admission_discharge_times"

#############################################
#############################################
#
# Admission Discharge Placeholder Labeler
#
#############################################
#############################################


class DummyAdmissionDischargeOntology:
    def get_dictionary(self):
        return [
            "zero",
            "Visit/IP",
            "two",
            "three",
        ]

    def get_children(self, *args) -> List[int]:
        return []


def _run_test_admission_discharge_placeholder(
    labeler, events_with_labels: EventsWithLabels, help_text: str = ""
):
    # Check Labels match admission start/end times
    true_labels: List[Tuple[datetime.datetime, Optional[bool]]] = [  # type: ignore
        y  # type: ignore
        for x in events_with_labels  # type: ignore
        for y in [(x[0].start, x[1]), (x[0].end, x[1])]  # type: ignore
        if isinstance(x[1], bool) or (x[1] is None) # type: ignore
    ]  # type: ignore
    patients: List[piton.Patient] = create_patients_list(
        10, [x[0] for x in events_with_labels]
    )
    labeled_patients: LabeledPatients = labeler.apply(patients=patients)
    for patient in patients:
        assert_labels_are_accurate(
            labeled_patients,
            patient.patient_id,
            true_labels,
            help_text=help_text,
        )


def test_admission_discharge_placeholder(tmp_path: pathlib.Path):
    ontology = DummyAdmissionDischargeOntology()
    labeler = DummyAdmissionDischargeLabeler(ontology)  # type: ignore
    # Multiple admission/discharges
    events_with_labels: EventsWithLabels = [
        # fmt: off
        (event((2000, 1, 1), 1, end=datetime.datetime(2000, 1, 2), visit_id=1, omop_table='visit_detail'), True),
        (event((2000, 1, 1), 1, end=datetime.datetime(2000, 1, 2), visit_id=1, omop_table='visit_occurrence'), 'skip'),
        (event((2000, 1, 31), 3), "skip"),
        (event((2000, 1, 31), 1, end=datetime.datetime(2000, 1, 31), visit_id=2, omop_table='visit_detail'), True),
        (event((2000, 1, 31), 1, end=datetime.datetime(2000, 1, 31), visit_id=2, omop_table='visit_occurrence'), 'skip'),
        #
        (event((2005, 1, 1), 1, end=datetime.datetime(2005, 1, 2), visit_id=3, omop_table='visit_detail'), True),
        (event((2000, 1, 31), 1, end=datetime.datetime(2000, 1, 31), visit_id=3, omop_table='visit_occurrence'), 'skip'),
        (event((2005, 1, 15), 2), "skip"),
        #
        (event((2010, 1, 1), 1, end=datetime.datetime(2010, 3, 1), visit_id=4, omop_table='visit_detail'), True),
        (event((2000, 1, 31), 1, end=datetime.datetime(2000, 1, 31), visit_id=4, omop_table='visit_occurrence'), 'skip'),
        (event((2010, 3, 10), 0), "skip"),
        (event((2010, 3, 30, 23, 59), 1, end=datetime.datetime(2010, 4, 1), visit_id=5, omop_table='visit_detail'), True),
        (event((2000, 1, 31), 1, end=datetime.datetime(2000, 1, 31), visit_id=5, omop_table='visit_occurrence'), 'skip'),
        (event((2010, 4, 10), 4, visit_id=5), "skip"),
        #
        (event((2015, 1, 1), 1, end=datetime.datetime(2015, 1, 2), visit_id=6, omop_table='visit_detail'), True),
        (event((2000, 1, 31), 1, end=datetime.datetime(2000, 1, 31), visit_id=6, omop_table='visit_occurrence'), 'skip'),
        (event((2015, 1, 10), 0), "skip"),
        (event((2015, 1, 10), 3), "skip"),
        (event((2015, 1, 20), 2), "skip"),
        (event((2015, 3, 1), 1, end=datetime.datetime(2015, 3, 2), visit_id=7, omop_table='visit_detail'), True),
        (event((2000, 1, 31), 1, end=datetime.datetime(2000, 1, 31), visit_id=7, omop_table='visit_occurrence'), 'skip'),
        #
        (event((2020, 1, 1), 1, end=datetime.datetime(2020, 1, 3), visit_id=8, omop_table='visit_detail'), True),
        (event((2000, 1, 31), 1, end=datetime.datetime(2000, 1, 31), visit_id=8, omop_table='visit_occurrence'), 'skip'),
        #
        (event((2020, 1, 10), 1, end=datetime.datetime(2020, 1, 20)), 'skip'),
        #
        (event((2020, 1, 10), 1, end=datetime.datetime(2020, 1, 20), visit_id=9), 'skip'),
        #
        (event((2020, 1, 10), 1, end=datetime.datetime(2020, 1, 20), visit_id=10, omop_table='visit_detail'), 'skip'),
        #
        (event((2020, 1, 10), 1, end=datetime.datetime(2020, 1, 20), visit_id=11, omop_table='visit_occurrence'), 'skip'),
        #
        (event((2020, 1, 10), 1, end=datetime.datetime(2020, 1, 20), visit_id=12, omop_table='visit_occurrence'), 'skip'),
        (event((2020, 1, 10), 1, end=datetime.datetime(2020, 1, 20), visit_id=12, omop_table='visit_detail'), True),
        # fmt: on
    ]
    _run_test_admission_discharge_placeholder(
        labeler,
        events_with_labels,
        help_text="test_admission_discharge_placeholder_multiple",
    )

    # Zero admission/discharges
    events_with_labels = [
        # fmt: off
        (event((2000, 1, 1), 0, end=datetime.datetime(2000, 1, 2)), "skip"),
        (event((2000, 1, 31), 3), "skip"),
        (event((2000, 1, 31), 4, end=datetime.datetime(2000, 1, 31)), "skip"),
        # fmt: on
    ]
    _run_test_admission_discharge_placeholder(
        labeler,
        events_with_labels,
        help_text="test_admission_discharge_placeholder_zero",
    )

    # Overlapping admission/discharges
    events_with_labels = [
        (event((2000, 1, 1), 1, end=datetime.datetime(2000, 1, 30), visit_id=1, omop_table='visit_detail'), True),
        (event((2020, 1, 1), 1, end=datetime.datetime(2020, 1, 30), visit_id=1, omop_table='visit_occurrence'), 'skip'),
        (event((2000, 1, 15), 1, end=datetime.datetime(2000, 2, 10), visit_id=2, omop_table='visit_detail'), True),
        (event((2020, 1, 1), 1, end=datetime.datetime(2020, 1, 30), visit_id=2, omop_table='visit_occurrence'), 'skip'),
        (event((2000, 1, 29), 1, end=datetime.datetime(2000, 2, 4), visit_id=3, omop_table='visit_detail'), True),
        (event((2020, 1, 1), 1, end=datetime.datetime(2020, 1, 30), visit_id=3, omop_table='visit_occurrence'), 'skip'),
    ]
    _run_test_admission_discharge_placeholder(
        labeler,
        events_with_labels,
        help_text="test_admission_discharge_placeholder_overlap",
    )

    # Test fail cases
    with pytest.raises(RuntimeError):
        # Every admission must have an `end` time
        events_with_labels = [
            (event((2000, 1, 1), 1, end=datetime.datetime(2000, 1, 30), visit_id=1, omop_table='visit_detail'), True),
            (event((2020, 1, 1), 1, end=datetime.datetime(2020, 1, 30), visit_id=1, omop_table='visit_occurrence'), 'skip'),
            (event((2000, 1, 15), 1, end=datetime.datetime(2000, 2, 10), visit_id=2, omop_table='visit_detail'), True),
            (event((2020, 1, 1), 1, end=datetime.datetime(2020, 1, 30), visit_id=2, omop_table='visit_occurrence'), 'skip'),
            (event((2000, 1, 29), 1, visit_id=3, omop_table='visit_detail'), True),
            (event((2020, 1, 1), 1, end=datetime.datetime(2020, 1, 30), visit_id=3, omop_table='visit_occurrence'), 'skip'),
        ]
        patient = piton.Patient(0, [x[0] for x in events_with_labels])
        labeler.label(patient)


#############################################
#############################################
#
# Readmission Labeler
#
#############################################
#############################################


class DummyReadmissionOntology:
    def get_dictionary(self):
        return [
            "zero",
            "Visit/IP",
            "two",
            "three",
        ]

    def get_children(self, *args) -> List[int]:
        return []


def test_readmission(tmp_path: pathlib.Path):
    # Test general readmission labeler on 30-day readmission task
    time_horizon = TimeHorizon(
        datetime.timedelta(seconds=1), datetime.timedelta(days=30)
    )
    ontology = DummyReadmissionOntology()
    labeler = InpatientReadmissionLabeler(ontology, time_horizon)  # type: ignore
    events_with_labels: EventsWithLabels = [
        # fmt: off
        (event((2000, 1, 1), 1, end=datetime.datetime(2000, 1, 2), visit_id=1, omop_table='visit_detail'), True),
        (event((2000, 1, 31), 3, visit_id=1), "skip"),
        (event((2000, 1, 31), 1, end=datetime.datetime(2000, 1, 31), visit_id=2, omop_table='visit_detail'), False),
        #
        (event((2005, 1, 1), 1, end=datetime.datetime(2005, 1, 2), visit_id=3, omop_table='visit_detail'), False),
        (event((2005, 1, 15), 2, visit_id=3), "skip"),
        #
        (event((2010, 1, 1), 1, end=datetime.datetime(2010, 3, 1), visit_id=4, omop_table='visit_detail'), True),
        (event((2010, 3, 10), 0, visit_id=4), "skip"),
        (event((2010, 3, 30, 23, 59), 1, end=datetime.datetime(2010, 4, 1), visit_id=5, omop_table='visit_detail'), False),
        (event((2010, 4, 10), 4, visit_id=5), "skip"),
        #
        (event((2015, 1, 1), 1, end=datetime.datetime(2015, 1, 2), visit_id=6, omop_table='visit_detail'), False),
        (event((2015, 1, 10), 0, visit_id=6), "skip"),
        (event((2015, 1, 10), 3, visit_id=6), "skip"),
        (event((2015, 1, 20), 2, visit_id=6), "skip"),
        (event((2015, 3, 1), 1, end=datetime.datetime(2015, 3, 2), visit_id=7, omop_table='visit_detail'), False),
        #
        (event((2020, 1, 1), 1, end=datetime.datetime(2020, 1, 3), visit_id=8, omop_table='visit_detail'), True),
        (event((2020, 1, 10), 1, end=datetime.datetime(2020, 1, 20), visit_id=9, omop_table='visit_detail'), None),
        #
        # visit occurrences for all visit_details
        (event((2020, 1, 10), 1, visit_id=1, omop_table='visit_occurrence'), 'skip'),
        (event((2020, 1, 10), 1, visit_id=2, omop_table='visit_occurrence'), 'skip'),
        (event((2020, 1, 10), 1, visit_id=3, omop_table='visit_occurrence'), 'skip'),
        (event((2020, 1, 10), 1, visit_id=4, omop_table='visit_occurrence'), 'skip'),
        (event((2020, 1, 10), 1, visit_id=5, omop_table='visit_occurrence'), 'skip'),
        (event((2020, 1, 10), 1, visit_id=6, omop_table='visit_occurrence'), 'skip'),
        (event((2020, 1, 10), 1, visit_id=7, omop_table='visit_occurrence'), 'skip'),
        (event((2020, 1, 10), 1, visit_id=8, omop_table='visit_occurrence'), 'skip'),
        (event((2020, 1, 10), 1, visit_id=9, omop_table='visit_occurrence'), 'skip'),
        # fmt: on
    ]
    patient = piton.Patient(0, [x[0] for x in events_with_labels])
    true_outcome_times: List[datetime.datetime] = [
        x[0].start for x in events_with_labels if x[0].omop_table == 'visit_detail'
    ]
    true_prediction_times: List[datetime.datetime] = [
        move_datetime_to_end_of_day(x[0].end) for x in events_with_labels if x[0].omop_table == 'visit_detail'
    ]
    assert labeler.get_time_horizon() == time_horizon
    assert labeler.get_outcome_times(patient) == true_outcome_times
    assert labeler.get_prediction_times(patient) == true_prediction_times
    run_test_for_labeler(
        labeler,
        events_with_labels,
        true_outcome_times=true_outcome_times,
        true_prediction_times=true_prediction_times,
        help_text="test_readmission_general",
    )

    # Test fail cases
    with pytest.raises(RuntimeError) as _:
        # Require that all `visit_detail` events have `end` specified
        events_with_labels = [
            # fmt: off
            (event((2000, 1, 1), 1, end=None, visit_id=7, omop_table='visit_detail'), "skip"),
            (event((2000, 1, 1), 1, end=datetime.datetime(2020, 1, 20), visit_id=7, omop_table='visit_occurrence'), "skip"),
            # fmt: on
        ]
        patient = piton.Patient(0, [x[0] for x in events_with_labels])
        labeler.get_prediction_times(patient)


#############################################
#############################################
#
# Inpatient Mortality Labeler
#
#############################################
#############################################


class DummyMortalityOntology:
    def get_dictionary(self):
        return [
            "zero",
            "Visit/IP",
            "Death Type/OMOP generated",
            "DEATH_CHILD",
            "four",
            "Condition Type/OMOP4822053",
        ]

    def get_children(self, code: int) -> List[int]:
        if code == 2:
            return [3]
        return []


def test_mortality(tmp_path: pathlib.Path):
    ontology = DummyMortalityOntology()
    labeler = InpatientMortalityLabeler(ontology)  # type: ignore
    events_with_labels: EventsWithLabels = [
        # fmt: off
        (event((2000, 1, 1), 1, end=datetime.datetime(2000, 1, 10), visit_id=1, omop_table='visit_detail'), False),  # admission
        (event((2000, 1, 9), 0, visit_id=1), "skip"),
        #
        # NOTE: InpatientMortalityLabeler goes by `visit_id`, not `end`, so this is True
        # (even tho it occurs outside of the admission's dates)
        (event((2001, 1, 1), 1, end=datetime.datetime(2001, 1, 5), visit_id=2, omop_table='visit_detail'), True),  # admission
        (event((2001, 1, 10), 2, visit_id=2), "skip"),  # event
        #
        (event((2002, 1, 30), 1, end=datetime.datetime(2002, 2, 10), visit_id=3, omop_table='visit_detail'), False),  # admission
        (event((2002, 2, 10, 1), 1, visit_id=3), "skip"),
        #
        # NOTE: InpatientMortalityLabeler goes by `visit_id`, not `end`, so this is False
        # b/c the event has no `visit_id`
        (event((2003, 1, 30), 1, end=datetime.datetime(2003, 2, 10), visit_id=4, omop_table='visit_detail'), False),  # admission
        (event((2003, 2, 9), 2, visit_id=40), "skip"),  # event
        #
        (event((2004, 4, 30), 1, end=datetime.datetime(2004, 5, 10), visit_id=5, omop_table='visit_detail'), True),  # admission
        (event((2004, 5, 9), 3, visit_id=5), "skip"),  # event
        #
        (event((2004, 4, 30), 10, end=datetime.datetime(2004, 5, 10), visit_id=6, omop_table='visit_detail'), True),  # admission
        (event((2004, 5, 9), 5, visit_id=6), "skip"),  # event
        #
        (event((2005, 1, 2), 1, end=datetime.datetime(2005, 5, 10), visit_id=7, omop_table='visit_detail'), False),  # admission
        (event((2005, 5, 9), 0, visit_id=70), "skip"),
        (event((2005, 5, 9), 2, visit_id=70), "skip"),  # event
        #
        # NOTE: No censoring since we have the end of the admission
        (event((2006, 1, 2), 1, end=datetime.datetime(2006, 5, 10), visit_id=8, omop_table='visit_detail'), False),
        # 
        # visit occurrences for all visit_details
        (event((2020, 1, 10), 1, visit_id=1, omop_table='visit_occurrence'), 'skip'),
        (event((2020, 1, 10), 1, visit_id=2, omop_table='visit_occurrence'), 'skip'),
        (event((2020, 1, 10), 1, visit_id=3, omop_table='visit_occurrence'), 'skip'),
        (event((2020, 1, 10), 1, visit_id=4, omop_table='visit_occurrence'), 'skip'),
        (event((2020, 1, 10), 1, visit_id=5, omop_table='visit_occurrence'), 'skip'),
        (event((2020, 1, 10), 1, visit_id=6, omop_table='visit_occurrence'), 'skip'),
        (event((2020, 1, 10), 1, visit_id=7, omop_table='visit_occurrence'), 'skip'),
        (event((2020, 1, 10), 1, visit_id=8, omop_table='visit_occurrence'), 'skip'),
        # fmt: on
    ]
    patient = piton.Patient(0, [x[0] for x in events_with_labels])
    assert labeler.outcome_codes == [2, 3, 5]
    run_test_for_labeler(
        labeler, 
        events_with_labels, 
        true_prediction_times=[ move_datetime_to_end_of_day(x[0].start) for x in events_with_labels if isinstance(x[1], bool) ],
        help_text="test_mortality"
    )


#############################################
#############################################
#
# Long Length of Stay Labeler
#
#############################################
#############################################


class DummyLOSOntology:
    def get_dictionary(self):
        return [
            "zero",
            "Visit/IP",
        ]

    def get_children(self, *args) -> List[int]:
        return []


def test_long_admission(tmp_path: pathlib.Path):
    ontology = DummyLOSOntology()
    long_time: datetime.timedelta = datetime.timedelta(days=7)
    labeler = InpatientLongAdmissionLabeler(ontology, long_time)  # type: ignore
    events_with_labels: EventsWithLabels = [
        # fmt: off
        (event((2000, 1, 1), 1, end=datetime.datetime(2000, 1, 10), visit_id=1, omop_table='visit_detail'), True),
        (event((2004, 4, 30), 1, end=datetime.datetime(2004, 5, 10), visit_id=2, omop_table='visit_detail'), True),
        (event((2006, 1, 2), 1, end=datetime.datetime(2006, 1, 5), visit_id=3, omop_table='visit_detail'), False),
        (event((2006, 1, 3), 0), "skip"),
        (event((2008, 1, 1), 1, end=datetime.datetime(2008, 1, 7, 23, 59), visit_id=4, omop_table='visit_detail'), False),
        (event((2010, 1, 1), 1, end=datetime.datetime(2010, 1, 8), visit_id=5, omop_table='visit_detail'), True),
        # 
        # visit occurrences for all visit_details
        (event((2020, 1, 10), 1, visit_id=1, omop_table='visit_occurrence'), 'skip'),
        (event((2020, 1, 10), 1, visit_id=2, omop_table='visit_occurrence'), 'skip'),
        (event((2020, 1, 10), 1, visit_id=3, omop_table='visit_occurrence'), 'skip'),
        (event((2020, 1, 10), 1, visit_id=4, omop_table='visit_occurrence'), 'skip'),
        (event((2020, 1, 10), 1, visit_id=5, omop_table='visit_occurrence'), 'skip'),
        # fmt: on
    ]
    assert labeler.long_time == long_time
    patient = piton.Patient(0, [x[0] for x in events_with_labels])
    true_prediction_times: List[datetime.datetime] = [
        move_datetime_to_end_of_day(x[0].start) for x in events_with_labels if x[0].omop_table == 'visit_detail'
    ]
    run_test_for_labeler(
        labeler, 
        events_with_labels, 
        true_prediction_times=true_prediction_times,
        help_text="test_long_admission"
    )

    # Test fail cases
    with pytest.raises(RuntimeError) as _:
        # Require that all `visit_detail` events have `end` specified
        events_with_labels = [
            # fmt: off
            (event((2000, 1, 1), 1, end=None, visit_id=7, omop_table='visit_detail'), "skip"),
            (event((2000, 1, 1), 1, end=datetime.datetime(2020, 1, 20), visit_id=7, omop_table='visit_occurrence'), "skip"),
            # fmt: on
        ]
        patient = piton.Patient(0, [x[0] for x in events_with_labels])
        labeler.label(patient)


# Local testing
if __name__ == "__main__":
    run_test_locally("../ignore/test_labelers/",test_get_inpatient_admission_discharge_times)
    run_test_locally(
        "../ignore/test_labelers/", test_admission_discharge_placeholder
    )
    run_test_locally("../ignore/test_labelers/", test_readmission)
    run_test_locally("../ignore/test_labelers/", test_mortality)
    run_test_locally("../ignore/test_labelers/", test_long_admission)