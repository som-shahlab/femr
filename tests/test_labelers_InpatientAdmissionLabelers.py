"""TODO"""
import datetime
import pytest

from typing import List

import piton.datasets
from piton.labelers.core import LabeledPatients, TimeHorizon
from piton.labelers.omop import CodeLabeler
from piton.labelers.omop_inpatient_admissions import (
    InpatientReadmissionLabeler,
    InpatientMortalityLabeler,
    InpatientLongAdmissionLabeler,
    _30DayReadmissionLabeler,
    _1WeekLongLOSLabeler
)

from tools import event, run_test_locally, EventsWithLabels, run_test_for_labeler

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

def test_readmission():
    # Test general readmission labeler on 30-day readmission task
    time_horizon = TimeHorizon(
        datetime.timedelta(seconds=1), datetime.timedelta(days=30)
    )
    ontology = DummyReadmissionOntology()
    labeler = InpatientReadmissionLabeler(ontology, time_horizon) # type: ignore
    events_with_labels: EventsWithLabels = [
        (event((2000, 1, 1), 1, end=datetime.datetime(2000, 1, 2)), True), # admission
        (event((2000, 1, 31), 3), 'skip'),
        (event((2000, 1, 31), 1, end=datetime.datetime(2000, 1, 31)), False), # admission
        #
        (event((2005, 1, 1), 1, end=datetime.datetime(2005, 1, 2)), False), # admission
        (event((2005, 1, 15), 2), 'skip'),
        #
        (event((2010, 1, 1), 1, end=datetime.datetime(2010, 3, 1)), True), # admission
        (event((2010, 3, 10), 0), 'skip'),
        (event((2010, 3, 30, 23, 59), 1, end=datetime.datetime(2010, 4, 1)), False), # admission
        (event((2010, 4, 10), 4), 'skip'),
        #
        (event((2015, 1, 1), 1, end=datetime.datetime(2015, 1, 2)), False), # admission
        (event((2015, 1, 10), 0), 'skip'),
        (event((2015, 1, 10), 3), 'skip'),
        (event((2015, 1, 20), 2), 'skip'),
        (event((2015, 3, 1), 1, end=datetime.datetime(2015, 3, 2)), False), # admission
        #
        (event((2020, 1, 1), 1, end=datetime.datetime(2020, 1, 3)), True), # admission
        (event((2020, 1, 10), 1, end=datetime.datetime(2020, 1, 20)), None), # admission
    ]
    patient = piton.Patient(0, [x[0] for x in events_with_labels])
    true_outcome_times: List[datetime.datetime] = [
        x[0].start for x in events_with_labels if x[0].code in [1]
    ]
    true_prediction_times: List[datetime.datetime] = [
        x[0].end for x in events_with_labels if x[0].code in [1]
    ]
    assert labeler.get_time_horizon() == time_horizon
    assert labeler.get_outcome_times(patient) == true_outcome_times
    assert labeler.get_prediction_times(patient) == true_prediction_times
    run_test_for_labeler(labeler, 
                         events_with_labels, 
                         true_outcome_times=true_outcome_times,
                         help_text="test_readmission_general")

    # Confirm 30-day readmission labeler matches this
    labeler2 = _30DayReadmissionLabeler(ontology) # type: ignore
    assert labeler.get_time_horizon() == labeler2.get_time_horizon()
    assert labeler.get_outcome_times(patient) == labeler2.get_outcome_times(patient)
    assert labeler.get_prediction_times(patient) == labeler2.get_prediction_times(patient)
    run_test_for_labeler(labeler2, 
                         events_with_labels, 
                         true_outcome_times=true_outcome_times,
                         help_text="test_readmission_30_day")
    
    # Test fail cases
    with pytest.raises(ValueError) as _:
        # Require that all events have `end` specified
        events_with_labels = [
            (event((2000, 1, 1), 1, end=None), 'skip'), # admission
        ]
        patient = piton.Patient(0, [x[0] for x in events_with_labels])
        labeler.get_prediction_times(patient)
    
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
    
def test_mortality():
    ontology = DummyMortalityOntology()
    labeler = InpatientMortalityLabeler(ontology) # type: ignore
    events_with_labels: EventsWithLabels = [
        (event((2000, 1, 1), 1, None, 0, end=datetime.datetime(2000, 1, 10)), False), # admission
        (event((2000, 1, 9), 0, None, 0), 'skip'),
        #
        # NOTE: InpatientMortalityLabeler goes by `visit_id`, not `end`, so this is True
        # (even tho it occurs outside of the admission's dates)
        (event((2001, 1, 1), 1, None, 1, end=datetime.datetime(2001, 1, 5)), True), # admission
        (event((2001, 1, 10), 2, None, 1), 'skip'), # event
        #
        (event((2002, 1, 30), 1, None, 2, end=datetime.datetime(2002, 2, 10)), False), # admission
        (event((2002, 2, 10, 1), 1, None, 2), 'skip'),
        #
        # NOTE: InpatientMortalityLabeler goes by `visit_id`, not `end`, so this is False
        # b/c the event has no `visit_id`
        (event((2003, 1, 30), 1, None, 3, end=datetime.datetime(2003, 2, 10)), False), # admission
        (event((2003, 2, 9), 2, None, None), 'skip'), # event
        #
        (event((2004, 4, 30), 1, None, 4, end=datetime.datetime(2004, 5, 10)), True), # admission
        (event((2004, 5, 9), 3, None, 4), 'skip'), # event
        #
        (event((2004, 4, 30), 1, None, 10, end=datetime.datetime(2004, 5, 10)), True), # admission
        (event((2004, 5, 9), 5, None, 10), 'skip'), # event
        #
        (event((2005, 1, 2), 1, None, 5, end=datetime.datetime(2005, 5, 10)), False), # admission
        (event((2005, 5, 9), 0, None, 5), 'skip'),
        (event((2005, 5, 11), 0, None, 5), 'skip'), # event
        #
        # NOTE: No censoring since we have the end of the admission
        (event((2006, 1, 2), 1, None, 6, end=datetime.datetime(2006, 5, 10)), False), # admission
    ]
    patient = piton.Patient(0, [x[0] for x in events_with_labels])
    true_prediction_times: List[datetime.datetime] = [
        x[0].start for x in events_with_labels if x[0].code in [1]
    ]
    assert labeler.outcome_codes == [ 2, 3, 5 ]
    run_test_for_labeler(labeler, 
                         events_with_labels,
                         help_text="test_mortality")
    
    # Test fail cases
    with pytest.raises(RuntimeError) as _:
        # Require that all events with a specific `visit_id` occur AFTER
        # the corresponding visit is started (i.e. no free-floating events)
        events_with_labels = [
            (event((2000, 1, 1, 1), 1, visit_id=1), 'skip'), # admission
            (event((2000, 1, 1), 2, visit_id=2), 'skip'),
            (event((2000, 1, 1, 1), 1, visit_id=2), 'skip'), # admission
        ]
        patient = piton.Patient(0, [x[0] for x in events_with_labels])
        labeler.label(patient)

class DummyLOSOntology:
    def get_dictionary(self):
        return [
            "zero",
            "Visit/IP",
        ]

    def get_children(self, *args) -> List[int]:
        return []

def test_long_admission():
    ontology = DummyLOSOntology()
    long_time: datetime.timedelta = datetime.timedelta(days=7)
    labeler = InpatientLongAdmissionLabeler(ontology, long_time) # type: ignore
    events_with_labels: EventsWithLabels = [
        (event((2000, 1, 1), 1, end=datetime.datetime(2000, 1, 10)), True), # admission
        (event((2004, 4, 30), 1, end=datetime.datetime(2004, 5, 10)), True), # admission
        (event((2006, 1, 2), 1, end=datetime.datetime(2006, 1, 5)), False), # admission
        (event((2006, 1, 3), 0), 'skip'),
        (event((2008, 1, 1), 1, end=datetime.datetime(2008, 1, 7, 23, 59)), False), # admission
        (event((2010, 1, 1), 1, end=datetime.datetime(2010, 1, 8)), True), # admission
    ]
    patient = piton.Patient(0, [x[0] for x in events_with_labels])
    assert labeler.long_time == long_time
    assert labeler.admission_codes == [1]
    run_test_for_labeler(labeler, 
                         events_with_labels,
                         help_text="test_long_admission")

    # Confirm 7-day LOS labeler matches this
    labeler2 = _1WeekLongLOSLabeler(ontology) # type: ignore
    assert labeler.long_time == labeler2.long_time
    assert labeler.admission_codes == labeler2.admission_codes
    run_test_for_labeler(labeler2, 
                         events_with_labels, 
                         help_text="test_long_admission_1_wek")
    
    
    # Test fail cases
    with pytest.raises(RuntimeError) as _:
        # Require that all events have an `end` time
        events_with_labels = [
            (event((2000, 1, 1, 1), 1, visit_id=1), 'skip'), # admission
        ]
        patient = piton.Patient(0, [x[0] for x in events_with_labels])
        labeler.label(patient)

# Local testing
if __name__ == '__main__':
    run_test_locally('../ignore/test_labelers/', test_readmission)
    run_test_locally('../ignore/test_labelers/', test_mortality)
    run_test_locally('../ignore/test_labelers/', test_long_admission)
