import numpy as np
import datetime
import os
from typing import List

import piton
import piton.datasets
from piton.labelers.core import LabeledPatients, TimeHorizon
from piton.labelers.labeling_functions import CodeLF, MortalityLF

def create_patients(events: List[piton.Event]) -> List[piton.Patient]:
    patients: List[piton.Patient] = []
    for patient_id in range(5):
        patients.append(piton.Patient(
            patient_id,
            events,
        ))
    return patients

def assert_labels_are_accurate(labeled_patients: LabeledPatients, patient_id: int, true_labels: List[bool]):
    """Passes if the labels in `labeled_patients` for `patient_id` exactly match the labels in `true_labels`
    """    
    for idx, (label, true_label) in enumerate(zip(labeled_patients[patient_id], true_labels)):
        assert label.value == true_label, f"Patient: {patient_id} #{idx} | {label.value} != {true_label}"

def assert_tuples_match_labels(labeled_patients: LabeledPatients):
    """Passes if tuples output by `as_list_of_label_tuples()` are the same as the `labels` in `labeled_patients`
    """    
    label_tuples = labeled_patients.as_list_of_label_tuples()
    for lt in label_tuples:
        patient_id = lt[0]
        label = lt[1]
        assert label in labeled_patients[patient_id]

def assert_np_arrays_match_labels(labeled_patients: LabeledPatients):
    """Passes if np.arrays output by `as_numpy_arrays()` are the same as the `labels` in `labeled_patients`
    """    
    for row in zip(labeled_patients.as_numpy_arrays()):
        print(row) 
    # assert label_numpy[0].shape[0] == \
    #         label_numpy[1].shape[0] == \
    #         label_numpy[2].shape[0] == \
    #         len(label_tuples)

def test_mortality_lf(events: List[piton.Event], true_labels: List[bool]):
    """Creates a MortalityLF for code 3, which corresponds to "Death Type/"
    """
    patients = create_patients(events)

    # 6. Create `Ontology` stub and run `MortalityPredictor`
    class DummyOntology:
        def get_dictionary(self):
            return [ "zero", "one", "two", "Death Type/", "four", ]
    ontology = DummyOntology()
    time_horizon_4_to_12_months = TimeHorizon(datetime.timedelta(days=120), datetime.timedelta(days=365))
    labeler = MortalityLF(ontology, time_horizon_4_to_12_months)
    labeled_patients = labeler.apply(patients)

    ## Selected the right code
    assert labeler.code == 3
    ## All label values are correct
    for patient_id in labeled_patients.keys():
        assert_labels_are_accurate(labeled_patients, patient_id, true_labels)
    ## Check that tuples are correct
    assert_tuples_match_labels(labeled_patients)
    ## Check that numpy are correct
    assert_np_arrays_match_labels(labeled_patients)

def test_code_lf(events: List[piton.Event], true_labels: List[bool]):
    """Creates a CodeLF for code '2' with time horizon of (0,180 days).
    """    
    patients = create_patients(events)
    
    # Create a CodeLF for Code 2
    time_horizon_6_months = TimeHorizon(datetime.timedelta(days=0),
                                        datetime.timedelta(days=180))
    labeler = CodeLF(2, time_horizon_6_months)

    # Check CodeLF's internal functions
    assert labeler.get_time_horizon() == time_horizon_6_months
    for p in patients:
        assert labeler.get_outcome_times(p) == [ event.time for event in p.events if event.code == 2]

    # Label patients and check that they're correct
    labeled_patients: LabeledPatients = labeler.apply(patients)

    # All label values are correct
    for patient_id in labeled_patients.keys():
        assert_labels_are_accurate(labeled_patients, patient_id, true_labels)
    ## Check that tuples are correct
    assert_tuples_match_labels(labeled_patients)
    ## Check that numpy are correct
    assert_np_arrays_match_labels(labeled_patients)

def test_labeler_save_and_load(events: List[piton.Event]):
    patients = create_patients(events)
    horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180)),
    labeler = CodeLF(2, horizon)
    labeled_patients = labeler.apply(patients)
    
    # Save labeler results
    path = '../tmp/test_labelers/CodeLF.pkl'
    labeled_patients.save_to_file(path)

    # Check that file was created
    assert os.path.exists(path)

    # Read in the output files and check that they're accurate
    labeled_patients_new = LabeledPatients.load_from_file(path)

    # Check that we successfully saved / loaded file contents
    assert labeled_patients_new == labeled_patients
    assert labeled_patients_new.as_list_of_label_tuples() == labeled_patients.as_list_of_label_tuples()
    for (orig, new) in zip(labeled_patients.as_numpy_arrays(), labeled_patients_new.as_numpy_arrays()):
        assert np.sum(orig != new) == 0

def test_time_horizons():
    tests = [
        [
            # (0, 180) days
            TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180)),
            create_patients([
                piton.Event(start=datetime.datetime(2015, 1, 3), code=2, value=float(34)),
            ]),
            [False, True, False, False, True, False, False, True],
        ],
        [
            # (1, 180) days
            TimeHorizon(datetime.timedelta(days=1), datetime.timedelta(days=180)),
            create_patients([
                piton.Event(start=datetime.datetime(2015, 1, 3), code=2, value=float(34)),
            ]),
            [],
        ],
        [
            # (180, 365) days
            TimeHorizon(datetime.timedelta(days=180), datetime.timedelta(days=365)),
            create_patients([
                piton.Event(start=datetime.datetime(2015, 1, 3), code=2, value=float(34)),
            ]),
            [],
        ],
        [
            # (0, 0) days
            TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=0)),
            create_patients([
                piton.Event(start=datetime.datetime(2015, 1, 3), code=2, value=float(34)),
            ]),
            [],
        ],
        [
            # (10, 10) days
            TimeHorizon(datetime.timedelta(days=10), datetime.timedelta(days=10)),
            create_patients([
                piton.Event(start=datetime.datetime(2015, 1, 3), code=2, value=float(34)),
            ]),
            [],
        ],
        [
            # (0, 100000) days
            TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=100000)),
            create_patients([
                piton.Event(start=datetime.datetime(2015, 1, 3), code=2, value=float(34)),
            ]),
            [],
        ],
        [
            # (5 hours, 100 days)
            TimeHorizon(datetime.timedelta(hours=5), datetime.timedelta(days=100)),
            create_patients([
                piton.Event(start=datetime.datetime(2015, 1, 3), code=2, value=float(34)),
            ]),
            [],
        ],
        
    ]
    for test in tests:
        horizon, patients, true_labels = test
        labeler = CodeLF(2, horizon)
        labeled_patients = labeler.apply(patients)
        for i in range(len(patients)):
            # all patients have same events
            assert_labels_are_accurate(labeled_patients, i, true_labels)

test_code_lf([
    piton.Event(start=datetime.datetime(1995, 1, 3), code=0, value=float(34)),
    piton.Event(start=datetime.datetime(2010, 1, 1), code=1, value=memoryview(b"test_value")),
    piton.Event(start=datetime.datetime(2010, 1, 5), code=2, value=None),
    piton.Event(start=datetime.datetime(2010, 6, 5), code=3, value=None),
    piton.Event(start=datetime.datetime(2011, 2, 5), code=2, value=None),
    piton.Event(start=datetime.datetime(2011, 7, 5), code=2, value=None),
    piton.Event(start=datetime.datetime(2012, 10, 5), code=3, value=None),
    piton.Event(start=datetime.datetime(2015, 6, 5, 0), code=2, value=None),
    piton.Event(start=datetime.datetime(2015, 6, 5, 10), code=2, value=None),
    piton.Event(start=datetime.datetime(2015, 6, 5, 11), code=2, value=None),
    piton.Event(start=datetime.datetime(2015, 6, 15, 10), code=2, value=None),
    piton.Event(start=datetime.datetime(2015, 6, 15, 11), code=2, value=None),
    piton.Event(start=datetime.datetime(2015, 6, 15, 12), code=2, value=None),
    piton.Event(start=datetime.datetime(2015, 10, 5, 11), code=2, value=None),
    piton.Event(start=datetime.datetime(2015, 10, 5, 11, 1), code=2, value=None),
    piton.Event(start=datetime.datetime(2015, 10, 5, 11, 2), code=2, value=None),
], [])
test_mortality_lf([
    piton.Event(start=datetime.datetime(1995, 1, 3), code=0, value=float(34)),
], [])
test_time_horizons()
test_labeler_save_and_load([
    piton.Event(start=datetime.datetime(1995, 1, 3), code=0, value=float(34)),
    piton.Event(start=datetime.datetime(2010, 1, 1), code=1, value=memoryview(b"test_value")),
    piton.Event(start=datetime.datetime(2010, 1, 5), code=2, value=None),
    piton.Event(start=datetime.datetime(2010, 6, 5), code=3, value=None),
    piton.Event(start=datetime.datetime(2011, 2, 5), code=2, value=None),
    piton.Event(start=datetime.datetime(2011, 7, 5), code=2, value=None),
    piton.Event(start=datetime.datetime(2012, 10, 5), code=3, value=None),
    piton.Event(start=datetime.datetime(2015, 6, 5, 0), code=2, value=None),
    piton.Event(start=datetime.datetime(2015, 6, 5, 10), code=2, value=None),
    piton.Event(start=datetime.datetime(2015, 6, 5, 11), code=2, value=None),
    piton.Event(start=datetime.datetime(2015, 6, 15, 10), code=2, value=None),
    piton.Event(start=datetime.datetime(2015, 6, 15, 11), code=2, value=None),
    piton.Event(start=datetime.datetime(2015, 6, 15, 12), code=2, value=None),
    piton.Event(start=datetime.datetime(2015, 10, 5, 11), code=2, value=None),
    piton.Event(start=datetime.datetime(2015, 10, 5, 11, 1), code=2, value=None),
    piton.Event(start=datetime.datetime(2015, 10, 5, 11, 2), code=2, value=None),
])