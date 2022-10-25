import numpy as np
import datetime
import os
from typing import List

import piton
import piton.datasets
from piton.labelers.core import LabeledPatients, TimeHorizon
from piton.labelers.labeling_functions import CodeLF, MortalityLF

# 1. Setup dummy patients with code ICDXXX -> List[Patient] which can sub in for PatientDatabase
dummy_events = [
    piton.Event(
        start=datetime.datetime(1995, 1, 3), # FALSE
        code=0,
        value=float(34),
    ),
    piton.Event(
        start=datetime.datetime(2010, 1, 1), # TRUE
        code=1,
        value=memoryview(b"test_value"),
    ),
    piton.Event(
        start=datetime.datetime(2010, 1, 5), # TRUE
        code=2,
        value=None,
    ),
    piton.Event(
        start=datetime.datetime(2010, 6, 5), # FALSE
        code=3,
        value=None,
    ),
    piton.Event(
        start=datetime.datetime(2011, 2, 5), # FALSE
        code=1,
        value=None,
    ),
    piton.Event(
        start=datetime.datetime(2011, 7, 5), # FALSE
        code=2,
        value=None,
    ),
    piton.Event(
        start=datetime.datetime(2012, 10, 5), # FALSE
        code=3,
        value=None,
    ),
    piton.Event(
        start=datetime.datetime(2015, 6, 5), # FALSE
        code=2,
        value=None,
    ),
]
patients: List[piton.Patient] = []
for patient_id in range(5):
    patients.append(piton.Patient(
        patient_id,
        dummy_events,
    ))

# 2. Create a CodeLF for Code #2
time_horizon_6_months = TimeHorizon(datetime.timedelta(days=0),
                                    datetime.timedelta(days=180))
labeler = CodeLF(2, time_horizon_6_months)

# 3. Label patients and check that they're correct
labeled_patients: LabeledPatients = labeler.apply(patients)

## All label values are correct
for patient_id, labels in labeled_patients.items():
    for idx, (label, true_label) in enumerate(zip(labels, [False, True, False, False, True, False, False, True])):
        assert label.value == true_label, f"Patient: {patient_id} #{idx} | {label.value} != {true_label}"
## Check that tuples version is correct
label_tuples = labeled_patients.as_list_of_label_tuples()
for lt in label_tuples:
    patient_id = lt[0]
    label = lt[1]
    assert label in labeled_patients[patient_id]
## Check that numpy version is correct
label_numpy = labeled_patients.as_numpy_arrays()
assert label_numpy[0].shape[0] == \
        label_numpy[1].shape[0] == \
        label_numpy[2].shape[0] == \
        len(label_tuples)

# 4. Save labeler results
path = '../ignore/test_labelers/CodeLF.pkl'
labeled_patients.save_to_file(path)

## Check that file was created
assert os.path.exists(path)

# 5. Read in the output files and check that they're accurate
labeled_patients_new = LabeledPatients.load_from_file(path)
labeled_patients_new_numpy = labeled_patients_new.as_numpy_arrays()

## Check that we successfully saved / loaded file contents
assert labeled_patients_new == labeled_patients
assert labeled_patients_new.as_list_of_label_tuples() == labeled_patients.as_list_of_label_tuples()
assert np.sum(label_numpy[0] != labeled_patients_new_numpy[0]) == 0
assert np.sum(label_numpy[1] != labeled_patients_new_numpy[1]) == 0
assert np.sum(label_numpy[2] != labeled_patients_new_numpy[2]) == 0

# 6. Create `Ontology` stub and run `MortalityPredictor`
class DummyOntology:
    def get_dictionary(self):
       return [ "zero", "one", "two", "Death Type/", "four", ]
ontology = DummyOntology()
time_horizon_4_to_12_months = TimeHorizon(datetime.timedelta(days=120),
                                          datetime.timedelta(days=365))
labeler = MortalityLF(ontology, time_horizon_4_to_12_months)
mortality_labels = labeler.apply(patients)

## Selected the right code
assert labeler.code == 3
## All label values are correct
for patient_id, labels in mortality_labels.items():
    for idx, (label, true_label) in enumerate(zip(labels, [False, True, True, False, False, False, False, False])):
        assert label.value == true_label, f"Patient: {patient_id} #{idx} | {label.value} != {true_label}"
## Check that tuples version is correct
label_tuples = mortality_labels.as_list_of_label_tuples()
for lt in label_tuples:
    patient_id = lt[0]
    label = lt[1]
    assert label in mortality_labels[patient_id]
## Check that numpy version is correct
label_numpy = mortality_labels.as_numpy_arrays()
assert label_numpy[0].shape[0] == \
        label_numpy[1].shape[0] == \
        label_numpy[2].shape[0] == \
        len(label_tuples)