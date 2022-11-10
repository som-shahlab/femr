import datetime
import os
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

import piton
import piton.datasets
from piton.labelers.core import Label, LabeledPatients, TimeHorizon
from piton.labelers.omop_labeling_functions import CodeLF, MortalityLF, IsMaleLF
from piton.featurizers.core import Featurizer, FeaturizerList, ColumnValue
from piton.featurizers.featurizers import AgeFeaturizer, CountFeaturizer


SHARED_EVENTS = [
    piton.Event(start=datetime.datetime(1995, 1, 3), code=0),
    piton.Event(start=datetime.datetime(2010, 1, 1), code=1, value=memoryview(b"test_value")),
    piton.Event(start=datetime.datetime(2010, 1, 5), code=2, value=True),
    piton.Event(start=datetime.datetime(2010, 6, 5), code=3, ),
    piton.Event(start=datetime.datetime(2011, 2, 5), code=2, value=True),
    piton.Event(start=datetime.datetime(2011, 7, 5), code=2, value=False),
    piton.Event(start=datetime.datetime(2012, 10, 5), code=3),
    piton.Event(start=datetime.datetime(2015, 6, 5, 0), code=2, value=False),
    piton.Event(start=datetime.datetime(2015, 6, 5, 10, 10), code=2, value=True),
    piton.Event(start=datetime.datetime(2015, 6, 15, 11), code=3),
    piton.Event(start=datetime.datetime(2016, 1, 1), code=2, value=True),
    piton.Event(start=datetime.datetime(2016, 3, 1, 10, 10, 10), code=2, value=False),
]

def create_patients(events: List[piton.Event], num_events:int=5) -> List[piton.Patient]:
    patients: List[piton.Patient] = []

    events = events[:num_events]
    for patient_id in range(NUM_PATIENTS):
        patients.append(
            piton.Patient(
                patient_id,
                events,
            )
        )
    return patients

NUM_PATIENTS = 5
NUM_EVENTS = 5
PATIENTS = create_patients(SHARED_EVENTS, num_events=NUM_EVENTS)


def test_age_featurizer():
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))
    labeler = CodeLF(2, time_horizon)
    labels = labeler.label(PATIENTS[0])

    featurizer = AgeFeaturizer(normalize=False)
    patient_features = featurizer.featurize(PATIENTS[0], labels)


    assert patient_features == [
        [ColumnValue(column=0, value=0.0)], 
        [ColumnValue(column=0, value=15.005479452054795)],
        [ColumnValue(column=0, value=15.016438356164384)],
        [ColumnValue(column=0, value=15.43013698630137)],
        [ColumnValue(column=0, value=16.101369863013698)]
    ]

    featurizer = AgeFeaturizer(normalize=True)
    featurizer_list = FeaturizerList([featurizer])
    featurizer_list.preprocess_featurizers(PATIENTS, labeler)
    featurized_patients = featurizer_list.featurize(PATIENTS, labeler)

    assert len(featurized_patients) == 4

    assert featurized_patients[0].dtype == 'float32'
    assert featurized_patients[1].dtype == 'float32'
    assert featurized_patients[2].dtype == 'int32'
    assert featurized_patients[3].dtype == 'datetime64[us]'

    assert len(featurized_patients[1]) == NUM_PATIENTS * NUM_EVENTS

    labels_per_patient = [0., 1., 1., 0., 1]
    all_labels = np.array(labels_per_patient * NUM_PATIENTS)
    assert np.sum(featurized_patients[1] == all_labels) == NUM_PATIENTS * NUM_EVENTS

    patient_ids = np.array(sorted([i for i in range(NUM_PATIENTS)] * NUM_PATIENTS))
    assert np.sum(featurized_patients[2] == patient_ids) == NUM_PATIENTS * NUM_EVENTS

    label_time = [event.start for patient in PATIENTS for event in patient.events]
    assert np.sum(featurized_patients[3] == label_time) == NUM_PATIENTS * NUM_EVENTS


def test_count_featurizer():
    assert 1 == 1


def test_count_bins_featurizer():
    assert 1 == 1


def test_complete_featurization():
    assert 1 == 1


def test_serialization_and_deserialization():
    assert 1 == 1
