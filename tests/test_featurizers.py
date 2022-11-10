import datetime
import os
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
import math

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

NUM_PATIENTS = 5
NUM_EVENTS = len(SHARED_EVENTS)

def create_patients() -> List[piton.Patient]:
    patients: List[piton.Patient] = []

    events = SHARED_EVENTS[:NUM_EVENTS]
    for patient_id in range(NUM_PATIENTS):
        patients.append(
            piton.Patient(
                patient_id,
                events,
            )
        )
    return patients

PATIENTS = create_patients()

class DummyOntology:
    def get_dictionary(self):
        return [
            memoryview("zero".encode("utf8")),
            memoryview("one".encode("utf8")),
            memoryview("two".encode("utf8")),
            memoryview("three".encode("utf8")),
            memoryview("four".encode("utf8")),
        ]


def _assert_featurized_patients_structure(featurized_patients, labels_per_patient):
    assert len(featurized_patients) == 4

    assert featurized_patients[0].dtype == 'float32'
    assert featurized_patients[1].dtype == 'float32'
    assert featurized_patients[2].dtype == 'int32'
    assert featurized_patients[3].dtype == 'datetime64[us]'

    assert len(featurized_patients[1]) == NUM_PATIENTS * NUM_EVENTS

    all_labels = np.array(labels_per_patient * NUM_PATIENTS)
    assert np.sum(featurized_patients[1] == all_labels) == NUM_PATIENTS * NUM_EVENTS

    patient_ids = np.array(sorted([i for i in range(NUM_PATIENTS)] * NUM_EVENTS))
    assert np.sum(featurized_patients[2] == patient_ids) == NUM_PATIENTS * NUM_EVENTS

    label_time = [event.start for patient in PATIENTS for event in patient.events]
    assert np.sum(featurized_patients[3] == label_time) == NUM_PATIENTS * NUM_EVENTS


def test_age_featurizer():
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))
    labeler = CodeLF(2, time_horizon)
    labels = labeler.label(PATIENTS[0])

    featurizer = AgeFeaturizer(normalize=False)
    patient_features = featurizer.featurize(PATIENTS[0], labels)

    assert patient_features[0] == [ColumnValue(column=0, value=0.0)]
    assert patient_features[5] == [ColumnValue(column=0, value=16.512328767123286)]
    assert patient_features[-1] == [ColumnValue(column=0, value=21.172602739726027)]

    featurizer = AgeFeaturizer(normalize=True)
    featurizer_list = FeaturizerList([featurizer])
    featurizer_list.preprocess_featurizers(PATIENTS, labeler)
    featurized_patients = featurizer_list.featurize(PATIENTS, labeler)

    assert featurized_patients[0].shape == (60, 1)

    labels_per_patient = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
    _assert_featurized_patients_structure(featurized_patients, labels_per_patient)


def test_count_featurizer():

    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))
    labeler = CodeLF(2, time_horizon)
    labels = labeler.label(PATIENTS[0])

    featurizer = CountFeaturizer(DummyOntology)
    featurizer.preprocess(PATIENTS[0], labels)
    patient_features = featurizer.featurize(PATIENTS[0], labels)

    assert featurizer.num_columns() == 4

    assert patient_features[0] == [ColumnValue(column=0, value=1)]
    assert patient_features[5] == [
        ColumnValue(column=0, value=1),
        ColumnValue(column=1, value=1),
        ColumnValue(column=2, value=3),
        ColumnValue(column=3, value=1)
    ]
    assert patient_features[-1] == [
        ColumnValue(column=0, value=1),
        ColumnValue(column=1, value=1),
        ColumnValue(column=2, value=7),
        ColumnValue(column=3, value=3)
    ]

    featurizer = CountFeaturizer(DummyOntology)
    featurizer_list = FeaturizerList([featurizer])
    featurizer_list.preprocess_featurizers(PATIENTS, labeler)
    featurized_patients = featurizer_list.featurize(PATIENTS, labeler)

    assert featurized_patients[0].shape == (60, 4)

    labels_per_patient = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
    _assert_featurized_patients_structure(featurized_patients, labels_per_patient)


def test_count_bins_featurizer():
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))
    labeler = CodeLF(2, time_horizon)
    labels = labeler.label(PATIENTS[0])

    time_bins = [90, 180, math.inf]
    featurizer = CountFeaturizer(DummyOntology, time_bins=time_bins)
    featurizer.preprocess(PATIENTS[0], labels)
    patient_features = featurizer.featurize(PATIENTS[0], labels)

    assert featurizer.num_columns() == 12
    assert patient_features[0] == [ColumnValue(column=0, value=1)]
    assert patient_features[6] == [
        ColumnValue(column=3, value=1),
        ColumnValue(column=8, value=1),
        ColumnValue(column=9, value=1),
        ColumnValue(column=10, value=3),
        ColumnValue(column=11, value=1)
    ]
    assert patient_features[-1] == [ColumnValue(column=2, value=2)]

    featurizer = CountFeaturizer(DummyOntology, time_bins=time_bins)
    featurizer_list = FeaturizerList([featurizer])
    featurizer_list.preprocess_featurizers(PATIENTS, labeler)
    featurized_patients = featurizer_list.featurize(PATIENTS, labeler)

    assert featurized_patients[0].shape == (60, 12)

    labels_per_patient = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
    _assert_featurized_patients_structure(featurized_patients, labels_per_patient)


def test_complete_featurization():
    assert 1 == 1


def test_serialization_and_deserialization():
    assert 1 == 1
