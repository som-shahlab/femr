import datetime
import math
import os
import pickle
import pathlib
import random
import contextlib
import io
import zstandard
import csv
from typing import List, cast, Optional, Tuple

import numpy as np
import scipy.sparse

import piton
import piton.datasets
from piton.featurizers.core import ColumnValue, FeaturizerList
from piton.featurizers.featurizers import AgeFeaturizer, CountFeaturizer
from piton.labelers.core import TimeHorizon, LabelingFunction, LabeledPatients
from piton.labelers.omop_labeling_functions import CodeLF


# SHARED_EVENTS = [
#     piton.Event(start=datetime.datetime(1995, 1, 3), code=0, value=34.5),
#     piton.Event(
#         start=datetime.datetime(2010, 1, 1),
#         code=1,
#         value="test_value",
#     ),
#     piton.Event(start=datetime.datetime(2010, 1, 5), code=2, value=1),
#     piton.Event(start=datetime.datetime(2010, 6, 5), code=3, value=None),
#     piton.Event(start=datetime.datetime(2010, 8, 5), code=2, value=None),
#     piton.Event(start=datetime.datetime(2011, 7, 5), code=2, value=None),
#     piton.Event(start=datetime.datetime(2012, 10, 5), code=3, value=None),
#     piton.Event(start=datetime.datetime(2015, 6, 5, 0), code=2, value=None),
#     piton.Event(
#         start=datetime.datetime(2015, 6, 5, 10, 10), code=2, value=None
#     ),
#     piton.Event(start=datetime.datetime(2015, 6, 15, 11), code=3, value=None),
#     piton.Event(start=datetime.datetime(2016, 1, 1), code=2, value=None),
#     piton.Event(
#         start=datetime.datetime(2016, 3, 1, 10, 10, 10), code=4, value=None
#     ),
# ]

# NUM_PATIENTS = 5
# NUM_EVENTS = len(SHARED_EVENTS)


# def create_patients() -> List[piton.Patient]:
#     patients: List[piton.Patient] = []

#     events = SHARED_EVENTS[:NUM_EVENTS]
#     for patient_id in range(NUM_PATIENTS):
#         patients.append(
#             piton.Patient(
#                 patient_id,
#                 events,
#             )
#         )
#     return patients


# PATIENTS = create_patients()

# class DummyOntology:
#     def get_dictionary(self):
#         return [
#             "zero", 
#             "one", 
#             "two", 
#             "three", 
#             "four"
#         ]


dummy_events = [
    piton.Event(start=datetime.datetime(1995, 1, 3), code=0, value=34.5),
    piton.Event(
        start=datetime.datetime(2010, 1, 1),
        code=1,
        value="test_value",
    ),
    piton.Event(start=datetime.datetime(2010, 1, 5), code=2, value=1),
    piton.Event(start=datetime.datetime(2010, 6, 5), code=3, value=None),
    piton.Event(start=datetime.datetime(2010, 8, 5), code=2, value=None),
    piton.Event(start=datetime.datetime(2011, 7, 5), code=2, value=None),
    piton.Event(start=datetime.datetime(2012, 10, 5), code=3, value=None),
    piton.Event(start=datetime.datetime(2015, 6, 5, 0), code=2, value=None),
    piton.Event(
        start=datetime.datetime(2015, 6, 5, 10, 10), code=2, value=None
    ),
    piton.Event(start=datetime.datetime(2015, 6, 15, 11), code=3, value=None),
    piton.Event(start=datetime.datetime(2016, 1, 1), code=2, value=None),
    piton.Event(
        start=datetime.datetime(2016, 3, 1, 10, 10, 10), code=4, value=None
    ),
]

NUM_EVENTS = len(dummy_events)
NUM_PATIENTS = 10

all_events: List[Tuple[int, piton.Event]] = []

for patient_id in range(10, 10 + NUM_PATIENTS):
    all_events.extend((patient_id, event) for event in dummy_events)


def create_events(tmp_path: pathlib.Path) -> piton.datasets.EventCollection:
    events = piton.datasets.EventCollection(os.path.join(tmp_path, "events"))

    # random.shuffle(all_events)

    chunks = 7
    events_per_chunk = (len(all_events) + chunks - 1) // chunks

    for i in range(7):
        with contextlib.closing(events.create_writer()) as writer:
            for patient_id, event in all_events[
                i * events_per_chunk : (i + 1) * events_per_chunk
            ]:
                writer.add_event(patient_id, event)

    return events


def create_patients(tmp_path: pathlib.Path) -> piton.datasets.PatientCollection:
    return create_events(tmp_path).to_patient_collection(
        os.path.join(tmp_path, "patients")
    )


def create_ontology(path_to_ontology_dir: str, concepts: List[str]):
    path_to_concept_file: str = os.path.join(
        path_to_ontology_dir, "concept", "concept.csv.zst"
    )
    os.makedirs(os.path.dirname(path_to_concept_file), exist_ok=True)
    os.makedirs(os.path.join(path_to_ontology_dir + "/concept_relationship/"), exist_ok = True)
    
    concept_map: Dict[str, int] = {}

    with io.TextIOWrapper(
        zstandard.ZstdCompressor(1).stream_writer(
            open(path_to_concept_file, "wb")
        )
    ) as o:
        writer = csv.DictWriter(
            o,
            fieldnames=[
                "concept_id", "concept_name", "domain_id", "vocabulary_id",
                "concept_class_id", "standard_concept", "concept_code", "valid_start_DATE",
                "valid_end_DATE", "invalid_reason", "load_table_id", "load_row_id",
            ],
        )

        writer.writeheader()

        next_code: int = 0
        for i, c in enumerate(concepts):
            code: int = i + next_code
            concept_map[c] = code
            writer.writerow(
                {
                    "concept_id": str(code),
                    "concept_name": c,
                    "domain_id": "Observation",
                    "vocabulary_id": "dummy",
                    "concept_class_id": "Observation",
                    "standard_concept": "",
                    "concept_code": c,
                    "valid_start_DATE": "1970-01-01",
                    "valid_end_DATE": "2099-12-31",
                    "invalid_reason": "",
                    "load_table_id": "custom_mapping",
                    "load_row_id": "",
                }
            )
    return concept_map

class DummyOntology:
    def get_dictionary(self):
        return [
            "zero", 
            "one", 
            "two", 
            "three", 
            "four"
        ]

def create_database(tmp_path) -> None:

    patient_collection = create_patients(tmp_path)
    with patient_collection.reader() as reader:
        all_patients = list(reader)

    path_to_ontology = os.path.join(tmp_path, "ontology")
    concepts = [ str(x) for x in DummyOntology().get_dictionary() ]
    concept_map = create_ontology(path_to_ontology, concepts)
    print(concept_map)

    path_to_database = os.path.join(tmp_path, "target")

    if not os.path.exists(path_to_database):
        os.mkdir(path_to_database)

    patient_collection.to_patient_database(
        path_to_database,
        path_to_ontology,  # concept.csv
        num_threads=2,
    ).close()


def _assert_featurized_patients_structure(
    labeled_patients, featurized_patients, labels_per_patient
):
    assert len(featurized_patients) == 4

    assert featurized_patients[0].dtype == "float32"
    assert featurized_patients[1].dtype == "int64"
    assert featurized_patients[2].dtype == "bool"
    assert featurized_patients[3].dtype == "datetime64[us]"

    assert len(featurized_patients[1]) == NUM_PATIENTS * len(labels_per_patient)

    patient_ids = np.array(
        sorted([i for i in range(NUM_PATIENTS)] * len(labels_per_patient))
    )
    assert (
        np.sum(featurized_patients[1] == patient_ids)
        == NUM_PATIENTS * len(labels_per_patient)
    )

    all_labels = np.array(labels_per_patient * NUM_PATIENTS)
    assert (
        np.sum(featurized_patients[2] == all_labels)
        == NUM_PATIENTS * len(labels_per_patient)
    )

    label_time = labeled_patients.as_numpy_arrays()[2]
    assert (
        np.sum(featurized_patients[3] == label_time)
        == NUM_PATIENTS * len(labels_per_patient)
    )


def create_labeled_patients(labeler: LabelingFunction, patients: List[piton.Patient]):
    pat_to_labels = {}

    for patient in patients:
        labels = labeler.label(patient)

        if len(labels) > 0:
            pat_to_labels[patient.patient_id] = labels

    labeled_patients = LabeledPatients(pat_to_labels, labeler.get_labeler_type())

    return labeled_patients

def _get_piton_codes(ontology, target_code):
    piton_concept_id = f"dummy/{DummyOntology().get_dictionary()[target_code]}"
    piton_target_code = ontology.get_dictionary().index(piton_concept_id)
    return piton_target_code

def test_age_featurizer(tmp_path: pathlib.Path):
    time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=180)
    )

    create_database(tmp_path)

    database_path = os.path.join(tmp_path, "target")
    database = piton.datasets.PatientDatabase(database_path)
    ontology = database.get_ontology()

    piton_target_code = _get_piton_codes(ontology, 2)
    piton_admission_code = _get_piton_codes(ontology, 3)

    labeler = CodeLF(piton_admission_code, piton_target_code, time_horizon=time_horizon)
    labels = labeler.label(database[0])

    featurizer = AgeFeaturizer(is_normalize=False)
    patient_features = featurizer.featurize(database[0], labels, ontology)

    assert patient_features[0] == [ColumnValue(column=0, value=15.43013698630137)]
    assert patient_features[1] == [
        ColumnValue(column=0, value=17.767123287671232)
    ]
    assert patient_features[-1] == [
        ColumnValue(column=0, value=20.46027397260274)
    ]

    labeled_patients = labeler.apply(database_path)

    featurizer = AgeFeaturizer(is_normalize=True)
    featurizer_list = FeaturizerList([featurizer])
    featurizer_list.preprocess_featurizers(database_path, labeled_patients)
    featurized_patients = featurizer_list.featurize(database_path, labeled_patients)

    assert featurized_patients[0].shape == (30, 1)

    labels_per_patient = [True, False, False]
    _assert_featurized_patients_structure(
        labeled_patients, featurized_patients, labels_per_patient
    )


def test_count_featurizer(tmp_path: pathlib.Path):

    time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=180)
    )
    create_database(tmp_path)

    database_path = os.path.join(tmp_path, "target")
    database = piton.datasets.PatientDatabase(database_path)
    ontology = database.get_ontology()

    piton_target_code = _get_piton_codes(ontology, 2)
    piton_admission_code = _get_piton_codes(ontology, 3)

    labeler = CodeLF(piton_admission_code, piton_target_code, time_horizon=time_horizon)
    labels = labeler.label(database[0])

    featurizer = CountFeaturizer()
    featurizer.preprocess(database[0], labels)
    patient_features = featurizer.featurize(database[0], labels, ontology)

    assert featurizer.get_num_columns() == 3

    assert patient_features[0] == [ColumnValue(column=0, value=1)]
    assert patient_features[1] == [ColumnValue(column=0, value=2), ColumnValue(column=1, value=2)]
    assert patient_features[2] == [ColumnValue(column=0, value=3), ColumnValue(column=1, value=4)]

    labeled_patients = labeler.apply(database_path)

    featurizer = CountFeaturizer(is_ontology_expansion=True)
    featurizer_list = FeaturizerList([featurizer])
    featurizer_list.preprocess_featurizers(database_path, labeled_patients)
    featurized_patients = featurizer_list.featurize(database_path, labeled_patients)

    assert featurized_patients[0].shape == (30, 3)

    labels_per_patient = [True, False, False]
    _assert_featurized_patients_structure(
        labeled_patients, featurized_patients, labels_per_patient
    )


def test_count_bins_featurizer(tmp_path: pathlib.Path):

    time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=180)
    )
    create_database(tmp_path)

    database_path = os.path.join(tmp_path, "target")
    database = piton.datasets.PatientDatabase(database_path)
    ontology = database.get_ontology()

    piton_target_code = _get_piton_codes(ontology, 2)
    piton_admission_code = _get_piton_codes(ontology, 3)

    labeler = CodeLF(piton_admission_code, piton_target_code, time_horizon=time_horizon)
    labels = labeler.label(database[0])

    time_bins = [90, 180, math.inf]
    featurizer = CountFeaturizer(is_ontology_expansion=True, time_bins=time_bins)
    featurizer.preprocess(database[0], labels)
    patient_features = featurizer.featurize(database[0], labels, ontology)

    assert featurizer.get_num_columns() == 9
    assert patient_features[0] == [
        ColumnValue(column=0, value=1), 
        ColumnValue(column=4, value=1)
    ]
    assert patient_features[1] == [
        ColumnValue(column=0, value=1), 
        ColumnValue(column=7, value=3),
        ColumnValue(column=6, value=1)
    ]
    assert patient_features[2] == [
        ColumnValue(column=1, value=1)
    ]

    labeled_patients = labeler.apply(database_path)

    featurizer = CountFeaturizer(is_ontology_expansion=True, time_bins=time_bins)
    featurizer_list = FeaturizerList([featurizer])
    featurizer_list.preprocess_featurizers(database_path, labeled_patients)
    featurized_patients = featurizer_list.featurize(database_path, labeled_patients)

    assert featurized_patients[0].shape == (30, 9)

    labels_per_patient = [True, False, False]
    _assert_featurized_patients_structure(
        labeled_patients, featurized_patients, labels_per_patient
    )


def test_complete_featurization(tmp_path: pathlib.Path):
    time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=180)
    )

    create_database(tmp_path)

    database_path = os.path.join(tmp_path, "target")
    database = piton.datasets.PatientDatabase(database_path)
    ontology = database.get_ontology()

    piton_target_code = _get_piton_codes(ontology, 2)
    piton_admission_code = _get_piton_codes(ontology, 3)

    labeler = CodeLF(piton_admission_code, piton_target_code, time_horizon=time_horizon)
    labeled_patients = labeler.apply(database_path)

    age_featurizer = AgeFeaturizer(is_normalize=True)
    age_featurizer_list = FeaturizerList([age_featurizer])
    age_featurizer_list.preprocess_featurizers(database_path, labeled_patients)
    age_featurized_patients = age_featurizer_list.featurize(database_path, labeled_patients)

    time_bins = [90, 180, math.inf]
    count_featurizer = CountFeaturizer(is_ontology_expansion=True, time_bins=time_bins)
    count_featurizer_list = FeaturizerList([count_featurizer])
    count_featurizer_list.preprocess_featurizers(database_path, labeled_patients)
    count_featurized_patients = count_featurizer_list.featurize(database_path, labeled_patients)

    age_featurizer = AgeFeaturizer(is_normalize=True)
    time_bins = [90, 180, math.inf]
    count_featurizer = CountFeaturizer(is_ontology_expansion=True, time_bins=time_bins)
    featurizer_list = FeaturizerList([age_featurizer, count_featurizer])
    featurizer_list.preprocess_featurizers(database_path, labeled_patients)
    featurized_patients = featurizer_list.featurize(database_path, labeled_patients)

    assert featurized_patients[0].shape == (30, 10)

    the_same = (
        featurized_patients[0].toarray()
        == scipy.sparse.hstack(
            (age_featurized_patients[0], count_featurized_patients[0])
        ).toarray()
    )

    assert the_same.all()


def save_to_file(object_to_save, path_to_file: str):
    """Save object to Pickle file."""
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "wb") as fd:
        pickle.dump(object_to_save, fd)


def load_from_file(path_to_file: str):
    """Load object from Pickle file."""
    with open(path_to_file, "rb") as fd:
        result = pickle.load(fd)
    return result


def test_serialization_and_deserialization(tmp_path: pathlib.Path):
    time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=180)
    )

    create_database(tmp_path)

    database_path = os.path.join(tmp_path, "target")
    database = piton.datasets.PatientDatabase(database_path)
    ontology = database.get_ontology()

    piton_target_code = _get_piton_codes(ontology, 2)
    piton_admission_code = _get_piton_codes(ontology, 3)

    labeler = CodeLF(piton_admission_code, piton_target_code, time_horizon=time_horizon)
    labeled_patients = labeler.apply(database_path)

    time_bins = [90, 180, math.inf]
    count_featurizer = CountFeaturizer(is_ontology_expansion=True, time_bins=time_bins)
    count_featurizer_list = FeaturizerList([count_featurizer])
    count_featurizer_list.preprocess_featurizers(database_path, labeled_patients)
    count_featurized_patient = count_featurizer_list.featurize(database_path, labeled_patients)

    save_to_file(count_featurizer_list, os.path.join(tmp_path, "count_featurizer_list.pickle"))
    save_to_file(count_featurized_patient, os.path.join(tmp_path, "count_featurized_patient.pickle"))
        
    # count_featurizer_list_loaded = load_from_file(
    #     "./count_featurizer_list.pickle"
    # )
    count_featurized_patient_loaded = load_from_file(
        os.path.join(tmp_path, "count_featurized_patient.pickle")
    )

    assert (
        count_featurized_patient_loaded[0].toarray()
        == count_featurized_patient[0].toarray()
    ).all()
    assert (
        count_featurized_patient_loaded[1] == count_featurized_patient[1]
    ).all()
    assert (
        count_featurized_patient_loaded[2] == count_featurized_patient[2]
    ).all()
    assert (
        count_featurized_patient_loaded[3] == count_featurized_patient[3]
    ).all()
