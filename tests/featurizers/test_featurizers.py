# flake8: noqa: E402
import datetime
import os
import pathlib
import sys
from typing import cast

import numpy as np
import pytest
import scipy.sparse

import femr
import femr.datasets
from femr.featurizers import ColumnValue, FeaturizerList
from femr.featurizers.featurizers import AgeFeaturizer, CountFeaturizer
from femr.labelers import TimeHorizon
from femr.labelers.omop import CodeLabeler

# Needed to import `tools` for local testing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import create_database, get_femr_code, load_from_pkl, run_test_locally, save_to_pkl  # type: ignore


def _assert_featurized_patients_structure(labeled_patients, featurized_patients, labels_per_patient):
    assert len(featurized_patients) == 4

    assert featurized_patients[0].dtype == "float32"
    assert featurized_patients[1].dtype == "int64"
    assert featurized_patients[2].dtype == "bool"
    assert featurized_patients[3].dtype == "datetime64[us]"

    assert len(featurized_patients[1]) == len(labeled_patients) * len(
        labels_per_patient
    ), f"len(featurized_patients[1]) = {len(featurized_patients[1])}"

    patient_ids = np.array(sorted(list(labeled_patients) * len(labels_per_patient)))
    assert np.sum(featurized_patients[1] == patient_ids) == len(labeled_patients) * len(labels_per_patient)

    all_labels = np.array(labels_per_patient * len(labeled_patients))
    assert np.sum(featurized_patients[2] == all_labels) == len(labeled_patients) * len(labels_per_patient)

    label_time = labeled_patients.as_numpy_arrays()[2]
    assert np.sum(featurized_patients[3] == label_time) == len(labeled_patients) * len(labels_per_patient)


def test_age_featurizer(tmp_path: pathlib.Path):
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))

    create_database(tmp_path)

    database_path = os.path.join(tmp_path, "target")
    database = femr.datasets.PatientDatabase(database_path)
    ontology = database.get_ontology()

    femr_outcome_code = get_femr_code(ontology, 2)
    femr_admission_code = get_femr_code(ontology, 3)

    labeler = CodeLabeler([femr_outcome_code], time_horizon, [femr_admission_code])

    patient: femr.Patient = cast(femr.Patient, database[next(iter(database))])
    labels = labeler.label(patient)
    featurizer = AgeFeaturizer(is_normalize=False)
    patient_features = featurizer.featurize(patient, labels, ontology)

    assert patient_features[0] == [(0, 15.43013698630137)]
    assert patient_features[1] == [(0, 17.767123287671232)]
    assert patient_features[-1] == [(0, 20.46027397260274)]

    labeled_patients = labeler.apply(path_to_patient_database=database_path)

    featurizer = AgeFeaturizer(is_normalize=True)
    featurizer_list = FeaturizerList([featurizer])
    featurizer_list.preprocess_featurizers(database_path, labeled_patients)
    featurized_patients = featurizer_list.featurize(database_path, labeled_patients)

    labels_per_patient = [True, False, False]
    assert featurized_patients[0].shape == (
        len(labeled_patients) * len(labels_per_patient),
        1,
    )
    _assert_featurized_patients_structure(labeled_patients, featurized_patients, labels_per_patient)


def test_count_featurizer(tmp_path: pathlib.Path):
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))
    create_database(tmp_path)

    database_path = os.path.join(tmp_path, "target")
    database = femr.datasets.PatientDatabase(database_path)
    ontology = database.get_ontology()

    femr_outcome_code = get_femr_code(ontology, 2)
    femr_admission_code = get_femr_code(ontology, 3)

    labeler = CodeLabeler([femr_outcome_code], time_horizon, [femr_admission_code])

    patient: femr.Patient = cast(femr.Patient, database[next(iter(database))])
    labels = labeler.label(patient)
    featurizer = CountFeaturizer()
    featurizer.preprocess(patient, labels, ontology)
    patient_features = featurizer.featurize(patient, labels, ontology)

    assert featurizer.get_num_columns() == 3, f"featurizer.get_num_columns() = {featurizer.get_num_columns()}"

    simple_patient_features = [{(featurizer.get_column_name(v.column), v.value) for v in a} for a in patient_features]

    assert simple_patient_features[0] == {("dummy/three", 1)}, f"patient_features[0] = {patient_features[0]}"
    assert simple_patient_features[1] == {
        ("dummy/three", 2),
        ("dummy/two", 2),
    }
    assert simple_patient_features[2] == {
        ("dummy/three", 3),
        ("dummy/two", 4),
    }

    labeled_patients = labeler.apply(path_to_patient_database=database_path)

    featurizer = CountFeaturizer(is_ontology_expansion=True)
    featurizer_list = FeaturizerList([featurizer])
    featurizer_list.preprocess_featurizers(database_path, labeled_patients)
    featurized_patients = featurizer_list.featurize(database_path, labeled_patients)

    labels_per_patient = [True, False, False]
    assert featurized_patients[0].shape == (
        len(labeled_patients) * len(labels_per_patient),
        3,
    )
    _assert_featurized_patients_structure(labeled_patients, featurized_patients, labels_per_patient)


def test_count_featurizer_with_values(tmp_path: pathlib.Path):
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))
    create_database(tmp_path)

    database_path = os.path.join(tmp_path, "target")
    database = femr.datasets.PatientDatabase(database_path)
    ontology = database.get_ontology()

    femr_outcome_code = get_femr_code(ontology, 2)
    femr_admission_code = get_femr_code(ontology, 3)

    labeler = CodeLabeler([femr_outcome_code], time_horizon, [femr_admission_code])

    patient: femr.Patient = cast(femr.Patient, database[next(iter(database))])
    labels = labeler.label(patient)
    featurizer = CountFeaturizer(numeric_value_decile=True, string_value_combination=True)
    featurizer.preprocess(patient, labels, ontology)
    patient_features = featurizer.featurize(patient, labels, ontology)

    assert featurizer.get_num_columns() == 8, f"featurizer.get_num_columns() = {featurizer.get_num_columns()}"

    simple_patient_features = [{(featurizer.get_column_name(v.column), v.value) for v in a} for a in patient_features]

    assert simple_patient_features[0] == {
        ("dummy/three", 1),
        ("dummy/zero [34.5, inf)", 1),
        ("dummy/one test_value", 2),
        ("dummy/one test_value", 2),
        ("dummy/two [1.0, inf)", 1),
        ("dummy/zero [34.5, inf)", 1),
    }, f"patient_features[0] = {patient_features[0]}"
    assert simple_patient_features[1] == {
        ("dummy/one test_value", 2),
        ("dummy/two [1.0, inf)", 1),
        ("dummy/zero [34.5, inf)", 1),
        ("dummy/three", 2),
        ("dummy/two", 2),
    }
    assert simple_patient_features[2] == {
        ("dummy/one test_value", 2),
        ("dummy/two [1.0, inf)", 1),
        ("dummy/zero [34.5, inf)", 1),
        ("dummy/three", 3),
        ("dummy/two", 4),
    }

    labeled_patients = labeler.apply(path_to_patient_database=database_path)

    featurizer = CountFeaturizer(is_ontology_expansion=True)
    featurizer_list = FeaturizerList([featurizer])
    featurizer_list.preprocess_featurizers(database_path, labeled_patients)
    featurized_patients = featurizer_list.featurize(database_path, labeled_patients)

    labels_per_patient = [True, False, False]
    assert featurized_patients[0].shape == (
        len(labeled_patients) * len(labels_per_patient),
        3,
    )
    _assert_featurized_patients_structure(labeled_patients, featurized_patients, labels_per_patient)


def test_count_featurizer_exclude_filter(tmp_path: pathlib.Path):
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))
    create_database(tmp_path)

    database_path = os.path.join(tmp_path, "target")
    database = femr.datasets.PatientDatabase(database_path)
    ontology = database.get_ontology()

    femr_outcome_code = get_femr_code(ontology, 2)
    femr_admission_code = get_femr_code(ontology, 3)

    labeler = CodeLabeler([femr_outcome_code], time_horizon, [femr_admission_code])

    patient: femr.Patient = cast(femr.Patient, database[next(iter(database))])
    labels = labeler.label(patient)

    count_nonempty_columns = lambda lol: len([x for x in lol if x])  # lol -> list of lists

    # Test filtering all codes
    featurizer = CountFeaturizer(excluded_event_filter=lambda _: True)
    featurizer.preprocess(patient, labels, ontology)
    patient_features = featurizer.featurize(patient, labels, ontology)

    assert count_nonempty_columns(patient_features) == 0

    # Test filtering no codes
    featurizer = CountFeaturizer(excluded_event_filter=lambda _: False)
    featurizer.preprocess(patient, labels, ontology)
    patient_features = featurizer.featurize(patient, labels, ontology)

    assert count_nonempty_columns(patient_features) == 3

    # Test filtering single code
    featurizer = CountFeaturizer(excluded_event_filter=lambda e: e.code == "dummy/three")
    featurizer.preprocess(patient, labels, ontology)
    patient_features = featurizer.featurize(patient, labels, ontology)

    assert count_nonempty_columns(patient_features) == 2


def test_count_bins_featurizer(tmp_path: pathlib.Path):
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))
    create_database(tmp_path)

    database_path = os.path.join(tmp_path, "target")
    database = femr.datasets.PatientDatabase(database_path)
    ontology = database.get_ontology()

    femr_outcome_code = get_femr_code(ontology, 2)
    femr_admission_code = get_femr_code(ontology, 3)

    labeler = CodeLabeler([femr_outcome_code], time_horizon, [femr_admission_code])

    patient: femr.Patient = cast(femr.Patient, database[next(iter(database))])
    labels = labeler.label(patient)
    time_bins = [
        datetime.timedelta(days=90),
        datetime.timedelta(days=180),
        datetime.timedelta(weeks=1e4),
    ]
    featurizer = CountFeaturizer(
        time_bins=time_bins,
    )
    featurizer.preprocess(patient, labels, ontology)
    patient_features = featurizer.featurize(patient, labels, ontology)

    assert set(featurizer.code_to_column_index.keys()) == {
        "dummy/four",
        "dummy/three",
        "dummy/two",
    }, f"featurizer.code_to_column_index = {featurizer.code_to_column_index}"
    assert featurizer.get_num_columns() == 9, f"featurizer.get_num_columns() = {featurizer.get_num_columns()}"

    simple_patient_features = [{(featurizer.get_column_name(v.column), v.value) for v in a} for a in patient_features]

    assert simple_patient_features[0] == {
        ("dummy/three_90 days, 0:00:00", 1),
    }, f"patient_features[0] = {patient_features[0]}"
    assert simple_patient_features[1] == {
        ("dummy/three_90 days, 0:00:00", 1),
        ("dummy/two_70000 days, 0:00:00", 2),
        ("dummy/three_70000 days, 0:00:00", 1),
    }, f"patient_features[1] = {patient_features[1]}"
    assert simple_patient_features[2] == {
        ("dummy/three_70000 days, 0:00:00", 2),
        ("dummy/two_90 days, 0:00:00", 2),
        ("dummy/three_90 days, 0:00:00", 1),
        ("dummy/two_70000 days, 0:00:00", 2),
    }, f"patient_features[2] = {patient_features[2]}"

    labeled_patients = labeler.apply(path_to_patient_database=database_path)

    featurizer = CountFeaturizer(is_ontology_expansion=True, time_bins=time_bins)
    featurizer_list = FeaturizerList([featurizer])
    featurizer_list.preprocess_featurizers(database_path, labeled_patients)
    featurized_patients = featurizer_list.featurize(database_path, labeled_patients)

    labels_per_patient = [True, False, False]
    assert featurized_patients[0].shape == (
        len(labeled_patients) * len(labels_per_patient),
        9,
    ), f"featurized_patients[0].shape = {featurized_patients[0].shape}"
    _assert_featurized_patients_structure(labeled_patients, featurized_patients, labels_per_patient)


def test_complete_featurization(tmp_path: pathlib.Path):
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))

    create_database(tmp_path)

    database_path = os.path.join(tmp_path, "target")
    database = femr.datasets.PatientDatabase(database_path)
    ontology = database.get_ontology()

    femr_outcome_code = get_femr_code(ontology, 2)
    femr_admission_code = get_femr_code(ontology, 3)

    labeler = CodeLabeler([femr_outcome_code], time_horizon, [femr_admission_code])
    labeled_patients = labeler.apply(path_to_patient_database=database_path)

    age_featurizer = AgeFeaturizer(is_normalize=True)
    age_featurizer_list = FeaturizerList([age_featurizer])
    age_featurizer_list.preprocess_featurizers(database_path, labeled_patients)
    age_featurized_patients = age_featurizer_list.featurize(database_path, labeled_patients)

    time_bins = [
        datetime.timedelta(days=90),
        datetime.timedelta(days=180),
        datetime.timedelta(weeks=1e5),
    ]
    count_featurizer = CountFeaturizer(is_ontology_expansion=True, time_bins=time_bins)
    count_featurizer_list = FeaturizerList([count_featurizer])
    count_featurizer_list.preprocess_featurizers(database_path, labeled_patients)
    count_featurized_patients = count_featurizer_list.featurize(database_path, labeled_patients)

    age_featurizer = AgeFeaturizer(is_normalize=True)
    time_bins = [
        datetime.timedelta(days=90),
        datetime.timedelta(days=180),
        datetime.timedelta(weeks=1e5),
    ]
    count_featurizer = CountFeaturizer(is_ontology_expansion=True, time_bins=time_bins)
    featurizer_list = FeaturizerList([age_featurizer, count_featurizer])
    featurizer_list.preprocess_featurizers(database_path, labeled_patients)
    featurized_patients = featurizer_list.featurize(database_path, labeled_patients)

    assert featurized_patients[0].shape == (len(labeled_patients) * 3, 10)

    the_same = (
        featurized_patients[0].toarray()
        == scipy.sparse.hstack((age_featurized_patients[0], count_featurized_patients[0])).toarray()
    )

    assert the_same.all()


def test_serialization_and_deserialization(tmp_path: pathlib.Path):
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))

    create_database(tmp_path)

    database_path = os.path.join(tmp_path, "target")
    database = femr.datasets.PatientDatabase(database_path)
    ontology = database.get_ontology()

    femr_outcome_code = get_femr_code(ontology, 2)
    femr_admission_code = get_femr_code(ontology, 3)

    labeler = CodeLabeler([femr_outcome_code], time_horizon, [femr_admission_code])
    labeled_patients = labeler.apply(path_to_patient_database=database_path)

    time_bins = [
        datetime.timedelta(days=90),
        datetime.timedelta(days=180),
        datetime.timedelta(weeks=1e5),
    ]
    count_featurizer = CountFeaturizer(is_ontology_expansion=True, time_bins=time_bins)
    count_featurizer_list = FeaturizerList([count_featurizer])
    count_featurizer_list.preprocess_featurizers(database_path, labeled_patients)
    count_featurized_patient = count_featurizer_list.featurize(database_path, labeled_patients)

    save_to_pkl(
        count_featurizer_list,
        os.path.join(tmp_path, "count_featurizer_list.pickle"),
    )
    save_to_pkl(
        count_featurized_patient,
        os.path.join(tmp_path, "count_featurized_patient.pickle"),
    )

    count_featurized_patient_loaded = load_from_pkl(os.path.join(tmp_path, "count_featurized_patient.pickle"))

    assert (count_featurized_patient_loaded[0].toarray() == count_featurized_patient[0].toarray()).all()
    assert (count_featurized_patient_loaded[1] == count_featurized_patient[1]).all()
    assert (count_featurized_patient_loaded[2] == count_featurized_patient[2]).all()
    assert (count_featurized_patient_loaded[3] == count_featurized_patient[3]).all()


if __name__ == "__main__":
    run_test_locally("../ignore/test_labelers/", test_complete_featurization)
