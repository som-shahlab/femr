import datetime
from typing import Any, List, Mapping, cast

import femr_test_tools
import meds
import meds_reader
import scipy.sparse

import femr
from femr.featurizers import FeaturizerList
from femr.featurizers.featurizers import AgeFeaturizer, CountFeaturizer
from femr.labelers import TimeHorizon
from femr.labelers.omop import CodeLabeler


def _assert_featurized_patients_structure(labels: List[meds.Label], features: Mapping[str, Any]):
    assert features["features"].dtype == "float32"
    assert features["patient_ids"].dtype == "int64"
    assert features["feature_times"].dtype == "datetime64[us]"

    assert features["feature_times"].shape[0] == len(labels)
    assert features["patient_ids"].shape[0] == len(labels)
    assert features["features"].shape[0] == len(labels)

    assert sorted(list(features["patient_ids"])) == sorted(list(label["patient_id"] for label in labels))
    assert sorted(list(features["feature_times"])) == sorted(list(label["prediction_time"] for label in labels))


def test_age_featurizer() -> None:
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))

    dataset = femr_test_tools.create_patients_dataset(100)

    labeler = CodeLabeler(["2"], time_horizon, ["3"])

    patient: meds_reader.Patient = dataset[0]
    labels = labeler.label(patient)
    featurizer = AgeFeaturizer(is_normalize=False)
    patient_features = featurizer.featurize(patient, labels)

    assert patient_features[0] == [(0, 15.43013698630137)]
    assert patient_features[1] == [(0, 17.767123287671232)]
    assert patient_features[-1] == [(0, 20.46027397260274)]

    all_labels = labeler.apply(dataset)

    featurizer = AgeFeaturizer(is_normalize=True)
    featurizer_list = FeaturizerList([featurizer])
    featurizer_list.preprocess_featurizers(dataset, all_labels)
    featurized_patients = featurizer_list.featurize(dataset, all_labels)

    _assert_featurized_patients_structure(all_labels, featurized_patients)


def test_count_featurizer() -> None:
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))

    dataset = femr_test_tools.create_patients_dataset(100)

    labeler = CodeLabeler(["2"], time_horizon, ["3"])

    patient: meds_reader.Patient = dataset[0]
    labels = labeler.label(patient)
    featurizer = CountFeaturizer()
    data = featurizer.get_initial_preprocess_data()
    featurizer.add_preprocess_data(data, patient, {patient.patient_id: labels})
    featurizer.encorperate_prepreprocessed_data([data])

    patient_features = featurizer.featurize(patient, labels)

    assert featurizer.get_num_columns() == 4, f"featurizer.get_num_columns() = {featurizer.get_num_columns()}"

    simple_patient_features = [{(featurizer.get_column_name(v.column), v.value) for v in a} for a in patient_features]

    assert simple_patient_features[0] == {
        ("SNOMED/184099003", 1),
        ("3", 1),
    }
    assert simple_patient_features[1] == {
        ("SNOMED/184099003", 1),
        ("3", 2),
        ("2", 2),
    }
    assert simple_patient_features[2] == {
        ("SNOMED/184099003", 1),
        ("3", 3),
        ("2", 4),
    }

    all_labels = labeler.apply(dataset)

    featurizer = CountFeaturizer()
    featurizer_list = FeaturizerList([featurizer])
    featurizer_list.preprocess_featurizers(dataset, all_labels)
    featurized_patients = featurizer_list.featurize(dataset, all_labels)

    _assert_featurized_patients_structure(all_labels, featurized_patients)


def test_count_featurizer_with_ontology() -> None:
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))

    dataset = femr_test_tools.create_patients_dataset(100)

    labeler = CodeLabeler(["2"], time_horizon, ["3"])

    patient: meds_reader.Patient = dataset[0]
    labels = labeler.label(patient)

    class DummyOntology:
        def get_all_parents(self, code):
            if code in ("2", "SNOMED/184099003"):
                return {"parent", code}
            else:
                return {code}

    featurizer = CountFeaturizer(is_ontology_expansion=True, ontology=cast(femr.ontology.Ontology, DummyOntology()))
    data = featurizer.get_initial_preprocess_data()
    featurizer.add_preprocess_data(data, patient, {patient.patient_id: labels})
    featurizer.encorperate_prepreprocessed_data([data])

    patient_features = featurizer.featurize(patient, labels)

    assert featurizer.get_num_columns() == 5, f"featurizer.get_num_columns() = {featurizer.get_num_columns()}"

    simple_patient_features = [{(featurizer.get_column_name(v.column), v.value) for v in a} for a in patient_features]

    assert simple_patient_features[0] == {
        ("SNOMED/184099003", 1),
        ("3", 1),
        ("parent", 1),
    }
    assert simple_patient_features[1] == {
        ("SNOMED/184099003", 1),
        ("3", 2),
        ("2", 2),
        ("parent", 3),
    }
    assert simple_patient_features[2] == {
        ("SNOMED/184099003", 1),
        ("parent", 5),
        ("3", 3),
        ("2", 4),
    }

    all_labels = labeler.apply(dataset)

    featurizer = CountFeaturizer(is_ontology_expansion=True, ontology=cast(femr.ontology.Ontology, DummyOntology()))
    featurizer_list = FeaturizerList([featurizer])
    featurizer_list.preprocess_featurizers(dataset, all_labels)
    featurized_patients = featurizer_list.featurize(dataset, all_labels)

    _assert_featurized_patients_structure(all_labels, featurized_patients)


def test_count_featurizer_with_values() -> None:
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))

    dataset = femr_test_tools.create_patients_dataset(100)

    labeler = CodeLabeler(["2"], time_horizon, ["3"])

    patient: meds_reader.Patient = dataset[0]
    labels = labeler.label(patient)
    featurizer = CountFeaturizer(numeric_value_decile=True, string_value_combination=True)
    data = featurizer.get_initial_preprocess_data()
    featurizer.add_preprocess_data(data, patient, {patient.patient_id: labels})
    featurizer.encorperate_prepreprocessed_data([data])

    patient_features = featurizer.featurize(patient, labels)

    assert featurizer.get_num_columns() == 7

    simple_patient_features = [{(featurizer.get_column_name(v.column), v.value) for v in a} for a in patient_features]

    assert simple_patient_features[0] == {
        ("SNOMED/184099003", 1),
        ("3", 1),
        ("2 [1.0, inf)", 1),
        ("1 test_value", 2),
    }

    assert simple_patient_features[1] == {
        ("SNOMED/184099003", 1),
        ("3", 2),
        ("2", 2),
        ("2 [1.0, inf)", 1),
        ("1 test_value", 2),
    }
    assert simple_patient_features[2] == {
        ("SNOMED/184099003", 1),
        ("3", 3),
        ("2", 4),
        ("2 [1.0, inf)", 1),
        ("1 test_value", 2),
    }

    all_labels = labeler.apply(dataset)

    featurizer = CountFeaturizer(numeric_value_decile=True, string_value_combination=True)
    featurizer_list = FeaturizerList([featurizer])
    featurizer_list.preprocess_featurizers(dataset, all_labels)
    featurized_patients = featurizer_list.featurize(dataset, all_labels)

    _assert_featurized_patients_structure(all_labels, featurized_patients)


def test_count_featurizer_exclude_filter() -> None:
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))

    dataset = femr_test_tools.create_patients_dataset(100)

    labeler = CodeLabeler(["2"], time_horizon, ["3"])

    patient: meds_reader.Patient = dataset[0]
    labels = labeler.label(patient)

    # Test filtering all codes
    featurizer = CountFeaturizer(excluded_event_filter=lambda _: True)
    data = featurizer.get_initial_preprocess_data()
    featurizer.add_preprocess_data(data, patient, {patient.patient_id: labels})
    featurizer.encorperate_prepreprocessed_data([data])

    assert featurizer.get_num_columns() == 0

    # Test filtering no codes
    featurizer = CountFeaturizer(excluded_event_filter=lambda _: False)
    data = featurizer.get_initial_preprocess_data()
    featurizer.add_preprocess_data(data, patient, {patient.patient_id: labels})
    featurizer.encorperate_prepreprocessed_data([data])

    assert featurizer.get_num_columns() == 4

    # Test filtering single code
    featurizer = CountFeaturizer(excluded_event_filter=lambda e: e.code == "3")
    data = featurizer.get_initial_preprocess_data()
    featurizer.add_preprocess_data(data, patient, {patient.patient_id: labels})
    featurizer.encorperate_prepreprocessed_data([data])

    assert featurizer.get_num_columns() == 3


def test_count_bins_featurizer() -> None:
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))

    dataset = femr_test_tools.create_patients_dataset(100)

    labeler = CodeLabeler(["2"], time_horizon, ["3"])

    patient: meds_reader.Patient = dataset[0]
    labels = labeler.label(patient)
    time_bins = [
        datetime.timedelta(days=90),
        datetime.timedelta(days=180),
        datetime.timedelta(weeks=1e4),
    ]
    featurizer = CountFeaturizer(
        time_bins=time_bins,
    )
    data = featurizer.get_initial_preprocess_data()
    featurizer.add_preprocess_data(data, patient, {patient.patient_id: labels})
    featurizer.encorperate_prepreprocessed_data([data])

    patient_features = featurizer.featurize(patient, labels)

    assert featurizer.get_num_columns() == 12

    simple_patient_features = [{(featurizer.get_column_name(v.column), v.value) for v in a} for a in patient_features]

    assert simple_patient_features[0] == {
        ("SNOMED/184099003_70000 days, 0:00:00", 1),
        ("3_90 days, 0:00:00", 1),
    }
    assert simple_patient_features[1] == {
        ("3_90 days, 0:00:00", 1),
        ("SNOMED/184099003_70000 days, 0:00:00", 1),
        ("3_70000 days, 0:00:00", 1),
        ("2_70000 days, 0:00:00", 2),
    }
    assert simple_patient_features[2] == {
        ("2_70000 days, 0:00:00", 2),
        ("2_90 days, 0:00:00", 2),
        ("SNOMED/184099003_70000 days, 0:00:00", 1),
        ("3_90 days, 0:00:00", 1),
        ("3_70000 days, 0:00:00", 2),
    }

    all_labels = labeler.apply(dataset)

    time_bins = [
        datetime.timedelta(days=90),
        datetime.timedelta(days=180),
        datetime.timedelta(weeks=1e4),
    ]
    featurizer = CountFeaturizer(
        time_bins=time_bins,
    )
    featurizer_list = FeaturizerList([featurizer])
    featurizer_list.preprocess_featurizers(dataset, all_labels)
    featurized_patients = featurizer_list.featurize(dataset, all_labels)

    _assert_featurized_patients_structure(all_labels, featurized_patients)


def test_complete_featurization() -> None:
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))

    dataset = femr_test_tools.create_patients_dataset(100)

    labeler = CodeLabeler(["2"], time_horizon, ["3"])

    all_labels = labeler.apply(dataset)

    age_featurizer = AgeFeaturizer(is_normalize=True)
    age_featurizer_list = FeaturizerList([age_featurizer])
    age_featurizer_list.preprocess_featurizers(dataset, all_labels)
    age_featurized_patients = age_featurizer_list.featurize(dataset, all_labels)

    time_bins = [
        datetime.timedelta(days=90),
        datetime.timedelta(days=180),
        datetime.timedelta(weeks=1e5),
    ]
    count_featurizer = CountFeaturizer(time_bins=time_bins)
    count_featurizer_list = FeaturizerList([count_featurizer])
    count_featurizer_list.preprocess_featurizers(dataset, all_labels)
    count_featurized_patients = count_featurizer_list.featurize(dataset, all_labels)

    age_featurizer = AgeFeaturizer(is_normalize=True)
    time_bins = [
        datetime.timedelta(days=90),
        datetime.timedelta(days=180),
        datetime.timedelta(weeks=1e5),
    ]
    count_featurizer = CountFeaturizer(time_bins=time_bins)
    featurizer_list = FeaturizerList([age_featurizer, count_featurizer])
    featurizer_list.preprocess_featurizers(dataset, all_labels)
    featurized_patients = featurizer_list.featurize(dataset, all_labels)

    assert featurized_patients["patient_ids"].shape == count_featurized_patients["patient_ids"].shape

    the_same = (
        featurized_patients["features"].toarray()
        == scipy.sparse.hstack((age_featurized_patients["features"], count_featurized_patients["features"])).toarray()
    )

    assert the_same.all()
