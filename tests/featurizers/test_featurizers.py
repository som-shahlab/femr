import datetime
from typing import Any, Mapping, cast

import femr_test_tools
import meds
import meds_reader
import pandas as pd
import scipy.sparse

import femr
from femr.featurizers import FeaturizerList
from femr.featurizers.featurizers import AgeFeaturizer, CountFeaturizer
from femr.labelers import TimeHorizon
from femr.labelers.omop import CodeLabeler


def _assert_featurized_subjects_structure(labels: pd.DataFrame, features: Mapping[str, Any]):
    assert features["features"].dtype == "float32"
    assert features["subject_ids"].dtype == "int64"
    assert features["feature_times"].dtype == "datetime64[us]"

    assert features["feature_times"].shape[0] == len(labels)
    assert features["subject_ids"].shape[0] == len(labels)
    assert features["features"].shape[0] == len(labels)

    assert sorted(list(features["subject_ids"])) == sorted(
        list(label.subject_id for label in labels.itertuples(index=False))
    )
    assert sorted(list(features["feature_times"])) == sorted(
        list(label.prediction_time for label in labels.itertuples(index=False))
    )


def test_age_featurizer() -> None:
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))

    dataset = femr_test_tools.create_subjects_dataset(100)

    labeler = CodeLabeler(["2"], time_horizon, ["3"])

    subject: meds_reader.Subject = dataset[0]
    labels = labeler.label(subject)
    featurizer = AgeFeaturizer(is_normalize=False)
    subject_features = featurizer.featurize(subject, labels)

    assert subject_features[0] == [(0, 15.43013698630137)]
    assert subject_features[1] == [(0, 17.767123287671232)]
    assert subject_features[-1] == [(0, 20.46027397260274)]

    all_labels = labeler.apply(dataset)

    featurizer = AgeFeaturizer(is_normalize=True)
    featurizer_list = FeaturizerList([featurizer])
    featurizer_list.preprocess_featurizers(dataset, all_labels)
    featurized_subjects = featurizer_list.featurize(dataset, all_labels)

    _assert_featurized_subjects_structure(all_labels, featurized_subjects)


def test_count_featurizer() -> None:
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))

    dataset = femr_test_tools.create_subjects_dataset(100)

    labeler = CodeLabeler(["2"], time_horizon, ["3"])

    subject: meds_reader.Subject = dataset[0]
    labels = labeler.label(subject)
    featurizer = CountFeaturizer()
    data = featurizer.get_initial_preprocess_data()
    featurizer.add_preprocess_data(data, subject, labels)
    featurizer.encorperate_prepreprocessed_data([data])

    subject_features = featurizer.featurize(subject, labels)

    assert featurizer.get_num_columns() == 4, f"featurizer.get_num_columns() = {featurizer.get_num_columns()}"

    simple_subject_features = [{(featurizer.get_column_name(v.column), v.value) for v in a} for a in subject_features]

    assert simple_subject_features[0] == {
        (meds.birth_code, 1),
        ("3", 1),
    }
    assert simple_subject_features[1] == {
        (meds.birth_code, 1),
        ("3", 2),
        ("2", 2),
    }
    assert simple_subject_features[2] == {
        (meds.birth_code, 1),
        ("3", 3),
        ("2", 4),
    }

    all_labels = labeler.apply(dataset)

    featurizer = CountFeaturizer()
    featurizer_list = FeaturizerList([featurizer])
    featurizer_list.preprocess_featurizers(dataset, all_labels)
    featurized_subjects = featurizer_list.featurize(dataset, all_labels)

    _assert_featurized_subjects_structure(all_labels, featurized_subjects)


def test_count_featurizer_with_ontology() -> None:
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))

    dataset = femr_test_tools.create_subjects_dataset(100)

    labeler = CodeLabeler(["2"], time_horizon, ["3"])

    subject: meds_reader.Subject = dataset[0]
    labels = labeler.label(subject)

    class DummyOntology:
        def get_all_parents(self, code):
            if code in ("2", meds.birth_code):
                return {"parent", code}
            else:
                return {code}

    featurizer = CountFeaturizer(is_ontology_expansion=True, ontology=cast(femr.ontology.Ontology, DummyOntology()))
    data = featurizer.get_initial_preprocess_data()
    featurizer.add_preprocess_data(data, subject, labels)
    featurizer.encorperate_prepreprocessed_data([data])

    subject_features = featurizer.featurize(subject, labels)

    assert featurizer.get_num_columns() == 5, f"featurizer.get_num_columns() = {featurizer.get_num_columns()}"

    simple_subject_features = [{(featurizer.get_column_name(v.column), v.value) for v in a} for a in subject_features]

    assert simple_subject_features[0] == {
        (meds.birth_code, 1),
        ("3", 1),
        ("parent", 1),
    }
    assert simple_subject_features[1] == {
        (meds.birth_code, 1),
        ("3", 2),
        ("2", 2),
        ("parent", 3),
    }
    assert simple_subject_features[2] == {
        (meds.birth_code, 1),
        ("parent", 5),
        ("3", 3),
        ("2", 4),
    }

    all_labels = labeler.apply(dataset)

    featurizer = CountFeaturizer(is_ontology_expansion=True, ontology=cast(femr.ontology.Ontology, DummyOntology()))
    featurizer_list = FeaturizerList([featurizer])
    featurizer_list.preprocess_featurizers(dataset, all_labels)
    featurized_subjects = featurizer_list.featurize(dataset, all_labels)

    _assert_featurized_subjects_structure(all_labels, featurized_subjects)


def test_count_featurizer_with_values() -> None:
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))

    dataset = femr_test_tools.create_subjects_dataset(100)

    labeler = CodeLabeler(["2"], time_horizon, ["3"])

    subject: meds_reader.Subject = dataset[0]
    labels = labeler.label(subject)
    featurizer = CountFeaturizer(numeric_value_decile=True, string_value_combination=True)
    data = featurizer.get_initial_preprocess_data()
    featurizer.add_preprocess_data(data, subject, labels)
    featurizer.encorperate_prepreprocessed_data([data])

    subject_features = featurizer.featurize(subject, labels)

    assert featurizer.get_num_columns() == 7

    simple_subject_features = [{(featurizer.get_column_name(v.column), v.value) for v in a} for a in subject_features]

    assert simple_subject_features[0] == {
        (meds.birth_code, 1),
        ("3", 1),
        ("2 [1.0, inf)", 1),
        ("1 test_value", 2),
    }

    assert simple_subject_features[1] == {
        (meds.birth_code, 1),
        ("3", 2),
        ("2", 2),
        ("2 [1.0, inf)", 1),
        ("1 test_value", 2),
    }
    assert simple_subject_features[2] == {
        (meds.birth_code, 1),
        ("3", 3),
        ("2", 4),
        ("2 [1.0, inf)", 1),
        ("1 test_value", 2),
    }

    all_labels = labeler.apply(dataset)

    featurizer = CountFeaturizer(numeric_value_decile=True, string_value_combination=True)
    featurizer_list = FeaturizerList([featurizer])
    featurizer_list.preprocess_featurizers(dataset, all_labels)
    featurized_subjects = featurizer_list.featurize(dataset, all_labels)

    _assert_featurized_subjects_structure(all_labels, featurized_subjects)


def test_count_featurizer_exclude_filter() -> None:
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))

    dataset = femr_test_tools.create_subjects_dataset(100)

    labeler = CodeLabeler(["2"], time_horizon, ["3"])

    subject: meds_reader.Subject = dataset[0]
    labels = labeler.label(subject)

    # Test filtering all codes
    featurizer = CountFeaturizer(excluded_event_filter=lambda _: True)
    data = featurizer.get_initial_preprocess_data()
    featurizer.add_preprocess_data(data, subject, labels)
    featurizer.encorperate_prepreprocessed_data([data])

    assert featurizer.get_num_columns() == 0

    # Test filtering no codes
    featurizer = CountFeaturizer(excluded_event_filter=lambda _: False)
    data = featurizer.get_initial_preprocess_data()
    featurizer.add_preprocess_data(data, subject, labels)
    featurizer.encorperate_prepreprocessed_data([data])

    assert featurizer.get_num_columns() == 4

    # Test filtering single code
    featurizer = CountFeaturizer(excluded_event_filter=lambda e: e.code == "3")
    data = featurizer.get_initial_preprocess_data()
    featurizer.add_preprocess_data(data, subject, labels)
    featurizer.encorperate_prepreprocessed_data([data])

    assert featurizer.get_num_columns() == 3


def test_count_bins_featurizer() -> None:
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))

    dataset = femr_test_tools.create_subjects_dataset(100)

    labeler = CodeLabeler(["2"], time_horizon, ["3"])

    subject: meds_reader.Subject = dataset[0]
    labels = labeler.label(subject)
    time_bins = [
        datetime.timedelta(days=90),
        datetime.timedelta(days=180),
        datetime.timedelta(weeks=1e4),
    ]
    featurizer = CountFeaturizer(
        time_bins=time_bins,
    )
    data = featurizer.get_initial_preprocess_data()
    featurizer.add_preprocess_data(data, subject, labels)
    featurizer.encorperate_prepreprocessed_data([data])

    subject_features = featurizer.featurize(subject, labels)

    assert featurizer.get_num_columns() == 12

    simple_subject_features = [{(featurizer.get_column_name(v.column), v.value) for v in a} for a in subject_features]

    assert simple_subject_features[0] == {
        (meds.birth_code + "_70000 days, 0:00:00", 1),
        ("3_90 days, 0:00:00", 1),
    }
    assert simple_subject_features[1] == {
        ("3_90 days, 0:00:00", 1),
        (meds.birth_code + "_70000 days, 0:00:00", 1),
        ("3_70000 days, 0:00:00", 1),
        ("2_70000 days, 0:00:00", 2),
    }
    assert simple_subject_features[2] == {
        ("2_70000 days, 0:00:00", 2),
        ("2_90 days, 0:00:00", 2),
        (meds.birth_code + "_70000 days, 0:00:00", 1),
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
    featurized_subjects = featurizer_list.featurize(dataset, all_labels)

    _assert_featurized_subjects_structure(all_labels, featurized_subjects)


def test_complete_featurization() -> None:
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))

    dataset = femr_test_tools.create_subjects_dataset(100)

    labeler = CodeLabeler(["2"], time_horizon, ["3"])

    all_labels = labeler.apply(dataset)

    age_featurizer = AgeFeaturizer(is_normalize=True)
    age_featurizer_list = FeaturizerList([age_featurizer])
    age_featurizer_list.preprocess_featurizers(dataset, all_labels)
    age_featurized_subjects = age_featurizer_list.featurize(dataset, all_labels)

    time_bins = [
        datetime.timedelta(days=90),
        datetime.timedelta(days=180),
        datetime.timedelta(weeks=1e5),
    ]
    count_featurizer = CountFeaturizer(time_bins=time_bins)
    count_featurizer_list = FeaturizerList([count_featurizer])
    count_featurizer_list.preprocess_featurizers(dataset, all_labels)
    count_featurized_subjects = count_featurizer_list.featurize(dataset, all_labels)

    age_featurizer = AgeFeaturizer(is_normalize=True)
    time_bins = [
        datetime.timedelta(days=90),
        datetime.timedelta(days=180),
        datetime.timedelta(weeks=1e5),
    ]
    count_featurizer = CountFeaturizer(time_bins=time_bins)
    featurizer_list = FeaturizerList([age_featurizer, count_featurizer])
    featurizer_list.preprocess_featurizers(dataset, all_labels)
    featurized_subjects = featurizer_list.featurize(dataset, all_labels)

    assert featurized_subjects["subject_ids"].shape == count_featurized_subjects["subject_ids"].shape

    the_same = (
        featurized_subjects["features"].toarray()
        == scipy.sparse.hstack((age_featurized_subjects["features"], count_featurized_subjects["features"])).toarray()
    )

    assert the_same.all()
