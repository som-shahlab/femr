import scipy.sparse

from ehr_ml.featurizer import (
    AgeFeaturizer,
    ColumnValue,
    CountFeaturizer,
    FeaturizerList,
)
from ehr_ml.labeler import ObservationGreaterThanValue

from test_utils import *  # noqa: F401; pylint: disable=unused-variable


def test_age_featurizer(dummy_patient, dummy_timeline):
    featurizer = AgeFeaturizer()

    print(dummy_patient)

    label_indices = {0, 2}

    featurizer.train(dummy_patient, label_indices)

    assert featurizer.num_columns() == 1

    transformed = featurizer.transform(dummy_patient, label_indices)

    print(transformed)

    assert transformed == [
        [ColumnValue(0, -0.7071067811865476)],
        [ColumnValue(0, 0.7071067811865476)],
    ]


def test_count_featurizer(dummy_patient, dummy_timeline, dummy_ontologies):
    featurizer = CountFeaturizer(dummy_timeline, dummy_ontologies)

    label_indices = {1}

    featurizer.train(dummy_patient, label_indices)

    assert featurizer.num_columns() == 5

    transformed = featurizer.transform(dummy_patient, label_indices)

    print(transformed)

    assert transformed == [
        [
            ColumnValue(column=0, value=2),
            ColumnValue(column=1, value=1),
            ColumnValue(column=2, value=1),
            ColumnValue(column=3, value=1),
            ColumnValue(column=4, value=1),
        ]
    ]


def test_count_bins_featurizer(dummy_patient, dummy_timeline, dummy_ontologies):
    featurizer = CountFeaturizer(
        dummy_timeline, dummy_ontologies, time_bins=[365 * 2, None]
    )

    label_indices = {1}

    featurizer.train(dummy_patient, label_indices)

    assert featurizer.num_columns() == 10

    transformed = featurizer.transform(dummy_patient, label_indices)

    print(transformed)

    assert transformed == [
        [
            ColumnValue(column=0, value=1),
            ColumnValue(column=3, value=1),
            ColumnValue(column=4, value=1),
            ColumnValue(column=5, value=1),
            ColumnValue(column=6, value=1),
            ColumnValue(column=7, value=1),
        ]
    ]


def test_complete_featurization(
    dummy_patient, dummy_timeline, dummy_ontologies
):
    age = AgeFeaturizer()
    count = CountFeaturizer(dummy_timeline, dummy_ontologies)
    featurizersAge = FeaturizerList([age])
    featurizersCount = FeaturizerList([count])
    featurizersBoth = FeaturizerList([age, count])

    labeler = ObservationGreaterThanValue(909, 1)

    featurizersAge.train_featurizers(dummy_timeline, labeler)
    featurizersCount.train_featurizers(dummy_timeline, labeler)
    featurizersBoth.train_featurizers(dummy_timeline, labeler)

    full_matrix, labels, _, _ = featurizersBoth.featurize(
        dummy_timeline, labeler
    )

    age_matrix, age_labels, _, _ = featurizersAge.featurize(
        dummy_timeline, labeler
    )

    count_matrix, count_labels, _, _ = featurizersCount.featurize(
        dummy_timeline, labeler
    )

    assert (labels == age_labels).all()
    assert (labels == count_labels).all()

    print(full_matrix.shape, age_matrix.shape, count_matrix.shape)

    the_same = (
        full_matrix.toarray()
        == scipy.sparse.hstack((age_matrix, count_matrix)).toarray()
    )

    assert the_same.all()
