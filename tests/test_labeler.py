from ehr_ml.labeler import Label, ObservationGreaterThanValue

from test_utils import *  # noqa: F401; pylint: disable=unused-variable


def test_long_admission_labeler(dummy_patient):
    labeler = ObservationGreaterThanValue(909, 1)

    labels = labeler.label(dummy_patient)

    print(labels)

    assert labels == [
        Label(day_index=0, is_positive=False),
        Label(day_index=1, is_positive=True),
    ]
