import datetime
import os
import pathlib
import pickle
from typing import List, Optional, Tuple

import numpy as np

import piton.datasets
from piton.labelers.core import Label, LabeledPatients, TimeHorizon
from piton.labelers.omop import CodeLabeler

# Needed to import `tools` for local testing
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import (
    EventsWithLabels,
    assert_labels_are_accurate,
    create_patients_list,
    event,
    run_test_locally,
    save_to_pkl,
)


def assert_tuples_match_labels(labeled_patients: LabeledPatients):
    """Passes if tuples output by `as_list_of_label_tuples()` are the same as the `labels` in `labeled_patients`."""
    label_tuples = labeled_patients.as_list_of_label_tuples()
    assert (
        len(label_tuples) == labeled_patients.get_num_labels()
    ), f"{len(label_tuples)} != {labeled_patients.get_num_labels()}"  # Needed in case `label_tuples = []`
    for lt in label_tuples:
        patient_id = lt[0]
        label = lt[1]
        assert label in labeled_patients[patient_id]


def assert_np_arrays_match_labels(labeled_patients: LabeledPatients):
    """Passes if np.arrays output by `as_numpy_arrays()` are the same as the `labels` in `labeled_patients`."""
    label_numpy: Tuple[
        np.ndarray, np.ndarray, np.ndarray
    ] = labeled_patients.as_numpy_arrays()
    assert (
        label_numpy[0].shape[0]
        == label_numpy[1].shape[0]
        == label_numpy[2].shape[0]
        == labeled_patients.get_num_labels()
    )
    for i in range(label_numpy[0].shape[0]):
        patient_id = label_numpy[0][i]
        assert (
            Label(
                value=bool(label_numpy[1][i])
                if label_numpy[1][i] is not None
                else None,
                time=label_numpy[2][i],
            )
            in labeled_patients[patient_id]
        ), f"{Label(value=bool(label_numpy[1][i]), time=label_numpy[2][i])} not in {labeled_patients[patient_id]}"


def test_labeled_patients(tmp_path: pathlib.Path) -> None:
    """Checks internal methods of `LabeledPatient`"""
    events_with_labels: EventsWithLabels = [
        (event((1995, 1, 3), 0, 34.5), "skip"),
        (event((2010, 1, 1), 1, "test_value"), "skip"),
        (event((2010, 1, 5), 2, 1), True),
        (event((2010, 6, 5), 3, True), "skip"),
        (event((2011, 2, 5), 2, None), False),
        (event((2011, 7, 5), 2, None), False),
        (event((2012, 10, 5), 3, None), "skip"),
        (event((2015, 6, 5, 0), 2, None), True),
        (event((2015, 6, 5, 10, 10), 2, None), True),
        (event((2015, 6, 15, 11), 3, None), "skip"),
        (event((2016, 1, 1), 2, None), None),
        (event((2016, 3, 1, 10, 10, 10), 2, None), None),
    ]
    patients: List[piton.Patient] = create_patients_list(
        10, [x[0] for x in events_with_labels]
    )
    true_labels: List[Tuple[datetime.datetime, Optional[bool]]] = [
        (x[0].start, x[1])
        for x in events_with_labels
        if isinstance(x[1], bool) or (x[1] is None)
    ]

    # Create Labeler
    time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=180)
    )
    labeler = CodeLabeler([3], time_horizon, prediction_codes=[2])

    # Create LabeledPatients
    patients_to_labels = {}
    for patient in patients:
        labels = labeler.label(patient)
        if len(labels) > 0:
            patients_to_labels[patient.patient_id] = labels
    labeled_patients: LabeledPatients = LabeledPatients(
        patients_to_labels, labeler.get_labeler_type()
    )

    # Data representations
    #   Check accuracy of Labels
    for i in range(len(patients)):
        assert_labels_are_accurate(labeled_patients, i, true_labels)
    #   Check that label counter is correct
    assert labeled_patients.get_num_labels() == len(true_labels) * len(
        labeled_patients
    ), (
        f"Number of labels in LabeledPatients ({labeled_patients.get_num_labels()}) "
        f"!= Expected number of labels ({len(true_labels)} * {len(labeled_patients)})"
    )
    #   Check that tuples are correct
    assert_tuples_match_labels(labeled_patients)
    #   Check that numpy are correct
    assert_np_arrays_match_labels(labeled_patients)

    # Saving / Loading
    #   Save labeler results
    path = os.path.join(tmp_path, "LabeledPatients.pkl")
    save_to_pkl(labeled_patients, path)

    #   Check that file was created
    assert os.path.exists(path)

    #   Read in the output files and check that they're accurate
    labeled_patients_new = pickle.load(open(path, "rb"))

    #   Check that we successfully saved / loaded file contents
    assert labeled_patients_new == labeled_patients
    assert (
        labeled_patients_new.as_list_of_label_tuples()
        == labeled_patients.as_list_of_label_tuples()
    )
    for (orig, new) in zip(
        labeled_patients.as_numpy_arrays(),
        labeled_patients_new.as_numpy_arrays(),
    ):
        assert np.sum(orig != new) == 0


run_test_locally("../ignore/test_labelers/", test_labeled_patients)
