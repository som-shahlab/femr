import datetime
import os
import pathlib
from typing import List, Optional, Tuple, cast

import numpy as np

import piton
import piton.datasets
from piton.labelers.core import (
    Label,
    LabeledPatients,
    LabelType,
    NLabelPerPatientLF,
    TimeHorizon,
    LabelingFunction,
    SurvivalValue
)
from piton.labelers.omop_labeling_functions import CodeLF, MortalityLF
from tools import (
    DUMMY_EVENTS,
    create_database,
    create_labeled_patients_list,
    create_patients_list,
    get_piton_code,
    load_from_pkl,
    save_to_pkl,
)


def assert_labels_are_accurate(
    labeled_patients: LabeledPatients,
    patient_id: int,
    true_labels: List[Optional[bool]] | List[bool],
    help_text: str = "",
):
    """Passes if the labels in `labeled_patients` for `patient_id` exactly match the labels in `true_labels`."""
    assert len(labeled_patients[patient_id]) == len(
        true_labels
    ), f"{len(labeled_patients[patient_id])} != {len(true_labels)}" + (
        help_text if help_text else ""
    )  # Needed in case `true_labels = []`
    for idx, (label, true_label) in enumerate(
        zip(labeled_patients[patient_id], true_labels)
    ):
        assert label.value == true_label, (
            f"Patient #{patient_id}, idx={idx} | {label.value} != {true_label}"
            + (help_text if help_text else "")
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
        if labeled_patients.labeler_type in [ 'boolean', 'numeric', 'categorical']:
            assert (
                Label(
                    value=label_numpy[1][i],
                    time=label_numpy[2][i],
                )
                in labeled_patients[patient_id]
            )
        elif labeled_patients.labeler_type in ['survival']:
            assert (
                Label(
                    value=SurvivalValue(
                        time_to_event=label_numpy[1][i][0],
                        is_censored=label_numpy[1][i][1],
                    ),
                    time=label_numpy[2][i],
                )
                in labeled_patients[patient_id]
            )
        else:
            assert False


def test_labeled_patients(tmp_path: pathlib.Path) -> None:
    """Checks internal methods of `LabeledPatient`"""

    time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=180)
    )

    create_database(tmp_path)

    database_path = os.path.join(tmp_path, "target")
    database = piton.datasets.PatientDatabase(database_path)
    ontology = database.get_ontology()

    piton_target_code = get_piton_code(ontology, 2)
    piton_admission_code = get_piton_code(ontology, 3)

    labeler = CodeLF(
        piton_admission_code, piton_target_code, time_horizon=time_horizon
    )
    labeled_patients = labeler.apply(database_path)

    true_labels = [
        # Assumes time horizon (0, 180) days + Code 2
        True,
        False,
        False,
    ]

    # Data representations
    #   Check that label counter is correct
    assert labeled_patients.get_num_labels() == len(true_labels) * len(
        labeled_patients
    ), f"{labeled_patients.get_num_labels()} != {len(true_labels)} * {len(labeled_patients)}"
    #   Check that tuples are correct
    assert_tuples_match_labels(labeled_patients)
    #   Check that numpy are correct
    assert_np_arrays_match_labels(labeled_patients)

    # Saving / Loading
    #   Save labeler results
    path = os.path.join(tmp_path, "CodeLF.pkl")
    save_to_pkl(labeled_patients, path)

    #   Check that file was created
    assert os.path.exists(path)

    #   Read in the output files and check that they're accurate
    labeled_patients_new = load_from_pkl(path)

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


def test_mortality_lf() -> None:
    """Creates a MortalityLF for code 3, which corresponds to "Death Type/" """
    patients = create_patients_list(DUMMY_EVENTS)
    true_labels = [
        # Assumes time horizon (0, 180) days + MortalityLF() where code is 3
        False,
        False,
        True,
    ]

    # 6. Create `Ontology` stub and run `MortalityPredictor`
    class DummyOntology:
        def get_dictionary(self):
            return [
                "zero",
                "one",
                "two",
                "Visit/IP",
                "Condition Type/OMOP4822053",
            ]

    dummy_ontology = DummyOntology()
    ontology = cast(piton.datasets.Ontology, dummy_ontology)
    time_horizon_12_months = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=365)
    )
    labeler = MortalityLF(ontology, time_horizon_12_months)

    labeled_patients = create_labeled_patients_list(labeler, patients)

    # Selected the right code
    assert labeler.code == 4
    # All label values are correct -- assumes all patients have same events
    for patient_id in labeled_patients.keys():
        assert_labels_are_accurate(labeled_patients, patient_id, true_labels)


def test_code_lf() -> None:
    """Creates a CodeLF for code '2' with time horizon of (0,180 days)."""
    patients = create_patients_list(DUMMY_EVENTS)
    true_labels = [
        # Assumes time horizon (0, 180) days + Code 2
        True,
        False,
        False,
    ]

    # Create a CodeLF for Code 2
    time_horizon_6_months = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=180)
    )
    labeler = CodeLF(3, 2, time_horizon_6_months)

    # Check CodeLF's internal functions
    assert labeler.get_time_horizon() == time_horizon_6_months
    for p in patients:
        assert labeler.get_outcome_times(p) == [
            event.start for event in p.events if event.code == 2
        ]

    # Label patients and check that they're correct
    labeled_patients = create_labeled_patients_list(labeler, patients)

    # All label values are correct -- assumes all patients have same events
    for patient_id in labeled_patients.keys():
        assert_labels_are_accurate(labeled_patients, patient_id, true_labels)


def test_time_horizons():
    tests = [
        [
            # Test #0
            # (0, 180) days
            TimeHorizon(
                datetime.timedelta(days=0), datetime.timedelta(days=180)
            ),
            create_patients_list(
                [
                    piton.Event(
                        start=datetime.datetime(2015, 1, 3), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 1, 3), code=1, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 10, 5), code=1, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2018, 1, 3), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2018, 3, 3), code=3, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2018, 5, 3), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2018, 5, 3, 11),
                        code=1,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2018, 5, 4), code=3, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2018, 12, 4), code=2, value=None
                    ),
                ]
            ),
            [
                True,
                False,
            ],
        ],
        [
            # Test #1
            # (180, 365) days
            TimeHorizon(
                datetime.timedelta(days=180), datetime.timedelta(days=365)
            ),
            create_patients_list(
                [
                    piton.Event(
                        start=datetime.datetime(2000, 1, 3), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2000, 10, 5), code=3, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2000, 10, 10),
                        code=2,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2002, 4, 5), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2002, 12, 5), code=3, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2003, 11, 1),
                        code=2,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2004, 1, 10), code=2, value=None
                    ),
                ]
            ),
            [
                False,
                True,
            ],
        ],
        [
            # Test #2
            # (0, 0) days
            TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=0)),
            create_patients_list(
                [
                    piton.Event(
                        start=datetime.datetime(2015, 1, 3), code=3, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 1, 3, 23, 59),
                        code=2,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 1, 5), code=1, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 1, 5, 10),
                        code=3,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 1, 6), code=2, value=None
                    ),
                ]
            ),
            [True, False],
        ],
        [
            # Test #3
            # (10, 10) days
            TimeHorizon(
                datetime.timedelta(days=10), datetime.timedelta(days=10)
            ),
            create_patients_list(
                [
                    piton.Event(
                        start=datetime.datetime(2015, 1, 3), code=3, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 1, 13, 23, 59),
                        code=2,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 1, 23), code=1, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 2, 2), code=3, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 2, 15, 11, 59),
                        code=2,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 3, 20), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 3, 29), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 3, 30), code=1, value=None
                    ),
                ]
            ),
            [True, False],
        ],
        [
            # Test #4
            # (5 hours, 10.5 hours)
            TimeHorizon(
                datetime.timedelta(hours=5),
                datetime.timedelta(hours=10, minutes=30),
            ),
            create_patients_list(
                [
                    piton.Event(
                        start=datetime.datetime(2015, 1, 1, 0, 0),
                        code=1,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 1, 1, 10, 29),
                        code=3,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 1, 2, 0, 0),
                        code=2,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 1, 2, 6, 0),
                        code=1,
                        value=None,
                    ),
                    #
                    piton.Event(
                        start=datetime.datetime(2016, 1, 1, 0, 0),
                        code=1,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2016, 1, 1, 10, 29),
                        code=1,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2016, 1, 1, 10, 30),
                        code=2,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2016, 1, 1, 10, 31),
                        code=1,
                        value=None,
                    ),
                    #
                    piton.Event(
                        start=datetime.datetime(2017, 1, 1, 0, 0),
                        code=1,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2017, 1, 1, 10, 29),
                        code=1,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2017, 1, 1, 10, 30),
                        code=3,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2017, 1, 2, 7, 31),
                        code=2,
                        value=None,
                    ),
                    #
                    piton.Event(
                        start=datetime.datetime(2017, 1, 2, 7, 31),
                        code=1,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2017, 1, 2, 20, 31),
                        code=2,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2018, 1, 1, 5),
                        code=1,
                        value=None,
                    ),
                    #
                    piton.Event(
                        start=datetime.datetime(2019, 1, 1, 0, 0),
                        code=1,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2019, 1, 1, 4, 59, 59),
                        code=1,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2019, 1, 1, 5),
                        code=2,
                        value=None,
                    ),
                ]
            ),
            [
                False,
                True,
            ],
        ],
    ]
    for test_idx, test in enumerate(tests):
        horizon, patients, true_labels = test
        labeler = CodeLF(3, 2, horizon)
        labeled_patients = create_labeled_patients_list(labeler, patients)
        for i in range(len(patients)):
            # all patients have same events
            assert_labels_are_accurate(
                labeled_patients,
                i,
                true_labels,
                help_text=f" | test #{test_idx}",
            )

class DummySurvivalLabeler(LabelingFunction):
    def label(self, patient: piton.Patient) -> List[Label]:
        return [
            Label(time=datetime.datetime(2000, 1, 1), value=SurvivalValue(datetime.timedelta(days=10), 0)),
            Label(time=datetime.datetime(2000, 4, 1), value=SurvivalValue(datetime.timedelta(days=15), 0)),
            Label(time=datetime.datetime(2000, 4, 1), value=SurvivalValue(datetime.timedelta(days=15, hours=10), 0)),
            Label(time=datetime.datetime(2000, 10, 1), value=SurvivalValue(datetime.timedelta(days=100), 1)),
        ]
    def get_labeler_type(self) -> LabelType:
        return "survival"

def test_survival_labels(tmp_path: pathlib.Path) -> None:

    create_database(tmp_path)
    database_path = os.path.join(tmp_path, "target")
    database = piton.datasets.PatientDatabase(database_path)

    labeler = DummySurvivalLabeler()
    patient = database[0]
    patient = cast(piton.Patient, patient)

    labeled_patients = labeler.apply(database_path)

    true_labels = [
        SurvivalValue(datetime.timedelta(days=10), 0), 
        SurvivalValue(datetime.timedelta(days=15), 0),
        SurvivalValue(datetime.timedelta(days=15, hours=10), 0),
        SurvivalValue(datetime.timedelta(days=100), 1),
    ]

    # Data representations
    #   Check that label counter is correct
    assert labeled_patients.get_num_labels() == len(true_labels) * len(labeled_patients), \
        f"{labeled_patients.get_num_labels()} != {len(true_labels)} * {len(labeled_patients)}"
    #   Check that tuples are correct
    assert_tuples_match_labels(labeled_patients)
    #   Check that numpy are correct
    assert_np_arrays_match_labels(labeled_patients)

    # Saving / Loading
    #   Save labeler results
    path = os.path.join(tmp_path, "labeled_patients.pkl")
    save_to_pkl(labeled_patients, path)

    #   Check that file was created
    assert os.path.exists(path)

    #   Read in the output files and check that they're accurate
    labeled_patients_new = load_from_pkl(path)

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
    
    


def test_NLabelPerPatientLF(tmp_path: pathlib.Path) -> None:
    """Checks NLabelPerPatient LabelingFunction"""

    time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=180)
    )

    create_database(tmp_path)

    database_path = os.path.join(tmp_path, "target")
    database = piton.datasets.PatientDatabase(database_path)
    ontology = database.get_ontology()

    piton_target_code = get_piton_code(ontology, 3)
    piton_admission_code = get_piton_code(ontology, 2)

    labeler = CodeLF(
        piton_admission_code, piton_target_code, time_horizon=time_horizon
    )
    patient = database[0]
    patient = cast(piton.Patient, patient)
    all_labels = labeler.label(patient)

    num_all_labels = len(all_labels)

    seed = 0
    num_labels = 6
    n_labeler = NLabelPerPatientLF(
        labeling_function=labeler, num_labels=num_labels, seed=seed
    )
    n_labels = n_labeler.label(patient)

    assert len(n_labels) == num_all_labels

    seeds = [0, 10, 40]
    num_labels = 3

    n_labels_list = []
    for seed in seeds:
        n_labeler = NLabelPerPatientLF(
            labeling_function=labeler, num_labels=num_labels, seed=seed
        )
        n_labels = n_labeler.label(patient)
        n_labels_list.append(n_labels)

    assert n_labels_list[0] == n_labels_list[0]

    for i in range(len(n_labels_list)):
        for j in range(i + 1, len(n_labels_list)):
            assert n_labels_list[i] != n_labels_list[j]