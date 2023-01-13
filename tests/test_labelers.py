import datetime
import os
import pathlib
import pickle
from typing import List, Optional, Tuple, cast

import numpy as np
import pickle

import piton
import piton.datasets
from piton.labelers.core import Label, LabeledPatients, TimeHorizon
from piton.labelers.omop_labeling_functions import CodeLF, MortalityLF

dummy_events = [
    piton.Event(start=datetime.datetime(1995, 1, 3), code=0, value=float(34)),
    piton.Event(
        start=datetime.datetime(2010, 1, 3),
        code=1,
        value="test_value",
    ),
    piton.Event(
        start=datetime.datetime(2010, 1, 5),
        code=2,
        value=None,
    ),
]

all_events: List[Tuple[int, piton.Event]] = []

for patient_id in range(10, 25):
    all_events.extend((patient_id, event) for event in dummy_events)


def create_events(tmp_path: pathlib.Path) -> piton.datasets.EventCollection:
    events = piton.datasets.EventCollection(os.path.join(tmp_path, "events"))

    random.shuffle(all_events)

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


# NUM_PATIENTS = 5

# SHARED_EVENTS = [
#     piton.Event(start=datetime.datetime(1995, 1, 3), code=0, value=34.5),
#     piton.Event(
#         start=datetime.datetime(2010, 1, 1),
#         code=1,
#         value="test_value",
#     ),
#     piton.Event(start=datetime.datetime(2010, 1, 5), code=2, value=1),
#     piton.Event(start=datetime.datetime(2010, 6, 5), code=3, value=True),
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
#         start=datetime.datetime(2016, 3, 1, 10, 10, 10), code=2, value=None
#     ),
# ]


def save_to_pkl(object_to_save, path_to_file: str):
    """Save object to Pickle file."""
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "wb") as fd:
        pickle.dump(object_to_save, fd)

def load_from_pkl(path_to_file: str):
    """Load object from Pickle file."""
    with open(path_to_file, "rb") as fd:
        result = pickle.load(fd)
    return result

# def create_patients(events: List[piton.Event]) -> List[piton.Patient]:
#     patients: List[piton.Patient] = []
#     for patient_id in range(NUM_PATIENTS):
#         patients.append(
#             piton.Patient(
#                 patient_id,
#                 events,
#             )
#         )
#     return patients


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
        assert (
            Label(
                value=bool(label_numpy[1][i]),
                time=label_numpy[2][i],
            )
            in labeled_patients[patient_id]
        )


def test_labeled_patients(tmp_path: pathlib.Path) -> None:
    """Checks internal methods of `LabeledPatient`"""
    patients = create_patients(tmp_path)
    true_labels = [
        # Assumes time horizon (0, 180) days + Code 2
        True, False, False
    ]

    time_horizon_6_months = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=180)
    )
    labeler = CodeLF(3, 2, time_horizon_6_months)

    patients_to_labels = {}

    for patient in patients:
        labels = labeler.label(patient)

        if len(labels) > 0:
            patients_to_labels[patient.patient_id] = labels

    labeled_patients = LabeledPatients(patients_to_labels, labeler.get_labeler_type())

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
    path = "../tmp/test_labelers/CodeLF.pkl"
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
    patients = create_patients(SHARED_EVENTS)
    true_labels = [
        # Assumes time horizon (0, 180) days + MortalityLF() where code is 3
        False,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        True,
    ]

    # 6. Create `Ontology` stub and run `MortalityPredictor`
    class DummyOntology:
        def get_dictionary(self):
            return [
                memoryview("zero".encode("utf8")),
                memoryview("one".encode("utf8")),
                memoryview("Visit/IP".encode("utf8")),
                memoryview("Condition Type/OMOP4822053".encode("utf8")),
                memoryview("four".encode("utf8")),
            ]

    dummy_ontology = DummyOntology()
    ontology = cast(piton.datasets.Ontology, dummy_ontology)
    time_horizon_4_to_12_months = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=180)
    )
    labeler = MortalityLF(ontology, time_horizon_4_to_12_months)
    labeled_patients = labeler.apply(patients)

    # Selected the right code
    assert labeler.code == 3
    # All label values are correct -- assumes all patients have same events
    for patient_id in labeled_patients.keys():
        assert_labels_are_accurate(labeled_patients, patient_id, true_labels)


def test_code_lf() -> None:
    """Creates a CodeLF for code '2' with time horizon of (0,180 days)."""
    patients = create_patients(SHARED_EVENTS)
    true_labels = [
        # Assumes time horizon (0, 180) days + CodeLF(2)
        False,
        True,
        True,
        False,
        True,
        True,
        False,
        True,
        True,
        False,
        True,
        True,
    ]

    # Create a CodeLF for Code 2
    time_horizon_6_months = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=180)
    )
    labeler = CodeLF(2, time_horizon_6_months)

    # Check CodeLF's internal functions
    assert labeler.get_time_horizon() == time_horizon_6_months
    for p in patients:
        assert labeler.get_outcome_times(p) == [
            event.start for event in p.events if event.code == 2
        ]

    # Label patients and check that they're correct
    labeled_patients: LabeledPatients = labeler.apply(patients)

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
            create_patients(
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
                        start=datetime.datetime(2018, 3, 3), code=2, value=None
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
                        start=datetime.datetime(2018, 5, 4), code=1, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2018, 12, 4), code=1, value=None
                    ),
                ]
            ),
            [
                True,
                True,
                False,
                True,
                True,
                True,
                False,
                False,
            ],
        ],
        [
            # Test #1
            # (1, 180) days
            TimeHorizon(
                datetime.timedelta(days=1), datetime.timedelta(days=180)
            ),
            create_patients(
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
                        start=datetime.datetime(2018, 1, 3), code=1, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2018, 3, 3), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2018, 5, 3), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2018, 5, 3, 11),
                        code=2,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2018, 5, 4), code=1, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2018, 12, 4), code=1, value=None
                    ),
                ]
            ),
            [
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
            ],
        ],
        [
            # Test #2
            # (180, 365) days
            TimeHorizon(
                datetime.timedelta(days=180), datetime.timedelta(days=365)
            ),
            create_patients(
                [
                    piton.Event(
                        start=datetime.datetime(2000, 1, 3), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2000, 10, 5), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2002, 1, 5), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2002, 4, 5), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2002, 12, 5), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2002, 12, 10),
                        code=1,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2004, 1, 10), code=2, value=None
                    ),
                ]
            ),
            [
                True,
                False,
                True,
                True,
                False,
                False,
            ],
        ],
        [
            # Test #3
            # (0, 0) days
            TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=0)),
            create_patients(
                [
                    piton.Event(
                        start=datetime.datetime(2015, 1, 3), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 1, 4), code=1, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 1, 5), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 1, 5, 10),
                        code=1,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 1, 6), code=2, value=None
                    ),
                ]
            ),
            [True, False, True, False, True],
        ],
        [
            # Test #4
            # (10, 10) days
            TimeHorizon(
                datetime.timedelta(days=10), datetime.timedelta(days=10)
            ),
            create_patients(
                [
                    piton.Event(
                        start=datetime.datetime(2015, 1, 3), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 1, 13), code=1, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 1, 23), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 2, 2), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 3, 10), code=1, value=None
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
            [False, True, True, False, True, False],
        ],
        [
            # Test #5
            # (0, 1e6) days
            TimeHorizon(
                datetime.timedelta(days=0), datetime.timedelta(days=1e6)
            ),
            create_patients(
                [
                    piton.Event(
                        start=datetime.datetime(2000, 1, 3), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2001, 10, 5), code=1, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2020, 10, 5), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2021, 10, 5), code=1, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2050, 1, 10), code=2, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(2051, 1, 10), code=1, value=None
                    ),
                    piton.Event(
                        start=datetime.datetime(5000, 1, 10), code=1, value=None
                    ),
                ]
            ),
            [
                True,
                True,
                True,
                True,
                True,
                False,
            ],
        ],
        [
            # Test #6
            # (5 hours, 10.5 hours)
            TimeHorizon(
                datetime.timedelta(hours=5),
                datetime.timedelta(hours=10, minutes=30),
            ),
            create_patients(
                [
                    piton.Event(
                        start=datetime.datetime(2015, 1, 1, 0, 0),
                        code=1,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 1, 1, 10, 29),
                        code=2,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 1, 1, 10, 30),
                        code=1,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2015, 1, 1, 10, 31),
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
                        code=1,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2017, 1, 1, 10, 31),
                        code=2,
                        value=None,
                    ),
                    #
                    piton.Event(
                        start=datetime.datetime(2018, 1, 1, 0, 0),
                        code=1,
                        value=None,
                    ),
                    piton.Event(
                        start=datetime.datetime(2018, 1, 1, 4, 59, 59),
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
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
        ],
    ]
    for test_idx, test in enumerate(tests):
        horizon, patients, true_labels = test
        labeler = CodeLF(2, horizon)
        labeled_patients = labeler.apply(patients)
        for i in range(len(patients)):
            # all patients have same events
            assert_labels_are_accurate(
                labeled_patients,
                i,
                true_labels,
                help_text=f" | test #{test_idx}",
            )


# # `LabeledPatients` class
# test_labeled_patients()

# # Labeling functions
# test_code_lf()
# test_mortality_lf()

# # Time Horizons
# test_time_horizons()
