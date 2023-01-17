import datetime
import os
import pathlib
import pickle
from typing import List, Optional, Union, cast
import shutil

import numpy as np

import piton
import piton.datasets
from piton.labelers.core import Label, LabeledPatients, TimeHorizon
from piton.labelers.omop import CodeLabeler, MortalityCodeLabeler

NUM_PATIENTS = 5

SHARED_EVENTS = [
    piton.Event(start=datetime.datetime(1995, 1, 3), code=0, value=34.5),
    piton.Event(
        start=datetime.datetime(2010, 1, 1),
        code=1,
        value="test_value",
    ),
    piton.Event(start=datetime.datetime(2010, 1, 5), code=2, value=1),
    piton.Event(start=datetime.datetime(2010, 6, 5), code=3, value=True),
    piton.Event(start=datetime.datetime(2011, 2, 5), code=2, value=None),
    piton.Event(start=datetime.datetime(2011, 7, 5), code=2, value=None),
    piton.Event(start=datetime.datetime(2012, 10, 5), code=3, value=None),
    piton.Event(start=datetime.datetime(2015, 6, 5, 0), code=2, value=None),
    piton.Event(
        start=datetime.datetime(2015, 6, 5, 10, 10), code=2, value=None
    ),
    piton.Event(start=datetime.datetime(2015, 6, 15, 11), code=3, value=None),
    piton.Event(start=datetime.datetime(2016, 1, 1), code=2, value=None),
    piton.Event(
        start=datetime.datetime(2016, 3, 1, 10, 10, 10), code=2, value=None
    ),
]

def create_patients(events: List[piton.Event]) -> List[piton.Patient]:
    patients: List[piton.Patient] = []
    for patient_id in range(NUM_PATIENTS):
        patients.append(
            piton.Patient(
                patient_id,
                events,
            )
        )
    return patients

def assert_labels_are_accurate(
    labeled_patients: LabeledPatients,
    patient_id: int,
    true_labels: Union[List[Optional[bool]], List[bool]],
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

def test_MortalityCodeLabeler() -> None:
    """Creates a MortalityCodeLabeler for code 3, which corresponds to "Death Type/" """
    patients = create_patients(SHARED_EVENTS)
    true_labels = [
        # Assumes time horizon (0, 180) days + MortalityCodeLabeler() where code is 3
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

    # Create `Ontology` stub
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
    
    # Run labeler
    time_horizon_4_to_12_months = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=180)
    )
    labeler = MortalityCodeLabeler(ontology, time_horizon_4_to_12_months)
    labeled_patients = labeler.apply(patients)

    # Selected the right code
    assert labeler.code == 3
    # All label values are correct -- assumes all patients have same events
    for patient_id in labeled_patients.keys():
        assert_labels_are_accurate(labeled_patients, patient_id, true_labels)


def test_CodeLabeler() -> None:
    """Creates a CodeLabeler for code '2' with time horizon of (0,180 days)."""
    patients = create_patients(SHARED_EVENTS)
    true_labels = [
        # Assumes time horizon (0, 180) days + CodeLabeler(2)
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

    # Create a CodeLabeler for Code 2
    time_horizon_6_months = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=180)
    )
    labeler = CodeLabeler([2], time_horizon_6_months)

    # Check CodeLabeler's internal functions
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
        labeler = CodeLabeler([2], horizon)
        labeled_patients = labeler.apply(patients)
        for i in range(len(patients)):
            # all patients have same events
            assert_labels_are_accurate(
                labeled_patients,
                i,
                true_labels,
                help_text=f" | test #{test_idx}",
            )


tmp_path = '../ignore/test_labelers/'
shutil.rmtree(tmp_path)
os.makedirs(tmp_path, exist_ok=True)

# Labeling functions
test_CodeLabeler()
test_MortalityCodeLabeler()
test_OMOPConceptCodeLabeler()
test_WithinVisitLabeler()
test_LupusDiseaseCodeLabeler()
test_HighHbA1cCodeLabeler()
test_HypoglycemiaCodeLabeler()
test_OpioidOverdoseLabeler()
test_CeliacTestLabeler()
test_IsMaleLabeler()

# Time Horizons
test_time_horizons()
