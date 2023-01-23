import datetime
import pathlib
from typing import List

import piton.datasets
from piton.labelers.core import TimeHorizon
from piton.labelers.omop_lab_values import (
    OMOPConceptOutcomeFromLabValueLabeler,
    ThrombocytopeniaLabValueLabeler,
    HyperkalemiaLabValueLabeler,
    HypoglycemiaLabValueLabeler,
    HyponatremiaLabValueLabeler,
    AnemiaLabValueLabeler,
    NeutropeniaLabValueLabeler,
    AcuteKidneyInjuryLabValueLabeler,
)
from tools import (
    EventsWithLabels,
    event,
    run_test_for_labeler,
    run_test_locally,
)


class DummyOntology:
    def get_dictionary(self):
        return [
            "zero",
            "one",
            "Visit/IP",
            "OMOP_CONCEPT_A",
            "OMOP_CONCEPT_B",
            "five",
            "six",
            "OMOP_CONCEPT_A_CHILD",
            "OMOP_CONCEPT_A_CHILD_CHILD",
            "nine",
            "ten",
            "eleven",
            "OMOP_CONCEPT_B_CHILD",
            "OMOP_CONCEPT_A_CHILD",
            "fourteen",
        ]

    def get_children(self, parent_code: int) -> List[int]:
        if parent_code == 3:
            return [7, 13]
        elif parent_code == 4:
            return [12]
        elif parent_code == 7:
            return [8]
        else:
            return []


class DummyLabeler1(OMOPConceptOutcomeFromLabValueLabeler):
    original_omop_concept_codes = [
        "OMOP_CONCEPT_A",
        "OMOP_CONCEPT_B",
    ]

    def value_to_label(self, value: float) -> str:
        if value < 50:
            return "severe"
        elif value < 100:
            return "moderate"
        elif value < 150:
            return "mild"
        return "normal"

    def normalize_value_with_units(self, value: float, unit: str) -> float:
        if unit == "mmol/L":
            # mmol/L
            # Original OMOP concept ID: 8753
            return value
        elif unit == "mEq/L":
            # mEq/L (1-to-1 -> mmol/L)
            # Original OMOP concept ID: 9557
            return value
        elif unit == "mg/dL":
            # mg / dL (divide by 18 to get mmol/L)
            # Original OMOP concept ID: 8840
            return value / 18
        raise ValueError(f"Unknown unit: {unit}")


class DummyLabeler2(OMOPConceptOutcomeFromLabValueLabeler):
    original_omop_concept_codes = [
        "OMOP_CONCEPT_B",
    ]

    def value_to_label(self, value: float) -> str:
        return "normal"

    def normalize_value_with_units(self, value: float, unit: str) -> float:
        return 0


def test_constructor():
    # Constructor 1
    time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=10)
    )
    ontology = DummyOntology()
    labeler = DummyLabeler1(ontology, time_horizon, "severe")  # type: ignore
    assert labeler.get_time_horizon() == time_horizon
    assert labeler.severity == "severe"
    assert set(labeler.outcome_codes) == {3,4,7,8,12,13}

    # Constructor 2
    time_horizon = TimeHorizon(
        datetime.timedelta(hours=1), datetime.timedelta(hours=12)
    )
    labeler = DummyLabeler2(ontology, time_horizon, "normal")
    assert labeler.get_time_horizon() == time_horizon
    assert labeler.severity == "normal"
    assert set(labeler.outcome_codes) == {4,12}


def test_labeling(tmp_path: pathlib.Path):
    time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=10)
    )
    ontology = DummyOntology()
    labeler = DummyThrombocytopeniaLabeler(ontology, time_horizon, "severe")  # type: ignore

    # Create patient
    events_with_labels: EventsWithLabels = [
        (event((2000, 1, 3), 0, None), "skip"),
        (event((2000, 1, 4), 7, None), "skip"),  # lab test
        (event((2000, 1, 5), 1, None), "skip"),
        (event((2002, 1, 3), 2, None), False),  # admission
        (event((2002, 10, 1), 2, None), True),  # admission
        (event((2002, 10, 5, 0), 3, None), "skip"),  # lab test
        (event((2002, 10, 5, 1), 5, None), "skip"),
        (event((2002, 10, 5, 2), 4, 20.5, unit="mmol/L"), "skip"),  # lab test
        (event((2002, 10, 5, 3), 6, None), "skip"),
        (event((2004, 3, 1), 2, None), True),  # admission
        (event((2004, 3, 2), 12, -20.5, unit="mmol/L"), "skip"),  # lab test
        (event((2004, 9, 1), 2, None), False),  # admission
        (event((2004, 9, 2), 12, 200.5, unit="mmol/L"), "skip"),  # lab test
        (event((2006, 5, 1), 2, None), False),  # admission
        (event((2006, 5, 2), 0, None), "skip"),
        (event((2006, 5, 3), 9, None), "skip"),
        (event((2006, 5, 3, 11), 8, None), "skip"),  # lab test
        (event((2008, 1, 3), 2, None), False),  # admission
        (event((2008, 1, 4), 12, 75.5, unit="mmol/L"), "skip"),  # lab test
        (event((2008, 11, 1), 2, None), True),  # admission
        (event((2008, 11, 4), 12, 45.5, unit="mmol/L"), "skip"),  # lab test
        (event((2008, 12, 30), 2, None), False),  # admission
        (event((2010, 1, 3), 2, None), None),  # admission
        (event((2010, 1, 4), 13, 125.5, unit="mmol/L"), "skip"),  # lab test
    ]
    patient = piton.Patient(0, [x[0] for x in events_with_labels])
    true_outcome_times: List[datetime.datetime] = [
        events_with_labels[7][0].start,
        events_with_labels[10][0].start,
        events_with_labels[20][0].start,
    ]
    assert (
        labeler.get_outcome_times(patient) == true_outcome_times
    ), f"{labeler.get_outcome_times(patient)}"
    assert labeler.get_prediction_times(patient) == [
        events_with_labels[3][0].start,
        events_with_labels[4][0].start,
        events_with_labels[9][0].start,
        events_with_labels[11][0].start,
        events_with_labels[13][0].start,
        events_with_labels[17][0].start,
        events_with_labels[19][0].start,
        events_with_labels[21][0].start,
        events_with_labels[22][0].start,
    ], f"{labeler.get_prediction_times(patient)}"

    run_test_for_labeler(
        labeler,
        events_with_labels,
        true_outcome_times=true_outcome_times,
        help_text="test_labeling",
    )


def test_units(tmp_path: pathlib.Path):
    # TODO: test unit normalization
    pass

def test_thrombocytopenia(tmp_path: pathlib.Path):
    # TODO
    return
    class DummyOntology:
        def get_dictionary(self):
            return [
                "zero",
                "Visit/IP",
                "OMOP_CONCEPT_A",
            ]
        def get_children(self, parent_code: int) -> List[int]:
            return [7, 13] if parent_code in [] else []

    time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=10)
    )
    ontology = DummyOntology()
    labeler = ThrombocytopeniaLabeler(ontology, time_horizon, "severe")  # type: ignore

    # Create patient
    events_with_labels: EventsWithLabels = [
        (event((2000, 1, 4), 7, None), "skip"),  # lab test
    ]
    true_outcome_times: List[datetime.datetime] = [
        events_with_labels[7][0].start,
        events_with_labels[10][0].start,
        events_with_labels[20][0].start,
    ]
    run_test_for_labeler(
        labeler,
        events_with_labels,
        true_outcome_times=true_outcome_times,
        help_text="test_thrombocytopenia",
    )

def test_hyperkalemia(tmp_path: pathlib.Path):
    # TODO
    pass
def test_hypoglycemia(tmp_path: pathlib.Path):
    # TODO
    pass
def test_hyponatremia(tmp_path: pathlib.Path):
    # TODO
    pass
def test_anemia(tmp_path: pathlib.Path):
    # TODO
    pass
def test_neutropenia(tmp_path: pathlib.Path):
    # TODO
    pass
def test_acuteKidneyInjury(tmp_path: pathlib.Path):
    # TODO
    pass

# Local testing
if __name__ == "__main__":
    run_test_locally("../ignore/test_labelers/", test_constructor)
    run_test_locally("../ignore/test_labelers/", test_labeling)
    run_test_locally("../ignore/test_labelers/", test_units)
