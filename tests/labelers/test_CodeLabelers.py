# flake8: noqa: E402
# mypy: ignore-errors
import datetime
import os
import pathlib
import sys
from typing import List

from femr.labelers import TimeHorizon
from femr.labelers.omop_lab_values import (
    AKICodeLabeler,
    AnemiaCodeLabeler,
    HyperkalemiaCodeLabeler,
    HypoglycemiaCodeLabeler,
    HyponatremiaCodeLabeler,
    NeutropeniaCodeLabeler,
    ThrombocytopeniaCodeLabeler,
)
from femr.labelers.omop import (
    CodeLabeler,
    LupusCodeLabeler,
    MortalityCodeLabeler,
    OMOPConceptCodeLabeler,
    get_death_concepts,
)

# Needed to import `tools` for local testing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import EventsWithLabels, event, run_test_for_labeler, run_test_locally

#############################################
#############################################
#
# Generic CodeLabeler
#
#############################################
#############################################


def test_outcome_codes(tmp_path: pathlib.Path):
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=10))
    # One outcome
    labeler = CodeLabeler(["2"], time_horizon)
    events_with_labels: EventsWithLabels = [
        (event((2015, 1, 3), 2, None), "duplicate"),
        (event((2015, 1, 3), 4, None), "duplicate"),
        (event((2015, 1, 3), 1, None), "duplicate"),
        (event((2015, 1, 3), 3, None), "skip"),
        (event((2015, 10, 5), 1, None), False),
        (event((2018, 1, 3), 2, None), "skip"),
        (event((2018, 3, 1), 4, None), False),
        (event((2018, 3, 3), 1, None), False),
        (event((2018, 5, 2), 5, None), True),
        (event((2018, 5, 3), 2, None), "skip"),
        (event((2018, 5, 3, 11), 1, None), False),
        (event((2018, 5, 4), 1, None), "duplicate"),
        (event((2018, 5, 4), 4, None), False),
        (event((2018, 11, 1), 5, None), False),
        (event((2018, 12, 4), 1, None), False),
        (event((2018, 12, 30), 4, None), "out of range"),
    ]
    run_test_for_labeler(labeler, events_with_labels, help_text="test_outcome_codes_one")

    # Zero outcome
    labeler = CodeLabeler([], time_horizon)
    events_with_labels = [
        (event((2015, 1, 3), 2, None), "duplicate"),
        (event((2015, 1, 3), 4, None), "duplicate"),
        (event((2015, 1, 3), 1, None), "duplicate"),
        (event((2015, 1, 3), 3, None), False),
        (event((2015, 10, 5), 1, None), False),
        (event((2018, 1, 3), 2, None), False),
        (event((2018, 3, 1), 4, None), False),
        (event((2018, 3, 3), 1, None), False),
        (event((2018, 5, 2), 5, None), False),
        (event((2018, 5, 3), 2, None), False),
        (event((2018, 5, 3, 11), 1, None), False),
        (event((2018, 5, 4), 1, None), "duplicate"),
        (event((2018, 5, 4), 4, None), False),
        (event((2018, 11, 1), 5, None), False),
        (event((2018, 12, 4), 1, None), False),
        (event((2018, 12, 30), 4, None), "out of range"),
    ]
    run_test_for_labeler(labeler, events_with_labels, help_text="test_outcome_codes_zero")

    # Multiple outcomes
    labeler = CodeLabeler(["1", "4"], time_horizon)
    events_with_labels = [
        (event((2015, 1, 3), 2, None), "duplicate"),
        (event((2015, 1, 3), 4, None), "duplicate"),
        (event((2015, 1, 3), 1, None), "duplicate"),
        (event((2015, 1, 3), 3, None), "skip"),
        (event((2015, 10, 5), 1, None), "skip"),
        (event((2018, 1, 3), 2, None), False),
        (event((2018, 3, 1), 4, None), "skip"),
        (event((2018, 3, 3), 1, None), "skip"),
        (event((2018, 5, 2), 5, None), False),
        (event((2018, 5, 3), 2, None), False),
        (event((2018, 6, 2), 0, None), True),
        (event((2018, 6, 3, 11), 1, None), "skip"),
        (event((2018, 6, 3, 23), 3, None), False),
        (event((2018, 9, 1), 3, None), True),
        (event((2018, 9, 4), 4, None), "skip"),
        (event((2018, 11, 1), 5, None), False),
        (event((2018, 12, 3), 0, None), True),
        (event((2018, 12, 4), 4, None), "skip"),
        (event((2018, 12, 30), 0, None), "out of range"),
    ]
    run_test_for_labeler(labeler, events_with_labels, help_text="test_outcome_codes_multiple")


def test_prediction_codes(tmp_path: pathlib.Path):
    # One outcome + multiple predictions
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=10))
    labeler = CodeLabeler(["2"], time_horizon, prediction_codes=["4", "5"])
    events_with_labels: EventsWithLabels = [
        (event((2015, 1, 3), 2, None), "skip"),
        (event((2015, 1, 3), 4, None), "skip"),
        (event((2015, 1, 3), 1, None), "skip"),
        (event((2015, 1, 3), 3, None), "skip"),
        (event((2015, 10, 5), 1, None), "skip"),
        (event((2018, 1, 3), 2, None), "skip"),
        (event((2018, 3, 1), 4, None), False),
        (event((2018, 3, 3), 1, None), "skip"),
        (event((2018, 5, 2), 5, None), True),
        (event((2018, 5, 3), 2, None), "skip"),
        (event((2018, 5, 3, 11), 1, None), "skip"),
        (event((2018, 5, 4), 4, None), False),
        (event((2018, 5, 4), 1, None), "skip"),
        (event((2018, 11, 1), 5, None), False),
        (event((2018, 12, 4), 1, None), "skip"),
        (event((2018, 12, 30), 4, None), "out of range"),
    ]
    run_test_for_labeler(labeler, events_with_labels, help_text="prediction_codes_one_outcomes")

    # Multiple outcomes + multiple predictions
    labeler = CodeLabeler(["2", "6", "7"], time_horizon, prediction_codes=["4", "5"])
    events_with_labels = [
        (event((2010, 1, 1), 2, None), "skip"),
        (event((2010, 1, 3), 4, None), True),
        (event((2010, 1, 8), 6, None), "skip"),
        (event((2010, 2, 1), 5, None), True),
        (event((2010, 2, 9), 7, None), "skip"),
        (event((2010, 2, 11), 4, None), False),
        (event((2015, 1, 3), 2, None), "skip"),
        (event((2015, 1, 3), 4, None), "skip"),
        (event((2015, 1, 3), 1, None), "skip"),
        (event((2015, 1, 3), 3, None), "skip"),
        (event((2015, 10, 5), 1, None), "skip"),
        (event((2018, 1, 3), 2, None), "skip"),
        (event((2018, 3, 1), 4, None), True),
        (event((2018, 3, 2), 7, None), "skip"),
        (event((2018, 3, 3), 1, None), "skip"),
        (event((2018, 5, 2), 5, None), True),
        (event((2018, 5, 3), 2, None), "skip"),
        (event((2018, 5, 3, 11), 1, None), "skip"),
        (event((2018, 5, 4), 4, None), False),
        (event((2018, 5, 4), 1, None), "skip"),
        (event((2018, 11, 1), 5, None), False),
        (event((2018, 12, 4), 1, None), "skip"),
        (event((2018, 12, 30), 4, None), "out of range"),
    ]
    run_test_for_labeler(
        labeler,
        events_with_labels,
        help_text="prediction_codes_multiple_outcomes",
    )

    # Multiple outcomes + no predictions
    labeler = CodeLabeler([2, 6, 7], time_horizon, prediction_codes=[])
    events_with_labels = [
        (event((2010, 1, 1), 2, None), "skip"),
        (event((2010, 1, 3), 4, None), "skip"),
        (event((2010, 1, 8), 6, None), "skip"),
        (event((2010, 2, 1), 5, None), "skip"),
        (event((2010, 2, 9), 7, None), "skip"),
        (event((2010, 2, 11), 4, None), "skip"),
        (event((2015, 1, 3), 2, None), "skip"),
        (event((2015, 1, 3), 4, None), "skip"),
        (event((2015, 1, 3), 1, None), "skip"),
        (event((2015, 1, 3), 3, None), "skip"),
        (event((2015, 10, 5), 1, None), "skip"),
        (event((2018, 1, 3), 2, None), "skip"),
        (event((2018, 3, 1), 4, None), "skip"),
        (event((2018, 3, 2), 7, None), "skip"),
        (event((2018, 3, 3), 1, None), "skip"),
        (event((2018, 5, 2), 5, None), "skip"),
        (event((2018, 5, 3), 2, None), "skip"),
        (event((2018, 5, 3, 11), 1, None), "skip"),
        (event((2018, 5, 4), 4, None), "skip"),
        (event((2018, 5, 4), 1, None), "skip"),
        (event((2018, 11, 1), 5, None), "skip"),
        (event((2018, 12, 4), 1, None), "skip"),
        (event((2018, 12, 30), 4, None), "skip"),
    ]
    run_test_for_labeler(
        labeler,
        events_with_labels,
        help_text="prediction_codes_zero_predictions",
    )


#############################################
#############################################
#
# Generic OMOPConceptCodeLabeler
#
#############################################
#############################################


class DummyOntology_OMOPConcept:
    def get_children(self, parent_code: str) -> List[str]:
        if parent_code == "OMOP_CONCEPT_A":
            return ["OMOP_CONCEPT_A_CHILD", "OMOP_CONCEPT_A_CHILD2"]
        elif parent_code == "OMOP_CONCEPT_B":
            return ["OMOP_CONCEPT_B_CHILD"]
        elif parent_code == "OMOP_CONCEPT_A_CHILD":
            return ["OMOP_CONCEPT_A_CHILD_CHILD"]
        else:
            return []


class DummyLabeler_OMOPConcept(OMOPConceptCodeLabeler):
    original_omop_concept_codes = [
        "OMOP_CONCEPT_A",
        "OMOP_CONCEPT_B",
    ]


def test_omop_concept_code_labeler(tmp_path: pathlib.Path):
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=10))
    ontology = DummyOntology_OMOPConcept()
    labeler = DummyLabeler_OMOPConcept(ontology, time_horizon, prediction_codes=["1", "2"])  # type: ignore
    assert set(labeler.outcome_codes) == {
        "OMOP_CONCEPT_A_CHILD_CHILD",
        "OMOP_CONCEPT_B",
        "OMOP_CONCEPT_B_CHILD",
        "OMOP_CONCEPT_A_CHILD2",
        "OMOP_CONCEPT_A",
        "OMOP_CONCEPT_A_CHILD",
    }
    assert labeler.prediction_codes == ["1", "2"]
    assert labeler.get_time_horizon() == time_horizon


#############################################
#############################################
#
# Specific instances of CodeLabeler
#
#############################################
#############################################


#############################################
# MortalityCodeLabeler
#############################################


class DummyOntology_Mortality:
    def get_children(self, parent_code: int) -> List[int]:
        return []


def test_death_concepts() -> None:
    expected_death_concepts: set = {
        "Death Type/OMOP generated",
        "Condition Type/OMOP4822053",
    }
    assert set(get_death_concepts()) == expected_death_concepts


def test_MortalityCodeLabeler() -> None:
    """Create a MortalityCodeLabeler for codes 3 and 6"""
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))
    events_with_labels: EventsWithLabels = [
        (event((1995, 1, 3), 0, 34.5), False),
        (event((2000, 1, 1), 1, "test_value"), True),
        (event((2000, 1, 5), 2, 1), True),
        (event((2000, 6, 5), "Condition Type/OMOP4822053", True), "skip"),
        (event((2005, 2, 5), 2, None), False),
        (event((2005, 7, 5), 2, None), False),
        (event((2010, 10, 5), 1, None), False),
        (event((2015, 2, 5, 0), 2, None), False),
        (event((2015, 7, 5, 0), 0, None), True),
        (event((2015, 11, 5, 10, 10), 2, None), True),
        (event((2015, 11, 15, 11), "Death Type/OMOP generated", None), "skip"),
        (event((2020, 1, 1), 2, None), "out of range"),
        (event((2020, 3, 1, 10, 10, 10), 2, None), "out of range"),
    ]

    ontology = DummyOntology_Mortality()

    # Run labeler
    labeler = MortalityCodeLabeler(ontology, time_horizon)  # type: ignore

    # Check that we selected the right codes
    assert set(labeler.outcome_codes) == {"Death Type/OMOP generated", "Condition Type/OMOP4822053"}

    run_test_for_labeler(labeler, events_with_labels, help_text="MortalityLabeler")


#############################################
# LupusCodeLabeler
#############################################


class DummyOntology_Lupus:
    def get_children(self, parent_code: int) -> List[int]:
        if parent_code == "SNOMED/55464009":
            return {"SNOMED_55464009", "Lupus_child_seven", "Lupus_child_nine", "Lupus_child_ten"}
        else:
            return []


def test_LupusCodeLabeler() -> None:
    """Create a LupusCodeLabeler for codes 3 and 6"""
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=180))
    events_with_labels: EventsWithLabels = [
        (event((1995, 1, 3), 0, 34.5), False),
        (event((2000, 1, 1), 1, "test_value"), True),
        (event((2000, 1, 5), 2, 1), True),
        (event((2000, 5, 5), "SNOMED/201436003", None), "skip"),
        (event((2005, 2, 5), 2, None), False),
        (event((2005, 7, 5), 2, None), False),
        (event((2010, 10, 5), 1, None), True),
        (event((2010, 10, 8), "Lupus_child_seven", None), "skip"),
        (event((2015, 2, 5, 0), 2, None), False),
        (event((2015, 7, 5, 0), 0, None), True),
        (event((2015, 11, 5, 10, 10), 2, None), True),
        (event((2015, 11, 15, 11), "SNOMED/55464009", None), "skip"),
        (event((2020, 1, 1), "Lupus_child_ten", None), "skip"),
        (event((2020, 3, 1, 10, 10, 10), 2, None), "out of range"),
    ]

    ontology = DummyOntology_Lupus()
    labeler = LupusCodeLabeler(ontology, time_horizon)  # type: ignore
    # Check that we selected the right codes
    assert set(labeler.outcome_codes) == {
        "SNOMED_55464009",
        "SNOMED/201436003",
        "Lupus_child_nine",
        "SNOMED/55464009",
        "Lupus_child_ten",
        "Lupus_child_seven",
    }

    run_test_for_labeler(labeler, events_with_labels, help_text="LupusCodeLabeler")


#############################################
#############################################
#
# Specific instances of OMOPConceptCodeLabeler
#
#############################################
#############################################


class DummyOntology_OMOPConcept_Specific:
    def __init__(self, new_codes: List[str]):
        self.new_codes = new_codes + ["", ""]

    def get_children(self, parent_code: str) -> List[str]:
        if parent_code == "child_1":
            return ["child_1_1"]
        elif parent_code == self.new_codes[0]:
            return ["child_1"]
        elif parent_code == self.new_codes[1]:
            return ["child_2"]
        return []


def _assert_labvalue_constructor_correct(
    labeler: OMOPConceptCodeLabeler,
    time_horizon: TimeHorizon,
    outcome_codes: set,
):
    print(set(labeler.outcome_codes))
    assert set(labeler.outcome_codes) == outcome_codes
    assert labeler.prediction_codes == ["1", "2"]
    assert labeler.get_time_horizon() == time_horizon


def _create_specific_labvalue_labeler(LabelerClass, outcome_codes: set):
    time_horizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=10))
    ontology = DummyOntology_OMOPConcept_Specific(LabelerClass.original_omop_concept_codes)
    labeler = LabelerClass(ontology, time_horizon, prediction_codes=["1", "2"])  # type: ignore
    _assert_labvalue_constructor_correct(labeler, time_horizon, outcome_codes)
    return labeler


def test_thrombocytopenia(tmp_path: pathlib.Path):
    outcome_codes = {"child_1_1", "child_2", "SNOMED/89627008", "child_1", "SNOMED/267447008"}
    _create_specific_labvalue_labeler(ThrombocytopeniaCodeLabeler, outcome_codes)


def test_hyperkalemia(tmp_path: pathlib.Path):
    outcome_codes = {"child_1", "SNOMED/14140009", "child_1_1"}
    _create_specific_labvalue_labeler(HyperkalemiaCodeLabeler, outcome_codes)


def test_hypoglycemia(tmp_path: pathlib.Path):
    outcome_codes = {
        "SNOMED/120731000119103",
        "child_2",
        "child_1",
        "SNOMED/52767006",
        "SNOMED/719216001",
        "SNOMED/302866003",
        "SNOMED/267384006",
        "SNOMED/421725003",
        "SNOMED/237637005",
        "SNOMED/237633009",
        "SNOMED/190448007",
        "child_1_1",
        "SNOMED/421437000",
        "SNOMED/230796005",
        "SNOMED/84371000119108",
    }
    _create_specific_labvalue_labeler(HypoglycemiaCodeLabeler, outcome_codes)


def test_hyponatremia(tmp_path: pathlib.Path):
    outcome_codes = {"child_2", "child_1", "SNOMED/89627008", "SNOMED/267447008", "child_1_1"}
    _create_specific_labvalue_labeler(HyponatremiaCodeLabeler, outcome_codes)


def test_anemia(tmp_path: pathlib.Path):
    outcome_codes = {
        "child_1",
        "SNOMED/713496008",
        "SNOMED/691411000119101",
        "SNOMED/691401000119104",
        "SNOMED/767657005",
        "child_2",
        "SNOMED/111570005",
        "SNOMED/271737000",
        "SNOMED/713349004",
        "child_1_1",
    }
    _create_specific_labvalue_labeler(AnemiaCodeLabeler, outcome_codes)


def test_neutropenia(tmp_path: pathlib.Path):
    outcome_codes = {"child_1", "SNOMED/165517008", "child_1_1"}
    _create_specific_labvalue_labeler(NeutropeniaCodeLabeler, outcome_codes)


def test_aki(tmp_path: pathlib.Path):
    outcome_codes = {"child_2", "child_1_1", "child_1", "SNOMED/298015003", "SNOMED/14669001", "SNOMED/35455006"}
    _create_specific_labvalue_labeler(AKICodeLabeler, outcome_codes)


# Local testing
if __name__ == "__main__":
    run_test_locally("../ignore/test_labelers/", test_prediction_codes)
    run_test_locally("../ignore/test_labelers/", test_outcome_codes)
    run_test_locally("../ignore/test_labelers/", test_omop_concept_code_labeler)
    run_test_locally("../ignore/test_labelers/", test_MortalityCodeLabeler)
    run_test_locally("../ignore/test_labelers/", test_LupusCodeLabeler)
    run_test_locally("../ignore/test_labelers/", test_death_concepts)
    run_test_locally("../ignore/test_labelers/", test_thrombocytopenia)
    run_test_locally("../ignore/test_labelers/", test_hyperkalemia)
    run_test_locally("../ignore/test_labelers/", test_hypoglycemia)
    run_test_locally("../ignore/test_labelers/", test_hyponatremia)
    run_test_locally("../ignore/test_labelers/", test_anemia)
    run_test_locally("../ignore/test_labelers/", test_neutropenia)
    run_test_locally("../ignore/test_labelers/", test_aki)
