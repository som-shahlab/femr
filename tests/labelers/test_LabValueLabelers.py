# flake8: noqa: E402
import datetime
import itertools
import os
import pathlib
import sys
from typing import List, Optional, Tuple

import pytest

import femr.datasets
from femr.labelers import TimeHorizon
from femr.labelers.omop import move_datetime_to_end_of_day
from femr.labelers.omop_lab_values import (
    AnemiaLabValueLabeler,
    HyperkalemiaLabValueLabeler,
    HypoglycemiaLabValueLabeler,
    HyponatremiaLabValueLabeler,
    InpatientLabValueLabeler,
    ThrombocytopeniaLabValueLabeler,
)

# Needed to import `tools` for local testing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import EventsWithLabels, event, run_test_for_labeler, run_test_locally

#############################################
#############################################
#
# Generic InpatientLabValueLabeler
#
# #############################################
#############################################


class DummyOntology_Generic:
    def get_children(self, parent_code: str) -> List[str]:
        if parent_code == "OMOP_CONCEPT_A":
            return ["OMOP_CONCEPT_A_CHILD", "OMOP_CONCEPT_A_CHILD2"]
        elif parent_code == "OMOP_CONCEPT_B":
            return ["OMOP_CONCEPT_B_CHILD"]
        elif parent_code == "OMOP_CONCEPT_A_CHILD":
            return ["OMOP_CONCEPT_A_CHILD_CHILD"]
        else:
            return []


class DummyLabeler1(InpatientLabValueLabeler):
    original_omop_concept_codes = [
        "OMOP_CONCEPT_A",
        "OMOP_CONCEPT_B",
    ]

    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        value = float(raw_value)
        if unit == "mmol/L":
            # mmol/L
            # Original OMOP concept ID: 8753
            value = value
        elif unit == "mEq/L":
            # mEq/L (1-to-1 -> mmol/L)
            # Original OMOP concept ID: 9557
            value = value
        elif unit == "mg/dL":
            # mg / dL (divide by 18 to get mmol/L)
            # Original OMOP concept ID: 8840
            value = value / 18
        else:
            raise ValueError(f"Unknown unit: {unit}")
        if value < 50:
            return "severe"
        elif value < 100:
            return "moderate"
        elif value < 150:
            return "mild"
        return "normal"


class DummyLabeler2(InpatientLabValueLabeler):
    original_omop_concept_codes = [
        "OMOP_CONCEPT_B",
    ]

    def value_to_label(self, raw_value: str, unit: Optional[str]) -> str:
        return "normal"


def test_constructor():
    # Constructor 1
    ontology = DummyOntology_Generic()
    labeler = DummyLabeler1(ontology, "severe")  # type: ignore
    assert labeler.severity == "severe"
    assert set(labeler.outcome_codes) == {
        "OMOP_CONCEPT_A",
        "OMOP_CONCEPT_A_CHILD2",
        "OMOP_CONCEPT_B_CHILD",
        "OMOP_CONCEPT_A_CHILD",
        "OMOP_CONCEPT_B",
        "OMOP_CONCEPT_A_CHILD_CHILD",
    }

    # Constructor 2
    labeler = DummyLabeler2(ontology, "normal")
    assert labeler.severity == "normal"
    assert set(labeler.outcome_codes) == {"OMOP_CONCEPT_B", "OMOP_CONCEPT_B_CHILD"}


def test_labeling(tmp_path: pathlib.Path):
    ontology = DummyOntology_Generic()
    labeler = DummyLabeler1(ontology, "severe")  # type: ignore

    # No prediction time adjustment
    events_with_labels: EventsWithLabels = [
        # fmt: off
        (event((2000, 1, 3), 0, None), "skip"),
        (event((2000, 1, 4), "OMOP_CONCEPT_A_CHILD", None), "skip"),  # lab test
        (event((2000, 1, 5), 2, None), "skip"),
        #
        (event((2002, 1, 3), "Visit/IP", end=datetime.datetime(2002, 10, 4), omop_table='visit_occurrence'), 'skip'), # admission
        #
        (event((2002, 10, 1), "Visit/IP", end=datetime.datetime(2002, 10, 10), omop_table='visit_occurrence'), True), # admission
        (event((2002, 10, 5, 0), "OMOP_CONCEPT_A", None), "skip"),  # lab test - 5
        (event((2002, 10, 5, 2), 5, None), "skip"),
        (event((2002, 10, 5, 2), "OMOP_CONCEPT_B", 20.5, unit="mmol/L"), "skip"),  # lab test - 7
        (event((2002, 10, 5, 3), 6, None), "skip"),
        #
        (event((2004, 3, 1), "Visit/IP", end=datetime.datetime(2004, 3, 2, 1), omop_table='visit_occurrence'), True), # admission
        (event((2004, 3, 2), "OMOP_CONCEPT_B_CHILD", -20.5, unit="mmol/L"), "skip"),  # lab test - 10
        #
        (event((2005, 9, 1), "Visit/IP", end=datetime.datetime(2005, 9, 4), omop_table='visit_occurrence'), False), # admission
        (event((2005, 9, 2), "OMOP_CONCEPT_B_CHILD", 200.5, unit="mmol/L"), "skip"),  # lab test - 12
        #
        (event((2006, 5, 1), "Visit/IP", end=datetime.datetime(2006, 5, 5), omop_table='visit_occurrence'), 'skip'), # admission
        (event((2006, 5, 2), 0, None), "skip"),
        (event((2006, 5, 3), 9, None), "skip"),
        (event((2006, 5, 3, 11), "OMOP_CONCEPT_A_CHILD_CHILD", None), "skip"),  # lab test - 16
        #
        (event((2008, 1, 3), "Visit/IP", end=datetime.datetime(2008, 1, 5), omop_table='visit_occurrence'), False), # admission
        (event((2008, 1, 4), "OMOP_CONCEPT_B_CHILD", 75.5, unit="mmol/L"), "skip"),  # lab test - 18
        #
        (event((2009, 11, 4), "Visit/IP", end=datetime.datetime(2009, 11, 4, 1), omop_table='visit_occurrence'), True), # admission
        (event((2009, 11, 4), "OMOP_CONCEPT_B_CHILD", 45.5, unit="mmol/L"), "skip"),  # lab test - 20
        #
        (event((2010, 10, 30), "Visit/IP", end=datetime.datetime(2010, 12, 4), omop_table='visit_occurrence'), 'skip'), # admission
        #
        (event((2011, 1, 3), "Visit/IP", end=datetime.datetime(2011, 1, 14), omop_table='visit_occurrence'), False), # admission
        (event((2011, 1, 4), "OMOP_CONCEPT_A_CHILD2", 125.5, unit="mmol/L"), "skip"),
        # fmt: on
    ]
    true_prediction_times: List[datetime.datetime] = [
        labeler.visit_start_adjust_func(x[0].start) for x in events_with_labels if isinstance(x[1], bool)
    ]
    true_outcome_times: List[datetime.datetime] = [
        events_with_labels[7][0].start,
        events_with_labels[10][0].start,
        events_with_labels[20][0].start,
    ]
    run_test_for_labeler(
        labeler,
        events_with_labels,
        true_prediction_times=true_prediction_times,
        true_outcome_times=true_outcome_times,
        help_text="test_labeling",
    )

    # Prediction time adjustment
    labeler = DummyLabeler1(ontology, "severe", visit_start_adjust_func=move_datetime_to_end_of_day)  # type: ignore
    events_with_labels = [
        # fmt: off
        # yes
        (event((2002, 1, 3), "Visit/IP", end=datetime.datetime(2002, 1, 10), omop_table='visit_occurrence'), True), # admission
        (event((2002, 1, 4), "OMOP_CONCEPT_B_CHILD", 45.5, unit="mmol/L"), "skip"),  # lab test - 1
        # no
        (event((2003, 1, 3), "Visit/IP", end=datetime.datetime(2003, 1, 10), omop_table='visit_occurrence'), False), # admission
        (event((2003, 1, 3), "OMOP_CONCEPT_B_CHILD", 45.5, unit="mmol/L"), "skip"),
        # lab test - 3
        # fmt: on
    ]
    true_prediction_times = [
        labeler.visit_start_adjust_func(x[0].start) for x in events_with_labels if isinstance(x[1], bool)
    ]
    true_outcome_times = [
        events_with_labels[1][0].start,
        events_with_labels[3][0].start,
    ]
    run_test_for_labeler(
        labeler,
        events_with_labels,
        true_prediction_times=true_prediction_times,
        true_outcome_times=true_outcome_times,
        help_text="test_labeling_shifted_prediction_time",
    )

    # Test fail cases
    with pytest.raises(RuntimeError):
        # error when visit_start_adjust_func() pushes `start` after `end`
        labeler = DummyLabeler1(ontology, "severe", visit_start_adjust_func=move_datetime_to_end_of_day)  # type: ignore
        events_with_labels = [
            # fmt: off
            (event((2009, 11, 4), "Visit/IP", end=datetime.datetime(2009, 11, 4, 1), omop_table='visit_occurrence'), True), # admission
            (event((2009, 11, 4), "OMOP_CONCEPT_B_CHILD", 45.5, unit="mmol/L"), "skip"),
            # fmt: on
        ]
        patient = femr.Patient(0, [x[0] for x in events_with_labels])
        labeler.label(patient)


#############################################
#############################################
#
# Specific instances of InpatientLabValueLabeler
#
#############################################
#############################################


def _assert_value_to_label_correct(
    labeler: InpatientLabValueLabeler,
    severe: float,
    moderate: float,
    mild: float,
    normal: float,
    unit: Optional[str],
):
    assert labeler.value_to_label(str(severe), unit) == "severe"
    assert labeler.value_to_label(str(moderate), unit) == "moderate"
    assert labeler.value_to_label(str(mild), unit) == "mild"
    assert labeler.value_to_label(str(normal), unit) == "normal"


def _create_specific_labvalue_labeler(
    LabelerClass,
    severity: str,
    outcome_codes: set,
):
    ontology = DummyOntology_Specific(LabelerClass.original_omop_concept_codes)
    labeler = LabelerClass(ontology, severity)  # type: ignore
    assert set(labeler.outcome_codes) == outcome_codes
    assert labeler.severity == severity
    return labeler


def _run_specific_labvalue_test(
    labeler: InpatientLabValueLabeler,
    outcome_codes: set,
    severe_values: List[Tuple[float, str]],
    moderate_values: List[Tuple[float, str]],
    mild_values: List[Tuple[float, str]],
    normal_values: List[Tuple[float, str]],
    help_text: str = "",
):
    """You must specify two values for each of mild/moderate/severe/normal, and the second element
    of each tuple must be the `unit` associated with that measurement."""
    # `itertools.cycle` enables us to test all the `outcome_codes` without knowing in advance how many there are
    outcome_codes_cyclic_iter = itertools.cycle(outcome_codes)
    events_with_labels: EventsWithLabels = [
        # fmt: off
        (event((2000, 1, 1), "Visit/IP", end=datetime.datetime(2000, 1, 3), omop_table='visit_occurrence'), True),  # admission
        (event((2000, 1, 2), next(outcome_codes_cyclic_iter),
               severe_values[0][0], unit=severe_values[0][1]), "skip"),  # lab test - severe - 1
        #
        (event((2001, 1, 1), "Visit/IP", end=datetime.datetime(2001, 1, 3), omop_table='visit_occurrence'), False),  # admission
        (event((2001, 1, 2), next(outcome_codes_cyclic_iter),
               moderate_values[0][0], unit=moderate_values[0][1]), "skip"),  # lab test - moderate
        #
        (event((2002, 1, 1), "Visit/IP", end=datetime.datetime(2002, 1, 3), omop_table='visit_occurrence'), False),  # admission
        (event((2002, 1, 2), next(outcome_codes_cyclic_iter),
               mild_values[0][0], unit=mild_values[0][1]), "skip"),  # lab test - mild
        #
        (event((2003, 1, 1), "Visit/IP", end=datetime.datetime(2003, 1, 3), omop_table='visit_occurrence'), False),  # admission
        (event((2003, 1, 2), next(outcome_codes_cyclic_iter),
               normal_values[0][0], unit=normal_values[0][1]), "skip"),  # lab test - normal
        #
        (event((2004, 1, 1), "Visit/IP", end=datetime.datetime(2004, 1, 10), omop_table='visit_occurrence'), True),  # admission
        (event((2004, 1, 2), next(outcome_codes_cyclic_iter),
               normal_values[1][0], unit=normal_values[1][1]), "skip"),  # lab test - normal
        (event((2004, 1, 9), next(outcome_codes_cyclic_iter),
               severe_values[1][0], unit=severe_values[1][1]), "skip"),  # lab test - severe - 10
        #
        (event((2005, 1, 1), "Visit/IP", end=datetime.datetime(2005, 1, 3), omop_table='visit_occurrence'), False),  # admission
        (event((2005, 1, 2), next(outcome_codes_cyclic_iter),
               mild_values[1][0], unit=mild_values[1][1]), "skip"),  # lab test - mild
        (event((2005, 1, 8), next(outcome_codes_cyclic_iter),
               moderate_values[1][0], unit=moderate_values[1][1]), "skip"),  # lab test - moderate
        #
        (event((2006, 1, 1), "Visit/IP", end=datetime.datetime(2006, 1, 3), omop_table='visit_occurrence'), False),  # admission
        (event((2006, 1, 2), next(outcome_codes_cyclic_iter),
               normal_values[1][0], unit=normal_values[1][1]), "skip"),
        # lab test - normal
        #
        # fmt: on
    ]
    true_prediction_times: List[datetime.datetime] = [
        labeler.visit_start_adjust_func(x[0].start) for x in events_with_labels if isinstance(x[1], bool)
    ]
    true_outcome_times: List[datetime.datetime] = [
        events_with_labels[1][0].start,
        events_with_labels[10][0].start,
    ]
    run_test_for_labeler(
        labeler,
        events_with_labels,
        true_prediction_times=true_prediction_times,
        true_outcome_times=true_outcome_times,
        help_text=help_text,
    )


class DummyOntology_Specific:
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


def test_thrombocytopenia(tmp_path: pathlib.Path):
    outcome_codes = {"child_1", "child_1_1", "LOINC/LP393218-5", "LOINC/LG32892-8", "child_2", "LOINC/777-3"}
    labeler = _create_specific_labvalue_labeler(
        ThrombocytopeniaLabValueLabeler,
        "severe",
        outcome_codes,
    )

    # Test value parsing
    _assert_value_to_label_correct(labeler, 49.9, 50.1, 149.9, 150.1, None)

    # Run tests
    _run_specific_labvalue_test(
        labeler,
        outcome_codes,
        [(30.5, "g/L"), (10, "g/L")],
        [(55, "g/L"), (99.1, "g/L")],
        [(130, "g/L"), (145.123, "g/L")],
        [(200, "g/L"), (400.1, "g/L")],
        "test_thrombocytopenia",
    )


def test_hyperkalemia(tmp_path: pathlib.Path):
    outcome_codes = {"child_1_1", "child_2", "child_1", "LOINC/LP386618-5", "LOINC/LG10990-6", "LOINC/LG7931-1", "LOINC/6298-4",
                    "LOINC/2823-3"}
    labeler = _create_specific_labvalue_labeler(
        HyperkalemiaLabValueLabeler,
        "severe",
        outcome_codes,
    )

    # Test value parsing
    _assert_value_to_label_correct(labeler, 7.1, 6.1, 5.55, 5.49, "mmol/l")

    # Create patient
    _run_specific_labvalue_test(
        labeler,
        outcome_codes,
        [(10, "mmol/L"), (7.01 * 18, "mg/dL")],
        [(6.1, "mmol/L"), (6.9, "mmol/L")],
        [(5.6, "mEq/L"), (5.99, "mmol/L")],
        [(0, "mEq/L"), (4 * 18, "mg/dL")],
        "test_hyperkalemia",
    )


def test_hypoglycemia(tmp_path: pathlib.Path):
    outcome_codes = {"child_2", "child_1_1", "SNOMED/33747003", "LOINC/LP416145-3", "child_1", "LOINC/14749-6",
                    "LOINC/15074-8"}
    labeler = _create_specific_labvalue_labeler(
        HypoglycemiaLabValueLabeler,
        "severe",
        outcome_codes,
    )

    # Test value parsing
    _assert_value_to_label_correct(labeler, 2.9, 3.49, 3.89, 5, "mmol/l")

    # Create patient
    _run_specific_labvalue_test(
        labeler,
        outcome_codes,
        [(0, "mmol/L"), (2 * 18, "mg/dL")],
        [(3.11, "mmol/L"), (3.49, "mmol/L")],
        [(3.51, "mmol/L"), (3.7, "mmol/L")],
        [(10, "mmol/L"), (3.99 * 18, "mg/dL")],
        "test_hypoglycemia",
    )


def test_hyponatremia(tmp_path: pathlib.Path):
    outcome_codes = {"child_1", "child_1_1", "child_2", "LOINC/LG11363-5", "LOINC/2951-2", "LOINC/2947-0"}
    labeler = _create_specific_labvalue_labeler(
        HyponatremiaLabValueLabeler,
        "severe",
        outcome_codes,
    )

    # Test value parsing
    _assert_value_to_label_correct(labeler, 124.9, 129.9, 134.99, 136, None)

    # Create patient
    _run_specific_labvalue_test(
        labeler,
        outcome_codes,
        [(124.451, "mmol/L"), (0, "mg/dL")],
        [(125, "mmol/L"), (129, "mmol/L")],
        [(134.01, "mmol/L"), (130.1, "mmol/L")],
        [(1000, "mmol/L"), (140, "mg/dL")],
        "test_hyponatremia",
    )


def test_anemia(tmp_path: pathlib.Path):
    outcome_codes: set = {"LOINC/LP392452-1", "child_1_1", "child_1"}
    labeler = _create_specific_labvalue_labeler(
        AnemiaLabValueLabeler,
        "severe",
        outcome_codes,
    )

    # Test value parsing
    _assert_value_to_label_correct(labeler, 69.9 / 10, 109.99 / 10, 119.999 / 10, 121 / 10, "g/dl")

    # Create patient
    _run_specific_labvalue_test(
        labeler,
        outcome_codes,
        [(30 / 10, "g/dL"), (65.1 * 100, "mg/dL")],
        [(100 / 10, "g/dL"), (109 / 10, "g/dL")],
        [(115 / 10, "g/dL"), (119.9 / 10, "g/dL")],
        [(150.123 / 10, "g/dL"), (200 * 100, "mg/dL")],
        "test_anemia",
    )


def test_neutropenia(tmp_path: pathlib.Path):
    # TODO
    pass


def test_aki(tmp_path: pathlib.Path):
    # TODO
    pass


#############################################
#############################################
#
# Other lab value related labelers
#
#############################################
#############################################


def test_celiac_test(tmp_path: pathlib.Path):
    # TODO
    pass


# Local testing
if __name__ == "__main__":
    run_test_locally("../ignore/test_labelers/", test_constructor)
    run_test_locally("../ignore/test_labelers/", test_labeling)
    run_test_locally("../ignore/test_labelers/", test_thrombocytopenia)
    run_test_locally("../ignore/test_labelers/", test_hyperkalemia)
    run_test_locally("../ignore/test_labelers/", test_hypoglycemia)
    run_test_locally("../ignore/test_labelers/", test_hyponatremia)
    run_test_locally("../ignore/test_labelers/", test_anemia)
    run_test_locally("../ignore/test_labelers/", test_neutropenia)
    run_test_locally("../ignore/test_labelers/", test_aki)
    run_test_locally("../ignore/test_labelers/", test_celiac_test)
