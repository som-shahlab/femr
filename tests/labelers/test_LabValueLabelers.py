# flake8: noqa: E402
import datetime
import itertools
import os
import pathlib
import sys
from typing import List, Optional, Tuple

from femr.labelers import LabeledPatients
from femr import Patient

from femr.labelers.benchmarks import (
    AnemiaInstantLabValueLabeler,
    HyperkalemiaInstantLabValueLabeler,
    HypoglycemiaInstantLabValueLabeler,
    HyponatremiaInstantLabValueLabeler,
    ThrombocytopeniaInstantLabValueLabeler,
)

# Needed to import `tools` for local testing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import EventsWithLabels, assert_labels_are_accurate, create_patients_list, event, run_test_for_labeler, run_test_locally

def _assert_value_to_label_correct(
    labeler,
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
    outcome_codes: set,
):
    ontology = DummyOntology_Specific(LabelerClass.original_omop_concept_codes)
    labeler = LabelerClass(ontology)  # type: ignore
    assert set(labeler.outcome_codes) == outcome_codes
    return labeler


def _run_specific_labvalue_test(
    labeler,
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
        (event((2000, 1, 1), "Visit/IP", end=datetime.datetime(2000, 1, 3), omop_table='visit_occurrence'), 'skip'),  # admission
        (event((2000, 1, 2), next(outcome_codes_cyclic_iter),
               severe_values[0][0], unit=severe_values[0][1]), 'severe'),  # lab test - severe
        #
        (event((2001, 1, 1), "Visit/IP", end=datetime.datetime(2001, 1, 3), omop_table='visit_occurrence'), 'skip'),  # admission
        (event((2001, 1, 2), next(outcome_codes_cyclic_iter),
               moderate_values[0][0], unit=moderate_values[0][1]), 'moderate'),  # lab test - moderate
        #
        (event((2002, 1, 1), "Visit/IP", end=datetime.datetime(2002, 1, 3), omop_table='visit_occurrence'), 'skip'),  # admission
        (event((2002, 1, 2), next(outcome_codes_cyclic_iter),
               mild_values[0][0], unit=mild_values[0][1]), 'mild'),  # lab test - mild
        #
        (event((2003, 1, 1), "Visit/IP", end=datetime.datetime(2003, 1, 3), omop_table='visit_occurrence'), 'skip'),  # admission
        (event((2003, 1, 2), next(outcome_codes_cyclic_iter),
               normal_values[0][0], unit=normal_values[0][1]), 'normal'),  # lab test - normal
        #
        (event((2004, 1, 1), "Visit/IP", end=datetime.datetime(2004, 1, 10), omop_table='visit_occurrence'), 'skip'),  # admission
        (event((2004, 1, 2), next(outcome_codes_cyclic_iter),
               normal_values[1][0], unit=normal_values[1][1]), 'normal'),  # lab test - normal
        (event((2004, 1, 9), next(outcome_codes_cyclic_iter),
               severe_values[1][0], unit=severe_values[1][1]), 'severe'),  # lab test - severe
        #
        (event((2005, 1, 1), "Visit/IP", end=datetime.datetime(2005, 1, 3), omop_table='visit_occurrence'), 'skip'),  # admission
        (event((2005, 1, 2), next(outcome_codes_cyclic_iter),
               mild_values[1][0], unit=mild_values[1][1]), 'mild'),  # lab test - mild
        (event((2005, 1, 8), next(outcome_codes_cyclic_iter),
               moderate_values[1][0], unit=moderate_values[1][1]), 'moderate'),  # lab test - moderate
        #
        (event((2006, 1, 1), "Visit/IP", end=datetime.datetime(2006, 1, 3), omop_table='visit_occurrence'), 'skip'),  # admission
        (event((2006, 1, 2), next(outcome_codes_cyclic_iter),
               normal_values[1][0], unit=normal_values[1][1]), 'normal'), # lab test - normal
        #
        # fmt: on
    ]
    true_labels: List[Tuple[datetime.datetime, str]] = [
        (x[0].start - datetime.timedelta(minutes=1), labeler.label_to_int(x[1])) for x in events_with_labels if x[1] != 'skip'
    ]
    patients: List[Patient] = create_patients_list(10, [x[0] for x in events_with_labels])
    labeled_patients: LabeledPatients = labeler.apply(patients=patients)

    # Check accuracy of Labels
    for patient in patients:
        assert_labels_are_accurate(
            labeled_patients,
            patient.patient_id,
            true_labels,
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
        ThrombocytopeniaInstantLabValueLabeler,
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
    outcome_codes = {
        "child_1_1",
        "child_2",
        "child_1",
        "LOINC/LP386618-5",
        "LOINC/LG10990-6",
        "LOINC/LG7931-1",
        "LOINC/6298-4",
        "LOINC/2823-3",
    }
    labeler = _create_specific_labvalue_labeler(
        HyperkalemiaInstantLabValueLabeler,
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
    outcome_codes = {
        "child_2",
        "child_1_1",
        "SNOMED/33747003",
        "LOINC/LP416145-3",
        "child_1",
        "LOINC/14749-6",
    }
    labeler = _create_specific_labvalue_labeler(
        HypoglycemiaInstantLabValueLabeler,
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
        HyponatremiaInstantLabValueLabeler,
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
        AnemiaInstantLabValueLabeler,
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
    run_test_locally("../ignore/test_labelers/", test_thrombocytopenia)
    run_test_locally("../ignore/test_labelers/", test_hyperkalemia)
    run_test_locally("../ignore/test_labelers/", test_hypoglycemia)
    run_test_locally("../ignore/test_labelers/", test_hyponatremia)
    run_test_locally("../ignore/test_labelers/", test_anemia)
    run_test_locally("../ignore/test_labelers/", test_neutropenia)
    run_test_locally("../ignore/test_labelers/", test_aki)
    run_test_locally("../ignore/test_labelers/", test_celiac_test)
