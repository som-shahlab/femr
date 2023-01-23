import datetime
import pathlib
from typing import List, Tuple
import pytest
import itertools

import piton.datasets
from piton.labelers.core import TimeHorizon
from piton.labelers.omop_lab_values import (
    OMOPConceptOutcomeFromLabValueLabeler,
    ThrombocytopeniaLabValueLabeler,
    HyperkalemiaLabValueLabeler,
    HypoglycemiaLabValueLabeler,
    HyponatremiaLabValueLabeler,
    AnemiaLabValueLabeler,
    CeliacTestLabeler,
    NeutropeniaLabValueLabeler,
    AcuteKidneyInjuryLabValueLabeler,
)

# Needed to import `tools` for local testing
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import (
    EventsWithLabels,
    event,
    run_test_for_labeler,
    run_test_locally,
)


#############################################
#############################################
#
# Generic OMOPConceptOutcomeFromLabValueLabeler
#
# #############################################
#############################################

class DummyOntology_Generic:
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
    ontology = DummyOntology_Generic()
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
    ontology = DummyOntology_Generic()
    labeler = DummyLabeler1(ontology, time_horizon, "severe")  # type: ignore

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






#############################################
#############################################
#
# Specific instances of OMOPConceptOutcomeFromLabValueLabeler
#
#############################################
#############################################


def _assert_value_to_label_correct(labeler: OMOPConceptOutcomeFromLabValueLabeler, severe: float, moderate: float, mild: float, normal: float):
    assert labeler.value_to_label(severe) == "severe"
    assert labeler.value_to_label(moderate) == "moderate"
    assert labeler.value_to_label(mild) == "mild"
    assert labeler.value_to_label(normal) == "normal"

def _assert_labvalue_constructor_correct(labeler: OMOPConceptOutcomeFromLabValueLabeler, time_horizon: TimeHorizon, severity: str, outcome_codes: set):
    assert set(labeler.outcome_codes) == outcome_codes
    assert labeler.severity == severity
    assert labeler.time_horizon == time_horizon
    
def _create_specific_labvalue_labeler(LabelerClass, original_omop_concept_codes: List[str], severity: str, outcome_codes: set):
    time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=10)
    )
    ontology = DummyOntology_Specific(original_omop_concept_codes)
    labeler = LabelerClass(ontology, time_horizon, severity)  # type: ignore
    _assert_labvalue_constructor_correct(labeler, time_horizon, severity, outcome_codes)
    return labeler
    
def _run_specific_labvalue_test(labeler: OMOPConceptOutcomeFromLabValueLabeler, 
                                outcome_codes: set,
                                severe_values: List[Tuple[float, str]], 
                                moderate_values: List[Tuple[float, str]], 
                                mild_values: List[Tuple[float, str]], 
                                normal_values: List[Tuple[float, str]],
                                help_text: str = ""):
    """You must specify two values for each of mild/moderate/severe/normal, and the second element
    of each tuple must be the `unit` associated with that measurement."""
    # this enables us to test all the `outcome_codes` without knowing in advance how many there are
    outcome_codes_cyclic_iter = itertools.cycle(outcome_codes)
    events_with_labels: EventsWithLabels = [
        (event((2000, 1, 1), 1, None), True),  # admission
        (event((2000, 1, 2), next(outcome_codes_cyclic_iter), severe_values[0][0], unit=severe_values[0][1]), 'skip'),  # lab test - severe
        #
        (event((2001, 1, 1), 1, None), False),  # admission
        (event((2001, 1, 2), next(outcome_codes_cyclic_iter), moderate_values[0][0], unit=moderate_values[0][1]), 'skip'),  # lab test - moderate
        #
        (event((2002, 1, 1), 1, None), False),  # admission
        (event((2002, 1, 2), next(outcome_codes_cyclic_iter), mild_values[0][0], unit=mild_values[0][1]), 'skip'),  # lab test - mild
        #
        (event((2003, 1, 1), 1, None), False),  # admission
        (event((2003, 1, 2), next(outcome_codes_cyclic_iter), normal_values[0][0], unit=normal_values[0][1]), 'skip'),  # lab test - normal
        #
        (event((2004, 1, 1), 1, None), True),  # admission
        (event((2004, 1, 2), next(outcome_codes_cyclic_iter), normal_values[1][0], unit=normal_values[1][1]), 'skip'),  # lab test - normal
        (event((2004, 1, 9), next(outcome_codes_cyclic_iter), severe_values[1][0], unit=severe_values[1][1]), 'skip'),  # lab test - severe
        #
        (event((2005, 1, 1), 1, None), False),  # admission
        (event((2005, 1, 2), next(outcome_codes_cyclic_iter), mild_values[1][0], unit=mild_values[1][1]), 'skip'),  # lab test - mild
        (event((2005, 1, 8), next(outcome_codes_cyclic_iter), moderate_values[1][0], unit=moderate_values[1][1]), 'skip'),  # lab test - moderate
        #
        (event((2006, 1, 1), 1, None), None),  # admission
        (event((2006, 1, 2), next(outcome_codes_cyclic_iter), normal_values[1][0], unit=normal_values[1][1]), 'skip'),
    ]
    true_outcome_times: List[datetime.datetime] = [
        events_with_labels[1][0].start,
        events_with_labels[10][0].start,
    ]
    run_test_for_labeler(
        labeler,
        events_with_labels,
        true_outcome_times=true_outcome_times,
        help_text=help_text,
    )

class DummyOntology_Specific:
    def __init__(self, new_codes: List[str]):
        self.new_codes = new_codes

    def get_dictionary(self):
        return [
            "zero",
            "Visit/IP",
            "child_1_1", # two
            "child_1", # three
            "child_2", # four
            "five",
        ] + self.new_codes

    def get_children(self, parent_code: int) -> List[int]:
        if parent_code == 3:
            return [2, ]
        elif parent_code == 6:
            return [3,]
        elif parent_code == 7:
            return [4]
        return []

def test_thrombocytopenia(tmp_path: pathlib.Path):
    original_omop_concept_codes: List[str] = [
        "LOINC/LP393218-5", 
        "LOINC/LG32892-8"
    ]
    outcome_codes: set = { 2, 3, 4, 6, 7}
    labeler = _create_specific_labvalue_labeler(ThrombocytopeniaLabValueLabeler, original_omop_concept_codes, 'severe', outcome_codes)

    # Test value parsing
    _assert_value_to_label_correct(labeler, 49.9, 50.1, 149.9, 150.1)
    assert labeler.normalize_value_with_units(100, "g/L") == 100
    assert labeler.normalize_value_with_units(100, None) == 100

    # Run tests
    _run_specific_labvalue_test(labeler,
                                outcome_codes,
                                [ (30.5, "g/L"), (10, "g/L") ],
                                [ (55, "g/L"), (99.1, "g/L") ],
                                [ (130, "g/L"), (145.123, "g/L") ],
                                [ (200, "g/L"), (400.1, "g/L") ],
                                'test_thrombocytopenia')

def test_hyperkalemia(tmp_path: pathlib.Path):
    original_omop_concept_codes: List[str] = [
        "LOINC/LG7931-1",
        "LOINC/LP386618-5",
        "LOINC/40653596",
    ]
    outcome_codes: set = { 2, 3, 4, 6, 7, 8, }
    labeler = _create_specific_labvalue_labeler(HyperkalemiaLabValueLabeler, original_omop_concept_codes, 'severe', outcome_codes)

    # Test value parsing
    _assert_value_to_label_correct(labeler, 7.1, 6.1, 5.55, 5.49)
    assert labeler.normalize_value_with_units(100, "mmol/L") == 100
    assert labeler.normalize_value_with_units(100, "mEq/L (extra text)") == 100
    assert labeler.normalize_value_with_units(100, "mg/dL (extra text)") == 100 / 18
    with pytest.raises(ValueError):
        labeler.normalize_value_with_units(100, "g/L")
    with pytest.raises(ValueError):
        labeler.normalize_value_with_units(100, None)

    # Create patient
    _run_specific_labvalue_test(labeler,
                                outcome_codes,
                                [ (10, "mmol/L"), (7.01 * 18, "mg/dL") ],
                                [ (6.1, "mmol/L"), (6.9, "mmol/L") ],
                                [ (5.6, "mEq/L"), (5.99, "mmol/L") ],
                                [ (0, "mEq/L"), (4 * 18, "mg/dL") ],
                                'test_hyperkalemia')

def test_hypoglycemia(tmp_path: pathlib.Path):
    original_omop_concept_codes: List[str] = [
        "SNOMED/33747003",
        "LOINC/LP416145-3",
    ]
    outcome_codes: set = { 2, 3, 4, 6, 7, }
    labeler = _create_specific_labvalue_labeler(HypoglycemiaLabValueLabeler, original_omop_concept_codes, 'severe', outcome_codes)

    # Test value parsing
    _assert_value_to_label_correct(labeler, 2.9, 3.49, 3.89, 5)
    assert labeler.normalize_value_with_units(100, "mmol/L") == 100
    assert labeler.normalize_value_with_units(100, "mg/dL (extra text)") == 100 / 18
    with pytest.raises(ValueError):
        labeler.normalize_value_with_units(100, "g/L")
    with pytest.raises(ValueError):
        labeler.normalize_value_with_units(100, None)
    
    # Create patient
    _run_specific_labvalue_test(labeler,
                                outcome_codes,
                                [ (0, "mmol/L"), (2 * 18, "mg/dL") ],
                                [ (3.11, "mmol/L"), (3.49, "mmol/L") ],
                                [ (3.51, "mmol/L"), (3.7, "mmol/L") ],
                                [ (10, "mmol/L"), (3.99 * 18, "mg/dL") ],
                                'test_hypoglycemia')

def test_hyponatremia(tmp_path: pathlib.Path):
    original_omop_concept_codes: List[str] = [
        "LOINC/LG11363-5",
    ]
    outcome_codes: set = { 2, 3, 6, }
    labeler = _create_specific_labvalue_labeler(HyponatremiaLabValueLabeler, original_omop_concept_codes, 'severe', outcome_codes)

    # Test value parsing
    _assert_value_to_label_correct(labeler, 124.9, 129.9, 134.99, 136)
    assert labeler.normalize_value_with_units(100, "g/L") == 100
    assert labeler.normalize_value_with_units(100, None) == 100
    
    # Create patient
    _run_specific_labvalue_test(labeler,
                                outcome_codes,
                                [ (124.451, "mmol/L"), (0, "mg/dL") ],
                                [ (125, "mmol/L"), (129, "mmol/L") ],
                                [ (134.01, "mmol/L"), (130.1, "mmol/L") ],
                                [ (1000, "mmol/L"), (140, "mg/dL") ],
                                'test_hyponatremia')

def test_anemia(tmp_path: pathlib.Path):
    original_omop_concept_codes: List[str] = [
        "LOINC/LP392452-1",
    ]
    outcome_codes: set = { 2, 3, 6, }
    labeler = _create_specific_labvalue_labeler(AnemiaLabValueLabeler, original_omop_concept_codes, 'severe', outcome_codes)

    # Test value parsing
    _assert_value_to_label_correct(labeler, 69.9, 109.99, 119.999, 121)
    assert labeler.normalize_value_with_units(100, "g/dL") == 100 * 10
    assert labeler.normalize_value_with_units(100, "mg/dL") == 100 / 100
    with pytest.raises(ValueError):
        labeler.normalize_value_with_units(100, "g/L")
    with pytest.raises(ValueError):
        labeler.normalize_value_with_units(100, None)
    
    # Create patient
    _run_specific_labvalue_test(labeler,
                                outcome_codes,
                                [ (30 / 10, "g/dL"), (65.1 * 100, "mg/dL") ],
                                [ (100 / 10, "g/dL"), (109 / 10, "g/dL") ],
                                [ (115 / 10, "g/dL"), (119.9 / 10, "g/dL") ],
                                [ (150.123 / 10, "g/dL"), (200 * 100, "mg/dL") ],
                                'test_anemia')

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
    run_test_locally("../ignore/test_labelers/", test_units)
    run_test_locally("../ignore/test_labelers/", test_thrombocytopenia)
    run_test_locally("../ignore/test_labelers/", test_hyperkalemia)
    run_test_locally("../ignore/test_labelers/", test_hypoglycemia)
    run_test_locally("../ignore/test_labelers/", test_hyponatremia)
    run_test_locally("../ignore/test_labelers/", test_anemia)
    run_test_locally("../ignore/test_labelers/", test_neutropenia)
    run_test_locally("../ignore/test_labelers/", test_aki)
    run_test_locally("../ignore/test_labelers/", test_celiac_test)