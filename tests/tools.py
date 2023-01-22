import datetime
import os
import pathlib
import pickle
from typing import List, Tuple, Callable, Union, Optional
import shutil
from inspect import signature

import piton
import piton.datasets
from piton.labelers.core import LabeledPatients, Label, Labeler

# 2nd elem of tuple -- 'skip' means no label, None means censored
EventsWithLabels = List[Tuple[piton.Event, Optional[Union[bool, str]]]]

def event(date: Tuple, code, value, visit_id=None):
    """A terser way to create a Piton Event."""
    hour, minute, seconds = 0, 0, 0
    if len(date) == 3:
        year, month, day = date
    elif len(date) == 4:
        year, month, day, hour = date
    elif len(date) == 5:
        year, month, day, hour, minute = date
    elif len(date) == 6:
        year, month, day, hour, minute, seconds = date
    else:
        raise ValueError(f"Invalid date: {date}")
    return piton.Event(
        start=datetime.datetime(
            year,
            month,
            day,
            hour,
            minute,
            seconds,
        ),
        code=code,
        value=value,
        visit_id=visit_id,
    )
    
def save_to_pkl(object_to_save, path_to_file: str):
    """Save object to Pickle file."""
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "wb") as fd:
        pickle.dump(object_to_save, fd)
        
def run_test_locally(str_path: str, test_func: Callable):
    """Run test locally the way Github Actions does (in a temporary directory `tmp_path`)."""
    tmp_path = pathlib.Path(str_path)
    shutil.rmtree(tmp_path)
    os.makedirs(tmp_path, exist_ok=True)
    if signature(test_func).parameters.get("tmp_path"):
        test_func(tmp_path)
    else:
       test_func()

def create_patients(num_patients: int, events: List[piton.Event]) -> List[piton.Patient]:
    """Creates a list of patients, each with the same events contained in `events`"""
    patients: List[piton.Patient] = []
    for patient_id in range(num_patients):
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
    assert patient_id in labeled_patients, f"patient_id={patient_id} not in labeled_patients"
    generated_labels: List[Label] = labeled_patients[patient_id]
    # Check that length of lists of labels are the same
    assert len(generated_labels) == len(true_labels), (
        f"{len(generated_labels)} != {len(true_labels)} | {help_text}" 
    )
    # Check that value of labels are the same
    for idx, (label, true_label) in enumerate(zip(generated_labels, true_labels)):
        assert label.value == true_label, (
            f"patient_id={patient_id}, label_idx={idx}, label={label}  |  "
            f"{label.value} (Assigned) != {true_label} (Expected)  |  "
            f"{help_text}"
        )

def run_test_for_labeler(labeler: Labeler, 
                         events_with_labels: EventsWithLabels,
                         help_text: str = "",) -> None:
    patients: List[piton.Patient] = create_patients(10, 
        [ x[0] for x in events_with_labels ]
    )
    true_labels: List[Optional[bool]] = [ 
        x[1] for x in events_with_labels if isinstance(x[1], bool) or (x[1] is None)
    ]
    labeled_patients: LabeledPatients = labeler.apply(patient_database=patients)
    
    # Check accuracy of Labels
    for i in range(len(patients)):
        assert_labels_are_accurate(labeled_patients, i, true_labels, help_text=help_text)
    
    # Check Labeler's internal functions
    for p in patients:
        assert labeler.get_outcome_times(p) == [
            event.start for event in p.events if event.code in labeler.outcome_codes
        ]

if __name__ == '__main__':
    pass