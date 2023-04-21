import contextlib
import csv
import datetime
import io
import os
import pathlib
import pickle
import shutil
from inspect import signature
from typing import Callable, Dict, List, Optional, Tuple, Union

import zstandard

import femr
import femr.datasets
from femr.labelers import Label, LabeledPatients, Labeler

# 2nd elem of tuple -- 'skip' means no label, None means censored
EventsWithLabels = List[Tuple[femr.Event, Union[bool, str]]]


def event(date: Tuple, code, value=None, visit_id=None, **kwargs):
    """A terser way to create a femr Event."""
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
    return femr.Event(
        start=datetime.datetime(year, month, day, hour, minute, seconds),
        code=str(code),
        value=value,
        visit_id=visit_id,
        **kwargs,
    )


DUMMY_EVENTS = [
    event((1995, 1, 3), 0, 34.5),
    event((2010, 1, 1), 1, "test_value"),
    event((2010, 1, 5), 2, 1),
    event((2010, 6, 5), 3, None),
    event((2010, 8, 5), 2, None),
    event((2011, 7, 5), 2, None),
    event((2012, 10, 5), 3, None),
    event((2015, 6, 5, 0), 2, None),
    event((2015, 6, 5, 10, 10), 2, None),
    event((2015, 6, 15, 11), 3, None),
    event((2016, 1, 1), 2, None),
    event((2016, 3, 1, 10, 10, 10), 4, None),
]

NUM_EVENTS = len(DUMMY_EVENTS)
NUM_PATIENTS = 10

DUMMY_CONCEPTS: List[str] = ["zero", "one", "two", "three", "four"]

ALL_EVENTS: List[Tuple[int, femr.Event]] = []
for patient_id in range(NUM_PATIENTS):
    ALL_EVENTS.extend((patient_id, event) for event in DUMMY_EVENTS)


def create_events(tmp_path: pathlib.Path) -> femr.datasets.EventCollection:
    event_path = os.path.join(tmp_path, "events")
    os.makedirs(event_path, exist_ok=True)
    events = femr.datasets.EventCollection(event_path)
    chunks = 7
    events_per_chunk = (len(ALL_EVENTS) + chunks - 1) // chunks

    for i in range(7):
        with contextlib.closing(events.create_writer()) as writer:
            for patient_id, event in ALL_EVENTS[i * events_per_chunk : (i + 1) * events_per_chunk]:
                raw_event = femr.datasets.RawEvent(
                    start=event.start, concept_id=int(event.code), value=event.value, visit_id=event.visit_id
                )
                writer.add_event(patient_id, raw_event)

    return events


def create_patients_list(num_patients: int, events: List[femr.Event]) -> List[femr.Patient]:
    """Creates a list of patients, each with the same events contained in `events`"""
    patients: List[femr.Patient] = []
    for patient_id in range(num_patients):
        patients.append(
            femr.Patient(
                patient_id,
                tuple(events),
            )
        )
    return patients


def create_patients(tmp_path: pathlib.Path) -> femr.datasets.PatientCollection:
    return create_events(tmp_path).to_patient_collection(os.path.join(tmp_path, "patients"))


def create_ontology(path_to_ontology_dir: str, concepts: List[str]):
    path_to_concept_file: str = os.path.join(path_to_ontology_dir, "concept", "concept.csv.zst")
    path_to_relationship_file: str = os.path.join(path_to_ontology_dir + "/concept_relationship/")
    os.makedirs(os.path.dirname(path_to_concept_file), exist_ok=True)
    os.makedirs(path_to_relationship_file, exist_ok=True)

    concept_map: Dict[str, int] = {}

    with io.TextIOWrapper(zstandard.ZstdCompressor(1).stream_writer(open(path_to_concept_file, "wb"))) as o:
        writer = csv.DictWriter(
            o,
            fieldnames=[
                "concept_id",
                "concept_name",
                "domain_id",
                "vocabulary_id",
                "concept_class_id",
                "standard_concept",
                "concept_code",
                "valid_start_DATE",
                "valid_end_DATE",
                "invalid_reason",
                "load_table_id",
                "load_row_id",
            ],
        )

        writer.writeheader()

        next_code: int = 0
        for i, c in enumerate(concepts):
            code: int = i + next_code
            concept_map[c] = code
            writer.writerow(
                {
                    "concept_id": str(code),
                    "concept_name": c,
                    "concept_code": c,
                    "domain_id": "Observation",
                    "vocabulary_id": "dummy",
                    "concept_class_id": "Observation",
                    "standard_concept": "",
                    "valid_start_DATE": "1970-01-01",
                    "valid_end_DATE": "2099-12-31",
                    "invalid_reason": "",
                    "load_table_id": "custom_mapping",
                    "load_row_id": "",
                }
            )
    return concept_map


def create_database(tmp_path: pathlib.Path, dummy_concepts: List[str] = []) -> None:
    patient_collection = create_patients(tmp_path)

    path_to_ontology = os.path.join(tmp_path, "ontology")
    if dummy_concepts == []:
        dummy_concepts = DUMMY_CONCEPTS
    create_ontology(path_to_ontology, dummy_concepts)

    path_to_database = os.path.join(tmp_path, "target")
    os.makedirs(path_to_database, exist_ok=True)

    patient_collection.to_patient_database(
        path_to_database,
        path_to_ontology,
        num_threads=2,
    ).close()


def get_femr_code(ontology, target_code, dummy_concepts: List[str] = []):
    if dummy_concepts == []:
        dummy_concepts = DUMMY_CONCEPTS
    femr_concept_id = f"dummy/{dummy_concepts[target_code]}"
    return femr_concept_id


def assert_labels_are_accurate(
    labeled_patients: LabeledPatients,
    patient_id: int,
    true_labels: List[Tuple[datetime.datetime, Optional[bool]]],
    help_text: str = "",
):
    """Passes if the labels in `labeled_patients` for `patient_id` exactly match the labels in `true_labels`."""
    assert patient_id in labeled_patients, f"patient_id={patient_id} not in labeled_patients"
    generated_labels: List[Label] = labeled_patients[patient_id]
    # Check that length of lists of labels are the same

    assert len(generated_labels) == len(
        true_labels
    ), f"len(generated): {len(generated_labels)} != len(expected): {len(true_labels)} | {help_text}"
    # Check that value of labels are the same
    for idx, (label, true_label) in enumerate(zip(generated_labels, true_labels)):
        assert label.value == true_label[1] and label.time == true_label[0], (
            f"patient_id={patient_id}, label_idx={idx}, label={label}  |  "
            f"{label.value} (Assigned) != {true_label} (Expected)  |  "
            f"{help_text}"
        )


def run_test_for_labeler(
    labeler: Labeler,
    events_with_labels: EventsWithLabels,
    true_outcome_times: Optional[List[datetime.datetime]] = None,
    true_prediction_times: Optional[List[datetime.datetime]] = None,
    help_text: str = "",
) -> None:
    patients: List[femr.Patient] = create_patients_list(10, [x[0] for x in events_with_labels])
    true_labels: List[Tuple[datetime.datetime, Optional[bool]]] = [
        (x[0].start, x[1]) for x in events_with_labels if isinstance(x[1], bool)
    ]
    if true_prediction_times is not None:
        # If manually specified prediction times, adjust labels from occurring at `event.start`
        # e.g. we may make predictions at `event.end` or `event.start + 1 day`
        true_labels = [(tp, tl[1]) for (tl, tp) in zip(true_labels, true_prediction_times)]
    labeled_patients: LabeledPatients = labeler.apply(patients=patients)

    # Check accuracy of Labels
    for patient in patients:
        assert_labels_are_accurate(
            labeled_patients,
            patient.patient_id,
            true_labels,
            help_text=help_text,
        )

    # Check Labeler's internal functions
    if hasattr(labeler, "get_outcome_times"):
        for p in patients:
            if true_outcome_times:
                # If manually specified outcome times, check that they are correct
                assert (
                    labeler.get_outcome_times(p) == true_outcome_times
                ), f"{labeler.get_outcome_times(p)} != {true_outcome_times} | {help_text}"
            else:
                # Otherwise, assume that outcome times are simply the start times of
                # events with codes in `outcome_codes`
                assert hasattr(labeler, "outcome_codes"), f"{labeler} is missing an `outcome_codes` attribute"
                assert labeler.get_outcome_times(p) == [
                    event.start for event in p.events if event.code in labeler.outcome_codes
                ], f"{labeler.get_outcome_times(p)} != {true_outcome_times} | {help_text}"


def create_labeled_patients_list(labeler: Labeler, patients: List[femr.Patient]):
    pat_to_labels = {}

    for patient in patients:
        labels = labeler.label(patient)

        if len(labels) > 0:
            pat_to_labels[patient.patient_id] = labels

    labeled_patients = LabeledPatients(pat_to_labels, labeler.get_labeler_type())

    return labeled_patients


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


def run_test_locally(str_path: str, test_func: Callable):
    """Run test locally the way Github Actions does (in a temporary directory `tmp_path`)."""
    tmp_path = pathlib.Path(str_path)
    shutil.rmtree(tmp_path)
    os.makedirs(tmp_path, exist_ok=True)
    if signature(test_func).parameters.get("tmp_path"):
        test_func(tmp_path)
    else:
        test_func()
