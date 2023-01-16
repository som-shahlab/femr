import contextlib
import csv
import datetime
import io
import os
import pathlib
import pickle
from typing import List, Tuple, Dict

import zstandard

import piton
import piton.datasets
from piton.labelers.core import LabeledPatients, LabelingFunction

dummy_events = [
    piton.Event(start=datetime.datetime(1995, 1, 3), code=0, value=34.5),
    piton.Event(
        start=datetime.datetime(2010, 1, 1),
        code=1,
        value="test_value",
    ),
    piton.Event(start=datetime.datetime(2010, 1, 5), code=2, value=1),
    piton.Event(start=datetime.datetime(2010, 6, 5), code=3, value=None),
    piton.Event(start=datetime.datetime(2010, 8, 5), code=2, value=None),
    piton.Event(start=datetime.datetime(2011, 7, 5), code=2, value=None),
    piton.Event(start=datetime.datetime(2012, 10, 5), code=3, value=None),
    piton.Event(start=datetime.datetime(2015, 6, 5, 0), code=2, value=None),
    piton.Event(
        start=datetime.datetime(2015, 6, 5, 10, 10), code=2, value=None
    ),
    piton.Event(start=datetime.datetime(2015, 6, 15, 11), code=3, value=None),
    piton.Event(start=datetime.datetime(2016, 1, 1), code=2, value=None),
    piton.Event(
        start=datetime.datetime(2016, 3, 1, 10, 10, 10), code=4, value=None
    ),
]

NUM_EVENTS = len(dummy_events)
NUM_PATIENTS = 10

all_events: List[Tuple[int, piton.Event]] = []

for patient_id in range(10, 10 + NUM_PATIENTS):
    all_events.extend((patient_id, event) for event in dummy_events)


def create_events(tmp_path: pathlib.Path) -> piton.datasets.EventCollection:
    events = piton.datasets.EventCollection(os.path.join(tmp_path, "events"))
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


def create_ontology(path_to_ontology_dir: str, concepts: List[str]):
    path_to_concept_file: str = os.path.join(
        path_to_ontology_dir, "concept", "concept.csv.zst"
    )
    os.makedirs(os.path.dirname(path_to_concept_file), exist_ok=True)
    os.makedirs(
        os.path.join(path_to_ontology_dir + "/concept_relationship/"),
        exist_ok=True,
    )

    concept_map: Dict[str, int] = {}

    with io.TextIOWrapper(
        zstandard.ZstdCompressor(1).stream_writer(
            open(path_to_concept_file, "wb")
        )
    ) as o:
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
                    "domain_id": "Observation",
                    "vocabulary_id": "dummy",
                    "concept_class_id": "Observation",
                    "standard_concept": "",
                    "concept_code": c,
                    "valid_start_DATE": "1970-01-01",
                    "valid_end_DATE": "2099-12-31",
                    "invalid_reason": "",
                    "load_table_id": "custom_mapping",
                    "load_row_id": "",
                }
            )
    return concept_map


class DummyOntology:
    def get_dictionary(self):
        return ["zero", "one", "two", "three", "four"]


dummy_ontology = DummyOntology()


def create_database(
    tmp_path: pathlib.Path, dummy_ontology: DummyOntology = dummy_ontology
) -> None:

    patient_collection = create_patients(tmp_path)
    with patient_collection.reader() as reader:
        _ = list(reader)

    path_to_ontology = os.path.join(tmp_path, "ontology")
    concepts = [str(x) for x in dummy_ontology.get_dictionary()]
    concept_map = create_ontology(path_to_ontology, concepts)
    print(concept_map)

    path_to_database = os.path.join(tmp_path, "target")

    if not os.path.exists(path_to_database):
        os.mkdir(path_to_database)

    patient_collection.to_patient_database(
        path_to_database,
        path_to_ontology,  # concept.csv
        num_threads=2,
    ).close()


def get_piton_codes(ontology, target_code):
    piton_concept_id = f"dummy/{DummyOntology().get_dictionary()[target_code]}"
    piton_target_code = ontology.get_dictionary().index(piton_concept_id)
    return piton_target_code


def create_patients_list(events: List[piton.Event]) -> List[piton.Patient]:
    patients: List[piton.Patient] = []
    for patient_id in range(NUM_PATIENTS):
        patients.append(
            piton.Patient(
                patient_id,
                events,
            )
        )
    return patients


def create_labeled_patients_list(
    labeler: LabelingFunction, patients: List[piton.Patient]
):
    pat_to_labels = {}

    for patient in patients:
        labels = labeler.label(patient)

        if len(labels) > 0:
            pat_to_labels[patient.patient_id] = labels

    labeled_patients = LabeledPatients(
        pat_to_labels, labeler.get_labeler_type()
    )

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
