from __future__ import annotations

from typing import Callable, Iterable, Tuple, Sequence, Iterator, Optional

import collections.abc
import contextlib
from .. import Event, Patient
from ..extension import datasets

from . import fileio
import os
import random


class EventCollection:
    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            os.mkdir(self.path)

    def __iter__(self) -> Iterator[Tuple[int, Event]]:
        for child in os.listdir(self.path):
            full_path = os.path.join(self.path, child)
            with contextlib.closing(fileio.EventReader(full_path)) as reader:
                for a in reader:
                    yield a

    def create_writer(self) -> fileio.EventWriter:
        return fileio.EventWriter(self.path)


class PatientCollection:
    def __init__(self, path: str):
        self.path = path

    def transform(
        self,
        target_path: str,
        transformer: Callable[[Patient], Optional[Patient]],
    ) -> PatientCollection:
        """
        Applies a transformation to the patient files in a folder to generate a modified output folder.
        Optionally supports multithreading.
        """
        os.mkdir(target_path)

        # TODO: Actually add multiproccesing support
        # TODO: Implement in C++ if extra speed is needed
        for child in os.listdir(self.path):
            source_path = os.path.join(self.path, child)
            destination_path = os.path.join(target_path, child)

            with contextlib.closing(fileio.PatientReader(source_path)) as i:
                with contextlib.closing(
                    fileio.PatientWriter(destination_path)
                ) as o:
                    for p in i:
                        result_patient = transformer(p)
                        if result_patient is not None:
                            o.add_patient(result_patient)

        return PatientCollection(target_path)


def convert_event_collection_to_patient_collection(
    event_collection: EventCollection, target_path: str, num_shards: int = 10
) -> PatientCollection:
    # TODO: Actually add multiproccesing support
    # TODO: Implement in C++ if extra speed is needed
    os.mkdir(target_path)
    patients = collections.defaultdict(list)

    for id, event in event_collection:
        patients[id].append(event)

    all_patient_ids = list(patients.keys())
    random.shuffle(all_patient_ids)

    patients_per_shard = (len(all_patient_ids) + num_shards - 1) // num_shards

    for i in range(num_shards):
        patient_ids = all_patient_ids[
            patients_per_shard * i : patients_per_shard * (i + 1)
        ]
        with contextlib.closing(fileio.PatientWriter(target_path)) as f:
            for patient in patient_ids:
                patients[patient].sort(key=lambda a: a.start)
                f.add_patient(
                    Patient(patient_id=patient, events=patients[patient])
                )

    return PatientCollection(target_path)


# Import from C++ extension

PatientDatabase = datasets.PatientDatabase

convert_patient_collection_to_patient_database = (
    datasets.convert_patient_collection_to_patient_database
)
