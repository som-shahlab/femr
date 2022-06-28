from __future__ import annotations

import argparse
from lib2to3.pytree import convert
from pathlib import PurePosixPath
from sys import prefix
from typing import Sequence, Mapping, Tuple

from .. import extension
from .. import Patient, Event
from .. import EventReader, PatientReader, PatientWriter

import datetime
import abc
import multiprocessing
import os
import gzip
import numbers
import random
import csv
import dataclasses
import itertools
import collections

from ..fileio import EventReader, PatientReader, PatientWriter

from dataclasses import dataclass
from typing import List, Callable, Iterable

def convert_events_to_patients(
    event_directory: str,
    target_directory: str,
    shards: int = 1,
    num_threads: int = 1,
):
    """
    Convert a directory containing event files to a directory containing patient files.
    Shards enables multiprocessing. Set the number of shards to the number of processors on your machine.
    """

    # TODO: Actually add multiproccesing support
    # TODO: Implement if C++ if extra speed is needed
    os.makedirs(target_directory, exist_ok=True)
    patients = collections.defaultdict(list)

    for root, dirs, files in os.walk(event_directory):
        for file in files:
            with EventReader(os.path.join(root, file)) as f:
                for id, event in f.get_events():
                    patients[id].append(event)

    all_patient_ids = list(patients.keys())
    random.shuffle(all_patient_ids)

    patients_per_shard = (len(all_patient_ids) + shards - 1) // shards

    for i in range(shards):
        patient_ids = all_patient_ids[
            patients_per_shard * i : patients_per_shard * (i + 1)
        ]
        with PatientWriter(
            os.path.join(target_directory, str(i) + ".csv.gz")
        ) as f:
            for patient in patient_ids:
                patients[patient].sort(key=lambda a: a.start)
                f.add_patient(Patient(patient_id=patient, events=patients[patient]))


def transform_patients(
    source: str,
    target: str,
    transformer: Callable[[Patient], Patient],
    num_threads: int = 1,
) -> None:
    """
    Applies a transformation to the patient files in a folder to generate a modified output folder.
    Optionally supports multithreading.
    """

    # TODO: Actually add multiproccesing support
    # TODO: Implement if C++ if extra speed is needed
    for root, dirs, files in os.walk(source):
        for file in files:
            full_path = PurePosixPath(root, file)
            relative_path = full_path.relative_to(source)
            target_path = PurePosixPath(target) / relative_path
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            with PatientReader(full_path) as i:
                with PatientWriter(target_path) as o:
                    for p in i.get_patients():
                        result_patient = transformer(p)
                        if result_patient is not None:
                            o.add_patient(result_patient)


def convert_patients_to_patient_collection(
    source: str, target: str, num_threads: int = 1
) -> None:
    """
    Convert a folder containing patient files to an extract.
    Optionally supports multithreading.
    """

    extension.patient_collection.convert_patients_to_patient_collection(source, target, num_threads)