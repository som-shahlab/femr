from __future__ import annotations

import argparse
from lib2to3.pytree import convert
from pathlib import PurePosixPath
from sys import prefix
from typing import Sequence, Mapping, Tuple

import datetime
import abc
import multiprocessing
import os
import gzip
import numbers
import csv
import dataclasses

from dataclasses import dataclass
from typing import List

from . import *


class CSVConverter(abc.ABC):
    """
    An interface for converting a csv into events.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_patient_id_field(self) -> str:
        """
        Return the field that contains the patient_id
        """

    @abc.abstractmethod
    def get_file_prefix(self) -> str:
        """
        Return the prefix for files this converter will trigger on.
        """
        ...

    @abc.abstractmethod
    def get_events(self, row: Mapping[str, str]) -> Sequence[Event]:
        """
        Return the events generated for a particular row.
        """
        ...


def run_csv_converter(args: Tuple[str, str, CSVConverter]) -> None:
    source, target, converter = args
    os.makedirs(os.path.dirname(target), exist_ok=True)

    with gzip.open(source, "rt") as f:
        reader = csv.DictReader(f)
        with EventWriter(target) as o:
            for _, row in zip(range(1000), reader):
                lower_row = {a.lower(): b for a, b in row.items()}
                for event in converter.get_events(lower_row):
                    o.add_event(row[converter.get_patient_id_field()], event)


def run_csv_converters(
    source_csvs: str,
    target_location: str,
    num_threads: int,
    converters: Sequence[CSVConverter],
) -> None:
    pool = multiprocessing.Pool(num_threads)

    os.makedirs(target_location, exist_ok=True)

    to_process = []

    for root, dirs, files in os.walk(source_csvs):
        for name in files:
            full_path = PurePosixPath(root, name)
            relative_path = full_path.relative_to(source_csvs)
            target_path = PurePosixPath(target_location) / relative_path
            matching_converters = [
                a
                for a in converters
                if a.get_file_prefix() in str(relative_path)
            ]

            if len(matching_converters) > 1:
                print(
                    "Multiple converters matched?",
                    full_path,
                    matching_converters,
                )
                print(1 / 0)
            elif len(matching_converters) == 0:
                pass
            else:
                converter = matching_converters[0]
                to_process.append((full_path, target_path, converter))

    pool.map(run_csv_converter, to_process, chunksize=1)

    pool.close()
    pool.join()
