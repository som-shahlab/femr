from __future__ import annotations

import abc
import contextlib
import csv
import io
import multiprocessing
import os
import sys
from typing import Mapping, Sequence, Tuple, Optional

import zstandard

from .. import Event
from ..datasets import EventCollection

csv.field_size_limit(sys.maxsize)


class CSVConverter(abc.ABC):
    """
    An interface for converting a csv into events.
    """

    def __init__(self) -> None:
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


def run_csv_converter(
    args: Tuple[str, EventCollection, CSVConverter, Optional[str]]
) -> None:
    source, target, converter, debug_file = args
    try:
        with contextlib.ExitStack() as stack:
            f = stack.enter_context(
                io.TextIOWrapper(
                    zstandard.ZstdDecompressor().stream_reader(
                        open(source, "rb")
                    )
                )
            )

            debug_writer = None

            reader = csv.DictReader(f)
            with contextlib.closing(target.create_writer()) as o:
                for row in reader:
                    lower_row = {a.lower(): b for a, b in row.items()}
                    events = converter.get_events(lower_row)
                    if events:
                        for event in events:
                            o.add_event(
                                int(row[converter.get_patient_id_field()]),
                                event,
                            )
                    else:
                        # This is a bad row, should be inspected further
                        if debug_file is not None:
                            if debug_writer is None:
                                os.makedirs(
                                    os.path.dirname(debug_file), exist_ok=True
                                )
                                debug_f = stack.enter_context(
                                    io.TextIOWrapper(
                                        zstandard.ZstdCompressor(
                                            level=1
                                        ).stream_writer(open(debug_file, "wb"))
                                    )
                                )
                                assert reader.fieldnames is not None
                                debug_writer = csv.DictWriter(
                                    debug_f,
                                    fieldnames=list(reader.fieldnames)
                                    + ["converter"],
                                )
                                debug_writer.writeheader()
                            row["converter"] = repr(converter)
                            debug_writer.writerow(row)

    except Exception as e:
        print("Failing on", source, converter)
        raise e


def run_csv_converters(
    source_csvs: str,
    target_location: str,
    converters: Sequence[CSVConverter],
    num_threads: int = 1,
    debug_folder: Optional[str] = None,
) -> EventCollection:

    if debug_folder:
        os.mkdir(debug_folder)

    target = EventCollection(target_location)

    files_per_converter = [0 for _ in converters]

    to_process = []

    for root, dirs, files in os.walk(source_csvs):
        for name in files:
            full_path = os.path.join(root, name)
            relative_path = os.path.relpath(full_path, source_csvs)
            matching_converters = [
                (i, a)
                for i, a in enumerate(converters)
                if (
                    str(relative_path).startswith(a.get_file_prefix() + ".csv")
                    or str(relative_path).startswith(a.get_file_prefix() + "/")
                )
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
                i, converter = matching_converters[0]
                files_per_converter[i] += 1

                if debug_folder is not None:
                    debug_path = os.path.join(debug_folder, relative_path)
                else:
                    debug_path = None

                to_process.append((full_path, target, converter, debug_path))

    for count, c in zip(files_per_converter, converters):
        if count == 0:
            print("Could not find any files for", c)

    with multiprocessing.Pool(num_threads) as pool:
        for _ in pool.imap_unordered(run_csv_converter, to_process):
            pass

    return target
