"""Helper utilties for converting CSV files into Event files."""

from __future__ import annotations

import abc
import collections
import contextlib
import csv
import io
import logging
import multiprocessing
import os
import sys
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import zstandard

from femr.datasets import EventCollection, RawEvent

# Note that we want to support huge CSV records
csv.field_size_limit(sys.maxsize)


class CSVExtractor(abc.ABC):
    """An interface for converting a csv into events."""

    def __init__(self) -> None:
        """Create the CSVConverter."""
        super().__init__()

    @abc.abstractmethod
    def get_patient_id_field(self) -> str:
        """Return the field that contains the patient_id."""

    @abc.abstractmethod
    def get_file_prefix(self) -> str:
        """Return the prefix for files this converter will trigger on."""
        ...

    @abc.abstractmethod
    def get_events(self, row: Mapping[str, str]) -> Sequence[RawEvent]:
        """Return the events generated for a particular row."""
        ...


def _run_csv_extractor(
    args: Tuple[str, EventCollection, CSVExtractor, str, Optional[str]]
) -> Tuple[str, Dict[str, int]]:
    """
    Run a single csv converter, returns the prefix and the count dicts.

    This function is supposed to run with a multiprocess pool.
    """
    source, target, extractor, delimiter, debug_file = args
    stats: Dict[str, int] = collections.defaultdict(int)
    try:
        with contextlib.ExitStack() as stack:
            f: Iterable[str]
            if source.endswith(".csv.zst"):
                # Support Zstandard compressed CSVs
                f = stack.enter_context(
                    io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(source, "rb")))
                )
            else:
                # Support normal CSVs
                f = stack.enter_context(open(source, "r"))

            debug_writer = None

            reader = csv.DictReader(f, delimiter=delimiter)
            with contextlib.closing(target.create_writer()) as o:
                for row in reader:
                    lower_row = {a.lower(): b for a, b in row.items()}
                    events = extractor.get_events(lower_row)
                    stats["input_rows"] += 1
                    if events:
                        stats["valid_rows"] += 1
                        for event in events:
                            stats["valid_events"] += 1
                            o.add_event(
                                int(lower_row[extractor.get_patient_id_field()]),
                                event,
                            )
                    else:
                        stats["invalid_rows"] += 1
                        # This is a bad row, should be inspected further
                        if debug_file is not None:
                            if debug_writer is None:
                                os.makedirs(os.path.dirname(debug_file), exist_ok=True)
                                if debug_file.endswith(".csv.zst"):
                                    # Support Zstandard compressed CSVs
                                    debug_f = stack.enter_context(
                                        io.TextIOWrapper(
                                            zstandard.ZstdCompressor(level=1).stream_writer(open(debug_file, "wb"))
                                        )
                                    )
                                else:
                                    # Support normal CSVs
                                    debug_f = stack.enter_context(open(debug_file, "w"))
                                assert reader.fieldnames is not None
                                debug_writer = csv.DictWriter(
                                    debug_f,
                                    fieldnames=list(reader.fieldnames) + ["extractor"],
                                )
                                debug_writer.writeheader()
                            row["extractor"] = repr(extractor)
                            debug_writer.writerow(row)
        return (extractor.get_file_prefix(), stats)

    except Exception as e:
        logging.error("Failing on %s %s", source, extractor)
        raise e


def run_csv_extractors(
    source_csvs: str,
    target_location: str,
    extractors: Sequence[CSVExtractor],
    num_threads: int = 1,
    delimiter: str = ",",
    debug_folder: Optional[str] = None,
    stats_dict: Optional[Dict[str, Dict[str, int]]] = None,
) -> EventCollection:
    """Run a collection of CSV converters over a directory, producing an EventCollection.

    Args:
        source_csvs: A path to the directory containing the source csvs.
        target_location: A path where you want to store the EventCollection.
        converters: A series of classes that implement the CSVConverter API.
        num_threads: The number of threads to use when converting.
        debug_folder: An optional directory where the unmapped rows should be stored for debuggin
        stats_dict: An optional dictionary to store statistics about the conversion process.


    Returns:
        An EventCollection storing the resulting events
    """
    stats: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))

    if debug_folder:
        os.makedirs(debug_folder, exist_ok=True)

    target = EventCollection(target_location)

    files_per_extractor = [0 for _ in extractors]

    to_process = []

    for root, dirs, files in os.walk(source_csvs):
        for name in files:
            full_path = os.path.join(root, name)
            relative_path = os.path.relpath(full_path, source_csvs)
            matching_extractors = [
                (i, a)
                for i, a in enumerate(extractors)
                if (
                    str(relative_path).startswith(a.get_file_prefix() + ".csv")
                    or str(relative_path).startswith(a.get_file_prefix() + "/")
                )
            ]
            if len(matching_extractors) > 1:
                raise RuntimeError("Multiple extractors matched " + full_path + " " + str(matching_extractors))
            elif len(matching_extractors) == 0:
                pass
            else:
                i, extractor = matching_extractors[0]
                files_per_extractor[i] += 1

                if debug_folder is not None:
                    debug_path = os.path.join(debug_folder, relative_path)
                else:
                    debug_path = None

                to_process.append((full_path, target, extractor, delimiter, debug_path))

    for count, c in zip(files_per_extractor, extractors):
        if count == 0:
            print("Could not find any files for extractor", c)

    with multiprocessing.Pool(num_threads) as pool:
        for prefix, s in pool.imap_unordered(_run_csv_extractor, to_process):
            for k, v in s.items():
                stats[prefix][k] += v

    if stats_dict is not None:
        stats_dict.update(stats)

    return target
