import csv
import datetime
import io
import os
import pathlib
from typing import Dict, Mapping, Sequence

import zstandard as zst

import femr
import femr.datasets
from femr.extractors.csv import run_csv_extractors


class DummyConverter(femr.extractors.csv.CSVExtractor):
    def get_patient_id_field(self) -> str:
        return "patient_id"

    def get_file_prefix(self) -> str:
        return "temp"

    def get_events(self, row: Mapping[str, str]) -> Sequence[femr.datasets.RawEvent]:
        e = femr.datasets.RawEvent(
            start=datetime.datetime(int(row["event_start"]), 1, 1),
            concept_id=int(row["event_code"]),
            value=row["event_value"],
            metadata="test",
        )
        return [e]


ROWS = [(p_id, 1995, 0, "test_value") for p_id in range(20)] + [(p_id, 2020, 1, 12.4) for p_id in range(10)]


def create_csv(tmp_path: pathlib.Path) -> str:
    path_to_file: str = os.path.join(tmp_path, "temp.csv")
    with open(path_to_file, "w") as fd:
        writer = csv.writer(fd)
        writer.writerow(["patient_id", "event_start", "event_code", "event_value"])
        writer.writerows(ROWS)
    return path_to_file


def create_csv_zst(tmp_path: pathlib.Path) -> str:
    path_to_file: str = os.path.join(tmp_path, "temp.csv.zst")
    with io.TextIOWrapper(zst.ZstdCompressor(1).stream_writer(open(path_to_file, "wb"))) as fd:
        writer = csv.writer(fd)
        writer.writerow(["patient_id", "event_start", "event_code", "event_value"])
        writer.writerows(ROWS)
    return path_to_file


def run_test(tmp_path: pathlib.Path):
    path_to_output: str = os.path.join(tmp_path, "event_collection")
    stats_dict: Dict[str, Dict[str, int]] = {}
    event_collection = run_csv_extractors(
        str(tmp_path),  # path to files
        path_to_output,
        [DummyConverter()],
        debug_folder=os.path.join(tmp_path, "lost_csv_rows/"),
        stats_dict=stats_dict,
    )
    with event_collection.reader() as event_reader:
        results = []
        for p, e in event_reader:
            results.append((p, e.start.year, e.concept_id, e.value))
        assert results == ROWS, "Events extracted from file do not match expected events."


def test_csv_zst(tmp_path: pathlib.Path) -> None:
    _ = create_csv_zst(tmp_path)
    run_test(tmp_path)


def test_csv(tmp_path: pathlib.Path) -> None:
    _ = create_csv(tmp_path)
    run_test(tmp_path)
