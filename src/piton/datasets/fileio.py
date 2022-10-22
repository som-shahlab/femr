from __future__ import annotations

import csv
import dataclasses
import datetime
import io
import numbers
import tempfile
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast

import zstandard

from .. import Event, Patient


class EventWriter:
    """
    Writes events into a file.
    """

    def __init__(self, path: str):
        """
        Open a file for writing.
        """

        self.file = tempfile.NamedTemporaryFile(
            dir=path, suffix=".csv.zst", delete=False
        )
        compressor = zstandard.ZstdCompressor(level=1)
        self.o = io.TextIOWrapper(
            compressor.stream_writer(self.file),
        )
        self.rows_written = 0
        self.writer = csv.DictWriter(
            self.o,
            fieldnames=[
                "patient_id",
                "start",
                "end",
                "code",
                "visit_id",
                "value",
                "event_type",
            ],
        )
        self.writer.writeheader()

    def add_event(self, patient_id: int, event: Event) -> None:
        """
        Add an event to the record.
        """
        self.rows_written += 1
        data: Dict[str, Any] = {}
        data["patient_id"] = patient_id
        for f in dataclasses.fields(event):
            data[f.name] = getattr(event, f.name)

        data["start"] = data["start"].isoformat()
        if data["end"]:
            data["end"] = data["end"].isoformat()

        if data["value"] is None:
            data["value"] = ""
        elif isinstance(data["value"], (int, float)):
            data["value"] = str(data["value"])
        else:
            data["value"] = bytes(data["value"]).decode("utf8")

        self.writer.writerow(data)

    def close(self) -> None:
        if self.rows_written == 0:
            raise RuntimeError("Event writer with zero rows?")
        self.o.close()


class EventReader:
    def __init__(self, filename: str):
        self.filename = filename
        decompressor = zstandard.ZstdDecompressor()
        self.o = io.TextIOWrapper(
            decompressor.stream_reader(open(self.filename, "rb"))
        )
        self.reader = csv.DictReader(self.o)

    def __iter__(self) -> Iterator[Tuple[int, Event]]:
        for row in self.reader:
            id = int(row["patient_id"])
            del row["patient_id"]
            row["start"] = datetime.datetime.fromisoformat(row["start"])

            if row["end"]:
                row["end"] = datetime.datetime.fromisoformat(row["end"])
            else:
                del row["end"]

            row["code"] = int(row["code"])

            if row["visit_id"]:
                row["visit_id"] = int(row["visit_id"])
            else:
                del row["visit_id"]

            if row["value"] == "":
                row["value"] = None
            else:
                try:
                    row["value"] = float(row["value"])
                except ValueError:
                    row["value"] = memoryview(row["value"].encode("utf8"))

            if not row["event_type"]:
                del row["event_type"]

            yield (id, Event(**cast(Any, row)))

    def close(self) -> None:
        self.o.close()


class PatientReader:
    def __init__(self, filename: str):
        self.reader = EventReader(filename)

    def __iter__(self) -> Iterator[Patient]:
        last_id: Optional[int] = None
        current_events: List[Event] = []
        for id, event in self.reader:
            if id != last_id:
                if last_id is not None:
                    patient = Patient(patient_id=last_id, events=current_events)
                    yield patient
                last_id = id
                current_events = [event]
            elif last_id is not None:
                current_events.append(event)

        if last_id is not None:
            patient = Patient(patient_id=last_id, events=current_events)
            yield patient

    def close(self) -> None:
        self.reader.close()


class PatientWriter:
    """
    Writes events into a file for later use in ehr_ml extraction.

    Note: this must be used in a context manager in order to close the file properly.
    """

    def __init__(self, path: str):
        """
        Open a file for writing.
        """
        self.writer = EventWriter(path)

    def add_patient(self, patient: Patient) -> None:
        """
        Add a patient to the record.
        """
        for event in patient.events:
            self.writer.add_event(patient.patient_id, event)

    def close(self) -> None:
        self.writer.close()
