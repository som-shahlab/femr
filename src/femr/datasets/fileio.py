"""FileIO utilities for reading and writing data."""
from __future__ import annotations

import base64
import csv
import datetime
import io
import pickle
import tempfile
import warnings
from typing import Any, Dict, Iterator, List, Optional, Tuple

import zstandard

from .. import Event, Patient


def _encode_value(a: float | str | None) -> str:
    """Try to decode a string value."""
    if a is None:
        return ""
    else:
        return str(a)


def _decode_value(a: str) -> float | str | None:
    """Try to decode a string value."""
    if a == "":
        return None
    try:
        return float(a)
    except ValueError:
        return a


class EventWriter:
    """Writes events into a file."""

    def __init__(self, path: str):
        """Open a file for writing."""
        self.file = tempfile.NamedTemporaryFile(dir=path, suffix=".csv.zst", delete=False)
        compressor = zstandard.ZstdCompressor(level=1)
        self.o = io.TextIOWrapper(
            compressor.stream_writer(self.file),
        )
        self.rows_written = 0
        self.writer = csv.DictWriter(
            self.o,
            fieldnames=["patient_id", "start", "code", "value", "metadata"],
        )
        self.writer.writeheader()

    def add_event(self, patient_id: int, event: Event) -> None:
        """Add an event to the record."""
        self.rows_written += 1
        data: Dict[str, Any] = {}

        data["patient_id"] = patient_id
        data["start"] = event.start.isoformat()
        data["code"] = str(event.code)
        data["value"] = _encode_value(event.value)
        data["metadata"] = base64.b64encode(
            pickle.dumps({a: b for a, b in event.__dict__.items() if a not in ("start", "code", "value")})
        ).decode("utf8")

        self.writer.writerow(data)

    def close(self) -> None:
        """Close the event writer."""
        if self.rows_written == 0:
            warnings.warn(f"Zero rows were written by the `EventWriter` for file: {self.file.name}")
        self.o.close()


class EventReader:
    """Read events from an event file."""

    def __init__(self, filename: str):
        """Open the event file."""
        self.filename = filename
        decompressor = zstandard.ZstdDecompressor()
        self.o = io.TextIOWrapper(decompressor.stream_reader(open(self.filename, "rb")))
        self.reader = csv.DictReader(self.o)

    def __iter__(self) -> Iterator[Tuple[int, Event]]:
        """Iterate over each event."""
        for row in self.reader:
            id = int(row["patient_id"])

            code = int(row["code"])
            start = datetime.datetime.fromisoformat(row["start"])
            value = _decode_value(row["value"])
            metadata = pickle.loads(base64.b64decode(row["metadata"]))

            yield (id, Event(start=start, code=code, value=value, **metadata))

    def close(self) -> None:
        """Close the event file."""
        self.o.close()


class PatientReader:
    """Read patients from a patient file."""

    def __init__(self, filename: str):
        """Open the file with the given filename."""
        self.reader = EventReader(filename)

    def __iter__(self) -> Iterator[Patient]:
        """Iterate over each patient."""
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
        """Close the patient reader."""
        self.reader.close()


class PatientWriter:
    """
    Writes events into a file for later use in femr extraction.

    Note: this must be used in a context manager in order to close the file properly.
    """

    def __init__(self, path: str):
        """Open a file for writing."""
        self.writer = EventWriter(path)

    def add_patient(self, patient: Patient) -> None:
        """Add a patient to the record."""
        for event in patient.events:
            self.writer.add_event(patient.patient_id, event)

    def close(self) -> None:
        """Close the patient writer."""
        self.writer.close()
