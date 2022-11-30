"""FileIO utilities for reading and writing data."""
from __future__ import annotations

import csv
import dataclasses
import datetime
import io
import pickle
import base64
import tempfile
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast

import zstandard

from .. import Event, Patient


class EventWriter:
    """Writes events into a file."""

    def __init__(self, path: str):
        """Open a file for writing."""
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
            fieldnames=["patient_id"]
            + [f.name for f in dataclasses.fields(Event)],
        )
        self.writer.writeheader()

    def _encode_field(self, field_value: str, field_name: str) -> Any:
        """Encode a field for serialization.

        Must stay in sync with `EventReader._decode_field()`

        Currently supports fields of the following types:
            - datetime.datetime
            - int
            - float
            - memoryview
            - str
            - None (optional)
        """
        if field_value is None:
            # None => ""
            return ""
        elif isinstance(field_value, datetime.datetime):
            # Datetime => ISO format string
            return field_value.isoformat()
        elif isinstance(field_value, int):
            # Int => String
            return str(field_value)
        elif isinstance(field_value, float):
            # Float => String
            return str(field_value)
        elif isinstance(field_value, memoryview):
            return bytes(field_value).decode("utf8")
        elif isinstance(field_value, str):
            return field_value
        elif isinstance(field_value, dict):
            return base64.b85encode(pickle.dumps(field_value)).decode("utf8")
        else:
            raise ValueError(
                f"EventWriter does not have a method for fields with the type of {field_name}"
            )

    def add_event(self, patient_id: int, event: Event) -> None:
        """Add an event to the record."""
        self.rows_written += 1
        data: Dict[str, Any] = {}
        data["patient_id"] = patient_id

        for field in dataclasses.fields(event):
            data[field.name] = self._encode_field(
                getattr(event, field.name), field.name
            )

        self.writer.writerow(data)

    def close(self) -> None:
        """Close the event writer."""
        if self.rows_written == 0:
            raise RuntimeError("Event writer with zero rows?")
        self.o.close()


class EventReader:
    """Read events from an event file."""

    def __init__(self, filename: str):
        """Open the event file."""
        self.filename = filename
        decompressor = zstandard.ZstdDecompressor()
        self.o = io.TextIOWrapper(
            decompressor.stream_reader(open(self.filename, "rb"))
        )
        self.reader = csv.DictReader(self.o)
        self.schema: Dict[str, List[str]] = {
            field.name: [x.strip() for x in field.type.split("|")]
            for field in dataclasses.fields(Event)
        }

    def _decode_field(self, field_value: str, field_name: str) -> Any:
        """Decode a field for deserialization.

        Must stay in sync with `EventWriter._encode_field()`

        This tries to decode the field in the order that its non-None types
        are listed in the `Event` class definition.
        For example, the field:
            ```
                value: float | str | None
            ```
        will first try to decode `value` into a `None`, then a `float`, finally a `str`

        Currently supports fields of the following types:
            - datetime.datetime
            - int
            - float
            - memoryview
            - str
            - None (optional)
        """
        field_types: List = self.schema[field_name]
        if field_value == "" and "None" in field_types:
            # Need to check `None` first b/c it's represented by the empty string,
            # so it will incorrectly trigger casts like `str()` below
            return None
        for field_type in field_types:
            try:
                if field_type == "datetime.datetime":
                    return datetime.datetime.fromisoformat(field_value)
                elif field_type == "int":
                    return int(field_value)
                elif field_type == "float":
                    return float(field_value)
                elif field_type == "str":
                    return str(field_value)
                elif field_type == "memoryview":
                    return memoryview(field_value.encode("utf8"))
                elif field_type == "Mapping[str, Any]":
                    return pickle.loads(
                        base64.b85decode(field_value.encode("utf8"))
                    )
                else:
                    raise NotImplementedError(
                        f"Unrecognized field type {field_type} for {field_name}"
                    )
            except ValueError:
                # An exception occurs when we guess the wrong `field_type` for a specific field.
                # This is expected to occur for fields with multiple possible non-None
                # types (e.g. 'value: float | str | None'). We just catch the error,
                # then proceed to trying to decode this field with the next possible type.
                continue
        raise ValueError(f"Could not decode {field_name} with {field_types}")

    def __iter__(self) -> Iterator[Tuple[int, Event]]:
        """Iterate over each event."""
        for row in self.reader:
            # Remove `patient_id` from Event since it's not a property of the `Event` object
            id = int(row["patient_id"])
            del row["patient_id"]
            for field in dataclasses.fields(Event):
                row[field.name] = self._decode_field(
                    row[field.name], field.name
                )
            yield (id, Event(**cast(Any, row)))

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
    Writes events into a file for later use in piton extraction.

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
