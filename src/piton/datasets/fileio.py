from __future__ import annotations
from optparse import Option

from .. import Event, Patient
import gzip
import csv
import datetime
import io

from typing import (
    Iterable,
    Tuple,
    Optional,
    cast,
    Any,
    List,
    IO,
    Sequence,
    Iterator,
)

import dataclasses

import tempfile


class EventWriter:
    """
    Writes events into a file.
    """

    def __init__(self, path: str):
        """
        Open a file for writing.
        """

        self.file = tempfile.NamedTemporaryFile(
            prefix=path, suffix="csv.gz", delete=False
        )
        self.o = io.TextIOWrapper(
            cast(IO[bytes], gzip.GzipFile(fileobj=self.file))
        )
        self.writer = csv.DictWriter(
            self.o,
            fieldnames=[
                "patient_id",
                "start",
                "end",
                "code",
                "value",
                "event_type",
                "id",
                "parent_id",
            ],
        )
        self.writer.writeheader()

    def add_event(self, patient_id: int, event: Event) -> None:
        """
        Add an event to the record.
        """
        data = dataclasses.asdict(event)
        data["patient_id"] = patient_id
        self.writer.writerow(data)

    def close(self) -> None:
        self.o.close()


class EventReader:
    def __init__(self, filename: str):
        self.filename = filename
        self.o = gzip.open(self.filename, "rt")
        self.reader = csv.DictReader(self.o)

    def __iter__(self) -> Iterator[Tuple[int, Event]]:
        for row in self.reader:
            id = int(row["patient_id"])
            del row["patient_id"]
            row["start"] = datetime.datetime.fromisoformat(row["start"])
            if row["end"] != "":
                row["end"] = datetime.datetime.fromisoformat(row["end"])
            else:
                row["end"] = None
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
