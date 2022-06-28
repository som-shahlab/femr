from __future__ import annotations

import itertools
import datetime

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class Patient:
    patient_id: int
    events: List[Event]


@dataclass
class Event:
    start: datetime.datetime

    code: str
    value: str | float | None = None

    end: datetime.datetime | None = None

    event_type: str | None = None

    id: int | None = None
    parent_id: int | None = None


class EventWriter:
    """
    Writes events into a file for later use in ehr_ml extraction.

    Note: this must be used in a context manager in order to close the file properly.
    """

    def __init__(self, filename):
        """
        Open a file for writing.
        """
        self.filename = filename

    def __enter__(self):
        self.o = gzip.open(self.filename, "wt")
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

        return self

    def add_event(self, patient_id: int, event: Event):
        """
        Add an event to the record.
        """
        data = dataclasses.asdict(event)
        data["patient_id"] = patient_id
        self.writer.writerow(data)

    def __exit__(self, exc_type, exc_value, traceback):
        self.o.close()

class EventReader:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.o = gzip.open(self.filename, "rt")
        self.reader = csv.DictReader(self.o)
        return self

    def get_events(self) -> Iterable[(int, Event)]:
        for row in self.reader:
            id = row["patient_id"]
            del row["patient_id"]
            row["start"] = datetime.datetime.fromisoformat(row["start"])
            if row["end"] != "":
                row["end"] = datetime.datetime.fromisoformat(row["end"])
            else:
                row["end"] = None
            yield (id, Event(**row))

    def __exit__(self, exc_type, exc_value, traceback):
        self.o.close()


class PatientReader:
    def __init__(self, filename):
        self.reader = EventReader(filename)

    def __enter__(self):
        self.reader = self.reader.__enter__()

    def get_patients(self) -> Iterable[Patient]:
        last_id = None
        for id, event in itertools.chain(
            self.reader.get_events(), [(None, None)]
        ):
            if id != last_id:
                if last_id is not None:
                    patient = Patient(
                        patient_id=last_id, events=current_events
                    )
                    yield patient
                last_id = id
                current_events = [event]
            elif last_id is not None:
                current_events.append(event)


    def __exit__(self, exc_type, exc_value, traceback):
        self.reader.close()

    
class PatientWriter:
    """
    Writes events into a file for later use in ehr_ml extraction.

    Note: this must be used in a context manager in order to close the file properly.
    """

    def __init__(self, filename):
        """
        Open a file for writing.
        """
        self.writer = EventWriter(filename)

    def __enter__(self):
        self.writer = self.writer.__enter__()

        return self

    def add_patient(self, patient: Patient):
        """
        Add a patient to the record.
        """
        for event in patient.events:
            self.add_patient(patient.patient_id, event)

    def __exit__(self, exc_type, exc_value, traceback):
        self.writer.close()