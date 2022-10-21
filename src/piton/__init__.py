from __future__ import annotations

import datetime
import enum
from dataclasses import dataclass, field, fields
from typing import Sequence
import numbers


@dataclass(frozen=True)
class Patient:
    patient_id: int
    events: Sequence[Event]


@dataclass(frozen=True)
class Event:
    start: datetime.datetime
    code: int # OMOP code

    end: datetime.datetime | None = None
    value: memoryview | float | None = None
    # TODO - Seems like it should be separated from the Event class as it creates unnecessary
    # interdependencies between Events (since visits are Events)
    visit_id: int | None = None

    # TODO - add the below property
    omop_table: str = None # OMOP table where this event comes from

    # TODO - rename or make __private (confusing)
    event_type: str | None = None # Clarity table name where this event comes from (for ETL purposes only)

    def __post_init__(self) -> None:
        if not (
            (self.value is None)
            or isinstance(self.value, (int, float, memoryview))
        ):
            raise TypeError("Invalid type of value passed to event", self.value)

    def __repr__(self) -> str:
        items = []
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if f.name == "value" and isinstance(value, memoryview):
                value = "'" + value.tobytes().decode("utf-8") + "'"
            items.append(f"{f.name}={value}")
        return "Event(" + ", ".join(items) + ")"
