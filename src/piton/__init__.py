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
    code: int

    end: datetime.datetime | None = None
    visit_id: int | None = None
    value: memoryview | float | None = None

    event_type: str | None = None

    # TODO: Implement the following
    # event_type: str | None = None

    # id: int | None = None
    # parent_id: int | None = None

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
