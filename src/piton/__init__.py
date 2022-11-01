"""The fundamental underlying Patient and Event datatypes for piton."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, fields
from typing import Sequence


@dataclass(frozen=True)
class Patient:
    """A patient."""

    patient_id: int
    events: Sequence[Event]


@dataclass(frozen=True)
class Event:
    """An event with a patient record."""

    start: datetime.datetime
    code: int  # Is this an OMOP code (or is it an index into your Piton Ontology object?)

    end: datetime.datetime | None = None
    value: memoryview | float | None = None
    # TODO - Seems like `visit_id` should be separated from the Event class as it creates a weird
    # interdependency between Events (since visits are Events)
    visit_id: int | None = None

    # TODO - add the below property
    omop_table: str = None  # OMOP table where this event comes from

    # TODO - rename or make __private (confusing)
    event_type: str | None = None  # Clarity table name where this event comes from (for ETL purposes only)

    def __post_init__(self) -> None:
        """Verify that the event is constructed correctly."""
        if not (
            (self.value is None)
            or isinstance(self.value, (int, float, memoryview))
        ):
            raise TypeError("Invalid type of value passed to event", self.value)

    def __repr__(self) -> str:
        """Convert an event to a string."""
        items = []
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if f.name == "value" and isinstance(value, memoryview):
                value = "'" + value.tobytes().decode("utf-8") + "'"
            items.append(f"{f.name}={value}")
        return "Event(" + ", ".join(items) + ")"
