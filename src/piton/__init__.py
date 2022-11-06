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
    """An event within a patient timeline."""

    ########
    # Required fields
    ########

    # Shared ID across all Events of the same type
    concept_id: int

    # Time interval over which this event occurred.
    # Only specify `start` if it's a single moment in time
    start: datetime.datetime

    ########
    # Optional fields
    ########

    # Time interval over which this event occurred.
    end: datetime.datetime | None = None

    # Value associated with Event.
    # If text, then use `memoryview` (e.g. clinical note)
    # If numeric, then use `float` (e.g. lab value)
    value: int | float | memoryview | None = None

    # Any data associated with this Event that we need to carry
    # through the ETL in order to apply transformations to the Event
    # Examples of fields within `metadata`:
    #   - visit_id
    #   - load_table_etl
    metadata: dict | None = None  # type: ignore

    def __post_init__(self) -> None:
        """Verify that the event is constructed correctly."""
        # Check `value`
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

    def __hash__(self) -> int:
        """Need to explicitly define for `self.metadata` since dicts() are unhashable by default."""
        return hash(
            (
                self.concept_id,
                self.start,
                self.end,
                self.value,
                str(self.metadata),
            )
        )
