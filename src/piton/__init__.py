"""The fundamental underlying Patient and Event datatypes for piton."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, List, Tuple


@dataclass
class Patient:
    """A patient."""

    patient_id: int
    events: List[Event]

    def resort(self) -> None:
        """Resort the events to maintain the day invariant"""
        self.events.sort()

class Event:
    """An event within a patient timeline."""

    ########
    # Required fields
    ########

    # Shared ID across all Events of the same type
    code: int

    # Time interval over which this event occurred.
    # Only specify `start` if it's a single moment in time
    start: datetime.datetime

    ########
    # Optional fields
    ########

    # Time interval over which this event occurred.
    end: datetime.datetime | None = None

    # Value associated with Event.
    value: int | float | str | None = None

    def __init__(
        self,
        start: datetime.datetime,
        code: int,
        end: datetime.datetime | None = None,
        value: float | str | None = None,
        **kwargs: Any,
    ) -> None:
        self.start = start
        self.code = code
        self.value = value
        self.end = end

        for a, b in kwargs.items():
            if b is not None:
                self.__dict__[a] = b

    def __getattr__(self, name: str) -> Any:
        return self.__dict__[name]

    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value

    def __lt__(self, other: Event) -> bool:
        def sort_key(
            a: Event,
        ) -> Tuple[datetime.datetime, int]:
            return (a.start, a.code)

        return sort_key(self) < sort_key(other)

    def __eq__(self, other: object) -> bool:
        return self.__dict__ == other.__dict__

    def __repr__(self) -> str:
        val_str = ", ".join(f"{a}={b}" for a, b in self.__dict__.items())
        return f"Event({val_str})"