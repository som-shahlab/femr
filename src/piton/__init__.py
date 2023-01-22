"""The fundamental underlying Patient and Event datatypes for piton."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class Patient:
    """A patient."""

    patient_id: int
    events: List[Event]

    def resort(self) -> None:
        """Resort the events to maintain the day invariant"""
        self.events.sort()


class Event:
    """An event with a patient record.

    NOTE: Non-None field types must be specified in the order you want them decoded.

    For example,
        ```
            value: float | str | None
        ```
    Will attempt to decode the `.value` property as a `None` first, then `float`, then `str`.
    """

    start: datetime.datetime
    code: int
    value: float | str | None

    # This class can also contain any number of other optional fields
    # Optional fields can be anything, but there are a couple that are considered "standard"
    #
    # - end: datetime, the end datetime for this event
    # - visit_id: int, the visit_id this event is tied to
    # - omop_table: str, the omop table this event was pulled from
    # - clarity_table: str, the clarity table where the event comes from

    def __init__(
        self,
        start: datetime.datetime,
        code: int,
        value: float | str | None = None,
        **kwargs: Any,
    ) -> None:
        self.start = start
        self.code = code
        self.value = value

        for a, b in kwargs.items():
            self.__dict__[a] = b

    def __getattr__(self, __name: str) -> Any:
        return None

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

    def __getstate__(self) -> Dict[str, Any]:
        """Make this object pickleable (write)"""
        return self.__dict__

    def __setstate__(self, d: Dict[str, Any]) -> None:
        """Make this object pickleable (read)"""
        for a, b in d.items():
            self.__dict__[a] = b
