"""The fundamental underlying Patient and Event datatypes for femr."""

from __future__ import annotations

import datetime
import importlib.metadata
from dataclasses import dataclass
from typing import Any, Dict, Sequence


@dataclass
class Patient:
    """A patient."""

    patient_id: int
    events: Sequence[Event]


class Event:
    """An event within a patient record."""

    start: datetime.datetime
    code: str
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
        code: str,
        value: float | str | None = None,
        **kwargs: Any,
    ) -> None:
        self.start = start
        self.code = code
        self.value = value

        for a, b in kwargs.items():
            if b is not None:
                self.__dict__[a] = b

    def __getattr__(self, __name: str) -> Any:
        return None

    def __eq__(self, other: object) -> bool:
        if other is None:
            return False

        def get_val(val: Any) -> Any:
            other = {}
            if val.__dict__ is not None:
                for a, b in val.__dict__.items():
                    if a not in ("code", "start", "value") and b is not None:
                        other[a] = b

            return (val.code, val.start, val.value, other)

        return bool(get_val(self) == get_val(other))

    def __repr__(self) -> str:
        val_str = ", ".join(f"{a}={b}" for a, b in self.__dict__.items() if b is not None)
        return f"Event({val_str})"

    def __getstate__(self) -> Dict[str, Any]:
        """Make this object pickleable (write)"""
        return self.__dict__

    def __setstate__(self, d: Dict[str, Any]) -> None:
        """Make this object pickleable (read)"""
        for a, b in d.items():
            self.__dict__[a] = b


try:
    # __package__ allows for the case where __name__ is "__main__"
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
