from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class RawPatient:
    """A patient."""

    patient_id: int
    events: List[RawEvent]

    def resort(self) -> None:
        """Resort the events to maintain the day invariant"""
        self.events.sort()


class RawEvent:
    """An event with a patient record.

    NOTE: Non-None field types must be specified in the order you want them decoded.

    For example,
        ```
            value: float | str | None
        ```
    Will attempt to decode the `.value` property as a `None` first, then `float`, then `str`.
    """

    start: datetime.datetime
    concept_id: int
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
        concept_id: int,
        value: float | str | None = None,
        **kwargs: Any,
    ) -> None:
        self.start = start
        self.concept_id = concept_id
        self.value = value

        for a, b in kwargs.items():
            if b is not None:
                self.__dict__[a] = b

    def __getattr__(self, __name: str) -> Any:
        return None

    def __setattr__(self, name: str, value: Any) -> None:
        if value is not None:
            self.__dict__[name] = value

    def __lt__(self, other: RawEvent) -> bool:
        def sort_key(
            a: RawEvent,
        ) -> Tuple[datetime.datetime, int]:
            return (a.start, a.concept_id)

        return sort_key(self) < sort_key(other)

    def __eq__(self, other: object) -> bool:
        if other is None:
            return False

        def get_val(val: Any) -> Any:
            other = {}
            if val.__dict__ is not None:
                for a, b in val.__dict__.items():
                    if a not in ("concept_id", "start", "value") and b is not None:
                        other[a] = b

            return (val.concept_id, val.start, val.value, other)

        return bool(get_val(self) == get_val(other))

    def __repr__(self) -> str:
        val_str = ", ".join(f"{a}={b}" for a, b in self.__dict__.items())
        return f"RawEvent({val_str})"

    def __getstate__(self) -> Dict[str, Any]:
        """Make this object pickleable (write)"""
        return self.__dict__

    def __setstate__(self, d: Dict[str, Any]) -> None:
        """Make this object pickleable (read)"""
        for a, b in d.items():
            self.__dict__[a] = b
