"""The fundamental underlying Patient and Event datatypes for piton."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, fields, field
from typing import Sequence, Mapping, Any, Optional


@dataclass(frozen=True)
class Patient:
    """A patient."""

    patient_id: int
    events: Sequence[Event]


@dataclass(frozen=True, order=True)
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
    value: float | str | None = None

    # Arbitrary metadata storage for piton.
    # Note that this is saved using pickle, so it must support pickle serialization / deserialization
    #
    # A couple of common fields within here are frequently defined:
    # - end: datetime, the end datetime for this event
    # - visit_id: int, the visit_id this event is tied to
    # - omop_table: str, the omop table this event was pulled from
    # - clarity_table: str, the clarity table where the event comes from
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def get_int(self, name: str) -> Optional[int]:
        """Get an integer value with the provided name."""
        value = self.metadata.get(name)
        if value is None:
            return value
        assert isinstance(value, int)
        return value

    def get_str(self, name: str) -> Optional[str]:
        """Get a string value with the provided name."""
        value = self.metadata.get(name)
        if value is None:
            return value
        assert isinstance(value, str)
        return value

    def get_datetime(self, name: str) -> Optional[datetime.datetime]:
        """Get a datetime value with the provided name."""
        value = self.metadata.get(name)
        if value is None:
            return value
        assert isinstance(value, datetime.datetime)
        return value

    def __post_init__(self) -> None:
        """Verify that the event is constructed correctly."""
        if not (
            (self.value is None) or isinstance(self.value, (int, float, str))
        ):
            raise TypeError("Invalid type of value passed to event", self.value)
