from __future__ import annotations

import datetime
import enum
from dataclasses import dataclass, field
from typing import Sequence


class ValueType(enum.Enum):
    NONE = 0
    NUMERIC = 1
    TEXT = 2


@dataclass
class Patient:
    patient_id: int
    events: Sequence[Event]


@dataclass(unsafe_hash=True)
class Event:
    start: datetime.datetime
    code: int

    end: datetime.datetime | None = None
    visit_id: int | None = None
    value: str | float | None = None
    value_type: ValueType = field(init=False)

    event_type: str | None = None

    def __post_init__(self) -> None:
        if self.value is None:
            self.value_type = ValueType.NONE
        elif isinstance(self.value, str):
            self.value_type = ValueType.TEXT
        else:
            self.value = float(self.value)
            self.value_type = ValueType.NUMERIC

    # TODO: Implement the following
    # event_type: str | None = None

    # id: int | None = None
    # parent_id: int | None = None
