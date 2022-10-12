from __future__ import annotations

import itertools
import datetime
import collections.abc

from dataclasses import dataclass
from typing import Iterable, List, Callable, Optional


@dataclass
class Patient:
    patient_id: int
    events: List[Event]


@dataclass
class Event:
    start: datetime.datetime

    code: str
    value: str | float | None = None

    end: datetime.datetime | None = None

    event_type: str | None = None

    id: int | None = None
    parent_id: int | None = None
