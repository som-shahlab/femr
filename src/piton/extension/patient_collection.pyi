# Implementation is in patient_collection_extension.cpp

from __future__ import annotations

import datetime
from typing import Iterator, Literal, Optional, Sequence, Union, List, Tuple, NewType

TextValue = NewType('TextValue', int)

class PatientCollection:
    def __init__(self, filename: str, readall: bool = ...): ...
    def get_patient(
        self,
        patient_id: int,
        start_date: Optional[datetime.date] = ...,
        end_date: Optional[datetime.date] = ...,
    ) -> Patient: ...
    def get_patients(
        self,
        patient_ids: Optional[Sequence[int]] = ...,
        start_date: Optional[datetime.date] = ...,
        end_date: Optional[datetime.date] = ...,
    ) -> Iterator[Patient]: ...
    def get_patient_ids(self) -> Sequence[int]: ...