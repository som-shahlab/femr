# Implementation is in patient_collection_extension.cpp

from __future__ import annotations

import datetime
from typing import (
    Iterator,
    Literal,
    Optional,
    Sequence,
    Union,
    List,
    Tuple,
    NewType,
)

from .. import Patient

from ..datasets import PatientCollection

TextValue = NewType("TextValue", int)

class PatientDatabase:
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
    def close(self) -> None: ...

def convert_patient_collection_to_patient_database(
    patient_collection: PatientCollection, target_path: str
) -> PatientDatabase: ...
