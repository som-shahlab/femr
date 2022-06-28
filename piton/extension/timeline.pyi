from __future__ import annotations

import datetime
from typing import Iterator, Literal, Optional, Sequence, Union, List, Tuple, NewType

TextValue = NewType('TextValue', int)

class TimelineReader:
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
    def get_original_patient_ids(self) -> Sequence[int]: ...
    def get_dictionary(self) -> TermDictionary: ...
    def get_value_dictionary(self) -> TermDictionary: ...

class TermDictionary:
    def map(self, term: str) -> Optional[TextValue]: ...
    def get_word(self, code: TextValue) -> Optional[str]: ...
    def get_items(self) -> List[Tuple[str, TextValue]]: ...

class Patient:
    patient_id: int
    days: Sequence[PatientDay]

class PatientDay:
    date: datetime.date
    age: int
    observations: Sequence[TextValue]
    observations_with_values: Sequence[ObservationWithValue]

ObservationWithValue = Union[
    NumericObservationWithValue, TextObservationWithValue
]

class TextObservationWithValue:
    code: TextValue
    is_text: Literal[True]
    text_value: TextValue

class NumericObservationWithValue:
    code: TextValue
    is_text: Literal[False]
    numeric_value: float

def create_timeline(
    event_dir: str,
    output_filename: str,
) -> None: ...