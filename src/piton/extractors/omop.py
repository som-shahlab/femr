"""A class and program for converting OMOP v5 sources to piton."""

from __future__ import annotations

import dataclasses
import datetime
from typing import Mapping, Optional, Sequence

from piton import Event
from piton.extractors.csv import CSVExtractor

OMOP_BIRTH = 4216316


class _DemographicsConverter(CSVExtractor):
    """Convert the OMOP demographics table to events."""

    def get_patient_id_field(self) -> str:
        return "person_id"

    def get_file_prefix(self) -> str:
        return "person"

    def get_events(self, row: Mapping[str, str]) -> Sequence[Event]:
        if row["birth_datetime"]:
            birth = datetime.datetime.fromisoformat(row["birth_datetime"])
        else:
            year = 1900
            month = 1
            day = 1

            if row["year_of_birth"]:
                year = int(row["year_of_birth"])
            else:
                raise RuntimeError(
                    "Should always have at least a year of birth?"
                )

            if row["month_of_birth"]:
                month = int(row["month_of_birth"])

            if row["day_of_birth"]:
                day = int(row["day_of_birth"])

            birth = datetime.datetime(year=year, month=month, day=day)

        return [
            # 4216316 is the OMOP birth code
            Event(start=birth, code=4216316, event_type=row["load_table_id"])
        ] + [
            Event(
                start=birth,
                code=int(row[target]),
                event_type=row["load_table_id"],
            )
            for target in [
                "gender_concept_id",
                "ethnicity_concept_id",
                "race_concept_id",
            ]
            if row[target] != "0"
        ]


def _try_numeric(val: str) -> float | memoryview | None:
    if val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return memoryview(val.encode("utf8"))


@dataclasses.dataclass
class _ConceptTableConverter(CSVExtractor):
    """A generic OMOP converter for handling tables that contain a single concept."""

    prefix: str

    file_suffix: str = ""
    concept_id_field: Optional[str] = None
    string_value_field: Optional[str] = None
    numeric_value_field: Optional[str] = None

    def get_patient_id_field(self) -> str:
        return "person_id"

    def get_file_prefix(self) -> str:
        if self.file_suffix:
            return self.prefix + "_" + self.file_suffix
        else:
            return self.prefix

    def _get_date(
        self, date_field: str, row: Mapping[str, str]
    ) -> Optional[datetime.datetime]:
        """Extract the highest resolution date from the raw data."""
        for attempt in (date_field + "time", date_field):
            if attempt in row and row[attempt] != "":
                return datetime.datetime.fromisoformat(row[attempt])

        return None

    def get_events(self, row: Mapping[str, str]) -> Sequence[Event]:
        def normalize_to_float_if_possible(
            field_name: Optional[str], value: memoryview | float | None
        ) -> memoryview | float | None:
            if field_name is not None:
                val = _try_numeric(row[field_name])
                if val is not None:
                    return val
            return value

        value = normalize_to_float_if_possible(self.string_value_field, None)
        value = normalize_to_float_if_possible(self.numeric_value_field, value)

        concept_id_field = self.concept_id_field or (
            self.prefix + "_concept_id"
        )
        code = int(row[concept_id_field])
        if code == 0:
            # The following are worth recovering even without the code ...
            if self.prefix == "note":
                code = 26
            elif self.prefix == "visit":
                code = 8
            else:
                return []

        if (self.prefix + "_start_date") in row:
            start = self._get_date(self.prefix + "_start_date", row)
            end = self._get_date(self.prefix + "_end_date", row)
        else:
            start = self._get_date(self.prefix + "_date", row)
            end = None

        if start is None:
            raise RuntimeError(
                "Could not find a date field for "
                + repr(self)
                + " "
                + repr(row)
            )

        if "visit_occurrence_id" in row and row["visit_occurrence_id"]:
            visit_id = int(row["visit_occurrence_id"])
        else:
            visit_id = None

        return [
            Event(
                start=start,
                code=code,
                value=value,
                end=end,
                visit_id=visit_id,
                event_type=row["load_table_id"],
            )
        ]


def get_omop_csv_extractors() -> Sequence[CSVExtractor]:
    """Get the list of OMOP Converters."""
    converters = [
        _DemographicsConverter(),
        _ConceptTableConverter(
            prefix="drug_exposure",
            concept_id_field="drug_concept_id",
        ),
        _ConceptTableConverter(
            prefix="visit",
            file_suffix="occurrence",
        ),
        _ConceptTableConverter(
            prefix="condition",
            file_suffix="occurrence",
        ),
        _ConceptTableConverter(
            prefix="death", concept_id_field="death_type_concept_id"
        ),
        _ConceptTableConverter(
            prefix="procedure",
            file_suffix="occurrence",
        ),
        _ConceptTableConverter(
            prefix="device_exposure", concept_id_field="device_concept_id"
        ),
        _ConceptTableConverter(
            prefix="measurement",
            string_value_field="value_source_value",
            numeric_value_field="value_as_number",
        ),
        _ConceptTableConverter(
            prefix="observation",
            string_value_field="value_as_string",
            numeric_value_field="value_as_number",
        ),
        _ConceptTableConverter(
            prefix="note",
            concept_id_field="note_class_concept_id",
            string_value_field="note_text",
        ),
    ]

    return converters
