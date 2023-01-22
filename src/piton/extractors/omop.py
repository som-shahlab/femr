"""A class and program for converting OMOP v5 sources to piton."""

from __future__ import annotations

import dataclasses
import datetime
from typing import Any, Dict, Mapping, Optional, Sequence

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
            Event(
                start=birth,
                code=4216316,
                omop_table="person",
                clarity_table=row["load_table_id"],
            )
        ] + [
            Event(
                start=birth,
                code=int(row[target]),
                omop_table="person",
                clarity_table=row["load_table_id"],
                source_code=row[target.replace("_concept_id", "_source_value")],
            )
            for target in [
                "gender_concept_id",
                "ethnicity_concept_id",
                "race_concept_id",
            ]
            if row[target] != "0"
        ]


def _get_date(
    row: Mapping[str, str], date_field: str
) -> Optional[datetime.datetime]:
    """Extract the highest resolution date from the raw data."""
    for attempt in (date_field + "time", date_field):
        if attempt in row and row[attempt] != "":
            return datetime.datetime.fromisoformat(row[attempt])

    return None


def _try_numeric(val: str) -> float | str | None:
    if val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return val


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

    def get_events(self, row: Mapping[str, str]) -> Sequence[Event]:
        def normalize_to_float_if_possible(
            field_name: Optional[str], value: str | float | None
        ) -> str | float | None:
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
            elif self.prefix == "visit_detail":
                code = 8
            else:
                return []

        if (self.prefix + "_start_date") in row:
            start = _get_date(row, self.prefix + "_start_date")
            end = _get_date(row, self.prefix + "_end_date")
        else:
            start = _get_date(row, self.prefix + "_date")
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

        if "unit_concept_id" in row and row["unit_concept_id"]:
            unit_concept_id = int(row["unit_concept_id"])
        else:
            unit_concept_id = None

        if "unit_source_value" in row and row["unit_source_value"]:
            unit = row["unit_source_value"]
        else:
            unit = None

        metadata: Dict[str, Any] = {
            "omop_table": self.get_file_prefix(),
            "clarity_table": row["load_table_id"],
        }

        if visit_id is not None:
            metadata["visit_id"] = visit_id

        if end is not None:
            metadata["end"] = end

        if unit is not None:
            metadata["unit"] = unit

        if unit_concept_id is not None:
            metadata["unit_concept_id"] = unit_concept_id

        source_code_column = concept_id_field.replace(
            "_concept_id", "_source_value"
        )
        source_code = row.get(source_code_column)
        metadata["source_code"] = source_code or ""

        return [Event(start=start, code=code, value=value, **metadata)]


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
        _ConceptTableConverter(
            prefix="visit_detail",
            concept_id_field="piton_visit_detail_concept_id",
        ),
    ]

    return converters
