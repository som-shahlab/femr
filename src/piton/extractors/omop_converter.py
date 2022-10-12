import tempfile
import argparse
from .. import Event, Patient
from ..datasets import (
    convert_patient_collection_to_patient_database,
    convert_event_collection_to_patient_collection,
)
from .csv_converter import CSVConverter, run_csv_converters

from typing import Sequence, Mapping, Optional
import os

import datetime


class DemographicsConverter(CSVConverter):
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

            if row["month_of_birth"]:
                month = int(row["month_of_birth"])

            if row["day_of_birth"]:
                day = int(row["day_of_birth"])

            birth = datetime.datetime(year=year, month=month, day=day)

        return [Event(start=birth, code="birth")] + [
            Event(start=birth, code=row[target])
            for target in [
                "gender_source_concept_id",
                "ethnicity_source_concept_id",
                "race_source_concept_id",
            ]
        ]


class StandardConceptTableConverter(CSVConverter):
    def __init__(self, prefix: str, date_field: str, concept_id_field: str):
        super().__init__()
        self.prefix = prefix
        self.date_field = date_field
        self.concept_id_field = concept_id_field

    def get_patient_id_field(self) -> str:
        return "person_id"

    def get_file_prefix(self) -> str:
        return self.prefix

    def get_events(self, row: Mapping[str, str]) -> Sequence[Event]:
        return [
            Event(
                start=datetime.datetime.fromisoformat(row[self.date_field]),
                code=row[self.concept_id_field],
            )
        ]


def get_omop_csv_converters() -> Sequence[CSVConverter]:
    converters = [
        DemographicsConverter(),
        StandardConceptTableConverter(
            "drug_exposure",
            "drug_exposure_start_date",
            "drug_source_concept_id",
        ),
        StandardConceptTableConverter(
            "condition_occurrence",
            "condition_start_date",
            "condition_source_concept_id",
        ),
        StandardConceptTableConverter(
            "death", "death_date", "death_type_concept_id"
        ),
        StandardConceptTableConverter(
            "procedure_occurrence",
            "procedure_date",
            "procedure_source_concept_id",
        ),
        StandardConceptTableConverter(
            "device_exposure",
            "device_exposure_start_date",
            "device_source_concept_id",
        ),
    ]

    return converters


class OmopTransformer:
    def __call__(self, patient: Patient) -> Optional[Patient]:
        new_events = []

        birth_date = None
        for event in patient.events:
            if event.code == "birth":
                birth_date = event.start

        if birth_date is None:
            return None

        for event in patient.events:
            if event.start > birth_date + datetime.timedelta(days=100 * 365):
                print("WAT", event)
                continue

            new_events.append(event)

        if len(new_events) <= 2:
            return None

        patient.events = new_events
        return patient


def extract_omop_program() -> None:
    parser = argparse.ArgumentParser(
        description="An extraction tool for OMOP v5 sources"
    )

    parser.add_argument(
        "omop_source",
        type=str,
        help="Path of the folder to the omop source",
    )

    parser.add_argument(
        "target_location",
        type=str,
        help="The place to store the extract",
    )

    parser.set_defaults(use_quotes=True)

    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as temp_dir:
        event_dir = os.path.join(temp_dir, "events")
        raw_patients_dir = os.path.join(temp_dir, "patients_raw")
        cleaned_patients_dir = os.path.join(temp_dir, "patients_cleaned")

        print("Converting to events")
        event_collection = run_csv_converters(
            args.omop_source, event_dir, 1, get_omop_csv_converters()
        )

        print("Converting to patients")
        patient_collection = convert_event_collection_to_patient_collection(
            event_collection, raw_patients_dir, 1
        )

        transformer = OmopTransformer()

        print("Transforming entries")
        patient_collection = patient_collection.transform(
            cleaned_patients_dir, transformer
        )

        print("Converting to extract")
        convert_patient_collection_to_patient_database(
            patient_collection, args.target_location
        ).close()
