"""A class and program for converting OMOP v5 sources to piton."""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import logging
import os
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Set, Tuple

from piton import Event, Patient
from piton.datasets import EventCollection, PatientCollection
from piton.extractors.csv_converter import CSVConverter, run_csv_converters


class _DemographicsConverter(CSVConverter):
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
class _ConceptTableConverter(CSVConverter):
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


def _get_omop_csv_converters() -> Sequence[CSVConverter]:
    """Get the list of OMOP Converters."""
    converters = [
        _DemographicsConverter(),
        _ConceptTableConverter(
            "drug_exposure",
            concept_id_field="drug_concept_id",
        ),
        _ConceptTableConverter(
            "visit",
            file_suffix="occurrence",
        ),
        _ConceptTableConverter(
            "condition",
            file_suffix="occurrence",
        ),
        _ConceptTableConverter(
            "death", concept_id_field="death_type_concept_id"
        ),
        _ConceptTableConverter(
            "procedure",
            file_suffix="occurrence",
        ),
        _ConceptTableConverter(
            "device_exposure", concept_id_field="device_concept_id"
        ),
        _ConceptTableConverter(
            "measurement",
            string_value_field="value_source_value",
            numeric_value_field="value_as_number",
        ),
        _ConceptTableConverter(
            "observation",
            string_value_field="value_as_string",
            numeric_value_field="value_as_number",
        ),
        _ConceptTableConverter(
            "note",
            concept_id_field="note_class_concept_id",
            string_value_field="note_text",
        ),
    ]

    return converters


def _remove_pre_birth(patient: Patient) -> Optional[Patient]:
    """Remove all events before the birth of a patient."""
    birth_date = None
    for event in patient.events:
        # 4216316 is the SNOMED Concept ID for Birth
        if event.code == 4216316:
            birth_date = event.start

    if birth_date is None:
        return None

    new_events = []
    for event in patient.events:
        if event.start >= birth_date:
            new_events.append(event)

    return Patient(patient_id=patient.patient_id, events=new_events)


def _remove_short_patients(
    patient: Patient, min_num_dates: int = 3
) -> Optional[Patient]:
    """Remove patients with too few timepoints."""
    if (
        len(set(event.start.date() for event in patient.events))
        <= min_num_dates
    ):
        return None
    else:
        return patient


def _move_billing_codes(patient: Patient) -> Patient:
    """Move billing codes to the end of each visit.

    One issue with our OMOP extract is that billing codes are incorrectly assigned at the start of the visit.
    This class fixes that by assigning them to the end of the visit.
    """
    end_visits: Dict[int, datetime.datetime] = {}

    for event in patient.events:
        if event.event_type in ("lpch_pat_enc", "shc_pat_enc"):
            if event.end is not None:
                if event.visit_id is None:
                    raise RuntimeError(
                        f"Expected visit id for visit? {patient.patient_id} {event}"
                    )
                if end_visits.get(event.visit_id, event.end) != event.end:
                    raise RuntimeError(
                        f"Multiple end visits? {end_visits.get(event.visit_id)} {event}"
                    )
                end_visits[event.visit_id] = event.end

    new_events = []

    for event in patient.events:
        if event.event_type in ("lpch_pat_enc_dx", "shc_pat_enc_dx"):
            if event.visit_id is None:
                raise RuntimeError(
                    f"Expected visit id for code {patient.patient_id} {event}"
                )
            end_visit = end_visits.get(event.visit_id)
            if end_visit is None:
                raise RuntimeError(
                    f"Expected visit end for code {patient.patient_id} {event}"
                )
            new_events.append(dataclasses.replace(event, start=end_visit))
        else:
            new_events.append(event)

    new_events.sort(key=lambda a: (a.start, a.code))

    return Patient(patient_id=patient.patient_id, events=new_events)


def _remove_nones(patient: Patient) -> Patient:
    """Remove duplicate codes w/in same day if duplicate code has None value.

    There is no point having a NONE value in a timeline when we have an actual value within the same day.

    This removes those unnecessary NONE values.
    """
    has_value: Set[Tuple[int, datetime.date]] = set()

    for event in patient.events:
        if event.value is not None:
            has_value.add((event.code, event.start.date()))

    new_events = []
    for event in patient.events:
        if (
            event.value is None
            and (event.code, event.start.date()) in has_value
        ):
            continue
        new_events.append(event)

    return Patient(patient.patient_id, new_events)


def _move_to_day_end(patient: Patient) -> Patient:
    """We assume that everything coded at midnight should actually be moved to the end of the day."""
    new_events = []
    for event in patient.events:
        if (
            event.start.hour == 0
            and event.start.minute == 0
            and event.start.second == 0
        ):
            new_time = (
                event.start
                + datetime.timedelta(days=1)
                - datetime.timedelta(seconds=1)
            )
            new_events.append(dataclasses.replace(event, start=new_time))
        else:
            new_events.append(event)

    new_events.sort(key=lambda a: (a.start, a.code))

    return Patient(patient.patient_id, new_events)


def _delta_encode(patient: Patient) -> Patient:
    """Delta encodes the patient.

    The idea behind delta encoding is that if we get duplicate values within a short amount of time
    (1 day for this code), there is not much point retaining the duplicate.

    This code removes all *sequential* duplicates within the same day.
    """
    last_value: Dict[Tuple[int, datetime.date], Any] = {}

    new_events = []
    for event in patient.events:
        key = (event.code, event.start.date())
        if key in last_value and last_value[key] == event.value:
            continue
        last_value[key] = event.value
        new_events.append(event)

    return Patient(patient.patient_id, new_events)


def _get_omop_transformations() -> Sequence[
    Callable[[Patient], Optional[Patient]]
]:
    """Get the list of current OMOP transformations."""
    # All of these transformations are information preserving
    transforms: Sequence[Callable[[Patient], Optional[Patient]]] = [
        _remove_pre_birth,
        _move_to_day_end,
        _move_billing_codes,
        _remove_nones,
        _delta_encode,
        _remove_short_patients,
    ]

    return transforms


def extract_omop_program() -> None:
    """Extract data from an OMOP v5 source to create a piton PatientDatabase."""
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

    parser.add_argument(
        "temp_location",
        type=str,
        help="The place to store temporary files",
        default=None,
    )

    parser.add_argument(
        "--num_threads",
        type=int,
        help="The number of threads to use",
        default=1,
    )

    parser.set_defaults(use_quotes=True)

    args = parser.parse_args()

    args.target_location = os.path.abspath(args.target_location)
    args.temp_location = os.path.abspath(args.temp_location)

    if not os.path.exists(args.target_location):
        os.mkdir(args.target_location)
    if not os.path.exists(args.temp_location):
        os.mkdir(args.temp_location)

    logFormatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    )
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(os.path.join(args.target_location, "log"))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(logging.INFO)
    rootLogger.info(f"Extracting from OMOP with arguments {args}")

    try:
        event_dir = os.path.join(args.temp_location, "events")
        raw_patients_dir = os.path.join(args.temp_location, "patients_raw")
        cleaned_patients_dir = os.path.join(
            args.temp_location, "patients_cleaned"
        )

        if not os.path.exists(event_dir):
            rootLogger.info("Converting to events")
            stats_dict: Dict[str, Dict[str, int]] = {}
            event_collection = run_csv_converters(
                args.omop_source,
                event_dir,
                _get_omop_csv_converters(),
                num_threads=args.num_threads,
                debug_folder=os.path.join(args.temp_location, "lost_csv_rows"),
                stats_dict=stats_dict,
            )
            rootLogger.info("Got converter statistics " + str(stats_dict))
            with open(
                os.path.join(args.target_location, "convert_stats.json"), "w"
            ) as f:
                json.dump(stats_dict, f)
        else:
            rootLogger.info("Already converted to events, skipping")
            event_collection = EventCollection(event_dir)

        if not os.path.exists(raw_patients_dir):
            rootLogger.info("Converting to patients")
            patient_collection = event_collection.to_patient_collection(
                raw_patients_dir,
                num_threads=args.num_threads,
            )
        else:
            rootLogger.info("Already converted to patients, skipping")
            patient_collection = PatientCollection(raw_patients_dir)

        if not os.path.exists(cleaned_patients_dir):
            stats_dict = {}
            rootLogger.info("Appling transformations")
            patient_collection = patient_collection.transform(
                cleaned_patients_dir,
                _get_omop_transformations(),
                num_threads=args.num_threads,
                stats_dict=stats_dict,
            )
            rootLogger.info("Got transform statistics " + str(stats_dict))
            with open(
                os.path.join(args.target_location, "transform_stats.json"), "w"
            ) as f:
                json.dump(stats_dict, f)
        else:
            rootLogger.info("Already applied transformations, skipping")
            patient_collection = PatientCollection(cleaned_patients_dir)

        if not os.path.exists(os.path.join(args.target_location, "meta")):
            rootLogger.info("Converting to extract")

            print("Converting to extract", datetime.datetime.now())
            patient_collection.to_patient_database(
                args.target_location, args.omop_source
            ).close()
        else:
            rootLogger.info("Already converted to extract, skipping")

    except Exception as e:
        rootLogger.critical(e, exc_info=True)
        raise e
