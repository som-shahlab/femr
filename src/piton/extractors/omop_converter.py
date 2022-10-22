from __future__ import annotations

import argparse
import collections
import contextlib
import datetime
import functools
import json
import logging
import numbers
import os
import random
import resource
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

from piton.datasets import EventCollection, PatientCollection

from .. import Event, Patient
from .csv_converter import CSVConverter, run_csv_converters


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


def try_numeric(val: str) -> float | memoryview | None:
    if val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return memoryview(val.encode("utf8"))


@dataclass
class ConceptTableConverter(CSVConverter):
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

    def get_date(
        self, date_field: str, row: Mapping[str, str]
    ) -> Optional[datetime.datetime]:
        for attempt in (date_field + "time", date_field):
            if attempt in row and row[attempt] != "":
                return datetime.datetime.fromisoformat(row[attempt])

        return None

    def get_events(self, row: Mapping[str, str]) -> Sequence[Event]:
        def helper(
            field_name: Optional[str], value: memoryview | float | None
        ) -> memoryview | float | None:
            if field_name is not None:
                val = try_numeric(row[field_name])
                if val is not None:
                    return val
            return value

        value = helper(self.string_value_field, None)
        value = helper(self.numeric_value_field, value)

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
            start = self.get_date(self.prefix + "_start_date", row)
            end = self.get_date(self.prefix + "_end_date", row)
        else:
            start = self.get_date(self.prefix + "_date", row)
            end = None

        if start is None:
            raise RuntimeError(
                "Could not find a date field for "
                + repr(self)
                + " "
                + repr(row)
            )

        if (self.prefix + "_visit_id") in row:
            visit_id = int(row[self.prefix + "_visit_id"])
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


def get_omop_csv_converters() -> Sequence[CSVConverter]:
    converters = [
        DemographicsConverter(),
        ConceptTableConverter(
            "drug_exposure",
            concept_id_field="drug_concept_id",
        ),
        ConceptTableConverter(
            "visit",
            file_suffix="occurrence",
        ),
        ConceptTableConverter(
            "condition",
            file_suffix="occurrence",
        ),
        ConceptTableConverter(
            "death", concept_id_field="death_type_concept_id"
        ),
        ConceptTableConverter(
            "procedure",
            file_suffix="occurrence",
        ),
        ConceptTableConverter(
            "device_exposure", concept_id_field="device_concept_id"
        ),
        ConceptTableConverter(
            "measurement",
            string_value_field="value_source_value",
            numeric_value_field="value_as_number",
        ),
        ConceptTableConverter(
            "observation",
            string_value_field="value_as_string",
            numeric_value_field="value_as_number",
        ),
        ConceptTableConverter(
            "note",
            concept_id_field="note_class_concept_id",
            string_value_field="note_text",
        ),
    ]

    return converters


def remove_pre_birth(patient: Patient) -> Optional[Patient]:
    birth_date = None
    for event in patient.events:
        if event.code == 4216316:
            birth_date = event.start

    if birth_date is None:
        return None

    new_events = []
    for event in patient.events:
        if event.start >= birth_date:
            new_events.append(event)

    return Patient(patient_id=patient.patient_id, events=new_events)


# There is no point holding onto patients with just a single time point
def remove_small_patients(patient: Patient) -> Optional[Patient]:
    if len(set(event.start for event in patient.events)) <= 2:
        return None
    else:
        return patient


# One big flaw in our setup is that billing codes are assigned to the wrong date
def move_billing_codes(patient: Patient) -> Patient:
    new_events = []

    for event in patient.events:
        match event.event_type:
            case _:
                new_events.append(event)

    new_events.sort(key=lambda a: a.start)

    return Patient(patient_id=patient.patient_id, events=new_events)


# Remove redunant rows
def remove_redundant_rows(patient: Patient) -> Patient:
    value_count: Dict[
        Tuple[int, datetime.datetime],
        Dict[Any, int],
    ] = collections.defaultdict(lambda: collections.defaultdict(int))

    for event in patient.events:
        value_count[(event.code, event.start)][event.value] += 1

    new_events = []

    for event in patient.events:
        counts = value_count[(event.code, event.start)]
        if event.value is None:
            # Remove if any of another type
            if any(v is not None for v in counts.keys()):
                continue

        if counts[event.value] >= 2:
            counts[event.value] -= 1
        else:
            new_events.append(event)

    return Patient(patient.patient_id, new_events)


# Remove unneeeded none values
def remove_nones(patient: Patient) -> Patient:
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


# Delta encode rows
def delta_encode(patient: Patient) -> Patient:
    last_value: Dict[Tuple[int, datetime.date], Any] = {}

    new_events = []
    for event in patient.events:
        key = (event.code, event.start.date())
        if key in last_value and last_value[key] == event.value:
            continue
        last_value[key] = event.value
        new_events.append(event)

    return Patient(patient.patient_id, new_events)


# Too many measurements overwhelms our analysis
# This does some dumb subsampling to try to keep things sane
def subsample_same_day(max_per_day: int, patient: Patient) -> Optional[Patient]:
    code_counts_per_day: Dict[
        Tuple[int, datetime.date], int
    ] = collections.defaultdict(int)
    for event in patient.events:
        code_counts_per_day[(event.code, event.start.date())] += 1

    new_events = []

    for event in patient.events:
        count = code_counts_per_day[(event.code, event.start.date())]
        prob = max_per_day / count
        if random.random() < prob:
            new_events.append(event)

    return Patient(patient.patient_id, new_events)


def mkdtemp_persistent(
    *args: Any,
    persistent: bool = True,
    force: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    if force is not None:

        @contextlib.contextmanager
        def foo() -> Any:
            yield force

        return foo()

    if persistent:

        @contextlib.contextmanager
        def normal_mkdtemp() -> Any:
            yield tempfile.mkdtemp(*args, **kwargs)

        return normal_mkdtemp()
    else:
        return tempfile.TemporaryDirectory(*args, **kwargs)


def extract_omop_program() -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

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
                get_omop_csv_converters(),
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

        # All of these transformations are information preserving
        transforms = [
            remove_pre_birth,
            remove_nones,
            delta_encode,
            move_billing_codes,
            remove_small_patients,
        ]

        # These transformations lose data and are disabled for now
        # Probably a good idea for the future though
        _disabled_transforms = [
            remove_redundant_rows,
            functools.partial(subsample_same_day, 10),
        ]
        _ = _disabled_transforms

        if not os.path.exists(cleaned_patients_dir):
            stats_dict = {}
            rootLogger.info("Appling transformations")
            patient_collection = patient_collection.transform(
                cleaned_patients_dir,
                transforms,
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
