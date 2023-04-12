"""
An ETL script for doing an end to end transform of SK OMOP into a PatientDatabase.
Note that the transforms are largely the same to that for STARR OMOP in order to
match the assumptions for timestamps. Exceptions ares:
1. SK-OMOP doesn't contain billing codes
2. at SK, categorical lab results are assigned a measurement value of 9999999.
   We are replacing these values with None.
3. missing birthdates(replaced with 1-1-1 in SK-OMOP) are replaced with 1900-1-1
"""

import argparse
import datetime
import functools
import json
import logging
import os
import resource
from typing import Callable, Dict, Optional, Sequence

from femr import Event, Patient
from femr.datasets import EventCollection, PatientCollection
from femr.extractors.csv import run_csv_extractors
from femr.extractors.omop import get_omop_csv_extractors
from femr.transforms import delta_encode, remove_nones
from femr.transforms.sickkids import replace_categorical_measurement_results, replace_default_birthdate
from femr.transforms.stanford import move_pre_birth, move_to_day_end, move_visit_start_to_first_event_start


def _is_visit_event(e: Event) -> bool:
    return e.omop_table == "visit_occurrence"


def _get_sk_transformations() -> Sequence[Callable[[Patient], Optional[Patient]]]:
    """Get the list of current OMOP transformations."""
    # All of these transformations are information preserving except
    # replace_categorical_measurement_results
    transforms: Sequence[Callable[[Patient], Optional[Patient]]] = [
        replace_default_birthdate,
        move_pre_birth,
        move_visit_start_to_first_event_start,
        move_to_day_end,
        replace_categorical_measurement_results,
        functools.partial(
            remove_nones,  # We have to keep visits in order to sync up visit_ids later in the process
            # If we ever remove or revisit visit_id, we would want to revisit this
            do_not_apply_to_filter=_is_visit_event,
        ),
        functools.partial(
            delta_encode,  # We have to keep visits in order to sync up visit_ids later in the process
            # If we ever remove or revisit visit_id, we would want to revisit this
            do_not_apply_to_filter=_is_visit_event,
        ),
    ]

    return transforms


def etl_sk_omop_program() -> None:
    """Extract data from an SK OMOP source to create a FEMR PatientDatabase."""
    parser = argparse.ArgumentParser(description="An extraction tool for SK-OMOP sources")

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

    args = parser.parse_args()

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

    args.target_location = os.path.abspath(args.target_location)
    args.temp_location = os.path.abspath(args.temp_location)

    if not os.path.exists(args.target_location):
        os.mkdir(args.target_location)
    if not os.path.exists(args.temp_location):
        os.mkdir(args.temp_location)

    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
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
        cleaned_patients_dir = os.path.join(args.temp_location, "patients_cleaned")

        if not os.path.exists(event_dir):
            rootLogger.info("Converting to events")
            stats_dict: Dict[str, Dict[str, int]] = {}
            event_collection = run_csv_extractors(
                args.omop_source,
                event_dir,
                get_omop_csv_extractors(),
                num_threads=args.num_threads,
                debug_folder=os.path.join(args.temp_location, "lost_csv_rows"),
                stats_dict=stats_dict,
                delimiter="\t",
            )
            rootLogger.info("Got converter statistics " + str(stats_dict))
            with open(os.path.join(args.target_location, "convert_stats.json"), "w") as f:
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
                _get_sk_transformations(),
                num_threads=args.num_threads,
                stats_dict=stats_dict,
            )
            rootLogger.info("Got transform statistics " + str(stats_dict))
            with open(os.path.join(args.target_location, "transform_stats.json"), "w") as f:
                json.dump(stats_dict, f)
        else:
            rootLogger.info("Already applied transformations, skipping")
            patient_collection = PatientCollection(cleaned_patients_dir)

        if not os.path.exists(os.path.join(args.target_location, "meta")):
            rootLogger.info("Converting to extract")

            print("Converting to extract", datetime.datetime.now())
            patient_collection.to_patient_database(
                args.target_location,
                args.omop_source,
                num_threads=args.num_threads,
                delimiter="\t",
            ).close()
        else:
            rootLogger.info("Already converted to extract, skipping")

    except Exception as e:
        rootLogger.critical(e, exc_info=True)
        raise e
