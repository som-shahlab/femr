"""An ETL script for doing an end to end transform of MIMIC-III-OMOP data into a PatientDatabase."""

import argparse
import datetime
import functools
import json
import logging
import os
import resource
from typing import Callable, Dict, Optional, Sequence
import shutil

from piton import Event, Patient
from piton.datasets import EventCollection, PatientCollection
from piton.extractors.csv import run_csv_extractors
from piton.extractors.omop import get_omop_csv_extractors
from piton.transforms import delta_encode, remove_nones


def _is_visit_event(e: Event) -> bool:
    return e.omop_table == "visit_occurrence"


def etl_mimic_omop_program() -> None:
    """Extract data from a MIMIC-III-OMOP source to create a piton PatientDatabase."""
    parser = argparse.ArgumentParser(description="An extraction tool for MIMIC-III-OMOP")

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
        "--is_force_refresh",
        action="store_true",
        help="If TRUE, force refresh the entire extract from scratch (will delete old copies)",
        default=False,
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
    logs_location: str = os.path.join(args.target_location, "log")

    # Force refresh
    if args.is_force_refresh:
        shutil.rmtree(args.target_location, ignore_errors=True)
        shutil.rmtree(args.temp_location, ignore_errors=True)

    # Create target, temp folders
    os.makedirs(args.target_location, exist_ok=True)
    os.makedirs(args.temp_location, exist_ok=True)

    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(logs_location)
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
                [],
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
            ).close()
        else:
            rootLogger.info("Already converted to extract, skipping")

    except Exception as e:
        rootLogger.critical(e, exc_info=True)
        raise e
