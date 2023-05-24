"""An ETL script for doing an end to end transform of our custom "simple" data format into a PatientDatabase."""

import argparse
import contextlib
import csv
import datetime
import io
import logging
import multiprocessing
import os
import resource
from typing import Dict, Iterable, List, Mapping, Set, Tuple

import zstandard

from femr.datasets import EventCollection, PatientCollection, RawEvent


def get_concept_ids_from_file(filename: str) -> Set[str]:
    resulting_concepts = set()

    with contextlib.ExitStack() as stack:
        f: Iterable[str]
        if filename.endswith(".csv.zst"):
            # Support Zstandard compressed CSVs
            f = stack.enter_context(io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(filename, "rb"))))
        else:
            # Support normal CSVs
            f = stack.enter_context(open(filename, "r"))

        reader = csv.DictReader(f)
        for row in reader:
            resulting_concepts.add(row["code"])

    return resulting_concepts


def convert_file_to_event_file(args: Tuple[str, Mapping[str, int], EventCollection]) -> None:
    filename, concept_map, collection = args

    with contextlib.ExitStack() as stack:
        f: Iterable[str]
        if filename.endswith(".csv.zst"):
            # Support Zstandard compressed CSVs
            f = stack.enter_context(io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(filename, "rb"))))
        else:
            # Support normal CSVs
            f = stack.enter_context(open(filename, "r"))

        writer = stack.enter_context(contextlib.closing(collection.create_writer()))

        reader = csv.DictReader(f)
        for row in reader:
            assert "/" in row["code"], f"Code must include vocabulary type with a / prefix, but {row} doesn't have one"
            event = RawEvent(
                start=datetime.datetime.fromisoformat(row["start"]),
                concept_id=concept_map[row["code"]],
                value=row["value"],
            )
            for k, v in row.items():
                if k not in ("start", "code", "value", "patient_id"):
                    if v == "":
                        v = None
                    elif k == "end":
                        v = datetime.datetime.fromisoformat(v)

                    setattr(event, k, v)

            writer.add_event(int(row["patient_id"]), event)


def etl_simple_femr_program() -> None:
    """Extract data from an generic OMOP source to create a femr PatientDatabase."""
    parser = argparse.ArgumentParser(description="An extraction tool for generic OMOP sources")

    parser.add_argument(
        "simple_source",
        type=str,
        help="Path of the folder to the simple femr input source",
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

    parser.add_argument(
        "--athena_download",
        type=str,
        help="An optional athena download to use for ontologies",
        default=None,
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
        patients_dir = os.path.join(args.temp_location, "patients")
        omop_dir = os.path.join(args.temp_location, "omop_dir")

        if not os.path.exists(event_dir):
            rootLogger.info("Converting to events")

            assert os.path.exists(args.simple_source), "Input file / directory is missing?"
            input_files: List[str]
            if os.path.isdir(args.simple_source):
                input_files = [os.path.join(args.simple_source, fname) for fname in os.listdir(args.simple_source)]
            else:
                input_files = [args.simple_source]

            concept_ids = set()
            with multiprocessing.Pool(args.num_threads) as pool:
                for f_concepts in pool.imap_unordered(get_concept_ids_from_file, input_files):
                    concept_ids |= f_concepts

            os.mkdir(omop_dir)
            with open(os.path.join(omop_dir, "concept.csv"), "w") as f:
                concept_id_map: Dict[str, int] = {}
                writer = csv.DictWriter(
                    f, ["concept_id", "concept_name", "vocabulary_id", "standard_concept", "concept_code"]
                )

                writer.writeheader()
                if args.athena_download:
                    with open(os.path.join(args.athena_download, "CONCEPT.csv"), "r") as f:
                        reader = csv.DictReader(f, delimiter="\t")
                        for row in reader:
                            del row["invalid_reason"]
                            del row["domain_id"]
                            del row["valid_end_date"]
                            del row["concept_class_id"]
                            del row["valid_start_date"]
                            writer.writerow(row)
                            concept_id_map[f'{row["vocabulary_id"]}/{row["concept_code"]}'] = int(row["concept_id"])

                for i, concept_id in enumerate(concept_ids):
                    if concept_id in concept_id_map:
                        continue
                    index = i + 11_000_000_000
                    prefix_index = concept_id.index("/")
                    vocab = concept_id[:prefix_index]
                    code = concept_id[prefix_index + 1 :]
                    writer.writerow(
                        {
                            "concept_id": index,
                            "concept_name": concept_id,
                            "vocabulary_id": vocab,
                            "standard_concept": "",
                            "concept_code": code,
                        }
                    )
                    concept_id_map[concept_id] = index

            if args.athena_download:
                with open(os.path.join(args.athena_download, "CONCEPT_RELATIONSHIP.csv"), "r") as f:
                    with open(os.path.join(omop_dir, "concept_relationship.csv"), "w") as wf:
                        reader = csv.DictReader(f, delimiter="\t")
                        assert reader.fieldnames is not None
                        writer = csv.DictWriter(wf, fieldnames=reader.fieldnames)
                        writer.writeheader()
                        for row in reader:
                            writer.writerow(row)
            else:
                os.mkdir(os.path.join(omop_dir, "concept_relationship"))

            event_collection = EventCollection(event_dir)
            with multiprocessing.Pool(args.num_threads) as pool:
                tasks = [(fname, concept_id_map, event_collection) for fname in input_files]
                for _ in pool.imap_unordered(convert_file_to_event_file, tasks):
                    pass
        else:
            rootLogger.info("Already converted to events, skipping")
            event_collection = EventCollection(event_dir)

        if not os.path.exists(patients_dir):
            rootLogger.info("Converting to patients")
            patient_collection = event_collection.to_patient_collection(
                patients_dir,
                num_threads=args.num_threads,
            )
        else:
            rootLogger.info("Already converted to patients, skipping")
            patient_collection = PatientCollection(patients_dir)

        if not os.path.exists(os.path.join(args.target_location, "meta")):
            rootLogger.info("Converting to extract")

            print("Converting to extract", datetime.datetime.now())
            patient_collection.to_patient_database(
                args.target_location,
                omop_dir,
                num_threads=args.num_threads,
            ).close()
        else:
            rootLogger.info("Already converted to extract, skipping")

    except Exception as e:
        rootLogger.critical(e, exc_info=True)
        raise e
