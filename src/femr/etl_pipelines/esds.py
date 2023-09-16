"""An ETL script for doing an end to end transform of the Event Stream Data Standard into a PatientDatabase."""

import argparse
import contextlib
import csv
import datetime
import logging
import multiprocessing
import os
import resource
from typing import Any, Dict, List, Mapping, Set, Tuple

import event_stream_data_standard as ESDS
import pyarrow as pa
import pyarrow.parquet as pq

from femr import Patient
from femr.datasets import EventCollection, PatientCollection, PatientDatabase, RawEvent


def get_codes_from_file(filename: str) -> Set[str]:
    resulting_concepts = set()

    source_file = pq.ParquetFile(filename)

    for row_group in range(0, source_file.num_row_groups):
        table = source_file.read_row_group(row_group)

        for batch in table.to_batches():
            for patient in batch.to_pylist():
                for measure in patient["static_measurements"]:
                    resulting_concepts.add(measure["code"])

                for event in patient["events"]:
                    for measure in event["measurements"]:
                        resulting_concepts.add(measure["code"])

    return resulting_concepts


def convert_file_to_event_file(args: Tuple[str, Mapping[str, int], EventCollection]) -> None:
    filename, concept_map, collection = args

    source_file = pq.ParquetFile(filename)

    def convert_measure_to_event(current_time, measurement):
        if measurement["datetime_value"] is not None:
            value = (measurement["datetime_value"] - current_time) / datetime.timedelta(days=1)
        elif measurement["numeric_value"] is not None:
            value = measurement["numeric_value"]
        elif measurement["text_value"] is not None:
            value = measurement["text_value"]
        else:
            value = None

        return RawEvent(
            start=current_time,
            concept_id=concept_map[measure["code"]],
            value=value,
        )

    with contextlib.closing(collection.create_writer()) as writer:
        for row_group in range(0, source_file.num_row_groups):
            table = source_file.read_row_group(row_group)
            for batch in table.to_batches():
                for patient in batch.to_pylist():
                    patient_id = patient["subject_id"]
                    birth_date = None
                    for measure in patient["static_measurements"]:
                        if measure["code"] == "Birth/Birth":
                            birth_date = measure["datetime_value"]
                    assert birth_date is not None
                    for measure in patient["static_measurements"]:
                        writer.add_event(patient_id, convert_measure_to_event(birth_date, measure))

                    for event in patient["events"]:
                        for measure in event["measurements"]:
                            writer.add_event(patient_id, convert_measure_to_event(event["time"], measure))


def etl_esds_program() -> None:
    """Extract data from an generic OMOP source to create a femr PatientDatabase."""
    parser = argparse.ArgumentParser(description="An extraction tool for Event Stream Data Standard sources")

    parser.add_argument(
        "esds_source",
        type=str,
        help="Path of the folder that contains ESDS parquet files",
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

            assert os.path.exists(args.esds_source), "Input file / directory is missing?"

            if os.path.isdir(args.esds_source):
                input_files = [os.path.join(args.esds_source, a) for a in os.listdir(args.esds_source)]
            else:
                input_files = [args.esds_source]

            codes = set()
            with multiprocessing.Pool(args.num_threads) as pool:
                for f_codes in pool.imap_unordered(get_codes_from_file, input_files):
                    codes |= f_codes

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

                for i, code in enumerate(codes):
                    if code in concept_id_map:
                        continue
                    index = i + 11_000_000_000
                    prefix_index = code.index("/")
                    vocab = code[:prefix_index]
                    concept_code = code[prefix_index + 1 :]
                    writer.writerow(
                        {
                            "concept_id": index,
                            "concept_name": code,
                            "vocabulary_id": vocab,
                            "standard_concept": "",
                            "concept_code": concept_code,
                        }
                    )
                    concept_id_map[code] = index

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


def write_shard(args: Tuple[str, str, int, int]) -> None:
    target_location, extract_path, shard_index, num_shards = args
    extract = PatientDatabase(extract_path)
    patient_ids = list(extract)

    patient_ids_per_shard = (len(patient_ids) + num_shards - 1) // num_shards

    start_index = patient_ids_per_shard * shard_index
    end_index = patient_ids_per_shard * (shard_index + 1)

    with pq.ParquetWriter(os.path.join(target_location, f"{shard_index}.parquet"), ESDS.patient) as writer:
        for i in range(start_index, end_index, 1_000):
            print("Processing patient", i, "out of", len(patient_ids))
            patient_objs = []
            for patient_id in patient_ids[i : min(end_index, i + 1_000)]:
                patient: Patient = extract[patient_id]

                birth_date = patient.events[0].start

                events: List[Dict[str, Any]] = []
                patient_obj = {
                    "subject_id": int(patient_id),
                    "static_measurements": [{"code": "Birth/Birth", "datetime_value": birth_date}],
                    "events": events,
                }

                for event in patient.events:
                    if len(events) == 0 or events[-1]["time"] != event.start:
                        events.append({"time": event.start, "measurements": []})
                    data: Dict[str, Any] = {"code": event.code}

                    if event.code in ("Birth/Birth", "SNOMED/184099003"):
                        continue

                    if event.value is None:
                        pass
                    elif isinstance(event.value, str):
                        data["text_value"] = event.value
                    else:
                        # Must be numeric
                        data["numeric_value"] = event.value
                    events[-1]["measurements"].append(data)

                patient_objs.append(patient_obj)

            table = pa.Table.from_pylist(patient_objs, schema=ESDS.patient)
            writer.write_table(table)


def export_esds_program() -> None:
    """Convert a patient database into an ESDS dataset."""
    parser = argparse.ArgumentParser(description="An extraction tool for Event Stream Data Standard sources")

    parser.add_argument(
        "extract_location",
        type=str,
        help="Path of the extract location to convert",
    )

    parser.add_argument(
        "target_location",
        type=str,
        help="The place to store the ESDS dataset",
    )

    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="The number of shards to create",
    )

    args = parser.parse_args()

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

    args.target_location = os.path.abspath(args.target_location)

    if not os.path.exists(args.target_location):
        os.mkdir(args.target_location)

    output_tasks = [(args.target_location, args.extract_location, i, args.num_shards) for i in range(args.num_shards)]

    with multiprocessing.Pool(args.num_shards) as pool:
        for _ in pool.imap_unordered(write_shard, output_tasks):
            pass
