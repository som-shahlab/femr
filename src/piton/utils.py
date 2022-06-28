from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

def set_up_logging(filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    logFormatter = logging.Formatter("%(asctime)s %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(filename, mode="w")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.INFO)

def inspect_patient_collection() -> None:
    parser = argparse.ArgumentParser(
        description="A tool for inspecting a piton patient_collection"
    )

    parser.add_argument(
        "extract_dir",
        type=str,
        help="Path of the folder to the ehr_ml extraction",
    )

    parser.add_argument(
        "patient_id", type=int, help="The patient id to inspect",
    )

    args = parser.parse_args()

    source_file = args.extract_dir
    timelines = TimelineReader(source_file)

    if args.patient_id is not None:
        patient_id = int(args.patient_id)
    else:
        patient_id = timelines.get_patient_ids()[0]

    patient = timelines.get_patient(patient_id)

    print(f"Patient: {patient.patient_id}")

    def value_to_str(value: Value) -> str:
        if value.type == ValueType.NONE:
            return ""
        elif value.type == ValueType.NUMERIC:
            return str(value.numeric_value)
        elif value.type == ValueType.TEXT:
            return value.text_value

    for i, event in enumerate(patient.events):
        print(f"--- Event {i}----")
        print(event.start_age)
        print(event.code + " " + value_to_str(event.value))