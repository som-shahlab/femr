"""An ETL script for doing an end to end transform of Stanford data into a PatientDatabase."""

import argparse
import functools
import json
import os
import shutil
from typing import Callable, Sequence

import datasets
import meds

from femr.transforms import delta_encode, remove_nones
from femr.transforms.stanford import (
    move_billing_codes,
    move_pre_birth,
    move_to_day_end,
    move_visit_start_to_first_event_start,
    switch_to_icd10cm,
    join_consecutive_day_visits,
)


def _is_visit_measurement(e: meds.Measurement) -> bool:
    return e["metadata"]["table"] == "visit"


def _get_stanford_transformations() -> Callable[[meds.Patient], meds.Patient]:
    """Get the list of current OMOP transformations."""
    # All of these transformations are information preserving
    transforms: Sequence[Callable[[meds.Patient], meds.Patient]] = [
        move_pre_birth,
        move_visit_start_to_first_event_start,
        move_to_day_end,
        switch_to_icd10cm,
        move_billing_codes,
        functools.partial(
            remove_nones,  # We have to keep visits in order to sync up visit_ids later in the process
            # If we ever remove or revisit visit_id, we would want to revisit this
            do_not_apply_to_filter=_is_visit_measurement,
        ),
        functools.partial(
            delta_encode,  # We have to keep visits in order to sync up visit_ids later in the process
            # If we ever remove or revisit visit_id, we would want to revisit this
            do_not_apply_to_filter=_is_visit_measurement,
        ),
        join_consecutive_day_visits,
    ]
    
    return lambda patient: functools.reduce(lambda r, f: f(r), transforms, patient)


def femr_stanford_omop_fixer_program() -> None:
    """Extract data from an Stanford STARR-OMOP v5 source to create a femr PatientDatabase."""
    parser = argparse.ArgumentParser(description="An extraction tool for STARR-OMOP v5 sources")

    parser.add_argument(
        "source_dataset",
        type=str,
        help="Path of the folder to source dataset",
    )

    parser.add_argument(
        "target_dataset",
        type=str,
        help="The place to store the extract",
    )

    parser.add_argument(
        "--num_proc",
        type=int,
        help="The number of threads to use",
        default=1,
    )

    args = parser.parse_args()

    if os.path.exists(args.target_dataset):
        shutil.rmtree(args.target_dataset)
    os.mkdir(args.target_dataset)

    dataset = datasets.Dataset.from_parquet(os.path.join(args.source_dataset, "data", "*"))
    
    fixed_patient = dataset.map(_get_stanford_transformations(), num_proc=args.num_proc, load_from_cache_file=False)

    os.mkdir(os.path.join(args.target_dataset, "data"))
    fixed_patient.to_parquet(os.path.join(args.target_dataset, "data", "data.parquet"))

    with open(os.path.join(args.source_dataset, "metadata.json")) as f:
        metadata = json.load(f)

    # Let's mark that we modified this dataset
    metadata["post_etl_name"] = "femr_stanford_omop_fixer"
    metadata["post_etl_version"] = "0.1"

    with open(os.path.join(args.target_dataset, "metadata.json"), "w") as f:
        json.dump(metadata, f)

if __name__ == '__main__':
    femr_stanford_omop_fixer_program()