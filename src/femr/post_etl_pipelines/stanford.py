"""An ETL script for doing an end to end transform of Stanford data into a SubjectDatabase.

This is a thin wrapper around the configurable site-fix plugin system
(:mod:`femr.post_etl_pipelines.site_fixes`) using the ``"stanford"`` profile, which
reproduces the historical ``femr_stanford_omop_fixer`` behavior exactly. For other
sites, use the ``femr_omop_fixer`` entrypoint with ``--profile`` or ``--config``.
"""

import argparse
import json
import os

import meds_reader
import meds_reader.transform

from femr.post_etl_pipelines.site_fixes import SITE_PROFILES, SubjectTransform, build_pipeline

POST_ETL_NAME = "femr_stanford_omop_fixer"
POST_ETL_VERSION = "0.1"


def _get_stanford_transformations() -> SubjectTransform:
    """Get the list of current OMOP transformations."""
    # All of these transformations are information preserving
    return build_pipeline(SITE_PROFILES["stanford"])


def femr_stanford_omop_fixer_program() -> None:
    """Extract data from an Stanford STARR-OMOP v5 source to create a femr SubjectDatabase."""
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

    meds_reader.transform.transform_meds_dataset(
        args.source_dataset, args.target_dataset, _get_stanford_transformations(), num_threads=args.num_proc
    )

    with open(os.path.join(args.target_dataset, "metadata/dataset.json")) as f:
        metadata = json.load(f)

    # Let's mark that we modified this dataset
    metadata["post_etl_name"] = POST_ETL_NAME
    metadata["post_etl_version"] = POST_ETL_VERSION

    with open(os.path.join(args.target_dataset, "metadata/dataset.json"), "w") as f:
        json.dump(metadata, f)
