"""
Preprocess MIMIC-III-OMOP to work with FEMR's OMOP pipeline.
"""

import argparse
import csv
import io
import os
from typing import List

import zstandard


def fix_drug_exposure_file_dates(path_to_file: str):
    # Change `visit_start_date` -> `visit_detail_start_date`

def fix_visit_detail_file_dates(path_to_file: str):
    # Change `visit_start_date` -> `visit_detail_start_date`
    print(path_to_file)
    fieldnames: List[str] = []
    with io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path_to_file, "rb"))) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = [ row for row in reader ]

    print(fieldnames, len(rows))
    # Fix `visit_start_date` -> `visit_detail_start_date`
    for i, field in enumerate(fieldnames):
        if field == "visit_start_date":
            fieldnames[i] = "visit_detail_start_date"
        elif field == "visit_end_date":
            fieldnames[i] = "visit_detail_end_date"
        if field == "visit_start_datetime":
            fieldnames[i] = "visit_detail_start_datetime"
        elif field == "visit_end_datetime":
            fieldnames[i] = "visit_detail_end_datetime"

    # Write rest of file unchanged
    exit()
    with io.TextIOWrapper(zstandard.ZstdCompressor(1).stream_writer(open(path_to_file, "wb"))) as o:
        writer = csv.DictWriter(o, fieldnames)
        writer.writeheader()
        for row in rows:
            print(row)
            print(row.values())
            exit()
            writer.writerow(row)

def fix_visit_detail(path_to_source_dir: str):
    path_to_visit_detail_dir: str = os.path.join(path_to_source_dir, "visit_detail")
    assert os.path.exists(path_to_visit_detail_dir), f"Could not find visit_detail directory at {path_to_visit_detail_dir}"
    for file in os.listdir(path_to_visit_detail_dir):
        if file.endswith(".csv.zst"):
            fix_visit_detail_file_dates(os.path.join(path_to_visit_detail_dir, file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply in-place fixes to MIMIC-III OMOP tables to work with FEMR's OMOP pipeline. This will overwrite the `source` folder's files.")
    parser.add_argument("source", type=str, help="The source OMOP folder")
    parser.add_argument(
        "--num_threads",
        type=int,
        help="The number of threads to use",
        default=1,
    )

    args = parser.parse_args()

    fix_visit_detail(args.source)
