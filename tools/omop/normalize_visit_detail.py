"""
A handy tool for adding a visit_detail_concept_id to the visit_detail table.

This makes visit_detail a "standard concept" table, which means it can be processed like many other OMOP tables.
"""

import argparse
import csv
import functools
import io
import multiprocessing.pool
import os
from typing import Dict, Mapping, Optional, Tuple

import zstandard


def get_care_site_concepts(root: str, child: str) -> Mapping[str, Tuple[str, Optional[str]]]:
    """Pull out the new care_site concept_ids that we have to map."""
    new_concepts: Dict[str, Tuple[str, Optional[str]]] = {}

    source_path = os.path.join(root, "care_site", child)
    with io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(source_path, "rb"))) as f:
        reader = csv.DictReader(f)
        for row in reader:
            care_site_id = row["care_site_id"]
            care_site_name = row["care_site_name"]
            parent_concept_id: Optional[str] = row["place_of_service_concept_id"]

            if care_site_id == "":
                continue

            if parent_concept_id == "0":
                parent_concept_id = None

            new_concepts[care_site_id] = (care_site_name, parent_concept_id)

    return new_concepts


def convert_row(row: Mapping[str, str], care_site_concepts: Mapping[str, str]) -> Mapping[str, str]:
    result = dict(**row)
    if row["care_site_id"] == "":
        result["femr_visit_detail_concept_id"] = "0"
    else:
        result["femr_visit_detail_concept_id"] = care_site_concepts[row["care_site_id"]]
    return result


def correct_rows(root: str, target: str, care_site_concepts: Mapping[str, str], child: str) -> None:
    """Using a concept_id map, fix incorrect mappings."""
    source_path = os.path.join(root, "visit_detail", child)
    out_path = os.path.join(target, "visit_detail", child)
    with io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(source_path, "rb"))) as f:
        with io.TextIOWrapper(zstandard.ZstdCompressor(1).stream_writer(open(out_path, "wb"))) as o:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            writer = csv.DictWriter(o, list(reader.fieldnames) + ["femr_visit_detail_concept_id"])
            writer.writeheader()
            for row in reader:
                new_row = convert_row(row, care_site_concepts)
                writer.writerow(new_row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add visit_detail_concept_id to visit_detail")
    parser.add_argument("source", type=str, help="The source OMOP folder")
    parser.add_argument("target", type=str, help="The location to create the result OMOP folder")
    parser.add_argument(
        "--num_threads",
        type=int,
        help="The number of threads to use",
        default=1,
    )

    args = parser.parse_args()

    os.mkdir(args.target)

    new_concepts: Dict[str, Tuple[str, Optional[str]]] = {}
    with multiprocessing.pool.Pool(args.num_threads) as pool:
        for directory in os.listdir(args.source):
            os.mkdir(os.path.join(args.target, directory))

            if directory != "visit_detail":
                for file in os.listdir(os.path.join(args.source, directory)):
                    os.link(
                        os.path.join(args.source, directory, file),
                        os.path.join(args.target, directory, file),
                    )

        for child_concepts in pool.imap_unordered(
            functools.partial(get_care_site_concepts, args.source),
            [os.path.join(args.source, "care_site", f) for f in os.listdir(os.path.join(args.source, "care_site"))],
        ):
            new_concepts |= child_concepts

        destination_path = os.path.join(args.target, "concept", "visit_detail.csv.zst")

        destination_rel_path = os.path.join(args.target, "concept_relationship", "visit_detail.csv.zst")
        with io.TextIOWrapper(zstandard.ZstdCompressor(1).stream_writer(open(destination_path, "wb"))) as o:
            with io.TextIOWrapper(zstandard.ZstdCompressor(1).stream_writer(open(destination_rel_path, "wb"))) as o2:
                writer = csv.DictWriter(
                    o,
                    fieldnames=[
                        "concept_id",
                        "concept_name",
                        "domain_id",
                        "vocabulary_id",
                        "concept_class_id",
                        "standard_concept",
                        "concept_code",
                        "valid_start_DATE",
                        "valid_end_DATE",
                        "invalid_reason",
                        "load_table_id",
                        "load_row_id",
                    ],
                )
                writer.writeheader()

                rel_writer = csv.DictWriter(
                    o2,
                    fieldnames=[
                        "concept_id_1",
                        "concept_id_2",
                        "relationship_id",
                        "valid_start_DATE",
                        "valid_end_DATE",
                        "invalid_reason",
                        "load_table_id",
                        "load_row_id",
                    ],
                )
                rel_writer.writeheader()

                next_concept_id = 3100000000
                new_concept_map = {}
                for i, (
                    care_site_id,
                    (
                        care_site_name,
                        parent_concept_id,
                    ),
                ) in enumerate(sorted(new_concepts.items())):
                    concept_id = str(i + next_concept_id)
                    new_concept_map[care_site_id] = concept_id
                    writer.writerow(
                        {
                            "concept_id": concept_id,
                            "concept_name": care_site_name,
                            "domain_id": "Care Site",
                            "vocabulary_id": "CARE_SITE",
                            "concept_class_id": "Care Site",
                            "standard_concept": "",
                            "concept_code": care_site_id,
                            "valid_start_DATE": "1970-01-01",
                            "valid_end_DATE": "2099-12-31",
                            "invalid_reason": "",
                            "load_table_id": "custom_mapping",
                            "load_row_id": "",
                        }
                    )
                    if parent_concept_id is not None:
                        rel_writer.writerow(
                            {
                                "concept_id_1": concept_id,
                                "concept_id_2": parent_concept_id,
                                "relationship_id": "Is a",
                                "valid_start_DATE": "1970-01-01",
                                "valid_end_DATE": "2099-12-31",
                                "invalid_reason": "",
                                "load_table_id": "custom mapping",
                                "load_row_id": "",
                            }
                        )

        for _ in pool.imap_unordered(
            functools.partial(correct_rows, args.source, args.target, new_concept_map),
            os.listdir(os.path.join(args.source, "visit_detail")),
        ):
            pass
