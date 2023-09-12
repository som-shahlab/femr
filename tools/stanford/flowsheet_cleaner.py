"""
A tool for cleaning up incorrectly ETLed flowsheet values in STARR-OMOP.

See the __main__ section for how to run this program.

Note that this assumes that the source data is already in CSVs compressed by zstandard.

The fundamental idea of this program is to do a two pass conversion, first to OMOP rows (with incorrect concept_ids).
And then a second pass to fix up the concept_ids.
"""

import argparse
import csv
import functools
import io
import json
import multiprocessing.pool
import os
import traceback
from typing import Dict, Mapping, Optional, Set, Tuple, cast

import zstandard


def convert_row(row: Mapping[str, str]) -> Optional[Dict[str, str]]:
    """Convert an incorrect row into a correct one, splitting out the JSON into the correct OMOP columns."""
    if (
        row["load_table_id"] in ("shc_ip_flwsht_meas", "lpch_ip_flwsht_meas")
        and row["observation_source_concept_id"] == "2000006253"
    ):
        first_data = json.loads(row["value_as_string"])

        # This field is currently broken unformately
        # second_data = json.loads(row["observation_source_value"])

        values = first_data["values"]

        if len(values) != 3:
            raise RuntimeError(f"Wrong number of values? {values}")

        sheet = ""
        value = ""
        units = ""
        name = ""

        for part in values:
            if part["source"] == "ip_flt_data.display_name":
                sheet = cast(str, part["value"]) or sheet
            elif part["source"] == "ip_flwsht_meas.meas_value":
                value = cast(str, part["value"]) or value
            elif part["source"] == "ip_flo_gp_data.units":
                units = cast(str, part["value"]) or units
            elif part["source"] == "ip_flo_gp_data.disp_name":
                name = cast(str, part["value"]) or name
            else:
                raise RuntimeError(f"Got invalid source {part}")

        new_row = dict(row)

        try:
            _ = float(cast(str, value))
            value_as_number = value
        except ValueError:
            value_as_number = ""

        # Note that we set the concept ids to 0 as we don't know what they should be yet
        if sheet:
            name = sheet + "/" + name
            new_row["observation_parent_id"] = "0"
            new_row["observation_parent_value"] = sheet
        else:
            new_row["observation_parent_id"] = ""
            new_row["observation_parent_value"] = ""

        new_row["observation_source_value"] = name

        new_row["value_as_number"] = value_as_number
        new_row["value_as_string"] = value
        if units is not None and units != "":
            new_row["unit_source_value"] = units
        else:
            new_row["unit_source_value"] = ""

        return new_row
    else:
        return None


def get_concepts_to_add(root: str, child: str) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    """Pull out the new concept_ids that we have to map."""
    new_concepts = set()
    new_relationships = set()

    try:
        source_path = os.path.join(root, "observation", child)
        with io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(source_path, "rb"))) as f:
            reader = csv.DictReader(f)
            for row in reader:
                new_row = convert_row(row)
                if new_row is None:
                    continue

                new_concepts.add(new_row["observation_source_value"])
                new_concepts.add(new_row["observation_parent_value"])

                new_relationships.add(
                    (
                        new_row["observation_source_value"],
                        new_row["observation_parent_value"],
                    )
                )

                if new_row["unit_source_value"] != "":
                    new_concepts.add(new_row["unit_source_value"])
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError("Failed " + root + " , " + child, e)

    return new_concepts, new_relationships


def correct_rows(root: str, target: str, mapping: Mapping[str, str], child: str) -> None:
    """Using a concept_id map, fix incorrect mappings.

    Currently, this only fixes json encoded observations by decoding them into OMOP fields.
    Future versions of this code might fix other issues."""
    source_path = os.path.join(root, "observation", child)
    out_path = os.path.join(target, "observation", child)
    with io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(source_path, "rb"))) as f:
        with io.TextIOWrapper(zstandard.ZstdCompressor(1).stream_writer(open(out_path, "wb"))) as o:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            writer = csv.DictWriter(o, reader.fieldnames)
            writer.writeheader()
            for row in reader:
                new_row = convert_row(row)
                if new_row is None:
                    writer.writerow(row)
                else:
                    new_row["observation_source_concept_id"] = mapping[new_row["observation_source_value"]]

                    del new_row["observation_parent_id"]
                    del new_row["observation_parent_value"]

                    if new_row["unit_source_value"] != "":
                        new_row["unit_concept_id"] = mapping[new_row["unit_source_value"]]

                    writer.writerow(new_row)


if __name__ == "__main__":
    forkserver = multiprocessing.get_context("forkserver")
    parser = argparse.ArgumentParser(description="Clean Stanford flowsheet data")
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

    new_concepts = set()
    new_relationships = set()
    with forkserver.Pool(args.num_threads) as pool:
        for directory in os.listdir(args.source):
            os.mkdir(os.path.join(args.target, directory))

            if directory != "observation":
                for file in os.listdir(os.path.join(args.source, directory)):
                    os.link(
                        os.path.join(args.source, directory, file),
                        os.path.join(args.target, directory, file),
                    )

        for child_concepts, child_relationships in pool.imap_unordered(
            functools.partial(get_concepts_to_add, args.source),
            os.listdir(os.path.join(args.source, "observation")),
        ):
            new_concepts |= child_concepts
            new_relationships |= child_relationships

        highest_concept_id = 0

        destination_path = os.path.join(args.target, "concept", "flowsheet.csv.zst")
        destination_rel_path = os.path.join(args.target, "concept_relationship", "flowsheet.csv.zst")
        with io.TextIOWrapper(zstandard.ZstdCompressor(1).stream_writer(open(destination_path, "wb"))) as o:
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

            next_concept_id = 3000000000
            new_concept_map = {}
            for i, c in enumerate(list(new_concepts)):
                concept_id = str(i + next_concept_id)
                new_concept_map[c] = concept_id
                writer.writerow(
                    {
                        "concept_id": concept_id,
                        "concept_name": c,
                        "domain_id": "Observation",
                        "vocabulary_id": "STANFORD_OBS",
                        "concept_class_id": "Observation",
                        "standard_concept": "",
                        "concept_code": c,
                        "valid_start_DATE": "1970-01-01",
                        "valid_end_DATE": "2099-12-31",
                        "invalid_reason": "",
                        "load_table_id": "custom_mapping",
                        "load_row_id": "",
                    }
                )

        with io.TextIOWrapper(zstandard.ZstdCompressor(1).stream_writer(open(destination_rel_path, "wb"))) as o2:
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

            for child_concept, parent_concept in new_relationships:
                child_concept_id = new_concept_map[child_concept]
                parent_concept_id = new_concept_map[parent_concept]

                rel_writer.writerow(
                    {
                        "concept_id_1": child_concept_id,
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
            os.listdir(os.path.join(args.source, "observation")),
        ):
            pass
