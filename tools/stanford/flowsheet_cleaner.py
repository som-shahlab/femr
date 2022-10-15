from __future__ import annotations

import argparse
import csv
import functools
import io
import json
import multiprocessing.pool
import os
from typing import Dict, Mapping, Optional, Set, cast

import zstandard


def convert_row(row: Mapping[str, str]) -> Optional[Dict[str, str]]:
    if row["load_table_id"] in ("shc_ip_flwsht_meas", "lpch_ip_flwsht_meas"):
        data = json.loads(row["value_as_string"])

        if len(data["values"]) != 3:
            raise RuntimeError("More than 3 values?")

        value = ""
        units = ""
        name = ""

        for part in data["values"]:
            if part["source"] == "ip_flwsht_meas.meas_value":
                value = cast(str, part["value"]) or value
            elif part["source"] == "ip_flo_gp_data.units":
                units = cast(str, part["value"]) or units
            elif part["source"] == "ip_flo_gp_data.disp_name":
                name = cast(str, part["value"]) or name
            else:
                raise RuntimeError("Got invalid source" + part)

        new_row = dict(row)

        try:
            _ = float(cast(str, value))
            value_as_number = value
        except ValueError:
            value_as_number = ""

        new_row["observation_concept_id"] = "0"
        new_row["observation_source_value"] = name
        new_row["value_as_number"] = value_as_number
        new_row["value_as_string"] = value
        if units is not None and units != "":
            new_row["unit_concept_id"] = "0"
            new_row["unit_source_value"] = units
        else:
            new_row["unit_concept_id"] = ""
            new_row["unit_source_value"] = ""

        return new_row
    else:
        return None


def get_new_sources(root: str, child: str) -> Set[str]:
    new_concepts = set()

    source_path = os.path.join(root, "observation", child)
    with io.TextIOWrapper(
        zstandard.ZstdDecompressor().stream_reader(open(source_path, "rb"))
    ) as f:
        reader = csv.DictReader(f)
        for row in reader:
            new_row = convert_row(row)
            if new_row is None:
                continue
            if new_row["observation_concept_id"] == "0":
                new_concepts.add(new_row["observation_source_value"])
            if new_row["unit_concept_id"] == "0":
                new_concepts.add(new_row["unit_source_value"])

    return new_concepts


def correct_rows(
    root: str, target: str, mapping: Mapping[str, str], child: str
) -> None:
    source_path = os.path.join(root, "observation", child)
    out_path = os.path.join(target, "observation", child)
    with io.TextIOWrapper(
        zstandard.ZstdDecompressor().stream_reader(open(source_path, "rb"))
    ) as f:
        with io.TextIOWrapper(
            zstandard.ZstdCompressor(1).stream_writer(open(out_path, "wb"))
        ) as o:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            writer = csv.DictWriter(o, reader.fieldnames)
            writer.writeheader()
            for row in reader:
                new_row = convert_row(row)
                if new_row is None:
                    writer.writerow(row)
                else:
                    new_row["observation_concept_id"] = mapping[
                        new_row["observation_source_value"]
                    ]

                    if new_row["unit_concept_id"] != "":
                        new_row["unit_concept_id"] = mapping[
                            new_row["unit_source_value"]
                        ]
                    writer.writerow(new_row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean Stanford flowsheet data"
    )
    parser.add_argument("source", type=str, help="The source OMOP folder")
    parser.add_argument(
        "target", type=str, help="The location to create the result OMOP folder"
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        help="The number of threads to use",
        default=1,
    )

    args = parser.parse_args()

    os.mkdir(args.target)

    new_concepts = set()
    with multiprocessing.pool.Pool(args.num_threads) as pool:
        for directory in os.listdir(args.source):
            os.mkdir(os.path.join(args.target, directory))

            if directory != "observation":
                for file in os.listdir(os.path.join(args.source, directory)):
                    os.link(
                        os.path.join(args.source, directory, file),
                        os.path.join(args.target, directory, file),
                    )

        for child_concepts in pool.imap_unordered(
            functools.partial(get_new_sources, args.source),
            os.listdir(os.path.join(args.source, "observation")),
        ):
            new_concepts |= child_concepts

        highest_concept_id = 0

        destination_path = os.path.join(
            args.target, "concept", "flowsheet.csv.gz"
        )
        with io.TextIOWrapper(
            zstandard.ZstdCompressor(1).stream_writer(
                open(destination_path, "wb")
            )
        ) as o:
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
        for _ in pool.imap_unordered(
            functools.partial(
                correct_rows, args.source, args.target, new_concept_map
            ),
            os.listdir(os.path.join(args.source, "observation")),
        ):
            pass
