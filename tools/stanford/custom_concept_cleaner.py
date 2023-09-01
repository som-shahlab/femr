"""
A tool for cleaning up bad custom concepts in the ontology.

STARR-OMOP annoyingly has codes that are "duplicates" of each other.

See the __main__ section for how to run this program.

Note that this assumes that the source data is already in CSVs compressed by zstandard.
"""

import argparse
import collections
import csv
import functools
import io
import multiprocessing.pool
import os
import traceback
from typing import Dict, List

import zstandard


def get_concepts_to_fix(root: str, child: str) -> Dict[str, List[str]]:
    result = collections.defaultdict(list)
    try:
        source_path = os.path.join(root, "concept", child)
        with io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(source_path, "rb"))) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["concept_id"]) < 2000000000:
                    # Ignore regular concepts
                    continue

                if row["vocabulary_id"] == "Vocabulary":
                    continue

                if row["vocabulary_id"].endswith("_to_value"):
                    continue

                code = row["vocabulary_id"] + "/" + row["concept_code"]
                result[code].append(row["concept_id"])
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError("Failed " + root + " , " + child, e)

    return result


def get_all_maps(root: str, target_root: str, remap_map: Dict[str, str], child: str) -> None:
    try:
        source_path = os.path.join(root, "concept_relationship", child)
        target_path = os.path.join(target_root, "concept_relationship", child)
        with io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(source_path, "rb"))) as f:
            with io.TextIOWrapper(zstandard.ZstdCompressor().stream_writer(open(target_path, "wb"))) as of:
                reader = csv.DictReader(f)
                writer = None
                for row in reader:
                    if writer is None:
                        assert reader.fieldnames is not None
                        writer = csv.DictWriter(of, fieldnames=reader.fieldnames)
                        writer.writeheader()
                    row["concept_id_1"] = remap_map.get(row["concept_id_1"], row["concept_id_1"])
                    row["concept_id_2"] = remap_map.get(row["concept_id_2"], row["concept_id_2"])
                    writer.writerow(row)
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError("Failed " + root + " , " + child, e)


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

    with forkserver.Pool(args.num_threads) as pool:
        for directory in os.listdir(args.source):
            os.mkdir(os.path.join(args.target, directory))

            if directory != "concept_relationship":
                for file in os.listdir(os.path.join(args.source, directory)):
                    os.link(
                        os.path.join(args.source, directory, file),
                        os.path.join(args.target, directory, file),
                    )

        all_concepts = collections.defaultdict(list)
        for child_concepts in pool.imap_unordered(
            functools.partial(get_concepts_to_fix, args.source),
            os.listdir(os.path.join(args.source, "concept")),
        ):
            for k, v in child_concepts.items():
                all_concepts[k].extend(v)

        concept_map: Dict[str, str] = {}

        # Fix typo in STARR-OMOP
        concept_map["5803"] = "5083"

        for k, v in all_concepts.items():
            if len(v) > 1:
                for i in v:
                    if i != min(v):
                        concept_map[i] = min(v)

        for _ in pool.imap_unordered(
            functools.partial(get_all_maps, args.source, args.target, concept_map),
            os.listdir(os.path.join(args.source, "concept_relationship")),
        ):
            pass

        concept_remap = os.path.join(args.target, "concept_remap.csv.zst")
        with io.TextIOWrapper(zstandard.ZstdCompressor(1).stream_writer(open(concept_remap, "wb"))) as o_bad:
            for a, b in concept_map.items():
                o_bad.write(a + "," + b + "\n")
