"""
A tool for cleaning up bad source concept ids in STARR-OMOP.

STARR-OMOP annoyingly has tons of broken source concept ids.

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
from typing import Any, Dict, List, Tuple

import zstandard


def get_concepts(root: str, child: str) -> Dict[str, List[Tuple[str, str]]]:
    result = collections.defaultdict(list)
    try:
        source_path = os.path.join(root, "concept", child)
        with io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(source_path, "rb"))) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["vocabulary_id"] not in ("ICD10CM", "ICD9CM"):
                    continue
                result[row["concept_code"]].append((row["vocabulary_id"], row["concept_id"]))
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError("Failed " + root + " , " + child, e)

    return result


def fix_rows(table: str, prefix: str, root: str, target_root: str, all_concepts: Dict[str, Any], child: str) -> None:
    try:
        source_path = os.path.join(root, table, child)
        target_path = os.path.join(target_root, table, child)
        with io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(source_path, "rb"))) as f:
            with io.TextIOWrapper(zstandard.ZstdCompressor().stream_writer(open(target_path, "wb"))) as of:
                reader = csv.DictReader(f)
                writer = None
                for row in reader:
                    if writer is None:
                        assert reader.fieldnames is not None
                        writer = csv.DictWriter(of, fieldnames=reader.fieldnames)
                        writer.writeheader()

                    mapped_values = all_concepts.get(row[f"{prefix}_source_value"], [])

                    if len(mapped_values) > 0:
                        if len(mapped_values) == 1:
                            # Easy, one mapped value
                            row[f"{prefix}_source_concept_id"] = mapped_values[0][1]
                        else:
                            assert len(mapped_values) == 2
                            # The v codes are rarely used, so we try to map to ICD9CM first
                            icd9 = [a for a in mapped_values if a[0] == "ICD9CM"]
                            assert len(icd9) == 1
                            row[f"{prefix}_source_concept_id"] = icd9[0][1]

                    if row["load_table_id"] in ("shc_medical_hx", "lpch_medical_hx"):
                        # These rows are really special and we explicitly don't want to use source_concept_id here
                        row[f"{prefix}_source_concept_id"] = "0"

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
            if os.path.isdir(os.path.join(args.source, directory)):
                os.mkdir(os.path.join(args.target, directory))

                if directory not in ("condition_occurrence", "observation"):
                    for file in os.listdir(os.path.join(args.source, directory)):
                        os.link(
                            os.path.join(args.source, directory, file),
                            os.path.join(args.target, directory, file),
                        )
            else:
                os.link(
                    os.path.join(args.source, directory),
                    os.path.join(args.target, directory),
                )

        all_concepts = collections.defaultdict(list)
        for child_concepts in pool.imap_unordered(
            functools.partial(get_concepts, args.source),
            os.listdir(os.path.join(args.source, "concept")),
        ):
            for k, v in child_concepts.items():
                all_concepts[k].extend(v)

        entries = [
            ("condition_occurrence", "condition"),
            ("observation", "observation"),
        ]

        for table, prefix in entries:
            for _ in pool.imap_unordered(
                functools.partial(fix_rows, table, prefix, args.source, args.target, all_concepts),
                os.listdir(os.path.join(args.source, table)),
            ):
                pass
