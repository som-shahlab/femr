"""
A tool for converting gziped files to zstandard.

See the __main__ section for how to run this program.

Note that this assumes that the source data is already in CSVs compressed by gzip
"""

import argparse
import collections
import csv
import functools
import gzip
import io
import multiprocessing.pool
import os
import shutil
import traceback
from typing import Dict, List

import zstandard


def recompress(source_target):
    source, target = source_target
    params = zstandard.ZstdCompressionParameters.from_level(-1)
    print(source)
    with gzip.open(source) as source_f:
        with zstandard.ZstdCompressor(compression_params=params).stream_writer(open(target, "wb")) as target_f:
            shutil.copyfileobj(source_f, target_f)


if __name__ == "__main__":
    forkserver = multiprocessing.get_context("forkserver")
    parser = argparse.ArgumentParser(description="Convert gzip to zstdandard")
    parser.add_argument("source", type=str, help="The source folder")
    parser.add_argument("target", type=str, help="The location to create the result folder")
    parser.add_argument(
        "--num_threads",
        type=int,
        help="The number of threads to use",
        default=1,
    )

    args = parser.parse_args()

    os.mkdir(args.target)

    paths = []

    with forkserver.Pool(args.num_threads) as pool:
        for directory in os.listdir(args.source):
            os.mkdir(os.path.join(args.target, directory))

            if not os.path.isdir(os.path.join(args.source, directory)):
                paths.append(
                    (os.path.join(args.source, directory), os.path.join(args.target, directory).replace(".gz", "zst"))
                )
            else:
                for file in os.listdir(os.path.join(args.source, directory)):
                    paths.append(
                        (
                            os.path.join(args.source, directory, file),
                            os.path.join(args.target, directory, file).replace(".gz", ".zst"),
                        )
                    )

        for _ in pool.imap_unordered(recompress, paths):
            pass
