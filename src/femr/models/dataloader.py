from __future__ import annotations

import argparse
import datetime
import logging
import math
import os
import pickle
import queue
import random
import threading
import sys
from typing import Any, List, Optional, Tuple, TypeVar

import jax
import msgpack
import numpy as np

import femr.datasets
import femr.extension.dataloader

T = TypeVar("T")

BatchLoader = femr.extension.dataloader.BatchLoader


def _index_thread(
    index_queue: queue.Queue[Optional[Tuple[int, int]]],
    seed: int,
    num_epochs: int,
    num_batch_threads: int,
    split: str,
    data_path: str,
    batch_info_path: str,
    num_batches: int,
) -> None:
    """Generate indices in random order and add them to the queue."""
    rng = random.Random(seed)
    step = 0
    for _ in range(num_epochs):
        order: List[int] = list(range(num_batches))
        rng.shuffle(order)

        for i in order:
            index_queue.put((i, step))
            step += 1

    for _ in range(num_batch_threads):
        index_queue.put(None)


def _batch_thread(
    index_queue: queue.Queue[Optional[Tuple[int, int]]],
    batch_queue: queue.Queue[Optional[Tuple[Any, int]]],
    data_path: str,
    batch_info_path: str,
    token_dropout: float,
    split: str,
) -> None:
    """Load batches according to the indices in the index thread and add them to the batch queue."""
    thread_loader = BatchLoader(data_path, batch_info_path, token_dropout=token_dropout)
    while True:
        next_item = index_queue.get()
        if next_item is None:
            batch_queue.put(None)
            break

        batch_index, step = next_item

        batch = thread_loader.get_batch(split, batch_index)
        if batch["num_indices"] == 0:
            batch_queue.put((None, step))
        else:
            batch = jax.tree_map(lambda a: jax.device_put(a, device=jax.devices("cpu")[0]), batch)
            batch_queue.put((batch, step))

    batch_queue.put(None)


class Batches:
    def __init__(
        self,
        data_path: str,
        batch_info_path: str,
        token_dropout: float,
        seed: int,
        num_epochs: int,
        num_batch_threads: int,
        num_batches: int,
        split: str = "train",
    ):
        print("Working with seed", seed, file=sys.stderr)
        """Create a multithreaded batch loader for the given batch info."""
        index_queue: queue.Queue[Optional[int]] = queue.Queue(maxsize=300)
        _ = index_queue

        self.batch_queue: queue.Queue[Optional[Any]] = queue.Queue(maxsize=5)

        batch_queue = self.batch_queue
        _ = batch_queue

        local = locals()

        batcher_thread = threading.Thread(
            target=_index_thread,
            kwargs={
                k: local[k]
                for k in (
                    "index_queue",
                    "seed",
                    "num_batch_threads",
                    "num_epochs",
                    "data_path",
                    "batch_info_path",
                    "num_batches",
                    "split",
                )
            },
            name="batch_thread",
            daemon=True,
        )
        batcher_thread.start()

        batcher_threads = [
            threading.Thread(
                target=_batch_thread,
                kwargs={
                    k: local[k]
                    for k in (
                        "index_queue",
                        "batch_queue",
                        "data_path",
                        "batch_info_path",
                        "data_path",
                        "token_dropout",
                        "split",
                    )
                },
                name="batch_thread",
                daemon=True,
            )
            for _ in range(num_batch_threads)
        ]

        for t in batcher_threads:
            t.start()

        self.remaining_threads = num_batch_threads

    def get_next(self) -> Optional[Any]:
        """Get the next batch, or None if we are out of batches."""
        next_item = None

        while next_item is None:
            next_item = self.batch_queue.get()
            if next_item is not None:
                return next_item
            else:
                self.remaining_threads -= 1
                if self.remaining_threads == 0:
                    return None


def create_batches() -> None:
    parser = argparse.ArgumentParser(prog="Create batches")
    parser.add_argument("directory", type=str, help="The target directory to contain the batches")
    parser.add_argument("--data_path", type=str, required=True, help="The path to the source extract")
    parser.add_argument("--dictionary_path", type=str, required=True, help="The path to the dictionary")
    parser.add_argument("--task", type=str, help="Either clmbr, survival_clmbr, or labeled_patients")
    parser.add_argument("--transformer_vocab_size", type=int, default=1024 * 64, help="Size of the transformer vocab")
    parser.add_argument(
        "--clmbr_survival_dictionary_path", type=str, help="The survival clmbr dictionary if running that task"
    )
    parser.add_argument("--labeled_patients_path", type=str, help="The labeled patients")
    parser.add_argument(
        "--is_hierarchical", default=False, action="store_true", help="Whether to use hierarchical embeddings"
    )
    parser.add_argument("--seed", default=97, type=int, help="The random seed used for data splitting")
    parser.add_argument(
        "--val_start",
        default=70,
        type=int,
        help="The start of the validation split (and thus end of the train split)",
    )
    parser.add_argument(
        "--test_start",
        default=85,
        type=int,
        help="The start of the test split (and thus end of the val split)",
    )
    parser.add_argument(
        "--batch_size",
        default=int(1 << 14),
        type=int,
        help="The batch size (in events). Must be a power of two",
    )
    parser.add_argument("--note_embedding_data", default=None, type=str, help="Note embedding data when using notes")
    parser.add_argument(
        "--limit_to_patients_file",
        default=None,
        type=str,
        help="A file containing the only patient_ids to allow in batches",
    )
    parser.add_argument("--limit_before_date", default=None, type=str, help="Limit the batches to before a given date")

    args = parser.parse_args()

    os.mkdir(args.directory)

    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(os.path.join(args.directory, "log"))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(logging.INFO)
    rootLogger.info(f"Preparing batches with {args}")

    with open(args.dictionary_path, "rb") as f:
        dictionary = msgpack.load(f, use_list=False)

        if args.is_hierarchical:
            dict_len = len(dictionary["ontology_rollup"])
        else:
            dict_len = len(dictionary["regular"])

        assert (
            args.transformer_vocab_size <= dict_len
        ), f"Transformer vocab size ({args.transformer_vocab_size}) must be <= len(dictionary) ({dict_len})"

    data = femr.datasets.PatientDatabase(args.data_path)

    if args.labeled_patients_path is not None:
        with open(args.labeled_patients_path, "rb") as f:
            labeled_patients = pickle.load(f)
            result_labels = []
            offsets = []
            total_events = 0
            total = 0
            for pid, labels in labeled_patients.items():
                birth_date = datetime.datetime.combine(data.get_patient_birth_date(pid), datetime.time.min)

                for label in labels:
                    age = (label.time - birth_date) / datetime.timedelta(minutes=1)
                    if labeled_patients.labeler_type == "boolean":
                        value = label.value
                    elif labeled_patients.labeler_type == "survival":
                        event_offset = label.value.time_to_event / datetime.timedelta(minutes=1)

                        if event_offset == 0:
                            continue

                        offsets.append(event_offset)
                        total += 1
                        total_events += not label.value.is_censored

                        value = {
                            "event_time": event_offset + age,
                            "is_censored": label.value.is_censored,
                        }
                    result_labels.append((int(pid), age, value))

            task = {
                "type": "labeled_patients",
                "labeler_type": labeled_patients.labeler_type,
                "labels": result_labels,
            }
            if labeled_patients.labeler_type == "survival":
                mean_time = np.mean(offsets)
                frac_events = total_events / total
                task["lambda"] = frac_events / mean_time

                print(frac_events, mean_time, task["lambda"])

    elif args.task == "survival_clmbr":
        with open(args.clmbr_survival_dictionary_path, "rb") as f:
            surv_dict = msgpack.load(f, use_list=False)
        task = {
            "type": "survival_clmbr",
            "survival_dict": surv_dict,
        }
    elif args.task == "clmbr":
        task = {"type": "clmbr", "vocab_size": 1024 * 8}
    else:
        rootLogger.error("Invalid task?")
        exit()

    if args.limit_to_patients_file:
        with open(args.limit_to_patients_file) as f:
            ids_to_limit = [int(a) for a in f]
        task["patient_ids"] = ids_to_limit

    if args.limit_before_date:
        limit_date = datetime.date.fromisoformat(args.limit_before_date)
        task["limit_date"] = limit_date.isoformat()

    max_size = math.log2(args.batch_size)
    assert int(max_size) == max_size, "Batch size must be a power of two"
    max_size = int(max_size)

    loader_config: Any = {
        "transformer": {
            "vocab_size": args.transformer_vocab_size,
            "dictionary": dictionary,
            "min_size": 5,
            "max_size": max_size,
            "is_hierarchical": args.is_hierarchical,
        },
        "task": task,
        "seed": args.seed,
        "splits": [
            ["train", 0, args.val_start],
            ["dev", args.val_start, args.test_start],
            ["test", args.test_start, 100],
        ],
    }

    if args.note_embedding_data:
        loader_config["transformer"]["note_embedding_data"] = args.note_embedding_data

    random.seed(loader_config["seed"])

    config_path = os.path.join(args.directory, "loader_config.msgpack")
    with open(config_path, "wb") as out:
        msgpack.dump(loader_config, out)

    logging.info("Wrote config ...")

    rootLogger.info("Starting to load")

    target_path = os.path.join(args.directory, "batch_info.msgpack")

    femr.extension.dataloader.create_batches(
        target_path,
        args.data_path,
        config_path,
    )

    rootLogger.info("Loaded")

    loader = femr.extension.dataloader.BatchLoader(args.data_path, target_path)

    rootLogger.info("Number of train patients %s", loader.get_number_of_batches("train"))
