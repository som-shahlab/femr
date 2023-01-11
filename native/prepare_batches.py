import os

os.environ["JAX_NUMPY_RANK_PROMOTION"] = "raise"

import argparse

parser = argparse.ArgumentParser(prog="Train")
parser.add_argument("directory", type=str)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--dictionary_path", type=str, required=True)
parser.add_argument("--task", type=str)
parser.add_argument("--clmbr_survival_dictionary_path", type=str)
parser.add_argument("--labeled_patients_path", type=str)
parser.add_argument("--is_hierarchical", default=False, action="store_true")
parser.add_argument("--subset_fraction", default=None, type=float)

args = parser.parse_args()

import dataclasses
import datetime
import json
import logging
import pickle
import random
from typing import TypeVar

import msgpack
import numpy as np

import piton.datasets
import piton.extension.dataloader

T = TypeVar("T")

os.mkdir(args.directory)

logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
)
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

data = piton.datasets.PatientDatabase(args.data_path)

if args.labeled_patients_path is not None:
    with open(args.labeled_patients_path, "rb") as f:
        labeled_patients = pickle.load(f)
        result_labels = []
        offsets = []
        total_events = 0
        total = 0
        for pid, labels in labeled_patients.items():
            birth_date = datetime.datetime.combine(
                data.get_patient_birth_date(pid), datetime.time.min
            )

            for label in labels:
                age = (label.time - birth_date) / datetime.timedelta(minutes=1)
                if labeled_patients.labeler_type == "boolean":
                    value = label.value
                elif labeled_patients.labeler_type == "survival":
                    event_age = (
                        label.value.event_time - birth_date
                    ) / datetime.timedelta(minutes=1)
                    event_offset = event_age - age

                    if event_offset == 0:
                        continue

                    offsets.append(event_offset)
                    total += 1
                    total_events += not label.value.is_censored

                    value = {
                        "event_time": event_age,
                        "is_censored": label.value.is_censored,
                    }
                result_labels.append((pid, age, value))

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

if args.subset_fraction is None:
    train_end = 70
else:
    train_end = int(args.subset_fraction * 70)

loader_config = {
    "transformer": {
        "vocab_size": 1024 * 64,
        "dictionary": dictionary,
        "min_size": 5,
        "max_size": 14,
        "is_hierarchical": args.is_hierarchical,
    },
    "task": task,
    "seed": 97,
    "splits": [
        ["train", 0, train_end],
        ["dev", 70, 85],
        ["test", 85, 100],
    ],
}

random.seed(loader_config["seed"])

config_path = os.path.join(args.directory, "loader_config.msgpack")
with open(config_path, "wb") as out:
    msgpack.dump(loader_config, out)

print("Wrote config ...")

rootLogger.info("Starting to load")

target_path = os.path.join(args.directory, "batch_info.msgpack")

piton.extension.dataloader.create_batches(
    target_path,
    args.data_path,
    config_path,
)

rootLogger.info("Loaded")

loader = piton.extension.dataloader.BatchLoader(args.data_path, target_path)

rootLogger.info(
    "Number of train patients %s", loader.get_number_of_batches("train")
)
