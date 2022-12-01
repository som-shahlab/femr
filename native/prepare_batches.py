import os

os.environ["JAX_NUMPY_RANK_PROMOTION"] = "raise"

import argparse

parser = argparse.ArgumentParser(prog="Train")
parser.add_argument("directory", type=str)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--dictionary_path", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--clmbr_survival_dictionary_path", type=str)

args = parser.parse_args()

import pickle
import piton.extension.dataloader
import logging
import msgpack
import random

from typing import TypeVar

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

if args.task == "survival_clmbr":
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

loader_config = {
    "transformer": {
        "vocab_size": 1024 * 64,
        "dictionary": dictionary,
        "min_size": 5,
        "max_size": 14,
    },
    "task": task,
    "seed": 97,
    "splits": [["train", 0, 70], ["dev", 70, 85], ["test", 85, 100]],
}

random.seed(loader_config["seed"])

config_path = os.path.join(args.directory, "loader_config.msgpack")
with open(config_path, "wb") as out:
    msgpack.dump(loader_config, out)

rootLogger.info("Starting to load")

piton.extension.dataloader.create_batches(os.path.join(args.directory, "batch_info.msgpack"), args.data_path, config_path)

rootLogger.info("Loaded")