import os

os.environ["JAX_NUMPY_RANK_PROMOTION"] = "raise"

import argparse

parser = argparse.ArgumentParser(prog="Compute predictions")
parser.add_argument("destination", type=str)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--batch_info_path", type=str, required=True)
parser.add_argument("--model_dir", type=str, required=True)

args = parser.parse_args()

import copy
import functools
import logging
import pickle
import queue
import random
import threading
from typing import Any, Dict, Mapping, Optional, TypeVar

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import msgpack
import optax

import piton.datasets
import piton.extension.dataloader
import piton.models.transformer

T = TypeVar("T")


with open(args.batch_info_path, "rb") as f:
    batch_info = msgpack.load(f, use_list=False)

with open(os.path.join(args.model_dir, "config.msgpack"), "rb") as f:
    config = msgpack.load(f, use_list=False)

random.seed(config["seed"])

config = hk.data_structures.to_immutable_dict(config)

loader = piton.extension.dataloader.BatchLoader(
    args.data_path, args.batch_info_path
)

logging.info(
    "Loaded batches %s %s",
    loader.get_number_of_batches("train"),
    loader.get_number_of_batches("dev"),
)


def model_fn(config, batch):
    model = piton.models.transformer.EHRTransformer(config)(batch, no_task=True)
    return model


dummy_batch = jax.tree_map(lambda a: jnp.array(a), loader.get_batch("train", 0))

logging.info(
    "Got dummy batch %s",
    str(jax.tree_map(lambda a: (a.shape, a.dtype, a.device()), dummy_batch)),
)

rng = jax.random.PRNGKey(42)
model = hk.transform(model_fn)

logging.info("Transformed the model function")

with open(os.path.join(args.model_dir, "best"), "rb") as f:
    params = pickle.load(f)

logging.info(
    "Done initing %s", str(jax.tree_map(lambda a: (a.shape, a.dtype), params))
)


def _cast_floating_to(tree: T, dtype: jnp.dtype) -> T:
    def conditional_cast(x):
        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
            x = x.astype(dtype)
        return x

    return jax.tree_util.tree_map(conditional_cast, tree)


@functools.partial(jax.jit, static_argnames="config")
def compute_repr(params, rng, config, batch):
    return model.apply(params, rng, config, batch)


import datetime
from typing import List, Tuple

import numpy as np

import piton.datasets

database = piton.datasets.PatientDatabase(args.data_path)

reprs = []
label_ages = []
label_pids = []

for split in ("train", "dev", "test"):
    # for dev_index in range(loader.get_number_of_batches(split)):
    for dev_index in range(loader.get_number_of_batches(split)):
        raw_batch = loader.get_batch(split, dev_index)
        batch = jax.tree_map(lambda a: jnp.array(a), raw_batch)

        repr, mask = compute_repr(
            _cast_floating_to(params, jnp.float16),
            rng,
            config,
            batch,
        )

        p_index = (
            batch["transformer"]["label_indices"]
            // batch["transformer"]["length"]
        )

        reprs.append(repr[: batch["num_indices"], :])
        label_ages.append(
            raw_batch["task"]["label_ages"][: batch["num_indices"]]
        )
        assert raw_batch["task"]["label_ages"].dtype == np.float64
        label_pids.append(batch["patient_ids"][p_index][: batch["num_indices"]])

reprs = np.array(jnp.concatenate(reprs, axis=0))
label_ages = np.array(np.concatenate(label_ages, axis=0))
assert label_ages.dtype == np.float64

label_pids = np.array(jnp.concatenate(label_pids, axis=0))

label_times = []

for pid, age in zip(label_pids, label_ages):
    birth_date = datetime.datetime.combine(
        database.get_patient_birth_date(pid), datetime.time.min
    )
    label_time = birth_date + datetime.timedelta(days=float(age))
    label_times.append(label_time)

result = {
    "data_path": args.data_path,
    "model": args.model_dir,
    "data_matrix": reprs,
    "patient_ids": label_pids,
    "labeling_time": np.array(label_times),
}

with open(args.destination, "wb") as f:
    pickle.dump(result, f)
