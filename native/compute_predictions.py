import os

os.environ["JAX_NUMPY_RANK_PROMOTION"] = "raise"

import argparse

parser = argparse.ArgumentParser(prog="Compute predictions")
parser.add_argument("target_path", type=str)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--batch_info_path", type=str, required=True)
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--labeled_patients_path", type=str, required=True)

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

data = piton.datasets.PatientDatabase(args.data_path)

logging.info(
    "Loaded batches %s %s",
    loader.get_number_of_batches("train"),
    loader.get_number_of_batches("dev"),
)


def model_fn(config, batch):
    model = piton.models.transformer.EHRTransformer(config)(batch)
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


@functools.partial(jax.jit, static_argnames="config")
def compute_loss(params, rng, config, batch):
    loss = model.apply(params, rng, config, batch)[0]
    return loss


@functools.partial(jax.jit, static_argnames="config")
def compute_loss_and_logits(params, rng, config, batch):
    return model.apply(params, rng, config, batch)


def compute_total_loss(split, params, rng, config):
    num_to_get = min(500, loader.get_number_of_batches(split))

    total_loss = 0
    total_indices = 0
    for i in range(num_to_get):
        batch = loader.get_batch(split, i)
        total_loss += (
            compute_loss(
                piton.models.transformer.convert_params(params, jnp.float16),
                rng,
                config,
                batch,
            )
            * batch["num_indices"]
        )
        total_indices += batch["num_indices"]

    return total_loss / total_indices


num_train_batches = loader.get_number_of_batches("train")

logging.info(
    "Starting train loss %s",
    compute_total_loss("train", params, rng, config),
)
dev_loss = compute_total_loss("dev", params, rng, config)
logging.info("Starting dev loss %s", dev_loss)

import datetime
from typing import List, Tuple

import numpy as np

import piton.datasets

split_to_evaluate = "test"

database = piton.datasets.PatientDatabase(args.data_path)

num_expected_labels = 0

with open(args.labeled_patients_path, "rb") as f:
    labeled_patients = pickle.load(f)
    actual_patients_to_evaluate = {}
    for pid, values in labeled_patients.items():
        if database.compute_split(97, pid) >= 85:
            actual_values = [v for v in values if v.time != v.value.event_time]
            if len(actual_values) == 0:
                continue
            actual_patients_to_evaluate[pid] = actual_values
            num_expected_labels += len(actual_values)

print(num_expected_labels)

predictions: Dict[Tuple[int, datetime.datetime], Tuple[int, float, float]] = {}

total_loss = 0
num_dev = loader.get_number_of_batches(split_to_evaluate)
print("Total num_dev", num_dev)
num_skipped = 0
# num_dev = 100
for dev_index in range(num_dev):
    if dev_index % ((num_dev + 19) // 20) == 0:
        print(dev_index, num_dev, len(predictions), num_skipped)
    raw_batch = loader.get_batch(split_to_evaluate, dev_index)
    batch = jax.tree_map(lambda a: jnp.array(a), raw_batch)

    mask = (
        batch["transformer"]["label_indices"]
        != batch["transformer"]["ages"].shape[0]
    )
    loss, logits = compute_loss_and_logits(
        piton.models.transformer.convert_params(params, jnp.float16),
        rng,
        config,
        batch,
    )

    if False:
        for index, age, logit, label in zip(
            batch["transformer"]["label_indices"],
            raw_batch["task"]["label_ages"],
            logits,
            batch["task"]["labels"],
        ):
            p_index = index // batch["transformer"]["length"]
            p_offset = index % batch["transformer"]["length"]

            if p_index >= batch["num_patients"]:
                continue

            pid = int(batch["patient_ids"][p_index])

            birth_date = datetime.datetime.combine(
                database.get_patient_birth_date(pid), datetime.time.min
            )

            prediction_date = birth_date + datetime.timedelta(days=float(age))

            k = (pid, prediction_date)
            v = (int(p_offset), float(logit), float(label))

            if k not in predictions:
                predictions[k] = v
            else:
                num_skipped += 1
                predictions[k] = max(predictions[k], v)
    else:
        for index, age, logit, event_time, is_censor in zip(
            batch["transformer"]["label_indices"],
            raw_batch["task"]["label_ages"],
            logits,
            batch["task"]["event_times"],
            batch["task"]["is_censor"],
        ):

            p_index = index // batch["transformer"]["length"]
            p_offset = index % batch["transformer"]["length"]

            if p_index >= batch["num_patients"]:
                continue

            # print(index, age, logit, event_time, is_censor)

            pid = int(batch["patient_ids"][p_index])

            birth_date = datetime.datetime.combine(
                database.get_patient_birth_date(pid), datetime.time.min
            )

            prediction_date = birth_date + datetime.timedelta(minutes=int(age))

            k = (pid, prediction_date)
            v = (
                int(p_offset),
                np.array(logit),
                float(event_time),
                float(is_censor),
            )

            target_labels = actual_patients_to_evaluate[pid]
            target_labels = [
                label
                for label in target_labels
                if label.time == prediction_date
            ]
            assert len(target_labels) == 1
            target_label = target_labels[0]
            assert is_censor == target_label.value.is_censored
            assert event_time == (
                target_label.value.event_time - target_label.time
            ) / datetime.timedelta(minutes=1)

            if k not in predictions:
                predictions[k] = v
            else:
                num_skipped += 1
                predictions[k] = max(predictions[k], v)


found_patients = {k[0] for k in predictions}

for k in actual_patients_to_evaluate:
    if k not in found_patients:
        print("Missing!", k)

print(len(predictions), num_expected_labels)

assert len(predictions) == num_expected_labels

pickle.dump(predictions, open(args.target_path, "wb"))

if False:
    labels = []
    logits = []

    for a, b in predictions.items():
        (_, logit, label) = b
        labels.append(label)
        logits.append(logit)

    import sklearn.metrics

    print(sklearn.metrics.roc_auc_score(labels, logits))
    print(sklearn.metrics.average_precision_score(labels, logits))

logits = []
is_censor = []
event_time = []

for a, b in predictions.items():
    (_, l, t, c) = b
    logits.append(l)
    is_censor.append(c)
    event_time.append(t)

logits = np.stack(logits, axis=0)
is_censor = np.array(is_censor, dtype=bool)
event_time = np.array(event_time)

print(logits.shape, is_censor.shape, event_time.shape)

time_bins = np.array(config["task"]["time_bins"]) * 60 * 24

import piton.extension.metrics

actual = piton.extension.metrics.compute_c_statistic(
    event_time, is_censor, time_bins, logits
)

print(actual)
