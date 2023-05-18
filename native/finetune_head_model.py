import datetime
import os
import sys

os.environ["JAX_NUMPY_RANK_PROMOTION"] = "raise"
print("Start", datetime.datetime.now())

import argparse

parser = argparse.ArgumentParser(prog="Compute predictions")
parser.add_argument("target_path", type=str)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--batch_info_path", type=str, required=True)
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--labeled_patients_path", type=str, required=True)
parser.add_argument("--subsample_frac", type=int, default=None)
parser.add_argument("--probe", type=str, default=None)

args = parser.parse_args()

if os.path.exists(args.target_path):
    sys.exit()

import copy
import functools
import logging
import pickle
import queue
import random
import threading
from typing import Any, Dict, Mapping, Optional, TypeVar

import jax.scipy.optimize

logging.basicConfig()
logging.root.setLevel(logging.INFO)

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import msgpack
import optax

import femr.datasets
import femr.extension.dataloader
import femr.models.dataloader
import femr.models.transformer

T = TypeVar("T")


from typing import List, Tuple

import numpy as np

import femr.datasets

database = femr.datasets.PatientDatabase(args.data_path)

inverse_map = {pid: i for i, pid in enumerate(database.keys())}

num_expected_labels = 0

actual_patients_to_evaluate = {}

with open(args.labeled_patients_path, "rb") as f:
    labeled_patients = pickle.load(f)
    for pid, values in labeled_patients.items():
        actual_values = [v for v in values if v.value.time_to_event != datetime.timedelta(minutes=0)]
        if len(actual_values) == 0:
            continue
        actual_values = values
        actual_patients_to_evaluate[pid] = actual_values
        num_expected_labels += len(actual_values)

with open(os.path.join(args.model_dir, "best"), "rb") as f:
    params = pickle.load(f)

if "code_weight" not in params["EHRTransformer/~/SurvivalCLMBRTask"]:
    params["EHRTransformer/~/SurvivalCLMBRTask"]["code_weight"] = params["EHRTransformer/~/SurvivalCLMBRTask"][
        "code_weights"
    ][:, :511]
    params["EHRTransformer/~/SurvivalCLMBRTask"]["code_weight_bias"] = params["EHRTransformer/~/SurvivalCLMBRTask"][
        "code_weights"
    ][:, 511:]

params = femr.models.transformer.convert_params(params, dtype=jnp.float16)

with open(args.batch_info_path, "rb") as f:
    batch_info = msgpack.load(f, use_list=False)

with open(os.path.join(args.model_dir, "config.msgpack"), "rb") as f:
    config = msgpack.load(f, use_list=False)

config = hk.data_structures.to_immutable_dict(config)

with open(
    "/local-scratch/nigam/projects/ethanid/piton/native/surv_clmbr_batches_new/batch_info.msgpack",
    "rb",
) as f:
    old_batch_task = msgpack.load(f)["config"]["task"]

    time_bins = jnp.array(old_batch_task["survival_dict"]["time_bins"] + [float("inf")])
    print(time_bins)

random.seed(config["seed"])
rng = jax.random.PRNGKey(42)

if True:
    loader = femr.extension.dataloader.BatchLoader(args.data_path, args.batch_info_path)

    logging.info(
        "Loaded batches %s %s",
        loader.get_number_of_batches("train"),
        loader.get_number_of_batches("dev"),
    )

    def model_fn(config, batch):
        model = femr.models.transformer.EHRTransformer(config)(batch, no_task=True)
        return model

    dummy_batch = jax.tree_map(lambda a: jnp.array(a), loader.get_batch("train", 0))

    logging.info(
        "Got dummy batch %s",
        str(jax.tree_map(lambda a: (a.shape, a.dtype, a.device()), dummy_batch)),
    )

    model = hk.transform(model_fn)

    logging.info("Transformed the model function")

    logging.info(
        "Done initing %s",
        str(jax.tree_map(lambda a: (a.shape, a.dtype), params)),
    )

    print("Computing reps", datetime.datetime.now())

    @functools.partial(jax.jit, static_argnames="config")
    def compute_repr(params, rng, config, batch, time_bins):
        repr, mask = model.apply(params, rng, config, batch)

        final_layer = params["EHRTransformer/~/SurvivalCLMBRTask/~/linear"]
        # final_layer = params["EHRTransformer/~/SurvivalTask/~/linear"]
        binned_reprs = jnp.dot(repr, final_layer["w"].astype(jnp.float16))
        binned_reprs += jnp.broadcast_to(final_layer["b"].astype(jnp.float16), binned_reprs.shape)
        num_time_bins = 8
        # num_time_bins = config['task']['num_time_bins']
        binned_reprs = binned_reprs.reshape(repr.shape[0], num_time_bins, -1)

        offsets = jnp.ones(
            (repr.shape[0], num_time_bins, 1),
            dtype=binned_reprs.dtype,
        )

        total_reps = jnp.concatenate((binned_reprs, offsets), axis=-1)

        tiled_bins = jnp.expand_dims(time_bins, 0)

        tiled_times = jnp.expand_dims(batch["task"]["event_times"], -1)

        time_in_bin = jnp.clip(
            tiled_times - tiled_bins[:, :-1],
            0,
            tiled_bins[:, 1:] - tiled_bins[:, :-1],
        )

        log_time_in_bin = jnp.log2(time_in_bin)

        # Marker of whether it is in the bin
        within_bin = jnp.logical_and(
            tiled_bins[:, :-1] <= tiled_times,
            tiled_times < tiled_bins[:, 1:],
        )

        is_event = jnp.expand_dims(~batch["task"]["is_censor"], 1) * within_bin

        return total_reps, log_time_in_bin, is_event

    reprs = []
    label_ages = []
    label_pids = []
    log_times = []
    is_events = []
    event_times = []
    is_censors = []
    split_indices = []

    for i, split in enumerate(("train", "dev", "test")):
        print("Starting batches", split, datetime.datetime.now())
        if False:
            batches = femr.models.dataloader.Batches(
                data_path=args.data_path,
                batch_info_path=args.batch_info_path,
                seed=config["seed"],
                num_batch_threads=4,
                token_dropout=0,
                num_epochs=1,
                split=split,
                num_batches=loader.get_number_of_batches(split),
            )
        print("Starting to process", split, datetime.datetime.now())

        # while True:
        #    next_one = batches.get_next()
        #    if not next_one:
        #        print("Done processing")
        #        break

        #    raw_batch, step = next_one
        #    if not raw_batch:
        #        print("Skipping none")
        #        continue
        for j in range(loader.get_number_of_batches(split)):
            raw_batch = loader.get_batch(split, j)

            batch = jax.tree_map(lambda a: jax.device_put(a, device=jax.devices("gpu")[0]), raw_batch)

            repr, log_time, is_event = compute_repr(
                params,
                rng,
                config,
                batch,
                time_bins,
            )

            p_index = batch["transformer"]["label_indices"] // batch["transformer"]["length"]
            p_index = p_index[: batch["num_indices"]]

            reprs.append(repr[: batch["num_indices"], :, :])
            label_ages.append(raw_batch["task"]["label_ages"][: batch["num_indices"]])
            assert raw_batch["task"]["label_ages"].dtype == np.uint32
            label_pids.append(raw_batch["patient_ids"][p_index])
            log_times.append(log_time[: batch["num_indices"], :])
            is_events.append(is_event[: batch["num_indices"], :])
            split_indices.append(np.ones(batch["num_indices"]) * i)
            event_times.append(raw_batch["task"]["event_times"][: batch["num_indices"]])
            is_censors.append(raw_batch["task"]["is_censor"][: batch["num_indices"]])

        print("Actually done", datetime.datetime.now())

    print("About to concat 1", datetime.datetime.now())
    reprs = jnp.concatenate(reprs, axis=0)
    print("About to concat 2", datetime.datetime.now())
    label_ages = jnp.concatenate(label_ages, axis=0)
    print("About to concat 3", datetime.datetime.now())
    assert label_ages.dtype == jnp.uint32
    label_pids = np.concatenate(label_pids, axis=0)
    assert label_pids.dtype == np.uint64
    print("About to concat 4", datetime.datetime.now())
    log_times = jnp.concatenate(log_times, axis=0)
    print("About to concat 5", datetime.datetime.now())
    is_events = jnp.concatenate(is_events, axis=0)
    print("About to concat 6", datetime.datetime.now())
    split_indices = jnp.concatenate(split_indices, axis=0)
    print("About to concat 7", datetime.datetime.now())

    event_times = jnp.concatenate(event_times, axis=0)
    print("About to concat 8", datetime.datetime.now())
    is_censors = jnp.concatenate(is_censors, axis=0)
    print("About to concat 9", datetime.datetime.now())

    print("Computed reprs")
    if True:
        with open("what.pkl", "wb") as f:
            pickle.dump(
                [
                    reprs,
                    label_ages,
                    label_pids,
                    log_times,
                    is_events,
                    split_indices,
                    event_times,
                    is_censors,
                ],
                f,
            )
    del model
    del loader
else:
    with open("what.pkl", "rb") as f:
        (
            reprs,
            label_ages,
            label_pids,
            log_times,
            is_events,
            split_indices,
            event_times,
            is_censors,
        ) = pickle.load(f)

print(reprs.dtype)
print(is_events.dtype)
print(log_times.dtype)
print(is_events.shape)

print(np.mean(split_indices == 0))

filter_tags = np.random.random(size=is_events.shape[0])
has_any = np.sum(is_events, axis=1)
print("Starting prevalence", np.mean(has_any))
filter_tags[has_any == 0] = 0

log_times = log_times.astype(log_times.dtype)


@functools.partial(jax.jit)
def compute_loss(beta, reprs, log_times, is_events, l=0):
    hazards = jnp.dot(reprs, beta)

    assert hazards.shape == is_events.shape

    event_loss = jnp.log(2) * (hazards * is_events).mean(dtype=jnp.float32)
    event_loss = -event_loss

    survival_loss = jnp.exp2(hazards + log_times).mean(dtype=jnp.float32)

    return survival_loss + event_loss + 0.5 * l * (beta[:-1] ** 2).sum()


@functools.partial(jax.jit)
def compute_grad(beta, data, l=0):
    reprs = data["reprs"]
    log_times = data["log_times"]
    is_events = data["is_events"]
    hazards = jnp.dot(reprs, beta)

    assert hazards.shape == is_events.shape

    event_loss = jnp.log(2) * (reprs * is_events.reshape(is_events.shape + (1,))).mean(axis=(0, 1), dtype=jnp.float32)
    event_loss = -event_loss

    survival_loss = jnp.log(2) * (jnp.exp2(hazards + log_times).reshape(hazards.shape + (1,)) * reprs).mean(
        axis=(0, 1), dtype=jnp.float32
    )

    mask = beta.at[-1].set(0)

    return survival_loss + event_loss + l * mask


@functools.partial(jax.jit)
def compute_hessian(beta, u, data, l=0):
    reprs = data["reprs"]
    log_times = data["log_times"]
    is_events = data["is_events"]

    hazards = jnp.dot(reprs, beta)
    factor = jnp.dot(reprs, u) ** 2

    assert hazards.shape == is_events.shape

    scale = jnp.log(2) * jnp.log(2) * jnp.exp2(hazards + log_times)

    survival_loss = (scale * factor).mean(dtype=jnp.float32)

    val = u.at[-1].set(0)

    return survival_loss + l * jnp.dot(val, val)


@functools.partial(jax.jit, donate_argnums=(0, 1, 2))
def conjugate_gradient(last_w, last_gradient, last_u, data, l):
    g = compute_grad(last_w, data, l=l)
    if last_gradient is None:
        u = g
    else:
        delta = g - last_gradient
        beta = jnp.dot(g, delta) / jnp.dot(last_u, delta)
        u = g - last_u * beta
    w = last_w - (jnp.dot(g, u) / compute_hessian(last_w, u, data, l=l)) * u
    return w, g, u


# for frac in [0.1, 0.2, 0.5, 1]:
for frac in [1]:
    print("WORKING ON", frac)
    train_mask = jnp.logical_and(split_indices == 0, filter_tags < frac)
    print("Total", np.mean(train_mask))

    if args.subsample_frac is not None:
        original = train_mask.sum()
        train_mask = train_mask * (np.random.rand(train_mask.shape[0]) * 100 < args.subsample_frac)
        print("Sampled from ", train_mask.sum(), "from", original)

    train_reprs = reprs[train_mask, :, :]
    train_log_times = log_times[train_mask, :]
    train_is_events = is_events[train_mask, :]

    train_has_any = np.sum(train_is_events, axis=1)
    print("Train prevalence", np.mean(train_has_any))

    rng = jax.random.PRNGKey(42)

    beta = jnp.ones(train_reprs.shape[-1], dtype=jnp.float32) / 1000
    beta = jnp.zeros(train_reprs.shape[-1], dtype=jnp.float32)
    beta = beta.at[-1].set(jnp.log2(jnp.array(batch_info["config"]["task"]["lambda"])))

    l = 0.1

    starting_loss = compute_loss(beta, train_reprs, train_log_times, train_is_events)

    if False:
        print(compute_grad(beta, train_reprs, train_log_times, train_is_events, l=l)[:10])
        print(jax.grad(compute_loss)(beta, train_reprs, train_log_times, train_is_events, l=l)[:10])
        hessian = jax.hessian(compute_loss)(beta, train_reprs, train_log_times, train_is_events, l=l)
        u = jax.random.uniform(rng, shape=beta.shape)
        a = compute_hessian(beta, u, train_reprs, train_log_times, train_is_events, l=l)
        print(a)
        print(u @ (hessian @ u.T))
        print(starting_loss)

    data = {"reprs": train_reprs, "log_times": train_log_times, "is_events": train_is_events}

    best_score = None
    best_hazards = None
    best_test = None
    best_beta = None

    start_l, end_l = -5, 1
    for l_exp in np.linspace(end_l, start_l, num=20):
        if l_exp == start_l:
            l = 0
        else:
            l = 10 ** (l_exp)
        # print("Starting to train", l, datetime.datetime.now())
        g = None
        u = None
        while True:
            # val = compute_loss(beta, train_reprs, train_log_times, train_is_events)
            beta, g, u = conjugate_gradient(beta, g, u, data, l)
            grad_norm = jnp.linalg.norm(g, ord=2)

            if grad_norm < 0.0001:
                break

        final_loss = compute_loss(beta, train_reprs, train_log_times, train_is_events)
        if False:
            print(
                "My optimizer",
                final_loss,
                jnp.linalg.norm(beta, ord=2),
                datetime.datetime.now(),
            )

        hazards = jnp.dot(reprs, beta)

        def get_c(split_index):
            mask = split_indices == split_index

            event_time = event_times[mask]
            is_censor = is_censors[mask]
            # print(split_index, mask.sum(), event_time, is_censor)
            # print(is_censor.sum(), is_censor.shape)

            limit_time = jnp.quantile(event_time[~is_censor], 0.9)
            is_censor = is_censor.at[event_time > limit_time].set(True)
            event_time = event_time.at[event_time > limit_time].set(limit_time)

            return femr.extension.metrics.compute_c_statistic(
                event_time,
                is_censor,
                time_bins[:-1],
                hazards[mask, :],
            )[0]

        for i, name in enumerate(["train", "dev", "test"]):
            score = get_c(i)
            print(name, score)
            if name == "dev":
                if best_score is None or score > best_score:
                    best_score, best_hazards, best_beta = score, hazards, np.array(beta)
                    best_test = get_c(2)

    print("Got best:", best_test)

    predictions: Dict[Tuple[int, datetime.datetime], Any] = {}

    num_skipped = 0

    hazards = np.array(best_hazards)
    label_pids = np.array(label_pids)
    label_ages = np.array(label_ages)
    event_times = np.array(event_times)
    is_censors = np.array(is_censors)

    for i in range(hazards.shape[0]):
        hazard = hazards[i, :]
        pid = int(label_pids[i])
        age = label_ages[i]
        event_time = event_times[i]
        is_censor = is_censors[i]

        birth_date = datetime.datetime.combine(database.get_patient_birth_date(pid), datetime.time.min)

        prediction_date = birth_date + datetime.timedelta(minutes=int(age))

        k = (inverse_map[pid], prediction_date)

        v = (
            prediction_date,
            np.array(hazard),
            float(event_time),
            float(is_censor),
        )

        target_labels = actual_patients_to_evaluate[pid]
        target_labels = [label for label in target_labels if label.time == prediction_date]
        assert len(target_labels) == 1
        target_label = target_labels[0]
        assert is_censor == target_label.value.is_censored
        assert event_time == (target_label.value.time_to_event) / datetime.timedelta(minutes=1)

        if k not in predictions:
            predictions[k] = v
        else:
            num_skipped += 1
            predictions[k] = max(predictions[k], v)

    found_patients = {k[0] for k in predictions}

    for k in actual_patients_to_evaluate:
        if inverse_map[k] not in found_patients:
            print("Missing!", k)

    if args.probe is not None:
        with open(args.probe, 'wb') as f:
            pickle.dump(best_beta, f)

    print(len(predictions), num_expected_labels)

    assert len(predictions) == num_expected_labels

    with open(args.target_path, 'wb') as f:
        pickle.dump(predictions, open(args.target_path, "wb"))
