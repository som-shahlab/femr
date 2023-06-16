import os

os.environ["JAX_NUMPY_RANK_PROMOTION"] = "raise"

import jax
import jax.numpy as jnp
import argparse
import copy
import functools
import logging
import pickle
import collections
import csv
import queue
import random
import sklearn.metrics
import threading
from typing import Any, Dict, Mapping, Optional, TypeVar

import jax.scipy.optimize


import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import datetime
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


@functools.partial(jax.jit, donate_argnums=(0, 1, 2), static_argnames=('compute_hessian', 'compute_grad'))
def conjugate_gradient(last_w, last_gradient, last_u, data, l, compute_hessian, compute_grad):
    g = compute_grad(last_w, data, l=l)
    if last_gradient is None:
        u = g
    else:
        delta = g - last_gradient
        beta = jnp.dot(g, delta) / jnp.dot(last_u, delta)
        u = g - last_u * beta
    w = last_w - (jnp.dot(g, u) / compute_hessian(last_w, u, data, l=l)) * u
    return w, g, u

def compute_logistic_loss(beta, data, l=0):
    reprs = data['reprs']
    labels = data['labels']
    
    hazards = jnp.dot(reprs, beta)

    return optax.sigmoid_binary_cross_entropy(hazards, labels).mean(dtype=jnp.float32) + 0.5 * l * (beta[:-1] ** 2).sum()

def compute_logistic_grad(beta, data, l=0):
    reprs = data['reprs']
    labels = data['labels']
    
    hazards = jnp.dot(reprs, beta)

    assert hazards.shape == labels.shape

    logit = jax.nn.sigmoid(hazards)
    inverse_logit = jax.nn.sigmoid(-hazards)
   
    weights = -labels * inverse_logit + (1 - labels) * logit
    weights = jnp.expand_dims(weights, axis=-1)
    
    mask = beta.at[-1].set(0)

    return (weights * reprs).mean(axis=0, dtype=jnp.float32) + l * mask

def compute_logistic_hessian(beta, u, data, l=0):
    reprs = data['reprs']
    labels = data['labels']
    
    hazards = jnp.dot(reprs, beta)
    
    logit = jax.nn.sigmoid(hazards)
    inverse_logit = jax.nn.sigmoid(-hazards)
    
    factor = jnp.dot(reprs, u) ** 2
    
    val = u.at[-1].set(0)

    return (factor * logit * inverse_logit).mean(axis=0, dtype=jnp.float32) + l * jnp.dot(val, val)


def compute_survival_loss(beta, data, l=0):
    reprs = data['reprs']
    log_times = data['log_times']
    is_events = data['is_events']

    hazards = jnp.dot(reprs, beta)

    assert hazards.shape == is_events.shape

    event_loss = jnp.log(2) * (hazards * is_events).mean(dtype=jnp.float32)
    event_loss = -event_loss

    survival_loss = jnp.exp2(hazards + log_times).mean(dtype=jnp.float32)

    return survival_loss + event_loss + 0.5 * l * (beta[:-1] ** 2).sum()

def compute_survival_grad(beta, data, l=0):
    reprs = data['reprs']
    log_times = data['log_times']
    is_events = data['is_events']
    hazards = jnp.dot(reprs, beta)

    assert hazards.shape == is_events.shape

    event_loss = jnp.log(2) * (reprs * is_events.reshape(is_events.shape + (1,))).mean(axis=(0, 1), dtype=jnp.float32)
    event_loss = -event_loss

    survival_loss = jnp.log(2) * (jnp.exp2(hazards + log_times).reshape(hazards.shape + (1,)) * reprs).mean(
        axis=(0, 1), dtype=jnp.float32
    )

    mask = beta.at[-1].set(0)

    return survival_loss + event_loss + l * mask

def compute_survival_hessian(beta, u, data, l=0):
    reprs = data['reprs']
    log_times = data['log_times']
    is_events = data['is_events']

    hazards = jnp.dot(reprs, beta)
    factor = jnp.dot(reprs, u) ** 2

    assert hazards.shape == is_events.shape

    scale = jnp.log(2) * jnp.log(2) * jnp.exp2(hazards + log_times)

    survival_loss = (scale * factor).mean(dtype=jnp.float32)

    val = u.at[-1].set(0)

    return survival_loss + l * jnp.dot(val, val)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def train_linear_probe() -> None:
    print("Start", datetime.datetime.now())

    parser = argparse.ArgumentParser(prog="Train a head model")
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batches_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)

    args = parser.parse_args()

    os.mkdir(args.output_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'log')),
            logging.StreamHandler()
        ]
    )
    
    database = femr.datasets.PatientDatabase(args.data_path)

    with open(os.path.join(args.model_dir, "best"), "rb") as f:
        params = pickle.load(f)

    params = femr.models.transformer.convert_params(params, dtype=jnp.float16)

    batch_info_path = os.path.join(args.batches_path, 'batch_info.msgpack')

    with open(batch_info_path, "rb") as f:
        batch_info = msgpack.load(f, use_list=False)

    with open(os.path.join(args.model_dir, "config.msgpack"), "rb") as f:
        config = msgpack.load(f, use_list=False)

    config = hk.data_structures.to_immutable_dict(config)

    random.seed(config["seed"])
    rng = jax.random.PRNGKey(42)

    assert batch_info['config']['task']['type'] == 'labeled_patients'
    labeler_type = batch_info['config']['task']['labeler_type']
    
    if labeler_type == "survival":
        time_bins = jnp.array(config['task']['time_bins'] + [float('inf')])

    if True:
        loader = femr.extension.dataloader.BatchLoader(args.data_path, batch_info_path)

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


        @functools.partial(jax.jit, static_argnames=("config"))
        def compute_repr(params, rng, config, batch):
            repr, mask = model.apply(params, rng, config, batch)

            if labeler_type == "survival":
                final_layer = params["EHRTransformer/~/SurvivalCLMBRTask/~/linear"]

                binned_reprs = jnp.dot(repr, final_layer["w"].astype(jnp.float16))
                binned_reprs += jnp.broadcast_to(final_layer["b"].astype(jnp.float16), binned_reprs.shape)
                num_time_bins = len(time_bins) - 1

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

                return total_reps, (batch["task"]["event_times"], batch["task"]["is_censor"], log_time_in_bin, is_event)

            elif labeler_type == "boolean":
                offsets = jnp.ones(
                    (repr.shape[0], 1),
                    dtype=repr.dtype,
                )
                total_repr = jnp.concatenate((repr, offsets), axis=-1)
                return total_repr, (batch["task"]["labels"],)

        reprs = []
        repr_ages = []
        repr_pids = []
        repr_offsets = []
        repr_split = []

        for i, split in enumerate(("train", "dev", "test")):
            print("Starting batches", split, datetime.datetime.now())
            print("Starting to process", split, datetime.datetime.now())

            for j in range(loader.get_number_of_batches(split)):
                raw_batch = loader.get_batch(split, j)

                batch = jax.tree_map(lambda a: jax.device_put(a, device=jax.devices("gpu")[0]), raw_batch)

                repr, label_data = compute_repr(
                    params,
                    rng,
                    config,
                    batch,
                )
                def slice(val):
                    if len(val.shape) == 3:
                        return val[:batch['num_indices'], :, :]
                    if len(val.shape) == 2:
                        return val[:batch['num_indices'], :]
                    elif len(val.shape) == 1:
                        return val[:batch['num_indices']]

                p_index = batch["transformer"]["label_indices"] // batch["transformer"]["length"]
                p_index = slice(p_index)

                reprs.append(slice(repr))
                assert repr.dtype == jnp.float16
                repr_ages.append(raw_batch["transformer"]["integer_ages"][slice(batch["transformer"]["label_indices"])])
                repr_pids.append(raw_batch["patient_ids"][p_index])
                repr_offsets.append(raw_batch["offsets"][p_index])
                repr_split.append(np.ones(batch['num_indices']) * i)

            print("Actually done", datetime.datetime.now())

        print("About to concat 1", datetime.datetime.now())
        reprs = jnp.concatenate(reprs, axis=0)
        print("About to concat 2", datetime.datetime.now())
        repr_ages = jnp.concatenate(repr_ages, axis=0)
        print("About to concat 3", datetime.datetime.now())
        assert repr_ages.dtype == jnp.uint32
        repr_pids = np.concatenate(repr_pids, axis=0)
        assert repr_pids.dtype == np.int64
        repr_offsets = jnp.concatenate(repr_offsets, axis=0)
        repr_split = jnp.concatenate(repr_split, axis=0)

        print("Computed reprs")
        if False:
            with open("what.pkl", "wb") as f:
                pickle.dump(
                    [
                        reprs,
                        repr_ages,
                        repr_pids,
                        repr_offsets,
                        repr_split,
                    ],
                    f,
                )
        del model
        del loader
    else:
        with open("what.pkl", "rb") as f:
            (
                reprs,
                repr_ages,
                repr_pids,
                repr_offsets,
                repr_split,
            ) = pickle.load(f)

    label_pids = np.array([val[0] for val in batch_info['config']['task']['labels']], dtype=np.uint64)
    label_ages = np.array([val[1] for val in batch_info['config']['task']['labels']], dtype=np.uint32)
    label_values = np.array([val[2] for val in batch_info['config']['task']['labels']])

    sort_indices = np.lexsort((label_ages, label_pids))
    
    label_pids = label_pids[sort_indices]
    label_ages = label_ages[sort_indices]
    label_values = label_values[sort_indices]


    repr_offsets = repr_offsets.astype(np.int32)
    
    sort_indices = np.lexsort((-repr_offsets, repr_ages, repr_pids))
    
    repr_offsets = repr_offsets[sort_indices]
    repr_ages = repr_ages[sort_indices]
    repr_pids = repr_pids[sort_indices]

    split_indices = []
    matching_indices = []
    
    j = 0
    deltas = []

    for label_pid, label_age, label_value in zip(label_pids, label_ages, label_values):
        while True:
            if j + 1 == len(repr_pids):
                break
            elif repr_pids[j] < label_pid:
                pass
            else:
                next_pid = repr_pids[j + 1]
                next_age = repr_ages[j + 1]
                next_offset = repr_offsets[j + 1]

                if next_pid != label_pid:
                    break
                
                if next_age > label_age:
                    break
            
            j += 1
        
        assert repr_pids[j] == label_pid
        assert repr_ages[j] <= label_age
        delta = label_age - repr_ages[j]
        deltas.append(delta)

        if j > 0 and repr_pids[j-1] == repr_pids[j] and repr_ages[j-1] == repr_ages[j]:
            assert repr_offsets[j] < repr_offsets[j - 1]
        
        split_indices.append(repr_split[j])
        matching_indices.append(j)

    reprs = reprs[sort_indices[matching_indices], :]

    split_indices = np.array(split_indices)

    counts = collections.defaultdict(int)

    for pid in label_pids:
        counts[database.compute_split(97, pid)] += 1
    
    train_mask = split_indices == 0
    train_val_mask = split_indices <= 1
    print("Percent train", np.mean(train_mask))
    
    def apply_mask(val, mask):
        if len(val.shape) == 3:
            return val[mask, :, :]
        if len(val.shape) == 2:
            return val[mask, :]
        elif len(val.shape) == 1:
            return val[mask]

    print("Train", sum(split_indices == 0))
    print("Valid", sum(split_indices == 1))
    print("Test", sum(split_indices == 2))
    print("Total", len(split_indices))


    data = {'reprs': reprs}
    
    if labeler_type == 'survival':
        event_time, is_censor, log_times, is_events = label_datas
        data['log_times'] = log_times
        data['is_events'] = is_events
        compute_grad = compute_survival_grad
        compute_hessian = compute_survival_hessian
    elif labeler_type == 'boolean':
        labels = label_values.astype(np.float32)
        data['labels'] = labels
        print("Prevalence", np.mean(labels))
        compute_grad = compute_logistic_grad
        compute_hessian = compute_logistic_hessian
    
    data = {k: apply_mask(v, train_mask) for k, v in data.items()}

    rng = jax.random.PRNGKey(42)

    beta = jnp.zeros(reprs.shape[-1], dtype=jnp.float16)

    if labeler_type == 'survival':
        beta = beta.at[-1].set(jnp.log2(jnp.array(batch_info["config"]["task"]["lambda"])))

    best_scores = None
    best_hazards = None
    best_beta = None
    best_l = None

    def get_c(hazards, split_index):
        mask = split_indices == split_index

        if labeler_type == 'survival':
            e_t = apply_mask(event_times, mask)
            is_c = apply_mask(is_censor, mask)

            limit_time = jnp.quantile(e_t[~is_c], 0.9)
            is_c = is_c.at[e_t > limit_time].set(True)
            e_t = e_t.at[e_t > limit_time].set(limit_time)

            return femr.extension.metrics.compute_c_statistic(
                e_t,
                is_c,
                time_bins[:-1],
                apply_mask(hazards, mask),
            )[0]
        elif labeler_type == 'boolean':
            ls = apply_mask(labels, mask)

            return sklearn.metrics.roc_auc_score(ls, apply_mask(hazards, mask))

    start_l, end_l = -5, 1
    for l_exp in np.linspace(end_l, start_l, num=20):
        if l_exp == start_l:
            l = 0
        else:
            l = 10 ** (l_exp)

        g = None
        u = None
        while True:
            beta, g, u = conjugate_gradient(beta, g, u, data, l, compute_hessian=compute_hessian, compute_grad=compute_grad)
            grad_norm = jnp.linalg.norm(g, ord=2)

            if grad_norm < 0.0001:
                break

        hazards = jnp.dot(reprs, beta)

        scores = [get_c(hazards, i) for i in range(3)]
        print(l, scores)
        if best_scores is None or scores[1] > best_scores[1]:
            best_scores, best_hazards, best_beta, best_l = scores, hazards, np.array(beta), l

    logging.info(f"Train AUROC {best_scores[0]}")
    logging.info(f"Valid AUROC {best_scores[1]}")
    logging.info(f"Test AUROC {best_scores[2]}")
    logging.info(f"L2 Strength {best_l}")
    
    with open(os.path.join(args.output_dir, 'probe.pkl'), 'wb') as f:
        pickle.dump(best_beta, f)
        
    prediction_dates = []
    for pid, age in zip(label_pids, label_ages):
        birth_date = datetime.datetime.combine(database.get_patient_birth_date(pid), datetime.time.min)
        prediction_dates.append(birth_date + datetime.timedelta(minutes=int(age)))

    with open(os.path.join(args.output_dir, 'predictions.pkl'), 'wb') as f:
        pickle.dump([sigmoid(best_hazards.astype(np.float32)), label_pids, label_values, prediction_dates], f)