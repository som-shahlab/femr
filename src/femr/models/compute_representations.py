import argparse
import datetime
import functools
import logging
import os
import pickle
import random
from typing import TypeVar

import haiku as hk
import jax
import jax.numpy as jnp
import jax.scipy.optimize
import msgpack
import numpy as np
import optax
import sklearn.metrics

import femr.datasets
import femr.extension.dataloader
import femr.models.dataloader
import femr.models.transformer

os.environ["JAX_NUMPY_RANK_PROMOTION"] = "raise"


T = TypeVar("T")


def compute_representations() -> None:
    print("Start", datetime.datetime.now())

    parser = argparse.ArgumentParser(prog="Train a head model")
    parser.add_argument("output_path", type=str)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batches_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    database = femr.datasets.PatientDatabase(args.data_path)

    with open(os.path.join(args.model_dir, "best"), "rb") as f:
        params = pickle.load(f)

    params = femr.models.transformer.convert_params(params, dtype=jnp.float16)

    batch_info_path = os.path.join(args.batches_path, "batch_info.msgpack")

    with open(batch_info_path, "rb") as f:
        batch_info = msgpack.load(f, use_list=False)

    with open(os.path.join(args.model_dir, "config.msgpack"), "rb") as f:
        config = msgpack.load(f, use_list=False)

    config = hk.data_structures.to_immutable_dict(config)

    random.seed(config["seed"])
    rng = jax.random.PRNGKey(42)

    assert batch_info["config"]["task"]["type"] == "labeled_patients"
    labeler_type = batch_info["config"]["task"]["labeler_type"]

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

        offsets = jnp.ones(
            (repr.shape[0], 1),
            dtype=repr.dtype,
        )
        return jnp.concatenate((repr, offsets), axis=-1)

    l_reprs = []
    l_repr_ages = []
    l_repr_pids = []
    l_repr_offsets = []

    for i, split in enumerate(("train", "dev", "test")):
        print("Starting batches", split, datetime.datetime.now())
        print("Starting to process", split, datetime.datetime.now())

        for j in range(loader.get_number_of_batches(split)):
            raw_batch = loader.get_batch(split, j)

            batch = jax.tree_map(lambda a: jax.device_put(a, device=jax.devices("gpu")[0]), raw_batch)

            repr = compute_repr(
                params,
                rng,
                config,
                batch,
            )

            def slice(val):
                if len(val.shape) == 3:
                    return val[: batch["num_indices"], :, :]
                if len(val.shape) == 2:
                    return val[: batch["num_indices"], :]
                elif len(val.shape) == 1:
                    return val[: batch["num_indices"]]

            p_index = batch["transformer"]["label_indices"] // batch["transformer"]["length"]
            p_index = slice(p_index)

            l_reprs.append(slice(repr))
            assert repr.dtype == jnp.float16
            l_repr_ages.append(raw_batch["transformer"]["integer_ages"][slice(batch["transformer"]["label_indices"])])
            l_repr_pids.append(raw_batch["patient_ids"][p_index])
            l_repr_offsets.append(raw_batch["offsets"][p_index])

    print("About to concat 1", datetime.datetime.now())
    reprs = jnp.concatenate(l_reprs, axis=0)
    print("About to concat 2", datetime.datetime.now())
    repr_ages = np.concatenate(l_repr_ages, axis=0)
    print("About to concat 3", datetime.datetime.now())

    assert repr_ages.dtype == np.uint32
    repr_pids = np.concatenate(l_repr_pids, axis=0)
    assert repr_pids.dtype == np.int64
    repr_offsets = np.concatenate(l_repr_offsets, axis=0)

    label_pids = np.array([val[0] for val in batch_info["config"]["task"]["labels"]], dtype=np.int64)
    label_ages = np.array([val[1] for val in batch_info["config"]["task"]["labels"]], dtype=np.uint32)
    label_values = np.array([val[2] for val in batch_info["config"]["task"]["labels"]])

    print("About to sort labels", datetime.datetime.now(), len(label_pids))

    sort_indices = np.lexsort((label_ages, label_pids))

    print("Done sorting labels", datetime.datetime.now())

    label_pids = label_pids[sort_indices]
    label_ages = label_ages[sort_indices]
    label_values = label_values[sort_indices]

    repr_offsets = repr_offsets.astype(np.int32)

    print("About to sort representations", datetime.datetime.now(), len(repr_offsets))
    sort_indices = np.lexsort((-repr_offsets, repr_ages, repr_pids))
    print("Done sorting representations", datetime.datetime.now())

    repr_offsets = repr_offsets[sort_indices].astype(np.uint32)
    repr_ages = repr_ages[sort_indices]
    repr_pids = repr_pids[sort_indices]

    matching_indices = femr.extension.dataloader.compute_repr_label_alignment(
        label_pids, label_ages, repr_pids, repr_ages, repr_offsets
    )

    print("Done sorting labels and representations", datetime.datetime.now())

    assert np.all(repr_ages[matching_indices] <= label_ages)
    assert np.all(repr_pids[matching_indices] == label_pids)

    print("Creating representations")

    reprs = np.array(reprs[sort_indices[matching_indices], :])

    print("Computing dates")

    prediction_dates = []
    for pid, age in zip(label_pids, label_ages):
        birth_date = datetime.datetime.combine(database.get_patient_birth_date(pid), datetime.time.min)
        prediction_dates.append(birth_date + datetime.timedelta(minutes=int(age)))

    print("About to save", datetime.datetime.now())

    with open(args.output_path, "wb") as of:
        pickle.dump([reprs, label_pids, label_values, np.array(prediction_dates)], of)