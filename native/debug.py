import os

os.environ["JAX_NUMPY_RANK_PROMOTION"] = "raise"

import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"


import pickle

import msgpack
import numpy as np
from jax import make_jaxpr

import piton.extension.dataloader
import piton.models.transformer

data_path = "/local-scratch/nigam/projects/ethanid/piton_1_extract"

dictionary_path = (
    "/local-scratch/nigam/projects/ethanid/piton/native/results/dictionary"
)

surv_dictionary_path = "/local-scratch/nigam/projects/ethanid/piton/native/results/survival_clmbr_dictionary"

print(surv_dictionary_path)

import piton.datasets

data = piton.datasets.PatientDatabase(data_path)
male_code = data.get_code_dictionary().index("Gender/M")

import json

dictionary = msgpack.load(open(dictionary_path, "rb"), use_list=False)

t = "survival_clmbr"

if t == "survival_clmbr":
    surv_dict = msgpack.load(open(surv_dictionary_path, "rb"), use_list=False)
    task = {"type": "survival_clmbr", "survival_dict": surv_dict, "dim": 256}
    print(task["dim"])
elif t == "clmbr":
    task = {"type": "clmbr", "vocab_size": 10_000}
else:
    labels = []

    if False:
        limit = 100
    else:
        limit = len(data)

    for patient_id in range(0, limit):
        patient = data[patient_id]
        is_male = any(event.code == male_code for event in patient.events)
        labels.append((patient.patient_id, 1, is_male))
    task = {"type": "binary", "labels": labels}

config = {
    "transformer": {
        "vocab_size": 50000,
        "dictionary": dictionary,
        "min_size": 5,
        "max_size": 13,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "n_heads": 12,
        "n_layers": 6,
        "rotary": "per_head",
        "attention_width": 256,
    },
    "task": task,
    "seed": 97,
    "splits": [["train", 0, 70], ["dev", 70, 85], ["test", 85, 100]],
    "learning_rate": 1e-3,
    "max_grad_norm": 1.0,
    "l2": 0,
    "n_epochs": 100,
}
print("WORKING WITH", config["learning_rate"])

with open("trash/config.json", "bw") as f:
    msgpack.dump(config, f)

loader = piton.extension.dataloader.BatchCreator(data_path, "trash/config.json")

print("Ready to go!")
print(loader.get_batch("train", 1))

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import optax

import piton.models.transformer


def roberta_classification_fn(batch):
    model = piton.models.transformer.EHRTransformer(config)(batch)
    return model


dummy_batch = jax.tree_map(lambda a: jnp.array(a), loader.get_batch("train", 1))
print("Batch info", jax.tree_map(lambda a: (a.shape, a.dtype), dummy_batch))

rng = jax.random.PRNGKey(42)
roberta_classifier = hk.transform(roberta_classification_fn)

params = roberta_classifier.init(
    rng,
    batch=dummy_batch,
)

print("Params info", jax.tree_map(lambda a: a.shape, params))

if False:
    print("loading from")

    loaded_params = pickle.load(open("result_clmbr/params_clmbr_110000", "rb"))
    loaded_params = jax.tree_map(lambda a: jnp.array(a), loaded_params)

    for k, v in loaded_params.items():
        if k in params:
            print("Using old ", k)
            params[k] = v


from typing import TypeVar

from jax import debug

T = TypeVar("T")


def _cast_floating_to(tree: T, dtype: jnp.dtype) -> T:
    def conditional_cast(x):
        if isinstance(x, (np.ndarray, jnp.ndarray)) and jnp.issubdtype(
            x.dtype, jnp.floating
        ):
            x = x.astype(dtype)
        return x

    return jax.tree_util.tree_map(conditional_cast, tree)


@jax.jit
def compute_loss(params, rng, batch):
    loss = roberta_classifier.apply(params, rng, batch)[0]
    return loss


def compute_total_loss(split, params, rng):
    total_loss = 0
    for i in range(100):
        batch = loader.get_batch(split, i)
        total_loss += compute_loss(
            _cast_floating_to(params, jnp.float16), rng, batch
        )

    return total_loss / 100


@jax.value_and_grad
def loss_value_and_grad(params, loss_scale, rng, batch):
    if False:
        print(
            "Inner params", jax.tree_map(lambda a: (a.shape, a.dtype), params)
        )
        print("Inner batch", jax.tree_map(lambda a: (a.shape, a.dtype), batch))
    loss = roberta_classifier.apply(params, rng, batch)[0]

    assert loss.dtype == jnp.float32

    post_scale = loss_scale.scale(loss)
    return post_scale


def apply_optimizer(params, grads, opt_state):
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state


@jax.jit
def update(params, loss_scale, rng, opt_state, batch):
    batch_loss, grads = loss_value_and_grad(
        _cast_floating_to(params, jnp.float16), loss_scale, rng, batch
    )

    batch_loss = loss_scale.unscale(batch_loss.astype(jnp.float32))
    grads = loss_scale.unscale(_cast_floating_to(grads, jnp.float32))

    grads_finite = jmp.all_finite(grads)

    loss_scale = loss_scale.adjust(grads_finite)

    new_params, opt_state = jmp.select_tree(
        grads_finite,
        apply_optimizer(params, grads, opt_state),
        (params, opt_state),
    )

    return new_params, opt_state, batch_loss, loss_scale


def make_lr_schedule(warmup_percentage, total_steps):
    def lr_schedule(step):
        percent_complete = step / total_steps
        before_peak = jax.lax.convert_element_type(
            (percent_complete <= warmup_percentage), np.float32
        )
        scale = (
            before_peak * (percent_complete / warmup_percentage)
            + (1 - before_peak)
        ) * (1 - percent_complete)
        return scale

    return lr_schedule


num_train_batches = loader.get_number_of_batches("train")

total_steps = config["n_epochs"] * num_train_batches
print("total steps", total_steps, "num train batches", num_train_batches)

lr_schedule = make_lr_schedule(warmup_percentage=0.1, total_steps=total_steps)
opt = optax.chain(
    optax.clip_by_global_norm(config["max_grad_norm"]),
    optax.adam(learning_rate=config["learning_rate"]),
    optax.scale_by_schedule(lr_schedule),
)
opt_state = opt.init(params)

import numpy as np

loss_scale = jmp.DynamicLossScale(jnp.array(2**15, dtype=jnp.float32))
# loss_scale = jmp.DynamicLossScale(jmp.half_dtype()(2))
print(loss_scale)

import time

total_batch = jnp.array(0)

print(compute_total_loss("train", params, rng))
print(compute_total_loss("dev", params, rng))
for step in range(total_steps):
    if step % 100 == 0:
        print(f"[Step {step}]")

    if step % 1000 == 1:
        start = time.time()

    if step % 1000 == 999:
        end = time.time()
        print(end - start)

    if step % 1000 == 0 and step != 0:
        with open(f"result_clmbr/params_{t}_{step}", "wb") as f:
            pickle.dump(params, f)
        print(loss_scale)
        print("Train loss", total_batch / 1000)
        print("Actual Train loss", compute_total_loss("train", params, rng))
        total_batch *= 0
        print("Dev loss", compute_total_loss("dev", params, rng))
    #        measure_current_performance(params, n_examples=100)

    # print(loss_scale)
    batch = loader.get_batch("train", step % num_train_batches)
    batch = jax.tree_map(lambda a: jnp.array(a), batch)
    # print(batch['task'])
    # print("Ready to go", jax.tree_map(lambda a: a.shape, batch))
    # Perform adam update
    # print(batch_labels)
    params, opt_state, batch_loss, loss_scale = update(
        params, loss_scale, rng, opt_state, batch
    )
    # print("Got it!", batch_loss, loss_scale)
    total_batch += batch_loss
