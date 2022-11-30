import os

os.environ["JAX_NUMPY_RANK_PROMOTION"] = "raise"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.98"

import argparse

parser = argparse.ArgumentParser(prog="Train")
parser.add_argument("directory", type=str)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--batch_info_path", type=str, required=True)
parser.add_argument("--learning_rate", type=float, required=True)
parser.add_argument("--rotary_type", type=str, required=True)
parser.add_argument("--clmbr_survival_dim", type=int)
parser.add_argument("--num_batch_threads", type=int)
parser.add_argument("--start_from_checkpoint", type=str)
parser.add_argument("--freeze_weights", default=False, action="store_true")
parser.add_argument("--token_dropout", type=float, default=0)
parser.add_argument("--internal_dropout", type=float, default=0)
parser.add_argument("--weight_decay", type=float, default=0)

args = parser.parse_args()

import copy
import functools
import json
import logging
import pickle
import queue
import random
import threading
from typing import Any, Optional, TypeVar

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import msgpack
import optax
import sklearn.metrics

import piton.datasets
import piton.extension.dataloader
import piton.models.transformer

T = TypeVar("T")

os.mkdir(args.directory)

logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
)
rootLogger = logging.getLogger()

rootLogger.handlers[0].setFormatter(logFormatter)

fileHandler = logging.FileHandler(os.path.join(args.directory, "log"))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

rootLogger.setLevel(logging.INFO)
rootLogger.info(f"Training model with {args}")

with open(args.batch_info_path, "rb") as f:
    batch_info = msgpack.load(f, use_list=False)

batch_config = batch_info["config"]

database = piton.datasets.PatientDatabase(args.data_path)

batch_task = batch_config["task"]
task = {}
task["type"] = batch_task["type"]
if batch_task["type"] == "survival_clmbr":
    task["num_time_bins"] = len(batch_task["survival_dict"]["time_bins"])
    task["num_codes"] = len(batch_task["survival_dict"]["codes"])
    task["dim"] = args.clmbr_survival_dim
elif batch_config["task"]["type"] == "clmbr":
    task["vocab_size"] = batch_task["vocab_size"]
elif batch_config["task"]["type"] == "labeled_patients":
    task["labeler_type"] = batch_task["labeler_type"]
    if task["labeler_type"] == "survival":
        # Currently need a lot of hacks to get this working right ...
        with open(
            "../../gpu_experiments/new_batches/batch_4_fixed_again_hire/batch_info.msgpack",
            "rb",
        ) as f:
            old_batch_task = msgpack.load(f)["config"]["task"]

            task["time_bins"] = old_batch_task["survival_dict"]["time_bins"]

        with open(
            "../../gpu_experiments/best_surv_model/config.msgpack", "rb"
        ) as f:
            old_config = msgpack.load(f)
            task["dim"] = 512
else:
    rootLogger.error("Invalid task? " + batch_task["task"])
    exit()

config = {
    "data_path": args.data_path,
    "batch_info_path": args.batch_info_path,
    "seed": batch_config["seed"],
    "task": task,
    "transformer": {
        "vocab_size": batch_config["transformer"]["vocab_size"],
        "hidden_size": 768,
        "intermediate_size": 3072,
        "n_heads": 12,
        "n_layers": 6,
        "rotary": args.rotary_type,
        "attention_width": (512 - 16),
        "internal_dropout": args.internal_dropout,
        "is_hierarchical": batch_config["transformer"]["is_hierarchical"],
    },
    "learning_rate": args.learning_rate,
    "max_grad_norm": 1.0,
    "weight_decay": args.weight_decay,
    "n_epochs": 100,
}

logging.info("Got config %s", config)

random.seed(config["seed"])

config_path = os.path.join(args.directory, "config.msgpack")
with open(config_path, "wb") as out:
    msgpack.dump(config, out)

config = hk.data_structures.to_immutable_dict(config)

loader = piton.extension.dataloader.BatchLoader(
    args.data_path, args.batch_info_path
)

logging.info(
    "Loaded batches %s %s",
    loader.get_number_of_batches("train"),
    loader.get_number_of_batches("dev"),
)


def model_fn(config, batch, is_training):
    model = piton.models.transformer.EHRTransformer(config)(batch, is_training)
    return model


dummy_batch = jax.tree_map(lambda a: jnp.array(a), loader.get_batch("train", 0))

logging.info(
    "Got dummy batch %s",
    str(jax.tree_map(lambda a: (a.shape, a.dtype, a.device()), dummy_batch)),
)

rng = jax.random.PRNGKey(42)
model = hk.transform(model_fn)

logging.info("Transformed the model function")

params = jax.jit(model.init, static_argnames=["config", "is_training"])(
    rng, config=config, batch=dummy_batch, is_training=True
)

if batch_task["type"] == "survival_clmbr":
    old_code_weight_bias = params["EHRTransformer/~/SurvivalCLMBRTask"][
        "code_weight_bias"
    ]
    new_code_weight_bias = (
        jnp.log2(jnp.array(batch_task["survival_dict"]["lambdas"]))
        .astype(dtype=old_code_weight_bias.dtype)
        .reshape(old_code_weight_bias.shape)
    )
    params["EHRTransformer/~/SurvivalCLMBRTask"][
        "code_weight_bias"
    ] = new_code_weight_bias
elif batch_task["type"] == "clmbr":
    pass
elif batch_task["type"] == "labeled_patients":
    if batch_task["labeler_type"] == "survival":
        old_weights = params["EHRTransformer/~/SurvivalTask"]["code_weight"]
        params["EHRTransformer/~/SurvivalTask"]["code_weight"] = old_weights.at[
            0, -1
        ].set(jnp.log2(batch_task["lambda"]))
else:
    rootLogger.error("Invalid task for postprocess?")
    exit()

non_fit_params = {}
if args.start_from_checkpoint is not None:
    with open(os.path.join(args.start_from_checkpoint, "best"), "rb") as f:
        checkpointed_weights = pickle.load(f)

        if (
            batch_task["type"] == "labeled_patients"
            and batch_task["labeler_type"] == "survival"
            and (
                "EHRTransformer/~/SurvivalCLMBRTask/~/linear"
                in checkpointed_weights
            )
        ):
            magic_layer = checkpointed_weights[
                "EHRTransformer/~/SurvivalCLMBRTask/~/linear"
            ]
            checkpointed_weights[
                "EHRTransformer/~/SurvivalTask/~/linear"
            ] = magic_layer

        for p, v in list(params.items()):
            if p in checkpointed_weights:
                value_to_replace = checkpointed_weights[p]
                for name, value in v.items():
                    if name not in value_to_replace:
                        logging.error(
                            "Could not find value? %s %s",
                            p,
                            name,
                        )
                        exit()
                    if value_to_replace[name].shape != value.shape:
                        logging.error(
                            "Invalid shape %s %s %s %s",
                            p,
                            name,
                            value.shape,
                            value_to_replace[name].shape,
                        )
                        exit()

                if args.freeze_weights:
                    non_fit_params[p] = checkpointed_weights[p]
                    del params[p]
                else:
                    params[p] = checkpointed_weights[p]
            else:
                logging.info("Have to train %s from scratch", p)
                params[p] = params[p]


non_fit_params = piton.models.transformer.convert_params(
    non_fit_params, jnp.float16
)
non_fit_params = hk.data_structures.to_immutable_dict(non_fit_params)

logging.info(
    "Done initing %s", str(jax.tree_map(lambda a: (a.shape, a.dtype), params))
)


@functools.partial(
    jax.jit,
    static_argnums=(
        3,
        5,
    ),
)
def compute_loss(
    params, non_fit_params, rng, config, batch, requires_logits=False
):
    total_params = params | hk.data_structures.to_mutable_dict(non_fit_params)
    loss, logits = model.apply(
        total_params, rng, config, batch, is_training=False
    )
    if requires_logits:
        return loss, logits
    else:
        return loss, None


def compute_total_loss(split, params, non_fit_params, rng, config):
    num_to_get = min(500, loader.get_number_of_batches(split))
    total_loss = 0
    total_indices = 0

    logits = []

    is_censor = []
    event_times = []
    labels = []

    for i in range(num_to_get):
        batch = loader.get_batch(split, i)
        if batch["num_indices"] == 0:
            print("Skipping ", i, " due to no indices")
            continue
        loss, logit = compute_loss(
            piton.models.transformer.convert_params(params, jnp.float16),
            non_fit_params,
            rng,
            config,
            batch,
            requires_logits=config["task"]["type"] == "labeled_patients",
        )
        total_loss += loss * batch["num_indices"]
        total_indices += batch["num_indices"]
        if config["task"]["type"] == "labeled_patients":
            if config["task"]["labeler_type"] == "survival":
                logits.append(logit[: batch["num_indices"], :])
                is_censor.append(
                    batch["task"]["is_censor"][: batch["num_indices"]]
                )
                event_times.append(
                    batch["task"]["event_times"][: batch["num_indices"]]
                )
            else:
                logits.append(logit[: batch["num_indices"]])
                labels.append(batch["task"]["labels"][: batch["num_indices"]])

    loss = float(total_loss / total_indices)

    if config["task"]["type"] == "labeled_patients":
        if config["task"]["labeler_type"] == "survival":
            logits = jnp.concatenate(logits, axis=0)
            is_censor = jnp.concatenate(is_censor, axis=0)
            event_times = jnp.concatenate(event_times, axis=0)

            c_statistic = piton.extension.metrics.compute_c_statistic(
                event_times,
                is_censor,
                jnp.array(config["task"]["time_bins"]) * 60 * 24,
                logits,
            )
        elif config["task"]["labeler_type"] == "binary":
            logits = jnp.concatenate(logits, axis=0)
            labels = jnp.concatenate(labels, axis=0)
            c_statistic = sklearn.metrics.roc_auc_score(labels, logits)
    else:
        c_statistic = -loss

    result = {
        "loss": loss,
        "c_statistic": c_statistic,
    }

    return result


@jax.value_and_grad
def loss_value_and_grad(params, non_fit_params, loss_scale, rng, config, batch):
    total_params = params | hk.data_structures.to_mutable_dict(non_fit_params)
    loss = model.apply(total_params, rng, config, batch, is_training=True)[0]

    assert loss.dtype == jnp.float32

    post_scale = loss_scale.scale(loss)
    return post_scale


def apply_optimizer(params, grads, opt_state):
    updates, opt_state = opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state


@functools.partial(
    jax.jit,
    static_argnums=(5,),
    donate_argnums=(0, 2, 3, 4, 6),
)
def update(params, non_fit_params, loss_scale, rng, opt_state, config, batch):
    batch_loss, grads = loss_value_and_grad(
        piton.models.transformer.convert_params(params, jnp.float16),
        non_fit_params,
        loss_scale,
        rng,
        config,
        batch,
    )

    batch_loss = loss_scale.unscale(batch_loss.astype(jnp.float32))
    grads = loss_scale.unscale(
        piton.models.transformer.convert_params(grads, jnp.float32)
    )

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
            (percent_complete <= warmup_percentage), jnp.float32
        )
        scale = (
            before_peak * (percent_complete / warmup_percentage)
            + (1 - before_peak)
        ) * (1 - percent_complete)
        return scale

    return lr_schedule


num_train_batches = loader.get_number_of_batches("train")

total_steps = config["n_epochs"] * num_train_batches
logging.info(
    "total steps %s num train batches %s", total_steps, num_train_batches
)


def should_decay(module_name, name, value):
    return name not in ("b", "scale", "embeddings", "code_weight_bias")


weight_decay_mask = hk.data_structures.map(should_decay, params)

logging.info("Applying decay mask %s", weight_decay_mask)

lr_schedule = make_lr_schedule(warmup_percentage=0.01, total_steps=total_steps)
weight_decay = args.weight_decay
logging.info("Using weight decay %s", weight_decay)
opt = optax.chain(
    optax.clip_by_global_norm(config["max_grad_norm"]),
    optax.adamw(
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        mask=weight_decay_mask,
    ),
    optax.scale_by_schedule(lr_schedule),
)
opt_state = opt.init(params)

loss_scale = jmp.DynamicLossScale(jnp.array(2**15, dtype=jnp.float32))

best_loss = float("inf")


logging.info("Starting loss scale %s", loss_scale)
logging.info(
    "Starting train loss %s",
    compute_total_loss("train", params, non_fit_params, rng, config),
)
dev_loss = compute_total_loss("dev", params, non_fit_params, rng, config)
logging.info("Starting dev loss %s", dev_loss)


class Batches:
    def __init__(self, num_batch_threads):
        index_queue: queue.Queue[Optional[int]] = queue.Queue(maxsize=300)

        def index_thread(
            index_queue, seed, total_steps, num_train_batches, num_batch_threads
        ):
            rng = random.Random(seed)
            order = None
            for step in range(total_steps):
                if step % num_train_batches == 0:
                    order = list(range(num_train_batches))
                    rng.shuffle(order)

                index_queue.put((order[step % num_train_batches], step))

            for _ in range(num_batch_threads):
                index_queue.put(None)

        batcher_thread = threading.Thread(
            target=index_thread,
            args=(
                index_queue,
                config["seed"],
                total_steps,
                num_train_batches,
                args.num_batch_threads,
            ),
            name="batch_thread",
            daemon=True,
        )
        batcher_thread.start()

        self.batch_queue: queue.Queue[Optional[Any]] = queue.Queue(maxsize=200)

        def batch_thread(
            index_queue, batch_queue, data_path, batch_info_path, token_dropout
        ):
            thread_loader = piton.extension.dataloader.BatchLoader(
                data_path, batch_info_path, token_dropout=token_dropout
            )
            while True:
                next_item = index_queue.get()
                if next_item is None:
                    batch_queue.put(None)
                    break

                batch_index, step = next_item

                batch = thread_loader.get_batch("train", batch_index)
                if batch["num_indices"] == 0:
                    batch_queue.put((None, step))
                else:
                    batch = jax.tree_map(lambda a: jnp.array(a), batch)
                    batch_queue.put((batch, step))

            batch_queue.put(None)

        batcher_threads = [
            threading.Thread(
                target=batch_thread,
                args=(
                    index_queue,
                    self.batch_queue,
                    args.data_path,
                    args.batch_info_path,
                    args.token_dropout,
                ),
                name="batch_thread",
                daemon=True,
            )
            for _ in range(num_batch_threads)
        ]

        for t in batcher_threads:
            t.start()

        self.remaining_threads = num_batch_threads

    def get_next(self):
        next_item = None

        while next_item is None:
            next_item = self.batch_queue.get()
            if next_item is not None:
                return next_item
            else:
                self.remaining_threads -= 1
                if self.remaining_threads == 0:
                    return None


batches = Batches(args.num_batch_threads)
rng_sequence = hk.PRNGSequence(rng)

per_limit = min(5000, num_train_batches)

while True:
    next_item = batches.get_next()
    if next_item is None:
        break

    batch, step = next_item

    if batch is None:
        continue

    if step % 100 == 0:
        logging.info(f"[Step {step}]")

    if (
        (step % per_limit == 0 and step != 0)
        or step == 500
        or step == 1500
        or step == 2500
    ):
        logging.info("Loss scale %s", loss_scale)
        logging.info(
            "Train loss %s",
            compute_total_loss("train", params, non_fit_params, rng, config),
        )
        dev_loss = compute_total_loss(
            "dev", params, non_fit_params, rng, config
        )
        dev_loss_metric = -dev_loss["c_statistic"]
        logging.info("Dev loss %s", dev_loss)
        if dev_loss_metric != dev_loss_metric or (
            loss_scale.loss_scale == 1 and loss_scale.counter == 0
        ):
            logging.info("Diverged, shutting down")
            break
        if dev_loss_metric < best_loss:
            best_loss = dev_loss_metric
            test_loss = compute_total_loss(
                "test", params, non_fit_params, rng, config
            )
            with open(os.path.join(args.directory, "best"), "wb") as out:
                total_params = params | hk.data_structures.to_mutable_dict(
                    non_fit_params
                )
                pickle.dump(total_params, out)
            with open(
                os.path.join(args.directory, "best_opt_state"), "wb"
            ) as out:
                pickle.dump(opt_state, out)
            with open(os.path.join(args.directory, "best_info"), "w") as out_t:
                out_t.write(f"Step {step}, Loss {dev_loss}")
            with open(
                os.path.join(args.directory, "best_test_loss"), "w"
            ) as out_t:
                json.dump(test_loss, out_t)
        logging.info("Continuing to train ...")

    params, opt_state, batch_loss, loss_scale = update(
        params,
        non_fit_params,
        loss_scale,
        next(rng_sequence),
        opt_state,
        config,
        batch,
    )
