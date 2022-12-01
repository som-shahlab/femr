import os

os.environ["JAX_NUMPY_RANK_PROMOTION"] = "raise"

import argparse

parser = argparse.ArgumentParser(prog="Train")
parser.add_argument("directory", type=str)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--batch_info_path", type=str, required=True)
parser.add_argument("--learning_rate", type=float, required=True)
parser.add_argument("--rotary_type", type=str, required=True)
parser.add_argument("--clmbr_survival_dim", type=int)
parser.add_argument("--num_batch_threads", type=int)

args = parser.parse_args()

import pickle
import piton.extension.dataloader
from immutabledict import immutabledict
import logging
import queue
import msgpack
import piton.models.transformer
import functools
import random
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import queue
import threading
import copy
import jmp

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
rootLogger.info(f"Training model with {args}")

with open(args.batch_info_path, "rb") as f:
    batch_info = msgpack.load(f, use_list=False)

batch_config = batch_info['config']


batch_task = batch_config['task']
task = {}
task['type'] = batch_task['type']
if batch_task['type'] == "survival_clmbr":
    task['num_time_bins'] = len(batch_task['survival_dict']['time_bins'])
    task['num_codes'] = len(batch_task['survival_dict']['codes'])
    task['dim'] = args.clmbr_survival_dim
elif batch_config['task']['type'] == "clmbr":
    task['vocab_size'] = batch_task['vocab_size']
else:
    rootLogger.error("Invalid task?")
    exit()

config = {
    "data_path": args.data_path,
    "batch_info_path": args.batch_info_path,
    "seed": batch_config["seed"],
    "task": task,
    "transformer": {
        "vocab_size": batch_config['transformer']['vocab_size'],
        "hidden_size": 768,
        "intermediate_size": 3072,
        "n_heads": 12,
        "n_layers": 6,
        "rotary": args.rotary_type,
        "attention_width": (512 - 16),
    },
    "learning_rate": args.learning_rate,
    "max_grad_norm": 1.0,
    "l2": 0,
    "n_epochs": 10,
}

random.seed(config["seed"])

config_path = os.path.join(args.directory, "config.msgpack")
with open(config_path, "wb") as out:
    msgpack.dump(config, out)

def to_immutable(d):
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = to_immutable(v)
        else:
            result[k] = v
    
    return immutabledict(result)

config = to_immutable(config)

loader = piton.extension.dataloader.BatchCreator(args.data_path, args.batch_info_path)

logging.info("Loaded batches %s %s", loader.get_number_of_batches("train"), loader.get_number_of_batches("dev"))

def model_fn(config, batch):
    model = piton.models.transformer.EHRTransformer(config)(batch)
    return model

dummy_batch = jax.tree_map(lambda a: jnp.array(a), loader.get_batch("train", 0))

logging.info("Got dummy batch %s", str(jax.tree_map(lambda a: (a.shape, a.dtype, a.device()), dummy_batch)))

rng = jax.random.PRNGKey(42)
model = hk.transform(model_fn)

logging.info("Transformed the model function")

params = jax.jit(model.init, static_argnames="config")(
    rng,
    config=config,
    batch=dummy_batch,
)

if batch_task['type'] == "survival_clmbr":
    old_weights = params['EHRTransformer/~/SurvivalCLMBRTask']['code_weights']
    manual_weights = jnp.log2(jnp.array(batch_task['survival_dict']['lambdas'])).astype(dtype=old_weights.dtype)
    params['EHRTransformer/~/SurvivalCLMBRTask']['code_weights'] = old_weights.at[:, -1].set(manual_weights)
elif batch_config['task']['type'] == "clmbr":
    pass
else:
    rootLogger.error("Invalid task?")
    exit()

logging.info("Done initing %s", str(jax.tree_map(lambda a: (a.shape, a.dtype), params)))

def _cast_floating_to(tree: T, dtype: jnp.dtype) -> T:
    def conditional_cast(x):
        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
            x = x.astype(dtype)
        return x

    return jax.tree_util.tree_map(conditional_cast, tree)


@functools.partial(jax.jit, static_argnames="config")
def compute_loss(params, rng, config, batch):
    loss = model.apply(params, rng, config, batch)[0]
    return loss


def compute_total_loss(split, params, rng, config):
    total_loss = 0
    num_to_get = min(500, loader.get_number_of_batches(split))
    for i in range(num_to_get):
        batch = loader.get_batch(split, i)
        total_loss += compute_loss( 
            _cast_floating_to(params, jnp.float16), rng, config, batch
        )

    return total_loss / num_to_get


@jax.value_and_grad
def loss_value_and_grad(params, loss_scale, rng, config, batch):
    loss = model.apply(params, rng, config, batch)[0]

    assert loss.dtype == jnp.float32

    post_scale = loss_scale.scale(loss)
    return post_scale


def apply_optimizer(params, grads, opt_state):
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state


@functools.partial(jax.jit, static_argnames="config")
def update(params, loss_scale, rng, opt_state, config, batch):
    batch_loss, grads = loss_value_and_grad(
        _cast_floating_to(params, jnp.float16), loss_scale, rng, config, batch
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

lr_schedule = make_lr_schedule(warmup_percentage=0.1, total_steps=total_steps)
opt = optax.chain(
    optax.clip_by_global_norm(config["max_grad_norm"]),
    optax.adam(learning_rate=config["learning_rate"]),
    optax.scale_by_schedule(lr_schedule),
)
opt_state = opt.init(params)

loss_scale = jmp.DynamicLossScale(jnp.array(2**15, dtype=jnp.float32))

best_loss = float("inf")

index_queue = queue.Queue(maxsize=300)

def index_thread(index_queue, seed, total_steps, num_train_batches, num_batch_threads):
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

batch_queue = queue.Queue(maxsize=300)

def batch_thread(index_queue, batch_queue, data_path, batch_info_path):
    thread_loader = piton.extension.dataloader.BatchCreator(
        data_path, batch_info_path
    )
    while True:
        next_item = index_queue.get()
        if next_item is None:
            batch_queue.put(None)
            break
        
        batch_index, step = next_item

        batch = thread_loader.get_batch(
            "train", batch_index
        )
        batch = jax.tree_map(lambda a: jnp.array(a), batch)
        batch_queue.put((batch, step))

    batch_queue.put(None)

batcher_threads = [threading.Thread(
    target=batch_thread,
    args=(
        index_queue,
        batch_queue,
        args.data_path,
        args.batch_info_path,
    ),
    name="batch_thread",
    daemon=True,
) for _ in range(args.num_batch_threads)]

remaining_threads = 0
for t in batcher_threads:
    remaining_threads += 1
    t.start()

logging.info("Starting loss scale %s", loss_scale)
logging.info("Starting train loss %s", compute_total_loss("train", params, rng, config))
dev_loss = compute_total_loss("dev", params, rng, config)
logging.info("Starting dev loss %s", dev_loss)

while True:
    next_item = batch_queue.get()
    if next_item is None:
        remaining_threads -= 1
        if remaining_threads == 0:
            break

    batch, step = next_item

    if step % 100 == 0:
        logging.info(f"[Step {step}]")

    if (step % 5000 == 0 and step != 0) or step == 500:
        logging.info("Loss scale %s", loss_scale)
        logging.info("Train loss %s", compute_total_loss("train", params, rng, config))
        dev_loss = compute_total_loss("dev", params, rng, config)
        logging.info("Dev loss %s", dev_loss)
        if dev_loss != dev_loss:
            logging.info("Diverged, shutting down")
            break
        if dev_loss < best_loss:
            best_loss = dev_loss
            with open(os.path.join(args.directory, "best"), "wb") as out:
                pickle.dump(params, out)
            with open(
                os.path.join(args.directory, "best_opt_state"), "wb"
            ) as out:
                pickle.dump(opt_state, out)
            with open(os.path.join(args.directory, "best_info"), "w") as out_t:
                out_t.write(f"Step {step}, Loss {dev_loss}")
        logging.info("Continuing to train ...")

    params, opt_state, batch_loss, loss_scale = update(
        params, loss_scale, rng, opt_state, config, batch
    )

for t in batcher_threads:
    t.join()
