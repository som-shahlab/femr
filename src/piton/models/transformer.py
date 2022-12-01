import piton.jax
import jax.numpy as jnp
import jax.lax as lax
import haiku as hk
from io import BytesIO
from jax import random

from jax import jit
from jax.random import PRNGKey
import jax
import optax
from jax import debug

# From https://github.com/kingoflolz/mesh-transformer-jax
def rotate_every_two_v2(x):
    flat_x = x.reshape(-1, x.shape[-1])

    x1 = flat_x[:, ::2]
    x2 = flat_x[:, 1::2]

    result = jnp.stack((-x2, x1), axis=-1).reshape(x.shape)

    assert x.dtype == result.dtype
    return result


def apply_rotary_pos_emb(x, sincos):
    sin, cos = sincos
    assert x.dtype == sin.dtype == cos.dtype

    if len(sin.shape) != len(x.shape):
        new_shape = (1,) + sin.shape
        sin = sin.reshape(new_shape)
        cos = cos.reshape(new_shape)

    return (x * cos) + (rotate_every_two_v2(x) * sin)


def fixed_pos_embedding(ages, dim, dtype):
    assert ages.dtype == jnp.float32
    assert len(ages.shape) == 1

    inv_freq = 1.0 / (10000 ** (jnp.linspace(0, 2, num=dim // 2)))
    inv_freq = inv_freq.reshape(1, dim // 2)
    assert inv_freq.dtype == jnp.float32

    ages = ages.reshape(ages.shape[0], 1)

    t = inv_freq * ages

    sin, cos = jnp.sin(t), jnp.cos(t)

    final_shape = (ages.shape[0], dim)

    sin = jnp.stack((sin, sin), axis=-1).reshape(final_shape).astype(dtype)
    cos = jnp.stack((cos, cos), axis=-1).reshape(final_shape).astype(dtype)

    return sin, cos


class TransformerBlock(hk.Module):
    def __init__(self, config):
        super().__init__("TransformerBlock")
        self.config = config

        self.norm = hk.LayerNorm(-1, True, True)
        self.input_proj = hk.Linear(
            output_size=3 * self.config["hidden_size"]
            + self.config["intermediate_size"],
        )
        self.output_proj = hk.Linear(
            self.config["hidden_size"],
            w_init=hk.initializers.TruncatedNormal(
                stddev=2
                / (
                    self.config["n_layers"]
                    * jnp.sqrt(self.config["hidden_size"])
                )
            ),
        )

    def __call__(self, x, normed_ages, pos_embed, batch):
        assert x.shape[1] == self.config["hidden_size"]
        assert len(x.shape) == 2

        x = self.norm(x)

        x_with_ages = jnp.concatenate(
            (x, jnp.expand_dims(normed_ages, -1)), axis=-1
        )
        assert x_with_ages.shape[0] == x.shape[0]
        assert x_with_ages.shape[1] == self.config["hidden_size"] + 1
        assert x_with_ages.dtype == x.dtype

        middle = self.input_proj(x)

        head_size = self.config["hidden_size"] // self.config["n_heads"]

        q, k, v, ff = jnp.split(
            middle,
            [i * self.config["hidden_size"] for i in range(1, 4)],
            axis=-1,
        )

        if self.config["rotary"] == "global":
            q = apply_rotary_pos_emb(q, pos_embed)
            k = apply_rotary_pos_emb(k, pos_embed)

        def move_to_batch(val):
            with_head = val.reshape(
                (x.shape[0], self.config["n_heads"], head_size)
            )
            with_head_at_start = with_head.transpose((1, 0, 2))
            return with_head_at_start

        q = move_to_batch(q)
        k = move_to_batch(k)
        v = move_to_batch(v)

        if self.config["rotary"] == "per_head":
            q = apply_rotary_pos_emb(q, pos_embed)
            k = apply_rotary_pos_emb(k, pos_embed)

        length_mask = batch["length"].astype(jnp.uint32)
        length_mask = ~(length_mask - 1)

        if hk.running_init():
            attn = jnp.zeros_like(q)
        else:
            attn = piton.jax.local_attention(
                q, k, v, length_mask, self.config["attention_width"]
            )

        def move_out_of_batch(val):
            with_head_at_start = val.transpose((1, 0, 2))
            return with_head_at_start.reshape(x.shape)

        shaped_attn = move_out_of_batch(attn)

        ff = jax.nn.gelu(ff)

        combined = jnp.concatenate((shaped_attn, ff), axis=-1)

        return x + self.output_proj(combined)


class Transformer(hk.Module):
    def __init__(self, config):
        super().__init__(name="Transformer")
        self.config = config
        self.norm = hk.LayerNorm(-1, True, True)
        self.embed = hk.Embed(
            vocab_size=self.config["vocab_size"],
            embed_dim=self.config["hidden_size"],
            w_init=hk.initializers.TruncatedNormal(
                stddev=1 / jnp.sqrt(self.config["hidden_size"])
            ),
        )

        self.layer_transform = hk.transform(
            lambda *args: TransformerBlock(config)(*args)
        )
        self.lifted_params = [
            hk.lift(self.layer_transform.init, name=f"loop_{i}")
            for i in range(self.config["n_layers"])
        ]

    def __call__(self, batch):

        ages = batch["ages"]
        assert ages.dtype == jnp.float32

        normed_ages = batch["normalized_ages"]
        assert normed_ages.dtype == jnp.float32

        x = self.embed(batch["tokens"])
        normed_ages = normed_ages.astype(x.dtype)

        if not hk.running_init():
            assert x.dtype == jnp.float16

        if self.config["rotary"] == "global":
            pos_embed = fixed_pos_embedding(
                ages, self.config["hidden_size"], x.dtype
            )
        elif self.config["rotary"] == "per_head":
            pos_embed = fixed_pos_embedding(
                ages,
                self.config["hidden_size"] // self.config["n_heads"],
                x.dtype,
            )
        elif self.config["rotary"] == "disabled":
            pos_embed = None
        else:
            raise RuntimeError("Invalid rotary embedding option")

        if hk.running_init():
            init_rngs = random.split(hk.next_rng_key(), len(self.lifted_params))
        else:
            init_rngs = [None for _ in self.lifted_params]

        all_params = [
            lifted(rng, x, normed_ages, pos_embed, batch)
            for lifted, rng in zip(self.lifted_params, init_rngs)
        ]
        flattened = [jax.tree_util.tree_flatten(a) for a in all_params]
        all_flat, all_defs = zip(*flattened)

        assert all(all_defs[0] == a for a in all_defs)

        all_stacked = [
            jnp.stack(tuple(a[i] for a in all_flat))
            for i in range(len(all_flat[0]))
        ]
        all_stacked_tree = jax.tree_util.tree_unflatten(
            all_defs[0], all_stacked
        )

        process = lambda v, params: (
            self.layer_transform.apply(
                params, None, v, normed_ages, pos_embed, batch
            ),
            None,
        )

        final_x = jax.lax.scan(process, x, all_stacked_tree)[0]

        assert final_x.dtype == x.dtype
        assert final_x.shape == x.shape

        return self.norm(final_x)


class TransformerFeaturizer(hk.Module):
    def __init__(self, config):
        super().__init__(name="TransformerFeaturizer")
        self.config = config
        self.transformer = Transformer(config)

    def __call__(self, batch):
        sequence_data = self.transformer(batch)

        assert len(sequence_data.shape) == 2

        mask = batch["label_indices"] != sequence_data.shape[0]

        return (
            jnp.take(
                sequence_data,
                batch["label_indices"],
                axis=0,
                unique_indices=True,
                indices_are_sorted=True,
                mode="fill",
                fill_value=0,
            ),
            mask,
        )


class BinaryTask(hk.Module):
    def __init__(self, config):
        super().__init__(name="BinaryClassifier")
        self.config = config
        self.final_layer = hk.Linear(output_size=1)

    def __call__(self, features, mask, batch):
        logits = self.final_layer(features)[:, 0]

        labels = batch["labels"]

        loss_vector = optax.sigmoid_binary_cross_entropy(logits, labels)
        masked_loss_vector = jnp.where(mask, loss_vector, 0)

        loss = masked_loss_vector.sum(dtype=jnp.float32) / mask.sum(
            dtype=jnp.float32
        )

        return loss, logits


class CLMBRTask(hk.Module):
    def __init__(self, config):
        super().__init__(name="CLMBRTask")
        self.config = config
        self.final_layer = hk.Linear(output_size=config["vocab_size"])

    def __call__(self, features, mask, batch):
        logits = self.final_layer(features)

        labels = batch["labels"]

        loss_vector = optax.softmax_cross_entropy_with_integer_labels(
            logits, labels
        )
        masked_loss_vector = jnp.where(mask, loss_vector, 0)

        loss = masked_loss_vector.sum(dtype=jnp.float32) / mask.sum(
            dtype=jnp.float32
        )

        return loss, logits


class SurvivalCLMBRTask(hk.Module):
    def __init__(self, config):
        super().__init__(name="SurvivalCLMBRTask")
        self.config = config
        num_codes = config["num_codes"]

        self.code_weights = hk.get_parameter(
            "code_weights",
            (num_codes, config["dim"]),
            init=hk.initializers.TruncatedNormal(stddev = 1/ jnp.sqrt(config["dim"]))
        )
        self.num_time_bins = config["num_time_bins"]
        self.dim = self.config["dim"]
        self.final_layer = hk.Linear(
            output_size=self.num_time_bins * (self.dim - 1)
        )

    def __call__(self, features, mask, batch):
        binned_reprs = self.final_layer(features).reshape(
            (features.shape[0], self.num_time_bins, self.dim - 1)
        )
        offsets = jnp.ones(
            (features.shape[0], self.num_time_bins, 1), dtype=features.dtype
        )

        total_reps = jnp.concatenate((binned_reprs, offsets), axis=-1)

        num_masked = mask.sum(dtype=jnp.float32)

        full_a = total_reps.reshape(-1, self.dim)

        assert full_a.dtype == features.dtype

        if not hk.running_init():
            survival_loss = piton.jax.exp_mean(
                full_a, self.code_weights, batch["sparse_time"]
            ) * (full_a.shape[0] / num_masked)
        else:
            survival_loss = 0

        event_loss = jnp.log(2) * piton.jax.embedding_dot(
            full_a, self.code_weights, batch["event_indices"]
        ).sum(dtype=jnp.float32)
        event_loss = -event_loss / (num_masked * self.code_weights.shape[0])

        logits = jnp.exp2(full_a @ self.code_weights.T)

        return (event_loss + survival_loss), logits


def create_task(config):
    if config["type"] == "binary":
        return BinaryTask(config)
    elif config["type"] == "clmbr":
        return CLMBRTask(config)
    elif config["type"] == "survival_clmbr":
        return SurvivalCLMBRTask(config)
    else:
        assert False


class EHRTransformer(hk.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(name="EHRTransformer")
        self.config = config
        self.featurizer = TransformerFeaturizer(self.config["transformer"])
        self.task_model = create_task(self.config["task"])

    def __call__(self, batch):
        print(
            "Compiling the transformer ...",
            batch["transformer"]["tokens"].shape,
            batch["transformer"]["label_indices"].shape,
        )
        features, mask = self.featurizer(batch["transformer"])
        return self.task_model(features, mask, batch["task"])
