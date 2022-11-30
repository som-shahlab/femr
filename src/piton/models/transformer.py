from __future__ import annotations

import logging

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax import debug, random

import piton.jax


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

        self.norm = hk.RMSNorm(-1)
        # self.norm = hk.LayerNorm(-1, True, True)
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

    def __call__(self, x, normed_ages, pos_embed, batch, is_training):
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

        result = self.output_proj(combined)
        if is_training and self.config["internal_dropout"] != 0:
            print(
                "Applying dropout to an internal layer",
                self.config["internal_dropout"],
            )
            result = hk.dropout(
                hk.next_rng_key(),
                rate=self.config["internal_dropout"],
                x=result,
            )

        return result


class Transformer(hk.Module):
    def __init__(self, config):
        super().__init__(name="Transformer")
        self.config = config
        self.in_norm = hk.RMSNorm(-1)
        self.out_norm = hk.RMSNorm(-1)
        # self.in_norm = hk.LayerNorm(-1, True, True)
        # self.out_norm = hk.LayerNorm(-1, True, True)
        self.embed = hk.Embed(
            vocab_size=self.config["vocab_size"],
            embed_dim=self.config["hidden_size"],
            w_init=hk.initializers.TruncatedNormal(stddev=1),
        )

        self.layer_transform = hk.transform(
            lambda *args: TransformerBlock(config)(*args)
        )
        self.lifted_params = [
            hk.lift(self.layer_transform.init, name=f"loop_{i}")
            for i in range(self.config["n_layers"])
        ]

    def __call__(self, batch, is_training):
        ages = batch["ages"]
        assert ages.dtype == jnp.float32

        normed_ages = batch["normalized_ages"]
        assert normed_ages.dtype == jnp.float32

        if self.config.get("is_hierarchical"):
            e = self.embed.embeddings

            assert e.dtype == jnp.float32

            x = piton.jax.gather_scatter(
                e, batch["sparse_token_indices"], batch["ages"].shape[0]
            )

            if "bad_tokens" in batch:
                alt = batch["bad_tokens"] @ e

                delta = jnp.abs(alt - x).max()

                debug.print(
                    "Got {a} {e} {d}",
                    a=x.sum(),
                    e=alt.sum(),
                    d=delta,
                )
        else:
            x = self.embed(batch["tokens"])

        dummy_values = jnp.ones((1, 1), dtype=x.dtype)

        x = jnp.where(
            batch["valid_tokens"].reshape((-1, 1)),
            x,
            dummy_values,
        )

        x = self.in_norm(x)

        if not hk.running_init():
            x = x.astype(jnp.float16)

        normed_ages = normed_ages.astype(x.dtype)

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

        layer_rngs = random.split(hk.next_rng_key(), len(self.lifted_params))

        all_params = [
            lifted(rng, x, normed_ages, pos_embed, batch, is_training)
            for lifted, rng in zip(self.lifted_params, layer_rngs)
        ]
        flattened = [jax.tree_util.tree_flatten(a) for a in all_params]
        all_flat, all_defs = zip(*flattened)

        assert all(all_defs[0] == a for a in all_defs)

        all_stacked = [
            jnp.stack(tuple(a[i] for a in all_flat))
            for i in range(len(all_flat[0]))
        ]

        all_stacked_tree = [
            jax.tree_util.tree_unflatten(all_defs[0], all_stacked),
            layer_rngs,
        ]

        def process(v, params_rng):
            params, rng = params_rng

            res = self.layer_transform.apply(
                params, rng, v, normed_ages, pos_embed, batch, is_training
            )
            return (v + res, None)

        final_x = jax.lax.scan(process, x, all_stacked_tree)[0]

        assert final_x.dtype == x.dtype
        assert final_x.shape == x.shape

        return self.out_norm(final_x)


class TransformerFeaturizer(hk.Module):
    def __init__(self, config):
        super().__init__(name="TransformerFeaturizer")
        self.config = config
        self.transformer = Transformer(config)

    def __call__(self, batch, is_training):
        sequence_data = self.transformer(batch, is_training)
        if is_training and self.config["internal_dropout"] != 0:
            print(
                "Applying dropout to the sequence data ",
                self.config["internal_dropout"],
            )
            sequence_data = hk.dropout(
                hk.next_rng_key(),
                rate=self.config["internal_dropout"],
                x=sequence_data,
            )

        mask = batch["label_indices"] != sequence_data.shape[0]
        assert len(sequence_data.shape) == 2

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


class BooleanTask(hk.Module):
    def __init__(self, config):
        super().__init__(name="BooleanClassifier")
        self.config = config
        self.final_layer = hk.Linear(output_size=1)

    def __call__(self, features, mask, batch, _is_training):
        logits = self.final_layer(features)[:, 0]

        labels = batch["labels"]

        loss_vector = optax.sigmoid_binary_cross_entropy(logits, labels)
        masked_loss_vector = jnp.where(mask, loss_vector, 0)

        loss = masked_loss_vector.sum(dtype=jnp.float32) / mask.sum(
            dtype=jnp.float32
        )

        return loss, logits


class SurvivalTask(hk.Module):
    def __init__(self, config):
        super().__init__(name="SurvivalTask")
        self.config = config
        self.time_bins = jnp.array(tuple(config["time_bins"]) + (float("inf"),))
        self.time_bins = self.time_bins * 60 * 24
        self.num_time_bins = len(config["time_bins"])
        self.dim = self.config["dim"]
        self.final_layer = hk.Linear(
            output_size=self.num_time_bins * (self.dim - 1)
        )

        self.code_weight = hk.get_parameter(
            "code_weight",
            (1, self.dim - 1),
            init=hk.initializers.TruncatedNormal(
                stddev=1 / jnp.sqrt(config["dim"])
            ),
        )

        self.code_weight_bias = hk.get_parameter(
            "code_weight_bias",
            (1, 1),
            init=hk.initializers.TruncatedNormal(
                stddev=1 / jnp.sqrt(config["dim"])
            ),
        )

    def __call__(self, features, mask, batch, _is_training):
        binned_reprs = self.final_layer(features).reshape(
            (features.shape[0], self.num_time_bins, self.dim - 1)
        )
        offsets = jnp.ones(
            (features.shape[0], self.num_time_bins, 1), dtype=features.dtype
        )

        times = batch["event_times"]

        tiled_bins = jnp.expand_dims(self.time_bins, 0)

        tiled_times = jnp.expand_dims(times, -1)

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

        is_event = jnp.expand_dims(~batch["is_censor"], 1) * within_bin
        assert log_time_in_bin.shape == is_event.shape

        total_reps = jnp.concatenate((binned_reprs, offsets), axis=-1)

        total_code_weight = jnp.concatenate(
            (self.code_weight, self.code_weight_bias), axis=-1
        )

        hazards = total_reps @ total_code_weight.T

        assert hazards.shape[-1] == 1
        hazards = hazards[:, :, 0]

        assert hazards.shape == is_event.shape

        num_masked = mask.sum()

        event_loss = jnp.log(2) * (hazards * is_event).sum(dtype=jnp.float32)
        event_loss = -event_loss / num_masked

        survival_loss = (
            jnp.exp2(hazards + log_time_in_bin).sum(dtype=jnp.float32)
            / num_masked
        )

        return (
            (event_loss + survival_loss),
            hazards,
        )


class CLMBRTask(hk.Module):
    def __init__(self, config):
        super().__init__(name="CLMBRTask")
        self.config = config
        self.final_layer = hk.Linear(output_size=config["vocab_size"])

    def __call__(self, features, mask, batch, _is_training):
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

        self.code_weight = hk.get_parameter(
            "code_weight",
            (num_codes, config["dim"] - 1),
            init=hk.initializers.TruncatedNormal(
                stddev=1 / jnp.sqrt(config["dim"])
            ),
        )

        self.code_weight_bias = hk.get_parameter(
            "code_weight_bias",
            (num_codes, 1),
            init=hk.initializers.TruncatedNormal(stddev=1),
        )

        self.num_time_bins = config["num_time_bins"]
        self.dim = self.config["dim"]
        self.final_layer = hk.Linear(
            output_size=self.num_time_bins * (self.dim - 1)
        )

    def __call__(self, features, mask, batch, _is_training):
        binned_reprs = self.final_layer(features).reshape(
            (features.shape[0], self.num_time_bins, self.dim - 1)
        )
        offsets = jnp.ones(
            (features.shape[0], self.num_time_bins, 1), dtype=features.dtype
        )

        total_reps = jnp.concatenate((binned_reprs, offsets), axis=-1)

        total_code_weight = jnp.concatenate(
            (self.code_weight, self.code_weight_bias), axis=-1
        )

        num_masked = mask.sum(dtype=jnp.float32)

        full_a = total_reps.reshape(-1, self.dim)

        assert full_a.dtype == features.dtype

        if not hk.running_init():
            survival_loss = piton.jax.exp_mean(
                full_a, total_code_weight, batch["sparse_time"]
            ) * (full_a.shape[0] / num_masked)
        else:
            survival_loss = 0

        event_loss = jnp.log(2) * piton.jax.embedding_dot(
            full_a, total_code_weight, batch["event_indices"]
        ).sum(dtype=jnp.float32)
        event_loss = -event_loss / (num_masked * total_code_weight.shape[0])

        logits = jnp.exp2(full_a @ total_code_weight.T)

        return (event_loss + survival_loss), logits


def create_task(config):
    if config["type"] == "clmbr":
        return CLMBRTask(config)
    elif config["type"] == "survival_clmbr":
        return SurvivalCLMBRTask(config)
    elif config["type"] == "labeled_patients":
        if config["labeler_type"] == "boolean":
            return BooleanTask(config)
        elif config["labeler_type"] == "survival":
            return SurvivalTask(config)
        else:
            assert False
    else:
        assert False


class EHRTransformer(hk.Module):
    def __init__(self, config):
        super().__init__(name="EHRTransformer")
        self.config = config
        self.featurizer = TransformerFeaturizer(self.config["transformer"])
        self.task_model = create_task(self.config["task"])

    def __call__(self, batch, is_training=False, no_task=False):
        print(
            "Compiling the transformer ...",
            batch["transformer"]["normalized_ages"].shape,
            batch["transformer"]["label_indices"].shape,
        )
        features, mask = self.featurizer(batch["transformer"], is_training)
        if no_task:
            return features, mask
        return self.task_model(features, mask, batch["task"], is_training)


def convert_params(tree: T, dtype: jnp.dtype) -> T:
    embed_k = "EHRTransformer/~/TransformerFeaturizer/~/Transformer/~/embed"

    def mapper(module_name, name, value):
        if module_name != embed_k:
            return conditional_cast(value)
        else:
            return value

    def conditional_cast(x):
        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
            x = x.astype(dtype)
        return x

    return hk.data_structures.map(mapper, tree)
