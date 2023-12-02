import math
from typing import TypeVar

import datasets
import flash_attn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

import femr.models.flash_attention
import femr.models.rmsnorm
import femr.models.tasks


# From https://github.com/kingoflolz/mesh-transformer-jax
def rotate_every_two_v2(x):
    flat_x = x.reshape(-1, x.shape[-1])

    x1 = flat_x[:, ::2]
    x2 = flat_x[:, 1::2]

    result = torch.stack((-x2, x1), axis=-1).reshape(x.shape)

    assert x.dtype == result.dtype
    return result


def fixed_pos_embedding(ages, dim, dtype):
    assert ages.dtype == torch.float32
    assert len(ages.shape) == 2

    inv_freq = 1.0 / (10000 ** (torch.linspace(0, 2, steps=dim // 2, device=ages.device)))
    inv_freq = inv_freq.reshape(1, 1, dim // 2)
    assert inv_freq.dtype == torch.float32

    ages = ages.reshape(ages.shape[0], ages.shape[1], 1)

    t = inv_freq * ages

    sin, cos = torch.sin(t), torch.cos(t)

    final_shape = (ages.shape[0], ages.shape[1], 1, dim)

    sin = torch.stack((sin, sin), axis=-1).reshape(final_shape).type(dtype)
    cos = torch.stack((cos, cos), axis=-1).reshape(final_shape).type(dtype)

    return sin, cos


def apply_rotary_pos_emb(x, sincos):
    sin, cos = sincos
    assert x.dtype == sin.dtype == cos.dtype

    if len(sin.shape) != len(x.shape):
        new_shape = (1,) + sin.shape
        sin = sin.reshape(new_shape)
        cos = cos.reshape(new_shape)

    return (x * cos) + (rotate_every_two_v2(x) * sin)


class FEMREncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm = femr.models.rmsnorm.RMSNorm(self.config["hidden_size"])
        self.attn_proj = nn.Linear(self.config["hidden_size"], self.config["hidden_size"] * 3)
        self.input_proj = nn.Linear(self.config["hidden_size"], self.config["intermediate_size"])
        self.output_proj = nn.Linear(
            self.config["hidden_size"] + self.config["intermediate_size"], self.config["hidden_size"]
        )

        self.activate = nn.GELU()

    def forward(self, x, pos_embed):
        x = self.norm(x)

        qkv = self.attn_proj(x)

        head_size = self.config["hidden_size"] // self.config["n_heads"]

        qkv = qkv.reshape(x.shape[0], x.shape[1], 3, self.config["n_heads"], head_size)

        q = apply_rotary_pos_emb(qkv[:, :, 0, :], pos_embed)
        k = apply_rotary_pos_emb(qkv[:, :, 1, :], pos_embed)
        v = qkv[:, :, 2, :]

        attn = femr.models.flash_attention.flash_attention_wrapper(q, k, v, self.config["attention_width"])

        attn = attn.reshape(x.shape)

        ff = self.input_proj(x)
        ff = self.activate(ff)

        combined = torch.concatenate((attn, ff), axis=-1)
        result = self.output_proj(combined)
        return result


class FEMRTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.in_norm = femr.models.rmsnorm.RMSNorm(self.config["hidden_size"])
        self.out_norm = femr.models.rmsnorm.RMSNorm(self.config["hidden_size"])

        if not self.config.get("is_hierarchical"):
            self.embed = nn.Embedding(self.config["vocab_size"], self.config["hidden_size"])
        else:
            # Need to be using an embedding bag here
            assert False

        self.layers = nn.ModuleList([FEMREncoderLayer(config) for _ in range(self.config["n_layers"])])

    def forward(self, batch):
        x = self.embed(batch["tokens"])

        x = self.in_norm(x)
        pos_embed = fixed_pos_embedding(batch["ages"], self.config["hidden_size"] // self.config["n_heads"], x.dtype)

        for layer in self.layers:
            x = x + layer(x, pos_embed)

        return self.out_norm(x)


class FEMRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = FEMRTransformer(self.config["transformer"])
        if "task" in self.config:
            self.task_model = femr.models.tasks.create_task(
                self.config["transformer"]["hidden_size"], self.config["task"]
            )

    def __call__(self, batch):
        features = self.transformer(batch["transformer"])
        if batch["task"] is not None:
            features = sequence_data.reshape(-1, sequence_data.shape[-1])
            features = features[batch["transformer"]["label_indices"], :]
            return self.task_model(features, batch["task"])
        else:
            return features
