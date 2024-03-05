from __future__ import annotations

import datetime
import math
from typing import Any, Dict, TypeVar

import datasets
import flash_attn
import torch
import torch.nn.functional as F
import transformers
from einops import rearrange, repeat
from torch import nn

import femr.models.flash_attention
import femr.models.rmsnorm


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
    def __init__(self, config: FEMRTransformerConfig):
        super().__init__()
        self.config = config
        self.norm = femr.models.rmsnorm.RMSNorm(self.config.hidden_size)
        self.input_proj = nn.Linear(
            self.config.hidden_size, self.config.hidden_size * 3 + self.config.intermediate_size
        )
        self.output_proj = nn.Linear(self.config.hidden_size + self.config.intermediate_size, self.config.hidden_size)
        self.activate = nn.GELU()

    def forward(self, x, pos_embed):
        x = self.norm(x)

        transformed = self.input_proj(x)

        ff = transformed[:, :, -self.config.intermediate_size :]
        qkv = transformed[:, :, : -self.config.intermediate_size]

        head_size = self.config.hidden_size // self.config.n_heads

        qkv = qkv.reshape(x.shape[0], x.shape[1], 3, self.config.n_heads, head_size)

        q = apply_rotary_pos_emb(qkv[:, :, 0, :, :], pos_embed)
        k = apply_rotary_pos_emb(qkv[:, :, 1, :, :], pos_embed)
        v = qkv[:, :, 2, :, :]

        attn = femr.models.flash_attention.flash_attention_wrapper(q, k, v, self.config.attention_width)

        attn = attn.reshape(x.shape)

        ff = self.activate(ff)

        combined = torch.concatenate((attn, ff), axis=-1)
        result = self.output_proj(combined)

        return result


class FEMRTransformerConfig(transformers.PretrainedConfig):
    def __init__(
        self,
        vocab_size: int = 32768,
        is_hierarchical: bool = False,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        n_heads: int = 12,
        n_layers: int = 6,
        attention_width: int = 496,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.is_hierarchical = is_hierarchical

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.attention_width = attention_width


class FEMRTransformer(nn.Module):
    def __init__(self, config: FEMRTransformerConfig):
        super().__init__()
        self.config = config

        self.in_norm = femr.models.rmsnorm.RMSNorm(self.config.hidden_size)
        self.out_norm = femr.models.rmsnorm.RMSNorm(self.config.hidden_size)

        if not self.config.is_hierarchical:
            self.embed = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        else:
            self.embed = nn.EmbeddingBag(self.config.vocab_size, self.config.hidden_size)

        self.layers = nn.ModuleList([FEMREncoderLayer(config) for _ in range(self.config.n_layers)])

    def forward(self, batch):
        if not self.config.is_hierarchical:
            x = self.embed(batch["tokens"])
        else:
            embedded = self.embed(batch["hier_tokens"], batch["hier_token_indices"])
            x = embedded[batch['hier_token_offsets'], :]

        x = self.in_norm(x)
        pos_embed = fixed_pos_embedding(batch["ages"], self.config.hidden_size // self.config.n_heads, x.dtype)

        for layer in self.layers:
            x = x + layer(x, pos_embed)

        final = self.out_norm(x)

        return final


class LabeledPatientTaskHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

    def forward(self, features: torch.Tensor, batch: Mapping[str, torch.Tensor]):
        return (
            batch["patient_ids"],
            batch["prediction_timestamps"].cpu().numpy().astype("datetime64[s]").astype(datetime.datetime),
            features,
        )


class CLMBRTaskHead(nn.Module):
    def __init__(self, hidden_size: int, clmbr_vocab_size: int, **kwargs):
        super().__init__()

        self.final_layer = nn.Linear(hidden_size, clmbr_vocab_size)

    def forward(self, features: torch.Tensor, batch: Mapping[str, torch.Tensor]):
        logits = self.final_layer(features)
        labels = batch["labels"]
        loss = F.cross_entropy(logits, labels)

        return loss, logits


class FEMRTaskConfig(transformers.PretrainedConfig):
    def __init__(self, task_type: str = "", **kwargs):
        super().__init__(**kwargs)
        self.task_type = task_type
        self.kwargs = kwargs

    def create_task_head(self, hidden_size: int):
        if self.task_type == "clmbr":
            return CLMBRTaskHead(hidden_size, **self.kwargs)
        elif self.task_type == "labeled_patients":
            return LabeledPatientTaskHead(hidden_size, **self.kwargs)


class FEMRModelConfig(transformers.PretrainedConfig):
    def __init__(self, transformer_config: Dict[str, Any] = None, task_config: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if transformer_config is None:
            transformer_config = {}
        self.transformer_config = FEMRTransformerConfig(**transformer_config)

        if task_config is not None:
            self.task_config = FEMRTaskConfig(**task_config)
        else:
            self.task_config = None

    @classmethod
    def from_transformer_task_configs(
        cls, transformer_config: FEMRTransformerConfig, task_config: FEMRTaskConfig
    ) -> FEMRModelConfig:
        if task_config is not None:
            task_config_dict = task_config.to_dict()
        else:
            task_config_dict = None

        return cls(transformer_config=transformer_config.to_dict(), task_config=task_config_dict)


class FEMRModel(transformers.PreTrainedModel):
    config_class = FEMRModelConfig

    def __init__(self, config: FEMRModelConfig, **kwargs):
        # Allow the task config to be ovewritten
        if "task_config" in kwargs:
            config.task_config = kwargs["task_config"]

        super().__init__(config)

        self.transformer = FEMRTransformer(self.config.transformer_config)
        if self.config.task_config is not None:
            self.task_model = self.config.task_config.create_task_head(self.config.transformer_config.hidden_size)

    def forward(self, batch: Mapping[str, Any]):
        features = self.transformer(batch["transformer"])
        if "task" in batch:
            features = features.reshape(-1, features.shape[-1])
            features = features[batch["transformer"]["label_indices"], :]
            return self.task_model(features, batch["task"])
        else:
            return (
                batch["patient_ids"],
                batch["transformer"]["timestamps"].cpu().numpy().astype("datetime64[s]").astype(datetime.datetime),
                features,
            )
