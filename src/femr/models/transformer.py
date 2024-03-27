from __future__ import annotations

import collections
import datetime
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import datasets
import meds
import numpy as np
import torch
import torch.nn.functional as F
import transformers
import xformers.ops
from torch import nn

import femr.models.processor
import femr.models.rmsnorm
import femr.models.tasks
import femr.models.tokenizer
import femr.models.xformers


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
    assert len(ages.shape) == 1

    inv_freq = 1.0 / (10000 ** (torch.linspace(0, 2, steps=dim // 2, device=ages.device)))
    inv_freq = inv_freq.reshape(1, 1, dim // 2)
    assert inv_freq.dtype == torch.float32

    ages = ages.reshape(ages.shape[0], 1)

    t = inv_freq * ages

    sin, cos = torch.sin(t), torch.cos(t)

    final_shape = (ages.shape[0], 1, dim)

    sin = torch.stack((sin, sin), axis=-1).reshape(final_shape).type(dtype)
    cos = torch.stack((cos, cos), axis=-1).reshape(final_shape).type(dtype)

    return sin, cos


def apply_rotary_pos_emb(x, sincos):
    sin, cos = sincos
    sin = sin.to(dtype=x.dtype)
    cos = cos.to(dtype=x.dtype)

    assert x.dtype == sin.dtype == cos.dtype, f"{x.dtype} {sin.dtype} {cos.dtype}"

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
        if self.config.hidden_act == "swiglu":
            hidden_mult = 2
        else:
            hidden_mult = 1

        self.input_proj = nn.Linear(
            self.config.hidden_size,
            self.config.hidden_size * 3 + hidden_mult * self.config.intermediate_size,
            bias=self.config.use_bias,
        )
        self.output_proj = nn.Linear(
            self.config.hidden_size + self.config.intermediate_size, self.config.hidden_size, bias=self.config.use_bias
        )

    def forward(self, x, normed_ages, pos_embed, attn_bias):
        x = self.norm(x)

        if self.config.use_normed_ages:
            x[:, -2] = normed_ages.to(dtype=x.dtype)
            x[:, -1] = (normed_ages**2).to(dtype=x.dtype)

        transformed = self.input_proj(x)

        ff = transformed[:, : -self.config.hidden_size * 3]
        qkv = transformed[:, -self.config.hidden_size * 3 :]

        head_size = self.config.hidden_size // self.config.n_heads

        qkv = qkv.reshape(x.shape[0], 3, self.config.n_heads, head_size)

        q = apply_rotary_pos_emb(qkv[:, 0, :, :], pos_embed)
        k = apply_rotary_pos_emb(qkv[:, 1, :, :], pos_embed)
        v = qkv[:, 2, :, :]

        attn = femr.models.xformers.memory_efficient_attention_wrapper(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            attn_bias=attn_bias,
        )

        attn = attn.reshape(x.shape)

        if self.config.hidden_act == "gelu":
            ff = F.gelu(ff)
        elif self.config.hidden_act == "swiglu":
            x1, x2 = ff.chunk(2, dim=-1)
            ff = F.silu(x1) * x2

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
        use_normed_ages: bool = False,
        use_bias: bool = True,
        hidden_act: str = "gelu",
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

        self.use_normed_ages = use_normed_ages

        self.use_bias = use_bias
        self.hidden_act = hidden_act


class FEMRTransformer(nn.Module):
    def __init__(self, config: FEMRTransformerConfig):
        super().__init__()
        self.config = config

        self.in_norm = femr.models.rmsnorm.RMSNorm(self.config.hidden_size)
        self.out_norm = femr.models.rmsnorm.RMSNorm(self.config.hidden_size)

        if not self.config.is_hierarchical:
            self.embed = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        else:
            self.embed_bag = nn.EmbeddingBag(
                num_embeddings=self.config.vocab_size,
                embedding_dim=self.config.hidden_size,
                mode="sum",
                include_last_offset=True,
            )

        self.layers = nn.ModuleList([FEMREncoderLayer(config) for _ in range(self.config.n_layers)])

    def forward(self, batch):
        """
        TODO: needs some documentation
        
        - what is the expected shape of batch["tokens"]?
        - '' '' '' batch["normalized_ages"], etc...
        """
        if not self.config.is_hierarchical:
            x = self.embed(batch["tokens"])
        else:
            x = self.embed_bag(batch["hierarchical_tokens"], batch["token_indices"], batch["hierarchical_weights"])

        x = self.in_norm(x)
        normed_ages = batch["normalized_ages"]
        pos_embed = fixed_pos_embedding(batch["ages"], self.config.hidden_size // self.config.n_heads, x.dtype)

        attn_bias = xformers.ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
            batch["patient_lengths"].tolist()
        ).make_local_attention(self.config.attention_width)

        for layer in self.layers:
            x = x + layer(x, normed_ages, pos_embed, attn_bias)

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
    def __init__(self, hidden_size: int, clmbr_vocab_size: int):
        super().__init__()

        self.final_layer = nn.Linear(hidden_size, clmbr_vocab_size)

    def forward(self, features: torch.Tensor, batch: Mapping[str, torch.Tensor]):
        logits = self.final_layer(features)
        labels = batch["labels"]
        loss = F.cross_entropy(logits, labels)

        return loss, logits


class MOTORTaskHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        pretraining_task_info: List[Tuple[str, float]],
        time_bins: List[float],
        final_layer_size: int,
    ):
        super().__init__()

        self.num_time_bins = len(time_bins) - 1
        self.num_tasks = len(pretraining_task_info)

        self.final_layer_size = final_layer_size
        self.final_layer = nn.Linear(hidden_size, self.num_time_bins * final_layer_size)

        self.task_layer = nn.Linear(self.final_layer_size, self.num_tasks)
        start_bias = torch.log2(torch.tensor([a[1] for a in pretraining_task_info], dtype=torch.float32))
        self.task_layer.bias.data = start_bias

    def forward(self, features: torch.Tensor, batch: Mapping[str, torch.Tensor], return_logits=False):
        time_independent_features = self.final_layer(features).reshape(
            features.shape[0], self.num_time_bins, self.final_layer_size
        )

        time_dependent_logits = self.task_layer(time_independent_features)

        assert (
            batch["log_time"].shape == time_dependent_logits.shape
        ), f"{time_dependent_logits.shape} {batch['log_time'].shape}"
        assert (
            batch["is_event"].shape == time_dependent_logits.shape
        ), f"{time_dependent_logits.shape} {batch['is_event'].shape}"

        survival_loss = torch.exp2(time_dependent_logits + batch["log_time"]).mean()
        event_loss = -math.log(2) * torch.where(batch["is_event"], time_dependent_logits, 0).mean()

        loss = survival_loss + event_loss

        if not return_logits:
            time_dependent_logits = None

        return loss, time_dependent_logits


class FEMRTaskConfig(transformers.PretrainedConfig):
    def __init__(self, task_type: str = "", task_kwargs: Mapping[str, Any] = {}, **kwargs):
        super().__init__(**kwargs)
        self.task_type = task_type
        self.task_kwargs = task_kwargs

    def create_task_head(self, hidden_size: int):
        if self.task_type == "clmbr":
            return CLMBRTaskHead(hidden_size, **self.task_kwargs)
        elif self.task_type == "labeled_patients":
            return LabeledPatientTaskHead(hidden_size, **self.task_kwargs)
        elif self.task_type == "motor":
            return MOTORTaskHead(hidden_size, **self.task_kwargs)


class FEMRModelConfig(transformers.PretrainedConfig):
    def __init__(
        self,
        transformer_config: Optional[Dict[str, Any]] = None,
        task_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if transformer_config is None:
            transformer_config = {}
        self.transformer_config = FEMRTransformerConfig(**transformer_config)

        self.task_config: Optional[FEMRTaskConfig]

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


def remove_first_dimension(data: Any) -> Any:
    if isinstance(data, collections.abc.Mapping):
        return {k: remove_first_dimension(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        assert data.shape[0] == 1
        return data.squeeze(dim=0)
    elif isinstance(data, np.ndarray):
        assert data.shape[0] == 1
        return np.squeeze(data, axis=0)
    elif isinstance(data, (int, float, np.number, np.bool_)):
        return data
    else:
        raise RuntimeError("Could not convert item of type " + str(type(data)))


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

    def forward(self, batch: Mapping[str, Any], return_loss=True):
        # Need a return_loss parameter for transformers.Trainer to work properly
        assert return_loss

        batch = remove_first_dimension(batch)

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


def compute_features(
    dataset: datasets.Dataset,
    model_path: str,
    labels: List[meds.Label],
    num_proc: int = 1,
    tokens_per_batch: int = 1024,
    device: Optional[torch.device] = None,
    ontology: Optional[femr.ontology.Ontology] = None,
) -> Dict[str, np.ndarray]:
    task = femr.models.tasks.LabeledPatientTask(labels)

    index = femr.index.PatientIndex(dataset, num_proc=num_proc)

    model = femr.models.transformer.FEMRModel.from_pretrained(model_path, task_config=task.get_task_config())
    tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained(model_path, ontology=ontology)
    processor = femr.models.processor.FEMRBatchProcessor(tokenizer, task=task)

    # get a dataset that only includes patients with labels
    filtered_data = task.filter_dataset(dataset, index)

    # BUG fix: if device = 0 then model was not being moved to device
    model = model.to(device)

    batches = processor.convert_dataset(
        filtered_data, tokens_per_batch=tokens_per_batch, min_samples_per_batch=1, num_proc=num_proc
    )

    # BUG: possible bug around here... batch being moved to device and then used with numpy functions?
    # issue: https://github.com/som-shahlab/femr/issues/184
    batches.set_format("pt", device=None)

    all_patient_ids = []
    all_feature_times = []
    all_representations = []

    # num_feature_times_processed = 0
    for batch in batches:
        # print(f"num feature times processed: {num_feature_times_processed}")
        # print(f"batch (len={len(batch)}):")
        # # print(batch)
        batch = processor.collate([batch])["batch"] #.to(device)
        with torch.no_grad():
            patient_ids, feature_times, representations = model(batch)
            all_patient_ids.append(patient_ids.cpu().numpy())
            all_feature_times.append(feature_times)
            all_representations.append(representations.cpu().numpy())
        # num_feature_times_processed += len(feature_times)
        # print(f"patient ids: {patient_ids}")
        # print(f"feature_times: {feature_times}")

    return {
        "patient_ids": np.concatenate(all_patient_ids),
        "feature_times": np.concatenate(all_feature_times),
        "features": np.concatenate(all_representations),
    }
