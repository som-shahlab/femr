from __future__ import annotations

import collections
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import meds
import meds_reader
import numpy as np
import torch
import torch.nn.functional as F
import transformers
import xformers.ops
from torch import nn
from tqdm import tqdm

import femr.models.config
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
    def __init__(self, config: femr.models.config.FEMRTransformerConfig, use_hawk=False):
        super().__init__()
        self.config = config
        self.use_hawk = use_hawk

        self.norm = femr.models.rmsnorm.RMSNorm(self.config.hidden_size)
        if self.config.hidden_act == "swiglu":
            hidden_mult = 2
        else:
            hidden_mult = 1

        if self.use_hawk:
            import torch_hawk

            self.input_proj = nn.Linear(
                self.config.hidden_size,
                hidden_mult * self.config.intermediate_size,
                bias=self.config.use_bias,
            )
            self.hawk_module = torch_hawk.RecurrentBlock(self.config.hidden_size, num_heads=self.config.n_heads)
        else:
            self.input_proj = nn.Linear(
                self.config.hidden_size,
                self.config.hidden_size * 3 + hidden_mult * self.config.intermediate_size,
                bias=self.config.use_bias,
            )

        self.output_proj = nn.Linear(
            self.config.hidden_size + self.config.intermediate_size, self.config.hidden_size, bias=self.config.use_bias
        )

    def forward(self, x, time_data, pos_embed, attn_bias, s):
        x = self.norm(x)

        if self.config.use_normed_ages:
            if self.use_hawk:
                all_time = time_data
            else:
                all_time = torch.concatenate((time_data, time_data**2), axis=-1)

            x[:, -all_time.shape[1] :] = all_time.to(dtype=x.dtype)

        transformed = self.input_proj(x)

        if self.use_hawk:
            attn = self.hawk_module(x, s)
            ff = transformed
        else:

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


class FEMRTransformer(nn.Module):
    def __init__(self, config: femr.models.config.FEMRTransformerConfig):
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

        self.layers = nn.ModuleList([FEMREncoderLayer(config, use_hawk=False) for i in range(self.config.n_layers)])

    def forward(self, batch, s):
        if not self.config.is_hierarchical:
            x = self.embed(batch["tokens"])
        else:
            x = self.embed_bag(batch["hierarchical_tokens"], batch["token_indices"], batch["hierarchical_weights"])

        x = self.in_norm(x)
        time_data = batch["time_data"]
        pos_embed = fixed_pos_embedding(batch["ages"], self.config.hidden_size // self.config.n_heads, x.dtype)

        attn_bias = xformers.ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
            batch["subject_lengths"].tolist()
        ).make_local_attention(self.config.attention_width)

        for layer in self.layers:
            x = x + layer(x, time_data, pos_embed, attn_bias, s)

        final = self.out_norm(x)

        return final


class LabeledSubjectTaskHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

    def forward(self, features: torch.Tensor, batch: Mapping[str, torch.Tensor], return_logits=False):
        return 0, {}


class CLMBRTaskHead(nn.Module):
    def __init__(self, hidden_size: int, clmbr_vocab_size: int):
        super().__init__()

        self.final_layer = nn.Linear(hidden_size, clmbr_vocab_size)

    def forward(self, features: torch.Tensor, batch: Mapping[str, torch.Tensor], return_logits=False):
        logits = self.final_layer(features)
        labels = batch["labels"]
        loss = F.cross_entropy(logits, labels)

        if not return_logits:
            logits = None

        return loss, {"logits": logits}


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

        self.task_time_bias = nn.Parameter(torch.zeros(1, self.num_time_bins, self.num_tasks))

        self.norm = femr.models.rmsnorm.RMSNorm(self.final_layer_size)

    def forward(self, features: torch.Tensor, batch: Mapping[str, torch.Tensor], return_logits=False):
        time_independent_features = self.final_layer(features).reshape(
            features.shape[0], self.num_time_bins, self.final_layer_size
        )

        time_dependent_logits = self.task_layer(self.norm(time_independent_features)) + self.task_time_bias
        # time_dependent_logits = self.task_layer(time_independent_features)

        assert (
            batch["log_time"].shape == time_dependent_logits.shape
        ), f"{time_dependent_logits.shape} {batch['log_time'].shape}"
        assert (
            batch["is_event"].shape == time_dependent_logits.shape
        ), f"{time_dependent_logits.shape} {batch['is_event'].shape}"

        # Force to always be negative
        # time_dependent_logits = -F.softplus(-time_dependent_logits)

        survival_loss = torch.exp2(time_dependent_logits + batch["log_time"]).mean()
        event_loss = -math.log(2) * torch.where(batch["is_event"], time_dependent_logits, 0).mean()

        # with torch.autocast(device_type="cuda", enabled=False):
        #     actual = torch.exp2(time_dependent_logits.type(torch.float32) + batch["log_time"].type(torch.float32))
        #     bad = torch.exp2(time_dependent_logits.type(torch.bfloat16) + batch["log_time"].type(torch.bfloat16)).type(torch.float32)

        #     bias = self.task_layer.bias.reshape(1, 1, -1)
        #     better = torch.exp2((time_dependent_logits - bias).type(torch.bfloat16) + (bias + batch["log_time"]).type(torch.bfloat16)).type(torch.float32)

        #     bad_error = torch.mean((actual - bad) **2)
        #     better_error = torch.mean((actual - better) **2)
        #     var = torch.var(actual)

        #     print(bad_error / var, better_error / var)

        def stats(a):
            a = a[torch.isfinite(a)]
            print(torch.mean(a), torch.std(a), torch.max(a), torch.min(a))

        # print(survival_loss, event_loss)
        # print(features.dtype, (time_dependent_logits + batch["log_time"]).dtype)
        # print(time_dependent_logits + batch["log_time"])
        # stats(batch["log_time"])
        # stats(self.task_layer.bias.reshape(1, 1, -1) + batch["log_time"])
        # print(self.task_layer.bias.reshape(1, 1, -1) + batch["log_time"])
        # print(self.task_layer.bias)
        # print(time_dependent_logits - self.task_layer.bias.reshape(1, 1, -1))

        # total_loss = time_dependent_logits + batch["log_time"]
        # max_loss = torch.max(total_loss)

        # max_location = torch.where(total_loss == max_loss)

        # print(max_loss, max_location)

        # # max_location[0][0] = 532

        # stats(features[max_location[0], :])
        # stats(time_independent_features[max_location[0], max_location[1], :])

        # stats(features[532, :])
        # stats(time_independent_features[532, max_location[1], :])

        # print("Log time", batch["log_time"][max_location])
        # print("Logits", time_dependent_logits[max_location])
        # print("Bias", self.task_layer.bias[max_location[2]][0])

        # # # print(batch["log_time"][max_location])

        # # print(features[max_location[0], :])
        # # print(time_independent_features[max_location[0], max_location[1], :])
        # # print(self.task_layer.weight[max_location[2], :])
        # print(self.task_layer.weight.shape)

        # task_vector = self.task_layer.weight[max_location[2], :].reshape(-1).type(torch.float32)
        # feature_vector = time_independent_features[max_location[0], max_location[1], :].reshape(-1).type(torch.float32)

        # assert task_vector.shape == feature_vector.shape

        # print("Recompute", (task_vector * feature_vector).sum() + self.task_layer.bias[max_location[2]][0])

        # features[max_location[0], :] = 0
        # time_independent_features[max_location[0], max_location[1], :] = 0

        # stats(features)
        # stats(time_independent_features)

        # time_dependent_logits[max_location[0], max_location[1], :] = 0

        # stats(time_dependent_logits)

        # print(time_dependent_logits - self.task_layer.bias.unsqueeze(0).unsqueeze(0))

        # print(batch["log_time"])
        # print(self.task_layer.bias.unsqueeze(0).unsqueeze(0) + batch["log_time"])

        loss = survival_loss + event_loss

        if not return_logits:
            time_dependent_logits = None

        return loss, {"time_dependent_logits": time_dependent_logits}


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
    config_class = femr.models.config.FEMRModelConfig

    def __init__(self, config: femr.models.config.FEMRModelConfig, **kwargs):
        # Allow the task config to be ovewritten
        if "task_config" in kwargs:
            config.task_config = kwargs["task_config"]

        super().__init__(config)

        if isinstance(self.config, femr.models.config.FEMRTransformerConfig):
            self.transformer = FEMRTransformer(self.config)
        else:
            self.transformer = FEMRTransformer(self.config.transformer_config)

        if "task_config" in self.config and self.config.task_config is not None:
            self.task_model = self.create_task_head()

    def create_task_head(self) -> nn.Module:
        hidden_size = self.config.transformer_config.hidden_size
        task_type = self.config.task_config.task_type
        task_kwargs = self.config.task_config.task_kwargs
        if task_type == "clmbr":
            return CLMBRTaskHead(hidden_size, **task_kwargs)
        elif task_type == "labeled_subjects":
            return LabeledSubjectTaskHead(hidden_size, **task_kwargs)
        elif task_type == "motor":
            return MOTORTaskHead(hidden_size, **task_kwargs)
        else:
            raise RuntimeError("Could not determine head for task " + task_type)

    def forward(self, batch: Mapping[str, Any], return_loss=True, return_logits=False, return_reprs=False):
        # Need a return_loss parameter for transformers.Trainer to work properly
        assert return_loss

        batch = remove_first_dimension(batch)

        s = torch.zeros_like(batch["subject_ids"])
        s[1:] = batch["subject_ids"][1:] != batch["subject_ids"][:-1]
        s = torch.cumsum(s, dim=0).type(torch.uint8)

        features = self.transformer(batch["transformer"], s)
        if "task" in batch and self.config.task_config is not None:
            features = features.reshape(-1, features.shape[-1])
            features = features[batch["transformer"]["label_indices"], :]
            loss, result = self.task_model(features, batch["task"], return_logits=return_logits)
            if return_reprs:
                result["representations"] = features
            if return_logits or return_reprs:
                result["timestamps"] = batch["transformer"]["timestamps"][batch["transformer"]["label_indices"]]
                result["subject_ids"] = batch["subject_ids"][batch["transformer"]["label_indices"]]
            return loss, result
        else:
            loss = 0
            features = features.reshape(-1, features.shape[-1])
            if "task" in batch:
                features = features[batch["transformer"]["label_indices"], :]
                result = {
                    "timestamps": batch["transformer"]["timestamps"][batch["transformer"]["label_indices"]],
                    "subject_ids": batch["subject_ids"][batch["transformer"]["label_indices"]],
                    "representations": features,
                }
            else:
                result = {
                    "timestamps": batch["transformer"]["timestamps"],
                    "subject_ids": batch["subject_ids"],
                    "representations": features,
                }

            return loss, result


def to_device(data: Any, device: torch.device) -> Any:
    if isinstance(data, collections.abc.Mapping):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (int, float, np.number, np.bool_)):
        return data
    else:
        raise RuntimeError("Could not move item of type " + str(type(data)))


def compute_features(
    db: meds_reader.SubjectDatabase,
    model_path: str,
    labels: Optional[List[meds.Label]] = None,
    num_proc: int = 1,
    tokens_per_batch: int = 1024,
    device: Optional[torch.device] = None,
    ontology: Optional[femr.ontology.Ontology] = None,
) -> Dict[str, np.ndarray]:
    """ "Compute features for a set of labels given a dataset and a model.

    Arguments:
        dataset: A HuggingFace dataset containing MEDS subjects
        model_path: A path to a saved pretrained model, including a saved tokenizer
        labels: MEDS labels to compute features for
        num_proc: The number of processors to use
        tokens_per_batch: The maximum number of tokens per batch
        device: Which type of compute to use
        ontology: A FEMR ontology object, which is necessary for models that use a hierarchical tokenizer

    Returns:
        A dictionary of numpy arrays, with three keys, "subject_ids", "feature_times" and "features"
         -  "subject_ids" and "feature_times" define the subject and time each feature refers to
         -  "features" provides the representations at each subject id and feature time
    """

    if labels is None:
        task = None
        filtered_data = db
    else:
        task = femr.models.tasks.LabeledSubjectTask(labels)
        filtered_data = db.filter(list(task.label_map.keys()))

    model = femr.models.transformer.FEMRModel.from_pretrained(
        model_path, task_config=femr.models.config.FEMRTaskConfig(task_type="labeled_subjects")
    )
    tokenizer = femr.models.tokenizer.HierarchicalTokenizer.from_pretrained(model_path, ontology=ontology)
    processor = femr.models.processor.FEMRBatchProcessor(tokenizer, task=task)

    if device:
        model = model.to(device)

    cpu_device = torch.device("cpu")

    batches = processor.convert_dataset(
        filtered_data, tokens_per_batch=tokens_per_batch, min_subjects_per_batch=1, num_proc=num_proc
    )

    batches.set_format("pt")

    loader = torch.utils.data.DataLoader(batches, num_workers=num_proc, pin_memory=True, collate_fn=processor.collate)

    all_subject_ids = []
    all_feature_times = []
    all_representations = []

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for batch in tqdm(loader):
                if device:
                    batch = to_device(batch, device)
                _, result = model(**batch, return_reprs=True)
                all_subject_ids.append(result["subject_ids"].to(cpu_device, non_blocking=True))
                all_feature_times.append(result["timestamps"].to(cpu_device, non_blocking=True))
                all_representations.append(result["representations"].to(cpu_device, non_blocking=True))

    torch.cuda.synchronize()

    all_subject_ids_np = torch.concatenate(all_subject_ids).numpy()
    all_feature_times_np = torch.concatenate(all_feature_times).numpy()
    all_representations_np = torch.concatenate(all_representations).numpy()

    return {
        "subject_ids": all_subject_ids_np,
        "feature_times": all_feature_times_np.astype("datetime64[s]"),
        "features": all_representations_np,
    }
