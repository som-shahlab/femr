from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import transformers


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


class FEMRTaskConfig(transformers.PretrainedConfig):
    def __init__(self, task_type: str = "", task_kwargs: Mapping[str, Any] = {}, **kwargs):
        super().__init__(**kwargs)
        self.task_type = task_type
        self.task_kwargs = task_kwargs


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
