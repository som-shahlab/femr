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
        """Defined a configuration for a FEMR Transformer.

        Arguments:
            vocab_size: The number of tokens in the vocabulary
            is_hierarchical: Whether to use a hierarchical vocabulary. See FEMRTokenizer for more information
            hidden_size: The internal representation size
            intermediate_size: The size of the FFN in the transformer layers
            n_heads: The number of attention heads
            n_layers: The number of transformer encoder layers
            attention_width: FEMR by default uses a local attention transformer with a width defined here
            use_normed_ages: Whether or not to provide normalized ages as a feature to the model
            use_bias: Whether or not to use bias terms in the transformer layers
            hidden_act: The type of activation function to use in the transformer
        """
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
        """A generic FEMR task definition. This holds state used for initalizing a tasks.py class.

        Task.get_task_config returns the task type and kwargs used to initialize this.

        Arguments:
            task_type: The name of the task.
            task_kwargs: Arbitrary arguments used to store state for that task.
        """
        super().__init__(**kwargs)
        self.task_type = task_type
        self.task_kwargs = task_kwargs


class FEMRModelConfig(transformers.PretrainedConfig):
    """A model config is defined as the combination of a transformer config and a task config."""

    def __init__(
        self,
        transformer_config: Optional[Dict[str, Any]] = None,
        task_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """A combination of a transformer config and a task config.

        It is possible to initialize this with only a transformer config, in which
        case the model will be configured for inference only.
        """
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
        """
        Combine a transformer configuration and task configuration into a model configuration.
        """
        if task_config is not None:
            task_config_dict = task_config.to_dict()
        else:
            task_config_dict = None

        return cls(transformer_config=transformer_config.to_dict(), task_config=task_config_dict)
