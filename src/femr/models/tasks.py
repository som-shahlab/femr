import abc
import datetime
from typing import Any, List, Mapping, Sequence

import event_stream_data_standard as ESDS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Task(nn.Module, abc.ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def start_batch(self) -> None:
        ...

    @abc.abstractmethod
    def start_patient(self, patient: ESDS.Patient) -> None:
        ...

    @abc.abstractmethod
    def needs_exact(self) -> bool:
        ...

    @abc.abstractmethod
    def add_event(
        self, current_date: datetime.datetime, next_date: datetime.datetime, next_features: Sequence[int]
    ) -> bool:
        ...

    @abc.abstractmethod
    def get_batch_data(self) -> Mapping[str, np.ndarray]:
        ...


class CLMBRTask(Task):
    def __init__(self, clmbr_vocab_size: int, *, hidden_size: int = 1):
        config = {"type": "clmbr", "clmbr_vocab_size": clmbr_vocab_size}

        super().__init__(config)

        self.final_layer = nn.Linear(hidden_size, self.config["clmbr_vocab_size"])

    def start_patient(self, patient: ESDS.Patient) -> None:
        pass

    def needs_exact(self) -> bool:
        return False

    def start_batch(self) -> None:
        self.batch_labels: List[int] = []

    def add_event(
        self, current_date: datetime.datetime, next_date: datetime.datetime, next_features: Sequence[int]
    ) -> bool:
        if len(next_features) == 0:
            return False

        if len(next_features) != 1:
            raise RuntimeError("Only supports one for right now")

        next_feature = next_features[0]

        if next_feature >= self.config["clmbr_vocab_size"]:
            return False

        self.batch_labels.append(next_feature)

        return True

    def get_batch_data(self) -> Mapping[str, np.ndarray]:
        return {"labels": np.array(self.batch_labels, dtype=np.int32)}

    def forward(self, features: torch.Tensor, batch: Mapping[str, torch.Tensor]):
        logits = self.final_layer(features)
        labels = batch["labels"]
        loss = F.cross_entropy(logits, labels)

        return loss, logits


task_mapping = {
    "clmbr": CLMBRTask,
}


def create_task(hidden_size: int, config: Mapping[str, Any]) -> Task:
    stripped_config = dict(config)
    del stripped_config["type"]
    assert config["type"] in task_mapping, f"{config['type']} not currently supported"

    task = task_mapping[config["type"]]

    return task(**stripped_config, hidden_size=hidden_size)
