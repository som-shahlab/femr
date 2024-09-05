from __future__ import annotations

import bisect
import collections
import datetime
import functools
import math
import os
from typing import Any, Dict, Iterator, List, Mapping, Optional, Set, Tuple, Union

import meds_reader
import msgpack
import numpy as np
import transformers

import femr.ontology
import femr.stat_utils
import femr.pat_utils
import pyarrow as pa


def train_tokenizer(
    db: meds_reader.SubjectDatabase,
    vocab_size: int,
    num_numeric: int = 1000,
) -> FlatTokenizer:
    """Train a FEMR tokenizer from the given dataset"""

    statistics = functools.reduce(
        agg_statistics,
        db.map(
            functools.partial(
                map_statistics,
                num_subjects=len(db),
            )
        ),
    )

    return FlatTokenizer(
        convert_statistics_to_msgpack(statistics, vocab_size, num_numeric)
    )


def agg_statistics(stats1, stats2):
    stats1["age_stats"].combine(stats2["age_stats"])

    for n in ("code_counts", "text_counts"):
        for k, v in stats2[n].items():
            stats1[n][k] += v

    if stats1.get("numeric_samples_by_lab"):
        for k, v in stats2["numeric_samples_by_lab"].items():
            stats1["numeric_samples_by_lab"][k].combine(v)

    return stats1


def normalize_unit(unit):
    if unit:
        return unit.lower().replace(" ", "")
    else:
        return None

def is_close_float(t, f):
    if f is None:
        return False
    try:
        v = float(t)
        return math.abs(f - v) < 0.01 * f
    except:
        return False

def map_statistics(
    subjects: Iterator[meds_reader.Subject],
    *,
    num_subjects: int,
) -> Mapping[str, Any]:
    age_stats = femr.stat_utils.OnlineStatistics()
    code_counts: Dict[str, float] = collections.defaultdict(float)

    numeric_samples_by_lab = collections.defaultdict(functools.partial(femr.stat_utils.ReservoirSampler, 1_000))

    text_counts: Dict[Any, float] = collections.defaultdict(float)

    for subject in subjects:
        total_events = len(subject.events)

        if total_events == 0:
            continue

        weight = 1.0 / (num_subjects * total_events)
        birth_date = femr.pat_utils.get_subject_birthdate(subject)
        for event in subject.events:
            if event.time is not None and event.time != birth_date:
                age_stats.add(weight, (event.time - birth_date).total_seconds())

            assert numeric_samples_by_lab is not None
            if event.numeric_value is not None:
                numeric_samples_by_lab[event.code].add(event.numeric_value, weight)
            elif event.text_value is not None:
                text_counts[(event.code, event.text_value)] += weight
            else:
                code_counts[event.code] += weight

    return {
        "age_stats": age_stats,
        "code_counts": code_counts,
        "text_counts": text_counts,
        "numeric_samples_by_lab": numeric_samples_by_lab,
    }


def convert_statistics_to_msgpack(
    statistics, vocab_size: int, num_numeric: int,
):
    vocab = []

    for code, weight in statistics["code_counts"].items():
        entry = {
            "type": "code",
            "code_string": code,
            "weight": weight * math.log(weight) + (1 - weight) * math.log(1 - weight),
        }
        vocab.append(entry)

    for (code, text), weight in statistics["text_counts"].items():
        entry = {
            "type": "text",
            "code_string": code,
            "text_string": text,
            "weight": weight * math.log(weight) + (1 - weight) * math.log(1 - weight),
        }
        vocab.append(entry)

    for code, reservoir in statistics["numeric_samples_by_lab"].items():
        weight = reservoir.total_weight / 10
        samples = reservoir.samples
        samples.sort()

        samples_per_bin = (len(samples) + 9) // 10

        for bin_index in range(0, 10):
            if bin_index == 0:
                start_val = float("-inf")
            else:
                if bin_index * samples_per_bin >= len(samples):
                    continue
                start_val = samples[bin_index * samples_per_bin]

            if bin_index == 9 or (bin_index + 1) * samples_per_bin >= len(samples):
                end_val = float("inf")
            else:
                end_val = samples[(bin_index + 1) * samples_per_bin]

            if start_val == end_val:
                continue

        entry = {
            "type": "numeric",
            "code_string": code,
            "val_start": start_val,
            "val_end": end_val,
            "weight": weight * math.log(weight) + (1 - weight) * math.log(1 - weight),
        }
        vocab.append(entry)
 

    vocab.sort(key=lambda a: a["weight"])
    vocab = vocab[:vocab_size]

    result = {
        "vocab": vocab,
        "age_stats": {
            "mean": statistics["age_stats"].mean(),
            "std": statistics["age_stats"].standard_deviation(),
        },
    }

    return result


class FlatTokenizer(transformers.utils.PushToHubMixin):
    def __init__(self, dictionary: Mapping[str, Any], ontology: Optional[femr.ontology.Ontology] = None):
        self.dictionary = dictionary

        self.is_hierarchical = dictionary["is_hierarchical"]

        if self.is_hierarchical:
            assert ontology is not None

        self.ontology = ontology

        self.dictionary = dictionary
        vocab = dictionary["vocab"]

        self.string_lookup = {}
        self.code_lookup = {}

        self.vocab_size = len(vocab)

        self.numeric_lookup = collections.defaultdict(list)
        for i, dict_entry in enumerate(vocab):
            if dict_entry["type"] == "code":
                self.code_lookup[dict_entry["code_string"]] = i
            elif dict_entry["type"] == "numeric":
                self.numeric_lookup[dict_entry["code_string"]].append(
                    (dict_entry["val_start"], dict_entry["val_end"], i)
                )
            elif dict_entry["type"] == "text":
                self.string_lookup[(dict_entry["code_string"], dict_entry["text_string"])] = i
            else:
                pass

    @classmethod
    def from_pretrained(
        self,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ):
        """
        Load the FEMR tokenizer.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing tokenization data saved using
                      [`save_pretrained`], e.g., `./my_data_directory/`.
            ontology: An ontology object for hierarchical tokenizers
            kwargs: Arguments for loading to pass to transformers.utils.hub.cached_file

        Returns:
            A FEMR Tokenizer
        """

        dictionary_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path, "dictionary.msgpack", **kwargs
        )

        with open(dictionary_file, "rb") as f:
            dictionary = msgpack.load(f)

        return FlatTokenizer(dictionary)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save the FEMR tokenizer.


        This method make sure the batch processor can then be re-loaded using the
        .from_pretrained class method.

        Args:
            save_directory (`str` or `os.PathLike`): The path to a directory where the tokenizer will be saved.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`PushToHubMixin.push_to_hub`] method.
        """
        assert not os.path.isfile(save_directory), f"Provided path ({save_directory}) should be a directory, not a file"

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", str(save_directory).split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        with open(os.path.join(save_directory, "dictionary.msgpack"), "wb") as f:
            msgpack.dump(self.dictionary, f)

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

    def start_subject(self):
        """Compute per-subject statistics that are required to generate features."""

        # This is currently a null-op, but is required for cost featurization
        pass

    def get_feature_codes(self, event: meds_reader.Event) -> Tuple[List[int], Optional[List[float]]]:
        """Get codes for the provided measurement and time"""

        # Note that time is currently not used in this code, but it is required for cost featurization
        if event.numeric_value is not None:
            for start, end, i in self.numeric_lookup.get(event.code, []):
                if start <= event.numeric_value < end:
                    return [i], None
            else:
                return [], None
        elif event.text_value is not None:
            value = self.string_lookup.get((event.code, event.text_value))
            if value is not None:
                return [value], None
            else:
                return [], None
        else:
            value = self.code_lookup.get(event.code)
            if value is not None:
                return [value], None
            else:
                return [], None

    def normalize_age(self, age: datetime.timedelta) -> float:
        return (age.total_seconds() - self.dictionary["age_stats"]["mean"]) / (self.dictionary["age_stats"]["std"])
