from __future__ import annotations

import bisect
import collections
import functools
import math
import datetime
import os
from typing import Any, Dict, Mapping, Optional, Set, Union

import msgpack
import numpy as np
import transformers

import femr.hf_utils
import femr.stat_utils
import meds


def train_tokenizer(
    dataset,
    vocab_size: int,
    is_hierarchical: bool = False,
    num_numeric: int = 1000,
    ontology: Optional[femr.ontology.Ontology] = None,
    num_proc: int = 1,
) -> FEMRTokenizer:
    """Train a FEMR tokenizer from the given dataset"""
    statistics = femr.hf_utils.aggregate_over_dataset(
        dataset,
        functools.partial(
            map_statistics, num_patients=len(dataset), is_hierarchical=is_hierarchical, ontology=ontology
        ),
        agg_statistics,
        num_proc=num_proc,
        batch_size=1_000,
    )
    return FEMRTokenizer(
        convert_statistics_to_msgpack(statistics, vocab_size, is_hierarchical, num_numeric, ontology), ontology
    )


def agg_statistics(stats1, stats2):
    stats1["age_stats"].combine(stats2["age_stats"])

    for n in ("code_counts", "text_counts"):
        for k, v in stats2[n].items():
            stats1[n][k] += v

    if stats1.get("numeric_samples"):
        stats1["numeric_samples"].combine(stats2["numeric_samples"])
    if stats1.get("numeric_samples_by_lab"):
        for k, v in stats2["numeric_samples_by_lab"].items():
            stats1["numeric_samples_by_lab"][k].combine(v)

    return stats1


def normalize_unit(unit):
    if unit:
        return unit.lower().replace(" ", "")
    else:
        return None


def map_statistics(
    batch, *, num_patients: int, is_hierarchical: bool, frac_values=0.05, ontology: Optional[femr.ontology.Ontology]
) -> Mapping[str, Any]:
    age_stats = femr.stat_utils.OnlineStatistics()
    code_counts: Dict[str, float] = collections.defaultdict(float)

    numeric_samples: Optional[femr.stat_utils.ReservoirSampler]
    numeric_samples_by_lab: Optional[Dict[str, femr.stat_utils.ReservoirSampler]]
    if is_hierarchical:
        assert ontology is not None
        numeric_samples = femr.stat_utils.ReservoirSampler(10_000)
        numeric_samples_by_lab = None
    else:
        numeric_samples = None
        numeric_samples_by_lab = collections.defaultdict(functools.partial(femr.stat_utils.ReservoirSampler, 1_000))

    text_counts: Dict[Any, float] = collections.defaultdict(float)

    for events in batch["events"]:
        total_events = 0
        for event in events:
            for measurement in event["measurements"]:
                total_events += 1

        if total_events == 0:
            continue

        weight = 1.0 / (num_patients * total_events)
        birth_date = events[0]["time"]
        code_set = set()
        text_set = set()
        pat_numeric_samples = []
        for event in events:
            for measurement in event["measurements"]:
                if measurement["code"] == meds.birth_code:
                    continue
                if event["time"] != birth_date:
                    age_stats.add(weight, (event["time"] - birth_date).total_seconds())
                if not is_hierarchical:
                    assert numeric_samples_by_lab is not None
                    if measurement["numeric_value"] is not None:
                        numeric_samples_by_lab[measurement["code"]].add(measurement["numeric_value"], weight)
                    elif measurement["text_value"] is not None:
                        text_counts[(measurement["code"], measurement["text_value"])] += weight
                    else:
                        code_counts[measurement["code"]] += weight
                else:
                    code_set.add(measurement["code"])

                    if measurement["text_value"] is not None and measurement["text_value"] != "":
                        text_set.add(measurement["text_value"])

                    if measurement.get("metadata") and normalize_unit(measurement["metadata"].get("unit")) is not None:
                        text_set.add(normalize_unit(measurement["metadata"]["unit"]))

                    if measurement["numeric_value"] is not None:
                        pat_numeric_samples.append(measurement["numeric_value"])

        if is_hierarchical:
            assert numeric_samples is not None
            assert ontology is not None
            final_codes: Set[str] = set()
            for code in code_set:
                final_codes |= ontology.get_all_parents(code)

            for code in final_codes:
                code_counts[code] += 1 / num_patients

            for text in text_set:
                text_counts[text] += 1 / num_patients

            for value in pat_numeric_samples:
                numeric_samples.add(value, 1 / (num_patients * len(pat_numeric_samples)))

    return {
        "age_stats": age_stats,
        "code_counts": code_counts,
        "text_counts": text_counts,
        "numeric_samples": numeric_samples,
        "numeric_samples_by_lab": numeric_samples_by_lab,
    }


def convert_statistics_to_msgpack(
    statistics, vocab_size: int, is_hierarchical: bool, num_numeric: int, ontology: Optional[femr.ontology.Ontology]
):
    vocab = []

    if not is_hierarchical:
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
    else:
        assert ontology
        for code, weight in statistics["code_counts"].items():
            baseline = min([1] + [statistics["code_counts"][parent] for parent in ontology.get_parents(code)])
            weight = weight / baseline

            weight = min(1, weight)

            if weight != 0 and weight != 1:
                entry = {
                    "type": "code",
                    "code_string": code,
                    "weight": baseline * (weight * math.log(weight) + (1 - weight) * math.log(1 - weight)),
                }
                vocab.append(entry)

        for text, weight in statistics["text_counts"].items():
            weight = min(1, weight)

            if weight != 0 and weight != 1:
                entry = {
                    "type": "text",
                    "text_string": text,
                    "weight": weight * math.log(weight) + (1 - weight) * math.log(1 - weight),
                }
                vocab.append(entry)

        numeric_samples = list(statistics["numeric_samples"].samples)

        if len(numeric_samples) > 0:
            assert num_numeric >= 1

            samples = sorted(list(set(np.percentile(numeric_samples, np.linspace(0, 1, num=num_numeric + 1)))))
            samples[0] = float("-inf")
            samples[-1] = float("inf")

            assert len(samples) >= 2

            for start_val, end_val in zip(samples, samples[1:]):
                entry = {
                    "type": "numeric",
                    "val_start": start_val,
                    "val_end": end_val,
                    "weight": -1,
                }
                vocab.append(entry)

    entry = {
        "type": "code",
        "code_string": meds.birth_code,
        "weight": -1,
    }
    vocab.append(entry)
    vocab.sort(key=lambda a: a["weight"])
    vocab = vocab[:vocab_size]

    result = {
        "vocab": vocab,
        "is_hierarchical": is_hierarchical,
        "age_stats": {
            "mean": statistics["age_stats"].mean(),
            "std": statistics["age_stats"].standard_deviation(),
        },
    }

    return result


class FEMRTokenizer(transformers.utils.PushToHubMixin):
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

        if not self.is_hierarchical:
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
        else:
            numeric_entries = []
            for i, dict_entry in enumerate(vocab):
                if dict_entry["type"] == "code":
                    if type(dict_entry["code_string"]) == list:
                        self.code_lookup[dict_entry["code_string"][0]] = i
                    else:
                        self.code_lookup[dict_entry["code_string"]] = i
                elif dict_entry["type"] == "numeric":
                    numeric_entries.append((dict_entry["val_start"], i))
                elif dict_entry["type"] == "text":
                    self.string_lookup[dict_entry["text_string"]] = i
                else:
                    pass
            numeric_entries.sort()
            if len(numeric_entries) > 0:
                self.numeric_values = [a[0] for a in numeric_entries[1:]]
                self.numeric_indices = [a[1] for a in numeric_entries]
            else:
                self.numeric_values = []
                self.numeric_indices = []

    @classmethod
    def from_pretrained(
        self,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        ontology: Optional[femr.ontology.Ontology] = None,
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

        return FEMRTokenizer(dictionary, ontology=ontology)

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

    def start_patient(self):
        pass

    def get_feature_codes(self, time: datetime.datetime, measurement: meds.Measurement):
        if self.is_hierarchical:
            codes = [
                self.code_lookup[parent]
                for parent in self.ontology.get_all_parents(measurement["code"])
                if parent in self.code_lookup
            ]
            weights = [1 / len(codes) for _ in codes]
            if measurement.get("metadata") and normalize_unit(measurement["metadata"].get("unit")) is not None:
                value = self.string_lookup.get(normalize_unit(measurement["metadata"]["unit"]))
                if value is not None:
                    codes.append(value)
                    weights.append(1)
            if measurement.get("numeric_value") is not None and len(self.numeric_indices) > 0:
                codes.append(self.numeric_indices[bisect.bisect(self.numeric_values, measurement["numeric_value"])])
                weights.append(1)
            if measurement.get("text_value") is not None:
                value = self.string_lookup.get(measurement["text_value"])
                if value is not None:
                    codes.append(value)
                    weights.append(1)

            return codes, weights
        else:
            if measurement.get("numeric_value") is not None:
                for start, end, i in self.numeric_lookup.get(measurement["code"], []):
                    if start <= measurement["numeric_value"] < end:
                        return [i], None
                else:
                    return [], None
            elif measurement.get("text_value") is not None:
                value = self.string_lookup.get((measurement["code"], measurement["text_value"]))
                if value is not None:
                    return [value], None
                else:
                    return [], None
            else:
                value = self.code_lookup.get(measurement["code"])
                if value is not None:
                    return [value], None
                else:
                    return [], None

    def normalize_age(self, age):
        return (age.total_seconds() - self.dictionary["age_stats"]["mean"]) / (self.dictionary["age_stats"]["std"])
