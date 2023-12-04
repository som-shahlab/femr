from __future__ import annotations

import collections
import datetime
import functools
import math
import os

import datasets
import msgpack
import transformers

import femr.hf_utils
import femr.stat_utils


def train_tokenizer(dataset, vocab_size, is_hierarchical=False, num_proc=1) -> FEMRTokenizer:
    """Train a FEMR tokenizer from the given dataset"""
    statistics = femr.hf_utils.aggregate_over_dataset(
        dataset,
        functools.partial(map_statistics, num_patients=len(dataset)),
        agg_statistics,
        num_proc=num_proc,
        batch_size=1_000,
    )
    return FEMRTokenizer(convert_statistics_to_msgpack(statistics, vocab_size, is_hierarchical))


def agg_statistics(stats1, stats2):
    stats1["age_stats"].combine(stats2["age_stats"])

    for n in ("code_counts", "hierarchical_code_counts", "text_counts"):
        for k, v in stats2[n].items():
            stats1[n][k] += v

    for k, v in stats2["numeric_samples"].items():
        stats1["numeric_samples"][k].combine(v)

    return stats1


def map_statistics(batch, *, num_patients):
    age_stats = femr.stat_utils.OnlineStatistics()
    code_counts = collections.defaultdict(float)
    hierarchical_code_counts = collections.defaultdict(float)

    text_counts = collections.defaultdict(float)
    numeric_samples = collections.defaultdict(functools.partial(femr.stat_utils.ReservoirSampler, 1_000))

    for patient_id, events in zip(batch["patient_id"], batch["events"]):
        total_events = 0
        for event in events:
            for measurement in event["measurements"]:
                total_events += 1

        if total_events == 0:
            continue

        weight = 1.0 / (num_patients * total_events)
        birth_date = events[0]["time"]
        for event in events:
            for measurement in event["measurements"]:
                age_stats.add(weight, (event["time"] - birth_date) / datetime.timedelta(minutes=1))
                if measurement["numeric_value"] is not None:
                    numeric_samples[measurement["code"]].add(measurement["numeric_value"], weight)
                elif measurement["text_value"] is not None:
                    text_counts[(measurement["code"], measurement["text_value"])] += weight
                else:
                    code_counts[measurement["code"]] += weight

    return {
        "age_stats": age_stats,
        "code_counts": code_counts,
        "hierarchical_code_counts": hierarchical_code_counts,
        "text_counts": text_counts,
        "numeric_samples": numeric_samples,
    }


def convert_statistics_to_msgpack(statistics, vocab_size, is_hierarchical):
    vocab = []

    if is_hierarchical:
        assert False, "not implemented yet"
    else:
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

    for code, reservoir in statistics["numeric_samples"].items():
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
        "is_hierarchical": is_hierarchical,
        "age_stats": {
            "mean": statistics["age_stats"].mean(),
            "std": statistics["age_stats"].standard_deviation(),
        },
    }

    return result


class FEMRTokenizer(transformers.utils.PushToHubMixin):
    def __init__(self, dictionary):
        assert not dictionary["is_hierarchical"], "Currently not supported"

        self.is_hierarchical = dictionary["is_hierarchical"]

        self.dictionary = dictionary
        vocab = dictionary["vocab"]

        self.numeric_lookup = collections.defaultdict(list)
        self.string_lookup = {}
        self.code_lookup = {}

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
    def from_pretrained(self, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
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
            kwargs: Arguments for loading to pass to transformers.utils.hub.cached_file

        Returns:
            A FEMR Tokenizer
        """

        dictionary_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path, "dictionary.msgpack", **kwargs
        )

        with open(dictionary_file, "rb") as f:
            dictionary = msgpack.load(f)

        return FEMRTokenizer(dictionary)

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
                Additional key word arguments passed along to the [`transformers.utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
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

    def get_feature_codes(self, measurement):
        if measurement.get("numeric_value") is not None:
            for start, end, i in self.numeric_lookup.get(measurement["code"], []):
                if start <= measurement["numeric_value"] < end:
                    return [i]
            else:
                return []
        elif measurement.get("text_value") is not None:
            value = self.string_lookup.get((measurement["code"], measurement["text_value"]))
            if value is not None:
                return [value]
            else:
                return []
        else:
            value = self.code_lookup.get(measurement["code"])
            if value is not None:
                return [value]
            else:
                return []

    def normalize_age(self, age):
        return (age - self.dictionary["age_stats"]["mean"]) / (self.dictionary["age_stats"]["std"])
