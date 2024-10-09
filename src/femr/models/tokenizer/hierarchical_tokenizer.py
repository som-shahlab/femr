from __future__ import annotations

import bisect
import collections
import datetime
import functools
import math
import os
from typing import Any, Dict, Iterator, List, Mapping, Optional, Set, Tuple, Union
import meds

import meds_reader
import msgpack
import numpy as np
import transformers
import pyarrow as pa

import femr.ontology
import femr.stat_utils
import femr.pat_utils

import pickle
import traceback


def agg_statistics(stats1, stats2):
    try:
        for k in stats1["age_stats"]:
            stats1["age_stats"][k].combine(stats2["age_stats"][k])

        for k, v in stats2["code_counts"].items():
            stats1["code_counts"][k] += v

        for k in stats1["property_samples"]:
            v1 = stats1["property_samples"][k]
            v2 = stats2["property_samples"][k]
            
            for k, v in v2["text_counts"].items():
                v1["text_counts"][k] += v
            v1["numeric_samples"].combine(v2["numeric_samples"])
            v1["numeric_count"] += v2["numeric_count"]

    except Exception as e:
        traceback.print_exc()
        raise e
        

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

def try_float(v):
    try:
        return float(v)
    except:
        return None


def map_statistics(
    subjects: Iterator[meds_reader.Subject],
    *,
    num_subjects: int,
    ontology: femr.ontology.Ontology,
    properties: Mapping[str, pa.DataType],
) -> Mapping[str, Any]:
    age_stats = collections.defaultdict(femr.stat_utils.OnlineStatistics)
    code_counts: Dict[str, float] = collections.defaultdict(float)

    
    bad_properties = {'code', 'time'}

    property_samples = {
        k: {
            'numeric_samples': femr.stat_utils.ReservoirSampler(10_000),
            'text_counts': collections.defaultdict(float),
            'numeric_count': 0,
        } for k in properties if k not in bad_properties
    }

    for subject in subjects:
        total_events = len(subject.events)

        if total_events == 0:
            continue

        weight = 1.0 / (num_subjects * total_events)
        birth_date = femr.pat_utils.get_subject_birthdate(subject)
        code_set = set()


        pat_samples = {
            k: {
                'numeric_samples': [],
                'text_counts': set(),
            } for k in property_samples
        }

        last_time = None

        for event in subject.events:
            if event.time is not None and event.time.date() > birth_date.date():
                age = (event.time - birth_date).total_seconds()
                age_stats["age"].add(weight, age)
                age_stats["log_age"].add(weight, math.log(1 + age))

            if event.time is not None and event.time.date() > birth_date.date() and last_time is not None and last_time.date() > birth_date.date():
                delta = (event.time - last_time).total_seconds()
                age_stats["delta"].add(weight, delta)
                age_stats["log_delta"].add(weight, math.log(1 + delta))

            last_time = event.time

            for k, v in event:
                if k == 'code':
                    code_set.add(v)
                elif k in pat_samples:
                    possib_float = try_float(v)
                    if possib_float is not None:
                        pat_samples[k]['numeric_samples'].append(possib_float)
                    else:
                        pat_samples[k]['text_counts'].add(str(v))


        final_codes: Set[str] = set()
        for code in code_set:
            final_codes |= ontology.get_all_parents(code)

        for code in final_codes:
            code_counts[code] += 1 / num_subjects

        for k, v in pat_samples.items():
            res = property_samples[k]
            for text in v['text_counts']:
                res['text_counts'][text] += 1 / num_subjects
            for value in v['numeric_samples']:
                res['numeric_samples'].add(value, 1 / (num_subjects * len(v['numeric_samples'])))
            res['numeric_count'] += len(v['numeric_samples']) / weight

    return {
        "age_stats": dict(age_stats),
        "code_counts": code_counts,
        "property_samples": property_samples,
    }


def entropy(p):
    return p * math.log(p) + (1 - p) * math.log(1 - p)

def convert_statistics_to_msgpack(
    statistics, vocab_size: int, num_numeric: int, ontology: femr.ontology.Ontology, min_fraction: float,
):
    vocab = []

    for code, weight in statistics["code_counts"].items():
        baseline = min([1] + [statistics["code_counts"][parent] for parent in ontology.get_parents(code)])
        weight = weight / baseline

        weight = min(1, weight)

        if code == meds.birth_code:
            baseline = 1
            weight = 0.5

        if weight != 0 and weight != 1:
            entry = {
                "type": "code",
                "code_string": code,
                "weight": baseline * (weight * math.log(weight) + (1 - weight) * math.log(1 - weight)),
            }
            vocab.append(entry)


    total_numeric_weight = sum(property_data["numeric_count"] for property_data in statistics["property_samples"].values())

    for property, property_data in statistics["property_samples"].items():
        for text, weight in property_data["text_counts"].items():
            weight = min(1, weight)

            if weight != 0 and weight != 1:
                entry = {
                    "type": "text",
                    "property": property,
                    "text_string": text,
                    "weight": weight * math.log(weight) + (1 - weight) * math.log(1 - weight),
                }
                vocab.append(entry)

        numeric_samples = list(property_data["numeric_samples"].samples)

        if len(numeric_samples) > 0:
            assert num_numeric >= 1

            frac_numeric = property_data["numeric_count"] / total_numeric_weight

            print("Has frac", property, frac_numeric)

            num_numeric_for_property = max(1, int(num_numeric * frac_numeric))

            bins = sorted(list(set(np.quantile(numeric_samples, np.linspace(0, 1, num=num_numeric_for_property + 1)))))
            bins = bins[1:-1]

            samples = [float('-inf')]
            samples.extend(bins)
            samples.append(float("inf"))

            assert len(samples) >= 2

            for start_val, end_val in zip(samples, samples[1:]):
                entry = {
                    "type": "numeric",
                    "property": property,
                    "val_start": start_val,
                    "val_end": end_val,
                    "weight": -1,
                }
                vocab.append(entry)

    min_entropy = entropy(min_fraction)

    vocab = [v for v in vocab if v["weight"] <= min_entropy]

    vocab.sort(key=lambda a: a["weight"])
    vocab = vocab[:vocab_size]

    age_stats_dict = {}
    for k, v in statistics["age_stats"].items():
        age_stats_dict[k] = {
            "mean": v.mean(),
            "std": v.standard_deviation(),
        }

    result = {
        "vocab": vocab,
        "age_stats": age_stats_dict,
    }

    return result


class HierarchicalTokenizer(transformers.utils.PushToHubMixin):
    @classmethod
    def train(
        self,
        db: meds_reader.SubjectDatabase,
        vocab_size: int,
        ontology: femr.ontology.Ontology,
        num_numeric: int = 1000,
        min_fraction: float = 1/1000,
        banned_properties: Set[str] = {}, 
    ) -> HierarchicalTokenizer:
        """Train a FEMR tokenizer from the given dataset"""

        properties = db.properties
        for banned in banned_properties:
            del properties[banned]

        statistics = functools.reduce(
            agg_statistics,
            db.map(
                functools.partial(
                    map_statistics,
                    num_subjects=len(db),
                    ontology=ontology,
                    properties = db.properties,
                )
            ),
        )

        whatever = convert_statistics_to_msgpack(statistics, vocab_size, num_numeric, ontology, min_fraction)

        return HierarchicalTokenizer(
            whatever, ontology
        )


    def __init__(self, dictionary: Mapping[str, Any], ontology: femr.ontology.Ontology):
        self.dictionary = dictionary

        self.ontology = ontology

        self.dictionary = dictionary
        vocab = dictionary["vocab"]

        self.string_lookup = collections.defaultdict(dict)
        self.code_lookup = {}

        self.vocab_size = len(vocab)

        numeric_entries = collections.defaultdict(list)
        for i, dict_entry in enumerate(vocab):
            if dict_entry["type"] == "code":
                self.code_lookup[dict_entry["code_string"]] = i
            elif dict_entry["type"] == "numeric":
                numeric_entries[dict_entry["property"]].append((dict_entry["val_start"], i))
            elif dict_entry["type"] == "text":
                self.string_lookup[dict_entry["property"]][dict_entry["text_string"]] = i
            else:
                pass

        self.numeric_values = {}
        self.numeric_indices = {}

        for k, v in numeric_entries.items():
            v.sort()
            if len(numeric_entries) > 0:
                self.numeric_values[k] = [a[0] for a in v[1:]]
                self.numeric_indices[k] = [a[1] for a in v]

    @classmethod
    def from_pretrained(
        self,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        ontology: femr.ontology.Ontology = None,
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

        return HierarchicalTokenizer(dictionary, ontology=ontology)

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
        assert self.ontology is not None
        codes = [
            self.code_lookup[parent]
            for parent in self.ontology.get_all_parents(event.code)
            if parent in self.code_lookup
        ]
        weights = [1 / len(codes) for _ in codes]

        for k, v in event:
            if k in self.numeric_indices and ((possib_float := try_float(v)) is not None):
                codes.append(self.numeric_indices[k][bisect.bisect(self.numeric_values[k], possib_float)])
                weights.append(1)
            elif k in self.string_lookup and ((possib_value := self.string_lookup[k].get(v)) is not None):
                codes.append(possib_value)
                weights.append(1)

        return codes, weights


    def get_time_data(self, age: datetime.timedelta, delta: datetime.timedelta) -> float:
        result = []

        for v, name in zip((age, delta), ("age", "delta")):
            for transform, transform_name in ((lambda a:a, ""), (lambda a: math.log(a + 1), "log_")):
                stats = self.dictionary["age_stats"][transform_name + name]
                if v is None:
                    result.append(0)
                else:
                    result.append((transform(v.total_seconds()) - stats["mean"]) / stats["std"])

        return result