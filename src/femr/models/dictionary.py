import collections
import datetime
import functools
import math

import femr.hf_utils
import femr.stat_utils


def create_dictionary(dataset, vocab_size, is_hierarchical=False, num_proc=1):
    statistics = femr.hf_utils.aggregate_over_dataset(
        dataset,
        functools.partial(map_statistics, num_patients=len(dataset)),
        agg_statistics,
        num_proc=num_proc,
        batch_size=1_000,
    )
    return convert_statistics_to_msgpack(statistics, vocab_size, is_hierarchical)


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


class FeatureLookup:
    def __init__(self, dictionary):
        assert not dictionary["is_hierarchical"], "Currently not supported"

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

    def get_feature_codes(self, measurement):
        if measurement["numeric_value"] is not None:
            for start, end, i in self.numeric_lookup.get(measurement["code"], []):
                if start <= measurement["numeric_value"] < end:
                    return [i]
            else:
                return []
        elif measurement["text_value"] is not None:
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
