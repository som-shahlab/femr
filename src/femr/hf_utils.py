import functools
import json
import pickle

import meds


def _agg_helper(*args, map_func):
    result = map_func(*args)
    return {"data": [pickle.dumps(result)]}


def aggregate_over_dataset(dataset, map_func, agg_func, batch_size, num_proc, with_indices=False):
    """Perform an aggregation over a huggingface dataset.

    This logic consists of two parts, map_func and agg_func.

    map_func takes a batch of data and converts it to an intermediate result.

    agg_func takes those intermediate results and combines them into a final result.
    """
    parts = dataset.map(
        functools.partial(_agg_helper, map_func=map_func),
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        with_indices=with_indices,
        keep_in_memory=True,
        new_fingerprint="invalid",
    )

    current = None
    for stat in parts:
        fixed_stat = pickle.loads(stat["data"])
        if current is None:
            current = fixed_stat
        else:
            current = agg_func(current, fixed_stat)

    return current

def dump_patient_to_json(patient: meds.Patient, path: str) -> None:
    """Dump a patient to a human-readable JSON file, with pretty print."""
    with open(path, "w") as f:
        json.dump(patient, f, indent=2, default=str)