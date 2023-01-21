"""A module for generating labels on patient timelines."""

from __future__ import annotations

import hashlib
import math
import struct


def compute_random_num(seed: int, num_1: int, num_2: int):
    network_num_1 = struct.pack("!I", num_1)
    network_num_2 = struct.pack("!I", num_2)
    network_seed = struct.pack("!I", seed)

    to_hash = network_seed + network_num_1 + network_num_2

    hash_object = hashlib.sha256()
    hash_object.update(to_hash)
    hash_value = hash_object.digest()

    result = 0
    for i in range(len(hash_value)):
        result = (result * 256 + hash_value[i]) % 100

    return result
