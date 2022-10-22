from __future__ import annotations

import contextlib
import functools
import itertools
import multiprocessing
import random
from typing import Any, Iterable, Iterator, List, Optional

import pytest

import piton.datasets.utils

"""
class MultiplexStreamingMerge(Generic[T]):
    def __init__(
        self,
        iterators: Sequence[Iterable[T]],
        get_shard: Callable[[T], int],
        num_shards: int,
        func: Callable[[Iterable[T]], None],
"""


@contextlib.contextmanager
def chunk_helper(chunks: List[int]) -> Iterator[List[int]]:
    yield chunks


@pytest.mark.parametrize(
    "module",
    [
        piton.datasets.utils.SingleThreadMultiplexStreamingMerge,
        piton.datasets.utils.MultiplexStreamingMerge,
    ],
)
def test_simple_random(module: Any) -> None:
    random.seed(2315423)

    total_items = [i for i in range(100) for _ in range(3)] + list(range(20))
    random.shuffle(total_items)

    chunks = [
        sorted(total_items[i * 20 : (i + 1) * 20])
        for i in range(len(total_items) // 20)
    ]

    assert sorted(list(itertools.chain.from_iterable(chunks))) == sorted(
        total_items
    )

    chunk_funcs = [functools.partial(chunk_helper, a) for a in chunks]

    num_shards = 7

    def get_shard(index: int) -> int:
        return index % num_shards

    final_queue: multiprocessing.Queue[Optional[int]] = multiprocessing.Queue()

    def add_to_queue(iterator: Iterable[int]) -> None:
        current_shard = None
        last_index: Optional[int] = None
        for i in iterator:
            if last_index is not None:
                assert i >= last_index

            last_index = i
            if current_shard is None:
                current_shard = get_shard(i)
            assert current_shard == get_shard(i)
            final_queue.put(i)
        final_queue.put(None)

    result = []

    with contextlib.closing(
        module(chunk_funcs, get_shard, num_shards, add_to_queue)
    ) as _:
        seen_done = 0
        while seen_done != num_shards:
            next_item = final_queue.get()
            if next_item is None:
                seen_done += 1
            else:
                result.append(next_item)

    assert sorted(result) == sorted(total_items)
