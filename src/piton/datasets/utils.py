from __future__ import annotations

import contextlib
import heapq
import itertools
import multiprocessing
from typing import (
    Any,
    Callable,
    ContextManager,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Collection,
)

T = TypeVar("T")


def convert_queue_to_iterable(
    queue: multiprocessing.Queue[Optional[T]],
) -> Iterable[T]:
    while True:
        next_item = queue.get()
        if next_item is None:
            break
        yield next_item


def multiplex_process(
    read_queues: Sequence[multiprocessing.Queue[Optional[T]]],
    write_queues: Sequence[multiprocessing.Queue[Optional[T]]],
    get_shard: Callable[[T], int],
    key: Callable[[T], Any],
) -> None:
    next_item_heap = []

    for i, queue in enumerate(read_queues):
        next_item = queue.get()
        if next_item is not None:
            next_item_heap.append((key(next_item), i, next_item, queue))

    heapq.heapify(next_item_heap)

    while len(next_item_heap) != 0:
        (_, i, next_item, queue) = heapq.heappop(next_item_heap)
        write_queues[get_shard(next_item)].put(next_item)

        next_item = queue.get()
        if next_item is not None:
            heapq.heappush(
                next_item_heap, (key(next_item), i, next_item, queue)
            )

    for write in write_queues:
        write.put(None)


def output_process(
    func: Callable[[Iterable[T]], None],
    write_queue: multiprocessing.Queue[Optional[T]],
) -> None:
    func(convert_queue_to_iterable(write_queue))


def input_process(
    iterator_func: Callable[[], ContextManager[Iterable[T]]],
    read_queue: multiprocessing.Queue[Optional[T]],
) -> None:
    with iterator_func() as iterator:
        for item in iterator:
            read_queue.put(item)
    read_queue.put(None)


class MultiplexStreamingMerge(Generic[T]):
    def __init__(
        self,
        iterator_funcs: Sequence[Callable[[], ContextManager[Iterable[T]]]],
        get_shard: Callable[[T], int],
        num_shards: int,
        func: Callable[[Iterable[T]], None],
        key: Callable[[T], Any] = lambda a: a,
        maxsize: int = 1000,
    ):
        self.read_queues: List[multiprocessing.Queue[Optional[T]]] = [
            multiprocessing.Queue(maxsize=maxsize) for _ in iterator_funcs
        ]

        self.write_queues: List[multiprocessing.Queue[Optional[T]]] = [
            multiprocessing.Queue(maxsize=maxsize) for _ in range(num_shards)
        ]

        self.queues = self.read_queues + self.write_queues

        self.input_processes = [
            multiprocessing.Process(
                target=input_process,
                args=(iterator_func, read_queue),
            )
            for iterator_func, read_queue in zip(
                iterator_funcs, self.read_queues
            )
        ]
        self.multiplex_process = multiprocessing.Process(
            target=multiplex_process,
            args=(self.read_queues, self.write_queues, get_shard, key),
        )

        self.output_processes = [
            multiprocessing.Process(
                target=output_process,
                args=(func, write_queue),
            )
            for write_queue in self.write_queues
        ]

        self.processes = (
            self.input_processes
            + self.output_processes
            + [self.multiplex_process]
        )

        for process in self.processes:
            process.start()

    def close(self) -> None:
        for process in self.processes:
            process.join()
            process.close()

        for queue in self.queues:
            queue.close()


def _helper(
    main_generator: Iterator[Tuple[int, T]], shard_index: int
) -> Iterator[T]:
    for index, item in main_generator:
        if index == shard_index:
            yield item


def _main_generator(
    iterator_funcs: Sequence[Callable[[], ContextManager[Iterable[T]]]],
    get_shard: Callable[[T], int],
    key: Callable[[T], Any],
) -> Iterator[Tuple[int, T]]:
    with contextlib.ExitStack() as stack:
        input_files = [
            iter(stack.enter_context(iterator_func()))
            for iterator_func in iterator_funcs
        ]

        next_item_heap = []

        for i, input_file in enumerate(input_files):
            next_item = next(input_file, None)
            if next_item is not None:
                next_item_heap.append(
                    (key(next_item), i, next_item, input_file)
                )

        heapq.heapify(next_item_heap)

        while len(next_item_heap) != 0:
            (_, i, next_item, input_file) = heapq.heappop(next_item_heap)
            yield (get_shard(next_item), next_item)

            next_item = next(input_file, None)
            if next_item is not None:
                heapq.heappush(
                    next_item_heap, (key(next_item), i, next_item, input_file)
                )


class SingleThreadMultiplexStreamingMerge(Generic[T]):
    def __init__(
        self,
        iterator_funcs: Sequence[Callable[[], ContextManager[Iterable[T]]]],
        get_shard: Callable[[T], int],
        num_shards: int,
        func: Callable[[Iterable[T]], None],
        key: Callable[[T], Any] = lambda a: a,
        maxsize: int = 1000,
    ):
        generators: Collection[Iterator[Tuple[int, T]]] = itertools.tee(
            _main_generator(iterator_funcs, get_shard, key), num_shards
        )

        for i, generator in enumerate(generators):
            func(_helper(generator, i))

    def close(self) -> None:
        pass
