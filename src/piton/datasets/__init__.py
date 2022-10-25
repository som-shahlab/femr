from __future__ import annotations

import collections.abc
import contextlib
import datetime
import functools
import itertools
import multiprocessing.pool
import os
import tempfile
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

from .. import Event, Patient
from ..extension import datasets as extension_datasets
from . import fileio, utils

USE_PYTHON_JOIN = False


def _get_sort_key(pid_and_event: Tuple[int, Event]) -> Any:
    return (pid_and_event[0], pid_and_event[1].start)


def _sort_helper(
    events: EventCollection,
    reader_funcs: Sequence[
        Callable[[], ContextManager[Iterable[Tuple[int, Event]]]]
    ],
) -> None:
    items: List[Tuple[int, Event]] = []
    for reader_func in reader_funcs:
        with reader_func() as reader:
            items.extend(reader)

    with contextlib.closing(events.create_writer()) as writer:
        items.sort(key=_get_sort_key)
        for i, e in items:
            writer.add_event(i, e)


def _get_patient_shard(num_shards: int, item: Tuple[int, Event]) -> int:
    return item[0] % num_shards


def _convert_to_patient_helper(
    target_path: str, events: Iterable[Tuple[int, Event]]
) -> None:
    current_patient_id: Optional[int] = None
    current_events: Optional[List[Event]] = None
    with contextlib.closing(fileio.PatientWriter(target_path)) as f:
        for (patient_id, event) in events:
            if current_patient_id != patient_id:
                if current_patient_id is not None:
                    assert current_events is not None
                    f.add_patient(
                        Patient(
                            patient_id=current_patient_id,
                            events=current_events,
                        )
                    )
                current_patient_id = patient_id
                current_events = []

            assert current_events is not None
            current_events.append(event)

        if current_patient_id is not None:
            assert current_events is not None
            f.add_patient(
                Patient(patient_id=current_patient_id, events=current_events)
            )


def _sharded_event_helper(
    path: str,
) -> ContextManager[Iterable[Tuple[int, Event]]]:
    return contextlib.closing(fileio.EventReader(path))


class EventCollection:
    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            os.mkdir(self.path)

    def sharded_readers(
        self,
    ) -> Sequence[Callable[[], ContextManager[Iterable[Tuple[int, Event]]]]]:
        return [
            functools.partial(
                _sharded_event_helper, os.path.join(self.path, child)
            )
            for child in os.listdir(self.path)
        ]

    @contextlib.contextmanager
    def reader(self) -> Iterator[Iterable[Tuple[int, Event]]]:
        with contextlib.ExitStack() as stack:
            sub_readers = [
                stack.enter_context(reader())
                for reader in self.sharded_readers()
            ]
            yield itertools.chain.from_iterable(sub_readers)

    def create_writer(self) -> fileio.EventWriter:
        return fileio.EventWriter(self.path)

    def sort(self, target_path: str, num_threads: int = 1) -> EventCollection:
        result = EventCollection(target_path)

        current_shards = self.sharded_readers()

        targets_per_current = (
            len(current_shards) + num_threads - 1
        ) // num_threads

        chunks = [
            current_shards[
                i * targets_per_current : (i + 1) * targets_per_current
            ]
            for i in range(num_threads)
        ]

        chunks = [chunk for chunk in chunks if chunk]

        with multiprocessing.pool.Pool(num_threads) as pool:
            for _ in pool.imap_unordered(
                functools.partial(_sort_helper, result), chunks
            ):
                pass

        return result

    def to_patient_collection(
        self, target_path: str, num_threads: int = 1
    ) -> PatientCollection:

        if USE_PYTHON_JOIN:
            os.mkdir(target_path)

            with tempfile.TemporaryDirectory() as tmp:
                sorted_self = self.sort(tmp, num_threads)
                print("Sorted", datetime.datetime.now())

                with contextlib.closing(
                    utils.MultiplexStreamingMerge(
                        sorted_self.sharded_readers(),
                        functools.partial(_get_patient_shard, num_threads),
                        num_threads,
                        functools.partial(
                            _convert_to_patient_helper, target_path
                        ),
                        _get_sort_key,
                    )
                ) as _:
                    pass
        else:
            extension_datasets.sort_and_join_csvs(
                self.path,
                target_path,
                ["patient_id", "start", "code"],
                ",",
                num_threads,
            )

        return PatientCollection(target_path)


def _sharded_patient_helper(path: str) -> ContextManager[Iterable[Patient]]:
    return contextlib.closing(fileio.PatientReader(path))


def _transform_helper(
    target_path: str,
    transforms: Sequence[Callable[[Patient], Optional[Patient]]],
    capture_statistics: bool,
    reader_func: Callable[[], ContextManager[Iterable[Patient]]],
) -> Optional[Dict[str, Dict[str, int]]]:
    if capture_statistics:
        information: Dict[str, Dict[str, int]] = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )
    with contextlib.closing(fileio.PatientWriter(target_path)) as o:
        with reader_func() as reader:
            for p in reader:
                if capture_statistics:
                    current_event_count = len(p.events)
                current_patient: Optional[Patient] = p
                for transform in transforms:
                    assert current_patient is not None
                    current_patient = transform(current_patient)
                    if current_patient is None:
                        if capture_statistics:
                            information[str(transform)]["lost_patients"] += 1
                            information[str(transform)][
                                "lost_events"
                            ] += current_event_count
                        break
                    else:
                        if capture_statistics:
                            new_events = len(current_patient.events)
                            information[str(transform)]["lost_events"] += (
                                current_event_count - new_events
                            )
                            current_event_count = new_events

                if current_patient is not None:
                    o.add_patient(current_patient)

    if capture_statistics:
        return {k: dict(v) for k, v in information.items()}
    else:
        return None


class PatientCollection:
    def __init__(self, path: str):
        self.path = path

    def sharded_readers(
        self,
    ) -> Sequence[Callable[[], ContextManager[Iterable[Patient]]]]:
        return [
            functools.partial(
                _sharded_patient_helper, os.path.join(self.path, child)
            )
            for child in os.listdir(self.path)
        ]

    @contextlib.contextmanager
    def reader(self) -> Iterator[Iterable[Patient]]:
        with contextlib.ExitStack() as stack:
            sub_readers = [
                stack.enter_context(reader())
                for reader in self.sharded_readers()
            ]
            yield itertools.chain.from_iterable(sub_readers)

    def transform(
        self,
        target_path: str,
        transform: Callable[[Patient], Optional[Patient]]
        | Sequence[Callable[[Patient], Optional[Patient]]],
        num_threads: int = 1,
        stats_dict: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> PatientCollection:
        """
        Applies a transformation to the patient files in a folder to generate a modified output folder.
        Optionally supports multithreading.
        """

        os.mkdir(target_path)

        if not isinstance(transform, collections.abc.Sequence):
            transform = [transform]

        total_stats: Dict[str, Dict[str, int]] = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )

        with multiprocessing.pool.Pool(num_threads) as pool:
            for stats in pool.imap_unordered(
                functools.partial(
                    _transform_helper,
                    target_path,
                    transform,
                    stats_dict is not None,
                ),
                self.sharded_readers(),
            ):
                if stats is not None:
                    for k, v in stats.items():
                        for sub_k, sub_v in v.items():
                            total_stats[k][sub_k] += sub_v
        if stats_dict is not None:
            stats_dict.update(total_stats)

        return PatientCollection(target_path)

    def to_patient_database(
        self, target_path: str, concept_path: str, num_threads: int = 1
    ) -> PatientDatabase:
        extension_datasets.convert_patient_collection_to_patient_database(
            self.path, concept_path, target_path, ",", num_threads
        )
        return PatientDatabase(target_path)


# Import from C++ extension

PatientDatabase = extension_datasets.PatientDatabase
