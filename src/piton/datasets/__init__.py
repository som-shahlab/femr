"""piton.datasets provides the main tools for doing raw data manipulation."""
from __future__ import annotations

import collections.abc
import contextlib
import functools
import itertools
import multiprocessing.pool
import os
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

import numpy as np

from piton import Event, Patient
from piton.datasets import fileio
from piton.extension import datasets as extension_datasets


def _get_sort_key(pid_and_event: Tuple[int, Event]) -> Any:
    """Get the sort key for an event."""
    return (pid_and_event[0], pid_and_event[1].start)


def _sort_readers(
    events: EventCollection,
    reader_funcs: Sequence[
        Callable[[], ContextManager[Iterable[Tuple[int, Event]]]]
    ],
) -> None:
    """Sort the provided reader_funcs and write out to the provided events."""
    items: List[Tuple[int, Event]] = []
    for reader_func in reader_funcs:
        with reader_func() as reader:
            items.extend(reader)

    with contextlib.closing(events.create_writer()) as writer:
        items.sort(key=_get_sort_key)
        for i, e in items:
            writer.add_event(i, e)


def _create_event_reader(
    path: str,
) -> ContextManager[Iterable[Tuple[int, Event]]]:
    """Create an event writer with a contextmanger."""
    return contextlib.closing(fileio.EventReader(path))


class EventCollection:
    """A datatype that represents an unordered collection of Events."""

    def __init__(self, path: str):
        """Create or open an EventCollection at the given path."""
        self.path = path
        if not os.path.exists(path):
            os.mkdir(self.path)

    def sharded_readers(
        self,
    ) -> Sequence[Callable[[], ContextManager[Iterable[Tuple[int, Event]]]]]:
        """Return a list of reader functions.

        Each resulting reader can be used in a multiprocessing.Pool to enable multiprocessing.
        """
        return [
            functools.partial(
                _create_event_reader, os.path.join(self.path, child)
            )
            for child in os.listdir(self.path)
        ]

    @contextlib.contextmanager
    def reader(self) -> Iterator[Iterable[Tuple[int, Event]]]:
        """Return a contextmanager that allows iteration over all of the events."""
        with contextlib.ExitStack() as stack:
            sub_readers = [
                stack.enter_context(reader())
                for reader in self.sharded_readers()
            ]
            yield itertools.chain.from_iterable(sub_readers)

    def create_writer(self) -> fileio.EventWriter:
        """Create an EventWriter."""
        return fileio.EventWriter(self.path)

    def sort(self, target_path: str, num_threads: int = 1) -> EventCollection:
        """Sort the collection and store the resulting output in the target_path."""
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
                functools.partial(_sort_readers, result), chunks
            ):
                pass

        return result

    def to_patient_collection(
        self, target_path: str, num_threads: int = 1
    ) -> PatientCollection:
        """Convert the EventCollection to a PatientCollection, which is stored in target_path."""
        extension_datasets.sort_and_join_csvs(
            self.path,
            target_path,
            np.dtype(
                [
                    ("patient_id", np.uint64),
                    ("start", np.datetime64),
                    ("code", np.uint64),
                ]
            ),
            ",",
            num_threads,
        )

        return PatientCollection(target_path)


def _sharded_patient_reader(path: str) -> ContextManager[Iterable[Patient]]:
    """Get a contextmanager for reading patients from a particular path."""
    return contextlib.closing(fileio.PatientReader(path))


def _transform_single_reader(
    target_path: str,
    transforms: Sequence[Callable[[Patient], Optional[Patient]]],
    capture_statistics: bool,
    reader_func: Callable[[], ContextManager[Iterable[Patient]]],
) -> Optional[Dict[str, Dict[str, int]]]:
    """Transform a single PatientReader, writing to a particular target_path."""
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
    """A PatientCollection is an unordered sequence of Patients."""

    def __init__(self, path: str):
        """Open a PatientCollection at a particular path."""
        self.path = path

    def sharded_readers(
        self,
    ) -> Sequence[Callable[[], ContextManager[Iterable[Patient]]]]:
        """Return a list of contextmanagers that allow sharded iteration of Patients."""
        return [
            functools.partial(
                _sharded_patient_reader, os.path.join(self.path, child)
            )
            for child in os.listdir(self.path)
        ]

    @contextlib.contextmanager
    def reader(self) -> Iterator[Iterable[Patient]]:
        """Return a single contextmanager that allows iteration over Patients."""
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
        """Apply a transformation to the patient files in a folder to generate a modified output folder."""
        os.mkdir(target_path)

        if not isinstance(transform, collections.abc.Sequence):
            transform = [transform]

        total_stats: Dict[str, Dict[str, int]] = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )

        with multiprocessing.pool.Pool(num_threads) as pool:
            for stats in pool.imap_unordered(
                functools.partial(
                    _transform_single_reader,
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
        """Convert a PatientCollection to a PatientDatabase."""
        extension_datasets.convert_patient_collection_to_patient_database(
            self.path, concept_path, target_path, ",", num_threads
        )
        return PatientDatabase(target_path)


# Import from C++ extension

PatientDatabase = extension_datasets.PatientDatabase
Ontology = extension_datasets.Ontology
