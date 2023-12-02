"""femr.datasets provides the main tools for doing raw data manipulation."""
from __future__ import annotations

import collections.abc
import contextlib
import functools
import itertools
import multiprocessing.pool
import os
import tempfile
from typing import Any, Callable, ContextManager, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import datasets
import event_stream_data_standard as ESDS
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from femr.datasets import fileio
from femr.datasets.types import RawEvent, RawPatient

femr_metadata = pa.struct([
    ('end', pa.timestamp('us')),
    ('visit_id', pa.int64()),
    ('omop_table', pa.string()),
    ('clarity_table', pa.string()),
    ('visit_id', pa.int64()),
    ('note_id', pa.int64()),
    ('unit', pa.string()),
])

femr_patient = ESDS.patient_schema(femr_metadata)

def _get_sort_key(pid_and_event: Tuple[int, RawEvent]) -> Any:
    """Get the sort key for an event."""
    return (pid_and_event[0], pid_and_event[1].start)


def _sort_readers(
    events: EventCollection,
    reader_funcs: Sequence[Callable[[], ContextManager[Iterable[Tuple[int, RawEvent]]]]],
) -> None:
    """Sort the provided reader_funcs and write out to the provided events."""
    items: List[Tuple[int, RawEvent]] = []
    for reader_func in reader_funcs:
        with reader_func() as reader:
            items.extend(reader)

    with contextlib.closing(events.create_writer()) as writer:
        items.sort(key=_get_sort_key)
        for i, e in items:
            writer.add_event(i, e)


def _create_event_reader(
    path: str,
) -> ContextManager[Iterable[Tuple[int, RawEvent]]]:
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
    ) -> Sequence[Callable[[], ContextManager[Iterable[Tuple[int, RawEvent]]]]]:
        """Return a list of reader functions.

        Each resulting reader can be used in a multiprocessing.Pool to enable multiprocessing.
        """
        return [
            functools.partial(_create_event_reader, os.path.join(self.path, child)) for child in os.listdir(self.path)
        ]

    @contextlib.contextmanager
    def reader(self) -> Iterator[Iterable[Tuple[int, RawEvent]]]:
        """Return a contextmanager that allows iteration over all of the events."""
        with contextlib.ExitStack() as stack:
            sub_readers = [stack.enter_context(reader()) for reader in self.sharded_readers()]
            yield itertools.chain.from_iterable(sub_readers)

    def create_writer(self) -> fileio.EventWriter:
        """Create an EventWriter."""
        return fileio.EventWriter(self.path)

    def sort(self, target_path: str, num_threads: int = 1) -> EventCollection:
        """Sort the collection and store the resulting output in the target_path."""
        result = EventCollection(target_path)

        current_shards = self.sharded_readers()

        targets_per_current = (len(current_shards) + num_threads - 1) // num_threads

        chunks = [current_shards[i * targets_per_current : (i + 1) * targets_per_current] for i in range(num_threads)]

        chunks = [chunk for chunk in chunks if chunk]

        with multiprocessing.pool.Pool(num_threads) as pool:
            for _ in pool.imap_unordered(functools.partial(_sort_readers, result), chunks):
                pass

        return result

    def to_patient_collection(self, target_path: str, num_threads: int = 1) -> PatientCollection:
        """Convert the EventCollection to a PatientCollection, which is stored in target_path."""
        assert not os.path.exists(target_path)


        event_files = [os.path.join(self.path, child) for child in os.listdir(self.path)]
        events = pl.scan_csv(event_files,
            schema={
                'patient_id': pl.Int64(),
                'concept_id': pl.Int64(),
                'start': pl.Datetime(),
                'metadata': pl.Utf8(),
                'value': pl.Utf8(),
            })
        events = events.sort(by=(pl.col("patient_id"), pl.col("start")))

        hashed_patient_id = pl.col("patient_id").hash() % num_threads
        events = events.with_columns(hashed_patient_id=hashed_patient_id)

        os.mkdir(target_path)

        partitioned_patients = events.collect().partition_by("hashed_patient_id")

        for i, patient in enumerate(partitioned_patients):
            patient.write_csv(os.path.join(target_path, f"{i}.csv"))

        return PatientCollection(target_path)


def _sharded_patient_reader(path: str) -> ContextManager[Iterable[RawPatient]]:
    """Get a contextmanager for reading patients from a particular path."""
    return contextlib.closing(fileio.PatientReader(path))


def _transform_single_reader(
    target_path: str,
    transforms: Sequence[Callable[[RawPatient], Optional[RawPatient]]],
    capture_statistics: bool,
    reader_func: Callable[[], ContextManager[Iterable[RawPatient]]],
) -> Optional[Dict[str, Dict[str, int]]]:
    """Transform a single PatientReader, writing to a particular target_path."""
    if capture_statistics:
        information: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    with contextlib.closing(fileio.PatientWriter(target_path)) as o:
        with reader_func() as reader:
            for p in reader:
                if capture_statistics:
                    current_event_count = len(p.events)
                current_patient: Optional[RawPatient] = p
                for transform in transforms:
                    assert current_patient is not None
                    current_patient = transform(current_patient)
                    if current_patient is None:
                        if capture_statistics:
                            information[str(transform)]["lost_patients"] += 1
                            information[str(transform)]["lost_events"] += current_event_count
                        break
                    else:
                        if capture_statistics:
                            new_events = len(current_patient.events)
                            information[str(transform)]["lost_events"] += current_event_count - new_events
                            current_event_count = new_events

                if current_patient is not None:
                    o.add_patient(current_patient)

    if capture_statistics:
        return {k: dict(v) for k, v in information.items()}
    else:
        return None

def _convert_to_parquet(target_path: str, concept_map: Mapping[int, str], reader_func: Callable[[], ContextManager[Iterable[RawPatient]]]) -> None:
    file = tempfile.NamedTemporaryFile(dir=target_path, suffix=".parquet", delete=False)
    current_patients = []

    with pq.ParquetWriter(file.name, schema=femr_patient) as w:
        with reader_func() as reader:
            for p in reader:
                events: List[ESDS.Event] = []
                current_measurement = None
                current_time = None
                for event in p.events:
                    event_dict: ESDS.Measurement = {
                        'code': concept_map[event.concept_id],
                    }

            pickle.dumps({a: b for a, b in event.__dict__.items() if a not in ("start", "concept_id", "value")})


                    if event.value is None:
                        pass
                    elif isinstance(event.value, str):
                        event_dict['text_value'] = event.value
                    elif isinstance(event.value, float):
                        event_dict['numeric_value'] = event.value

                    if current_time is None:
                        current_time = event.start
                        current_measurement = [event_dict]
                    elif current_time == event.start:
                        current_measurement.append(event_dict)
                    else:
                        events.append({'time': current_time, 'measurements': current_measurement})
                        current_time = event.start
                        current_measurement = [event_dict]

                if current_measurement is not None:
                    assert current_time is not None
                    events.append({'time': current_time, 'measurements': current_measurement})

                current_patients.append({
                    'patient_id': p.patient_id,
                    'events': events
                })
                if p.patient_id == 29925175:
                    print(current_patients[-1])

                if len(current_patients) > 10_000:
                    w.write_batch(pa.RecordBatch.from_pylist(current_patients, schema=femr_patient))
                    current_patients = []

        w.write_batch(pa.RecordBatch.from_pylist(current_patients, schema=femr_patient))


class PatientCollection:
    """A PatientCollection is an unordered sequence of Patients."""

    def __init__(self, path: str):
        """Open a PatientCollection at a particular path."""
        self.path = path

    def sharded_readers(
        self,
    ) -> Sequence[Callable[[], ContextManager[Iterable[RawPatient]]]]:
        """Return a list of contextmanagers that allow sharded iteration of Patients."""
        return [
            functools.partial(_sharded_patient_reader, os.path.join(self.path, child))
            for child in os.listdir(self.path)
        ]

    @contextlib.contextmanager
    def reader(self) -> Iterator[Iterable[RawPatient]]:
        """Return a single contextmanager that allows iteration over Patients."""
        with contextlib.ExitStack() as stack:
            sub_readers = [stack.enter_context(reader()) for reader in self.sharded_readers()]
            yield itertools.chain.from_iterable(sub_readers)

    def transform(
        self,
        target_path: str,
        transform: Callable[[RawPatient], Optional[RawPatient]]
        | Sequence[Callable[[RawPatient], Optional[RawPatient]]],
        num_threads: int = 1,
        stats_dict: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> PatientCollection:
        """Apply a transformation to the patient files in a folder to generate a modified output folder."""
        os.mkdir(target_path)

        if not isinstance(transform, collections.abc.Sequence):
            transform = [transform]

        total_stats: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))

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

    def to_huggingface_dataset(
        self,
        target_path: str,
        concept_path: str,
        num_threads: int = 1,
        delimiter: str = ",",
    ) -> datasets.Dataset:
        """Convert a PatientCollection to a PatientDatabase."""

        dataset_path = os.path.join(target_path, "dataset")
        os.mkdir(dataset_path)

        concept_table = pl.scan_csv(
            [os.path.join(concept_path, 'concept', a) for a in os.listdir(os.path.join(concept_path, 'concept'))],
            schema={
                'concept_id': pl.Int64(),
                'vocabulary_id': pl.Utf8(),
                'concept_code': pl.Utf8(),
            }
        )
        result = concept_table.select(pl.col("concept_id"), pl.col("vocabulary_id") + "/" + pl.col("concept_code"))
        collected_result = result.collect().to_dict()
        concept_map = dict(zip(collected_result['concept_id'], collected_result['vocabulary_id']))

        with multiprocessing.pool.Pool(num_threads) as pool:
            for _ in pool.imap_unordered(
                functools.partial(
                    _convert_to_parquet,
                    dataset_path,
                    concept_map,
                ),
                self.sharded_readers(),
            ):
                pass
