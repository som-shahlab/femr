import contextlib
import datetime
import os
import pathlib
import random
from typing import List, Optional, Tuple

import femr
import femr.datasets

dummy_events = [
    femr.datasets.RawEvent(start=datetime.datetime(1995, 1, 3), concept_id=0, value=float(34)),
    femr.datasets.RawEvent(
        start=datetime.datetime(2010, 1, 3),
        concept_id=1,
        value="test_value",
    ),
    femr.datasets.RawEvent(
        start=datetime.datetime(2010, 1, 5),
        concept_id=2,
        value=None,
    ),
]

all_events: List[Tuple[int, femr.datasets.RawEvent]] = []

for patient_id in range(10, 25):
    all_events.extend((patient_id, event) for event in dummy_events)


def create_events(tmp_path: pathlib.Path) -> femr.datasets.EventCollection:
    events = femr.datasets.EventCollection(os.path.join(tmp_path, "events"))

    random.shuffle(all_events)

    chunks = 7
    events_per_chunk = (len(all_events) + chunks - 1) // chunks

    for i in range(7):
        with contextlib.closing(events.create_writer()) as writer:
            for patient_id, event in all_events[i * events_per_chunk : (i + 1) * events_per_chunk]:
                writer.add_event(patient_id, event)

    return events


def create_patients(tmp_path: pathlib.Path) -> femr.datasets.PatientCollection:
    return create_events(tmp_path).to_patient_collection(os.path.join(tmp_path, "patients"))


def test_events(tmp_path: pathlib.Path) -> None:
    events = create_events(tmp_path)

    print(all_events[0])
    with events.reader() as reader:
        read_events = list(reader)

    print(read_events[0])
    assert sorted(read_events) == sorted(all_events)


def test_sort_events(tmp_path: pathlib.Path) -> None:
    events = create_events(tmp_path)

    sorted_events = events.sort(os.path.join(tmp_path, "sorted_events"), num_threads=2)

    with sorted_events.reader() as reader:
        all_sorted_events = list(reader)

    assert sorted(all_sorted_events) == sorted(all_events)

    for reader_func in sorted_events.sharded_readers():
        with reader_func() as reader:
            s_events = list(reader)
            assert sorted(s_events, key=lambda a: (a[0], a[1].start)) == s_events


def test_patients(tmp_path: pathlib.Path) -> None:
    patients = create_patients(tmp_path)

    with patients.reader() as reader:
        all_patients = list(reader)

    assert sorted(p.patient_id for p in all_patients) == sorted(range(10, 25))

    for patient in all_patients:
        assert patient.events == dummy_events


def transform_func(a: femr.datasets.RawPatient) -> Optional[femr.datasets.RawPatient]:
    if a.patient_id == 10:
        return None
    return femr.datasets.RawPatient(
        patient_id=a.patient_id,
        events=[
            femr.datasets.RawEvent(
                start=event.start,
                concept_id=event.concept_id,
                value="foo",
            )
            for event in a.events
        ],
    )


def test_transform_patients(tmp_path: pathlib.Path) -> None:
    patients = create_patients(tmp_path)

    transformed_patients = patients.transform(os.path.join(tmp_path, "transformed_patients"), transform_func)

    with transformed_patients.reader() as reader:
        all_patients = list(reader)

    assert set(p.patient_id for p in all_patients) == set(range(11, 25))

    for patient in all_patients:
        better_dummy_events = [
            femr.datasets.RawEvent(
                start=event.start,
                concept_id=event.concept_id,
                value="foo",
            )
            for event in dummy_events
        ]
        assert sorted(patient.events) == sorted(better_dummy_events)
