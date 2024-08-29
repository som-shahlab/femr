from __future__ import annotations

import collections
import dataclasses
import datetime
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import meds
import meds_reader
import pandas as pd

from femr.labelers import Label, Labeler

# 2nd elem of tuple -- 'skip' means no label, None means censored
EventsWithLabels = List[Tuple[Tuple[Tuple, int, Any], Union[bool, str]]]


DUMMY_EVENTS = [
    ((1995, 1, 3), meds.birth_code, None),
    ((2010, 1, 1), 1, "test_value"),
    ((2010, 1, 1), 1, "test_value"),
    ((2010, 1, 5), 2, 1),
    ((2010, 6, 5), 3, None),
    ((2010, 8, 5), 2, None),
    ((2011, 7, 5), 2, None),
    ((2012, 10, 5), 3, None),
    ((2015, 6, 5, 0), 2, None),
    ((2015, 6, 5, 10, 10), 2, None),
    ((2015, 6, 15, 11), 3, None),
    ((2016, 1, 1), 2, None),
    ((2016, 3, 1, 10, 10, 10), 4, None),
]

NUM_EVENTS = len(DUMMY_EVENTS)
NUM_PATIENTS = 10


@dataclasses.dataclass
class DummyEvent:
    time: datetime.datetime
    code: str
    text_value: Optional[str] = None
    numeric_value: Optional[float] = None
    visit_id: Optional[int] = None
    table: Optional[str] = None
    clarity_table: Optional[str] = None
    end: Optional[datetime.datetime] = None

    def __getattr__(self, name: str) -> Any:
        return None


@dataclasses.dataclass
class DummySubject:
    subject_id: int
    events: Sequence[DummyEvent]


class DummyDatabase(dict):
    def filter(self, subject_ids):
        return DummyDatabase({p: self[p] for p in subject_ids})

    def map(
        self,
        map_func,
    ) -> Any:
        return [map_func(self.values())]

    def map_with_data(self, map_func, data, assume_sorted) -> Any:
        entries = collections.defaultdict(list)

        for row in data.itertuples(index=False):
            entries[row.subject_id].append(row)

        temp = []
        for k, v in entries.items():
            temp.append((self[k], v))

        return [map_func(temp)]


def create_subjects_dataset(
    num_subjects: int, events: List[Tuple[Tuple, Any, Any]] = DUMMY_EVENTS
) -> meds_reader.SubjectDatabase:
    """Creates a list of subjects, each with the same events contained in `events`"""

    converted_events: List[DummyEvent] = []

    for event in events:
        if isinstance(event[1], int):
            code = str(event[1])
        else:
            code = event[1]

        dummy_event = DummyEvent(time=datetime.datetime(*event[0]), code=code)

        if isinstance(event[2], str):
            dummy_event.text_value = event[2]
        else:
            dummy_event.numeric_value = event[2]

        converted_events.append(dummy_event)

    result = DummyDatabase(
        (subject_id, DummySubject(subject_id, converted_events)) for subject_id in range(num_subjects)
    )
    return cast(meds_reader.SubjectDatabase, result)


def assert_labels_are_accurate(
    labeled_subjects: pd.DataFrame,
    subject_id: int,
    true_labels: List[Tuple[datetime.datetime, Optional[bool]]],
    help_text: str = "",
):
    """Passes if the labels in `labeled_subjects` for `subject_id` exactly match the labels in `true_labels`."""
    generated_labels: List[Label] = [a for a in labeled_subjects.itertuples(index=False) if a.subject_id == subject_id]
    # Check that length of lists of labels are the same

    assert len(generated_labels) == len(
        true_labels
    ), f"len(generated): {len(generated_labels)} != len(expected): {len(true_labels)} | {help_text}"
    # Check that value of labels are the same
    for idx, (label, true_label) in enumerate(zip(generated_labels, true_labels)):
        assert label.boolean_value == true_label[1] and label.prediction_time == true_label[0], (
            f"subject_id={subject_id}, label_idx={idx}, label={label}  |  "
            f"{label} (Assigned) != {true_label} (Expected)  |  "
            f"{help_text}"
        )


def run_test_for_labeler(
    labeler: Labeler,
    events_with_labels: EventsWithLabels,
    true_outcome_times: Optional[List[datetime.datetime]] = None,
    true_prediction_times: Optional[List[datetime.datetime]] = None,
    help_text: str = "",
) -> None:
    subjects: meds_reader.SubjectDatabase = create_subjects_dataset(10, [x[0] for x in events_with_labels])

    true_labels: List[Tuple[datetime.datetime, Optional[bool]]] = [
        (datetime.datetime(*x[0][0]), x[1]) for x in events_with_labels if isinstance(x[1], bool)
    ]
    if true_prediction_times is not None:
        # If manually specified prediction times, adjust labels from occurring at `event.start`
        # e.g. we may make predictions at `event.end` or `event.start + 1 day`
        true_labels = [(tp, tl[1]) for (tl, tp) in zip(true_labels, true_prediction_times)]
    labeled_subjects: List[meds.Label] = labeler.apply(subjects)

    # Check accuracy of Labels
    for subject_id in subjects:
        assert_labels_are_accurate(
            labeled_subjects,
            subject_id,
            true_labels,
            help_text=help_text,
        )
