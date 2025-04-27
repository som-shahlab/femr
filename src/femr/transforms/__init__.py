"""A collection of general use transforms."""

import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import meds_reader
import meds_reader.transform


def remove_nones(
    subject: meds_reader.transform.MutableSubject,
    do_not_apply_to_filter: Optional[Callable[[meds_reader.Event], bool]] = None,
) -> meds_reader.transform.MutableSubject:
    """Remove duplicate codes w/in same day if duplicate code has None value.

    There is no point having a NONE value in a timeline when we have an actual value within the same day.

    This removes those unnecessary NONE values.
    """
    do_not_apply_to_filter = do_not_apply_to_filter or (lambda _: False)
    has_value: Set[Tuple[str, datetime.date]] = set()

    for event in subject.events:
        value = (event.numeric_value, None if 'text_value' not in event else event.text_value)
        if any(v is not None for v in value):
            has_value.add((event.code, event.time.date()))

    new_events: List[meds_reader.transform.MutableEvent] = []
    for event in subject.events:
        value = (event.numeric_value, None if 'text_value' not in event else event.text_value)
        if (
            all(v is None for v in value)
            and (event.code, event.time.date()) in has_value
            and not do_not_apply_to_filter(event)
        ):
            # Skip this event as already in there
            continue

        new_events.append(event)

    subject.events = new_events
    subject.events.sort(key=lambda a: a.time)

    return subject


def delta_encode(
    subject: meds_reader.transform.MutableSubject,
    do_not_apply_to_filter: Optional[Callable[[meds_reader.Event], bool]] = None,
) -> meds_reader.transform.MutableSubject:
    """Delta encodes the subject.

    The idea behind delta encoding is that if we get duplicate values within a short amount of time
    (1 day for this code), there is not much point retaining the duplicate.

    This code removes all *sequential* duplicates within the same day.
    """
    do_not_apply_to_filter = do_not_apply_to_filter or (lambda _: False)

    last_value: Dict[Tuple[str, datetime.date], Any] = {}

    new_events: List[meds_reader.transform.MutableEvent] = []
    for event in subject.events:
        key = (event.code, event.time.date())
        value = (event.numeric_value, event.text_value)
        if key in last_value and last_value[key] == value and not do_not_apply_to_filter(event):
            continue
        last_value[key] = value
        new_events.append(event)

    subject.events = new_events
    subject.events.sort(key=lambda a: a.time)

    return subject


def fix_events(subject: meds_reader.transform.MutableSubject) -> meds_reader.transform.MutableSubject:
    """After a series of transformations, sometimes the subject structure gets a bit messed up.
    The usual issues are either duplicate event times or missorted events.

    This does a final cleanup pass to meet the MEDS requirements.
    """
    subject.events = sorted(subject.events, key=lambda a: a.time)

    return subject
