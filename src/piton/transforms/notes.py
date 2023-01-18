"""Collection of useful transformations for clinical notes featurizers."""

from __future__ import annotations

import datetime
from typing import List

from .. import Event
from ..featurizers.featurizers_notes import NotesProcessed
from ..labelers.core import Label


def remove_short_notes(
    notes: NotesProcessed, label: Label, **kwargs
) -> NotesProcessed:
    """Remove all notes from `notes` whose character length < `min_char_count`.
    `notes` is a list of tuples, where each tuple is: (event idx of note, Event)
    """
    min_char_count: int = kwargs.get("min_char_count", 0)
    new_notes: NotesProcessed = []
    for note in notes:
        text: str = str(note[1].value)
        if len(text) >= min_char_count:
            new_notes.append(note)
    return new_notes


def keep_only_notes_matching_codes(
    notes: NotesProcessed,
    label: Label,
    **kwargs,
) -> NotesProcessed:
    """Keep only notes that have a `code` contained in `keep_notes_with_codes`."""
    codes: List[int] = kwargs.get("keep_notes_with_codes", [])
    new_notes: NotesProcessed = []
    for note in notes:
        if note[1].code in codes:
            new_notes.append(note)
    return new_notes


def remove_notes_after_label(
    notes: NotesProcessed, label: Label, **kwargs
) -> NotesProcessed:
    """Remove all notes whose `start` > `label.time`."""
    new_notes: NotesProcessed = []
    for note in notes:
        if note[1].start <= label.time:
            new_notes.append(note)
    return new_notes


def join_all_notes(
    notes: NotesProcessed, label: Label, **kwargs
) -> NotesProcessed:
    """Join all notes from `notes` together into one long string."""
    text: str = " ".join([note[1].value for note in notes])  # type: ignore
    # Give it an arbitrary `start` and `code` (b/c merged notes don't have one)
    note = Event(start=datetime.datetime(0, 0, 0), code=0, value=text)
    return [(0, note)]


def keep_only_last_n_chars(
    notes: NotesProcessed, label: Label, **kwargs
) -> NotesProcessed:
    """Keep the last `n_chars` from each note."""
    n_chars: int = kwargs.get("keep_last_n_chars", None)
    if n_chars is None:
        return notes
    new_notes: NotesProcessed = []
    for note in notes:
        text: str = str(note[1].value)
        event = Event(
            start=note[1].start, code=note[1].code, value=text[:n_chars]
        )
        new_notes.append((note[0], event))
    return new_notes
