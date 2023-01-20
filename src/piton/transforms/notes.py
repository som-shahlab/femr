"""Collection of useful transformations for clinical notes featurizers."""

from __future__ import annotations

import datetime
from typing import List, Optional

from .. import Event
from ..featurizers.featurizers_notes import Note
from ..labelers.core import Label


def remove_short_notes(
    notes: List[Note], 
    label: Label, 
    min_char_count: int = 0,
    **kwargs
) -> List[Note]:
    """Remove all notes from `notes` whose character length < `min_char_count`.
    `notes` is a list of tuples, where each tuple is: (event idx of note, Event)
    """
    new_notes: List[Note] = []
    for note in notes:
        text: str = str(note.event.value)
        if len(text) >= min_char_count:
            new_notes.append(note)
    return new_notes


def keep_only_notes_matching_codes(
    notes: List[Note],
    label: Label,
    keep_notes_with_codes: List[int] = [],
    **kwargs,
) -> List[Note]:
    """Keep only notes that have a `code` contained in `keep_notes_with_codes`."""
    new_notes: List[Note] = []
    for note in notes:
        if note.event.code in keep_notes_with_codes:
            new_notes.append(note)
    return new_notes


def remove_notes_after_label(
    notes: List[Note], 
    label: Label,
    **kwargs
) -> List[Note]:
    """Remove all notes whose `start` > `label.time`."""
    new_notes: List[Note] = []
    for note in notes:
        if note.event.start <= label.time:
            new_notes.append(note)
    return new_notes


def join_all_notes(
    notes: List[Note], 
    label: Label, 
    **kwargs
) -> List[Note]:
    """Join all notes from `notes` together into one long string."""
    text: str = " ".join([note[1].value for note in notes])  # type: ignore
    # Give it an arbitrary `start` and `code` (b/c merged notes don't have one)
    last_note_start: datetime.datetime = notes[-1].event.start
    note = Event(start=last_note_start, code=0, value=text)
    return [Note(0, note)]


def keep_only_last_n_chars(
    notes: List[Note], 
    label: Label, 
    keep_last_n_chars: Optional[int] = None,
    **kwargs
) -> List[Note]:
    """Keep the last `n_chars` from each note."""
    if keep_last_n_chars is None:
        return notes
    new_notes: List[Note] = []
    for note in notes:
        text: str = str(note.event.value)
        event = Event(
            start=note.event.start, code=note.event.code, value=text[-keep_last_n_chars:]
        )
        new_notes.append(Note(note.event_idx, event))
    return new_notes
