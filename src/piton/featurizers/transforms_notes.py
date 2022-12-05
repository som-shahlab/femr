"""Collection of useful transformations for clinical notes featurizers."""

from __future__ import annotations

from typing import List

from .. import Event
from ..labelers.core import Label
from .featurizers_notes import NotesProcessed

def remove_short_notes(
    notes: NotesProcessed, label: Label, **kwargs
) -> NotesProcessed:
    """Remove all notes from `notes` whose character length < `min_char_count`.
    `notes` is a list of tuples, where each tuple is: (event idx of note, Event)
    """
    min_char_count: int = kwargs.get("min_char_count", 0)
    new_notes: NotesProcessed = []
    for note in notes:
        assert isinstance(note[1].value, memoryview)
        text: str = bytes(note[1].value).decode("utf8")
        if len(text) >= min_char_count:
            new_notes.append(note)
    return new_notes

def keep_only_notes_matching_codes(
    notes: NotesProcessed, label: Label, **kwargs,
) -> NotesProcessed:
    """Keep only notes that have a `code` contained in `codes`."""
    codes: List[int] = kwargs.get('codes', [])
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
    text: str = " ".join([bytes(note[1].value).decode("utf8") for note in notes])  # type: ignore
    note = Event(start=0, code=0, value=text)
    return [(0, note)]