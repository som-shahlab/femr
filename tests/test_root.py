import datetime

import piton


def test_event_to_string():
    e = piton.Event(start=datetime.datetime(1900, 1, 3), concept_id=3)
    assert str(e) == "Event(concept_id=3, start=1900-01-03 00:00:00)"

    e = piton.Event(start=datetime.datetime(1900, 1, 3), concept_id=3, value=3)
    assert str(e) == "Event(concept_id=3, start=1900-01-03 00:00:00, value=3)"

    e = piton.Event(
        start=datetime.datetime(1900, 1, 3), concept_id=3, value=3.3
    )
    assert str(e) == "Event(concept_id=3, start=1900-01-03 00:00:00, value=3.3)"

    e = piton.Event(
        start=datetime.datetime(1900, 1, 3),
        concept_id=3,
        value=memoryview(b"test"),
    )
    assert (
        str(e) == "Event(concept_id=3, start=1900-01-03 00:00:00, value='test')"
    )

    e = piton.Event(
        start=datetime.datetime(1900, 1, 3), concept_id=3, value=None
    )
    assert str(e) == "Event(concept_id=3, start=1900-01-03 00:00:00)"
