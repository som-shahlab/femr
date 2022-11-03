import datetime

import piton


def test_event_to_string():
    e = piton.Event(datetime.datetime(1900, 1, 3), code=3)
    assert str(e) == "Event(start=1900-01-03 00:00:00, code=3)"

    e = piton.Event(datetime.datetime(1900, 1, 3), code=3, value=3)
    assert str(e) == "Event(start=1900-01-03 00:00:00, code=3, value=3)"

    e = piton.Event(
        datetime.datetime(1900, 1, 3), code=3, value=memoryview(b"test")
    )
    assert str(e) == "Event(start=1900-01-03 00:00:00, code=3, value='test')"
