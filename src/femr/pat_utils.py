import datetime

import meds
import meds_reader


def get_subject_birthdate(subject: meds_reader.Subject) -> datetime.datetime:
    for e in subject.events:
        if e.code == meds.birth_code:
            return e.time
    raise ValueError("Couldn't find subject birthdate -- Subject has no events " + str(subject.events[:5]))
