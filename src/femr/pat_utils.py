import datetime

import meds
import meds_reader


def get_patient_birthdate(patient: meds_reader.Patient) -> datetime.datetime:
    for e in patient.events:
        if e.code == meds.birth_code:
            return e.time
    raise ValueError("Couldn't find patient birthdate -- Patient has no events " + str(patient.events[:5]))
