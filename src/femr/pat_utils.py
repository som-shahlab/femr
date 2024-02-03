import datetime

import meds


def get_patient_birthdate(patient: meds.Patient) -> datetime.datetime:
    for e in patient["events"]:
        for m in e["measurements"]:
            if m["code"] == meds.birth_code:
                return e["time"]
    raise ValueError("Couldn't find patient birthdate -- Patient has no birth events (which must be {meds.birth_code}): " + str(patient["events"][:5]))
