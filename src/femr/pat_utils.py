import datetime

import meds


# assume that first patient event is their birthdate
def get_patient_birthdate(patient: meds.Patient) -> datetime.datetime:
    if len(patient["events"]) > 0:
        return patient["events"][0]["time"]
    raise ValueError("Couldn't find patient birthdate -- Patient has no events")
