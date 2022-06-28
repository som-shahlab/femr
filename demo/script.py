import piton
import piton.fileio
import piton.transform
import datetime
import os
import json

import dataclasses

if not os.path.exists('events'):
    os.mkdir('events')

patient_id = 10

dummy_events = [
    piton.Event(start = datetime.date(1995, 1, 3), code='birth', event_type = 'birth'),
    piton.Event(start = datetime.date(2010, 1, 3), code='E11.4', event_type = 'diagnosis_code'),
    piton.Event(start = datetime.date(2010, 1, 5), code='C80.1', event_type = 'diagnosis_code'),
]

with piton.fileio.EventWriter('events/example.csv.gz') as w:
    for dummy_event in dummy_events:
        w.add_event(patient_id, dummy_event)

if not os.path.exists('raw_patients'):
    os.mkdir('raw_patients')

piton.transform.convert_events_to_patients('events', 'raw_patients', 10)

if not os.path.exists('processed_patients'):
    os.mkdir('processed_patients')

def transform(input: piton.Patient) -> piton.Patient:
    input.events = [a for a in input.events if a.code != 'E11.4'] # Remove E11.4 for some reason
    return input

piton.transform.transform_patients('raw_patients', 'processed_patients', transform)

piton.transform.convert_patients_to_patient_collection('processed_patients', 'patient_collection')

with piton.fileio.PatientCollectionReader('patient_collection') as collection:
    patient = collection.get_patient(10)

print(patient)

print(patient.birth_date)

for event in patient.events:
    print(event.start_age, event.code, event.event_type)

# How to convert to JSON

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        elif isinstance(o, datetime.datetime):
            return o.isoformat()
        return super().default(o)

for root, dirs, files in os.walk('processed_patients'):
    for file in files:
        with piton.fileio.PatientReader(os.path.join(root, file)) as r:
            for patient in r.get_patients():
                print(json.dumps(patient, cls=EnhancedJSONEncoder))