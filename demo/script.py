import piton
import piton.fileio
import piton.transforms
import datetime
import os
import json

import dataclasses

#####################################
##### Part 1: Create events #########
#####################################

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
        


#####################################
##### Part 2: Create patients #######
#####################################

if not os.path.exists('raw_patients'):
    os.mkdir('raw_patients')

# Converts the events directory "events" to the patient directory "raw_patients"
piton.transforms.convert_events_to_patients('events', 'raw_patients', shards=10)


#############################################################
##### Part 3: Apply a transformation to the  patients #######
#############################################################

if not os.path.exists('processed_patients'):
    os.mkdir('processed_patients')

def transform(input: piton.Patient) -> piton.Patient:
    input.events = [a for a in input.events if a.code != 'E11.4'] # Remove E11.4 for some reason
    return input

piton.transforms.transform_patients('raw_patients', 'processed_patients', transform)


#####################################################################
##### Part 4: Convert the patients to a patient collection ##########
#####################################################################

piton.transforms.convert_patients_to_patient_collection('processed_patients', 'patient_collection')

with piton.fileio.PatientCollectionReader('patient_collection') as collection:
    # The key part of a patient collection is that it supports queries
    patient = collection.get_patient(10)

    
    
print(patient)

print(patient.birth_date)

for event in patient.events:
    print(event.start_age, event.code, event.event_type)
