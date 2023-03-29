"""
FEMR is primarily designed to work with OMOP datasources, but it is possible to provide
a simpler input format that will work with some parts of the FEMR pipeline.

This tutorial documents that simple schema and shows how to use it.

The input schema is a folder of csv files, where each csv file has at minimum the following columns:

patient_id, start, code, value

Each row in a file corresponds to an event.

- patient_id is some id for the patient that has the event
  patient_id must be a 64 bit unsigned integer

- start is the start timestamp for that event, ideally when the event is initially recorded in the database.
  start must be an ISO 8601 timestamp string

- code is a string identifying what the event is. It must internally consist of a vocab signifier and the code itself, split by a "/" character.
  For example ICD10CM/E11.4

- value is a value associated with the event. It can either be a numeric value, an arbitrary string, or an empty string (indicating no value).

You may also add arbitrary columns for any csv file. Those will be added to each event. The columns can vary between csv files.
We recommend adding columns to note dosage, visit_ids, and lab units, source Clarity tables, etc, etc.

The first row (in time) for each patient is considered their birth event.

Ordering of rows for each patient, or patient rows being split across files doesn't matter.
Everything will be resorted and joined as part of the ETL process.
"""

import os

TARGET_DIR = 'trash/simple_femr'
os.mkdir(TARGET_DIR)

INPUT_DIR = os.path.join(TARGET_DIR, 'simple_input')
os.mkdir(INPUT_DIR)

"""
Write an example file.
"""

with open(os.path.join(INPUT_DIR, 'example.csv'), 'w') as f:
    f.write('patient_id,start,code,value,dosage\n')
    f.write('2,1994-01-03,Birth/Birth,,\n') # First event is always birth
    
    f.write('2,1994-03-06,Drug/Atorvastatin,,50mg\n') # Example usage of dosage
    
    f.write('2,1994-02-03,ICD10CM/E11.4,,\n') # Note how events can be out of order 

    f.write('2,1994-07-09,Vitals/Blood Pressure,150,\n') # Example use of a numeric value

"""
Convert the directory to an extract
"""

LOG_DIR = os.path.join(TARGET_DIR, 'logs')
EXTRACT_DIR = os.path.join(TARGET_DIR, 'extract')

os.system(f"etl_simple_femr {INPUT_DIR} {EXTRACT_DIR} {LOG_DIR} --num_threads 2")

"""
Open and look at the data.
"""

import femr.datasets

database = femr.datasets.PatientDatabase(EXTRACT_DIR)

# We have one patient
print("Num patients", len(database))

# Print out that patient
patient = database[0]
print(patient)

# Note that the patient ids get remapped, you can unmap with the database
original_id = database.get_original_patient_id(0)
print("Oiringla id for patient 0", original_id)

# Also note that concepts have been mapped to integers
print("What code 3 means", database.get_code_dictionary()[3]) # Returns what code=0 means in the database

# You can pull things like dosage by looking at the event
for event in patient.events:
    print(event, event.dosage)