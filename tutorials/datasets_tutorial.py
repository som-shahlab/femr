# type: ignore
from __future__ import annotations

import contextlib
import datetime
import os
import shutil

import piton
import piton.datasets

# This tutorial covers the two major dataset types in piton, EventCollection
# PatientCollection

target_directory = "dataset_tutorial_target"

if os.path.exists(target_directory):
    shutil.rmtree(target_directory)
os.makedirs(target_directory)


#####################################
# Part 1: Create events             #
#####################################

# Piton stores events within EventCollections, which are an unordered series of events
# Note that there are three types of events, numeric, binary, and None

dummy_events = [
    piton.Event(
        start=datetime.datetime(1995, 1, 3),
        code=0,
        value=float(34),
    ),
    piton.Event(
        start=datetime.datetime(2010, 1, 3),
        code=1,
        value="test_value",
    ),
    piton.Event(
        start=datetime.datetime(2010, 1, 5),
        code=2,
        value=None,
    ),
]

events = piton.datasets.EventCollection(
    os.path.join(target_directory, "events")
)

# Once we create an events object we can just start writing
# Create_writer() creates a writer object that allows you to add events to the EventCollection
# It automatically creates a new file for events
with contextlib.closing(events.create_writer()) as writer:
    for event in dummy_events:
        # Note that these are getting written to disk as part of the EventCollection
        writer.add_event(patient_id=30, event=event)


# We can also iterate over events
# Note that we need to create a reader object as thees are natively stored on disk

with events.reader() as reader:
    for event in reader:
        print(event)


#####################################
# Part 2: Create patients           #
#####################################

# Piton stores patients within PatientCollections
# These are simply generated right from EventCollections

patients = events.to_patient_collection(
    os.path.join(target_directory, "patients")
)

# We can iterate over patients

with patients.reader() as reader:
    for patient in reader:
        print(patient)


#############################################################
# Part 3: Apply a transformation to the patients            #
#############################################################

# Piton allows you to perform modifications on patients in a straightforward manner


def transform(input: piton.Patient) -> piton.Patient:
    return piton.Patient(
        patient_id=input.patient_id,
        events=[
            a
            for a in input.events
            if a.value
            != b"test_value"  # Note that text values are stored as bytes
        ],  # Remove test_value for some reason
    )


transformed_patients: piton.datasets.PatientCollection = patients.transform(
    os.path.join(target_directory, "transformed_patients"), transform
)

with transformed_patients.reader() as reader:
    for patient in reader:
        print(patient)
