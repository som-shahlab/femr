from typing import Sequence

from femr import Event, Patient
from femr.datasets import EventCollection, PatientCollection
from femr.datasets.fileio import PatientWriter
from femr.extension.datasets import PatientDatabase
from femr.extractors.csv import run_csv_extractors


def create_patient_database(patient: Sequence[Patient], tmp_path="./") -> PatientDatabase:
    """Create a PatientDatabase from a sequence of Patients."""

    # Create event collection
    # TODO: This requires raw csv files ...
    event_collection = run_csv_extractors(...)

    # Create patient collection
    patient_collection = event_collection.to_patient_collection()

    # Create patient database

    return patient_collection.to_patient_database()
