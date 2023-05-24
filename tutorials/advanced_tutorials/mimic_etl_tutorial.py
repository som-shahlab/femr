"""
A simple ETL script that documents the minimum set of OMOP required for FEMR to function and provides an example
converting MIMIC to that minimal OMOP subset.

Note that this ETL is very minimal and is not intended to be a full OMOP ETL.

In particular, it doesn't try to harmonize codes to standard concepts at all.
"""

import csv
import dataclasses
import datetime
import io
import itertools
import os
import pickle
from typing import List, Optional, Tuple

TARGET_DIR = "trash/mimic_omop_conversion"

"""
We assume you have already donwloaded a copy of MIMIC to MIMIC_DIR
"""

MIMIC_DIR = "/local-scratch/nigam/projects/clmbr_text_assets/data/mimic_iii/mimic-iii-unzipped"
# Simple flag to do a partial extract to run faster
PARTIAL_EXTRACT = True


def verbose_mimic_open(mimic_table, read_full=False):
    """A utility to verbosely open a mimic file, with support for partial reads."""
    print("Processing", mimic_table)
    file = open(os.path.join(MIMIC_DIR, mimic_table + ".csv"))

    if not read_full and PARTIAL_EXTRACT:
        lines = [file.readline() for _ in range(10000)]
        file.close()
        return io.StringIO("\n".join(lines))
    else:
        return file


OMOP_DIR = os.path.join(TARGET_DIR, "mimic_omop")
EXTRACT_DIR = os.path.join(TARGET_DIR, "mimic_extract")

os.makedirs(OMOP_DIR, exist_ok=True)

"""
First, we need to start with an existing Athena ontology dump.

Athena is the OHDSI vocab management tool and it has a download feature to get ontology dumps.

See https://athena.ohdsi.org/

This tutorial assumes that this download has already been done and the corresponding Athena dump can be found at VOCAB_DIR
"""

VOCAB_DIR = "/local-scratch/nigam/projects/ethanid/mimic_tutorial_data/athena"

print("Reading the OMOP vocab for initial use")
code_to_concept_id_map = {}

with open(os.path.join(VOCAB_DIR, "concept.csv")) as f:
    # Note that Athena annoyingly provides their "csvs" in tab delimited form ...
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        code_to_concept_id_map[row["vocabulary_id"] + "/" + row["concept_code"]] = row["concept_id"]


"""
Note that the existing Athena dump will not contain all the codes we need.

So we will inevetiably have to add some custom codes of our own.

We keep track of these extra codes using an extra codes list, to be added to the concept table at the end.
"""
extra_code_offset = int(1e8)
extra_codes: List[Tuple[str, str]] = []


def get_concept_id(code, description=None, require_exists=False):
    """Get the concept id for a given code. If the code is not in the current concept list, it gets added."""
    if description is None:
        description = code
    if code not in code_to_concept_id_map:
        assert not require_exists
        # Need to manually add this code.
        code_to_concept_id_map[code] = extra_code_offset + len(extra_codes)
        extra_codes.append((code, description))

    return code_to_concept_id_map[code]


"""
We want to manually handle certain MIMIC concepts to provide additional metadata.
In a future ETL you would want to actually try to map these to OMOP standard concepts.
"""
for item_source in ("D_ITEMS", "D_LABITEMS"):
    with verbose_mimic_open(item_source, read_full=True) as rf:
        reader = csv.DictReader(rf)
        for row in reader:
            code = f"MIMIC-Item/{row['ITEMID']}"
            description = row["LABEL"]

            get_concept_id(code, description)

"""
The main part of the ETL begins.

The overall idea is that we process all the MIMIC tables, pulling out events of interest.

We will then insert these events in the OMOP observation table, which can hold arbitrary events.

Note that there are some special cases where we need to put some data in the person, visit_occurrence and
drug_exposure tables.
"""


@dataclasses.dataclass
class EventData:
    """A helper class to hold onto events we want to save."""

    person_id: str
    code: str
    time: str
    visit_id: str = ""
    value: str = ""

    def __post_init__(self) -> None:
        assert self.time != ""

        assert datetime.datetime.fromisoformat(self.time) is not None


# Keep track of events to handle later
events: List[EventData]
events = []

# Process the patient table
with verbose_mimic_open("PATIENTS", read_full=True) as rf:
    reader = csv.DictReader(rf)

    # Note that we also need to create an OMOP person table for special data from this table
    with open(os.path.join(OMOP_DIR, "person.csv"), "w") as wf:
        fields = ["person_id", "birth_datetime"]
        for t in ["gender", "ethnicity", "race"]:
            fields.append(t + "_concept_id")
        writer = csv.DictWriter(wf, fields)
        writer.writeheader()

        for row in reader:
            gender_code_map = {
                "M": "Gender/M",
                "F": "Gender/F",
            }
            writer.writerow(
                {
                    "person_id": row["SUBJECT_ID"],
                    "birth_datetime": row["DOB"],
                    "gender_concept_id": get_concept_id(gender_code_map[row["GENDER"]], require_exists=True),
                    "ethnicity_concept_id": 0,
                    "race_concept_id": 0,
                }
            )

            # For some reason, mimic also stores deaths in the PATIENTS table
            if row["EXPIRE_FLAG"] == "1":
                events.append(
                    EventData(person_id=row["SUBJECT_ID"], code="Condition Type/OMOP4822053", time=row["DOD"])
                )

admission_end_map = {}

# Process the admissions
with verbose_mimic_open("ADMISSIONS", read_full=True) as rf:
    with open(os.path.join(OMOP_DIR, "visit_occurrence.csv"), "w") as wf:
        reader = csv.DictReader(rf)
        writer = csv.DictWriter(
            wf,
            fieldnames=[
                "person_id",
                "visit_occurence_id",
                "visit_concept_id",
                "visit_start_datetime",
                "visit_end_datetime",
            ],
        )
        writer.writeheader()

        for row in reader:
            admission_end_map[row["HADM_ID"]] = row["DISCHTIME"]
            writer.writerow(
                {
                    "person_id": row["SUBJECT_ID"],
                    "visit_occurence_id": row["HADM_ID"],
                    "visit_concept_id": get_concept_id("MIMIC-Admission_type/" + row["ADMISSION_TYPE"]),
                    "visit_start_datetime": row["ADMITTIME"],
                    "visit_end_datetime": row["DISCHTIME"],
                }
            )

            events.append(
                EventData(
                    person_id=row["SUBJECT_ID"],
                    code="MIMIC-Discharge_location/" + row["DISCHARGE_LOCATION"],
                    time=row["DISCHTIME"],
                    visit_id=row["HADM_ID"],
                )
            )

            for field in ["Admission_location", "Insurance", "Language", "Religion", "Marital_status", "Ethnicity"]:
                events.append(
                    EventData(
                        person_id=row["SUBJECT_ID"],
                        code=f"MIMIC-{field}/{row[field.upper()]}",
                        time=row["ADMITTIME"],
                        visit_id=row["HADM_ID"],
                    )
                )


# Process a bunch of tables that are all based on ITEMIDs
for table in (
    "CHARTEVENTS",
    "DATETIMEEVENTS",
    "INPUTEVENTS_CV",
    "INPUTEVENTS_MV",
    "LABEVENTS",
    "OUTPUTEVENTS",
    "PROCEDUREEVENTS_MV",
):
    with verbose_mimic_open(table) as rf:
        reader = csv.DictReader(rf)
        for row in reader:
            code = f"MIMIC-Item/{row['ITEMID']}"

            # Make sure the code is in the dictionary ...
            get_concept_id(code, require_exists=True)

            events.append(
                EventData(
                    person_id=row["SUBJECT_ID"],
                    code=code,
                    time=row.get("STORETIME") or row["CHARTTIME"],
                    visit_id=row["HADM_ID"],
                    value=row.get("VALUENUM") or row.get("VALUE") or row.get("AMOUNT"),
                )
            )

# Process CPT events (which are billing codes that are tied to an actual date)
with verbose_mimic_open("CPTEVENTS") as rf:
    reader = csv.DictReader(rf)
    for row in reader:
        events.append(
            EventData(
                person_id=row["SUBJECT_ID"],
                code=f"MIMIC-CPT/{row['CPT_CD']}",
                time=row["CHARTDATE"] or admission_end_map[row["HADM_ID"]],
                visit_id=row["HADM_ID"],
            )
        )

# Process visit oriented billing codes
for billing_table, billing_type in (("DIAGNOSES_ICD", "ICD9"), ("PROCEDURES_ICD", "ICD9"), ("DRGCODES", "DRG")):
    with verbose_mimic_open(billing_table) as rf:
        reader = csv.DictReader(rf)
        for row in reader:
            events.append(
                EventData(
                    person_id=row["SUBJECT_ID"],
                    code=f"MIMIC-{billing_type}/{row[billing_type + '_CODE']}",
                    time=admission_end_map[row["HADM_ID"]],
                    visit_id=row["HADM_ID"],
                )
            )

# Process the prescriptions
with verbose_mimic_open("PRESCRIPTIONS") as rf:
    with open(os.path.join(OMOP_DIR, "drug_exposure.csv"), "w") as wf:
        reader = csv.DictReader(rf)
        writer = csv.DictWriter(
            wf,
            fieldnames=[
                "person_id",
                "visit_occurence_id",
                "drug_concept_id",
                "drug_exposure_start_datetime",
                "drug_exposure_end_datetime",
            ],
        )
        writer.writeheader()

        for row in reader:
            writer.writerow(
                {
                    "person_id": row["SUBJECT_ID"],
                    "visit_occurence_id": row["HADM_ID"],
                    "drug_concept_id": get_concept_id("MIMIC-Drug/" + row["DRUG"]),
                    "drug_exposure_start_datetime": row["STARTDATE"],
                    "drug_exposure_end_datetime": row["ENDDATE"],
                }
            )

"""
Write out the obtained events.
The observation table in OMOP can be used as a generic place to store events
"""

with open(os.path.join(OMOP_DIR, "observation.csv"), "w") as wf:
    writer = csv.DictWriter(
        wf,
        fieldnames=[
            "person_id",
            "observation_concept_id",
            "observation_datetime",
            "value_as_string",
            "visit_occurrence_id",
        ],
    )
    writer.writeheader()

    for event in events:
        writer.writerow(
            {
                "person_id": event.person_id,
                "observation_concept_id": get_concept_id(event.code),
                "observation_datetime": event.time,
                "value_as_string": event.value or "",
                "visit_occurrence_id": event.visit_id or "",
            }
        )


"""
FEMR relies on a subset of OMOP's vocabulary.

In particular, we need the concept and concept_relationship tables.

Here we populate those tables, in actual csv form.
"""

print("Copying the vocab tables")


def copy_and_convert_to_csv(name, extra_rows=[]):
    with open(os.path.join(OMOP_DIR, name), "w") as o:
        with open(os.path.join(VOCAB_DIR, name)) as i:
            reader = csv.reader(i, delimiter="\t")
            writer = csv.writer(o)

            for row in itertools.chain(reader, extra_rows):
                writer.writerow(row)


# Add our custom concepts to the corresponding tables
extra_concept_rows = [
    [
        extra_code_offset + i,
        description,
        "Custom",
        code.split("/")[0],
        "Custom",
        "",
        "/".join(code.split("/")[1:]),
        "",
        "",
        "",
    ]
    for i, (code, description) in enumerate(extra_codes)
]

copy_and_convert_to_csv("concept.csv", extra_concept_rows)
copy_and_convert_to_csv("concept_relationship.csv")


"""
Finally ready to perform the final stage of ETL.

As we have converted to OMOP, we can use the generic OMOP pipeline.
"""

os.system(f"etl_generic_omop {OMOP_DIR} {EXTRACT_DIR} {EXTRACT_DIR}_logs")

"""
We can now load and examine the resulting extract.
"""

import femr.datasets

database = femr.datasets.PatientDatabase(EXTRACT_DIR)

print("The database has", len(database), "patients")
