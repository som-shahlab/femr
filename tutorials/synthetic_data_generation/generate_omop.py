import csv
import dataclasses
import datetime
import io
import itertools
import os
import pickle
import random
from typing import List, Optional, Tuple

OMOP_DIR = "../etl_input/omop"


code_to_concept_id_map = {}

extra_code_offset = 1
extra_codes: List[Tuple[str, str]] = []


def get_concept_id(code):
    """Get the concept id for a given code. If the code is not in the current concept list, it gets added."""
    if code not in code_to_concept_id_map:
        # Need to manually add this code.
        code_to_concept_id_map[code] = extra_code_offset + len(extra_codes)
        extra_codes.append(code)

    return code_to_concept_id_map[code]


birth_dates = []

# Note that we also need to create an OMOP person table for special data from this table
with open(os.path.join(OMOP_DIR, "person.csv"), "w") as wf:
    fields = ["person_id", "birth_datetime"]
    for t in ["gender", "ethnicity", "race"]:
        fields.append(t + "_concept_id")
    writer = csv.DictWriter(wf, fields)
    writer.writeheader()

    for pid in range(100):
        gender = random.choice(["Gender/M", "Gender/F"])
        dob = datetime.datetime(random.randint(1970, 2000), 1, 1)
        birth_dates.append(dob)

        writer.writerow(
            {
                "person_id": pid,
                "birth_datetime": dob,
                "gender_concept_id": get_concept_id(gender),
                "ethnicity_concept_id": 0,
                "race_concept_id": 0,
            }
        )

observation_folder = os.path.join(OMOP_DIR, "observation")

os.makedirs(observation_folder, exist_ok=True)

for j in range(10):
    with open(os.path.join(observation_folder, f"{j}.csv"), "w") as wf:
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

        for pid in range(j * 10, (j + 1) * 10):
            current_time = birth_dates[pid]
            for i in range(20):
                current_time += datetime.timedelta(days=random.randint(100, 200))
                random_code = f"CODE/{random.randint(1000, 5000)}"

                writer.writerow(
                    {
                        "person_id": pid,
                        "observation_concept_id": get_concept_id(random_code),
                        "observation_datetime": current_time,
                        "value_as_string": "",
                        "visit_occurrence_id": "",
                    }
                )

with open(os.path.join(OMOP_DIR, "concept.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "concept_id",
            "concept_name",
            "domain_id",
            "vocabulary_id",
            "concept_class_id",
            "standard_concept",
            "concept_code",
            "valid_start_date",
            "valid_end_date",
            "invalid_reason",
        ]
    )
    for i, code in enumerate(extra_codes):
        writer.writerow(
            [
                extra_code_offset + i,
                code,
                "Custom",
                code.split("/")[0],
                "Custom",
                "",
                "/".join(code.split("/")[1:]),
                "",
                "",
                "",
            ]
        )

with open(os.path.join(OMOP_DIR, "concept_relationship.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["concept_id_1", "concept_id_2", "relationship_id", "valid_start_date", "valid_end_date", "invalid_reason"]
    )
