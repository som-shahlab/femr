import argparse
import datetime
import json
import os
import pickle
import random

import jsonschema
import meds
import meds_reader
import pyarrow
import pyarrow.parquet

import femr.ontology
import femr.transforms

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="generate_subjects", description="Create synthetic data")
    parser.add_argument("athena", type=str)
    parser.add_argument("destination", type=str)
    args = parser.parse_args()

    random.seed(4533)

    def get_random_subject(subject_id):
        epoch = datetime.datetime(1990, 1, 1)
        birth = epoch + datetime.timedelta(days=random.randint(100, 1000))
        current_date = birth

        gender = "Gender/" + random.choice(["F", "M"])
        race = "Race/" + random.choice(["White", "Non-White"])

        rows = []

        birth_codes = [meds.birth_code, gender, race]

        for birth_code in birth_codes:
            rows.append({"subject_id": subject_id, "time": birth, "code": birth_code})

        code_cats = ["ICD9CM", "RxNorm"]
        for code in range(random.randint(1, 10 + (20 if gender == "Gender/F" else 0))):
            code_cat = random.choice(code_cats)
            if code_cat == "RxNorm":
                code = str(random.randint(0, 10000))
            else:
                code = str(random.randint(0, 10000))
                if len(code) > 3:
                    code = code[:3] + "." + code[3:]
            current_date = current_date + datetime.timedelta(days=random.randint(1, 100))
            code = code_cat + "/" + code
            rows.append({"subject_id": subject_id, "time": current_date, "code": code})

        return rows

    subjects = []
    for i in range(200):
        subjects.extend(get_random_subject(i))

    subject_schema = meds.schema.data_schema()

    subject_table = pyarrow.Table.from_pylist(subjects, subject_schema)

    os.makedirs(os.path.join(args.destination, "data"), exist_ok=True)
    os.makedirs(os.path.join(args.destination, "metadata"), exist_ok=True)

    pyarrow.parquet.write_table(subject_table, os.path.join(args.destination, "data", "subjects.parquet"))

    metadata = {
        "dataset_name": "femr synthetic datata",
        "dataset_version": "1",
        "etl_name": "synthetic data",
        "etl_version": "1",
        "code_metadata": {},
    }

    jsonschema.validate(instance=metadata, schema=meds.dataset_metadata_schema)

    with open(os.path.join(args.destination, "metadata", "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print("Converting")
    os.system(f"meds_reader_convert {args.destination} {args.destination}_meds")

    print("Opening database")

    with meds_reader.SubjectDatabase(args.destination + "_meds", num_threads=6) as database:
        print("Creating ontology")
        ontology = femr.ontology.Ontology(args.athena)

        print("Pruning ontology")
        ontology.prune_to_dataset(database, remove_ontologies=("SNOMED"))

        with open(os.path.join(args.destination, "ontology.pkl"), "wb") as f:
            pickle.dump(ontology, f)
