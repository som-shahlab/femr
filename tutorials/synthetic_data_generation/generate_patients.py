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

    parser = argparse.ArgumentParser(prog="generate_patients", description="Create synthetic data")
    parser.add_argument("athena", type=str)
    parser.add_argument("destination", type=str)
    args = parser.parse_args()

    random.seed(4533)

    def get_random_patient(patient_id):
        epoch = datetime.datetime(1990, 1, 1)
        birth = epoch + datetime.timedelta(days=random.randint(100, 1000))
        current_date = birth

        gender = "Gender/" + random.choice(["F", "M"])
        race = "Race/" + random.choice(["White", "Non-White"])

        patient = {
            "patient_id": patient_id,
            "events": [],
        }

        birth_codes = [meds.birth_code, gender, race]

        for birth_code in birth_codes:
            patient["events"].append({"time": birth, "code": birth_code})

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
            patient["events"].append({"time": current_date, "code": code})

        return patient

    patients = []
    for i in range(200):
        patients.append(get_random_patient(i))

    patient_schema = meds.schema.patient_schema()

    patient_table = pyarrow.Table.from_pylist(patients, patient_schema)

    os.makedirs(os.path.join(args.destination, "data"), exist_ok=True)

    pyarrow.parquet.write_table(patient_table, os.path.join(args.destination, "data", "patients.parquet"))

    metadata = {
        "dataset_name": "femr synthetic datata",
        "dataset_version": "1",
        "etl_name": "synthetic data",
        "etl_version": "1",
        "code_metadata": {},
    }

    jsonschema.validate(instance=metadata, schema=meds.dataset_metadata)

    with open(os.path.join(args.destination, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print("Converting")
    os.system(f"convert_to_meds_reader {args.destination} {args.destination}_meds")

    print("Opening database")

    with meds_reader.PatientDatabase(args.destination + "_meds", num_threads=6) as database:
        print("Creating ontology")
        ontology = femr.ontology.Ontology(args.athena)

        print("Pruning ontology")
        ontology.prune_to_dataset(database, remove_ontologies=("SNOMED"))

        with open(os.path.join(args.destination, "ontology.pkl"), "wb") as f:
            pickle.dump(ontology, f)
