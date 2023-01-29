import argparse
import datetime
import os
import pickle
import time
from typing import List, Optional

import piton
import piton.datasets
from piton.featurizers.core import FeaturizerList
from piton.featurizers.featurizers import AgeFeaturizer, CountFeaturizer
from piton.labelers.core import NLabelsPerPatientLabeler, TimeHorizon
from piton.labelers.omop import HighHbA1cCodeLabeler, LupusCodeLabeler
from piton.labelers.omop_inpatient_admissions import (
    DummyAdmissionDischargeLabeler,
    InpatientLongAdmissionLabeler,
    InpatientMortalityLabeler,
    InpatientReadmissionLabeler,
)
from piton.labelers.omop_lab_values import (
    AnemiaLabValueLabeler,
    HyperkalemiaLabValueLabeler,
    HypoglycemiaLabValueLabeler,
    HyponatremiaLabValueLabeler,
    ThrombocytopeniaLabValueLabeler,
)

"""
Example running:

    # /local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract_v5 \

To generate admission/discharge placeholder labels on 1% extract:

    python3 tutorials/1_run_featurizers.py \
        /local-scratch/nigam/projects/mwornow/data/1_perct_extract_01_11_23 \
        /local-scratch/nigam/projects/clmbr_text_assets/data/features/admission_discharge/ \
        --labeling_function admission_discharge \
        --num_threads 20

To generate admission/discharge placeholder labels on 100% extract:

    python3 tutorials/1_run_featurizers.py \
        /local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract_v5 \
        /local-scratch/nigam/projects/clmbr_text_assets/data/features/admission_discharge/ \
        --labeling_function admission_discharge \
        --num_threads 20

To run a real labeler:

    python3 tutorials/1_run_featurizers.py \
        /local-scratch/nigam/projects/mwornow/data/1_perct_extract_01_11_23 \
        /local-scratch/nigam/projects/clmbr_text_assets/data/features/lupus/ \
        --labeling_function lupus \
        --max_labels_per_patient 5 \
        --num_threads 20

    python3 tutorials/1_run_featurizers.py \
        /local-scratch/nigam/projects/mwornow/data/1_perct_extract_01_11_23 \
        /local-scratch/nigam/projects/clmbr_text_assets/data/features/thrombocytopenia_lab/ \
        --labeling_function thrombocytopenia_lab \
        --max_labels_per_patient 5 \
        --num_threads 20

    python3 tutorials/1_run_featurizers.py \
        /local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract_v5 \
        /local-scratch/nigam/projects/clmbr_text_assets/data/features/hyperkalemia_lab/ \
        --labeling_function hyperkalemia_lab \
        --max_labels_per_patient 5 \
        --num_threads 20

    python3 tutorials/1_run_featurizers.py \
        /local-scratch/nigam/projects/mwornow/data/1_perct_extract_01_11_23 \
        /local-scratch/nigam/projects/clmbr_text_assets/data/features/hypoglycemia_lab/ \
        --labeling_function hypoglycemia_lab \
        --max_labels_per_patient 5 \
        --num_threads 20

    python3 tutorials/1_run_featurizers.py \
        /local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract_v5 \
        /local-scratch/nigam/projects/clmbr_text_assets/data/features/hyponatremia_lab/ \
        --labeling_function hyponatremia_lab \
        --max_labels_per_patient 5 \
        --num_threads 20

    python3 tutorials/1_run_featurizers.py \
        /local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract_v5 \
        /local-scratch/nigam/projects/clmbr_text_assets/data/features/anemia_lab/ \
        --labeling_function anemia_lab \
        --max_labels_per_patient 5 \
        --num_threads 20
"""


def save_to_pkl(object_to_save, path_to_file: str):
    """Save object to pkl file."""
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "wb") as fd:
        pickle.dump(object_to_save, fd)


LABELING_FUNCTIONS: List[str] = [
    # All admission/discharge times
    "admission_discharge",
    # CLMBR code-based tasks
    "mortality",
    "long_los",
    "readmission",
    # Other code-based tasks
    "lupus",
    "high_hba1c",
    # Lab-value tasks
    "thrombocytopenia_lab",
    "hyperkalemia_lab",
    "hypoglycemia_lab",
    "hyponatremia_lab",
    "anemia_lab",
]

if __name__ == "__main__":
    START_TIME = time.time()

    def print_log(name: str, content: str):
        print(f"{int(time.time() - START_TIME)} | {name} | {content}")

    parser = argparse.ArgumentParser(description="Run Piton featurization")

    parser.add_argument(
        "path_to_patient_database",
        type=str,
        help="Path of folder to the Piton PatientDatabase. Example: '/local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract_v5/'",
    )

    parser.add_argument(
        "path_to_output_dir",
        type=str,
        help=(
            "Path to save files output by featurizer."
            " This folder will contain these files: labeled_patients.pkl, preprocessed_featurizers.pkl, and featurized_patients.pkl."
            " Example: '/local-scratch/nigam/projects/rthapa84/data/mortality/'"
        ),
    )

    parser.add_argument(
        "--labeling_function",
        type=str,
        help="Name of labeling function to create.",
        choices=LABELING_FUNCTIONS,
        default=LABELING_FUNCTIONS[0],
    )

    parser.add_argument(
        "--num_threads",
        type=int,
        help="The number of threads to use",
        default=1,
    )

    parser.add_argument(
        "--max_labels_per_patient",
        type=int,
        help="Max number of labels to keep per patient (excess labels are randomly discarded)",
        default=None,
    )

    parser.add_argument(
        "--num_patients",
        type=int,
        help="Number of patients to use (used for DEBUGGING)",
        default=None,
    )

    # Parse CLI args
    args = parser.parse_args()
    PATH_TO_PATIENT_DATABASE: str = args.path_to_patient_database
    PATH_TO_OUTPUT_DIR: str = args.path_to_output_dir
    NUM_THREADS: int = args.num_threads
    NUM_PATIENTS: Optional[int] = args.num_patients
    MAX_LABELS_PER_PATIENT: int = args.max_labels_per_patient

    # create directories to save files
    PATH_TO_SAVE_LABELED_PATIENTS: str = os.path.join(PATH_TO_OUTPUT_DIR, "labeled_patients.pkl")
    PATH_TO_SAVE_PREPROCESSED_FEATURIZERS: str = os.path.join(PATH_TO_OUTPUT_DIR, "preprocessed_featurizers.pkl")
    PATH_TO_SAVE_FEATURIZED_PATIENTS: str = os.path.join(PATH_TO_OUTPUT_DIR, "featurized_patients.pkl")
    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)

    # Load PatientDatabase + Ontology
    database = piton.datasets.PatientDatabase(PATH_TO_PATIENT_DATABASE)
    ontology = database.get_ontology()
    print_log("PatientDatabase", "Loaded from: " + PATH_TO_PATIENT_DATABASE)

    # Define the labeling function.
    year_time_horizon: TimeHorizon = TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=365))
    if args.labeling_function == "admission_discharge":
        labeler = DummyAdmissionDischargeLabeler(ontology)
    elif args.labeling_function == "mortality":
        labeler = InpatientMortalityLabeler(ontology)
    elif args.labeling_function == "long_los":
        labeler = InpatientReadmissionLabeler(ontology)
    elif args.labeling_function == "readmission":
        labeler = InpatientReadmissionLabeler(ontology)
    elif args.labeling_function == "lupus":
        labeler = LupusCodeLabeler(ontology, year_time_horizon)
    elif args.labeling_function == "thrombocytopenia_lab":
        labeler = ThrombocytopeniaLabValueLabeler(ontology, "severe")
    elif args.labeling_function == "hyperkalemia_lab":
        labeler = HyperkalemiaLabValueLabeler(ontology, "severe")
    elif args.labeling_function == "hypoglycemia_lab":
        labeler = HypoglycemiaLabValueLabeler(ontology, "severe")
    elif args.labeling_function == "hyponatremia_lab":
        labeler = HyponatremiaLabValueLabeler(ontology, "severe")
    elif args.labeling_function == "anemia_lab":
        labeler = AnemiaLabValueLabeler(ontology, "severe")
    else:
        raise ValueError(
            f"Labeling function `{args.labeling_function}` not supported. Must be one of: {LABELING_FUNCTIONS}."
        )
    print_log("Labeler", f"Using Labeler `{args.labeling_function}`")

    # Determine how many labels to keep per patient
    if not args.labeling_function == "admission_discharge":
        # Don't throw out labels for admission/discharge placeholder, otherwise
        # defeats the purpose of this labeler
        labeler = NLabelsPerPatientLabeler(labeler, seed=0, num_labels=MAX_LABELS_PER_PATIENT)
        print_log(
            "Labeler",
            f"Keeping max of {MAX_LABELS_PER_PATIENT} labels per patient",
        )
    else:
        print_log("Labeler", f"Keeping ALL labels per patient")

    print_log("Labeling Patients", "Starting")
    labeled_patients = labeler.apply(
        path_to_patient_database=PATH_TO_PATIENT_DATABASE,
        num_threads=NUM_THREADS,
        num_patients=NUM_PATIENTS,
    )
    save_to_pkl(labeled_patients, PATH_TO_SAVE_LABELED_PATIENTS)
    print_log("Labeling Patients", "Finished")
    print("Length of labeled_patients", len(labeled_patients))

    # Lets use both age and count featurizer
    age = AgeFeaturizer()
    count = CountFeaturizer(is_ontology_expansion=True)
    featurizer_age_count = FeaturizerList([age, count])

    # Preprocessing the featurizers, which includes processes such as normalizing age.
    print_log("Preprocessing Featurizer", "Starting")
    featurizer_age_count.preprocess_featurizers(PATH_TO_PATIENT_DATABASE, labeled_patients, NUM_THREADS)
    save_to_pkl(featurizer_age_count, PATH_TO_SAVE_PREPROCESSED_FEATURIZERS)
    print_log("Preprocessing Featurizer", "Finished")

    print_log("Featurize Patients", "Starting")
    results = featurizer_age_count.featurize(PATH_TO_PATIENT_DATABASE, labeled_patients, NUM_THREADS)
    save_to_pkl(results, PATH_TO_SAVE_FEATURIZED_PATIENTS)
    print_log("Featurize Patients", "Finished")

    print_log("FINISH", "Done")
