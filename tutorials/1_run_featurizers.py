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
from piton.labelers.omop import (
    HighHbA1cCodeLabeler,
    IsMaleLabeler,
    MortalityCodeLabeler,
    LupusCodeLabeler,
)
from piton.labelers.omop_lab_values import (
    ThrombocytopeniaLabValueLabeler,
    HyperkalemiaLabValueLabeler,
    HypoglycemiaLabValueLabeler,
)

"""
Example running:

    # /local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract_v5 \

python3 1_run_featurizers.py \
    /local-scratch/nigam/projects/mwornow/data/1_perct_extract_01_11_23 \
    /local-scratch/nigam/projects/clmbr_text_assets/data/features/lupus/labeled_patients.pkl \
    /local-scratch/nigam/projects/clmbr_text_assets/data/features/lupus/preprocessed_featurizers.pkl \
    /local-scratch/nigam/projects/clmbr_text_assets/data/features/lupus/featurized_patients.pkl \
    --labeling_function lupus \
    --num_threads 20
"""


def save_to_pkl(object_to_save, path_to_file: str):
    """Save object to pkl file."""
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "wb") as fd:
        pickle.dump(object_to_save, fd)


LABELING_FUNCTIONS: List[str] = [
    "mortality", 
    "is_male", 
    "lupus",
    "high_hba1c",
    "thrombocytopenia",
    "hyperkalemia"
    "hypoglycemia",
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
        "path_to_labeled_patients",
        type=str,
        help="Path to file containing the Piton LabeledPatients. Example: '/local-scratch/nigam/projects/rthapa84/data/mortality_labeled_patients_test.pkl'",
    )

    parser.add_argument(
        "path_to_save_preprocessed_featurizers",
        type=str,
        help="Path to file to save preprocessed Featurizers. Example: '/local-scratch/nigam/projects/rthapa84/data/mortality_preprocessed_featurizers_test.pkl'",
    )

    parser.add_argument(
        "path_to_save_featurized_patients",
        type=str,
        help="Path to file to save features for patients. Example: '/local-scratch/nigam/projects/rthapa84/data/mortality_featurized_patients_test.pkl'",
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
        "--num_patients",
        type=int,
        help="Number of patients to use (used for DEBUGGING)",
        default=None,
    )

    # Parse CLI args
    args = parser.parse_args()
    PATH_TO_PATIENT_DATABASE: str = args.path_to_patient_database
    PATH_TO_LABELED_PATIENTS: str = args.path_to_labeled_patients
    PATH_TO_SAVE_PREPROCESSED_FEATURIZERS: str = (
        args.path_to_save_preprocessed_featurizers
    )
    PATH_TO_SAVE_FEATURIZED_PATIENTS: str = (
        args.path_to_save_featurized_patients
    )
    num_threads: int = args.num_threads
    num_patients: Optional[int] = args.num_patients

    # create directories to save files
    os.makedirs(
        os.path.dirname(os.path.abspath(PATH_TO_SAVE_PREPROCESSED_FEATURIZERS)),
        exist_ok=True,
    )
    os.makedirs(
        os.path.dirname(os.path.abspath(PATH_TO_SAVE_FEATURIZED_PATIENTS)),
        exist_ok=True,
    )

    # Load PatientDatabase + Ontology
    database = piton.datasets.PatientDatabase(PATH_TO_PATIENT_DATABASE)
    ontology = database.get_ontology()
    print_log("PatientDatabase", "Loaded from: " + PATH_TO_PATIENT_DATABASE)

    # Define the labeling function.
    if args.labeling_function == "high_hba1c":
        labeler = HighHbA1cCodeLabeler(ontology)
    elif args.labeling_function == "mortality":
        time_horizon = TimeHorizon(
            datetime.timedelta(days=0), datetime.timedelta(days=365)
        )
        labeler = MortalityCodeLabeler(ontology, time_horizon)
    elif args.labeling_function == "lupus":
        time_horizon = TimeHorizon(
            datetime.timedelta(days=0), datetime.timedelta(days=365)
        )
        labeler = LupusCodeLabeler(ontology, time_horizon)
    elif args.labeling_function == "is_male":
        labeler = IsMaleLabeler(ontology)
    elif args.labeling_function == "thrombocytopenia":
        time_horizon = TimeHorizon(
            datetime.timedelta(days=0), datetime.timedelta(days=365)
        )
        labeler = ThrombocytopeniaLabValueLabeler(ontology,
                                                  time_horizon,
                                                  'severe')
    elif args.labeling_function == "hyperkalemia":
        time_horizon = TimeHorizon(
            datetime.timedelta(days=0), datetime.timedelta(days=365)
        )
        labeler = HyperkalemiaLabValueLabeler(ontology,
                                                  time_horizon,
                                                  'severe')
    elif args.labeling_function == "hypoglycemia":
        time_horizon = TimeHorizon(
            datetime.timedelta(days=0), datetime.timedelta(days=365)
        )
        labeler = HypoglycemiaLabValueLabeler(ontology,
                                                  time_horizon,
                                                  'severe')
    else:
        raise ValueError(
            f"Labeling function `{args.labeling_function}` not supported. Must be one of: {LABELING_FUNCTIONS}."
        )

    # grabbing just one label at random from all the labels
    one_label_labeler = NLabelsPerPatientLabeler(labeler, seed=0, num_labels=1)
    print_log("Labeler", "Instantiated Labeler: " + args.labeling_function)

    print_log("Labeling Patients", "Starting")
    labeled_patients = one_label_labeler.apply(
        path_to_patient_database=PATH_TO_PATIENT_DATABASE, num_threads=num_threads, num_patients=num_patients
    )
    save_to_pkl(labeled_patients, PATH_TO_LABELED_PATIENTS)
    print_log("Labeling Patients", "Finished")
    print("Length of labeled_patients", len(labeled_patients))

    # Lets use both age and count featurizer
    age = AgeFeaturizer()
    count = CountFeaturizer(is_ontology_expansion=True)
    featurizer_age_count = FeaturizerList([age, count])

    # Preprocessing the featurizers, which includes processes such as normalizing age.
    print_log("Preprocessing Featurizer", "Starting")
    featurizer_age_count.preprocess_featurizers(
        PATH_TO_PATIENT_DATABASE, labeled_patients, num_threads
    )
    save_to_pkl(featurizer_age_count, PATH_TO_SAVE_PREPROCESSED_FEATURIZERS)
    print_log("Preprocessing Featurizer", "Finished")

    print_log("Featurize Patients", "Starting")
    results = featurizer_age_count.featurize(
        PATH_TO_PATIENT_DATABASE, labeled_patients, num_threads
    )
    save_to_pkl(results, PATH_TO_SAVE_FEATURIZED_PATIENTS)
    print_log("Featurize Patients", "Finished")

    print_log("FINISH", "Done")
