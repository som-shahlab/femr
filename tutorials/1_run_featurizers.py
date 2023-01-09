import datetime
import os
import argparse
import pickle
from typing import Optional, List
import piton
import piton.datasets
from piton.labelers.core import TimeHorizon, OneLabelPerPatient
from piton.labelers.omop_labeling_functions import CodeLF, MortalityLF, IsMaleLF, DiabetesLF, HighHbA1cLF
from piton.featurizers.core import Featurizer, FeaturizerList
from piton.featurizers.featurizers import AgeFeaturizer, CountFeaturizer

def save_to_pkl(object_to_save, path_to_file: str):
    """Save object to Pickle file."""
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "wb") as fd:
        pickle.dump(object_to_save, fd)

def load_from_pkl(path_to_file: str):
    """Load object from Pickle file."""
    with open(path_to_file, "rb") as fd:
        result = pickle.load(fd)
    return result

NUM_PATIENTS = 1000 # None if wants to run on all patients
NUM_THREADS = 20
LABELING_FUNCTIONS: List[str] = ['mortality', 'diabetes', 'is_male', 'high_hba1c']

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="Run Piton featurization"
    )

    parser.add_argument(
        "path_to_piton_db",
        type=str,
        help="Path of the folder to the Piton PatientDatabase. Example: '/local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2'",
    )

    parser.add_argument(
        "path_to_labeled_patients",
        type=str,
        help="Path to the Piton LabeledPatient. Example: '/local-scratch/nigam/projects/rthapa84/data/mortality_labeled_patients_test.pickle'",
    )

    parser.add_argument(
        "path_to_preprocessed_featurizers",
        type=str,
        help="Path to the Piton XXXX. Example: '/local-scratch/nigam/projects/rthapa84/data/mortality_preprocessed_featurizers_test.pickle'",
    )
    
    parser.add_argument(
        "path_to_featurized_data",
        type=str,
        help="Path to the Piton XXXX. Example: '/local-scratch/nigam/projects/rthapa84/data/mortality_featurized_patients_test.pickle'",
    )
    
    parser.add_argument(
        "--labeling_function",
        type=str,
        help="Name of labeling function to create.",
        options=LABELING_FUNCTIONS,
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
    
    

    args = parser.parse_args()
    PATH_TO_PITON_DB: str = args.path_to_piton_db
    PATH_TO_SAVE_MATRIX: str = args.path_to_save_matrix
    PATH_TO_LABELED_PATIENTS: str = args.path_to_labeled_patients
    PATH_TO_PREPROCESSED_FEATURIZERS_DATA: str = args.path_to_preprocessed_featurizers
    PATH_TO_FEATURIZED_DATA: str = args.path_to_featurized_data
    NUM_THREADS: int = args.num_threads
    num_patients: Optional[int] = args.num_patients

    start_time = datetime.datetime.now()

    # Load PatientDatabase + Ontology
    data = piton.datasets.PatientDatabase(PATH_TO_PITON_DB)
    ontology = data.get_ontology()

    # Define the labeling function. 
    time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=365)
    )
    if args.labeling_function == 'high_hba1c':
        labeler = HighHbA1cLF(ontology)
    elif args.labeling_function == 'mortality':
        labeler = MortalityLF(ontology, time_horizon)
    elif args.labeling_function == 'mortality':
        labeler = DiabetesLF(ontology, time_horizon)
    elif args.labeling_function == 'mortality':
        labeler = IsMaleLF(ontology)
    else:
        raise ValueError(f"Labeling function `{args.labeling_function}` not supported. Must be one of: {LABELING_FUNCTIONS}.")
    
    # grabbing just one label at random from all the labels
    one_label_labeler = OneLabelPerPatient(labeler)
    print("Instantiated Labelers")

    labeled_patients = one_label_labeler.apply(PATH_TO_PITON_DB, NUM_THREADS, num_patients=num_patients)
    save_to_pkl(labeled_patients, PATH_TO_LABELED_PATIENTS)
    print("Finished Labeling Patients: ", datetime.datetime.now() - start_time)

    # Lets use both age and count featurizer 
    age = AgeFeaturizer()
    count = CountFeaturizer(rollup=True)
    featurizer_age_count = FeaturizerList([age, count])

    # Preprocessing the featurizers, which includes processes such as normalizing age. 
    featurizer_age_count.preprocess_featurizers(labeled_patients, PATH_TO_PITON_DB, NUM_THREADS)
    save_to_pkl(featurizer_age_count, PATH_TO_PREPROCESSED_FEATURIZERS_DATA)
    print("Finished Preprocessing Featurizers: ", datetime.datetime.now() - start_time)

    results = featurizer_age_count.featurize(labeled_patients, PATH_TO_PITON_DB, NUM_THREADS)
    save_to_pkl(results, PATH_TO_FEATURIZED_DATA)
    print("Finished Training Featurizers: ", datetime.datetime.now() - start_time)

    print("Total Time: ", datetime.datetime.now() - start_time)







