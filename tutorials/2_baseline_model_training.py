import datetime
import os
from typing import List, Tuple

import numpy as np
from sklearn import metrics

import piton
import piton.datasets
from piton.labelers.core import Label, LabeledPatients, TimeHorizon
from piton.labelers.omop_labeling_functions import CodeLF, MortalityLF, IsMaleLF, DiabetesLF, HighHbA1cLF
from piton.featurizers.core import Featurizer, FeaturizerList
from piton.featurizers.featurizers import AgeFeaturizer, CountFeaturizer
from piton.extension import datasets as extension_datasets

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost as xgb
import pickle
import datetime

start_time = datetime.datetime.now()

def save_to_file(object_to_save, path_to_file: str):
    """Save object to Pickle file."""
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "wb") as fd:
        pickle.dump(object_to_save, fd)

def load_from_file(path_to_file: str):
    """Load object from Pickle file."""
    with open(path_to_file, "rb") as fd:
        result = pickle.load(fd)
    return result


# Please update this path with your extract of piton as noted in previous notebook. 
PATH_TO_PITON_DB= '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2'
PATH_TO_SAVE_MATRIX = "/share/pi/nigam/rthapa84/data"
LABELED_PATIENTS = "HighHbA1c_labeled_patients_v2.pickle"
PREPROCESSED_FEATURIZERS_DATA = "HighHbA1c_preprocessed_featurizers_v2.pickle"
FEATURIZED_DATA = "HighHbA1c_featurized_patients_v2.pickle"

NUM_PATIENTS = None
NUM_THREADS = 20

if __name__ == '__main__':

    #PATH_TO_PITON_DB = '/local-scratch/nigam/projects/clmbr_text_assets/data/piton_database_1_perct/'
    #PATH_TO_SAVE_MATRIX = "./"

    # Patient database
    data = piton.datasets.PatientDatabase(PATH_TO_PITON_DB)

    # Ontology 
    ontology = data.get_ontology()

    # patients = data
    # print("Finished creating patient database")
    # patients = [data[idx] for idx in range(500)]
    # Define time horizon for labeling purpose based on your use case. 
    # Note: Some labeling function may not take any time_horizon

    time_horizon = TimeHorizon(
            datetime.timedelta(days=0), datetime.timedelta(days=365)
        )

    # Define the mortality labeling function. 
    labeler = HighHbA1cLF(ontology)
    # labeler = MortalityLF(ontology, time_horizon)
    # labeler = DiabetesLF(ontology, time_horizon)
    print("Instantiated Labelers")

    # labeled_patients = load_from_file(os.path.join(PATH_TO_SAVE_MATRIX, LABELED_PATIENTS))

    labeled_patients = labeler.apply(PATH_TO_PITON_DB, NUM_THREADS, num_patients=NUM_PATIENTS)
    save_to_file(labeled_patients, os.path.join(PATH_TO_SAVE_MATRIX, LABELED_PATIENTS))

    print("Finished Labeling Patients: ", datetime.datetime.now() - start_time)

    # Lets use both age and count featurizer 
    age = AgeFeaturizer()
    count = CountFeaturizer(rollup=True)
    featurizer_age_count = FeaturizerList([age, count])

    # Preprocessing the featurizers, which includes processes such as normalizing age. 

    # featurizer_age_count = load_from_file(os.path.join(PATH_TO_SAVE_MATRIX, PREPROCESSED_FEATURIZERS_DATA))
    
    featurizer_age_count.preprocess_featurizers(labeled_patients, PATH_TO_PITON_DB, NUM_THREADS)
    save_to_file(featurizer_age_count, os.path.join(PATH_TO_SAVE_MATRIX, PREPROCESSED_FEATURIZERS_DATA))

    print("Finished Preprocessing Featurizers: ", datetime.datetime.now() - start_time)

    results = featurizer_age_count.featurize(labeled_patients, PATH_TO_PITON_DB, NUM_THREADS)
    print("Finished Training Featurizers: ", datetime.datetime.now() - start_time)
    # print(results[0].toarray(), results[1])
    save_to_file(results, os.path.join(PATH_TO_SAVE_MATRIX, FEATURIZED_DATA))

    end_time = datetime.datetime.now()
    delta = (end_time - start_time)
    print(delta)







