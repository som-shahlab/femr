import datetime
import os
from typing import List, Tuple

import numpy as np
from sklearn import metrics

import piton
import piton.datasets
from piton.labelers.core import Label, LabeledPatients, TimeHorizon
from piton.labelers.omop_labeling_functions import CodeLF, MortalityLF, IsMaleLF, DiabetesLF
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
LABELED_PATIENTS = "diabetes_labeled_patients_v4.pickle"
# PREPROCESSED_FEATURIZERS_DATA = "0_70_diabetes_preprocessed_featurizers_v1.pickle"
# FEATURIZED_DATA = "0_70_diabetes_featurized_patients_v1.pickle"

NUM_PATIENTS = None
NUM_THREADS = 20
SEED = 97

def slice_labeled_patients(labeled_patients, PATH_TO_PITON_DB, seed=SEED):
    database = piton.datasets.PatientDatabase(PATH_TO_PITON_DB)

    pids = np.array(labeled_patients.get_all_patient_ids())
    hashed_pids = np.array([database.compute_split(seed, pid) for pid in pids])

    patients_to_labels = labeled_patients.get_patients_to_labels() 
    labeler_type = labeled_patients.get_labeler_type()

    # train_bool = (hashed_pids <= 20)
    # dev_bool = ((hashed_pids >= 70) & (hashed_pids < 85))

    train_pids_idx = np.where((hashed_pids < 20))
    dev_pids_idx = np.where(((hashed_pids >= 70) & (hashed_pids < 85)))

    all_pids_idx = np.concatenate((train_pids_idx, dev_pids_idx), axis=None)
    print(len(all_pids_idx))

    new_patients_to_labels = {}

    new_patients_to_labels = {pids[pid_idx]: patients_to_labels[pids[pid_idx]] for pid_idx in all_pids_idx}

    return LabeledPatients(new_patients_to_labels, labeler_type)

    # print((hashed_pids <= 20).sum(), ((hashed_pids >= 70) & (hashed_pids < 85)).sum())

    # print(type(pids))

if __name__ == '__main__':

    database = piton.datasets.PatientDatabase(PATH_TO_PITON_DB)
    labeled_patients = load_from_file(os.path.join(PATH_TO_SAVE_MATRIX, LABELED_PATIENTS))

    patients_to_labels = labeled_patients.get_patients_to_labels()

    print(len(labeled_patients))
    print(labeled_patients.get_labeler_type())
    print(len(labeled_patients.get_patients_to_labels()))
    print(list(labeled_patients.get_patients_to_labels().keys())[:10])
    print(labeled_patients.get_patients_to_labels()[2594151])

    new_labeled_patients = slice_labeled_patients(labeled_patients, PATH_TO_PITON_DB)

    print(len(new_labeled_patients))
