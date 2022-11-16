import datetime
import os
from typing import List, Tuple, Set

import numpy as np
from sklearn import metrics

import piton
import piton.datasets
from piton.labelers.core import Label, LabeledPatients, TimeHorizon, LabelingFunction, SurvivalValue
from piton.labelers.omop_labeling_functions import CodeLF, MortalityLF, IsMaleLF, DiabetesLF, HighHbA1cLF
from piton.featurizers.core import Featurizer, FeaturizerList
from piton.featurizers.featurizers import AgeFeaturizer, CountFeaturizer
from piton.extension import datasets as extension_datasets

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
# import xgboost as xgb
import pickle
import datetime
from collections import deque
import random

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
PATH_TO_PITON_DB= '/local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2'

PATH_TO_SAVE_MATRIX = "/local-scratch/nigam/projects/ethanid/piton_labeling/tutorials/survival_tmp"
LABELED_PATIENTS = "labels"
SUBSET_LABELED_PATIENTS = "subset_labels"
FEATURES = "features"

NUM_PATIENTS = 100000
NUM_THREADS = 20

labels = {
    "pancreatic_cancer": "SNOMED/372003004",
    "celiac_disease": "SNOMED/396331005",
    "lupus": "SNOMED/55464009",
    "heart_attack": "SNOMED/57054005",
    "stroke": "SNOMED/432504007",
    "NAFL": "SNOMED/197321007",
}

import random

if __name__ == '__main__':
    # Patient database
    data = piton.datasets.PatientDatabase(PATH_TO_PITON_DB)

    # Ontology 
    ontology = data.get_ontology()

    for name, code_name in labels.items():
        labeled_patients = load_from_file(os.path.join(PATH_TO_SAVE_MATRIX, SUBSET_LABELED_PATIENTS, name + ".pickle"))

        age = AgeFeaturizer()
        count = CountFeaturizer()
        featurizer_age_count = FeaturizerList([age, count])
        
        featurizer_age_count.preprocess_featurizers(labeled_patients, PATH_TO_PITON_DB)
        full_matrix, labels, pids, times = featurizer_age_count.featurize(labeled_patients, PATH_TO_PITON_DB)
        
        features = {'full_matrix': full_matrix, 'labels': labels, 'pids': pids, 'times': times}
        save_to_file(features, os.path.join(PATH_TO_SAVE_MATRIX, FEATURES, name + ".pickle"))
