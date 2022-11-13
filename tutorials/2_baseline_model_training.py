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


# Please update this path with your extract of piton as noted in previous notebook. 
# PATH_TO_PITON_DB= '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract'
# PATH_TO_SAVE_MATRIX = "/home/rthapa84/data"

PATH_TO_PITON_DB = '/local-scratch/nigam/projects/clmbr_text_assets/data/piton_database_1_perct/'
PATH_TO_SAVE_MATRIX = "./"

# Patient database
data = piton.datasets.PatientDatabase(PATH_TO_PITON_DB)

# Ontology 
ontology = data.get_ontology()

patients = data

# Define time horizon for labeling purpose based on your use case. 
# Note: Some labeling function may not take any time_horizon

num_threads = 10

time_horizon = TimeHorizon(
        datetime.timedelta(days=0), datetime.timedelta(days=365)
    )

# Define the mortality labeling function. 
# labeler = MortalityLF(ontology, time_horizon)
labeler = DiabetesLF(ontology, time_horizon)

labeled_patients = labeler.apply(patients, PATH_TO_PITON_DB, num_threads)

# Lets use both age and count featurizer 
age = AgeFeaturizer()
count = CountFeaturizer(rollup=True)
featurizer_age_count = FeaturizerList([age, count])

# Preprocessing the featurizers, which includes processes such as normalizing age. 
featurizer_age_count.preprocess_featurizers(patients, labeled_patients, PATH_TO_PITON_DB, num_threads)

results = featurizer_age_count.featurize(patients, labeled_patients, PATH_TO_PITON_DB, num_threads)

save_to_file(results, os.path.join(PATH_TO_SAVE_MATRIX, "test_diabetes_matrix.pickle"))

end_time = datetime.datetime.now()
delta = (end_time - start_time)
print(delta)
save_to_file(delta, os.path.join(PATH_TO_SAVE_MATRIX, "total_time.pickle"))







