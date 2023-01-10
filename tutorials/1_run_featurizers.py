import datetime
import os
from typing import List, Tuple

import numpy as np
from sklearn import metrics

import piton
import piton.datasets
from piton.labelers.core import Label, LabeledPatients, TimeHorizon, OneLabelPerPatient
from piton.labelers.omop_labeling_functions import CodeLF, MortalityLF, IsMaleLF, DiabetesLF, HighHbA1cLF
from piton.featurizers.core import Featurizer, FeaturizerList
from piton.featurizers.featurizers import AgeFeaturizer, CountFeaturizer
from piton.featurizers import save_to_file, load_from_file
from piton.extension import datasets as extension_datasets

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
# import xgboost as xgb
import pickle
import datetime


# Please update this path with your extract of piton as noted in previous notebook. 
# PATH_TO_PITON_DB = '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2'
# PATH_TO_SAVE_MATRIX = "/share/pi/nigam/rthapa84/data"
# LABELED_PATIENTS = "mortality_labeled_patients_v1.pickle"
# PREPROCESSED_FEATURIZERS_DATA = "mortality_preprocessed_featurizers_v1.pickle"
# FEATURIZED_DATA = "mortality_featurized_patients_v1.pickle"

PATH_TO_PITON_DB = '/local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2'
PATH_TO_SAVE_MATRIX = "/local-scratch/nigam/projects/rthapa84/data"
LABELED_PATIENTS = "gender_labeled_patients_v2.pickle"
PREPROCESSED_FEATURIZERS_DATA = "gender_preprocessed_featurizers_test.pickle"
FEATURIZED_DATA = "gender_featurized_patients_test.pickle"

NUM_PATIENTS = None # None if wants to run on all patients
NUM_THREADS = 20

if __name__ == '__main__':

    start_time = datetime.datetime.now()

    # Patient database
    data = piton.datasets.PatientDatabase(PATH_TO_PITON_DB)

    # Ontology 
    ontology = data.get_ontology()

    time_horizon = TimeHorizon(
            datetime.timedelta(days=0), datetime.timedelta(days=365)
        )

    # Define the gender labeling function. 
    # labeler = HighHbA1cLF(ontology)
    # labeler = MortalityLF(ontology, time_horizon)
    # labeler = DiabetesLF(ontology, time_horizon)
    labeler = IsMaleLF(ontology)
    
    # grabbing just one label at random from all the labels
    one_label_labeler = OneLabelPerPatient(labeler)
    print("Instantiated Labelers")

    # labeled_patients = load_from_file(os.path.join(PATH_TO_SAVE_MATRIX, LABELED_PATIENTS))

    labeled_patients = one_label_labeler.apply(PATH_TO_PITON_DB, NUM_THREADS, num_patients=NUM_PATIENTS)
    save_to_file(labeled_patients, os.path.join(PATH_TO_SAVE_MATRIX, LABELED_PATIENTS))

    print("Finished Labeling Patients: ", datetime.datetime.now() - start_time)

    """
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
    print("Total Time: ", delta)

    """







