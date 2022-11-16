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

def _get_all_children(ontology: extension_datasets.Ontology, code:int) -> Set[int]:

    children_code_set = set([code])
    parent_deque = deque([code])

    while len(parent_deque) == 0:
        temp_parent_code = parent_deque.popleft()
        for temp_child_code in ontology.get_children(temp_parent_code):
            children_code_set.add(temp_child_code)
            parent_deque.append(temp_child_code)

    return children_code_set


class SurvivalLabeler(LabelingFunction):
    def __init__(self, ontology, code, required_days):
        self.required_days = required_days
        self.codes = _get_all_children(ontology, code)
        
        self.allowed_codes = set()
        
        for code in range(len(ontology.get_dictionary())):
            code_str = bytes(ontology.get_dictionary()[code]).decode('utf8')
            if code_str.startswith('SNOMED/'):
                self.allowed_codes.add(code)
                
        print("Allowed", len(self.allowed_codes))
    
    def label(self, patient):
        birth_date = datetime.datetime.combine(patient.events[0].start.date(), datetime.time.min)
        censor_time = patient.events[-1].start
        
        possible_times = []
        first_history = None
        first_code = None
        
        for event in patient.events:
            if event.value is not None or event.code not in self.allowed_codes:
                continue
                
            if first_history is None and (event.start - birth_date) > datetime.timedelta(days=10):
                first_history = event.start
            
            is_event = event.code in self.codes
            if first_code is None and is_event:
                first_code = event.start
                
            
            if first_history is not None and first_code is None and (event.start - first_history) > datetime.timedelta(days=self.required_days):
                possible_times.append(event.start)
        
        possible_times = [a for a in possible_times if a != first_code]
        if len(possible_times) == 0:
            return []
        
        selected_time = random.choice(possible_times)
        is_censored = first_code is None
        
        if is_censored:
            event_time = censor_time
        else:
            event_time = first_code
        
        survival_value = SurvivalValue(event_time=event_time, is_censored=is_censored)
        result = [Label(time=selected_time, value=survival_value, label_type=self.get_labeler_type())]
        return result
    
    def get_labeler_type(self):
        return "survival"

# Please update this path with your extract of piton as noted in previous notebook. 
PATH_TO_PITON_DB= '/local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2'

LABELED_PATIENTS = "labels"
# PREPROCESSED_FEATURIZERS_DATA = "HighHbA1c_preprocessed_featurizers_v2.pickle"
# FEATURIZED_DATA = "HighHbA1c_featurized_patients_v2.pickle"

if False:
    NUM_PATIENTS = 100_000
    PATH_TO_SAVE_MATRIX = "/local-scratch/nigam/projects/ethanid/piton_labeling/tutorials/survival_tmp"
else:
    NUM_PATIENTS = None
    PATH_TO_SAVE_MATRIX = "/local-scratch/nigam/projects/ethanid/piton_labeling/tutorials/survival_full"
NUM_THREADS = 20

labels = {
    "pancreatic_cancer": "SNOMED/372003004",
    "celiac_disease": "SNOMED/396331005",
    "lupus": "SNOMED/55464009",
    "heart_attack": "SNOMED/57054005",
    "stroke": "SNOMED/432504007",
    "NAFL": "SNOMED/197321007",
}

if __name__ == '__main__':

    #PATH_TO_PITON_DB = '/local-scratch/nigam/projects/clmbr_text_assets/data/piton_database_1_perct/'
    #PATH_TO_SAVE_MATRIX = "./"

    # Patient database
    data = piton.datasets.PatientDatabase(PATH_TO_PITON_DB)

    # Ontology 
    ontology = data.get_ontology()

    
    for name, code_name in labels.items():
        code = ontology.get_dictionary().index(code_name)
        print("FOUND!", code_name, code)

        labeler = SurvivalLabeler(ontology, code, 365)
        # print(labeler)

        # labeled_patients = load_from_file(os.path.join(PATH_TO_SAVE_MATRIX, LABELED_PATIENTS))

        labeled_patients = labeler.apply(PATH_TO_PITON_DB, NUM_THREADS, num_patients=NUM_PATIENTS)
        print("Got", len(labeled_patients), "for", name)
        save_to_file(labeled_patients, os.path.join(PATH_TO_SAVE_MATRIX, LABELED_PATIENTS, name + ".pickle"))

        print("Finished Labeling Patients: ", datetime.datetime.now() - start_time)

#     # Lets use both age and count featurizer 
#     age = AgeFeaturizer()
#     count = CountFeaturizer(rollup=True)
#     featurizer_age_count = FeaturizerList([age, count])

#     # Preprocessing the featurizers, which includes processes such as normalizing age. 

#     # featurizer_age_count = load_from_file(os.path.join(PATH_TO_SAVE_MATRIX, PREPROCESSED_FEATURIZERS_DATA))
    
#     featurizer_age_count.preprocess_featurizers(labeled_patients, PATH_TO_PITON_DB, NUM_THREADS)
#     save_to_file(featurizer_age_count, os.path.join(PATH_TO_SAVE_MATRIX, PREPROCESSED_FEATURIZERS_DATA))

#     print("Finished Preprocessing Featurizers: ", datetime.datetime.now() - start_time)

#     results = featurizer_age_count.featurize(labeled_patients, PATH_TO_PITON_DB, NUM_THREADS)
#     print("Finished Training Featurizers: ", datetime.datetime.now() - start_time)
#     # print(results[0].toarray(), results[1])
#     save_to_file(results, os.path.join(PATH_TO_SAVE_MATRIX, FEATURIZED_DATA))

#     end_time = datetime.datetime.now()
#     delta = (end_time - start_time)
#     print(delta)







