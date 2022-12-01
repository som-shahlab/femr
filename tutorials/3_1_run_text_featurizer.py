import piton
import piton.datasets
import pickle
import numpy as np

from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from piton.featurizers.featurizers import TextFeaturizer
from piton.featurizers import save_to_file, load_from_file
from typing import Tuple, List
import multiprocessing
import datetime
import os


# Please update this path with your extract of piton as noted in previous notebook. 
path_to_model = "/local-scratch/nigam/projects/clmbr_text_assets/models/Clinical-Longformer"
path_to_labeled_patients = "/local-scratch/nigam/projects/rthapa84/data/mortality_labeled_patients_test.pickle"
database_path = "/local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2"
path_to_save = "/local-scratch/nigam/projects/rthapa84/data/"
num_threads = 20
min_char = 100
max_char = 10000
max_length = 4096
padding = True
truncation = True
num_patients = 1000 # None if you want to run on entire patients
batch_size = 1024
prefix = "test_mortality"


if __name__ == '__main__':
    # start_time = datetime.datetime.now()

    labeled_patients = load_from_file(path_to_labeled_patients)
    print("Labeled Patients Loaded")

    text_featurizer = TextFeaturizer(labeled_patients, database_path, num_patients=num_patients)

    text_featurizer.accumulate_text(
        path_to_save,
        num_threads=num_threads,
        min_char=min_char,
        prefix=prefix,
    )

    path_to_text_data = os.path.join(path_to_save, f"{prefix}_text_data.pickle")

    path_to_save_tokenized_data = os.path.join(path_to_save, f"{prefix}_tokenized_data")
    print("Starting tokenization")
    text_featurizer.tokenize_text(
        path_to_model,
        path_to_save_tokenized_data,
        path_to_text_data,
        prefix=prefix,
        num_threads=num_threads,
        max_length=max_length, 
        padding=padding, 
        truncation=truncation, 
        batch_size=batch_size,
    )