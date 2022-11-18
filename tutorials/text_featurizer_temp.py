import piton
import piton.datasets
import pickle
import numpy as np

from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from piton.featurizers.featurizers import TextFeaturizer
from typing import Tuple, List
import multiprocessing
import datetime
import os


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
# PATH_TO_PITON_DB= '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2'
PATH_TO_SAVE_MATRIX = "/local-scratch/nigam/projects/rthapa84/data"
TEXT_EMBEDDINGS_PATH = "test_diabetes_text_embeddings.pickle"

path_to_model = "/local-scratch/nigam/projects/clmbr_text_assets/models/Clinical-Longformer"
# path_to_model = "/local-scratch/nigam/projects/clmbr_text_assets/models/Bio_ClinicalBERT"
path_to_labeled_patients = "/local-scratch/nigam/projects/rthapa84/data/HighHbA1c_labeled_patients_v3.pickle"
database_path = "/local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2"
path_to_save = "/local-scratch/nigam/projects/rthapa84/data/"
num_threads = 20
num_threads_gpu = 2
min_char = 100
max_char = 10000
max_length = 1024
padding = True
truncation = True
chunk_size = 10
num_patients = None


if __name__ == '__main__':

    tokenized_text = load_from_file(os.path.join(path_to_save, "temp_tokenized_text.pickle"))
    print(len(tokenized_text))