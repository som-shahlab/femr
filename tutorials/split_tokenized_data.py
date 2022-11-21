import piton
import piton.datasets
import pickle
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from piton.featurizers import save_to_file, load_from_file
from piton.featurizers.featurizers import TextFeaturizer, _get_text_embeddings
from typing import Tuple, List
import datetime
import os, sys


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


path_to_data = "/share/pi/nigam/rthapa84/data"
TEXT_EMBEDDINGS_PATH = "v1_diabetes_tokenized_text.pickle"

path_to_save = "/share/pi/nigam/rthapa84/data/diabetes_tokenized_data"

print("Loading File")
tokenized_text_list = load_from_file(os.path.join(path_to_data, TEXT_EMBEDDINGS_PATH))
print("File Loaded")
for i, tokenized_text in tqdm(enumerate(tokenized_text_list)):
    save_to_file(tokenized_text, os.path.join(path_to_save, f"{i}_tokenized_data.pickle"))

print("Done")
