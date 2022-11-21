import pickle
from tqdm import tqdm
from piton.featurizers import save_to_file, load_from_file
import os, sys


path_to_data = "/local-scratch/nigam/projects/rthapa84/data"
TEXT_EMBEDDINGS_PATH = "v1_mortality_tokenized_text.pickle"

path_to_save = "/local-scratch/nigam/projects/rthapa84/data/mortality_tokenized_data"

print("Loading File")
tokenized_text_list = load_from_file(os.path.join(path_to_data, TEXT_EMBEDDINGS_PATH))
print("File Loaded")
for i, tokenized_text in tqdm(enumerate(tokenized_text_list)):
    save_to_file(tokenized_text, os.path.join(path_to_save, f"{i}_tokenized_data.pickle"))

print("Done")
