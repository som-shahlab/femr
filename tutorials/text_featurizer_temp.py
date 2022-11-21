# import piton
# import piton.datasets
import pickle
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
# from piton.featurizers import save_to_file, load_from_file
# from piton.featurizers.featurizers import TextFeaturizer, _get_text_embeddings
from typing import Tuple, List
import datetime
import os, sys
import logging


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


def _get_text_embeddings(args):

    device = torch.device("cuda")

    file_name, path_to_model, params_dict = args
    tokenized_text_data = load_from_file(file_name)

    model = AutoModel.from_pretrained(path_to_model)
    model = model.to(device)

    batch_size: int = params_dict["batch_size"]
    train_loader = [
        {
            "input_ids": tokenized_text_data["input_ids"][x : x + batch_size],
            # "token_type_ids": tokenized_text_data["token_type_ids"][x : x + batch_size],
            "attention_mask": tokenized_text_data["attention_mask"][x : x + batch_size],
        }
        for x in range(0, tokenized_text_data["input_ids"].shape[0], batch_size)
    ]

    outputs = []
    with torch.no_grad():
        for batch in tqdm(train_loader):
            output = model(
                input_ids=batch["input_ids"].to(device),
                # token_type_ids=batch["token_type_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            outputs.append(output.last_hidden_state.detach().cpu())

    token_embeddings = torch.cat(outputs)
    embedding_tensor = token_embeddings[:, 0, :].squeeze()
    embedding_numpy = embedding_tensor.cpu().detach().numpy()

    return embedding_numpy


# Please update this path with your extract of piton as noted in previous notebook. 
# PATH_TO_PITON_DB= '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2'
# PATH_TO_SAVE_MATRIX = "/local-scratch/nigam/projects/rthapa84/data"
path_to_data = "/share/pi/nigam/rthapa84/data/mortality_tokenized_data"
# TEXT_EMBEDDINGS_PATH = "v1_diabetes_tokenized_text.pickle"

path_to_model = "/share/pi/nigam/rthapa84/models/Clinical-Longformer"
path_to_save = "/share/pi/nigam/rthapa84/data/mortality_shards"

# path_to_model = "/local-scratch/nigam/projects/clmbr_text_assets/models/Bio_ClinicalBERT"
# path_to_save = "/local-scratch/nigam/projects/rthapa84/data/"

num_threads: int = 20
num_threads_gpu: int = 1
min_gpu_size: int = 20
min_char: int = 100
max_char: int = 10000
max_length: int = 1024 
padding: bool = True 
truncation: bool = True
chunk_size: int = 1000
batch_size: int = 16
num_patients: int = 100

params_dict = {
    "min_char": min_char, 
    "max_char": max_char,
    "max_length": max_length, 
    "padding": padding, 
    "truncation": truncation, 
    "chunk_size": chunk_size, 
    "min_gpu_size": min_gpu_size, 
    "batch_size": batch_size
}


if __name__ == '__main__':
    shard_index = int(sys.argv[1])
    num_shards = int(sys.argv[2])

    file_names = [os.path.join(path_to_data, f"{i}_tokenized_data.pickle") for i in range(20)]

    # tokenized_text_list = load_from_file(os.path.join(path_to_data, TEXT_EMBEDDINGS_PATH))

    num_per_shard = (len(file_names) + num_shards - 1) // num_shards

    logging.info("Before the Loop")

    embeddings_list = []
    for file_name in file_names[shard_index * num_per_shard: (shard_index + 1) * num_per_shard]:
        logging.info(f"File Name: {file_name}")
        embeddings_list.append(_get_text_embeddings((file_name, path_to_model, params_dict)))
        logging.info(f"Finished: {file_name}")
    
    embeddings = np.concatenate(embeddings_list, axis=0)
    logging.info("Finished Concatenating")
    save_to_file(embeddings, os.path.join(path_to_save, f"{shard_index}_embeddings.pickle"))








    

    # tokenized_text_list = load_from_file(os.path.join(path_to_save, "temp_tokenized_text.pickle"))
    # print(len(tokenized_text_list))

    # num_threads: int = 1
    # num_threads_gpu: int = 1
    # min_gpu_size: int = 20
    # min_char: int = 100
    # max_char: int = 10000
    # max_length: int = 512 
    # padding: bool = True 
    # truncation: bool = True
    # chunk_size: int = 1000
    # batch_size: int = 4096
    # num_patients: int = None

    # params_dict = {
    #     "min_char": min_char, 
    #     "max_char": max_char,
    #     "max_length": max_length, 
    #     "padding": padding, 
    #     "truncation": truncation, 
    #     "chunk_size": chunk_size, 
    #     "min_gpu_size": min_gpu_size, 
    #     "batch_size": batch_size
    # }

    # start_time = datetime.datetime.now()
    # print("Starting Generating Embedding")

    # model = AutoModel.from_pretrained(path_to_model)

    # embeddings = []
    # for tokenized_data in tokenized_text_list:
    #     outputs = model(**tokenized_data)
    #     batch_embedding_tensor = outputs.last_hidden_state[:, 0, :].squeeze()
    #     batch_embedding_numpy = batch_embedding_tensor.cpu().detach().numpy()
    #     embeddings.append(batch_embedding_numpy)

    # embeddings = np.concatenate(embeddings)

    # print("Finished Generating Embedding: ", datetime.datetime.now() - start_time)
    # save_to_file(embeddings, os.path.join(path_to_save, f"temp_embeddings.pickle"))
    # print("Embedding Saved: ", datetime.datetime.now() - start_time)


    # # embeddings_list = []
    # # for tokenized_text in tokenized_text_list:
    # #     embeddings_list.append(_get_text_embeddings((tokenized_text, path_to_model, params_dict)))
    
    # # print("Finished Generating Embedding: ", datetime.datetime.now() - start_time)
    # # embeddings = np.concatenate(embeddings_list)
    # # save_to_file(embeddings, os.path.join(path_to_save, f"temp_embeddings.pickle"))

