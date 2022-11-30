"""
Note that this file needs to be run in Carina at the current state. 
That is because, in order to generate embeddings, the process needs 
to be run on GPU. This file is written specifically to be run on
Carina GPU. This file goes along with 3_2_generate_embeddings.sh bash file. 
The way to run this on carina is 

sbatch 3_2_generate_embeddings.sh

Make sure you have trasported over the tokenized data and other necessary information 
that this process requires from nero to specified location in carina. You can use
sftp command for that. For example, one of the data that you need to transport over 
is test_mortality_tokenized_data.
"""


import pickle
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from piton.featurizers import save_to_file, load_from_file
from typing import Tuple, List
import datetime
import os, sys
import logging


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
path_to_data = "/share/pi/nigam/rthapa84/data/test_mortality_tokenized_data"
path_to_model = "/share/pi/nigam/rthapa84/models/Clinical-Longformer"
path_to_save = "/share/pi/nigam/rthapa84/data/test_mortality_discharge_shards"

if not os.path.exists(path_to_save):
    os.mkdir(path_to_save)

batch_size: int = 16
num_patients: int = 1000

params_dict = {
    "chunk_size": chunk_size, 
    "batch_size": batch_size
}


if __name__ == '__main__':
    shard_index = int(sys.argv[1])
    num_shards = int(sys.argv[2])

    num_of_files = len(os.listdir(path_to_data))
    file_names = [os.path.join(path_to_data, f"{i}_tokenized_data.pickle") for i in range(num_of_files)]
    num_per_shard = (len(file_names) + num_shards - 1) // num_shards

    embeddings_list = []
    for file_name in file_names[shard_index * num_per_shard: (shard_index + 1) * num_per_shard]:
        logging.info(f"File Name: {file_name}")
        embeddings_list.append(_get_text_embeddings((file_name, path_to_model, params_dict)))
        logging.info(f"Finished: {file_name}")
    
    embeddings = np.concatenate(embeddings_list, axis=0)
    logging.info("Finished Concatenating")
    save_to_file(embeddings, os.path.join(path_to_save, f"{shard_index}_embeddings.pickle"))