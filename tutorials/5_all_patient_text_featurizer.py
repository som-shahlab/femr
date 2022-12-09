
import os
import piton
import itertools
import numpy as np
import multiprocessing
from tqdm import tqdm
import datetime
import piton.datasets
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from piton.featurizers import save_to_file, load_from_file

def _get_all_patient_text_data(args):
    
    database_path, path_to_save, pids, params_dict, thread_idx = args
    database = piton.datasets.PatientDatabase(database_path)

    note_piton_codes = [database.get_code_dictionary().index(concept_id) for concept_id in params_dict["note_concept_ids"]]
    chunk_size: int = params_dict["chunk_size"]
    pids_loader = [pids[x : x + chunk_size] for x in range(0, len(pids), chunk_size)]

    for batch_idx, pids_batch in tqdm(enumerate(pids_loader), total=len(pids_loader)):

        data = {}
        for patient_id in pids_batch:
            one_patient_dict = {"text_data": [], "event_ids": []}
            patient = database[patient_id]

            for event_id, event in enumerate(patient.events):
                if len(note_piton_codes) > 0 and event.code not in note_piton_codes:
                    continue

                if type(event.value) is not memoryview:
                    continue

                text_data = bytes(event.value).decode("utf-8")

                # if database.get_text_count(text_data) > 1:
                #     continue

                if len(text_data) < params_dict["min_char"]:
                    continue

                one_patient_dict["text_data"].append(text_data)
                one_patient_dict["event_ids"].append(event_id)
            
            if len(one_patient_dict["text_data"]) == 0:
                continue
            data[patient_id] = one_patient_dict
        
        save_to_file(data, os.path.join(path_to_save, f"{thread_idx}_{batch_idx}_text_data.pickle"))

    return thread_idx


def _get_tokenized_text(args):

    text_files, path_to_model, path_to_save, params_dict = args
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)

    for text_file in tqdm(text_files):

        file_suffix = text_file.split("/")[-1][:-17]
        text_data_dict = load_from_file(text_file)

        tokenized_text_data_dict = {}
        for patient_id, text_data in text_data_dict.items():
            tokenized_text_data_dict[patient_id] = {}
            patient_text_data = text_data["text_data"]

            if len(patient_text_data) == 0:
                continue

            notes_tokenized = tokenizer(
                                    patient_text_data,
                                    padding=params_dict["padding"],
                                    truncation=params_dict["truncation"],
                                    max_length=params_dict["max_length"],
                                    return_tensors="pt",
                                )

            input_ids = notes_tokenized['input_ids'].numpy().astype(np.uint16)
            attention_mask = (input_ids != 1)*1
            seq_lens = attention_mask.sum(axis=1)
            event_ids = text_data["event_ids"]

            # print(seq_lens)
            reverse_idx = np.argsort(seq_lens)[::-1]

            seq_lens = np.array(seq_lens[reverse_idx])
            input_ids = np.array(input_ids[reverse_idx])
            event_ids = np.array(event_ids)[reverse_idx]

            # seq_lens, input_ids, event_ids = zip(*sorted(zip(seq_lens, input_ids, event_ids), reverse=True))

            tokenized_text_data_dict[patient_id]["tokenized_data"] = input_ids
            tokenized_text_data_dict[patient_id]["event_ids"] = event_ids
            tokenized_text_data_dict[patient_id]["seq_lens"] = seq_lens

        save_to_file(tokenized_text_data_dict, os.path.join(path_to_save, f"{file_suffix}_tokenized_data.pickle"))
    
    return None


def _get_text_embeddings(args):

    text_files, path_to_model, path_to_save, params_dict = args
    model = AutoModel.from_pretrained(path_to_model)

    for text_file in text_files:

        file_suffix = text_file.split("/")[-1][:-22]
        tokenized_data_dict = load_from_file(text_file)

        embeded_text_data_dict = {}
        for patient_id, tokenized_data in tokenized_data_dict.items():
            embeded_text_data_dict[patient_id] = {}
            patient_tokenized_data = tokenized_data["tokenized_data"]

            output = model(
                input_ids=patient_tokenized_data["input_ids"],
                attention_mask=patient_tokenized_data["attention_mask"],
            )
            output = output.last_hidden_state.detach().cpu()
            embedding_tensor = output[:, 0, :].squeeze()
            embedding_numpy = embedding_tensor.cpu().detach().numpy()

            embeded_text_data_dict[patient_id]["embeded_data"] = embedding_numpy
            embeded_text_data_dict[patient_id]["event_ids"] = tokenized_data["event_ids"]
    
        save_to_file(embeded_text_data_dict, os.path.join(path_to_save, f"{file_suffix}_embeded_data.pickle"))


class TextFeaturizer:
    def __init__(
        self,
        database_path: str,
        num_patients: None,
        random_seed: int = 1
):
        self.database_path = database_path
        self.random_seed = random_seed
        self.num_patients = num_patients
    
    def preprocess_text(self):
        pass

    def accumulate_text(
        self, 
        path_to_save: str, 
        num_threads: int = 1,
        min_char: int = 100, 
        note_concept_ids: list= [],
        chunk_size: int = 10000
    ):
        if self.num_patients is not None:
            pids = [i for i in range(self.num_patients)]
        else:
            database = piton.datasets.PatientDatabase(self.database_path)
            num_patients = len(database)
            pids = [i for i in range(num_patients)]

        print(len(pids))
        params_dict = {
            "min_char": min_char, 
            "note_concept_ids": note_concept_ids, 
            "chunk_size": chunk_size
        }

        # Text Acculumation
        start_time = datetime.datetime.now()
        print("Starting text accumulation")
        
        pids_parts = np.array_split(pids, num_threads)
        tasks = [(self.database_path, path_to_save, pid_part, params_dict, thread_idx) for thread_idx, pid_part in enumerate(pids_parts)]
        ctx = multiprocessing.get_context('forkserver')
        with ctx.Pool(num_threads) as pool:
            batch_idx_list = list(pool.imap(_get_all_patient_text_data, tasks))

        print("Finished text accumulation: ", datetime.datetime.now() - start_time)

    def tokenize_text(
        self, 
        path_to_model: str,
        path_to_save: str,
        path_to_text_data: str, 
        num_threads: int = 5,
        max_length: int = 512, 
        padding: bool = True, 
        truncation: bool = True, 
    ):

        text_files = [os.path.join(path_to_text_data, text_file_name) for text_file_name in os.listdir(path_to_text_data)]
        params_dict = {
            "padding": padding, 
            "truncation": truncation, 
            "max_length": max_length
        }

        start_time = datetime.datetime.now()
        print("Starting Tokenization")
        text_files_parts = np.array_split(text_files, num_threads)
        tasks = [(text_files_part, path_to_model, path_to_save, params_dict) for text_files_part in text_files_parts]
        ctx = multiprocessing.get_context('forkserver')
        with ctx.Pool(num_threads) as pool:
            tokenized_text_list = list(pool.imap(_get_tokenized_text, tasks))

        print("Finished text tokenization: ", datetime.datetime.now() - start_time)
    
    def generate_embeddings(self):
        # Currently this needs to be done separately as a separate script to be run in GPU in Carina. 
        # Check ...
        pass


# Procedure note, Progress Note, Discharge Summary, and Nurse Notes respectively
note_concept_ids = ["LOINC/28570-0", "LOINC/11506-3", "LOINC/18842-5", "LOINC/34746-8"]
# note_concept_ids = ["LOINC/11506-3", "LOINC/18842-5", "LOINC/34746-8"]
# note_concept_ids = []
# path_to_model = "/local-scratch/nigam/projects/clmbr_text_assets/models/Clinical-Longformer"
path_to_model = "/local-scratch/nigam/projects/clmbr_text_assets/models/Bio_ClinicalBERT"
path_to_piton_db = "/local-scratch/nigam/projects/rthapa84/data/1_perct_piton_extract_12_07_22"
# path_to_piton_db = '/local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2'
path_to_save = "/local-scratch/nigam/projects/rthapa84/data/"
# path_to_save = "/local-scratch-nvme/nigam/projects/rthapa84/data"
path_to_save_text_data = os.path.join(path_to_save, "text_data_chunks")
path_to_save_tokenized_data = os.path.join(path_to_save, "tokenized_data_chunks")
num_threads = 20
min_char = 100
num_patients = None # None if you want to run on entire patients
chunk_size = 20000
max_length = 1024
padding = True
truncation = True

if __name__ == '__main__':

    text_featurizer = TextFeaturizer(path_to_piton_db, num_patients=num_patients)

    # if not os.path.exists(path_to_save_text_data):
    #     os.makedirs(path_to_save_text_data)

    # text_featurizer.accumulate_text(
    #     path_to_save_text_data,
    #     num_threads=num_threads,
    #     min_char=min_char,
    #     note_concept_ids=note_concept_ids,
    #     chunk_size=chunk_size, 
    # )

    if not os.path.exists(path_to_save_tokenized_data):
        os.makedirs(path_to_save_tokenized_data)

    text_featurizer.tokenize_text(
        path_to_model,
        path_to_save_tokenized_data,
        path_to_save_text_data,
        num_threads=num_threads,
        max_length=max_length, 
        padding=padding, 
        truncation=truncation, 
    )

    # params_dict = {
    #         "min_char": min_char, 
    #         "note_concept_ids": note_concept_ids, 
    #         "chunk_size": chunk_size
    #     }
    
    # pids = [i for i in range(num_patients)]
    # thread_idx = 0

    # args = path_to_piton_db, path_to_save_text_data, pids, params_dict, thread_idx
    # _get_all_patient_text_data(args)



