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


# def save_to_file(object_to_save, path_to_file: str):
#     """Save object to Pickle file."""
#     os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
#     with open(path_to_file, "wb") as fd:
#         pickle.dump(object_to_save, fd)

# def load_from_file(path_to_file: str):
#     """Load object from Pickle file."""
#     with open(path_to_file, "rb") as fd:
#         result = pickle.load(fd)
#     return result


# Please update this path with your extract of piton as noted in previous notebook. 
# PATH_TO_PITON_DB= '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2'
# PATH_TO_SAVE_MATRIX = "/local-scratch/nigam/projects/rthapa84/data"
TEXT_EMBEDDINGS_PATH = "test_diabetes_text_embeddings.pickle"

path_to_model = "/share/pi/nigam/rthapa84/models/Clinical-Longformer"
# path_to_model = "/local-scratch/nigam/projects/clmbr_text_assets/models/Clinical-Longformer"
# path_to_model = "/local-scratch/nigam/projects/clmbr_text_assets/models/Bio_ClinicalBERT"
# HighHbA1c_labeled_patients_v3
# mortality_labeled_patients_v1
# path_to_labeled_patients = "/local-scratch/nigam/projects/rthapa84/data/HighHbA1c_labeled_patients_v3.pickle"
path_to_labeled_patients = "/share/pi/nigam/rthapa84/data/mortality_labeled_patients_v1.pickle"
# database_path = "/local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2"
database_path = "/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2"
path_to_save = "/share/pi/nigam/rthapa84/data"
num_threads = 20
num_threads_gpu = 2
min_char = 100
max_char = 10000
max_length = 1024
padding = True
truncation = True
chunk_size = 10
num_patients = 1000
batch_size = 1024
prefix = "v1_mortality"


if __name__ == '__main__':
    # start_time = datetime.datetime.now()

    labeled_patients = load_from_file(path_to_labeled_patients)
    print("Labeled Patients Loaded")

    # print(len(labeled_patients))
    # exit()

    text_featurizer = TextFeaturizer(labeled_patients, database_path)

    print("Starting text featurization")
    result_tuple = text_featurizer.featurize(path_to_model, 
                                             path_to_save, 
                                             num_threads=num_threads, 
                                             max_char=max_char,
                                             num_threads_gpu=num_threads_gpu, 
                                             num_patients=num_patients, 
                                             max_length=max_length, 
                                             prefix=prefix, 
                                             batch_size=batch_size)
    print("Text Featurization Finished")

    # print(result_tuple[0].shape)
    # path_to_save = os.path.join(PATH_TO_SAVE_MATRIX, TEXT_EMBEDDINGS_PATH)
    # save_to_file(result_tuple, path_to_save)
    # print(f"Embeddings Saved at {path_to_save}")

    # end_time = datetime.datetime.now()
    # delta = (end_time - start_time)
    # print("Total Duration of the run: ", delta)





# def _get_patient_text_data(patient, labels):
#     text_for_each_label = []

#     label_idx = 0
#     current_text = []
#     for event in patient.events:
#         while event.start > labels[label_idx].time:
#             label_idx += 1
#             text_for_each_label.append(" ".join(current_text))

#             if label_idx >= len(labels):
#                 return text_for_each_label

#         if type(event.value) is not memoryview:
#             continue

#         text_data = bytes(event.value).decode("utf-8")

#         if len(text_data) < MAX_CHAR:
#             continue

#         current_text.append(text_data)

#     if label_idx < len(labels):
#         for label in labels[label_idx:]:
#             text_for_each_label.append(" ".join(current_text))

#     return text_for_each_label
    


# def _run_text_featurizer(args):
    
#     database_path, pids, labeled_patients, path_to_model = args
    
#     # database_path, pids, labeled_patients, path_to_model = args
#     database = piton.datasets.PatientDatabase(database_path)
#     tokenizer = AutoTokenizer.from_pretrained(path_to_model)
#     model = AutoModel.from_pretrained(path_to_model)
    
#     data = []
#     patient_ids = []
#     result_labels = []
#     labeling_time = []
    
#     for patient_id in pids:
#         patient = database[patient_id]
#         labels = labeled_patients.pat_idx_to_label(patient_id)

#         if len(labels) == 0:
#             continue
        
#         patient_text_data = _get_patient_text_data(patient, labels)
        
#         for i, label in enumerate(labels):
#             data.append(patient_text_data[i])
#             result_labels.append(label.value)
#             patient_ids.append(patient.patient_id)
#             labeling_time.append(label.time)
    
#     embeddings = []
#     for chunk in range(0, len(data), CHUNK_SIZE):
#         notes_tokenized = tokenizer(
#                                 data[chunk:chunk+CHUNK_SIZE],
#                                 padding=padding,
#                                 truncation=truncation,
#                                 max_length=max_length,
#                                 return_tensors="pt",
#                             )
#         outputs = model(**notes_tokenized)
#         batch_embedding_tensor = outputs.last_hidden_state[:, 0, :].squeeze()
#         batch_embedding_numpy = batch_embedding_tensor.cpu().detach().numpy()
#         embeddings.append(batch_embedding_numpy)
    
#     embeddings = np.concatenate(embeddings)
            
#     return embeddings, result_labels, patient_ids, labeling_time



# if __name__ == '__main__':
#     start_time = datetime.datetime.now()
#     pids = sorted(labeled_patients.get_all_patient_ids())[:200]
    

#     pids_parts = np.array_split(pids, num_threads)

#     tasks = [(database_path, pid_part, labeled_patients, path_to_model) for pid_part in pids_parts]

#     ctx = multiprocessing.get_context('forkserver')
#     with ctx.Pool(num_threads) as pool:
#         text_featurizers_tuple_list = list(pool.imap(_run_text_featurizer, tasks))

#     embeddings_list = []
#     result_labels_list = []
#     patient_ids_list = []
#     labeling_time_list = []
#     for result in text_featurizers_tuple_list:
#         embeddings_list.append(result[0])
#         result_labels_list.append(result[1])
#         patient_ids_list.append(result[2])
#         labeling_time_list.append(result[3])
    
#     embeddings = np.concatenate(embeddings_list)
#     result_labels = np.concatenate(result_labels_list, axis=None)
#     patient_ids = np.concatenate(patient_ids_list, axis=None)
#     labeling_time = np.concatenate(labeling_time_list, axis=None)

#     result_tuple = (
#             embeddings,
#             result_labels,
#             patient_ids,
#             labeling_time,
#         )

#     print(embeddings.shape)
#     save_to_file(result_tuple, os.path.join(PATH_TO_SAVE_MATRIX, TEXT_EMBEDDINGS_PATH))

#     end_time = datetime.datetime.now()
#     delta = (end_time - start_time)
#     print(delta)






        


