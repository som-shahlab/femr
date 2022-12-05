import piton
import pickle
import torch
import datetime
from typing import List
from piton.featurizers.featurizers_notes import NoteFeaturizer
from piton.datasets import PatientDatabase
import os
from ..labelers.core import LabeledPatients, Label
from icecream import ic

ic.configureOutput(includeContext=True, prefix=lambda: str(datetime.datetime.now()))

# Please update this path with your extract of piton as noted in previous notebook. 
PATH_TO_MODELS_DIR = "/local-scratch/nigam/projects/clmbr_text_assets/models/"
PATH_TO_PATIENT_DATABASE_1_PCT_EXTRACT = "/local-scratch/nigam/projects/clmbr_text_assets/data/piton_database_1_perct/" # n = 33,771
PATH_TO_PATIENT_DATABASE_FULL_EXTRACT = "/local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2" # n = 2,730,411
PATH_TO_LABELED_PATIENTS_DIABETES = "/local-scratch/nigam/projects/rthapa84/data/HighHbA1c_labeled_patients_v3.pickle" # n = 418,465
PATH_TO_LABELED_PATIENTS_MORATLITY = "/local-scratch/nigam/projects/rthapa84/data/mortality_labeled_patients_v1.pickle" # n = XXXXXX
PATH_TO_LABELED_PATIENTS_TEST = "/local-scratch/nigam/projects/rthapa84/data/mortality_labeled_patients_test.pickle" # n = XXXXXX
PATH_TO_CACHE_DIR = "/local-scratch/nigam/projects/mwornow/data/"

MODELS = [
    'Clinical-Longformer',
    'Bio_ClinicalBERT'
]

def get_gpus_with_minimum_free_memory(min_mem: float = 5) -> List[int]:
    """Return a list of GPU devices with at least `min_mem` free memory is in GB."""
    devices = []
    num_gpus: int = torch.cuda.device_count()
    for i in range(num_gpus):
        free, __ = torch.cuda.mem_get_info(i)
        if free >= min_mem * 1e9:
            devices.append(i)
    return devices

if __name__ == '__main__':
    # Constants
    path_to_cache_dir = os.path.join(PATH_TO_CACHE_DIR, 'test_mortality')
    path_to_model: str = os.path.join(PATH_TO_MODELS_DIR, 'Bio_ClinicalBERT')
    n_cpu_jobs: int = 20
    gpu_devices: List = get_gpus_with_minimum_free_memory(10)
    
    # Load LabeledPatients
    with open(PATH_TO_LABELED_PATIENTS_TEST, "rb") as fd:
        labeled_patients: LabeledPatients = pickle.load(fd)
    
    # Get valid codes for notes to keep
    data = PatientDatabase(PATH_TO_PATIENT_DATABASE_FULL_EXTRACT)
    valid_note_codes = [ data.get_code_dictionary().index(x) for x in [ "LOINC/28570-0", "LOINC/11506-3", "LOINC/18842-5", "LOINC/LP173418-7" ] ]
    
    
    note_featurizer = NoteFeaturizer(path_to_patient_database=PATH_TO_PATIENT_DATABASE_FULL_EXTRACT,
                                    path_to_tokenizer=path_to_model,
                                    path_to_embedder=path_to_model,
                                    path_to_cache_dir=path_to_cache_dir,
                                    path_to_output_dir=path_to_cache_dir, 
                                    n_cpu_jobs=n_cpu_jobs,
                                    gpu_devices=gpu_devices,
                                    params_preprocessor = {
                                        "min_char_count: ": 100,
                                        "codes" : valid_note_codes,
                                    },
                                    params_tokenizer = {
                                        "max_length": 4096,
                                        "padding": True,
                                        "truncation": True,
                                    },
                                    params_embedder = {
                                        "embed_method": piton.featurizers.featurizers_notes.embed_with_cls,
                                        "batch_size": 256,
                                    },
                                    preprocess_transformations = [
                                        piton.featurizers.transforms_notes.keep_only_notes_matching_codes,
                                        piton.featurizers.transforms_notes.remove_notes_after_label,
                                        piton.featurizers.transforms_notes.remove_short_notes,
                                        piton.featurizers.transforms_notes.join_all_notes,
                                    ])

    print("Starting text featurization")
    result_tuple = note_featurizer.featurize(labeled_patients, is_debug = True)
    print("Finished text featurization")
    print(result_tuple[0].shape)