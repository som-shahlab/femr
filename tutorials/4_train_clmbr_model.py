"""
    This script walks through the various steps to train and use CLMBR.

    In order to use this script, the assumption is that you already have a set of labels and an extract
"""

import os
import pickle


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train CLMBR models")
    parser.add_argument(
        "path_to_patient_database",
        type=str,
        help="Path of folder to the FEMR PatientDatabase. Example: '/local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract_v5'",
    )

    parser.add_argument(
        "path_to_labeled_patients",
        type=str,
        help="Path to the file containing labels for patients.",
    )

    parser.add_argument(
        "path_to_model",
        type=str,
        help="Path to the file containing pretrained CLMBR model.",
    )

    args = parser.parse_args()

    # EXTRACT_LOCATION = "/local-scratch/nigam/projects/ethanid/shared_tutorial_files/piton_new3_extract"
    #LABELS = "/local-scratch/nigam/projects/ethanid/shared_tutorial_files/lupus/labeled_patients.pkl"
    # EXTRACT_LOCATION = "/local-scratch/nigam/projects/zphuo/data/omop_extract_PHI/som-nero-phi-nigam-starr.shahlab_omop_cdm5_subset_2023_05_06_extract_no_observation_v2"
    # LABELS = "/local-scratch/nigam/projects/zphuo/data/omop_extract_PHI/Allmortality_10000label_0to12m/labeled_patients.pkl"
    #MODEL_PATH = '/local-scratch/nigam/projects/zphuo/models/clmbr_lr_1e-05_wd_0.0_id_0.0_td_0.0_rt_global_maxiter_1000000_hs_768_is_3072_nh_12_nl_12_aw_512_obs/clmbr_model_old'


    EXTRACT_LOCATION: str =  args.path_to_patient_database
    LABELS: str = args.path_to_labeled_patients
    MODEL_PATH: str = args.path_to_model

    TEMP_STORAGE = "trash/clmbr_data"

    os.makedirs(TEMP_STORAGE, exist_ok=True)


    """
    The first step of training CLMBR is creating a dictionary, that helps map codes to integers that can be used within a neural network.
    """

    DICTIONARY_PATH = os.path.join(TEMP_STORAGE, "dictionary")

    if not os.path.exists(DICTIONARY_PATH):
        assert 0 == os.system(f"clmbr_create_dictionary {DICTIONARY_PATH} --data_path {EXTRACT_LOCATION}")

    """
    The second step of training CLMBR is to prepare the batches that will actually get fed into the neural network.
    """

    CLMBR_BATCHES = os.path.join(TEMP_STORAGE, "clmbr_batches")

    if not os.path.exists(CLMBR_BATCHES):
        assert 0 == os.system(
            f"clmbr_create_batches {CLMBR_BATCHES} --data_path {EXTRACT_LOCATION} --dictionary {DICTIONARY_PATH} --task clmbr"
        )

    """
    Given the batches, it is now possible to train CLMBR. By default it will train for 100 epochs, with early stopping.
    """

    #MODEL_PATH = os.path.join(TEMP_STORAGE, "clmbr_model")
 

    if not os.path.exists(MODEL_PATH):
        assert 0 == os.system(
            f"clmbr_train_model {MODEL_PATH} --data_path {EXTRACT_LOCATION} --batches_path {CLMBR_BATCHES} --learning_rate 1e-4 --rotary_type per_head --num_batch_threads 3 --max_iter 1"
        )

    """
    You now have a complete CLMBR model. It is now time to generate representations for your task of interest.

    The first step of doing so is to generate batches for that task.
    """

    TASK_BATCHES = os.path.join(TEMP_STORAGE, "task_batches")

    if not os.path.exists(TASK_BATCHES):
        assert 0 == os.system(
            f"clmbr_create_batches {TASK_BATCHES} --data_path {EXTRACT_LOCATION} --dictionary {DICTIONARY_PATH} --task labeled_patients --labeled_patients_path {LABELS}"
        )

    REPRESENTATIONS = os.path.join(TEMP_STORAGE, "clmbr_reprs")

    """
    Finally, you can generate features for that generated batches.
    """

    if not os.path.exists(REPRESENTATIONS):
        assert 0 == os.system(
            f"clmbr_compute_representations {REPRESENTATIONS} --data_path {EXTRACT_LOCATION} --batches_path {TASK_BATCHES} --model_dir {MODEL_PATH}"
        )

    """
    Open the resulting representations and take a look at the data matrix.
    """

    with open(REPRESENTATIONS, "rb") as f:
        reprs = pickle.load(f)

        print(reprs.keys())

        print("Pulling the data for the first label")
        print("Patient id", reprs["patient_ids"][0])
        print("Label time", reprs["labeling_time"][0])
        print("Representation", reprs["data_matrix"][0, :16])
