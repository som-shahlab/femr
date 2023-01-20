import argparse
import os
import pickle
import time
from typing import Callable, List

import piton
from piton.datasets import PatientDatabase
from piton.featurizers.featurizers_notes import NoteFeaturizer
from piton.labelers.core import LabeledPatients
from piton.transforms.notes import (
    join_all_notes,
    keep_only_last_n_chars,
    keep_only_notes_matching_codes,
    remove_notes_after_label,
    remove_short_notes,
)

"""
Example running:

python3 3_run_text_featurizer.py \
    /local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract_v5 \
    /local-scratch/nigam/projects/mwornow/data/mortality_labeled_patients_v1.pickle \
    /local-scratch/nigam/projects/clmbr_text_assets/models/Clinical-Longformer/ \
    /local-scratch/nigam/projects/mwornow/data/output_mortality_longformer/ \
    /local-scratch/nigam/projects/mwornow/data/temp_mortality_longformer/ \
    --num_threads 20 \
    --gpu_devices 0 1 3 4 5 6 \
    --preprocessor__transformations keep_only_notes_matching_codes remove_notes_after_label remove_short_notes join_all_notes keep_only_last_n_chars \
    --preprocessor__min_note_char_count 100 \
    --preprocessor__keep_last_n_chars 4096 \
    # --preprocessor__keep_notes_of_type discharge \
    --tokenizer__max_length 4096 \
    --tokenizer__padding \
    --tokenizer__truncation \
    --embedder__method cls \
    --embedder__batch_size 32 \
    --is_force_refresh
"""

EMBEDDER_METHODS = [
    "cls",
]
NOTE_TYPES = [
    "discharge",
    "procedure",
    "progress",
    "note",
]
PREPROCESSOR_TRANFORMATIONS = [
    "keep_only_notes_matching_codes",
    "remove_notes_after_label",
    "remove_short_notes",
    "join_all_notes",
    "keep_only_last_n_chars",
]


def get_gpus_with_minimum_free_memory(min_mem: float = 5) -> List[int]:
    """Return a list of GPU devices with at least `min_mem` free memory is in GB."""
    import torch

    devices = []
    num_gpus: int = torch.cuda.device_count()
    for i in range(num_gpus):
        free, __ = torch.cuda.mem_get_info(i)
        if free >= min_mem * 1e9:
            devices.append(i)
    return devices


if __name__ == "__main__":
    START_TIME = time.time()

    def print_log(name: str, content: str):
        print(f"{int(time.time() - START_TIME)} | {name} | {content}")

    parser = argparse.ArgumentParser(description="Run Piton text featurizer")
    parser.add_argument(
        "path_to_patient_database",
        type=str,
        help="Path of folder to the Piton PatientDatabase. Example: '/local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract_v5'",
    )
    parser.add_argument(
        "path_to_labeled_patients",
        type=str,
        help="Path to file containing the Piton LabeledPatients. Example: '/local-scratch/nigam/projects/mwornow/data/mortality_labeled_patients_v1.pickle'",
    )
    parser.add_argument(
        "path_to_huggingface_model",
        type=str,
        help="Path of folder containing the HuggingFace model for text featurization. Example: '/local-scratch/nigam/projects/clmbr_text_assets/models/Clinical-Longformer/'",
    )
    parser.add_argument(
        "path_to_save_featurized_notes",
        type=str,
        help="Path to folder to save features for notes. Example: '/local-scratch/nigam/projects/mwornow/data/output_mortality_longformer/'",
    )
    parser.add_argument(
        "path_to_temp_dir",
        type=str,
        help="Path to folder where temporary files will be written. Example: '/local-scratch/nigam/projects/mwornow/data/temp_mortality_longformer/'",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        help="The number of threads to use",
        default=1,
    )
    parser.add_argument(
        "--gpu_devices",
        type=int,
        nargs="*",
        help="The GPU devices to use for running the HuggingFace Model to generate embeddings. Example: '0 1 2 3'",
        default=[],
    )
    parser.add_argument(
        "--is_force_refresh",
        action=argparse.BooleanOptionalAction,
        help="If specified, then recreate all aspects of this featurization from scratch. Otherwise, this wil try to re-use previous work stored in the `path_to_temp_dir`.",
        default=False,
    )
    parser.add_argument(
        "--preprocessor__transformations",
        type=str,
        nargs="*",
        choices=PREPROCESSOR_TRANFORMATIONS,
        help="List of transformations to apply to notes (in sequential order) before featurization. Example: 'keep_only_notes_matching_codes remove_notes_after_label remove_short_notes join_all_notes keep_only_last_n_chars'",
        default=[],
    )
    parser.add_argument(
        "--preprocessor__min_note_char_count",
        type=int,
        help="Remove notes with fewer than this many characters. If not specified, defaults to keeping all notes.",
        default=None,
    )
    parser.add_argument(
        "--preprocessor__keep_last_n_chars",
        type=int,
        help="Keep only the last `n` characters of each note",
        default=None,
    )
    parser.add_argument(
        "--preprocessor__keep_notes_of_type",
        type=str,
        nargs="*",
        choices=NOTE_TYPES,
        help="Keep only notes that are of this type. If not specified, defaults to keeping all notes. Example: 'discharge nurse'",
        default=[],
    )
    # parser.add_argument(
    #     "--preprocessor__valid_note_codes",
    #     type=str,
    #     nargs='*',
    #     help="Keep only notes with these event codes. Example: 'LOINC/28570-0 LOINC/11506-3 LOINC/18842-5 LOINC/LP173418-7'",
    #     default=[],
    # )
    parser.add_argument(
        "--tokenizer__max_length",
        type=int,
        help="Context window size for tokenizer",
        default=None,
    )
    parser.add_argument(
        "--tokenizer__padding",
        action=argparse.BooleanOptionalAction,
        help="If specified, then pad notes to `max_length`. Otherwise, no padding is applied.",
        default=False,
    )
    parser.add_argument(
        "--tokenizer__truncation",
        action=argparse.BooleanOptionalAction,
        help="If specified, then truncate notes to `max_length`. Otherwise, no truncation is applied",
        default=False,
    )
    parser.add_argument(
        "--embedder__method",
        type=str,
        choices=EMBEDDER_METHODS,
        help="The method to use for embedding. Default to using the 'cls' token to represent a note",
        default=EMBEDDER_METHODS[0],
    )
    parser.add_argument(
        "--embedder__batch_size",
        type=int,
        help="The batch size to use for feeding notes into the HuggingFace Model. Make sure this is small enough to fit onto your GPU.",
        default=None,
    )

    # Parse CLI args
    args = parser.parse_args()
    PATH_TO_PATIENT_DATABASE: str = args.path_to_patient_database
    PATH_TO_LABELED_PATIENTS: str = args.path_to_labeled_patients
    PATH_TO_HUGGINGFACE_MODEL: str = args.path_to_huggingface_model
    PATH_TO_OUTPUT_DIR: str = args.path_to_save_featurized_notes
    PATH_TO_TEMP_DIR: str = args.path_to_temp_dir
    num_threads: int = args.num_threads
    gpu_devices: List = (
        args.gpu_devices
        if len(args.gpu_devices) > 0
        else get_gpus_with_minimum_free_memory(10)
    )
    is_force_refresh: bool = args.is_force_refresh
    os.makedirs(PATH_TO_TEMP_DIR, exist_ok=True)
    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)
    assert num_threads > 0, "ERROR - `num_threads` must be greater than 0"
    assert (
        len(gpu_devices) > 0
    ), f"ERROR - Not enough GPUs specified. Must specify at least 1."

    num_patients_per_chunk: int = (
        20000  # Use 20,000 patients per chunk - TODO: Make this a CLI arg?
    )

    # Load code dictionary
    data = PatientDatabase(PATH_TO_PATIENT_DATABASE)
    code_dictionary = data.get_code_dictionary()

    # Load LabeledPatients
    with open(PATH_TO_LABELED_PATIENTS, "rb") as fd:
        labeled_patients: LabeledPatients = pickle.load(fd)
        print_log(
            "LabeledPatients",
            f"# of loaded labeled patients = {len(labeled_patients)}",
        )

    # Filter by note type
    valid_note_source_codes: List[str] = []
    for t in args.preprocessor__keep_notes_of_type:
        if t == "discharge":
            valid_note_source_codes += [
                "LOINC/18842-5",
            ]  # 699
        elif t == "procedure":
            valid_note_source_codes += [
                "LOINC/28570-0",
            ]  # 11
        elif t == "progress":
            valid_note_source_codes += [
                "LOINC/11506-3",
            ]  # 10
        elif t == "note":
            valid_note_source_codes += [
                "LOINC/LP173418-7",
            ]  # 19
        else:
            raise NotImplementedError(
                f"Codes for note type {t} have not been implemented yet."
            )
    valid_note_codes: List[int] = [
        code_dictionary.index(x) for x in valid_note_source_codes
    ]

    # Choose embedding method
    if args.embedder__method == "cls":
        embed_method: Callable = (
            piton.featurizers.featurizers_notes.embed_with_cls
        )
    else:
        raise ValueError(
            f"Invalid `embed_method` ({args.embedder__method}) specified"
        )

    # Set up preprocessing transformations
    preprocess_transformations: List[Callable] = []
    for t in args.preprocessor__transformations:
        if t == "keep_only_notes_matching_codes":
            preprocess_transformations.append(keep_only_notes_matching_codes)
        elif t == "remove_notes_after_label":
            preprocess_transformations.append(remove_notes_after_label)
        elif t == "remove_short_notes":
            preprocess_transformations.append(remove_short_notes)
        elif t == "join_all_notes":
            preprocess_transformations.append(join_all_notes)
        elif t == "keep_only_last_n_chars":
            preprocess_transformations.append(keep_only_last_n_chars)
        else:
            raise ValueError(
                f"Invalid preprocess transformation ({t}) specified"
            )

    # Logging
    print_log(
        "ArgParse",
        f"Keep only notes with these Piton event codes: {valid_note_codes}",
    )
    print_log(
        "ArgParse",
        f"    ...which correspond to these source codes: {valid_note_source_codes}",
    )
    print_log("ArgParse", f"Use these GPUs: {gpu_devices}")
    print_log(
        "ArgParse", f"Use this embedding method: '{embed_method.__name__}'"
    )
    print_log(
        "ArgParse",
        f"Apply these transformations in order: {[ x.__name__ for x in preprocess_transformations ]}",
    )

    # Run note featurizer
    note_featurizer = NoteFeaturizer(
        path_to_patient_database=PATH_TO_PATIENT_DATABASE,
        path_to_tokenizer=PATH_TO_HUGGINGFACE_MODEL,
        path_to_embedder=PATH_TO_HUGGINGFACE_MODEL,
        path_to_temp_dir=PATH_TO_TEMP_DIR,
        path_to_output_dir=PATH_TO_OUTPUT_DIR,
        n_cpu_jobs=num_threads,
        gpu_devices=gpu_devices,
        params_preprocessor={
            "min_char_count": args.preprocessor__min_note_char_count,
            "keep_last_n_chars": args.preprocessor__keep_last_n_chars,
            "keep_notes_with_codes": valid_note_codes,
        },
        params_tokenizer={
            "tokenizer_max_length": args.tokenizer__max_length,
            "tokenizer_padding": args.tokenizer__padding,
            "tokenizer_truncation": args.tokenizer__truncation,
        },
        params_embedder={
            "embed_method": embed_method,
            "batch_size": 32,
        },
        preprocess_transformations=preprocess_transformations,
    )

    print_log("NoteFeaturizer", "Starting")
    result_tuple = note_featurizer.featurize(
        labeled_patients,
        num_patients_per_chunk=num_patients_per_chunk,
        is_force_refresh=is_force_refresh,
        is_debug=False,
    )
    print_log("NoteFeaturizer", "Finished")
    print_log("NoteFeaturizer", f"Result shape: {result_tuple[0].shape}")
