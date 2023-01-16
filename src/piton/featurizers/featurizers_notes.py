from __future__ import annotations

import multiprocessing
import os
import pickle
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import transformers
from icecream import ic
from nptyping import Int, NDArray, Shape
from torchtyping import TensorType
from transformers import AutoModel, AutoTokenizer

<<<<<<< HEAD
=======

>>>>>>> ab5430786a61a65d5b57e2490cd02ea0e0292b05
from .. import Event, Patient
from ..datasets import PatientDatabase
from ..labelers.core import Label, LabeledPatients
from .core import ColumnValue

NotesTokenized = Dict[
    str, TensorType["n_notes", "max_note_token_count", int]
]  # noqa
NotesEmbedded = TensorType["n_notes", "embedding_length"]  # noqa
NotesEmbeddedByToken = TensorType[
    "n_notes", "max_note_token_count", "embedding_length"
]  # noqa
NotesProcessed = List[Tuple[int, Event]]  # event_idx, event (note)
PatientLabelNotesTuple = Tuple[
    int, int, NotesProcessed
]  # patient_id, label_idx, notes


START_TIME = time.time()


def print_log(name: str, content: str):
    print(f"{int(time.time() - START_TIME)} | {name} | {content}")


def save_to_pkl(object_to_save, path_to_file: str):
    """Save object to Pickle file."""
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "wb") as fd:
        print(fd is None, object_to_save is None)
        pickle.dump(object_to_save, fd)


def load_from_pkl(path_to_file: str):
    """Load object from Pickle file."""
    with open(path_to_file, "rb") as fd:
        result = pickle.load(fd)
    return result


# Helper functions for loading/saving
# Saving
def save_preprocessed_notes_chunk(path_to_folder: str, chunk: int, notes: List):
    """Save `notes_preprocessed_{chunk}.pkl`."""
    path_to_file: str = os.path.join(
        path_to_folder, f"notes_preprocessed_{chunk}.pkl"
    )
    save_to_pkl(notes, path_to_file)


def save_tokenized_notes_chunk(
    path_to_folder: str, chunk: int, notes: NotesTokenized
):
    """Save `notes_tokenized_{chunk}.pkl`."""
    path_to_file: str = os.path.join(
        path_to_folder, f"notes_tokenized_{chunk}.pkl"
    )
    save_to_pkl(notes, path_to_file)


def save_embedded_notes_chunk(
    path_to_folder: str, chunk: int, notes: NotesEmbedded
):
    """Save `notes_embedded_{chunk}.pt`."""
    path_to_file: str = os.path.join(
        path_to_folder, f"notes_embedded_{chunk}.pt"
    )
    torch.save(notes, path_to_file)


# Loading
def load_preprocessed_note_chunk(path_to_folder: str, chunk: int) -> List:
    """Load `notes_preprocessed_{chunk}.pkl`."""
    path_to_file = os.path.join(
        path_to_folder, f"notes_preprocessed_{chunk}.pkl"
    )
    return load_from_pkl(path_to_file)


def load_tokenized_note_chunk(
    path_to_folder: str, chunk: int
) -> NotesTokenized:
    """Load `notes_tokenized_{chunk}.pkl`."""
    path_to_file: str = os.path.join(
        path_to_folder, f"notes_tokenized_{chunk}.pkl"
    )
    return load_from_pkl(path_to_file)


def load_embedded_note_chunk(path_to_folder: str, chunk: int) -> NotesEmbedded:
    """Load `notes_embedded_{chunk}.pt`."""
    path_to_file = os.path.join(path_to_folder, f"notes_embedded_{chunk}.pt")
    return torch.load(path_to_file)


# Check if file exists
def is_exist_preprocessed_note_chunk(path_to_folder: str, chunk: int) -> bool:
    """Return TRUE if this chunk exists."""
    path_to_file = os.path.join(
        path_to_folder, f"notes_preprocessed_{chunk}.pkl"
    )
    return os.path.exists(path_to_file)


def is_exist_tokenized_note_chunk(path_to_folder: str, chunk: int) -> bool:
    """Return TRUE if this chunk exists."""
    path_to_file = os.path.join(path_to_folder, f"notes_tokenized_{chunk}.pkl")
    return os.path.exists(path_to_file)


def is_exist_embedded_note_chunk(path_to_folder: str, chunk: int) -> bool:
    """Return TRUE if this chunk exists."""
    path_to_file = os.path.join(path_to_folder, f"notes_embedded_{chunk}.pkl")
    return os.path.exists(path_to_file)


# Embedding aggregation
def embed_with_mean_over_all_tokens(
    embeddings: NotesEmbeddedByToken,
) -> NotesEmbedded:
    return torch.mean(embeddings, dim=1).clone()


def embed_with_cls(embeddings: NotesEmbeddedByToken) -> NotesEmbedded:
    return embeddings[:, 0, :].squeeze()


class NoteFeaturizer:
    def __init__(
        self,
        path_to_patient_database: str = "",
        path_to_tokenizer: str = "",
        path_to_embedder: str = "",
        path_to_temp_dir: str = "",
        path_to_output_dir: str = "",
        preprocess_transformations: List = [],
        params_preprocessor: Optional[dict] = None,
        params_tokenizer: Optional[dict] = None,
        params_embedder: Optional[dict] = None,
        n_cpu_jobs: int = 1,
        gpu_devices: List[int] = [],
    ):
        self.path_to_patient_database: str = (
            path_to_patient_database  # path to load PatientDatabase
        )
        self.path_to_temp_dir: str = path_to_temp_dir  # path to store intermediate files generated by featurizer
        self.path_to_tokenizer: str = (
            path_to_tokenizer  # path to HuggingFace Tokenizer
        )
        self.path_to_embedder: str = (
            path_to_embedder  # path to HuggingFace Model
        )
        self.path_to_output_dir: str = (
            path_to_output_dir  # path to store final output of featurizer
        )
        self.params_preprocessor = params_preprocessor
        self.params_tokenizer = params_tokenizer
        self.params_embedder = params_embedder
        self.preprocess_transformations: List[
            Callable
        ] = preprocess_transformations
        self.n_cpu_jobs: int = n_cpu_jobs
        self.gpu_devices: List[int] = gpu_devices

    def get_num_columns(self) -> int:
        return 768

    def is_needs_preprocessing(self) -> bool:
        return False

    def get_name(self) -> str:
        return "TextFeaturizer"

    def preprocess_parallel(self, args: Tuple) -> List[PatientLabelNotesTuple]:
        """Return a pre-processed version of each note in `notes`.

        Based on `args`, this creates a list of transformation functions which will be
        sequentially applied to each note to "preprocess" them. Each transformation function
        takes in a string and outputs a transformed version of that string. Thus, they can
        be composed together in arbitrary orderings.
        """
        patient_database: PatientDatabase = PatientDatabase(args[0])
        patient_ids: List[int] = args[1]
        labeled_patients: LabeledPatients = args[2]
        transformations: List[Callable] = args[3]
        path_to_temp_dir: str = args[4]
        is_force_refresh: bool = args[5]
        params: dict = args[6]

        chunk_id = patient_ids[0]  # identify chunk by its first patient ID
        if (not is_force_refresh) and is_exist_preprocessed_note_chunk(
            path_to_temp_dir, chunk_id
        ):
            print_log("preprocess", f"loading pre-written chunk #{chunk_id}")
            return load_preprocessed_note_chunk(path_to_temp_dir, chunk_id)

        # Apply transformation for each (patient, label) combo
        print_log(
            "preprocess", f"applying transformations to chunk #{chunk_id}"
        )
        notes_for_labels: List[PatientLabelNotesTuple] = []
        percent_done: float = 0.05  # note: need this hacky way of doing tqdm b/c of tqdm's printouts will be ruined by multiprocessing
        for patient_idx, patient_id in enumerate(patient_ids):
            patient: Patient = patient_database[patient_id]  # type: ignore
            labels: List[Label] = labeled_patients.get_labels_from_patient_idx(
                patient_id
            )
            for label_idx, label in enumerate(labels):
                # All events that have a `value` of type `str` are clinical notes
                notes: NotesProcessed = [
                    (event_idx, event)
                    for event_idx, event in enumerate(patient.events)
                    if isinstance(event.value, str)
                ]
                # Apply transformations sequentially to `notes`
                for transform in transformations:
                    notes = transform(notes, label, **params)
                notes_for_labels.append((patient_id, label_idx, notes))
            # logging
            if patient_idx / len(patient_ids) > percent_done:
                print_log(
                    "preprocess",
                    f"done with {int(percent_done * 100)}% of chunk #{chunk_id}",
                )
                percent_done += 0.05

        save_preprocessed_notes_chunk(
            path_to_temp_dir, chunk_id, notes_for_labels
        )
        print_log("preprocess", f"finished chunk #{chunk_id}")
        return notes_for_labels

    def tokenize_parallel(self, args: Tuple) -> NotesTokenized:
        """Return a tokenized version of each note in `notes`.

        Returns:
            NotesTokenized: Output of running `tokenizer()` on `notes`.
            Dictionary with three keys:
                - input_ids (TensorType['n_notes', 'max_note_token_count', int]): Vocabulary indices that
                    correspond to each token in the sentence
                - token_type_ids (Optional[TensorType['n_notes', 'max_note_token_count', int]]): Which
                    sequence a token belongs to IF there is more than one sequence
                    (i.e. for [CLS] SEQUENCE_A [SEP] SEQUENCE_B [SEP])
                - attention_mask (TensorType['n_notes', 'max_note_token_count', int]): 1 if token should
                    be attended to; 0 otherwise
                Note that the ordering of the notes in `NotesTokenized` is the same as their original
                ordering in `notes_preprocessed`
        """
        patient_ids: List[int] = args[0]
        path_to_tokenizer: str = args[1]
        path_to_temp_dir: str = args[2]
        is_force_refresh: bool = args[3]
        params = args[4]

        chunk_id = patient_ids[0]  # identify chunk by its first patient ID
        if (not is_force_refresh) and is_exist_tokenized_note_chunk(
            path_to_temp_dir, chunk_id
        ):
            print_log("tokenize", f"loading pre-written chunk #{chunk_id}")
            return load_tokenized_note_chunk(path_to_temp_dir, chunk_id)

        notes_for_labels: List[
            PatientLabelNotesTuple
        ] = load_preprocessed_note_chunk(path_to_temp_dir, chunk_id)

        # Parse arguments
        tokenizer = AutoTokenizer.from_pretrained(
            path_to_tokenizer, use_fast=True, batched=True
        )
        max_length = params.get("tokenizer_max_length", 512)
        padding = params.get("tokenizer_padding", True)
        truncation = params.get("tokenizer_truncation", True)

        # Run notes through an already-trained tokenizer
<<<<<<< HEAD
        text: List = [ note[1].value for (_, _, notes) in notes_for_labels for note in notes ]
=======
        text: List = [
            note[1].value
            for (__, __, notes) in notes_for_labels
            for note in notes
        ]
>>>>>>> ab5430786a61a65d5b57e2490cd02ea0e0292b05
        notes_tokenized: NotesTokenized = tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )

        # Save to file
        save_tokenized_notes_chunk(path_to_temp_dir, chunk_id, notes_tokenized)
        print_log("tokenize", f"finished chunk #{chunk_id}")

        return notes_tokenized

    def embed_parallel(self, args: Tuple) -> List[NotesEmbedded]:
        """Return an embedding vector for each note in `notes_tokenized`."""
        patient_ids_chunks: List[List[int]] = args[0]
        device: str = args[1]
        path_to_embedder: str = args[2]
        path_to_temp_dir: str = args[3]
        is_force_refresh: bool = args[4]
        params: dict = args[5]

        notes_embedded: List[NotesEmbedded] = []

        for patient_ids in patient_ids_chunks:
            chunk_id = patient_ids[0]  # identify chunk by its first patient ID
            if (not is_force_refresh) and is_exist_embedded_note_chunk(
                path_to_temp_dir, chunk_id
            ):
                print_log("embed", f"loading pre-written chunk #{chunk_id}")
                notes_embedded.append(
                    load_embedded_note_chunk(path_to_temp_dir, chunk_id)
                )
                continue

            # Get tokenization of notes
            notes_tokenized: NotesTokenized = load_tokenized_note_chunk(
                path_to_temp_dir, chunk_id
            )
            model = AutoModel.from_pretrained(path_to_embedder)
            embed_method: Callable = params.get("embed_method", embed_with_cls)

            # Move to GPU
            if device != "cpu":
                t = torch.cuda.get_device_properties(device).total_memory
                r = torch.cuda.memory_reserved(device)
                a = torch.cuda.memory_allocated(device)
                print_log(
                    "embed",
                    f"chunk #{chunk_id} | device {device}, total (mem): {t}, reserved: {r}, allocated: {a}, free: {t - r - a}",
                )

            try:
                model = model.to(device)
            except Exception as e:
                print_log("embed", str(e))
                raise InterruptedError(
                    f"Error loading model onto device {device}"
                )
            print_log("embed", f"chunk #{chunk_id} | device {device}")

            # Setup dataloader for model
            batch_size: int = params.get("batch_size", 4096)
            train_loader = [
                {
                    key: notes_tokenized[key][x : x + batch_size]
                    for key in notes_tokenized.keys()
                }
                for x in range(
                    0, notes_tokenized["input_ids"].shape[0], batch_size
                )
            ]
            print_log(
                "embed",
                f"chunk #{chunk_id} | device {device} | # of batches of size {batch_size}: {len(train_loader)}",
            )

            # Get embedding for each token for each note
            outputs: List[
                transformers.tokenization_utils_base.BatchEncoding
            ] = []
            with torch.no_grad():
                for batch in train_loader:
                    output = model(
                        input_ids=batch["input_ids"].to(device),
                        token_type_ids=batch["token_type_ids"].to(device)
                        if "token_type_ids" in batch
                        else None,
                        attention_mask=batch["attention_mask"].to(device)
                        if "attention_mask" in batch
                        else None,
                    )
                    result: TensorType[
                        "batch_size", "max_note_token_count", "embedding_length"
                    ] = output.last_hidden_state.detach().cpu()
                    outputs.append(result)

            # Merge together all token embeddings
            print_log("embed", f"chunk #{chunk_id} | concatenating embeddings")
            token_embeddings: NotesEmbeddedByToken = torch.cat(outputs)

            assert (
                token_embeddings.shape[0]
                == notes_tokenized["input_ids"].shape[0]
            )
            assert (
                token_embeddings.shape[1]
                == notes_tokenized["input_ids"].shape[1]
            )

            # Derive a single embedding representation for this note from its token embeddings
            # Note: Need clone(), otherwise will keep original Tensor when save to file
            #   see: https://discuss.pytorch.org/t/saving-tensor-with-torch-save-uses-too-much-memory/46865
            #   or: https://pytorch.org/docs/stable/notes/serialization.html#preserve-storage-sharing
            print_log("embed", f"chunk #{chunk_id} | aggregating embeddings")
            note_embeddings: NotesEmbedded = embed_method(
                token_embeddings
            ).clone()
            assert note_embeddings.shape[0] == token_embeddings.shape[0]
            assert note_embeddings.shape[1] == token_embeddings.shape[2]

            # Save to file
            print_log("embed", f"chunk #{chunk_id} | saving embeddings")
            save_embedded_notes_chunk(
                path_to_temp_dir, chunk_id, note_embeddings
            )
            reload = load_embedded_note_chunk(path_to_temp_dir, chunk_id)
            assert reload.shape == note_embeddings.shape
            assert type(reload) == type(note_embeddings)

            notes_embedded.append(note_embeddings)
            print_log("embed", f"finished chunk #{chunk_id}")

        return notes_embedded

    def featurize(
        self,
        labeled_patients: LabeledPatients,
        num_patients_per_chunk: Optional[int] = None,
        is_force_refresh: bool = False,
        is_debug: bool = False,
    ) -> List[List[ColumnValue]]:
        """Run all steps of note processing pipeline in sequence (preprocess -> tokenize -> embed)."""

        print_log("featurize", "START")
        # A 'chunk' is a single intermediate file used to shard computations
        num_patients_per_chunk = num_patients_per_chunk or 5000

        patient_ids: List[int] = sorted(labeled_patients.get_all_patient_ids())
        if is_debug:
            patient_ids = patient_ids[:10000]  # 2 chunks
        num_chunks: int = int(
            np.ceil(len(patient_ids) / num_patients_per_chunk)
        )
        patient_ids_by_chunk: List[NDArray] = np.array_split(
            patient_ids, num_chunks
        )
        print_log(
            "featurize",
            f"Using {num_chunks} chunks, with {num_patients_per_chunk} patients per chunk",
        )

        # preprocess notes
        print_log("featurize", "Starting Preprocessing...")
        tasks: List[Tuple] = [
            (
                self.path_to_patient_database,
                patient_ids_in_chunk,
                labeled_patients,
                self.preprocess_transformations,
                self.path_to_temp_dir,
                is_force_refresh,
                self.params_preprocessor,
            )
            for patient_ids_in_chunk in patient_ids_by_chunk
        ]
        ctx = multiprocessing.get_context("forkserver")
        with ctx.Pool(self.n_cpu_jobs) as pool:
            preprocess_parallel_result: List[
                List[PatientLabelNotesTuple]
            ] = list(pool.imap(self.preprocess_parallel, tasks))
        patient_label_notes_tuples: List[PatientLabelNotesTuple] = [
            y for x in preprocess_parallel_result for y in x
        ]
        np_patient_ids: NDArray[Shape["n_patients,1"], Int] = np.array(
            [x[0] for x in patient_label_notes_tuples]
        )
        np_label_idxs: NDArray[Shape["n_patients,1"], Any] = np.array(
            [x[1] for x in patient_label_notes_tuples]
        )
        print_log("featurize", "Finished Preprocessing...")

        # tokenize notes
        print_log("featurize", "Starting Tokenization...")
        tasks: List[Tuple] = [
            (
                patient_ids_in_chunk,
                self.path_to_tokenizer,
                self.path_to_temp_dir,
                is_force_refresh,
                self.params_tokenizer,
            )
            for patient_ids_in_chunk in patient_ids_by_chunk
        ]
        ctx = multiprocessing.get_context("forkserver")
        with ctx.Pool(self.n_cpu_jobs) as pool:
            _: List[NotesTokenized] = list(
                pool.imap(self.tokenize_parallel, tasks)
            )
        print_log("featurize", "Finished Tokenization...")

        # embed notes
        print_log("featurize", "Starting Embedding...")
        # Note: need to have num GPUs < num chunks to avoid multiple threads locking trying to read same chunk
        n_gpu_jobs: int = min(num_chunks, len(self.gpu_devices))
        n_chunks_per_gpu: int = int(np.ceil(num_chunks / n_gpu_jobs))
        print_log(
            "featurize",
            f"Choosing {n_gpu_jobs} devices from ({str(self.gpu_devices)}), with {n_chunks_per_gpu} chunks per device",
        )
        tasks: List[Tuple] = [
            (
                patient_ids_by_chunk[
                    i * n_chunks_per_gpu : (i + 1) * n_chunks_per_gpu
                ],
                self.gpu_devices[i],
                self.path_to_embedder,
                self.path_to_temp_dir,
                is_force_refresh,
                self.params_embedder,
            )
            for i in range(0, n_gpu_jobs)
        ]
        ctx = multiprocessing.get_context("forkserver")
        with ctx.Pool(n_gpu_jobs) as pool:
            embed_parallel_result: List[NotesEmbedded] = list(
                pool.imap(self.embed_parallel, tasks)
            )
        embeddings: NDArray[
            Shape["n_patients", "embedding_length"], float
        ] = np.vstack(
            [y.numpy() for x in embed_parallel_result for y in x]
        )  # unwrap nested lists
        print_log("featurize", "Finished Embedding...")

        # save features to file
        assert (
            np_patient_ids.shape[0]
            == np_label_idxs.shape[0]
            == embeddings.shape[0]
        )
        result_tuple: Tuple[NDArray, NDArray, NDArray] = (
            embeddings,
            np_patient_ids,
            np_label_idxs,
        )
        save_to_pkl(
            result_tuple, os.path.join(self.path_to_output_dir, f"features.pkl")
        )
        print_log("featurize", "DONE")
        # TODO: Fix typing
        return result_tuple
