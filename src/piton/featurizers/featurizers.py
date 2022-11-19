from __future__ import annotations

import datetime
import os
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Iterator, List, Mapping, Optional, Tuple

from .. import Patient
from ..extension import datasets as extension_datasets
from ..labelers.core import Label
from . import Dictionary, OnlineStatistics
from .core import ColumnValue, Featurizer
from ..datasets import PatientDatabase
import torch
from . import get_gpus_with_minimum_free_memory
import numpy as np
import multiprocessing
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from . import save_to_file, load_from_file


# TODO - replace this with a more flexible/less hacky way to allow the user to
# manage patient attributes (like age)
def get_patient_birthdate(patient: Patient) -> datetime.datetime:
    return patient.events[0].start if len(patient.events) > 0 else None


class AgeFeaturizer(Featurizer):
    """
    Produces the (possibly normalized) age at each label timepoint.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.age_statistics = OnlineStatistics()

    def preprocess(self, patient: Patient, labels: List[Label]) -> None:
        if not self.needs_preprocessing():
            return

        patient_birth_date = get_patient_birthdate(patient)
        for label in labels:
            age = (label.time - patient_birth_date).days / 365
            self.age_statistics.add(age)

    def num_columns(self) -> int:
        return 1

    def featurize(
        self, patient: Patient, labels: List[Label], ontology: extension_datasets.Ontology,
    ) -> List[List[ColumnValue]]:
        all_columns: List[List[ColumnValue]] = []

        patient_birth_date = get_patient_birthdate(patient)
        for label in labels:
            age = (label.time - patient_birth_date).days / 365
            if self.normalize:
                standardized_age = (age - self.age_statistics.mean()) / (
                    self.age_statistics.standard_deviation()
                )
                all_columns.append([ColumnValue(0, standardized_age)])
            else:
                all_columns.append([ColumnValue(0, age)])

        return all_columns

    def to_dict(self) -> Dict[str, Any]:
        return {"age_statistics": self.age_statistics.to_dict()}

    def from_dict(self, data: Mapping[str, Any]) -> None:
        self.age_statistics = OnlineStatistics(data["age_statistics"])

    def needs_preprocessing(self) -> bool:
        return self.normalize


class CountFeaturizer(Featurizer):
    """
    Produces one column per each diagnosis code, procedure code, and prescription code.
    The value in each column is the count of how many times that code appears in the patient record
    before the corresponding label.
    """

    def __init__(
        self,
        # ontology: extension_datasets.Ontology,
        rollup: bool = False,
        exclusion_codes: List[int] = [],
        time_bins: Optional[
            List[Optional[int]]
        ] = None,  # [90, 180] refers to [0-90, 90-180]; [90, 180, math.inf] refers to [0-90, 90-180, 180-inf]
    ):
        self.patient_codes: Dictionary = Dictionary()
        self.exclusion_codes = set(exclusion_codes)
        self.time_bins = time_bins
        # self.ontology = ontology
        self.rollup = rollup

    def get_codes(self, code: int, ontology: extension_datasets.Ontology) -> Iterator[int]:
        if code not in self.exclusion_codes:
            if self.rollup:
                for subcode in ontology.get_all_parents(code):
                    yield subcode
            else:
                yield code

    def preprocess(self, patient: Patient, labels: List[Label]):
        """Adds every event code in this patient's timeline to `patient_codes`"""
        for event in patient.events:
            if event.value is None:
                self.patient_codes.add(event.code)

    def num_columns(self) -> int:
        if self.time_bins is None:
            return len(self.patient_codes)
        else:
            return len(self.time_bins) * len(self.patient_codes)

    def featurize(
        self, patient: Patient, labels: List[Label], ontology: extension_datasets.Ontology,
    ) -> List[List[ColumnValue]]:
        all_columns: List[List[ColumnValue]] = []

        if self.time_bins is None:
            current_codes: Dict[int, int] = defaultdict(int)

            label_idx = 0
            for event in patient.events:
                while event.start > labels[label_idx].time:
                    label_idx += 1
                    all_columns.append(
                        [
                            ColumnValue(column, count)
                            for column, count in current_codes.items()
                        ]
                    )

                    if label_idx >= len(labels):
                        return all_columns

                if event.value is not None:
                    continue

                for code in self.get_codes(event.code, ontology):
                    if code in self.patient_codes:
                        current_codes[self.patient_codes.transform(code)] += 1

            if label_idx < len(labels):
                for label in labels[label_idx:]:
                    all_columns.append(
                        [
                            ColumnValue(column, count)
                            for column, count in current_codes.items()
                        ]
                    )
                

                # if label_idx == len(labels) - 1:
                #     all_columns.append(
                #         [
                #             ColumnValue(column, count)
                #             for column, count in current_codes.items()
                #         ]
                #     )
                #     break

        else:
            codes_per_bin: Dict[int, Deque[Tuple[int, datetime.date]]] = {
                i: deque() for i in range(len(self.time_bins) + 1)
            }

            code_counts_per_bin: Dict[int, Dict[int, int]] = {
                i: defaultdict(int) for i in range(len(self.time_bins) + 1)
            }

            label_idx = 0
            for event in patient.events:
                code = event.code
                while (
                    label_idx < len(labels)
                    and event.start > labels[label_idx].time
                ):
                    label_idx += 1
                    all_columns.append(
                        [
                            ColumnValue(
                                self.patient_codes.transform(code)
                                + i * len(self.patient_codes),
                                count,
                            )
                            for i in range(len(self.time_bins))
                            for code, count in code_counts_per_bin[i].items()
                        ]
                    )
                for code in self.get_codes(code, ontology):
                    if code in self.patient_codes:
                        codes_per_bin[0].append((code, event.start))
                        code_counts_per_bin[0][code] += 1

                for i, max_time in enumerate(self.time_bins):
                    # if i + 1 == len(self.time_bins):
                    #     continue

                    if max_time is None:
                        # This means that this bin accepts everything
                        continue

                    while len(codes_per_bin[i]) > 0:
                        next_code, next_date = codes_per_bin[i][0]

                        if (event.start - next_date).days <= max_time:
                            break
                        else:
                            codes_per_bin[i + 1].append(
                                codes_per_bin[i].popleft()
                            )

                            code_counts_per_bin[i][next_code] -= 1
                            if code_counts_per_bin[i][next_code] == 0:
                                del code_counts_per_bin[i][next_code]

                            code_counts_per_bin[i + 1][next_code] += 1

                # print(codes_per_bin, " | ", code_counts_per_bin)
                # print()
                if label_idx == len(labels) - 1:
                    all_columns.append(
                        [
                            ColumnValue(
                                self.patient_codes.transform(code)
                                + i * len(self.patient_codes),
                                count,
                            )
                            for i in range(len(self.time_bins) - 1)
                            for code, count in code_counts_per_bin[i].items()
                        ]
                    )
                    break

        return all_columns

    def to_dict(self) -> Dict[str, Any]:
        return {"patient_codes": self.patient_codes.to_dict()}

    def from_dict(self, data: Mapping[str, Any]) -> None:
        self.patient_codes = Dictionary(data["patient_codes"])

    def needs_preprocessing(self) -> bool:
        return True


def _get_one_patient_text_data(patient, labels, min_char, max_char):
    text_for_all_label = []

    label_idx = 0
    current_text = []
    for event in patient.events:
        while event.start > labels[label_idx].time:
            label_idx += 1

            combined_text = " ".join(current_text)
            # if len(combined_text) == 0:
            #     combined_text = " "
            # else:
            #     text_for_all_label.append(combined_text)
            text_for_all_label.append(combined_text[-max_char:])

            if label_idx >= len(labels):
                return text_for_all_label

        if type(event.value) is not memoryview:
            continue

        text_data = bytes(event.value).decode("utf-8")

        if len(text_data) < min_char:
            continue

        current_text.append(text_data)

    if label_idx < len(labels):
        for label in labels[label_idx:]:
            combined_text = " ".join(current_text)
            # if len(combined_text) == 0:
            #     combined_text = " "
            # else:
            #     text_for_all_label.append(combined_text)
            text_for_all_label.append(combined_text[-max_char:])

    return text_for_all_label


def _get_all_patient_text_data(args):
    
    database_path, pids, labeled_patients, params_dict = args
    database = PatientDatabase(database_path)

    data = []
    patient_ids = []
    result_labels = []
    labeling_time = []
    
    for patient_id in pids:
        patient = database[patient_id]
        labels = labeled_patients.pat_idx_to_label(patient_id)

        assert len(labels) == 1  # for now since we are only doing 1 label per patient

        # if len(labels) == 0:
        #     continue
        
        patient_text_data = _get_one_patient_text_data(patient, labels, params_dict["min_char"], params_dict["max_char"])

        assert len(labels) == len(patient_text_data)
        
        for i, label in enumerate(labels):

            data.append(patient_text_data[i])
            result_labels.append(label.value)
            patient_ids.append(patient.patient_id)
            labeling_time.append(label.time)
    
    return (
        data,
        result_labels,
        patient_ids,
        labeling_time,
    )


def _get_tokenized_text(args):

    text_data, path_to_model, params_dict = args
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)

    notes_tokenized = tokenizer(
                            list(text_data),
                            padding=params_dict["padding"],
                            truncation=params_dict["truncation"],
                            max_length=params_dict["max_length"],
                            return_tensors="pt",
                        )
    return notes_tokenized


def _get_text_embeddings(args):

    tokenized_text_data, path_to_model, params_dict = args
    model = AutoModel.from_pretrained(path_to_model)

    # gpu_device_ids: List[int] = get_gpus_with_minimum_free_memory(params_dict['min_gpu_size'])

    # if len(gpu_device_ids) == 0:
    #     raise Exception("No GPUs available that meet the minimum free mem (GB) requirements")

    # device = gpu_device_ids[1]
    # # model = torch.nn.DataParallel(model, device_ids=gpu_device_ids)
    # model = model.to(device)
    # print(f"embed | using devices {gpu_device_ids}")

    batch_size: int = params_dict["batch_size"]
    train_loader = [
        {
            "input_ids": tokenized_text_data["input_ids"][x : x + batch_size],
            # "token_type_ids": tokenized_text_data["token_type_ids"][x : x + batch_size],
            "attention_mask": tokenized_text_data["attention_mask"][x : x + batch_size],
        }
        for x in range(0, tokenized_text_data["input_ids"].shape[0], batch_size)
    ]

    # print(device, f"# of batches of size {batch_size}:", len(train_loader))

    outputs = []
    for batch in tqdm(train_loader):
        output = model(
                input_ids=batch["input_ids"],
                # token_type_ids=batch["token_type_ids"].to(device),
                attention_mask=batch["attention_mask"],
            )
        outputs.append(output.last_hidden_state.detach().cpu())
    
    token_embeddings = torch.cat(outputs)
    embedding_tensor = token_embeddings[:, 0, :].squeeze()
    embedding_numpy = embedding_tensor.cpu().detach().numpy()

    # with torch.no_grad():
    #     for batch in tqdm(train_loader):
    #         output = model(
    #             input_ids=batch["input_ids"].to(device),
    #             # token_type_ids=batch["token_type_ids"].to(device),
    #             attention_mask=batch["attention_mask"].to(device),
    #         )
    #         outputs.append(output.last_hidden_state.detach().cpu())

    # token_embeddings = torch.cat(outputs)
    # embedding_tensor = token_embeddings[:, 0, :].squeeze()
    # embedding_numpy = embedding_tensor.cpu().detach().numpy()

    return embedding_numpy


class TextFeaturizer:
    def __init__(
        self,
        labeled_patients: LabeledPatients,
        database_path: str,
        random_seed: int = 1
):
        self.labeled_patients = labeled_patients
        self.database_path = database_path
        self.random_seed = random_seed
    
    def preprocess_text(self):
        pass
    
    def featurize(
        self, 
        path_to_model: str,
        path_to_save: str, 
        prefix: str = "temp",
        num_threads: int = 1,
        num_threads_gpu: int = 1, 
        min_gpu_size: int = 20,
        min_char: int = 100, 
        max_char: int = 10000, 
        max_length: int = 512, 
        padding: bool = True, 
        truncation: bool = True, 
        chunk_size: int = 1000,
        batch_size: int = 4096,
        num_patients: int = None
    ):

        pids = sorted(self.labeled_patients.get_all_patient_ids())
        if num_patients is not None:
            pids = pids[:num_patients]

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

        # Text Acculumation
        start_time = datetime.datetime.now()
        print("Starting text accumulation")
        pids_parts = np.array_split(pids, num_threads)
        tasks = [(self.database_path, pid_part, self.labeled_patients, params_dict) for pid_part in pids_parts]
        ctx = multiprocessing.get_context('forkserver')
        with ctx.Pool(num_threads) as pool:
            patient_text_data_list = list(pool.imap(_get_all_patient_text_data, tasks))

        # print(len(patient_text_data_list))

        text_data = np.concatenate([patient_text_data[0] for patient_text_data in patient_text_data_list], axis=None)
        result_labels = np.concatenate([patient_text_data[1] for patient_text_data in patient_text_data_list], axis=None)
        patient_ids = np.concatenate([patient_text_data[2] for patient_text_data in patient_text_data_list], axis=None)
        labeling_time = np.concatenate([patient_text_data[3] for patient_text_data in patient_text_data_list], axis=None)

        save_to_file((text_data, result_labels, patient_ids, labeling_time), os.path.join(path_to_save, f"{prefix}_text_data.pickle"))
        save_to_file((result_labels, patient_ids, labeling_time), os.path.join(path_to_save, f"{prefix}_meta_data.pickle"))

        print("Finished text accumulation: ", datetime.datetime.now() - start_time)

        # Generate Tokenization
        print("Starting Tokenization")
        text_data_parts = np.array_split(text_data, num_threads)
        tasks = [(text_data_part, path_to_model, params_dict) for text_data_part in text_data_parts]
        ctx = multiprocessing.get_context('forkserver')
        with ctx.Pool(num_threads) as pool:
            tokenized_text_list = list(pool.imap(_get_tokenized_text, tasks))

        save_to_file(tokenized_text_list, os.path.join(path_to_save, f"{prefix}_tokenized_text.pickle"))
        
        print("Finished Tokenization: ", datetime.datetime.now() - start_time)

        # print(tokenized_text_list[0].keys())
        # exit()
        print("Starting Generating Embedding")
        # embeddings_list = []
        # for tokenized_text in tokenized_text_list:
        #     embeddings_list.append(_get_text_embeddings((tokenized_text, path_to_model, params_dict)))
        
        # print("Finished Generating Embedding: ", datetime.datetime.now() - start_time)

        # Generate Embeddings
        tasks = [(tokenized_text, path_to_model, params_dict) for tokenized_text in tokenized_text_list]
        ctx = multiprocessing.get_context('forkserver')
        with ctx.Pool(num_threads) as pool:
            embeddings_list = list(pool.imap(_get_text_embeddings, tasks))
        embeddings = np.concatenate(embeddings_list)

        result_tuple = (
            embeddings,
            result_labels,
            patient_ids,
            labeling_time,
        )

        save_to_file(result_tuple, os.path.join(path_to_save, f"{prefix}_embeddings.pickle"))

        return result_tuple
