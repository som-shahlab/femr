import os
import pickle
import tensorflow as tf

import piton.extension.dataloader
import msgpack
import numpy as np

data_path = "/local-scratch/nigam/projects/ethanid/piton_1_extract"

dictionary_path = (
    "/local-scratch/nigam/projects/ethanid/piton/native/results/dictionary"
)

surv_dictionary_path = "/local-scratch/nigam/projects/ethanid/piton/native/results/survival_clmbr_dictionary"

import piton.datasets

data = piton.datasets.PatientDatabase(data_path)
male_code = data.get_code_dictionary().index("Gender/M")

import json

dictionary = msgpack.load(open(dictionary_path, "rb"), use_list=False)

if True:
    surv_dict = msgpack.load(open(surv_dictionary_path, "rb"), use_list=False)
    print(surv_dict.keys())
    task = {"type": "survival_clmbr", "survival_dict": surv_dict}
elif False:
    task = {"type": "clmbr", "vocab_size": 10_000}
else:
    labels = []

    if False:
        limit = 100
    else:
        limit = len(data)

    for patient_id in range(0, limit):
        patient = data[patient_id]
        is_male = any(event.code == male_code for event in patient.events)
        labels.append((patient.patient_id, 1, is_male))
    task = {"type": "binary", "labels": labels}

config = {
    "task": task,
    "seed": 97,
    "vocab_size": 50000,
    "dictionary": dictionary,
    "max_size": 13,
    "splits": [["train", 0, 70], ["dev", 70, 85], ["test", 85, 100]],
    "hidden_size": 768,
    "intermediate_size": 3072,
    "n_heads": 12,
    "n_layers": 6,
    "n_classes": 2,
    "learning_rate": 1e-4,
    "max_grad_norm": 1.0,
    "l2": 0,
    "rotary": "disabled",
    "n_epochs": 100,
}
print("WORKING WITH", config["learning_rate"])

print(len(config["dictionary"]))

with open("trash/config.json", "bw") as f:
    msgpack.dump(config, f)

loader = piton.extension.dataloader.BatchCreator(data_path, "trash/config.json")

print("Starting to load ...")

root_dir = "splits2"
os.mkdir(root_dir)


for split in config["splits"]:
    split_name = split[0]

    split_dir = os.path.join(root_dir, split_name)

    os.mkdir(split_dir)

    shards = 10
    num_batches = loader.get_number_of_batches(split_name)
    items_per_shard = (num_batches + shards - 1) // shards

    for i in range(shards):
        with tf.io.TFRecordWriter(
            os.path.join(split_dir, f"{i}.tfrecord")
        ) as writer:
            for i in range(
                i * items_per_shard, min((i + 1) * items_per_shard, num_batches)
            ):
                result = loader.get_batch(split_name, i)
                if False:
                    for k, v in result["transformer"].items():
                        print(k, v.shape)
                    task_result = result["task"]
                    for k, v in task_result.items():
                        print(k, v.shape)
                        print(v)
                # pickle.dump(loader.get_batch(split_name, i), open("blah", "wb"))

                data = pickle.dumps(result)
                writer.write(data)
