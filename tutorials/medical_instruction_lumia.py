#!/usr/bin/env python
# coding: utf-8

# In[3]:


import multiprocessing
import os
import pickle
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

# In[4]:


def reservoir_sampling(iterable, n, seed=1234):
    np.random.seed(seed)
    i = 0
    pool = []
    for item in iterable:
        if len(pool) < n:
            pool.append(item)
        else:
            k = np.random.randint(0, i)
            if k < n:
                pool[k] = item

        i += 1
    return pool


# In[5]:


class ReservoirSampler:
    def __init__(self, k, rng_seed):
        self.k = k
        self.total = 0
        self.values = []
        self.rng = random.Random(rng_seed)

    def add(self, value):
        if len(self.values) < self.k:
            self.values.append(value)
        else:
            r = self.rng.randint(0, self.total)
            if r < self.k:
                self.values[r] = value

        self.total += 1


def sample_from_lumia(args):
    path, train_file, path_to_save, i, k, seed, num_threads = args
    print(train_file)

    reservior_sampler = ReservoirSampler(int(k / num_threads), seed)

    # exclude_shard = ['train.0.1684831482.jsonl.gz', 'train.1.1684831482.jsonl.gz']
    exclude_shard = []

    if train_file not in exclude_shard:
        df = pd.read_json(os.path.join(path, train_file), lines=True, compression="gzip")
        print(df.shape)
        for _, row in df.iterrows():
            reservior_sampler.add(row.tolist())

        results = reservior_sampler.values
        results = np.array(results)
        df_out = pd.DataFrame(
            {"uid": results[:, 0], "person_id": results[:, 1], "split": results[:, 2], "text": results[:, 3]}
        )

        df_out.to_json(path_to_save + f"lumia_trainsplit_sample_{i}.jsonl.gz", compression="gzip")


if __name__ == "__main__":
    k = 10000
    seed = 1234

    path = "/local-scratch/nigam/projects/jfries/crfm/datasets/pretraining/shc/markup_codes_notes_desc_dedup_x8_v1/"
    path_to_save = "/local-scratch/nigam/projects/zphuo/data/medical_instruction/"
    train_file_ls = []
    for i, train_file in enumerate(os.listdir(path)):
        if "train" in train_file:
            train_file_ls.append(train_file)
    num_threads = len(train_file_ls)

    tasks = [(path, train_file, path_to_save, i, k, seed, num_threads) for i, train_file in enumerate(train_file_ls)]

    ctx = multiprocessing.get_context("forkserver")

    with ctx.Pool(num_threads) as pool:
        parallel_result = list(pool.imap(sample_from_lumia, tasks))
