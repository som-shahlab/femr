#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd 
import random
import os
import pickle
from tqdm import tqdm
import multiprocessing


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

    path, train_file, path_to_save, i = args
    print(train_file)

    exclude_shard = ['train.0.1684831482.jsonl.gz', 'train.1.1684831482.jsonl.gz']

    if train_file not in exclude_shard:
        
        df = pd.read_json(os.path.join(path, train_file), lines=True, compression='gzip')
        print(df.shape)
        for i, text in enumerate(df['text']):
            print(i, end=' ')
            reservior_sampler.add(text)

        results = reservior_sampler.values
        
        with open(path_to_save + f'lumia_pretraining_data_{i}.pkl', 'wb') as f:
            pickle.dump(results, f)

if __name__ == '__main__':
    k = 10000
    seed = 1234
    reservior_sampler = ReservoirSampler(k, seed)

    path = '/share/pi/nigam/projects/zphuo/data/lumia/download_frazier/'
    path_to_save = '/share/pi/nigam/projects/zphuo/data/medical_instruction/'

    tasks = [(path, train_file, path_to_save, i) for i, train_file in enumerate(os.listdir(path)) if 'train' in train_file]

    ctx = multiprocessing.get_context("forkserver")

    num_threads = len([train_file for train_file in os.listdir(path) if 'train' in train_file])
    with ctx.Pool(num_threads) as pool:
        parallel_result = list(pool.imap(sample_from_lumia, tasks))