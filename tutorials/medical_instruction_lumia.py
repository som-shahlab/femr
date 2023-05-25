#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd 
import random
import os
import pickle
from tqdm import tqdm


# In[3]:


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


# In[4]:


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


# In[5]:


path = '/local-scratch/nigam/projects/jfries/crfm/datasets/pretraining/shc/markup_codes_notes_desc_dedup_x8_v1'
exlude_shard = ['train.0.1684831482.jsonl.gz', 'train.1.1684831482.jsonl.gz']

all_gz_ls = os.listdir(path)
all_train_ls = []
for gz_file in all_gz_ls:
    if 'train' in gz_file:
        if gz_file not in exlude_shard:
            all_train_ls.append(gz_file)

k = 10000
seed = 1234
reservior_sampler = ReservoirSampler(k, seed)

for train_file in all_train_ls:
    print(train_file)
    df = pd.read_json(os.path.join(path, train_file), lines=True, compression='gzip')

    for text in tqdm(df['text']):
        reservior_sampler.add(text)

results = reservior_sampler.values


path_to_save = '/local-scratch/nigam/projects/zphuo/data/medical_instruction/'
with open(path_to_save + 'lumia_pretraining_data.pkl', 'wb') as f:
    pickle.dump(results, f)
    


# In[ ]:




