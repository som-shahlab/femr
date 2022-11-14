import datetime
import os
from typing import List, Tuple
import pickle

import numpy as np
from sklearn import metrics

import piton

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgbm


with open("/share/pi/nigam/rthapa84/data/test_diabetes_matrix.pickle", "rb") as f:
    featurized_data = pickle.load(f)
    

feature_matrix = featurized_data[0].toarray()

