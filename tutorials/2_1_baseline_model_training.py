#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import datetime
import os
from typing import List, Tuple
import pickle

import numpy as np
from sklearn import metrics

import piton
import piton.datasets

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgbm


# In[3]:



# Please update this path with your extract of piton as noted in previous notebook. 
PATH_TO_PITON_DB = '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2'
PATH_TO_SAVE_MATRIX = "/share/pi/nigam/rthapa84/data"
LABELED_PATIENTS = "HighHbA1c_labeled_patients.pickle"
FEATURIZED_DATA = "test_HighHbA1cLF_featurized_patients.pickle"
SEED = 97

#PATH_TO_PITON_DB = '/local-scratch/nigam/projects/clmbr_text_assets/data/piton_database_1_perct/'
#PATH_TO_SAVE_MATRIX = "./"

# Patient database
database = piton.datasets.PatientDatabase(PATH_TO_PITON_DB)

with open(os.path.join(PATH_TO_SAVE_MATRIX, FEATURIZED_DATA), 'rb') as f:
    featurized_data = pickle.load(f)

print("Data loaded")


# In[4]:


len(featurized_data)


# In[5]:

feature_matrix, labels, patient_ids = featurized_data[0], featurized_data[1], featurized_data[2]

new_patient_ids = np.array([database.compute_split(SEED, pid) for pid in patient_ids])

X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels, train_size = 0.8)

# train_bool_mask = (new_patient_ids < 70)
# validation_patient_ids = (new_patient_ids >= 70 and new_patient_ids < 85)
# test_bool_mask = (new_patient_ids >= 85)

# X_train = feature_matrix[train_bool_mask]
# y_train = labels[train_bool_mask]

# print("Finished Training data split")

# X_test = feature_matrix[test_bool_mask]
# y_test = labels[test_bool_mask]

print("Finished Data split")


# Logistic Regresion
# model = LogisticRegression().fit(X_train, y_train)
# y_pred_proba = model.predict_proba(X_test)[::,1]
# auc = metrics.roc_auc_score(y_test, y_pred_proba)
# print(auc)

model = xgb.XGBClassifier()
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[::,1]
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print(auc)



# In[9]:


# len(patient_ids)

# print("Here")
# X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels, train_size = 0.8)

# print("Data Splitted")

# # Logistic Regresion
# model = LogisticRegression().fit(X_train, y_train)
# y_pred_proba = model.predict_proba(X_test)[::,1]
# auc = metrics.roc_auc_score(y_test, y_pred_proba)
# print(auc)

# In[10]:


# patient_ids_unique = list(set(patient_ids))
# len(patient_ids_unique)


# In[11]:


# train_patient_ids, test_patient_ids = train_test_split(patient_ids_unique, train_size = 0.8)
# train_bool_mask = np.in1d(patient_ids, train_patient_ids)
# test_bool_mask = np.in1d(patient_ids, test_patient_ids)
# X_train = feature_matrix[train_bool_mask]
# y_train = labels[train_bool_mask]
# X_test = feature_matrix[test_bool_mask]
# y_test = labels[test_bool_mask]


# print("Prevalence on total dataset:", sum(labels)/len(labels))
# print("Prevalence on training dataset:", sum(y_train)/len(y_train))
# print("Prevalence on testing dataset:", sum(y_test)/len(y_test))


# # In[ ]:


# # Logistic Regresion
# model = LogisticRegression().fit(X_train, y_train)
# y_pred_proba = model.predict_proba(X_test)[::,1]
# auc = metrics.roc_auc_score(y_test, y_pred_proba)
# print(auc)


# In[ ]:


# XGBoost
# params = {
#     "n_estimators": 50, 
#     "max_depth": 2
# }

# model = xgb.XGBClassifier()
# model.fit(X_train, y_train)
# y_pred_proba = model.predict_proba(X_test)[::,1]
# auc = metrics.roc_auc_score(y_test, y_pred_proba)
# print(auc)


# In[ ]:


# LightGBM

