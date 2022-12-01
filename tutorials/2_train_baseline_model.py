import datetime
import os
from typing import List, Tuple
import pickle

import numpy as np
from sklearn import metrics

import piton
import piton.datasets
from piton.featurizers import save_to_file, load_from_file

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve, auc

from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgbm


# Please update this path with your extract of piton as noted in previous notebook. 
PATH_TO_PITON_DB = '/local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2'
PATH_TO_SAVE_MATRIX = "/local-scratch/nigam/projects/rthapa84/data"
LABELED_PATIENTS = "mortality_labeled_patients_test.pickle"
FEATURIZED_DATA = "mortality_featurized_patients_test.pickle"
SEED = 97

database = piton.datasets.PatientDatabase(PATH_TO_PITON_DB)

featurized_data = load_from_file(os.path.join(PATH_TO_SAVE_MATRIX, FEATURIZED_DATA))
print("Data loaded")

feature_matrix, labels, patient_ids = featurized_data[0], featurized_data[1], featurized_data[2]

hashed_pids = np.array([database.compute_split(SEED, pid) for pid in patient_ids])
train_pids_idx = np.where((hashed_pids < 70))[0]
dev_pids_idx = np.where(((hashed_pids >= 70) & (hashed_pids < 85)))[0]
X_train = feature_matrix[train_pids_idx]
y_train = labels[train_pids_idx]
X_test = feature_matrix[dev_pids_idx]
y_test = labels[dev_pids_idx]

print("Training Size:", X_train.shape)
print("Testing Size:", X_test.shape)

print("Prevalence on total dataset:", round(sum(labels)/len(labels), 4))
print("Prevalence on training dataset:", round(sum(y_train)/len(y_train), 4))
print("Prevalence on testing dataset:", round(sum(y_test)/len(y_test), 4))

# Logistic Regresion
model = LogisticRegressionCV().fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[::,1]
auroc = metrics.roc_auc_score(y_test, y_pred_proba)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
auc_precision_recall = auc(recall, precision)
print("LR (auroc, auprc): ", auroc, auc_precision_recall)


model = xgb.XGBClassifier()
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[::,1]
auroc = metrics.roc_auc_score(y_test, y_pred_proba)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
auc_precision_recall = auc(recall, precision)
print("XGB (auroc, auprc): ", auroc, auc_precision_recall)


model = lgbm.LGBMClassifier()
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[::,1]
auroc = metrics.roc_auc_score(y_test, y_pred_proba)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
auc_precision_recall = auc(recall, precision)
print("LGBM (auroc, auprc): ", auroc, auc_precision_recall)
