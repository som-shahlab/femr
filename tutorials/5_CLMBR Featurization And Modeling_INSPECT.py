#!/usr/bin/env python
# coding: utf-8

# # Using CLMBR to generate features and training models on those features
# 
# We can use a trained CLMBR model to generate features and then use those features in a logistic regression model.

import warnings

# Suppress specific FutureWarning from awswrangler module
warnings.filterwarnings("ignore", message="promote has been superseded by mode='default'.", category=FutureWarning, module="pyarrow")
warnings.filterwarnings("ignore", message="promote has been superseded by mode='default'.", category=FutureWarning, module="datasets")
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="The max_iter was reached which means the coef_ did not converge")

import shutil
import os

os.environ["HF_DATASETS_CACHE"] = '/share/pi/nigam/projects/zphuo/.cache'

TARGET_DIR = 'trash/tutorial_5_INSPECT'

if os.path.exists(TARGET_DIR):
    shutil.rmtree(TARGET_DIR)

os.mkdir(TARGET_DIR)

num_proc = 20


import femr.models.transformer
import pyarrow.csv
import datasets
import pickle
import pyarrow as pa
import pyarrow.compute as pc

# First, we compute our features

label_columns = '12_month_PH'

# Load some labels
# labels = pyarrow.csv.read_csv('input/labels.csv').to_pylist()
label_csv_subset = '/share/pi/nigam/projects/zphuo/data/PE/inspect/timelines_smallfiles_meds/cohort_0.2.0_master_file_anon_subset.csv'
labels_table = pyarrow.csv.read_csv(label_csv_subset)


import femr.models.transformer
import pyarrow.csv
import datasets

# First, we compute our features

label_columns = '12_month_PH'

# Load some labels
print('loading labels...')
# labels = pyarrow.csv.read_csv('input/labels.csv').to_pylist()
label_csv_subset = '/share/pi/nigam/projects/zphuo/data/PE/inspect/timelines_smallfiles_meds/cohort_0.2.0_master_file_anon_subset.csv'
labels_table = pyarrow.csv.read_csv(label_csv_subset)

import pandas as pd
label_df = pd.read_csv(label_csv_subset)
label_df = label_df[['patient_id', 'split', ]]
label_df.rename(columns={'split': 'split_name'}, inplace=True)
inspect_split_csv = '/share/pi/nigam/projects/zphuo/repos/femr/tutorials/trash/tutorial_6_INSEPCT/motor_model/main_split.csv'
label_df.to_csv(inspect_split_csv, index=False)

# filter out censored
print('filtering out censored...')
selected_table = labels_table.select(['patient_id', 'procedure_time', label_columns])
filtered_table = selected_table.filter(pa.compute.field(label_columns) != "Censored")

# cast to bool
print('casting to bool...')
casted_column = pc.cast(filtered_table.column(label_columns), target_type=pa.bool_())
filtered_table = filtered_table.set_column(filtered_table.schema.get_field_index(label_columns), pa.field(label_columns, pa.bool_()), casted_column)

print('change column names...')
columns = {name: filtered_table.column(name) for name in filtered_table.column_names}
columns['prediction_time'] = columns.pop('procedure_time')
columns['boolean_value'] = columns.pop(label_columns)
filtered_table = pa.Table.from_arrays(list(columns.values()), names=list(columns.keys()))

labels = filtered_table.to_pylist()

# Load our data
# dataset = datasets.Dataset.from_parquet("input/meds/data/*")
print('loading data...')
parquet_folder = '/share/pi/nigam/projects/zphuo/data/PE/inspect/timelines_smallfiles_meds/data_subset/*'
dataset = datasets.Dataset.from_parquet(parquet_folder)

model_name = "StanfordShahLab/clmbr-t-base"

print('computing features...')
features = femr.models.transformer.compute_features(dataset, model_name, labels, num_proc=num_proc, tokens_per_batch=128)

# We have our features
print('feature shapes:')
for k, v in features.items():
    print(k, v.shape)


# # Joining features and labels
# 
# Given a feature set, it's important to be able to join a set of labels to those features.
# 
# This can be done with femr.featurizers.join_labels

import femr.featurizers
print('joining features and labels...')
import pdb; pdb.set_trace()
features_and_labels = femr.featurizers.join_labels(features, labels)

for k, v in features_and_labels.items():
    print(k, v.shape)


# # Data Splitting
# 
# When using a pretrained CLMBR model, we have to be very careful to use the splits used for the original model

import femr.splits
import numpy as np
print('loading splits...')
# We split into a global training and test set
split = femr.splits.PatientSplit.load_from_csv(inspect_split_csv)

train_mask = np.isin(features_and_labels['patient_ids'], split.train_patient_ids)
test_mask = np.isin(features_and_labels['patient_ids'], split.test_patient_ids)

percent_train = .70
X_train, y_train = (
    features_and_labels['features'][train_mask],
    features_and_labels['boolean_values'][train_mask],
)
X_test, y_test = (
    features_and_labels['features'][test_mask],
    features_and_labels['boolean_values'][test_mask],
)


# # Building Models
# 
# The generated features can then be used to build your standard models. In this case we construct both logistic regression and XGBoost models and evaluate them.
# 
# Performance is perfect since our task (predicting gender) is 100% determined by the features

import xgboost as xgb
import sklearn.linear_model
import sklearn.metrics
import sklearn.preprocessing

def run_analysis(title: str, y_train, y_train_proba, y_test, y_test_proba):
    print(f"---- {title} ----")
    print("Train:")
    print_metrics(y_train, y_train_proba)
    print("Test:")
    print_metrics(y_test, y_test_proba)

def print_metrics(y_true, y_proba):
    y_pred = y_proba > 0.5
    auroc = sklearn.metrics.roc_auc_score(y_true, y_proba)
    aps = sklearn.metrics.average_precision_score(y_true, y_proba)
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    f1 = sklearn.metrics.f1_score(y_true, y_pred)
    print("\tAUROC:", auroc)
    print("\tAPS:", aps)
    print("\tAccuracy:", accuracy)
    print("\tF1 Score:", f1)

print('training models...')
model = sklearn.linear_model.LogisticRegressionCV(penalty="l2", solver="liblinear").fit(X_train, y_train)
y_train_proba = model.predict_proba(X_train)[::, 1]
y_test_proba = model.predict_proba(X_test)[::, 1]
run_analysis("Logistic Regression", y_train, y_train_proba, y_test, y_test_proba)

