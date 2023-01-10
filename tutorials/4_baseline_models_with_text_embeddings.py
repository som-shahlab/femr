import pickle
import sklearn
import piton.datasets
from piton.featurizers import save_to_file, load_from_file
import sklearn.linear_model
import sklearn.metrics
import numpy as np
import os
import xgboost as xgb
import lightgbm as lgbm
import pandas as pd

# For replicating the data, please use these paths. This expriment was run on carina
# Please make sure the paths are correct for the server that you are working on.

# This is what ethan gave (only exists on Carina)
features_path = "/share/pi/nigam/rthapa84/data/hba1c_reps"  #CLMBR representation
labels_path = "/share/pi/nigam/rthapa84/data/HighHbA1c_labeled_patients_v3.pickle"
data_path = "/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2"

data = piton.datasets.PatientDatabase(data_path)

with open(features_path, "rb") as f:
    features = pickle.load(f)

with open(labels_path, "rb") as f:
    labeled_patients = pickle.load(f)


# CLMBR Model Representation and Performance

data_matrix, patient_ids, labeling_time = [
    features[k] for k in ("data_matrix", "patient_ids", "labeling_time")
]
labels = []

for pid, time in zip(patient_ids, labeling_time):
    raw_labels = [
        label.value for label in labeled_patients[pid] if label.time == time
    ]
    assert len(raw_labels) == 1
    labels.append(raw_labels[0])

labels = np.array(labels)

hashes = np.array([data.compute_split(97, pid) for pid in patient_ids])
train_pids = hashes < 70
val_pids = (hashes >= 70) & (hashes < 85)

if True:
    model = sklearn.linear_model.LogisticRegressionCV()
    model.fit(data_matrix[train_pids, :], labels[train_pids])

    with open("temp_model.pickle", "wb") as f:
        pickle.dump(model, f)
else:
    with open("temp_model.pickle", "rb") as f:
        model = pickle.load(f)

train_preds = model.predict_log_proba(data_matrix[train_pids, :])[:, 1]
print(sklearn.metrics.roc_auc_score(labels[train_pids], train_preds))
print(sklearn.metrics.average_precision_score(labels[train_pids], train_preds))


val_preds = model.predict_log_proba(data_matrix[val_pids, :])[:, 1]
print(sklearn.metrics.roc_auc_score(labels[val_pids], val_preds))
print(sklearn.metrics.average_precision_score(labels[val_pids], val_preds))


# Performance with only Diabetes Text Embeddings

path_to_diabetes_shards = "/share/pi/nigam/rthapa84/data/diabetes_shards"
path_to_diabetes_metadata = "/share/pi/nigam/rthapa84/data/v1_diabetes_meta_data.pickle"

file_list = [f"{i}_embeddings.pickle" for i in range(10)]

array_list = []

for file_name in file_list:
    test_embedding = load_from_file(os.path.join(path_to_diabetes_shards, file_name))
    print(test_embedding.shape)
    array_list.append(test_embedding)

embeddings = np.concatenate(array_list)
embeddings.shape

diabetes_meta_data = load_from_file(path_to_diabetes_metadata)

diabetes_meta_data

diabetes_text_features = {
    "data_matrix": embeddings, 
    "labels": diabetes_meta_data[0],
    "patient_ids": diabetes_meta_data[1], 
    "labeling_time": diabetes_meta_data[2], 
}


# Performance with just text data

data_matrix, labels, patient_ids, labeling_time = [
    diabetes_text_features[k] for k in ("data_matrix", "labels", "patient_ids", "labeling_time")
]


hashes = np.array([data.compute_split(97, pid) for pid in patient_ids])
train_pids = hashes < 70
val_pids = (hashes >= 70) & (hashes < 85)


# Logistic Regression
model = sklearn.linear_model.LogisticRegressionCV()
model.fit(data_matrix[train_pids, :], labels[train_pids])
train_preds = model.predict_log_proba(data_matrix[train_pids, :])[:, 1]
print(sklearn.metrics.roc_auc_score(labels[train_pids], train_preds))
print(sklearn.metrics.average_precision_score(labels[train_pids], train_preds))

val_preds = model.predict_log_proba(data_matrix[val_pids, :])[:, 1]
print(sklearn.metrics.roc_auc_score(labels[val_pids], val_preds))
print(sklearn.metrics.average_precision_score(labels[val_pids], val_preds))


# XGB
model = xgb.XGBClassifier(early_stopping=True)
model.fit(data_matrix[train_pids, :], labels[train_pids])
train_preds = model.predict_proba(data_matrix[train_pids, :])[::,1]
print(sklearn.metrics.roc_auc_score(labels[train_pids], train_preds))
print(sklearn.metrics.average_precision_score(labels[train_pids], train_preds))

val_preds = model.predict_proba(data_matrix[val_pids, :])[::,1]
print(sklearn.metrics.roc_auc_score(labels[val_pids], val_preds))
print(sklearn.metrics.average_precision_score(labels[val_pids], val_preds))

# LGBM
model = lgbm.LGBMClassifier()
model.fit(data_matrix[train_pids, :], labels[train_pids])
train_preds = model.predict_proba(data_matrix[train_pids, :])[::,1]
print(sklearn.metrics.roc_auc_score(labels[train_pids], train_preds))
print(sklearn.metrics.average_precision_score(labels[train_pids], train_preds))

val_preds = model.predict_proba(data_matrix[val_pids, :])[::,1]
print(sklearn.metrics.roc_auc_score(labels[val_pids], val_preds))
print(sklearn.metrics.average_precision_score(labels[val_pids], val_preds))


# Now, let us concatenate clmbr embeddigs with Diabetes text embeddings
clmbr_df = pd.DataFrame(features["data_matrix"])
clmbr_df["patient_ids"] = features["patient_ids"]
clmbr_df["labeling_time"] = features["labeling_time"]

text_df = pd.DataFrame(diabetes_text_features["data_matrix"])
text_df["patient_ids"] = diabetes_text_features["patient_ids"]
text_df["labeling_time"] = diabetes_text_features["labeling_time"]
combined_df = pd.merge(clmbr_df, text_df, on=['patient_ids','labeling_time'])

combined_data_matirx = combined_df.drop(columns=['patient_ids','labeling_time']).to_numpy()
patient_ids = combined_df["patient_ids"].to_numpy()
labeling_time = pd.to_datetime(combined_df["labeling_time"]).to_numpy()

combined_features = {
    "data_matrix": combined_data_matirx, 
    "patient_ids": patient_ids, 
    "labeling_time": labeling_time
}

with open(labels_path, "rb") as f:
    labeled_patients = pickle.load(f)
    
data_matrix, patient_ids, labeling_time = [
    combined_features[k] for k in ("data_matrix", "patient_ids", "labeling_time")
]

labels = []
for pid, time in zip(patient_ids, labeling_time):
    raw_labels = [
        label.value for label in labeled_patients[pid]
    ]
    assert len(raw_labels) == 1
    labels.append(raw_labels[0])

labels = np.array(labels)

hashes = np.array([data.compute_split(97, pid) for pid in patient_ids])
train_pids = hashes < 70
val_pids = (hashes >= 70) & (hashes < 85)

# Logistic Regression
model = sklearn.linear_model.LogisticRegressionCV()
model.fit(data_matrix[train_pids, :], labels[train_pids])

train_preds = model.predict_log_proba(data_matrix[train_pids, :])[:, 1]
print(sklearn.metrics.roc_auc_score(labels[train_pids], train_preds))
print(sklearn.metrics.average_precision_score(labels[train_pids], train_preds))

val_preds = model.predict_log_proba(data_matrix[val_pids, :])[:, 1]
print(sklearn.metrics.roc_auc_score(labels[val_pids], val_preds))
print(sklearn.metrics.average_precision_score(labels[val_pids], val_preds))

# XGB
model = xgb.XGBClassifier(early_stopping=True)
model.fit(data_matrix[train_pids, :], labels[train_pids])

train_preds = model.predict_proba(data_matrix[train_pids, :])[::,1]
print(sklearn.metrics.roc_auc_score(labels[train_pids], train_preds))
print(sklearn.metrics.average_precision_score(labels[train_pids], train_preds))

val_preds = model.predict_proba(data_matrix[val_pids, :])[::,1]
print(sklearn.metrics.roc_auc_score(labels[val_pids], val_preds))
print(sklearn.metrics.average_precision_score(labels[val_pids], val_preds))

# LGBM
model = lgbm.LGBMClassifier()
model.fit(data_matrix[train_pids, :], labels[train_pids])
train_preds = model.predict_proba(data_matrix[train_pids, :])[::,1]
print(sklearn.metrics.roc_auc_score(labels[train_pids], train_preds))
print(sklearn.metrics.average_precision_score(labels[train_pids], train_preds))

val_preds = model.predict_proba(data_matrix[val_pids, :])[::,1]
print(sklearn.metrics.roc_auc_score(labels[val_pids], val_preds))
print(sklearn.metrics.average_precision_score(labels[val_pids], val_preds))


# Performance with only Mortality Text Embeddings

path_to_mortality_shards = "/share/pi/nigam/rthapa84/data/mortality_shards"
path_to_mortality_metadata = "/share/pi/nigam/rthapa84/data/v1_mortality_meta_data.pickle"
labels_path = "/share/pi/nigam/rthapa84/data/mortality_labeled_patients_v1.pickle"

file_list = [f"{i}_embeddings.pickle" for i in range(10)]

array_list = []

for file_name in file_list:
    test_embedding = load_from_file(os.path.join(path_to_mortality_shards, file_name))
    print(test_embedding.shape)
    array_list.append(test_embedding)

embeddings = np.concatenate(array_list)
mortality_meta_data = load_from_file(path_to_mortality_metadata)

mortality_text_features = {
    "data_matrix": embeddings, 
    "labels": mortality_meta_data[0],
    "patient_ids": mortality_meta_data[1], 
    "labeling_time": mortality_meta_data[2], 
}

data_matrix, labels, patient_ids, labeling_time = [
    mortality_text_features[k] for k in ("data_matrix", "labels", "patient_ids", "labeling_time")
]

hashes = np.array([data.compute_split(97, pid) for pid in patient_ids])
train_pids = hashes < 70
val_pids = (hashes >= 70) & (hashes < 85)


# Logistic Regression
model = sklearn.linear_model.LogisticRegressionCV()
model.fit(data_matrix[train_pids, :], labels[train_pids])

train_preds = model.predict_log_proba(data_matrix[train_pids, :])[:, 1]
print(sklearn.metrics.roc_auc_score(labels[train_pids], train_preds))
print(sklearn.metrics.average_precision_score(labels[train_pids], train_preds))

val_preds = model.predict_log_proba(data_matrix[val_pids, :])[:, 1]
print(sklearn.metrics.roc_auc_score(labels[val_pids], val_preds))
print(sklearn.metrics.average_precision_score(labels[val_pids], val_preds))

# XGB
model = xgb.XGBClassifier(early_stopping=True)
model.fit(data_matrix[train_pids, :], labels[train_pids])
train_preds = model.predict_proba(data_matrix[train_pids, :])[::,1]
print(sklearn.metrics.roc_auc_score(labels[train_pids], train_preds))
print(sklearn.metrics.average_precision_score(labels[train_pids], train_preds))

val_preds = model.predict_proba(data_matrix[val_pids, :])[::,1]
print(sklearn.metrics.roc_auc_score(labels[val_pids], val_preds))
print(sklearn.metrics.average_precision_score(labels[val_pids], val_preds))

# LGBM
model = lgbm.LGBMClassifier()
model.fit(data_matrix[train_pids, :], labels[train_pids])
train_preds = model.predict_proba(data_matrix[train_pids, :])[::,1]
print(sklearn.metrics.roc_auc_score(labels[train_pids], train_preds))
print(sklearn.metrics.average_precision_score(labels[train_pids], train_preds))

val_preds = model.predict_proba(data_matrix[val_pids, :])[::,1]
print(sklearn.metrics.roc_auc_score(labels[val_pids], val_preds))
print(sklearn.metrics.average_precision_score(labels[val_pids], val_preds))


# Let us combined CLMBR representation with Mortality text data

features_path = "/share/pi/nigam/rthapa84/data/mortality_clmbr_tuned"
labels_path = "/share/pi/nigam/rthapa84/data/mortality_labeled_patients_v1.pickle"

with open(features_path, "rb") as f:
    features = pickle.load(f)
    
with open(labels_path, "rb") as f:
    labeled_patients = pickle.load(f)

clmbr_df = pd.DataFrame(features["data_matrix"])
clmbr_df["patient_ids"] = features["patient_ids"]
clmbr_df["labeling_time"] = features["labeling_time"]

text_df = pd.DataFrame(mortality_text_features["data_matrix"])
text_df["patient_ids"] = mortality_text_features["patient_ids"]
text_df["labeling_time"] = mortality_text_features["labeling_time"]
combined_df = pd.merge(clmbr_df, text_df, on=['patient_ids','labeling_time'])
combined_data_matirx = combined_df.drop(columns=['patient_ids','labeling_time']).to_numpy()
patient_ids = combined_df["patient_ids"].to_numpy()
labeling_time = pd.to_datetime(combined_df["labeling_time"]).to_numpy()

combined_features = {
    "data_matrix": combined_data_matirx, 
    "patient_ids": patient_ids, 
    "labeling_time": labeling_time
}

with open(labels_path, "rb") as f:
    labeled_patients = pickle.load(f)
    
data_matrix, patient_ids, labeling_time = [
    combined_features[k] for k in ("data_matrix", "patient_ids", "labeling_time")
]

labels = []
for pid, time in zip(patient_ids, labeling_time):
    raw_labels = [
        label.value for label in labeled_patients[pid]
    ]
    assert len(raw_labels) == 1
    labels.append(raw_labels[0])

labels = np.array(labels)

hashes = np.array([data.compute_split(97, pid) for pid in patient_ids])
train_pids = hashes < 70
val_pids = (hashes >= 70) & (hashes < 85)

# Logistic Regression
model = sklearn.linear_model.LogisticRegressionCV()
model.fit(data_matrix[train_pids, :], labels[train_pids])

train_preds = model.predict_log_proba(data_matrix[train_pids, :])[:, 1]
print(sklearn.metrics.roc_auc_score(labels[train_pids], train_preds))
print(sklearn.metrics.average_precision_score(labels[train_pids], train_preds))

val_preds = model.predict_log_proba(data_matrix[val_pids, :])[:, 1]
print(sklearn.metrics.roc_auc_score(labels[val_pids], val_preds))
print(sklearn.metrics.average_precision_score(labels[val_pids], val_preds))

# XGB
model = xgb.XGBClassifier()
model.fit(data_matrix[train_pids, :], labels[train_pids])

train_preds = model.predict_proba(data_matrix[train_pids, :])[::,1]
print(sklearn.metrics.roc_auc_score(labels[train_pids], train_preds))
print(sklearn.metrics.average_precision_score(labels[train_pids], train_preds))

val_preds = model.predict_proba(data_matrix[val_pids, :])[::,1]
print(sklearn.metrics.roc_auc_score(labels[val_pids], val_preds))
print(sklearn.metrics.average_precision_score(labels[val_pids], val_preds))

# LGBM
model = lgbm.LGBMClassifier()
model.fit(data_matrix[train_pids, :], labels[train_pids])
train_preds = model.predict_proba(data_matrix[train_pids, :])[::,1]
print(sklearn.metrics.roc_auc_score(labels[train_pids], train_preds))
print(sklearn.metrics.average_precision_score(labels[train_pids], train_preds))

val_preds = model.predict_proba(data_matrix[val_pids, :])[::,1]
print(sklearn.metrics.roc_auc_score(labels[val_pids], val_preds))
print(sklearn.metrics.average_precision_score(labels[val_pids], val_preds))


