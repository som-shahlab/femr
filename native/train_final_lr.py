import pickle

import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.metrics

import piton.datasets

features_path = "hba1c_reps"
labels_path = "/local-scratch/nigam/projects/ethanid/gpu_experiments/mortality_labeled_patients_v1.pickle"
labels_path = "/local-scratch/nigam/projects/ethanid/gpu_experiments/HighHbA1c_labeled_patients_v3.pickle"
data_path = "/local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2"

data = piton.datasets.PatientDatabase(data_path)


with open(features_path, "rb") as f:
    features = pickle.load(f)

with open(labels_path, "rb") as f:
    labeled_patients = pickle.load(f)

data_matrix, patient_ids, labeling_time = [features[k] for k in ("data_matrix", "patient_ids", "labeling_time")]
labels = []

for pid, time in zip(patient_ids, labeling_time):
    raw_labels = [label.value for label in labeled_patients[pid] if label.time == time]
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
