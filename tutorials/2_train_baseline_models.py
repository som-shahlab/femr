import argparse
import os
import pickle
import time
from typing import List, Tuple

import numpy as np
import xgboost as xgb
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import auc, precision_recall_curve
from sklearn.preprocessing import MaxAbsScaler

import piton
import piton.datasets

"""
Example running:

Note: Please make sure to first install xgboost. `pip install xgboost`

python3 tutorials/2_train_baseline_models.py \
    /local-scratch/nigam/projects/mwornow/data/1_perct_extract_01_11_23 \
    /local-scratch/nigam/projects/clmbr_text_assets/data/features/lupus/featurized_patients.pkl \
    --percent_train 0.8 \
    --split_seed 0 \
    --num_threads 20
"""


def load_from_pkl(path_to_file: str):
    """Load object from pkl file."""
    with open(path_to_file, "rb") as fd:
        result = pickle.load(fd)
    return result


if __name__ == "__main__":
    START_TIME = time.time()

    def print_log(name: str, content: str):
        print(f"{int(time.time() - START_TIME)} | {name} | {content}")

    parser = argparse.ArgumentParser(description="Train baseline models (LR, XGBoost, etc.)")

    parser.add_argument(
        "path_to_patient_database",
        type=str,
        help="Path of folder to the Piton PatientDatabase. Example: '/local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract_v5'",
    )

    parser.add_argument(
        "path_to_featurized_patients",
        type=str,
        help="Path to the file containing features for patients. Example: '/local-scratch/nigam/projects/rthapa84/data/mortality_featurized_patients_test.pkl'",
    )

    parser.add_argument(
        "--num_threads",
        type=int,
        help="The number of threads to use",
        default=1,
    )

    parser.add_argument(
        "--percent_train",
        type=float,
        help="Percentage of total dataset to use for training",
        default=0.7,
    )

    parser.add_argument(
        "--split_seed",
        type=int,
        help="Seed to use to split data into train/test splits -- used by PatientDatabase.compute_split(seed)",
        default=0,
    )

    # Parse CLI args
    args = parser.parse_args()
    PATH_TO_PATIENT_DATABASE: str = args.path_to_patient_database
    PATH_TO_FEATURIZED_PATIENTS: str = args.path_to_featurized_patients
    num_threads: int = int(args.num_threads)
    split_seed: int = int(args.split_seed)
    percent_train: float = float(args.percent_train)
    assert 0 < percent_train < 1, f"percent_train must be between 0 and 1, not {percent_train}"

    # Load PatientDatabase
    database = piton.datasets.PatientDatabase(PATH_TO_PATIENT_DATABASE)
    print_log("PatientDatabase", "Loaded from: " + PATH_TO_PATIENT_DATABASE)

    # Load featurized patients
    featurized_data = load_from_pkl(PATH_TO_FEATURIZED_PATIENTS)
    feature_matrix, patient_ids, label_values, label_times = (
        featurized_data[0],
        featurized_data[1],
        featurized_data[2],
        featurized_data[3],
    )
    print_log("Featurized Patients", f"Loaded from: {PATH_TO_FEATURIZED_PATIENTS}")
    print_log("Featurized Patients", f"Feature matrix shape: {feature_matrix.shape}")
    print_log("Featurized Patients", f"Patient IDs shape: {len(patient_ids)}")
    print_log("Featurized Patients", f"Label values shape: {label_values.shape}")
    print_log("Featurized Patients", f"Label times shape: {label_times.shape}")

    # Ignore all censored data
    label_values = label_values.astype(np.float32)
    feature_matrix = feature_matrix
    patient_ids = patient_ids
    label_times = label_times
    label_values = label_values

    # Train/test splits
    print_log(
        "Dataset Split",
        f"Splitting dataset ({round(percent_train, 3)} / {round(1 - percent_train, 3)}) (train / test), with seed {split_seed}",
    )
    hashed_pids = np.array([database.compute_split(split_seed, pid) for pid in patient_ids])
    train_pids_idx = np.where(hashed_pids < (percent_train * 100))[0]
    test_pids_idx = np.where(hashed_pids >= (percent_train * 100))[0]
    X_train, y_train = (
        feature_matrix[train_pids_idx],
        label_values[train_pids_idx],
    )
    X_test, y_test = feature_matrix[test_pids_idx], label_values[test_pids_idx]
    print_log(
        "Dataset Split",
        f"Train shape: X = {X_train.shape}, Y = {y_train.shape}",
    )
    print_log("Dataset Split", f"Test shape: X = {X_test.shape}, Y = {y_test.shape}")
    print_log(
        "Dataset Split",
        f"Prevalence: Total = {round(float(np.mean(label_values)), 3)}, Train = {round(float(np.mean(y_train)), 3)}, Test = {round(float(np.mean(y_test)), 3)}",
    )
    print_log(
        "Dataset Split",
        f"# of Positives: Total = {int(np.sum(label_values))}, Train = {int(np.sum(y_train))}, Test = {int(np.sum(y_test))}",
    )

    def run_analysis(title: str, y_train, y_train_proba, y_test, y_test_proba):
        print(f"---- {title} ----")
        print("Train:")
        print_metrics(y_train, y_train_proba)
        print("Test:")
        print_metrics(y_test, y_test_proba)

    def print_metrics(y_true, y_proba):
        y_pred = y_proba > 0.5
        auroc = metrics.roc_auc_score(y_true, y_proba)
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        auprc = auc(recall, precision)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred)
        print("\tAUROC:", auroc)
        print("\tAUPRC:", auprc)
        print("\tAccuracy:", accuracy)
        print("\tF1 Score:", f1)

    # Logistic Regresion
    print_log("Logistic Regression", "Training")
    scaler = MaxAbsScaler().fit(
        X_train
    )  # best for sparse data: see https://scikit-learn.org/stable/modules/preprocessing.html#scaling-sparse-data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegressionCV(n_jobs=num_threads, penalty="l2", solver="liblinear").fit(X_train_scaled, y_train)
    y_train_proba = model.predict_proba(X_train_scaled)[::, 1]
    y_test_proba = model.predict_proba(X_test_scaled)[::, 1]
    run_analysis("Logistic Regression", y_train, y_train_proba, y_test, y_test_proba)
    print_log("Logistic Regression", "Done")

    # XGBoost
    print_log("XGBoost", "Training")
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    y_train_proba = model.predict_proba(X_train)[::, 1]
    y_test_proba = model.predict_proba(X_test)[::, 1]
    run_analysis("XGBoost", y_train, y_train_proba, y_test, y_test_proba)
    print_log("XGBoost", "Done")
