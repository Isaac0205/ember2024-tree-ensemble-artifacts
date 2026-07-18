"""
This script is used to evaluate an XGBoost model's ability to detect newly emerging malware families (to reproduce Section 4.4 in the original paper)
"""

import os
import struct
import argparse
import pickle
import numpy as np
from collections import Counter
from sklearn.metrics import auc, precision_recall_curve
import thrember


if __name__ == "__main__":

    # Used to set up the script so that it can take inputs from the command line
    parser = argparse.ArgumentParser(description="Evaluate XGBoost model on strictly emerging families.")
    parser.add_argument("--data-dir", type=str, help="Path to the data partition.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained XGBoost model.")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    MODEL_PATH = args.model_path

    # The function below was used to read only the first 8 bytes of the .dat file to extract the row count
    def get_dat_nrows(path: str) -> int:
        with open(path, "rb") as f:
            n_rows, n_cols = struct.unpack("ii", f.read(8))
            return n_rows

    # Loading the test set features and labels from the dataset dir.
    X_test, y_malicious = thrember.read_vectorized_features(DATA_DIR, subset="test")

    # Loading family labels for the test set using memory mapping to keep memory usage low
    family_y_test_path = os.path.join(DATA_DIR, "family class min=1", "y_test.dat")
    n_test = len(y_malicious)
    y_family_test_full = np.fromfile(family_y_test_path, dtype=np.int32, count=n_test)

    # Loading family labels for the training set
    family_y_train_path = os.path.join(DATA_DIR, "family class min=1", "y_train.dat")
    n_train = get_dat_nrows(os.path.join(DATA_DIR, "X_train.dat")) # reads only 8 bytes
    y_family_train = np.fromfile(family_y_train_path, dtype=np.int32, count=n_train)

    # Count family occurrences in train and test whilst excluding all unlabelled samples
    train_counts = Counter(y_family_train[y_family_train != -1])
    test_counts  = Counter(y_family_test_full[y_family_test_full != -1])

    # Identify strictly emerging families which are families that have at least 10 samples in test set and zero samples in the training set
    emerging_families = [
        fam for fam in test_counts
        if test_counts[fam] >= 10 and train_counts.get(fam, 0) == 0
    ]

    # Print emerging families
    print(f"Found {len(emerging_families)} strictly emerging families")
    print("Family IDs:", emerging_families)
    print("Samples per family:", [test_counts[f] for f in emerging_families])

    # Since emerging_families contains only malicious files, we mix in benign samples for a realistic class distribution to build evaluation set.
    # Extract benign rows
    benign_idx   = np.where(y_malicious == 0)[0]
    # Extract malware rows
    emerging_idx = np.where(np.isin(y_family_test_full, emerging_families))[0]

    # Create evaluation matrix
    X_eval = np.vstack([X_test[benign_idx], X_test[emerging_idx]])
    y_eval = np.zeros(len(X_eval), dtype=int)
    y_eval[len(benign_idx):] = 1

    print(f"Eval set: {len(benign_idx):,} benign + {len(emerging_idx):,} emerging malicious")

    # Load the model that was serialised after training
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Evaluate model. XGBoost's predict_proba() outputs probabilities for each class. Column 0 is the probability of being benign while column 1 is the probability of being malicious
    y_pred = model.predict_proba(X_eval)[:, 1]

    # Calculate PR AUC 
    precision, recall, _ = precision_recall_curve(y_eval, y_pred)
    pr_auc = auc(recall, precision)

    print(f"\nPR AUC on strictly emerging families = {pr_auc:.4f}")