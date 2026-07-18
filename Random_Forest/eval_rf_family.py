"""
This script is used for evaluating a Random Forest EMBER2024 family classification model (to reproduce Section 4.5 in the original paper)
"""

import argparse
import pickle
import thrember
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


if __name__ == "__main__":

    # Used to set up the script so that it can take inputs from the command line
    parser = argparse.ArgumentParser(description="Evaluate Random Forest family classifier on EMBER2024")
    parser.add_argument("data_dir", type=str, help="Path to the directory containing the EMBER2024 dataset")
    parser.add_argument("model_path", type=str, help="Path to the trained Random Forest family model.")
    args = parser.parse_args()

    # Load the model that was serialised after training
    with open(args.model_path, "rb") as f:
        model = pickle.load(f)

    # Load test set features and family labels from the dataset dir.
    X_test, y_test = thrember.read_vectorized_features(args.data_dir, "test")

    # Remove samples without a family label 
    valid_mask = y_test != -1
    X_test = X_test[valid_mask]
    y_test = y_test[valid_mask]

    # Random Forest's predict() calls predict_proba() and takes argmax.
    # To avoid memory issues with large probability matrix, we process in batches (10,000 samples at a time)
    BATCH_SIZE = 10_000
    # Pre-allocate the array which will store the final predictions
    y_pred = np.empty(len(X_test), dtype=np.int32)
    
    # Process the test set in chunks
    for start in range(0, len(X_test), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(X_test))
        y_pred[start:end] = model.predict(X_test[start:end])


    # Compute classification metrics matching those reported in original paper.
    accuracy = accuracy_score(y_test, y_pred)
    macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Print results 
    print(f"Accuracy: {accuracy:.4f}   ")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Total test samples used: {len(y_test)}")
    print(f"Number of families: {len(np.unique(y_test))}")
