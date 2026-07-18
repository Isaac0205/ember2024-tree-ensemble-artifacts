"""
This script is used for evaluating a LightGBM EMBER2024 family classification model (to reproduce Section 4.5 in the original paper)
"""

import argparse
import thrember
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if __name__ == "__main__":
    
    # Used to set up the script so that it can take inputs from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to the directory containing the EMBER2024 dataset")
    parser.add_argument("model_path", type=str, help="Path to the trained family model")
    args = parser.parse_args()

    # Load the model that was serialised after training
    model = lgb.Booster(model_file=args.model_path)

    # Load test set features and family labels from the dataset dir.
    X_test, y_test = thrember.read_vectorized_features(args.data_dir, "test")

    # Remove samples without a family label 
    valid_mask = y_test != -1
    X_test = X_test[valid_mask]
    y_test = y_test[valid_mask]

    # LigthGBM outputs probabilities for each family class.
    y_pred_prob = model.predict(X_test)
    # Convert the probabilities to family predictions by selecting the class with the highest probability for each sample.
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Compute classification metrics matching those reported in original paper.
    accuracy = accuracy_score(y_test, y_pred)
    macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Print results 
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Total test samples used: {len(y_test)}")
    print(f"Number of families: {len(np.unique(y_test))}")