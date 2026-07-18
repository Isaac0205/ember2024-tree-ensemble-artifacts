"""
This script is used to evaluate the trained Random Forest malicious/benign classifier on unseen data.
"""

import pickle
import argparse
import thrember
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve


if __name__ == "__main__":

    # Used to set up the script so that it can take inputs from the command line
    parser = argparse.ArgumentParser(
        description="Evaluate Random Forest binary classifier on EMBER2024"
    )
    parser.add_argument("data_dir", type=str, help="Path to the directory containing the EMBER2024 dataset.")
    parser.add_argument("model_path", type=str, help="Path to the trained Random Forest model.")
    args = parser.parse_args()

    # Load the model that was serialised after training
    with open(args.model_path, "rb") as f:
        model = pickle.load(f)

    # Load the test set features and labels from the dataset dir.
    X_test, y_test = thrember.read_vectorized_features(args.data_dir, "test")
    
    # Evaluate on the test set. Random Forest's predict_proba() outputs probabilities for each class 
    # For binary classification, column 0 is the probability of being benign while column 1 is the probability of being malicious
    y_pred = model.predict_proba(X_test)[:, 1]

    # Calculate area under the ROC curve and PR curve for test set
    roc_auc = roc_auc_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)
    print(f"ROC AUC on test set: {roc_auc}")
    print(f"PR AUC on test set: {pr_auc}")

    # Plot and save the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    plt.title("ROC Curve for EMBER2024 Random Forest Model")
    plt.plot(fpr, tpr, color='black')
    plt.xlim(0.00005, 1.0)
    plt.ylim(0.65, 1.02)
    plt.xscale("log")  # Log scale used to see low FPR region better

    # Used to highlight the TPR when FPR is at 1%
    fpr_target = 0.01 
    index = np.argmin(np.abs(fpr - fpr_target))
    tpr_at_fpr_01 = tpr[index]
    plt.plot(
        [fpr_target, fpr_target, 0],
        [0, tpr_at_fpr_01, tpr_at_fpr_01],
        color='red', linestyle='--', label="TPR at 1% FPR"
    )

    plt.xlabel("False Positive Rate (log scale)")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.savefig("RF_Classifier_ROC_AUC.pdf")
    print(f"TPR of test set at FPR 0.1: {tpr_at_fpr_01}")

    # Load the challenge set features and labels from the dataset dir.
    X_challenge, y_challenge = thrember.read_vectorized_features(args.data_dir, "challenge")

    # Combine with benign files in test set
    # Since challnge set is 100% malicious, mix in benign test samples for a more realistic evaluation
    X_test_benign = X_test[y_test == 0]
    y_test_benign = y_test[y_test == 0]
    X_challenge_eval = np.concatenate((X_test_benign, X_challenge), axis=0)
    y_challenge_eval = np.concatenate((y_test_benign, y_challenge), axis=0)

    # Evaluate model on challenge set and compute ROC AUC and PR AUC
    y_pred_challenge = model.predict_proba(X_challenge_eval)[:, 1]
    roc_auc_challenge = roc_auc_score(y_challenge_eval, y_pred_challenge)
    precision_c, recall_c, _ = precision_recall_curve(y_challenge_eval, y_pred_challenge)
    pr_auc_challenge = auc(recall_c, precision_c)
    print(f"ROC AUC on challenge set: {roc_auc_challenge}")
    print(f"PR AUC on challenge set: {pr_auc_challenge}")
