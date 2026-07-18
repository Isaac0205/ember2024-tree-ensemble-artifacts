"""
This script is used to evaluate the trained LightGBM malicious/benign classifier on unseen data.
"""

import argparse
import thrember
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve

if __name__ == "__main__":
    
    # Used to set up the script so that it can take inputs from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to the directory containing the EMBER2024 dataset.")
    parser.add_argument("model_path", type=str, help="Path to save the trained model.")
    args = parser.parse_args()

    # Load the model that was serialised after training
    model  = lgb.Booster(model_file=args.model_path)

    # Load the test set features and labels from the dataset dir.
    X_test, y_test = thrember.read_vectorized_features(args.data_dir, "test")

    # Evaluate on the test set
    y_pred = model.predict(X_test)

    # Calculate area under the ROC curve and PR curve for test set
    roc_auc = roc_auc_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)
    print(f"ROC AUC on test set: {roc_auc}")
    print(f"PR AUC on test set: {pr_auc}")

    # Plot and save the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    plt.title("ROC Curve for EMBERv3 LightGBM Model")
    plt.plot(fpr, tpr, color='black')
    plt.xlim(0.00005, 1.0)
    plt.ylim(0.65, 1.02)
    plt.xscale("log") # Log scale used to see low FPR region better

    # Used to highlight the the TPR when FPR is at 1%
    fpr_target = 0.01
    index = np.argmin(np.abs(fpr - fpr_target))
    tpr_at_fpr_01 = tpr[index]
    plt.plot([fpr_target, fpr_target, 0], [0, tpr_at_fpr_01, tpr_at_fpr_01], color='red', linestyle='--', label="TPR at 1% FPR")
    
    plt.xlabel("False Positive Rate (log scale)")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.savefig("Classifier_ROC_AUC.pdf")
    print(f"TPR of test set at FPR 0.1: {tpr_at_fpr_01}")

    # Load the challenge set features and labels from the dataset dir.
    X_challenge, y_challenge = thrember.read_vectorized_features(args.data_dir, "challenge")

    # Combine with benign files in test set
    # Since challnge set is 100% malicious, mix in benign test samples for a more realistic evaluation
    X_test_benign = X_test[y_test == 0]
    y_test_benign = y_test[y_test == 0]
    X_challenge = np.concatenate((X_test_benign, X_challenge), axis=0)
    y_challenge = np.concatenate((y_test_benign, y_challenge), axis=0)

    # Evaluate model on challenge set and compute ROC AUC and PR AUC
    y_pred = model.predict(X_challenge)
    roc_auc = roc_auc_score(y_challenge, y_pred)
    precision, recall, _ = precision_recall_curve(y_challenge, y_pred)
    pr_auc = auc(recall, precision)
    print(f"ROC AUC on challenge set: {roc_auc}")
    print(f"PR AUC on challenge set: {pr_auc}")
