"""
This script is used to evaluate One-vs-Rest (OvR) LightGBM classifiers for EMBER2024 multi-label tasks (to reproduce Section 4.6 in the original paper).
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Used to suppresses NaN warnings from metrics calculations

import pickle
import argparse
import thrember
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc
)

if __name__ == "__main__":
    
    # Used to set up the script so that it can take inputs from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to the directory containing the EMBER2024 dataset.")
    parser.add_argument("model_path", type=str, help="Path to OvR model")
    parser.add_argument("--task-name", type=str, default="Multi-Label Task", help="Name for output (e.g. Behavior)")
    args = parser.parse_args()

    # Load the pickled list of OvR models
    with open(args.model_path, "rb") as f:
        models = pickle.load(f)
    print(f"Loaded {len(models)} binary LightGBM classifiers (OvR)")
    
    # Load test set features and multi-labels from the dataset dir.
    X_test, y_test = thrember.read_vectorized_features(args.data_dir, "test")
    n_samples, n_tags = y_test.shape
    print(f"Test samples: {n_samples:,}, Tags: {n_tags}")

    # Each model in list is responsible for one tag. 
    # We run each model separately and store its probability scores in the matching column of y_pred.
    y_pred = np.zeros_like(y_test, dtype=float)
    for i, booster in enumerate(models):
        y_pred[:, i] = booster.predict(X_test)

    # y_pred currently holds continuous scores. Convert to hard predictions by thresholding at 0.5
    y_pred_bin = (y_pred >= 0.5).astype(int)

    # Calculate macro-averaged precision, recall and F1 
    precision = precision_score(y_test, y_pred_bin, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred_bin, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred_bin, average="macro", zero_division=0)

    # Compute macro-averaged ROC AUC per tag then average, skipping tags which are all 0's or all 1's and instead saving NaN as a placeholder
    auc_scores = []
    for i in range(n_tags):
        try:
            # Compute ROC AUC for i'th tag
            auc_i = roc_auc_score(y_test[:, i], y_pred[:, i])
            auc_scores.append(auc_i)
        except ValueError:
            auc_scores.append(np.nan)  

    # Average across tags whilst ignoring any NaN placeholders
    auc_macro = np.nanmean(auc_scores)
    
    # Print results 
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"AUC: {auc_macro:.4f}")

    # Plot PR curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test.ravel(), y_pred.ravel())
    pr_auc = auc(recall_curve, precision_curve)

    plt.figure(figsize=(6, 6))
    plt.title(f"Precision-Recall Curve - {args.task_name}")
    plt.plot(recall_curve, precision_curve, color='black')
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.02)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.text(0.6, 0.1, f"PR AUC = {pr_auc:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.savefig(f"{args.task_name.replace(' ', '_')}_PR_Curve.pdf", bbox_inches='tight')