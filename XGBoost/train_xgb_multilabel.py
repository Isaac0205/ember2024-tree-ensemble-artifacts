"""
This script trains an XGBoost One-vs-Rest (OvR) classifier for EMBER2024 multi-label tasks (to reproduce Section 4.6 in the original paper)
"""

import os
import json
import pickle
import argparse
import numpy as np
import thrember
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    # Used to set up the script so that it can take inputs from the command line
    parser = argparse.ArgumentParser(description="Train multi-label OvR XGBoost model (EMBER2024 Section 4.6)")
    parser.add_argument("data_dir", type=str, help="Path to the directory containing vectorised features")
    parser.add_argument("model_path", type=str, help="Path to save the pickled list of OvR XGB models.")
    parser.add_argument("--config-file", type=str, default="xgb_config_tag.json", help="XGBoost config JSON file.")
    args = parser.parse_args()

    # Used to validate the data directory and config file (checks that paths exists)
    if not os.path.isdir(args.data_dir):
        raise ValueError(f"Not a directory: {args.data_dir}")
    if not os.path.isfile(args.config_file):
        raise ValueError(f"Config file not found: {args.config_file}")

    # Parse the XGBoost JSON config file specified by the user.
    with open(args.config_file, "r") as f:
        params = json.load(f)

    # Load test set features and labels from the dataset dir.
    X, y = thrember.read_vectorized_features(args.data_dir, "train")

    # Check that y shape is not 1D (i.e. not binary/multiclass)
    if len(y.shape) != 2:
        raise ValueError("Binary/multiclass tasks require train_xgb.py instead.")

    # Extract number of files and number of labels per file
    n_samples, n_tags = y.shape
    print(f"Training samples: {n_samples:,}")
    print(f"Number of tags: {n_tags}")

    # Create array of sample indices for splitting
    all_idx = np.arange(n_samples)

    # We split indices (not the data directly) into 90% train / 10% validation to avoid creating multiple copies of X
    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=0.1,
        random_state=params.get("random_state", 0)
    )
    
    # Create training and validation feature matrices 
    X_train = X[train_idx]
    X_val   = X[val_idx]

    # List to store a single XGBoost model per tag
    xgb_models = []
    
    # Train separate binary classifier for each tag 
    for i in range(n_tags):
        
        # Extract labels for just this tag from the full label matrix
        y_i = y[:, i]
        y_train_i = y_i[train_idx]
        y_val_i = y_i[val_idx]

        # XGBoost handles class imbalance using scale_pos_weight
        # This is XGBoost's equivalent to class_weight="balanced" in Random Forest
        pos_train = int(y_train_i.sum()) # Count number of positive samples for this tag
        neg_train = len(y_train_i) - pos_train # Count number of negative samples for this tag
        spw = neg_train / pos_train if pos_train > 0 else 1 # Calculate scale_pos_weight

        # Initialise XGBoost with hyperparameters from config file passed in by user
        model_i = XGBClassifier(**params, scale_pos_weight=spw, early_stopping_rounds=10)

        # Train the XGBoost binary classifier for this specific tag
        model_i.fit(
            X_train, y_train_i,
            eval_set=[(X_val, y_val_i)],
            verbose=False
        )

        # Store trained model in our list
        xgb_models.append(model_i)

        # Count the number of positive samples which exist for this tag
        print(f"[Tag {i+1:3d}/{n_tags}] trained | positives: {pos_train:,} | scale_pos_weight: {spw:.2f}")

    # Create output directory if it does not already exist
    os.makedirs(os.path.dirname(os.path.abspath(args.model_path)), exist_ok=True)
    
    # Serialise the list of OvR models to disk 
    with open(args.model_path, "wb") as f:
        pickle.dump(xgb_models, f)