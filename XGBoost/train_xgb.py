"""
This script is used to train an XGBoost classifier on EMBER2024 feature vectors.
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
    parser = argparse.ArgumentParser(description="Train XGBoost classifier on EMBER2024 features")
    parser.add_argument("data_dir", type=str, help="Path to the directory containing the EMBER2024 dataset.")
    parser.add_argument("model_path", type=str, help="Path to save the trained model")
    parser.add_argument("--config-file", type=str, default="xgb_config.json", help="Path to XGBoost config JSON file.")
    args = parser.parse_args()

    # Used to validate the data directory and config file (checks that paths exists)
    if not os.path.isdir(args.data_dir):
        raise ValueError(f"Not a directory: {args.data_dir}")
    if not os.path.isfile(args.config_file):
        raise ValueError(f"Not a file: {args.config_file}")

    # Parse the XGBoost JSON config file specified by the user.
    with open(args.config_file, "r") as f:
        params = json.load(f)
        
    # Load test set features and labels from the dataset dir.
    X, y = thrember.read_vectorized_features(args.data_dir, "train")

    # Check that y shape is not 2D (i.e. not multi-label)
    if len(y.shape) != 1:
        raise ValueError("Multi-label tasks require train_xgb_multilabel.py instead.")

    # Remove unlabelled samples
    valid_mask = y != -1
    X = X[valid_mask]
    y = y[valid_mask]

    # Remap class labels to be 0 indexed (XGBoost requirement)
    unique_classes = np.unique(y) # Get all unique class labels from the data   
    label_map = {old: new for new, old in enumerate(unique_classes)}
    y = np.array([label_map[label] for label in y]) # Apply mapping to all labels

    num_classes = len(np.unique(y))
    n_total = len(y)
    print(f"Total labeled samples: {n_total:,}")
    print(f"Unique classes: {num_classes}")

    # Split data into train and validation sets (90/10 split with stratification)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=params.get("random_state", 0)
    )
    print(f"Train: {len(y_train):,} samples")
    print(f"Validation: {len(y_val):,} samples")

    # Detect the type of task based on number of classes
    if num_classes == 2:
        print("Task: Binary classification")
        # Initialise XGBoost with hyperparameters from passed config file from user
        model = XGBClassifier(**params)
        # Training XGBoost model
        model.fit(X_train, y_train)
    else:
        print(f"Task: Multiclass classification ({num_classes} families)")
        # For multiclass case we add objective and number of classes as well as early stopping
        model = XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=num_classes,
            early_stopping_rounds=10
        )
        # Training XGBoost model
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

    # Create output directory if it does not already exist
    os.makedirs(os.path.dirname(os.path.abspath(args.model_path)), exist_ok=True)

    # Saving the trained model to disk
    with open(args.model_path, "wb") as f:
        if num_classes == 2:
            pickle.dump(model, f)
        else:
            # For multiclass case we also save the label_map to remap predictions back to original labels
            pickle.dump({"model": model, "label_map": label_map}, f)