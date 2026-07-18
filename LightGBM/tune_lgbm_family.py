"""
This script allows for the hyperparameter tuning of LightGBM for EMBER2024 family classification task using RandomizedSearchCV.
"""

import os
import json
import argparse
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
import thrember

# Chosen threshold during tuning, beyond which subsampling is applied
SUBSAMPLE_THRESHOLD = 300_000

if __name__ == "__main__":

    # Used to set up the script so that it can take inputs from the command line
    parser = argparse.ArgumentParser(
        description="Tune LightGBM Family Classifier on EMBER2024"
    )
    parser.add_argument("data_dir", type=str, help="Path to data partition directory")
    parser.add_argument("output_config", type=str, help="Path to save the best hyperparameter JSON config")
    parser.add_argument("--base-config", type=str, required=True, help="Path to original lgbm_config_family.json")
    parser.add_argument("--n-iter", type=int, default=30, help="Number of RandomizedSearch iterations")
    args = parser.parse_args()

    # Used to validate the data directory and config file (checks that paths exists)
    if not os.path.isdir(args.data_dir):
        raise ValueError(f"Data directory not found: {args.data_dir}")
    if not os.path.isfile(args.base_config):
        raise ValueError(f"Base config file not found: {args.base_config}")

    # Used to create the output directory if it does not already exist
    output_dir = os.path.dirname(args.output_config)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Parse the LightGBM JSON config file to load the baseline hyperparameters
    with open(args.base_config, "r") as f:
        base_params = json.load(f)

    # Load the test set features and labels from the dataset dir.
    X, y = thrember.read_vectorized_features(args.data_dir, "train")

    # Remove all unlabelled samples
    valid_mask = y != -1
    X = X[valid_mask]
    y = y[valid_mask]
    n_total = len(y)
    n_families_full = len(np.unique(y))
    print(f"Total labeled family samples: {n_total:,}")
    print(f"Unique families: {n_families_full:,}")

    # If the dataset exceeds the chosen threshold, take a stratified subsample for tuning.
    if n_total > SUBSAMPLE_THRESHOLD:

        # Creating the stratified subsample while preserving chronological order
        indices = np.arange(n_total)

        # Stratified split may fail if some families have only 1 sample for instance
        try:
            tune_idx, _ = train_test_split(
                indices,
                train_size=SUBSAMPLE_THRESHOLD,
                stratify=y,
                random_state=42
            )
        except ValueError as e:
            print(f"Stratified split failed ({e}).")
            
            # Falling back to random split which may under-represent rare families
            tune_idx, _ = train_test_split(
                indices,
                train_size=SUBSAMPLE_THRESHOLD,
                random_state=42
            )

        tune_idx = np.sort(tune_idx) # Indices are sorted to maintain original order
        X_tune = X[tune_idx]
        y_tune = y[tune_idx]

        # Checking class coverage (no families missing) to avoid any silent errors during training
        n_families_subsample = len(np.unique(y_tune))
        coverage = 100.0 * n_families_subsample / n_families_full
        print(f"Subsample size: {len(y_tune):,}")
        print(f"Families in subsample: {n_families_subsample:,} / {n_families_full:,} ({coverage:.1f}%)")

        if n_families_subsample < n_families_full:
            missing = n_families_full - n_families_subsample
            print(f"{missing} rare families absent from subsample.")
    else:
        # Using full training dataset for tuning.
        X_tune = X
        y_tune = y
        n_families_subsample = n_families_full

    # Chosen 6 dimensional hyperparameter search space
    param_distributions = {
        "num_leaves":        [64, 128, 256, 512],
        "learning_rate":     [0.05, 0.1, 0.15],
        "min_data_in_leaf":  [10, 20, 50, 100],
        "n_estimators":      [50, 100, 200],        
        "feature_fraction":  [0.6, 0.8, 0.9, 1.0],
        "bagging_fraction":  [0.7, 0.8, 0.9, 1.0],
    }

    # Chosen CV strategy
    cv = StratifiedKFold(n_splits=3, shuffle=False)

    # Fixed params are loaded from the base config file. The 6 searched params above are excluded here
    estimator = lgb.LGBMClassifier(
        boosting_type=base_params.get("boosting", "gbdt"),
        objective=base_params.get("objective", "multiclass"),
        max_depth=base_params.get("max_depth", -1),
        min_child_weight=base_params.get("min_sum_hessian_in_leaf", 0.001),
        subsample_freq=base_params.get("bagging_freq", 1),
        colsample_bynode=base_params.get("feature_fraction_bynode", 0.9),
        reg_alpha=base_params.get("lambda_l1", 0),
        reg_lambda=base_params.get("lambda_l2", 1.0),
        max_delta_step=base_params.get("max_delta_step", 0),
        is_unbalance=base_params.get("is_unbalance", True),
        random_state=base_params.get("seed", 0),
        n_jobs=-1,
        verbose=-1,
    )

    # Configuring the randomised hyperparameter search
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=args.n_iter,
        scoring="f1_macro",
        cv=cv,
        verbose=3,
        random_state=42,
        n_jobs=1,
        error_score=0.0
    )
    
    # Performing the hyperparameter search using the selected categorical feature indices
    search.fit(X_tune, y_tune, categorical_feature=[2, 3, 4, 5, 6, 701, 702])
    
    # Disaplying the results
    print(f"Best F1 Macro (CV on subsample): {search.best_score_:.6f}")
    print("Best Parameters:")
    for k, v in search.best_params_.items():
        print(f"{k}: {v}")

    # Merging the chosen best params back into the base config while converting sklearn param names back to native LightGBM config keys. All other base config keys are preserved unchanged
    best_params = search.best_params_
    tuned_params = base_params.copy()
    tuned_params["num_iterations"]    = best_params["n_estimators"]
    tuned_params["learning_rate"]     = best_params["learning_rate"]
    tuned_params["num_leaves"]        = best_params["num_leaves"]
    tuned_params["min_data_in_leaf"]  = best_params["min_data_in_leaf"]
    tuned_params["feature_fraction"]  = best_params["feature_fraction"]
    tuned_params["bagging_fraction"]  = best_params["bagging_fraction"]
    
    # Save the tuned config to disk
    with open(args.output_config, "w") as f:
        json.dump(tuned_params, f, indent=4)
