"""
This script is used to train a LightGBM classifier on EMBER2024 feature vectors.

Different configuration files should be used depending on the task type:
 - Benign/malicious detection: lgbm_config.json
 - Malware family classification: lgbm_config_family.json
 - All other classification tasks: lgbm_config_tag.json
"""

import os
import json
import thrember
import argparse
import lightgbm as lgb
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    # Used to set up the script so that it can take inputs from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to the directory containing the EMBER2024 dataset.")
    parser.add_argument("model_path", type=str, help="Path to save the trained model.")
    parser.add_argument("--config-file", type=str, default="lgbm_config.json", help="Path to LightGBM config file.")
    args = parser.parse_args()

    # Used to validate the data directory and config file (checks that paths exists)
    if not os.path.isdir(args.data_dir):
        raise ValueError(f"Not a directory: {args.data_dir}")
    if not os.path.isfile(args.config_file):
        raise ValueError(f"Not a file: {args.config_file}")

    # Parse the LightGBM JSON config file specified by the user.
    fit_params = json.load(open(args.config_file, "r"))

    # Invoke thrember.train_model for dataset loading and model fitting.
    model = thrember.train_model(args.data_dir, params=fit_params)

    # Saving the trained model to disk
    model.save_model(args.model_path, num_iteration=model.best_iteration)