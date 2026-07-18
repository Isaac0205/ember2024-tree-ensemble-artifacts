"""
This script trains a LightGBM One-vs-Rest (OvR) classifier for EMBER2024 multi-label tasks (to reproduce Section 4.6 in the original paper)
"""

import os
import json
import thrember
import argparse
import pickle

if __name__ == "__main__":

    # Used to set up the script so that it can take inputs from the command line
    parser = argparse.ArgumentParser(description="Train multi-label OvR LightGBM model (EMBER2024 Section 4.6)")
    parser.add_argument("data_dir", type=str, help="Path to the directory containing vectorised features")
    parser.add_argument("model_path", type=str, help="Path to save the pickled list of OvR models")
    parser.add_argument("--config-file", type=str, default="lgbm_config_tag.json", help="LightGBM config JSON file (use lgbm_config_tag.json for all tag tasks)")
    args = parser.parse_args()

    # Used to validate the data directory and config file (checks that paths exists)
    if not os.path.isdir(args.data_dir):
        raise ValueError(f"Not a directory: {args.data_dir}")
    if not os.path.isfile(args.config_file):
        raise ValueError(f"Config file not found: {args.config_file}")

    # Parse the LightGBM JSON config file specified by the user.
    fit_params = json.load(open(args.config_file, "r"))

    # Invoke thrember.train_ovr_model to train one independent binary classifier per tag column
    models = thrember.train_ovr_model(args.data_dir, params=fit_params)

    # Create the output directory if it does not already exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    # Serialise the list of OvR models to disk 
    with open(args.model_path, "wb") as f:
        pickle.dump(models, f)