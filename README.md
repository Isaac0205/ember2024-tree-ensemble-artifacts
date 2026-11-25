# Re-Benchmarking EMBER2024: A Comparison of Tree-Based Ensembles Against Evasive and Emerging Malware

This repository is the artifact accompanying the submitted paper. It contains the source code, fixed seeds, untuned configurations and selected tuned configurations used to compare LightGBM, XGBoost and Random Forest on EMBER2024.

## Artifact Scope

The artifact supports the paper's reported evaluations of:

- aggregate and format-specific binary malware detection;
- cross-format and All PE evaluation;
- evasive challenge-set detection;
- emerging family detection; and
- multi-label attribute prediction for Behaviour, File Property, Packer, Exploit and Group.

Raw EMBER2024 data are not included because of their size. Trained model binaries are hosted separately in an [anonymous model archive](https://anonymous-hf.up.railway.app/a/04dlw71uy6fj/). The dataset can be reconstructed through the official `thrember` pipeline, and the supplied scripts can recreate the models from the included configurations. No live malware binaries are distributed in this repository.

Family-classification scripts and configurations are retained for completeness, but the multiclass family-classification task is outside the evaluation reported in the paper and is not required to reproduce its tables.

## Repository Contents

| Directory | Contents |
| --- | --- |
| `Exploratory_Data_Analysis` | Dataset tour, label statistics and figures |
| `LightGBM` | LightGBM training, tuning and evaluation scripts and JSON configurations |
| `Random_Forest` | Random Forest training, tuning and evaluation scripts and JSON configurations |
| `XGBoost` | XGBoost training, tuning and evaluation scripts and JSON configurations |

Within each model directory:

- `*_config.json` contains the untuned binary configuration.
- `*_config_tag.json` contains the untuned multi-label configuration.
- `*_tuned_<partition>.json` contains the configuration selected for a binary partition.
- `train_*` scripts train binary or multi-label models.
- `tune_*_binary.py` applies the common randomised-search protocol.
- `eval_*` scripts evaluate standard, challenge, emerging-family or multi-label tasks.

## Dataset Preparation

Download and vectorise EMBER2024 using the official repository:

<https://github.com/FutureComputing4AI/EMBER2024>

The scripts expect the resulting data to follow this simplified structure:

```text
data/
+-- All/
|   +-- X_train.dat
|   +-- X_test.dat
|   +-- X_challenge.dat
|   +-- y_challenge.dat
|   +-- Malicious-benign/
|   |   +-- y_train.dat
|   |   +-- y_test.dat
|   +-- family class min=1/
|   |   +-- y_train.dat
|   |   +-- y_test.dat
|   +-- behaviour class min=10/
|   +-- file_property class min=10/
|   +-- packer class min=10/
|   +-- exploit class min=10/
|   +-- group class min=10/
+-- All PE/
+-- Win32/
+-- Win64/
+-- Dot_Net/
+-- APK/
+-- ELF/
+-- PDF/
```

Each format-specific directory contains its own feature matrices, challenge data and label subdirectories. Before training or evaluating a task, copy that task's `y_train.dat` and `y_test.dat` into the relevant partition root. For binary detection on All Files, for example:

```console
cp "data/All/Malicious-benign/y_train.dat" "data/All/y_train.dat"
cp "data/All/Malicious-benign/y_test.dat" "data/All/y_test.dat"
```

The same procedure applies to a multi-label category by selecting the corresponding `class min=10` directory.

## Reproducibility Settings

The supplied files preserve the seeds used in the experiments:

- LightGBM uses `seed=0`, including its bagging and feature-fraction seeds.
- Random Forest uses `random_state=0`.
- XGBoost uses `random_state=0`.
- Stratified tuning subsampling and randomised search use `random_state=42`.

Binary hyperparameter searches use thirty randomised trials with three-fold stratified cross-validation. At most 300,000 stratified training samples are used during search; partitions below this cap are searched in full. The selected configuration is then retrained on the complete training partition. Multi-label models use the untuned tag configurations, as described in the paper.

## Reproducing a Binary Run

The repository already contains the selected tuned configuration for every reported partition, so rerunning the search is optional.

```console
mkdir models
```

LightGBM example:

```console
python LightGBM/train_lgbm.py "data/All" "models/lgbm_all.model" --config-file LightGBM/lgbm_tuned_All.json
python LightGBM/eval_lgbm.py "data/All" "models/lgbm_all.model"
```

Random Forest example:

```console
python Random_Forest/train_rf.py "data/Win32" "models/rf_win32.pkl" --config-file Random_Forest/rf_tuned_Win32.json
python Random_Forest/eval_rf.py "data/Win32" "models/rf_win32.pkl"
```

XGBoost example:

```console
python XGBoost/train_xgb.py "data/All" "models/xgb_all.pkl" --config-file XGBoost/xgb_tuned_All.json
python XGBoost/eval_xgb.py "data/All" "models/xgb_all.pkl"
```

To rerun a binary search, provide a partition, an output path and the relevant untuned configuration:

```console
python LightGBM/tune_lgbm_binary.py "data/All" "LightGBM/lgbm_tuned_reproduced.json" --base-config LightGBM/lgbm_config.json --n-iter 30
```

## Emerging Family Evaluation

Emerging-family evaluation reuses a trained All Files binary model and the `family class min=1` labels:

```console
python LightGBM/eval_emerging_families.py --data-dir "data/All" --model-path "models/lgbm_all.model"
python Random_Forest/eval_rf_emerging_families.py --data-dir "data/All" --model-path "models/rf_all.pkl"
python XGBoost/eval_xgb_emerging_families.py --data-dir "data/All" --model-path "models/xgb_all.pkl"
```

## Multi-Label Evaluation

After copying one category's `class min=10` labels into the All Files partition root, train and evaluate its One-vs-Rest models. For example:

```console
python XGBoost/train_xgb_multilabel.py "data/All" "models/xgb_behaviour.pkl" --config-file XGBoost/xgb_config_tag.json
python XGBoost/eval_xgb_multilabel.py "data/All" "models/xgb_behaviour.pkl" --task-name Behaviour
```

Equivalent LightGBM and Random Forest scripts are provided in their respective directories.

## Exploratory Analysis

`Exploratory_Data_Analysis/EMBER_2024_Tour.ipynb` reproduces dataset composition, label-distribution and family-size analyses. Set `DATA_DIR` in the notebook to the vectorised All Files partition before running its data-loading cells. The figures used by the analysis are also supplied as standalone PNG files.

## External Resources

- Official EMBER2024 code and dataset instructions: <https://github.com/FutureComputing4AI/EMBER2024>
- Original authors' released LightGBM benchmark models: <https://huggingface.co/joyce8/EMBER2024-benchmark-models>

