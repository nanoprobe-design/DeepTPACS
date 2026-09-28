# DeepTPACS

[![Website](https://img.shields.io/badge/Website-www.TPACS.top-0A7A5A?style=for-the-badge&logo=googlechrome&logoColor=white)](https://www.TPACS.top)
[![License](https://img.shields.io/badge/License-MIT-black?style=for-the-badge)](LICENSE)

Official code repository for:

**Graph Neural Network-Driven Discovery of AIEgen Nanoprobes with Exceptional Two-Photon Action Cross-Sections for Ultra-Deep Imaging**

DeepTPACS predicts the logarithm of the two-photon action cross-section from molecular SMILES representations and contains the GNN training, cross-validation, inference, interpretation, chemical-space, and auxiliary analysis code used in the study.

## Release contents

This release includes:

- source code for DeepTPACS and comparison GNN models;
- a released DeepTPACS checkpoint at `model/trained.pt`;
- the original Hyperopt index file at `config/DeepTPACS_best_params_index.txt`;
- the resolved numerical hyperparameters at `config/DeepTPACS_resolved_hyperparameters.yaml`;
- a small reference demo at `demo/input_smiles.csv`;
- a portable CSV inference entry point, `predict_csv.py`;
- checkpoint checksums and release metadata;
- scripts to collect the tested software environment and create a numerical demo reference output;
- reproducibility and manuscript-methods documentation under `docs/` and `REPRODUCIBILITY.md`.

The checkpoint SHA-256 is:

```text
edc7eb36f6e0f5860741b93f547e5501cc079657256da81a02a955e2c9e0ec0e
```

Verify the release with:

```bash
python tools/verify_release.py
```

## System requirements

### Operating system

The code is written in Python and is intended for Linux or Windows systems supported by PyTorch, DGL, DGLLife, and RDKit. GPU acceleration is optional for inference but recommended for training.

The released demo was validated on:

- Linux 6.11.0-19-generic (x86_64, glibc 2.39);
- Python 3.10.18;
- PyTorch 2.8.0+cu128;
- DGL 2.4.0+cu124;
- DGLLife 0.3.2;
- RDKit 2025.03.6;
- EasyDict 1.13;
- CUDA 12.8 as reported by PyTorch and cuDNN 9.10.2;
- NVIDIA GeForce RTX 5080 (three GPUs were present; the demo used `cuda:0`).

The complete validated package list is recorded in `environment/TESTED_ENVIRONMENT.md`,
`environment/tested_environment.json`, and `environment/TESTED_PACKAGE_VERSIONS.txt`. The validated EasyDict version is 1.13.

### Core dependencies

Core DeepTPACS training and inference require Python plus the packages listed in `requirements-core.txt`, including PyTorch, DGL, DGLLife, RDKit, NumPy, pandas, scikit-learn, SciPy, matplotlib, NetworkX, EasyDict, and Hyperopt.

Optional classical-ML workflows additionally use the packages in `requirements-optional-ml.txt`.
The released GNN demo was validated in the environment above. In that validation environment,
`optuna`, `xgboost`, and `tensorflow` were not installed, so the classical-ML workflow should not be
described as validated on that same environment unless those optional packages are installed and tested separately.

### Classical-ML sample workflow

The classical ML baselines use the same 100-row sample CSV as the public GNN example. Generate the ML inputs from SMILES with:

```bash
python get-descriptors.py --input data_process/TPACS_sample_data.csv
python data-processing.py
```

This creates `ml/input_files/ori_data/{daylight,morgan,atompair,toptorsion,descriptors_rdkit}_data.csv`. For example, a Daylight-fingerprint XGBoost workflow can then be run with:

```bash
python model_params_opt.py XGB --descriptors daylight --n_trials 300
python ml_train.py XGB --descriptors daylight --mode train
```

No separate raw classical-ML dataset is required; the descriptor matrices are derived from `TPACS_sample_data.csv`.


The original combined dependency list is retained in `requirements.txt`.

### Hardware

No non-standard hardware is required for the small inference demo. CPU execution is supported. A CUDA-capable NVIDIA GPU is recommended for model training and large-scale virtual screening.

## Installation

A typical clean-environment installation is:

```bash
conda create -n deeptpacs python=3.10 -y
conda activate deeptpacs

pip install -r requirements-core.txt
```

If your platform requires a specific CUDA/CPU build of PyTorch or DGL, install the compatible build following the corresponding official package instructions before rerunning the remaining dependencies.

Then verify the released model files:

```bash
python tools/verify_release.py
```

**Typical installation time:** approximately **10 min** for the core DeepTPACS/GNN environment on a standard desktop computer with a stable internet connection. This is a typical estimate for environment creation, dependency installation, and release verification rather than a measured benchmark; the exact time varies with network speed, package-cache state, hardware, and the selected PyTorch/DGL build. Optional classical-ML dependencies are not included.

## Demo

The demo contains three molecules and exercises the released checkpoint end-to-end.

Run:

```bash
python predict_csv.py \
  --input demo/input_smiles.csv \
  --output demo/demo_output.csv \
  --device auto
```

A successful run creates `demo/demo_output.csv` containing the original input columns plus:

- `predicted_log10_TPACS` — DeepTPACS prediction for the manuscript target `lg(TPACS)`;
- `predicted_TPACS` — `10 ** predicted_log10_TPACS`, provided for convenience.

It should report three predictions and the output path. The deposited validation run used `cuda:0`,
predicted all three molecules, and completed in **5.297 s**. The exact numerical reference is committed
as `demo/reference_output.csv`; see `demo/EXPECTED_OUTPUT.md` for the values.

The complete validation command is:

```bash
python tools/run_validation.py
```

The deposited outputs from that command are already included as `environment/TESTED_ENVIRONMENT.md`,
`environment/tested_environment.json`, `environment/VALIDATION_REPORT.txt`, and
`demo/reference_output.csv`.

## Using DeepTPACS on your own data

Prepare a CSV file with a column named `SMILES`, for example:

```text
id,SMILES
mol_1,CCO
mol_2,c1ccccc1
```

Run:

```bash
python predict_csv.py \
  --input my_molecules.csv \
  --output my_predictions.csv \
  --device auto
```

`--device auto` uses `cuda:0` when CUDA is available and otherwise uses CPU. The released checkpoint and parameter-index file are used by default; alternative paths can be supplied with `--checkpoint` and `--params`.

## Released model configuration

The stored Hyperopt indices resolve against the search spaces in `gnn_train.py` as follows:

| Parameter | Stored index | Resolved value |
|---|---:|---:|
| dropout | 0 | 0.0 |
| graph feature size | 2 | 300 |
| L2 weight decay | 2 | 1e-6 |
| learning rate | 1 | 3.1622776601683795e-4 |
| GNN layers | 1 | 3 |
| attentive readout timesteps | 0 | 2 |

The checkpoint architecture is consistent with a 30-dimensional atom feature vector, 13-dimensional bond feature vector, 300-dimensional learned graph representation, three GNN layers, two attentive readout steps, and one regression output.

## Model training

The default DeepTPACS training entry point is:

```bash
python gnn_train.py DeepTPACS 0 gnn 6
```

In the provided code, this configures a maximum of 300 epochs, batch size 32, early-stopping patience 50, 30 Hyperopt/TPE trials, random seed 42, and 10-fold cross-validation. The target is `lg(TPACS)` and the default input file is `data_process/TPACS_sample_data.csv`.

Training outputs are written under `results/feature_6/` and checkpoints under `checkpoint/feature_6/`. These generated directories are intentionally excluded from version control.

To aggregate the per-fold prediction files after training, run:

```bash
python calc_kfold_preds_metrics.py DeepTPACS gnn 6
```

## Heatmap and embedding analyses

Generate the configured heatmap example:

```bash
python heatmap.py
```

Extract a graph-level embedding for one molecule:

```bash
python get_single_tsne_embedding.py "<SMILES>" example
```

Both scripts use `model/trained.pt` and `config/DeepTPACS_best_params_index.txt` by default.

## Large-scale virtual screening

`predict_smiles.py` is retained for the original large chemical-space workflow. Its default input path is now repository-relative (`data_process/chemical_space.npy`) rather than a developer-specific Windows drive. Large generated `.npy`, `.npz`, and `.bin` files remain excluded from Git.

## Reproducing manuscript results

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the code-to-result workflow and exact model settings.

Classical-ML data note: the classical machine-learning workflows use the same public 100-row `data_process/TPACS_sample_data.csv` sample dataset. The `ml/` directory is a generated workspace rather than a second raw dataset. Run `get-descriptors.py` followed by `data-processing.py` to regenerate the descriptor/fingerprint matrices under `ml/input_files/`, after which `model_params_opt.py` and `ml_train.py` can be run for the classical baselines.

## Repository structure

```text
DeepTPACS/
├── README.md
├── REPRODUCIBILITY.md
├── LICENSE
├── CITATION.cff
├── CHECKSUMS.sha256
├── requirements.txt
├── requirements-core.txt
├── requirements-optional-ml.txt
├── predict_csv.py
├── gnn_train.py
├── predict_smiles.py
├── heatmap.py
├── calc_kfold_preds_metrics.py
├── get_single_tsne_embedding.py
├── model/
│   ├── DeepTPACS.py
│   ├── GAT.py
│   ├── GCN.py
│   ├── MPNN.py
│   ├── Weave.py
│   └── trained.pt
├── config/
│   ├── DeepTPACS_best_params_index.txt
│   ├── DeepTPACS_resolved_hyperparameters.yaml
│   └── model_release.json
├── data_process/
│   └── TPACS_sample_data.csv
├── demo/
│   ├── input_smiles.csv
│   ├── reference_output.csv
│   └── EXPECTED_OUTPUT.md
├── environment/
│   ├── TESTED_ENVIRONMENT.md
│   ├── tested_environment.json
│   ├── TESTED_PACKAGE_VERSIONS.txt
│   └── VALIDATION_REPORT.txt
├── tools/
│   ├── verify_release.py
│   ├── collect_environment.py
│   └── run_validation.py
└── docs/
    ├── PSEUDOCODE_AND_CODE_DESCRIPTION.md
    └── CLASSICAL_ML_SAMPLE_WORKFLOW.md
```

## License

The source code is released under the MIT License; see `LICENSE`.

## Citation

If you use this repository or the TPACS online platform, please cite:

**Graph Neural Network-Driven Discovery of AIEgen Nanoprobes with Exceptional Two-Photon Action Cross-Sections for Ultra-Deep Imaging**


## Methods and pseudocode

A detailed, code-grounded description of the DeepTPACS workflow and pseudocode for molecular graph generation, hyperparameter optimization, cross-validation, inference, and the classical-ML sample workflow is provided in `docs/PSEUDOCODE_AND_CODE_DESCRIPTION.md`. The relevant material can be incorporated into the Methods or Supplementary Methods of the associated manuscript.
