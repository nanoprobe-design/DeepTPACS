# DeepTPACS

[![Website](https://img.shields.io/badge/Website-www.TPACS.top-0A7A5A?style=for-the-badge&logo=googlechrome&logoColor=white)](https://www.TPACS.top)
[![License](https://img.shields.io/badge/License-MIT-black?style=for-the-badge)](LICENSE)

Official codebase for the paper:

**Graph Neural Network-Driven Discovery of AIEgen Nanoprobes with Exceptional Two-Photon Action Cross-Sections for Ultra-Deep Imaging**

## Online Platform

**[www.TPACS.top](https://www.TPACS.top)**

Interactive website for TPACS prediction.

## Overview

DeepTPACS provides the main computational workflows used in this study, including:

- GNN-based TPACS model training
- k-fold evaluation and metric aggregation
- prediction from SMILES using a trained checkpoint
- chemical space generation for donor/acceptor design exploration
- classical ML baselines with data, checkpoints, and results stored under `ml/`

## Repository Structure

- `gnn_train.py`: main GNN training entrypoint
- `predict_smiles.py`: batch prediction with the trained model
- `heatmap.py`: heatmap / attribution visualization
- `chemical space generation.py`: chemical space construction utilities
- `analyze_hole_electron_regions.py`: cube-file analysis for hole/electron regions
- `GNN_QY_UTILS.py`: shared graph and utility functions
- `model/`: model definitions and trained checkpoint
- `calc_kfold_preds_metrics.py`: aggregate k-fold test prediction metrics
- `get_single_tsne_embedding.py`: embedding extraction for a single molecule
- `ml/`: classical ML workspace for input data, checkpoints, and result outputs

## Quick Start

Train the GNN model:

```bash
python gnn_train.py DeepTPACS 0 gnn 6
```

Run SMILES prediction with the trained checkpoint:

```bash
python predict_smiles.py
```

Generate a heatmap example:

```bash
python heatmap.py
```

Run chemical space generation:

```bash
python "chemical space generation.py"
```

Aggregate k-fold metrics:

```bash
python calc_kfold_preds_metrics.py DeepTPACS gnn 6
```

Extract a single-molecule embedding:

```bash
python get_single_tsne_embedding.py "<SMILES>" TBT
```

Run classical ML training:

```bash
python ml_train.py XGB --descriptors daylight
```

Run classical ML hyperparameter optimization:

```bash
python model_params_opt.py XGB --descriptors daylight --n_trials 300
```

## Model Checkpoint

The repository currently uses:

- `model/trained.pt`

This checkpoint is used by downstream scripts such as `predict_smiles.py`, `heatmap.py`, and `get_single_tsne_embedding.py`.

## Environment

Main dependencies include:

- Python
- PyTorch
- DGL
- DGLLife
- RDKit
- NumPy
- Pandas
- scikit-learn
- XGBoost
- TensorFlow
- Optuna
- matplotlib

See `requirements.txt` for the current dependency list.

## Citation

If you use this repository or the `www.TPACS.top` platform in academic work, please cite:

**Graph Neural Network-Driven Discovery of AIEgen Nanoprobes with Exceptional Two-Photon Action Cross-Sections for Ultra-Deep Imaging**
