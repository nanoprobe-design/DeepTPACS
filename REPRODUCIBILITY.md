# DeepTPACS reproducibility guide

This document maps the released code and files to the main computational workflow. It deliberately distinguishes what is present in the repository from items that must be supplied separately if they were used for the final manuscript figures.

## 1. Released inputs

### GNN sample dataset

`data_process/TPACS_sample_data.csv` contains 100 rows with the columns:

`id`, `molecular_name`, `SMILES`, `state`, `QY`, `TPAS`, `log(TPACS)`, and `DOI`.

`gnn_train.py` is configured by default to read this file and to train on the target `lg(TPACS)` derived in the code's data-loading workflow.

### Released model

- Checkpoint: `model/trained.pt`
- SHA-256: `edc7eb36f6e0f5860741b93f547e5501cc079657256da81a02a955e2c9e0ec0e`
- Checkpoint top-level fields: `model_state_dict`, `timestep`
- Stored timestep: 265
- Weight tensors: 46

### Hyperparameters

The original stored indices are in `config/DeepTPACS_best_params_index.txt`:

```text
dropout : 0
graph_feat_size : 2
l2 : 2
lr : 1
num_layers : 1
num_timesteps : 0
```

Against the search spaces in `gnn_train.py`, these resolve to:

```text
dropout = 0.0
graph_feat_size = 300
weight_decay (L2) = 1e-6
learning_rate = 10^-3.5 = 3.1622776601683795e-4
num_layers = 3
num_timesteps = 2
```

The resolved values are stored in `config/DeepTPACS_resolved_hyperparameters.yaml`.

## 2. Molecular graph representation

For the default `feature_num=6`, the training code takes the first six entries of `FEATURE_ORDER`, sorts the selected feature indices, and concatenates the corresponding atom descriptors. The released checkpoint confirms a 30-dimensional atom input.

Bond features are produced by `GNNBondFeaturizer` and the checkpoint is consistent with 13-dimensional bond inputs. Molecules are converted from SMILES to bidirectional graphs with DGLLife. The collate function adds graph self-loops before batching.

## 3. Hyperparameter selection

`gnn_train.py` defines a Hyperopt/TPE search over model hyperparameters. For DeepTPACS, the relevant search spaces are:

- L2 weight decay: `[0, 1e-8, 1e-6, 1e-4]`
- learning rate: `[10^-2.5, 10^-3.5, 10^-1.5]`
- GNN layers: `[2, 3, 4, 5]`
- attentive readout timesteps: `[2, 3, 4, 5]`
- graph feature size: `[100, 200, 300]`
- dropout: `[0, 0.1, 0.2]`

The code performs 30 optimization evaluations (`opt_iters=30`) on a 90/10 ShuffleSplit using random seed 42 and minimizes the validation MSE after training with early stopping.

## 4. Ten-fold evaluation

Run:

```bash
python gnn_train.py DeepTPACS 0 gnn 6
```

The code uses:

- random seed: 42;
- 10-fold `KFold` with shuffling;
- maximum epochs: 300;
- training batch size: 32;
- evaluation batch size: 32;
- early-stopping patience: 50;
- Adam optimizer;
- `ReduceLROnPlateau` scheduler with factor 0.8, patience 5, and minimum learning rate 1e-6;
- L1 loss in the first half of training and weighted MAE thereafter;
- exponential moving average (EMA, decay 0.999);
- stochastic weight averaging (SWA) beginning at 80% of the configured maximum epoch count.

Per-fold predictions are written to:

```text
results/feature_6/training_kfold/preds/
```

Aggregate them with:

```bash
python calc_kfold_preds_metrics.py DeepTPACS gnn 6
```

This produces overall MSE, MAE, and R2 from the concatenated held-out fold predictions.

## 5. Reference inference demo

Run:

```bash
python predict_csv.py \
  --input demo/input_smiles.csv \
  --output demo/demo_output.csv \
  --device auto
```

The deposited reference demo was validated by running:

```bash
python tools/run_validation.py
```

Release verification passed. The demo ran on `cuda:0`, predicted three molecules, and completed in
**5.297 s**. The tested software/hardware environment is recorded in
`environment/TESTED_ENVIRONMENT.md` and `environment/tested_environment.json`, and the exact predictions
are stored in `demo/reference_output.csv`.

## 6. Model interpretation

Run:

```bash
python heatmap.py
```

The script uses the released checkpoint, resolved architecture via the index file, and a predefined example molecule/order to generate heatmap-related outputs under `results/heatmap/ordered/`.

## 7. Single-molecule graph embedding

Run:

```bash
python get_single_tsne_embedding.py "<SMILES>" example
```

The graph-level embedding is written under `results/virtual_screen_data/`.

## 8. Large chemical-space workflow

`chemical space generation.py` constructs candidate structures and `predict_smiles.py` performs the original partitioned large-scale prediction workflow. Because the potentially very large generated arrays are intentionally not stored in Git, users must generate or provide the chemical-space `.npy` input before running the full virtual-screening script.

## 9. Classical machine-learning baselines

The classical machine-learning baselines use the same public 100-row `data_process/TPACS_sample_data.csv` file. No separate raw ML dataset is required. The `ml/` directory is generated locally from this CSV.

Generate RDKit descriptors and fingerprints:

```bash
python get-descriptors.py --input data_process/TPACS_sample_data.csv
python data-processing.py
```

The first command reads `SMILES` and automatically detects `log(TPACS)` (or the legacy spelling `lg(TPACS)`) as the regression target. It writes RDKit descriptors and Morgan, Daylight/RDK, atom-pair and topological-torsion fingerprints to `ml/input_files/`. `data-processing.py` then creates the matrices expected by `ml_train.py` under `ml/input_files/ori_data/`.

Example classical baseline:

```bash
python model_params_opt.py XGB --descriptors daylight --n_trials 300
python ml_train.py XGB --descriptors daylight --mode train
```

Thus, the 100-row CSV is the source dataset for the classical-ML sample workflow; descriptor matrices and the `ml/` workspace are reproducibly generated artifacts, not missing source data.

## 10. Full-data requirement

The file name `TPACS_sample_data.csv` and its 100-row size indicate that the public snapshot should be treated as a sample/demo dataset unless it is in fact the complete dataset used for the manuscript. If manuscript model fitting or reported cross-validation metrics used more observations, the complete input dataset must be made available through the repository or a persistent data repository to enable full numerical reproduction.

## 11. Final release checks

The release-integrity check and GNN reference demo have been completed and their outputs are deposited.
Before creating the manuscript release tag:

1. The typical core installation time is approximately 10 min, as stated in `README.md`; actual installation time varies with network speed and platform-specific package builds.
2. If the classical-ML scripts are to be described as validated, install and test the optional `optuna`, `xgboost`, and `tensorflow` dependencies and record that environment separately.
3. Ensure that any complete dataset required to reproduce reported quantitative results is deposited or linked from the manuscript Data Availability statement; the committed 100-row file is explicitly a sample/demo dataset.
4. Create a tagged GitHub release and, preferably, archive that tag in a DOI-issuing repository such as Zenodo.
