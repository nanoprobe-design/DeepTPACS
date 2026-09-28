# Classical ML sample-data workflow

The classical machine-learning examples use the same 100-row raw dataset as the public sample GNN workflow:

`data_process/TPACS_sample_data.csv`

The file contains SMILES strings and the target column `log(TPACS)`. No second raw classical-ML dataset is required.

## Generate descriptors/fingerprints

```bash
python get-descriptors.py --input data_process/TPACS_sample_data.csv
python data-processing.py
```

Generated inputs are written under `ml/input_files/` and `ml/input_files/ori_data/`.

## Hyperparameter optimization and training example

```bash
python model_params_opt.py XGB --descriptors daylight --n_trials 300
python ml_train.py XGB --descriptors daylight --mode train
```

The same pattern can be used for the other supported classical algorithms and descriptor types.

## Validation performed while preparing this release

The descriptor-generation and preprocessing chain was executed successfully on the supplied 100-row sample CSV in a clean working copy. The generated matrices were then removed from the release because they are reproducible derived artifacts.
