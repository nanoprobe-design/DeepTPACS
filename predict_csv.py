#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable DeepTPACS prediction from a CSV file containing a SMILES column.

Example
-------
python predict_csv.py --input demo/input_smiles.csv --output demo/demo_output.csv
"""

from __future__ import annotations

import argparse
import math
from functools import partial
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd
import torch
from dgl.data.utils import Subset
from dgllife.data import csv_dataset
from dgllife.utils import (
    BaseAtomFeaturizer,
    ConcatFeaturizer,
    atom_chirality_type_one_hot,
    atom_degree_one_hot,
    atom_hybridization_one_hot,
    atom_implicit_valence_one_hot,
    atom_is_chiral_center,
    atom_type_one_hot,
    smiles_to_bigraph,
)
from rdkit import Chem
from torch.utils.data import DataLoader

from GNN_QY_UTILS import GNNBondFeaturizer, collate_molgraphs, set_random_seed
from model.DeepTPACS import DeepTPACSPredictor_


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "model" / "trained.pt"
DEFAULT_PARAMS = PROJECT_ROOT / "config" / "DeepTPACS_best_params_index.txt"
DEFAULT_CACHE = PROJECT_ROOT / "results" / "prediction_graphs.bin"
TASK_NAME = "lg(TPACS)"

FEATURE_ORDER = [1, 2, 0, 4, 13, 5, 6, 7, 8, 11, 9, 10, 12, 14, 3]
ATOM_FEATURE_BUILDERS = [
    partial(atom_type_one_hot, allowable_set=["C", "N", "O", "F", "Si", "S", "Cl"], encode_unknown=True),
    partial(atom_degree_one_hot, allowable_set=list(range(6))),
    atom_implicit_valence_one_hot,
    None,  # unused feature slot retained to preserve the training feature ordering
    partial(atom_hybridization_one_hot, encode_unknown=True),
    atom_is_chiral_center,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    atom_chirality_type_one_hot,
    None,
]

HYPER_SPACES = {
    "l2": [0.0, 1e-8, 1e-6, 1e-4],
    "lr": [10 ** -2.5, 10 ** -3.5, 10 ** -1.5],
    "num_layers": [2, 3, 4, 5],
    "num_timesteps": [2, 3, 4, 5],
    "graph_feat_size": [100, 200, 300],
    "dropout": [0.0, 0.1, 0.2],
}


def read_index_params(path: Path) -> Dict[str, int]:
    params: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            params[key.strip()] = int(value.strip())
    required = {"dropout", "graph_feat_size", "l2", "lr", "num_layers", "num_timesteps"}
    missing = required.difference(params)
    if missing:
        raise ValueError(f"Missing hyperparameter indices in {path}: {sorted(missing)}")
    return params


def build_atom_featurizer(feature_num: int = 6) -> BaseAtomFeaturizer:
    selected_indices = sorted(FEATURE_ORDER[:feature_num])
    builders = []
    for idx in selected_indices:
        builder = ATOM_FEATURE_BUILDERS[idx]
        if builder is None:
            raise ValueError(f"Feature index {idx} is not defined in the portable predictor")
        builders.append(builder)
    return BaseAtomFeaturizer(featurizer_funcs={"h": ConcatFeaturizer(builders)})


def build_model(params: Dict[str, int], atom_featurizer: BaseAtomFeaturizer) -> DeepTPACSPredictor_:
    model = DeepTPACSPredictor_(
        node_feat_size=atom_featurizer.feat_size("h"),
        edge_feat_size=GNNBondFeaturizer.feat_size("e"),
        num_layers=HYPER_SPACES["num_layers"][params["num_layers"]],
        num_timesteps=HYPER_SPACES["num_timesteps"][params["num_timesteps"]],
        graph_feat_size=HYPER_SPACES["graph_feat_size"][params["graph_feat_size"]],
        dropout=HYPER_SPACES["dropout"][params["dropout"]],
        n_tasks=1,
    )
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)


def validate_smiles(smiles: Sequence[str]) -> None:
    invalid = [i for i, smi in enumerate(smiles) if Chem.MolFromSmiles(str(smi)) is None]
    if invalid:
        raise ValueError(f"Invalid SMILES at zero-based row indices: {invalid}")


def predict(input_csv: Path, output_csv: Path, checkpoint: Path, params_file: Path,
            batch_size: int, device: torch.device, cache_file: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_csv)
    if "SMILES" not in frame.columns:
        raise ValueError(f"{input_csv} must contain a column named 'SMILES'")
    if frame.empty:
        raise ValueError(f"{input_csv} contains no rows")

    smiles = frame["SMILES"].astype(str).tolist()
    validate_smiles(smiles)

    atom_featurizer = build_atom_featurizer(feature_num=6)
    params = read_index_params(params_file)
    model = build_model(params, atom_featurizer).to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    dataset_frame = pd.DataFrame({"SMILES": smiles, TASK_NAME: [0.0] * len(smiles)})
    dataset = csv_dataset.MoleculeCSVDataset(
        df=dataset_frame,
        smiles_to_graph=smiles_to_bigraph,
        node_featurizer=atom_featurizer,
        edge_featurizer=GNNBondFeaturizer,
        smiles_column="SMILES",
        cache_file_path=str(cache_file),
        load=False,
        init_mask=False,
    )
    loader = DataLoader(
        Subset(dataset, range(len(dataset))),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_molgraphs,
    )

    outputs_all = []
    with torch.no_grad():
        for _, batch_graph, _ in loader:
            atom_feats = batch_graph.ndata["h"].to(device)
            bond_feats = batch_graph.edata["e"].to(device)
            batch_graph = batch_graph.to(device)
            outputs = model(batch_graph, atom_feats, bond_feats)
            outputs_all.extend(float(x) for x in outputs.detach().cpu().reshape(-1))

    if len(outputs_all) != len(frame):
        raise RuntimeError(f"Expected {len(frame)} predictions, obtained {len(outputs_all)}")

    result = frame.copy()
    result["predicted_log10_TPACS"] = outputs_all
    result["predicted_TPACS"] = [math.pow(10.0, value) for value in outputs_all]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict log10(TPACS) from a CSV containing SMILES.")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV with a SMILES column")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def main() -> None:
    args = parse_args()
    set_random_seed(42)
    device = resolve_device(args.device)
    result = predict(
        input_csv=args.input,
        output_csv=args.output,
        checkpoint=args.checkpoint,
        params_file=args.params,
        batch_size=args.batch_size,
        device=device,
        cache_file=args.cache,
    )
    print(f"Device: {device}")
    print(f"Predicted {len(result)} molecules")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
