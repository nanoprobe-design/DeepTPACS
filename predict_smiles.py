#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: Run batch prediction from SMILES using the trained checkpoint.
Usage: python predict_smiles.py
Author: Yibin ZHANG
"""

import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from dgl.data.utils import Subset
from dgllife.data import csv_dataset
from torch.utils.data import DataLoader

from GNN_QY_UTILS import *
from model.DeepTPACS import DeepTPACSPredictor_
from model.GAT import GATPredictor_
from model.GCN import GCNPredictor_
from model.MPNN import MPNNPredictor_
from model.Weave import WeavePredictor_


MODEL_NAME = "DeepTPACS"
TASK = "lg(TPACS)"
FEAT_NUMBER = 6
BATCH_SIZE = 5000
NUM_PARTS = 1000
TEST_FIRST_N = None
BUILD_BIN_ONLY = False
PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

INPUT_NPY = Path("D:/GNN_TPCS/virtual_screen_data/chemical_space.npy")
OUTPUT_NPZ = Path("D:/GNN_TPCS/virtual_screen_data/chemical_space_predictions.npz")
OUTPUT_PART_TEMPLATE = "chemical_space_predictions_part_{part_index}.npz"
CACHE_BIN = Path("D:/GNN_TPCS/virtual_screen_data/chemical_space.bin")
CHECKPOINT = PROJECT_ROOT / "model" / "trained.pt"
BEST_PARAMS_FILE = PROJECT_ROOT / "results" / f"{MODEL_NAME}_best_params_{TASK}.txt"

FEATURE_ORDER = [1, 2, 0, 4, 13, 5, 6, 7, 8, 11, 9, 10, 12, 14, 3]
ATOM_FEATURE_BUILDERS = [
    partial(atom_type_one_hot, allowable_set=["C", "N", "O", "F", "Si", "S", "Cl"], encode_unknown=True),
    partial(atom_degree_one_hot, allowable_set=list(range(6))),
    atom_implicit_valence_one_hot,
    partial(atom_total_degree_one_hot, allowable_set=list(range(6))),
    partial(atom_hybridization_one_hot, encode_unknown=True),
    atom_is_chiral_center,
    atom_total_num_H_one_hot,
    atom_formal_charge_one_hot,
    atom_num_radical_electrons_one_hot,
    atom_mass,
    atom_is_in_ring_one_hot,
    atom_is_aromatic_one_hot,
    atom_chiral_tag_one_hot,
    atom_chirality_type_one_hot,
    atom_explicit_valence_one_hot,
]


@dataclass(frozen=True)
class PredictionConfig:
    model_name: str = MODEL_NAME
    task_name: str = TASK
    feat_number: int = FEAT_NUMBER
    batch_size: int = BATCH_SIZE
    num_parts: int = NUM_PARTS
    test_first_n: int | None = TEST_FIRST_N
    build_bin_only: bool = BUILD_BIN_ONLY
    device: torch.device = DEVICE
    input_npy: Path = INPUT_NPY
    output_npz: Path = OUTPUT_NPZ
    cache_bin: Path = CACHE_BIN
    checkpoint: Path = CHECKPOINT
    best_params_file: Path = BEST_PARAMS_FILE

    def part_output_path(self, part_index: int) -> Path:
        return self.output_npz.with_name(OUTPUT_PART_TEMPLATE.format(part_index=part_index))

    def part_cache_path(self, part_index: int) -> Path:
        return self.cache_bin.with_name(f"{self.cache_bin.stem}_part_{part_index}.bin")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_featurizers(feat_number: int):
    feature_indices = sorted(FEATURE_ORDER[:feat_number])
    atom_feature_lists = [ATOM_FEATURE_BUILDERS[idx] for idx in feature_indices]
    atom_featurizer = BaseAtomFeaturizer(featurizer_funcs={"h": ConcatFeaturizer(atom_feature_lists)})
    return atom_featurizer, GNNBondFeaturizer


def get_hyper_space(model_name: str) -> Dict[str, Sequence]:
    hidden_feats = (
        [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256), (64, 64, 128), (32, 32, 64)]
        if model_name in {"gat", "gatv2", "weave"}
        else [
            (32, 32),
            (32, 64),
            (64, 32),
            (64, 64),
            (64, 128),
            (128, 64),
            (128, 128),
            (256, 256),
            (32, 32, 32),
            (64, 64, 64),
            (128, 128, 128),
            (256, 256, 256),
            (64, 64, 128),
            (32, 32, 64),
        ]
    )
    return {
        "hidden_feats": hidden_feats,
        "predictor_hidden_feats": [8, 16, 32, 64, 128, 256],
        "num_layer_set2set": [2, 3, 4],
        "num_heads": [(2, 2, 2), (3, 3, 3), (4, 4, 4), (4, 3, 2), (2, 3, 4)],
        "num_layers": [2, 3, 4, 5],
        "num_timesteps": [2, 3, 4, 5],
        "node_out_feats": [64, 32, 16],
        "edge_hidden_feats": [64, 32, 16],
        "graph_feat_size": [100, 200, 300],
        "dropout": [0, 0.1, 0.2],
    }


def read_best_params(best_params_file: Path) -> Dict[str, int]:
    params: Dict[str, int] = {}
    with best_params_file.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            params[key.strip()] = int(value.strip())
    return params


def instantiate_model(model_name: str, atom_featurizer, bond_featurizer, h_space: Dict[str, Sequence], best_params: Dict[str, int]):
    if model_name == "gcn":
        return GCNPredictor_(
            in_feats=atom_featurizer.feat_size("h"),
            hidden_feats=h_space["hidden_feats"][best_params["hidden_feats"]],
            predictor_hidden_feats=h_space["predictor_hidden_feats"][best_params["predictor_hidden_feats"]],
        )
    if model_name == "gat":
        return GATPredictor_(
            in_feats=atom_featurizer.feat_size("h"),
            hidden_feats=h_space["hidden_feats"][best_params["hidden_feats"]],
            num_heads=h_space["num_heads"][best_params["num_heads"]],
            predictor_hidden_feats=h_space["predictor_hidden_feats"][best_params["predictor_hidden_feats"]],
            agg_modes=["mean", "mean", "mean"],
        )
    if model_name == "mpnn":
        return MPNNPredictor_(
            node_in_feats=atom_featurizer.feat_size("h"),
            edge_in_feats=bond_featurizer.feat_size("e"),
            node_out_feats=h_space["node_out_feats"][best_params["node_out_feats"]],
            edge_hidden_feats=h_space["edge_hidden_feats"][best_params["edge_hidden_feats"]],
            num_layer_set2set=h_space["num_layer_set2set"][best_params["num_layer_set2set"]],
        )
    if model_name == "DeepTPACS":
        return DeepTPACSPredictor_(
            node_feat_size=atom_featurizer.feat_size("h"),
            edge_feat_size=bond_featurizer.feat_size("e"),
            num_layers=h_space["num_layers"][best_params["num_layers"]],
            num_timesteps=h_space["num_timesteps"][best_params["num_timesteps"]],
            dropout=h_space["dropout"][best_params["dropout"]],
            graph_feat_size=h_space["graph_feat_size"][best_params["graph_feat_size"]],
        )
    if model_name == "weave":
        return WeavePredictor_(
            node_in_feats=atom_featurizer.feat_size("h"),
            edge_in_feats=bond_featurizer.feat_size("e"),
            num_gnn_layers=h_space["num_layers"][best_params["num_layers"]],
            gnn_hidden_feats=h_space["edge_hidden_feats"][best_params["edge_hidden_feats"]],
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def load_model(config: PredictionConfig, atom_featurizer, bond_featurizer):
    if not config.best_params_file.exists():
        raise FileNotFoundError(f"Best params file not found: {config.best_params_file}")
    if not config.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {config.checkpoint}")

    best_params = read_best_params(config.best_params_file)
    h_space = get_hyper_space(config.model_name)
    model = instantiate_model(config.model_name, atom_featurizer, bond_featurizer, h_space, best_params).to(config.device)
    stopper = EarlyStopping(mode="higher", patience=1, filename=str(config.checkpoint))
    stopper.load_checkpoint(model)
    return model


def load_smiles(config: PredictionConfig) -> List[str]:
    if not config.input_npy.exists():
        raise FileNotFoundError(f"Input file not found: {config.input_npy}")

    smiles_array = np.load(config.input_npy, allow_pickle=True)
    if config.test_first_n is None:
        smiles_list = list(smiles_array)
        print(f"Full mode enabled: using all {len(smiles_list)} SMILES")
        return smiles_list

    smiles_list = list(smiles_array[: config.test_first_n])
    print(f"Test mode enabled: using first {len(smiles_list)} SMILES")
    return smiles_list


def build_dataset(smiles_part: Sequence[str], config: PredictionConfig, atom_featurizer, bond_featurizer, cache_path: Path):
    frame = pd.DataFrame({"SMILES": list(smiles_part), config.task_name: [0.0] * len(smiles_part)})
    return csv_dataset.MoleculeCSVDataset(
        df=frame.loc[:, ["SMILES", config.task_name]],
        smiles_to_graph=smiles_to_bigraph,
        node_featurizer=atom_featurizer,
        edge_featurizer=bond_featurizer,
        smiles_column="SMILES",
        cache_file_path=str(cache_path),
        load=cache_path.exists(),
        init_mask=False,
    )


def build_data_loader(dataset, batch_size: int):
    return DataLoader(
        Subset(dataset, range(len(dataset))),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_molgraphs,
    )


@torch.no_grad()
def predict_batch(model_name: str, model, data_loader, device: torch.device) -> Tuple[List[str], List[float]]:
    smiles_list: List[str] = []
    predictions: List[float] = []
    model.eval()

    for smiles, batch_graph, _ in data_loader:
        atom_feats = batch_graph.ndata["h"].to(device)
        bond_feats = batch_graph.edata["e"].to(device)
        batch_graph = batch_graph.to(device)

        if model_name in {"gcn", "gat", "gatv2"}:
            outputs = model(batch_graph, atom_feats)
        else:
            outputs = model(batch_graph, atom_feats, bond_feats)

        smiles_list.extend(smiles)
        predictions.extend(float(value) for value in outputs.detach().cpu().reshape(-1))

    return smiles_list, predictions


def save_prediction_part(part_output_path: Path, part_index: int, smiles_list: Sequence[str], predictions: Sequence[float]) -> None:
    ensure_parent_dir(part_output_path)
    np.savez_compressed(
        part_output_path,
        id=np.array([f"chemical_space_part_{part_index}_{idx}" for idx in range(len(smiles_list))], dtype=object),
        SMILES=np.array(smiles_list, dtype=object),
        predictions=np.array(predictions, dtype=float),
    )


def count_saved_prediction_parts(config: PredictionConfig) -> Tuple[int, List[int], int, int]:
    total_count = 0
    count_gt_5 = 0
    existing_parts = 0
    missing_parts: List[int] = []

    for part_index in range(1, config.num_parts + 1):
        part_output_path = config.part_output_path(part_index)
        if not part_output_path.exists():
            missing_parts.append(part_index)
            continue

        with np.load(part_output_path, allow_pickle=True) as data:
            predictions = data["predictions"]
            total_count += len(predictions)
            count_gt_5 += int(np.sum(predictions > 5))
            existing_parts += 1

    return existing_parts, missing_parts, total_count, count_gt_5


def save_merged_predictions(output_path: Path, smiles_list: Sequence[str], predictions: Sequence[float]) -> None:
    ensure_parent_dir(output_path)
    np.savez_compressed(
        output_path,
        id=np.array([f"chemical_space_{idx}" for idx in range(len(smiles_list))], dtype=object),
        SMILES=np.array(smiles_list, dtype=object),
        predictions=np.array(predictions, dtype=float),
    )


def main() -> None:
    start_time = time.time()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    set_random_seed(0)

    config = PredictionConfig()
    atom_featurizer, bond_featurizer = build_featurizers(config.feat_number)
    smiles_list = load_smiles(config)
    model = None if config.build_bin_only else load_model(config, atom_featurizer, bond_featurizer)

    all_smiles: List[str] = []
    all_predictions: List[float] = []
    split_smiles = np.array_split(np.array(smiles_list, dtype=object), config.num_parts)

    for part_index, smiles_part in enumerate(split_smiles, start=1):
        if len(smiles_part) == 0:
            continue

        cache_path = config.part_cache_path(part_index)
        output_path = config.part_output_path(part_index)
        dataset = build_dataset(smiles_part.tolist(), config, atom_featurizer, bond_featurizer, cache_path)
        data_loader = build_data_loader(dataset, config.batch_size)

        print(f"Processing part {part_index}/{config.num_parts}: {len(smiles_part)} SMILES")
        print(f"Using cached graphs: {cache_path.exists()} ({cache_path})")

        if config.build_bin_only:
            print(f"Built bin for part {part_index}/{config.num_parts}")
            continue

        if output_path.exists():
            print(f"Skipped existing prediction for part {part_index}/{config.num_parts}: {output_path}")
            continue

        part_smiles, part_predictions = predict_batch(config.model_name, model, data_loader, config.device)
        save_prediction_part(output_path, part_index, part_smiles, part_predictions)
        print(f"Saved part predictions to: {output_path}")
        print(f"Part {part_index} count of predictions > 5: {int(np.sum(np.array(part_predictions) > 5))}")
        all_smiles.extend(part_smiles)
        all_predictions.extend(part_predictions)

    if config.build_bin_only:
        elapsed_minutes = (time.time() - start_time) / 60
        print("Bin build only mode finished.")
        print(f"Total SMILES count: {len(smiles_list)}")
        print(f"Time is {elapsed_minutes:.2f} min")
        return

    if all_smiles:
        save_merged_predictions(config.output_npz, all_smiles, all_predictions)
        print(f"Saved current-run predictions to: {config.output_npz}")
    else:
        print("No new predictions were generated in this run.")

    existing_parts, missing_parts, total_count, count_gt_5 = count_saved_prediction_parts(config)
    elapsed_minutes = (time.time() - start_time) / 60
    print(f"Read saved prediction parts: {existing_parts}/{config.num_parts}")
    if missing_parts:
        print(f"Missing prediction parts count: {len(missing_parts)}")
    print(f"Total saved SMILES count: {total_count}")
    print(f"Count of saved predictions > 5: {count_gt_5}")
    print(f"Time is {elapsed_minutes:.2f} min")


if __name__ == "__main__":
    main()
