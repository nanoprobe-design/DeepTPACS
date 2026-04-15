#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: Extract a graph-level embedding for a single SMILES string.
Usage: python get_single_tsne_embedding.py "<SMILES>" [suffix]
Author: Yibin ZHANG
"""

import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Sequence

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


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_NAME = "DeepTPACS"
TASK_NAME = "lg(TPACS)"
FEAT_NUMBER = 6
BATCH_SIZE = 1
RANDOM_SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
class SingleEmbeddingConfig:
    smiles_input: str
    suffix: str
    model_name: str = MODEL_NAME
    task_name: str = TASK_NAME
    feat_number: int = FEAT_NUMBER
    batch_size: int = BATCH_SIZE
    device: torch.device = DEVICE
    project_root: Path = PROJECT_ROOT

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results" / "virtual_screen_data"

    @property
    def output_npz(self) -> Path:
        suffix = f"_{self.suffix}" if self.suffix else ""
        return self.results_dir / f"single_smiles_embedding{suffix}.npz"

    @property
    def cache_file(self) -> Path:
        suffix = f"_{self.suffix}" if self.suffix else ""
        return self.results_dir / f"single_smiles_embedding_graph{suffix}.bin"

    @property
    def checkpoint(self) -> Path:
        return self.project_root / "model" / "trained.pt"

    @property
    def best_params_file(self) -> Path:
        return self.project_root / "results" / f"{self.model_name}_best_params_{self.task_name}.txt"


def parse_args(argv: List[str]) -> SingleEmbeddingConfig:
    smiles_input = argv[1] if len(argv) > 1 else 'CC(C=C1)=CC=C1/C(C2=CC=C(C)C=C2)=C(C3=CC=C(C4=CC=C(C5=CC=C(/C(C6=CC=CC=C6)=C(C7=CC=C(C)C=C7)\C8=CC=C(C)C=C8)C=C5)C=C4)C=C3)\C9=CC=CC=C9'
    suffix = argv[2] if len(argv) > 2 else 'TBT'
    return SingleEmbeddingConfig(smiles_input=smiles_input, suffix=suffix)


def build_featurizers(feat_number: int):
    selected = sorted(FEATURE_ORDER[:feat_number])
    atom_features = [ATOM_FEATURE_BUILDERS[idx] for idx in selected]
    atom_featurizer = BaseAtomFeaturizer(featurizer_funcs={"h": ConcatFeaturizer(atom_features)})
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


def load_model(config: SingleEmbeddingConfig, atom_featurizer, bond_featurizer):
    best_params = read_best_params(config.best_params_file)
    h_space = get_hyper_space(config.model_name)
    model = instantiate_model(config.model_name, atom_featurizer, bond_featurizer, h_space, best_params).to(config.device)
    stopper = EarlyStopping(mode="higher", patience=50, filename=str(config.checkpoint))
    stopper.load_checkpoint(model)
    return model


def build_dataset(config: SingleEmbeddingConfig, atom_featurizer, bond_featurizer):
    config.results_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"SMILES": [config.smiles_input]})
    return csv_dataset.MoleculeCSVDataset(
        df=frame,
        smiles_to_graph=smiles_to_bigraph,
        node_featurizer=atom_featurizer,
        edge_featurizer=bond_featurizer,
        smiles_column="SMILES",
        cache_file_path=str(config.cache_file),
        load=False,
        init_mask=False,
    )


def build_loader(dataset, batch_size: int):
    return DataLoader(
        Subset(dataset, np.arange(len(dataset))),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_molgraphs,
    )


@torch.no_grad()
def collect_embeddings(model, data_loader, device: torch.device):
    embeddings = []
    smiles_list = []
    model.eval()

    for smiles, batch_graph, labels in data_loader:
        atom_feats = batch_graph.ndata["h"].to(device)
        bond_feats = batch_graph.edata["e"].to(device)
        batch_graph = batch_graph.to(device)
        labels = labels.to(device)
        _, _, graph_embedding = model(batch_graph, atom_feats, bond_feats, get_node_weight=True, readout=True)
        embeddings.append(graph_embedding.cpu().squeeze().numpy())
        smiles_list.append(smiles[0])

    return np.asarray(embeddings), smiles_list


def main(argv: List[str]) -> None:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    set_random_seed(seed=RANDOM_SEED)

    config = parse_args(argv)
    atom_featurizer, bond_featurizer = build_featurizers(config.feat_number)
    dataset = build_dataset(config, atom_featurizer, bond_featurizer)
    data_loader = build_loader(dataset, config.batch_size)
    model = load_model(config, atom_featurizer, bond_featurizer)
    embedding_array, smiles_list = collect_embeddings(model, data_loader, config.device)

    np.savez_compressed(file=config.output_npz, embedding_array=embedding_array, SMILES=smiles_list)
    print(embedding_array.shape)
    print("Done")


if __name__ == "__main__":
    main(sys.argv)
