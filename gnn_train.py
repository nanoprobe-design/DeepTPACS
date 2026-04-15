#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: Train DeepTPACS and related GNN models with hyperparameter search and k-fold evaluation.
Usage: python gnn_train.py <model_name> <gpu_id> [gnn|desc_type] [feature_num]
Author: Yibin ZHANG
"""

import os
import shutil
import sys
import time
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
import networkx as nx
from dgl.data.utils import Subset
from dgllife.data import csv_dataset
from easydict import EasyDict
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import KFold, ShuffleSplit
from tqdm import tqdm
from torch.nn import MSELoss, L1Loss
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from GNN_QY_UTILS import *
from model.DeepTPACS import DeepTPACSPredictor_, DeepTPACSPredictorFusion_
from model.GAT import GATPredictor_
from model.GCN import GCNPredictor_
from model.MPNN import MPNNPredictor_
from model.Weave import WeavePredictor_


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TASKS = ["lg(TPACS)"]
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


def set_random_seed(seed=42):
    import random
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def copy_to(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                param.copy_(self.shadow[name])


def weighted_mae_loss(outputs, labels):
    weights = (labels - labels.mean()).abs()
    weights = weights / (weights.mean() + 1e-8)
    weights = torch.clamp(weights, 0.2, 5.0)
    return (weights * (outputs - labels).abs()).mean()


def build_config(argv):
    desc_type = argv[3] if len(argv) > 3 else "gnn"
    feature_num = int(argv[4]) if len(argv) > 4 else 6
    config = {
        "model_name": argv[1],  # gcn, gat, mpnn, DeepTPACS, gatv2, weave
        "len_dataset": None,
        "epochs": 300,
        "batch_size": 32,
        "eval_batch_size": 32,
        "patience": 50,
        "opt_iters": 30,
        "device": torch.device("cuda:" + str(argv[2]) if torch.cuda.is_available() else "CPU"),
        "input": "SMILES",
        "tasks": list(DEFAULT_TASKS),
        "full_path": PROJECT_ROOT,
        "file_name": "data_process/TPACS_sample_data.csv",
        "desc_file": "input_files/descriptors_rdkit.csv",
        "use_desc_fusion": False,
        "desc_type": desc_type,  # argv[3]: rdkit, morgan, daylight, atompair, toptorsion
        "node_mask_rate": 0.0,
        "edge_drop_rate": 0.0,
        "feature_num": feature_num,
    }
    return EasyDict(config)


def build_hyper_spaces(model_name):
    h_space = {
        "l2": [0, 10 ** -8, 10 ** -6, 10 ** -4],
        "lr": [10 ** -2.5, 10 ** -3.5, 10 ** -1.5],
        "hidden_feats": [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256), (64, 64, 128), (32, 32, 64)]
        if model_name in ["gat", "gatv2", "weave"]
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
        ],
        "predictor_hidden_feats": [8, 16, 32, 64, 128, 256],
        "num_layer_set2set": [2, 3, 4],
        "num_heads": [(2, 2, 2), (3, 3, 3), (4, 4, 4), (4, 3, 2), (2, 3, 4)],
        "num_layers": [2, 3, 4, 5],
        "num_timesteps": [2, 3, 4, 5],
        "node_out_feats": [64, 32, 16],
        "edge_hidden_feats": [64, 32, 16],
        "graph_feat_size": [100, 200, 300],
        "dropout": [0, 0.1, 0.2],
        "agg_modes": [("mean", "mean", "mean"), ("flatten", "flatten", "flatten"), ("flatten", "mean", "mean"), ("mean", "flatten", "flatten")],
    }

    Hspace = {
        "gcn": dict(
            l2=hp.choice("l2", h_space["l2"]),
            lr=hp.choice("lr", h_space["lr"]),
            hidden_feats=hp.choice("hidden_feats", h_space["hidden_feats"]),
            predictor_hidden_feats=hp.choice("predictor_hidden_feats", h_space["predictor_hidden_feats"]),
        ),
        "mpnn": dict(
            l2=hp.choice("l2", h_space["l2"]),
            lr=hp.choice("lr", h_space["lr"]),
            node_out_feats=hp.choice("node_out_feats", h_space["node_out_feats"]),
            edge_hidden_feats=hp.choice("edge_hidden_feats", h_space["edge_hidden_feats"]),
            num_layer_set2set=hp.choice("num_layer_set2set", h_space["num_layer_set2set"]),
        ),
        "gat": dict(
            l2=hp.choice("l2", h_space["l2"]),
            lr=hp.choice("lr", h_space["lr"]),
            hidden_feats=hp.choice("hidden_feats", h_space["hidden_feats"]),
            num_heads=hp.choice("num_heads", h_space["num_heads"]),
            predictor_hidden_feats=hp.choice("predictor_hidden_feats", h_space["predictor_hidden_feats"]),
        ),
        "gatv2": dict(
            l2=hp.choice("l2", h_space["l2"]),
            lr=hp.choice("lr", h_space["lr"]),
            hidden_feats=hp.choice("hidden_feats", h_space["hidden_feats"]),
            num_heads=hp.choice("num_heads", h_space["num_heads"]),
            agg_modes=hp.choice("agg_modes", h_space["agg_modes"]),
            predictor_hidden_feats=hp.choice("predictor_hidden_feats", h_space["predictor_hidden_feats"]),
        ),
        "weave": dict(
            l2=hp.choice("l2", h_space["l2"]),
            lr=hp.choice("lr", h_space["lr"]),
            num_layers=hp.choice("num_layers", h_space["num_layers"]),
            edge_hidden_feats=hp.choice("edge_hidden_feats", h_space["edge_hidden_feats"]),
        ),
        "DeepTPACS": dict(
            l2=hp.choice("l2", h_space["l2"]),
            lr=hp.choice("lr", h_space["lr"]),
            num_layers=hp.choice("num_layers", h_space["num_layers"]),
            num_timesteps=hp.choice("num_timesteps", h_space["num_timesteps"]),
            dropout=hp.choice("dropout", h_space["dropout"]),
            graph_feat_size=hp.choice("graph_feat_size", h_space["graph_feat_size"]),
        ),
    }
    return h_space, Hspace[model_name]


def build_index_space(model_name, h_space):
    if model_name == "gcn":
        return dict(
            l2=hp.choice("l2", list(range(len(h_space["l2"])))),
            lr=hp.choice("lr", list(range(len(h_space["lr"])))),
            hidden_feats=hp.choice("hidden_feats", list(range(len(h_space["hidden_feats"])))),
            predictor_hidden_feats=hp.choice("predictor_hidden_feats", list(range(len(h_space["predictor_hidden_feats"])))),
        )
    if model_name == "mpnn":
        return dict(
            l2=hp.choice("l2", list(range(len(h_space["l2"])))),
            lr=hp.choice("lr", list(range(len(h_space["lr"])))),
            node_out_feats=hp.choice("node_out_feats", list(range(len(h_space["node_out_feats"])))),
            edge_hidden_feats=hp.choice("edge_hidden_feats", list(range(len(h_space["edge_hidden_feats"])))),
            num_layer_set2set=hp.choice("num_layer_set2set", list(range(len(h_space["num_layer_set2set"])))),
        )
    if model_name == "gat":
        return dict(
            l2=hp.choice("l2", list(range(len(h_space["l2"])))),
            lr=hp.choice("lr", list(range(len(h_space["lr"])))),
            hidden_feats=hp.choice("hidden_feats", list(range(len(h_space["hidden_feats"])))),
            num_heads=hp.choice("num_heads", list(range(len(h_space["num_heads"])))),
            predictor_hidden_feats=hp.choice("predictor_hidden_feats", list(range(len(h_space["predictor_hidden_feats"])))),
        )
    if model_name == "gatv2":
        return dict(
            l2=hp.choice("l2", list(range(len(h_space["l2"])))),
            lr=hp.choice("lr", list(range(len(h_space["lr"])))),
            hidden_feats=hp.choice("hidden_feats", list(range(len(h_space["hidden_feats"])))),
            num_heads=hp.choice("num_heads", list(range(len(h_space["num_heads"])))),
            agg_modes=hp.choice("agg_modes", list(range(len(h_space["agg_modes"])))),
            predictor_hidden_feats=hp.choice("predictor_hidden_feats", list(range(len(h_space["predictor_hidden_feats"])))),
        )
    if model_name == "weave":
        return dict(
            l2=hp.choice("l2", list(range(len(h_space["l2"])))),
            lr=hp.choice("lr", list(range(len(h_space["lr"])))),
            num_layers=hp.choice("num_layers", list(range(len(h_space["num_layers"])))),
            edge_hidden_feats=hp.choice("edge_hidden_feats", list(range(len(h_space["edge_hidden_feats"])))),
        )
    if model_name == "DeepTPACS":
        return dict(
            l2=hp.choice("l2", list(range(len(h_space["l2"])))),
            lr=hp.choice("lr", list(range(len(h_space["lr"])))),
            num_layers=hp.choice("num_layers", list(range(len(h_space["num_layers"])))),
            num_timesteps=hp.choice("num_timesteps", list(range(len(h_space["num_timesteps"])))),
            dropout=hp.choice("dropout", list(range(len(h_space["dropout"])))),
            graph_feat_size=hp.choice("graph_feat_size", list(range(len(h_space["graph_feat_size"])))),
        )
    raise ValueError("Unsupported model_name: {}".format(model_name))


def _parse_param_value(raw):
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def load_best_params(save_best_params):
    best_opt_para = {}
    with open(save_best_params, "r") as f:
        for line in f.readlines():
            key, value = line.split(":")
            best_opt_para[key.strip()] = _parse_param_value(value.strip())
    return best_opt_para


def save_best_params(save_best_params, best_opt_para):
    with open(save_best_params, "w") as f:
        for key in sorted(best_opt_para.keys()):
            f.write("{}: {}\n".format(key, best_opt_para[key]))


def _resolve_param(h_space, best_opt_para, key):
    value = best_opt_para[key]
    if isinstance(value, (int, np.integer)):
        return h_space[key][value]
    return value


def apply_tunable_args(args, best_opt_para):
    return args


def build_featurizers(feature_num):
    feature_index = sorted(FEATURE_ORDER[:feature_num])
    return feature_index, ATOM_FEATURE_BUILDERS


def build_atom_featurizer(feature_num):
    feature_index, atom_feature_lists_all = build_featurizers(feature_num)
    atom_feature_lists = [atom_feature_lists_all[i] for i in feature_index]
    atom_featurizer = BaseAtomFeaturizer(featurizer_funcs={"h": ConcatFeaturizer(atom_feature_lists)})
    return feature_index, atom_featurizer


def get_best_params_path(args):
    return os.path.join(args.full_path, "results", f"{args.model_name}_best_params_{args.tasks[0]}.txt")


def build_dataset(df, args, AtomFeaturizer, BondFeaturizer):
    cache_file_path = args.file_name.replace(".csv", "_feature{}.bin".format(args.feature_num))
    dataset = csv_dataset.MoleculeCSVDataset(
        df=df.loc[:, [args.input] + args.tasks],
        smiles_to_graph=smiles_to_bigraph,
        node_featurizer=AtomFeaturizer,
        edge_featurizer=BondFeaturizer,
        smiles_column=args.input,
        cache_file_path=cache_file_path,
        load=False,
        init_mask=False,
    )
    return dataset


def build_train_test_loaders(dataset, train_index, test_index, args, collate_fn=collate_molgraphs):
    train_loader = DataLoader(
        Subset(dataset, train_index),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        Subset(dataset, test_index),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, test_loader


def build_model(args, AtomFeaturizer, BondFeaturizer, h_space, best_opt_para):
    if args.model_name == "gcn":
        model = GCNPredictor_(
            in_feats=AtomFeaturizer.feat_size("h"),
            hidden_feats=h_space["hidden_feats"][best_opt_para["hidden_feats"]],
            predictor_hidden_feats=h_space["predictor_hidden_feats"][best_opt_para["predictor_hidden_feats"]],
        )
    elif args.model_name == "gat":
        model = GATPredictor_(
            in_feats=AtomFeaturizer.feat_size("h"),
            hidden_feats=h_space["hidden_feats"][best_opt_para["hidden_feats"]],
            num_heads=h_space["num_heads"][best_opt_para["num_heads"]],
            predictor_hidden_feats=h_space["predictor_hidden_feats"][best_opt_para["predictor_hidden_feats"]],
        )
    elif args.model_name == "mpnn":
        model = MPNNPredictor_(
            node_in_feats=AtomFeaturizer.feat_size("h"),
            edge_in_feats=BondFeaturizer.feat_size("e"),
            node_out_feats=h_space["node_out_feats"][best_opt_para["node_out_feats"]],
            edge_hidden_feats=h_space["edge_hidden_feats"][best_opt_para["edge_hidden_feats"]],
            num_layer_set2set=h_space["num_layer_set2set"][best_opt_para["num_layer_set2set"]],
        )
    elif args.model_name == "weave":
        model = WeavePredictor_(
            node_in_feats=AtomFeaturizer.feat_size("h"),
            edge_in_feats=BondFeaturizer.feat_size("e"),
            num_gnn_layers=h_space["num_layers"][best_opt_para["num_layers"]],
            gnn_hidden_feats=h_space["edge_hidden_feats"][best_opt_para["edge_hidden_feats"]],
        )
    elif args.model_name == "DeepTPACS":
        if getattr(args, "use_desc_fusion", False):
            if not getattr(args, "desc_dim", None):
                raise ValueError("desc_dim must be set for descriptor fusion.")
            model = DeepTPACSPredictorFusion_(
                node_feat_size=AtomFeaturizer.feat_size("h"),
                edge_feat_size=BondFeaturizer.feat_size("e"),
                desc_dim=args.desc_dim,
                num_layers=h_space["num_layers"][best_opt_para["num_layers"]],
                num_timesteps=h_space["num_timesteps"][best_opt_para["num_timesteps"]],
                dropout=_resolve_param(h_space, best_opt_para, "dropout"),
                graph_feat_size=h_space["graph_feat_size"][best_opt_para["graph_feat_size"]],
            )
        else:
            model = DeepTPACSPredictor_(
                node_feat_size=AtomFeaturizer.feat_size("h"),
                edge_feat_size=BondFeaturizer.feat_size("e"),
                num_layers=h_space["num_layers"][best_opt_para["num_layers"]],
                num_timesteps=h_space["num_timesteps"][best_opt_para["num_timesteps"]],
                dropout=_resolve_param(h_space, best_opt_para, "dropout"),
                graph_feat_size=h_space["graph_feat_size"][best_opt_para["graph_feat_size"]],
            )
    else:
        raise ValueError("Unsupported model_name: {}".format(args.model_name))
    return model


def train_one_epoch(model, data_loader, loss_func, optimizer, args, ema=None, mask_embed=None):
    model.train()
    train_metric = Meter()
    for _, batch_data in enumerate(data_loader):
        if len(batch_data) == 4:
            smiles, batch_graph, labels, descs = batch_data
        else:
            smiles, batch_graph, labels = batch_data
            descs = None
        if args.edge_drop_rate > 0:
            batch_graph = apply_edge_dropping_batch(batch_graph, args.edge_drop_rate)
        atom_feats = batch_graph.ndata["h"]
        bond_feats = batch_graph.edata["e"]

        batch_graph = batch_graph.to(args.device)
        labels = labels.to(args.device)
        atom_feats = atom_feats.to(args.device)
        bond_feats = bond_feats.to(args.device)
        if descs is not None:
            descs = descs.to(args.device)

        if args.node_mask_rate > 0 and mask_embed is not None:
            mask = torch.rand(atom_feats.shape[0], device=atom_feats.device) < args.node_mask_rate
            atom_feats = atom_feats.clone()
            atom_feats[mask] = mask_embed

        if args.model_name in ["gcn", "gat", "gatv2"]:
            outputs = model(batch_graph, atom_feats)
        elif args.model_name in ["DeepTPACS"]:
            if descs is not None:
                outputs = model(batch_graph, atom_feats, bond_feats, descs)
            else:
                outputs, _, _ = model(batch_graph, atom_feats, bond_feats, get_node_weight=True, readout=True)
        else:
            outputs = model(batch_graph, atom_feats, bond_feats)

        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)

        train_metric.update(outputs, labels)

    mse_score = np.mean(train_metric.compute_metric("mse"))
    mae_score = np.mean(train_metric.compute_metric("mae"))
    r2_score = np.mean(train_metric.compute_metric("r2"))
    pearson_score = np.mean(train_metric.compute_metric("pearson"))
    return {"mse": mse_score, "mae": mae_score, "r2": r2_score, "pearson": pearson_score}


def eval_one_epoch(model, data_loader, args, epoch=0, heat=False, verbose=False, outputs_result=None, labels_result=None):
    model.eval()
    eval_metric = Meter()
    count = 0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            if len(batch_data) == 4:
                smiles, batch_graph, labels, descs = batch_data
            else:
                smiles, batch_graph, labels = batch_data
                descs = None
            atom_feats = batch_graph.ndata["h"]
            bond_feats = batch_graph.edata["e"]
            batch_graph = batch_graph.to(args.device)
            labels = labels.to(args.device)
            atom_feats = atom_feats.to(args.device)
            bond_feats = bond_feats.to(args.device)
            if descs is not None:
                descs = descs.to(args.device)

            if heat and batch_id == 0:
                if args.model_name in ["gcn", "gat", "gatv2"]:
                    outputs = model(batch_graph, atom_feats, epoch, batch_id)
                else:
                    if args.model_name in ["DeepTPACS"] and descs is not None:
                        outputs = model(batch_graph, atom_feats, bond_feats, descs)
                    else:
                        outputs = model(batch_graph, atom_feats, bond_feats)
            else:
                if args.model_name in ["gcn", "gat", "gatv2"]:
                    outputs = model(batch_graph, atom_feats)
                elif args.model_name in ["DeepTPACS"]:
                    if descs is not None:
                        outputs = model(batch_graph, atom_feats, bond_feats, descs)
                    else:
                        outputs, _, _ = model(batch_graph, atom_feats, bond_feats, get_node_weight=True, readout=True)
                        count += 1
                else:
                    outputs = model(batch_graph, atom_feats, bond_feats)

            eval_metric.update(outputs, labels)

            if verbose and outputs_result is not None and labels_result is not None:
                for i in range(len(outputs)):
                    outputs_result.append(outputs[i].cpu().detach().numpy()[0])
                    labels_result.append(labels[i].cpu().detach().numpy()[0])

    mse_score = np.mean(eval_metric.compute_metric("mse"))
    mae_score = np.mean(eval_metric.compute_metric("mae"))
    r2_score = np.mean(eval_metric.compute_metric("r2"))
    pearson_score = np.mean(eval_metric.compute_metric("pearson"))
    return {"mse": mse_score, "mae": mae_score, "r2": r2_score, "pearson": pearson_score}


def ensure_dir(path):
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def resolve_desc_path(args):
    if getattr(args, "desc_type", None):
        desc_map = {
            "rdkit": "descriptors_rdkit.csv",
            "morgan": "morgan_fp.csv",
            "daylight": "daylight_fp.csv",
            "atompair": "atompair_fp.csv",
            "toptorsion": "toptorsion_fp.csv",
        }
        if args.desc_type not in desc_map:
            raise ValueError("Unsupported desc_type: {}".format(args.desc_type))
        # Use different descriptor files by desc_type (generated by get-descriptors.py).
        desc_path = os.path.join("input_files", desc_map[args.desc_type])
    else:
        desc_path = args.desc_file

    if not os.path.isabs(desc_path):
        desc_path = os.path.join(args.full_path, desc_path)
    return desc_path


def load_descriptors(desc_path, standardize=True):
    df_desc = pd.read_csv(desc_path)
    if "SMILES" not in df_desc.columns:
        raise ValueError("SMILES column not found in {}".format(desc_path))
    drop_cols = [col for col in ["SMILES", "y"] if col in df_desc.columns]
    desc_cols = [col for col in df_desc.columns if col not in drop_cols]
    if not desc_cols:
        raise ValueError("No descriptor columns found in {}".format(desc_path))
    df_desc = df_desc.dropna(subset=["SMILES"])
    desc_values = df_desc[desc_cols].apply(pd.to_numeric, errors="coerce")
    desc_values = desc_values.replace([np.inf, -np.inf], np.nan)
    if standardize:
        mean = desc_values.mean(axis=0)
        std = desc_values.std(axis=0)
        std = std.replace(0.0, 1.0)
        desc_values = (desc_values - mean) / std
    desc_values = desc_values.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    desc_map = {}
    for idx, smiles in enumerate(df_desc["SMILES"].tolist()):
        if smiles in desc_map:
            continue
        desc_map[smiles] = desc_values.iloc[idx].to_numpy()
    return desc_map, len(desc_cols)


class MoleculeDescDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, desc_map, desc_dim):
        self.base_dataset = base_dataset
        self.desc_map = desc_map
        self.desc_dim = desc_dim

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        smiles, graph, label = self.base_dataset[idx]
        desc = self.desc_map.get(smiles)
        if desc is None:
            raise KeyError("Descriptor not found for SMILES: {}".format(smiles))
        desc = torch.tensor(desc, dtype=torch.float32)
        return smiles, graph, label, desc


def collate_molgraphs_with_desc(data):
    smiles, graphs, labels, descs = map(list, zip(*data))
    for i in range(len(graphs)):
        graphs[i] = dgl.add_self_loop(graphs[i])

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)
    descs = torch.stack(descs, dim=0)
    return smiles, bg, labels, descs


def _edge_drop_mask(g, drop_rate):
    if drop_rate <= 0:
        return torch.ones(g.num_edges(), dtype=torch.bool)

    u, v = g.edges()
    u = u.tolist()
    v = v.tolist()

    pair_to_eid = {}
    for eid, (src, dst) in enumerate(zip(u, v)):
        pair_to_eid[(src, dst)] = eid

    undirected = []
    for src, dst in zip(u, v):
        if src < dst and (dst, src) in pair_to_eid:
            undirected.append((src, dst))

    nx_g = nx.Graph()
    nx_g.add_nodes_from(range(g.num_nodes()))
    nx_g.add_edges_from(undirected)
    bridge_set = set(nx.bridges(nx_g))

    ring_mask = None
    if "e" in g.edata and g.edata["e"].dim() == 2 and g.edata["e"].shape[1] > 5:
        ring_mask = g.edata["e"][:, 5] == 1

    keep = torch.ones(g.num_edges(), dtype=torch.bool)
    for src, dst in undirected:
        if (src, dst) in bridge_set or (dst, src) in bridge_set:
            continue
        eid = pair_to_eid[(src, dst)]
        rev_eid = pair_to_eid.get((dst, src))
        if rev_eid is None:
            continue
        if ring_mask is not None and (ring_mask[eid] or ring_mask[rev_eid]):
            continue
        if np.random.rand() < drop_rate:
            keep[eid] = False
            keep[rev_eid] = False
    return keep


def apply_edge_dropping_batch(batch_graph, drop_rate):
    graphs = dgl.unbatch(batch_graph)
    new_graphs = []
    for g in graphs:
        mask = _edge_drop_mask(g, drop_rate)
        try:
            g_new = dgl.edge_subgraph(g, mask, preserve_nodes=True)
        except TypeError:
            g_new = dgl.edge_subgraph(g, mask, relabel_nodes=False)
        new_graphs.append(g_new)
    return dgl.batch(new_graphs)


def update_bn_dgl(model, data_loader, args):
    model.train()
    for module in model.modules():
        if hasattr(module, "reset_running_stats"):
            module.reset_running_stats()
        if hasattr(module, "momentum"):
            module.momentum = None

    with torch.no_grad():
        for batch_data in data_loader:
            if len(batch_data) == 4:
                _, batch_graph, labels, descs = batch_data
            else:
                _, batch_graph, labels = batch_data
                descs = None
            atom_feats = batch_graph.ndata["h"]
            bond_feats = batch_graph.edata["e"]

            batch_graph = batch_graph.to(args.device)
            labels = labels.to(args.device)
            atom_feats = atom_feats.to(args.device)
            bond_feats = bond_feats.to(args.device)
            if descs is not None:
                descs = descs.to(args.device)

            if args.model_name in ["gcn", "gat", "gatv2"]:
                model(batch_graph, atom_feats)
            elif args.model_name in ["DeepTPACS"]:
                if descs is not None:
                    model(batch_graph, atom_feats, bond_feats, descs)
                else:
                    model(batch_graph, atom_feats, bond_feats, get_node_weight=True, readout=True)
            else:
                model(batch_graph, atom_feats, bond_feats)

def run_hyperopt(args, h_space, dataset, AtomFeaturizer, BondFeaturizer, collate_fn):
    dataset_index = np.arange(args.len_dataset)
    sp = ShuffleSplit(train_size=0.9, test_size=0.1, random_state=rand_seed)
    train_index, test_index = next(sp.split(dataset_index))

    space = build_index_space(args.model_name, h_space)
    trials = Trials()
    progress = tqdm(total=args.opt_iters, desc="Hyperopt", dynamic_ncols=True)
    best_mse = [float("inf")]
    best_model_file = os.path.join(
        args.checkpoint_dir, "hyperopt", "{}_{}_hyperopt_best.pt".format(args.model_name, args.tasks[0])
    )

    def objective(param_idx):
        local_args = EasyDict(args.copy())
        apply_tunable_args(local_args, param_idx)
        train_loader, test_loader = build_train_test_loaders(
            dataset, train_index, test_index, local_args, collate_fn=collate_fn
        )

        model = build_model(local_args, AtomFeaturizer, BondFeaturizer, h_space, param_idx)
        model.to(local_args.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=_resolve_param(h_space, param_idx, "lr"),
            weight_decay=_resolve_param(h_space, param_idx, "l2"),
        )
        mask_embed = None
        if local_args.node_mask_rate > 0:
            mask_embed = nn.Parameter(torch.zeros(AtomFeaturizer.feat_size("h"), device=local_args.device))
            optimizer.add_param_group({"params": [mask_embed]})
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=5, min_lr=1e-6
        )
        model_dir = os.path.join(args.checkpoint_dir, "hyperopt")
        ensure_dir(model_dir)
        model_file = os.path.join(
            args.checkpoint_dir, "hyperopt", "{}_{}_hyperopt.pt".format(args.model_name, args.tasks[0])
        )
        stopper = EarlyStopping(mode="lower", patience=local_args.patience, filename=model_file)

        loss_func = L1Loss()
        for _ in range(args.epochs):
            train_one_epoch(model, train_loader, loss_func, optimizer, local_args, mask_embed=mask_embed)
            test_scores = eval_one_epoch(model, test_loader, local_args)
            early_stop = stopper.step(test_scores["mae"], model)
            scheduler.step(test_scores["mae"])
            if early_stop:
                break

        stopper.load_checkpoint(model)
        test_scores = eval_one_epoch(model, test_loader, local_args)
        if test_scores["mse"] < best_mse[0]:
            best_mse[0] = test_scores["mse"]
            if os.path.exists(model_file):
                shutil.copyfile(model_file, best_model_file)
        progress.update(1)
        progress.set_postfix(
            model=args.model_name,
            mse="{:.4f}".format(test_scores["mse"]),
            r2="{:.4f}".format(test_scores["r2"]),
            best="{:.4f}".format(best_mse[0]),
        )
        return {"loss": test_scores["mse"], "status": STATUS_OK}

    best_opt_para = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=args.opt_iters,
        trials=trials,
        rstate=np.random.default_rng(rand_seed),
    )
    progress.close()
    return best_opt_para


def run_feature_experiments(args):
    h_space, _ = build_hyper_spaces(args.model_name)
    desc_tag = args.desc_type if getattr(args, "use_desc_fusion", False) else "gnn"
    feat_tag = "feature_{}".format(args.feature_num)
    args.results_dir = os.path.join(args.full_path, "results", feat_tag)
    args.checkpoint_dir = os.path.join(args.full_path, "checkpoint", feat_tag)
    best_params_path = get_best_params_path(args)

    df = pd.read_csv(os.path.join(args.full_path, args.file_name))
    args.len_dataset = len(df)
    feature_index_selected, AtomFeaturizer = build_atom_featurizer(args.feature_num)

    for _ in range(1):
        print(
            "model_name: {}, desc_type: {}, feature_num: {}, feature_index_selected: {}".format(
                args.model_name, args.desc_type, args.feature_num, feature_index_selected
            )
        )

        BondFeaturizer = GNNBondFeaturizer
        dataset = build_dataset(df, args, AtomFeaturizer, BondFeaturizer)
        collate_fn = collate_molgraphs
        if args.model_name == "DeepTPACS" and getattr(args, "use_desc_fusion", False):
            desc_path = resolve_desc_path(args)
            desc_map, desc_dim = load_descriptors(desc_path)
            args.desc_dim = desc_dim
            dataset = MoleculeDescDataset(dataset, desc_map, desc_dim)
            collate_fn = collate_molgraphs_with_desc

        ensure_dir(os.path.dirname(best_params_path))
        if os.path.exists(best_params_path):
            best_opt_para = load_best_params(best_params_path)
        else:
            best_opt_para = run_hyperopt(args, h_space, dataset, AtomFeaturizer, BondFeaturizer, collate_fn)
            save_best_params(best_params_path, best_opt_para)
            print("best params saved to: {}".format(best_params_path))
        apply_tunable_args(args, best_opt_para)

        dataset_index = np.arange(args.len_dataset)
        kf = KFold(n_splits=10, shuffle=True, random_state=rand_seed)

        fold_scores = []
        rng = np.random.default_rng(42)
        for fold, (train_index, test_index) in enumerate(kf.split(dataset_index), start=1):
            epoch_metrics = []
            rng.shuffle(train_index)
            rng.shuffle(test_index)
            train_loader, test_loader = build_train_test_loaders(
                dataset, train_index, test_index, args, collate_fn=collate_fn
            )

            model = build_model(args, AtomFeaturizer, BondFeaturizer, h_space, best_opt_para)
            model.to(args.device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=_resolve_param(h_space, best_opt_para, "lr"),
                weight_decay=_resolve_param(h_space, best_opt_para, "l2"),
            )
            mask_embed = None
            if args.node_mask_rate > 0:
                mask_embed = nn.Parameter(torch.zeros(AtomFeaturizer.feat_size("h"), device=args.device))
                optimizer.add_param_group({"params": [mask_embed]})
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.8, patience=5, min_lr=1e-6
            )
            ema = EMA(model, decay=0.999)
            ema_model = build_model(args, AtomFeaturizer, BondFeaturizer, h_space, best_opt_para)
            ema_model.to(args.device)
            ema.copy_to(ema_model)
            swa_model = AveragedModel(model).to(args.device)
            stage1_end = max(1, int(args.epochs * 0.5))
            stage2_end = max(stage1_end + 1, int(args.epochs * 0.8))
            swa_start = stage2_end

            model_dir = os.path.join(args.checkpoint_dir, "training_kfold")
            ensure_dir(model_dir)
            model_file = os.path.join(
                args.checkpoint_dir, "training_kfold", "{}_{}_fold_{}.pt".format(args.model_name, args.tasks[0], fold)
            )
            best_r2_file = os.path.join(
                args.checkpoint_dir,
                "training_kfold",
                "{}_{}_fold_{}_best_r2.pt".format(args.model_name, args.tasks[0], fold),
            )
            best_r2 = -float("inf")
            stopper = EarlyStopping(mode="lower", patience=args.patience, filename=model_file)

            loss_func = L1Loss()
            for epoch in range(args.epochs):
                s_t = time.time()
                if epoch >= stage1_end:
                    active_loss = weighted_mae_loss
                else:
                    active_loss = loss_func
                train_scores = train_one_epoch(model, train_loader, active_loss, optimizer, args, ema=ema, mask_embed=mask_embed)
                test_scores = eval_one_epoch(model, test_loader, args)
                early_stop = stopper.step(test_scores["mae"], model)
                scheduler.step(test_scores["mae"])
                alpha_abs = None
                if hasattr(model, "alpha"):
                    alpha_abs = float(model.alpha.detach().abs().cpu())

                if test_scores["r2"] > best_r2:
                    best_r2 = test_scores["r2"]
                    torch.save(model.state_dict(), best_r2_file)
                ema.copy_to(ema_model)
                if (epoch + 1) % 5 == 0:
                    ema_scores = eval_one_epoch(ema_model, test_loader, args)
                    if ema_scores["r2"] > best_r2:
                        best_r2 = ema_scores["r2"]
                        torch.save(ema_model.state_dict(), best_r2_file)
                if epoch >= swa_start:
                    swa_model.update_parameters(model)
                e_t = time.time()
                epoch_metrics.append(
                    {
                        "fold": fold,
                        "epoch": epoch,
                        "lr": optimizer.param_groups[0]["lr"],
                        "alpha_abs": alpha_abs,
                        "train_mse": train_scores["mse"],
                        "train_mae": train_scores["mae"],
                        "train_r2": train_scores["r2"],
                        "train_pearson": train_scores["pearson"],
                                    }
                )
                print(
                    "Model:{}, Fold:{}, Epoch:{}/{}, Feature:{}, LR:{:.6g}, MSE:{:.4f}/{:.4f}, MAE:{:.4f}/{:.4f}, "
                    "R2:{:.4f}/{:.4f}, Pearson:{:.4f}/{:.4f}, Alpha:{}, Time:{:.2f}s".format(
                        args.model_name + "-" + args.desc_type,
                        fold,
                        epoch,
                        args.epochs,
                        args.feature_num,
                        optimizer.param_groups[0]["lr"],
                        train_scores["mse"],
                        test_scores["mse"],
                        train_scores["mae"],
                        test_scores["mae"],
                        train_scores["r2"],
                        test_scores["r2"],
                        train_scores["pearson"],
                        test_scores["pearson"],
                        "n/a" if alpha_abs is None else "{:.4f}".format(alpha_abs),
                        e_t - s_t,
                    )
                )
                if early_stop:
                    break

            if swa_model.n_averaged > 0:
                update_bn_dgl(swa_model, train_loader, args)
                swa_scores = eval_one_epoch(swa_model, test_loader, args)
                if swa_scores["r2"] > best_r2:
                    best_r2 = swa_scores["r2"]
                    torch.save(swa_model.module.state_dict(), best_r2_file)

            if os.path.exists(best_r2_file):
                model.load_state_dict(torch.load(best_r2_file, map_location=args.device))
            else:
                stopper.load_checkpoint(model)
            outputs_result = []
            labels_result = []
            train_scores = eval_one_epoch(model, train_loader, args)
            test_scores = eval_one_epoch(model, test_loader, args, verbose=True, outputs_result=outputs_result, labels_result=labels_result)

            print("model_name: {}, train_scores: {}".format(args.model_name, train_scores))
            print("model_name: {}, test_scores: {}".format(args.model_name, test_scores))

            fold_scores.append(
                {
                    "fold": fold,
                    "train_mse": train_scores["mse"],
                    "train_mae": train_scores["mae"],
                    "train_r2": train_scores["r2"],
                    "train_pearson": train_scores["pearson"],
                }
            )

            preds_dir = os.path.join(args.results_dir, "training_kfold", "preds")
            ensure_dir(preds_dir)
            df_preds = pd.DataFrame({"y_true": labels_result, "y_pred": outputs_result})
            df_preds.to_csv(
                os.path.join(
                    preds_dir,
                    "{}_{}_{}_fold_{}_preds.csv".format(args.model_name, args.tasks[0], desc_tag, fold),
                ),
                index=False,
            )

            metrics_dir = os.path.join(args.results_dir, "training_kfold", "metrics")
            ensure_dir(metrics_dir)
            df_epoch = pd.DataFrame(epoch_metrics)
            df_epoch.to_csv(
                os.path.join(
                    metrics_dir,
                    "{}_{}_{}_fold_{}_metrics.csv".format(args.model_name, args.tasks[0], desc_tag, fold),
                ),
                index=False,
            )

        results_dir = os.path.join(args.results_dir, "training_kfold")
        ensure_dir(results_dir)
        df_scores = pd.DataFrame(fold_scores)
        metric_cols = ["train_mse", "train_mae", "train_r2", "train_pearson"]
        mean_row = {"fold": "mean"}
        std_row = {"fold": "std"}
        for col in metric_cols:
            mean_row[col] = df_scores[col].mean()
            std_row[col] = df_scores[col].std()
        df_scores = pd.concat([df_scores, pd.DataFrame([mean_row, std_row])], ignore_index=True)
        train_summary_path = os.path.join(
            results_dir,
            "{}_{}_{}_train_kfold.csv".format(args.model_name, args.tasks[0], desc_tag),
        )
        df_scores.to_csv(train_summary_path, index=False)
        print("train results saved to: {}".format(train_summary_path))

    print("Done")


if __name__ == "__main__":
    start_time = time.time()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    rand_seed = 42
    set_random_seed(seed=rand_seed)

    args = build_config(sys.argv)
    print(args.device)
    print(args)
    run_feature_experiments(args)
