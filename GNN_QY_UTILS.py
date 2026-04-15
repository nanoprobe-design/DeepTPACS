#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: Provide shared graph construction, metrics, and visualization utilities.
Usage: Imported by training, prediction, visualization, and analysis scripts.
Author: Yibin ZHANG
"""

import numpy as np
import torch
import random
import os
import sys
from matplotlib import pyplot as plt
from functools import partial
from dgllife.utils import *
from matplotlib.colors import ColorConverter
from rdkit import Chem
import dgl
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, r2_score
import torch.nn.functional as F
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import Draw
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import pickle
import time
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  #
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  #

GNNBondFeaturizer = BaseBondFeaturizer( #11
    featurizer_funcs={'e': ConcatFeaturizer([
        bond_type_one_hot, # 4
        bond_is_conjugated_one_hot, # 1
        bond_is_in_ring_one_hot, # 1
        partial(bond_stereo_one_hot, # 5
                allowable_set=[Chem.rdchem.BondStereo.STEREONONE,
                               Chem.rdchem.BondStereo.STEREOANY,
                               Chem.rdchem.BondStereo.STEREOZ,
                               Chem.rdchem.BondStereo.STEREOE], encode_unknown=True)])})

def collate_molgraphs(data):
    # print(data[0])
    # assert len(data[0]) in [3, 4], 'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    # if len(data[0]) == 3:
    #     smiles, graphs, labels = map(list, zip(*data))
    #     masks = None
    # else:
    #     smiles, graphs, labels, masks = map(list, zip(*data))
    smiles, graphs, labels = map(list, zip(*data))

    for i in range(len(graphs)):
        # print(graphs[i])
        graphs[i] = dgl.add_self_loop(graphs[i])

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    # if masks is None:
    #     masks = torch.ones(labels.shape)
    # else:
    #     masks = torch.stack(masks, dim=0)
    # print(len(smiles), ' ', len(bg), ' ', len(labels))
    # sys.exit(0)
    return smiles, bg, labels
#
class Meter(object):
    """Track and summarize model performance on a dataset for (multi-label) prediction.

    When dealing with multitask learning, quite often we normalize the labels so they are
    roughly at a same scale. During the evaluation, we need to undo the normalization on
    the predicted labels. If mean and std are not None, we will undo the normalization.

    Currently we support evaluation with 4 metrics:

    * ``pearson r2``
    * ``mae``
    * ``rmse``
    * ``roc auc score``

    Parameters
    ----------
    mean : torch.float32 tensor of shape (T) or None.
        Mean of existing training labels across tasks if not ``None``. ``T`` for the
        number of tasks. Default to ``None`` and we assume no label normalization has been
        performed.
    std : torch.float32 tensor of shape (T)
        Std of existing training labels across tasks if not ``None``. Default to ``None``
        and we assume no label normalization has been performed.

    Examples
    --------
    Below gives a demo for a fake evaluation epoch.
    """
    def __init__(self, mean=None, std=None):
        self.mask = []
        self.y_pred = []
        self.y_true = []

        if (mean is not None) and (std is not None):
            self.mean = mean.cpu()
            self.std = std.cpu()
        else:
            self.mean = None
            self.std = None

    def update(self, y_pred, y_true, mask=None):
        """Update for the result of an iteration

        Parameters
        ----------
        y_pred : float32 tensor
            Predicted labels with shape ``(B, T)``,
            ``B`` for number of graphs in the batch and ``T`` for the number of tasks
        y_true : float32 tensor
            Ground truth labels with shape ``(B, T)``
        mask : None or float32 tensor
            Binary mask indicating the existence of ground truth labels with
            shape ``(B, T)``. If None, we assume that all labels exist and create
            a one-tensor for placeholder.
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        if mask is None:
            self.mask.append(torch.ones(self.y_pred[-1].shape))
        else:
            self.mask.append(mask.detach().cpu())

    def update_list(self, y_pred, y_true, mask=None):
        """Update for the result of an iteration

        Parameters
        ----------
        y_pred : float32 tensor
            Predicted labels with shape ``(B, T)``,
            ``B`` for number of graphs in the batch and ``T`` for the number of tasks
        y_true : float32 tensor
            Ground truth labels with shape ``(B, T)``
        mask : None or float32 tensor
            Binary mask indicating the existence of ground truth labels with
            shape ``(B, T)``. If None, we assume that all labels exist and create
            a one-tensor for placeholder.
        """
        self.y_pred.append(y_pred)
        self.y_true.append(y_true)
        if mask is None:
            self.mask.append(torch.ones(self.y_pred[-1].shape))
        else:
            self.mask.append(mask.detach().cpu())

    def _finalize(self):
        """Prepare for evaluation.

        If normalization was performed on the ground truth labels during training,
        we need to undo the normalization on the predicted labels.

        Returns
        -------
        mask : float32 tensor
            Binary mask indicating the existence of ground
            truth labels with shape (B, T), B for batch size
            and T for the number of tasks
        y_pred : float32 tensor
            Predicted labels with shape (B, T)
        y_true : float32 tensor
            Ground truth labels with shape (B, T)
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)

        if (self.mean is not None) and (self.std is not None):
            # To compensate for the imbalance between labels during training,
            # we normalize the ground truth labels with training mean and std.
            # We need to undo that for evaluation.
            y_pred = y_pred * self.std + self.mean

        return mask, y_pred, y_true

    def _reduce_scores(self, scores, reduction='none'):
        """Finalize the scores to return.

        Parameters
        ----------
        scores : list of float
            Scores for all tasks.
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        if reduction == 'none':
            return scores
        elif reduction == 'mean':
            return np.mean(scores)
        elif reduction == 'sum':
            return np.sum(scores)
        else:
            raise ValueError(
                "Expect reduction to be 'none', 'mean' or 'sum', got {}".format(reduction))

    def multilabel_score(self, score_func, reduction='none'):
        """Evaluate for multi-label prediction.

        Parameters
        ----------
        score_func : callable
            A score function that takes task-specific ground truth and predicted labels as
            input and return a float as the score. The labels are in the form of 1D tensor.
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        mask, y_pred, y_true = self._finalize()
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            task_score = score_func(task_y_true, task_y_pred)
            if task_score is not None:
                scores.append(task_score)
        return self._reduce_scores(scores, reduction)

    def pearson(self, reduction='none'):
        """Compute squared Pearson correlation coefficient."""

        def score(y_true, y_pred):
            return pearsonr(y_true.cpu().numpy(), y_pred.cpu().numpy())[0]

        return self.multilabel_score(score, reduction)
    def pearson_r2(self, reduction='none'):
        """Compute squared Pearson correlation coefficient.

        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        def score(y_true, y_pred):
            return pearsonr(y_true.numpy(), y_pred.numpy())[0] ** 2
        return self.multilabel_score(score, reduction)

    def mae(self, reduction='none'):
        """Compute mean absolute error.

        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        def score(y_true, y_pred):
            return F.l1_loss(y_true, y_pred).data.item()
        return self.multilabel_score(score, reduction)

    def mse(self, reduction='none'):
        """Compute root mean square error.

        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        def score(y_true, y_pred):
            return torch.sqrt(F.mse_loss(y_pred, y_true).cpu()).item()
        return self.multilabel_score(score, reduction)

    def roc_auc_score(self, reduction='none'):
        """Compute the area under the receiver operating characteristic curve (roc-auc score)
        for binary classification.

        ROC-AUC scores are not well-defined in cases where labels for a task have one single
        class only (e.g. positive labels only or negative labels only). In this case we will
        simply ignore this task and print a warning message.

        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks.

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        # Todo: This function only supports binary classification and we may need
        #  to support categorical classes.
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'
        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'ROC AUC score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                return roc_auc_score(y_true.long().numpy(), torch.sigmoid(y_pred).numpy())
        return self.multilabel_score(score, reduction)

    def pr_auc_score(self, reduction='none'):
        """Compute the area under the precision-recall curve (pr-auc score)
        for binary classification.

        PR-AUC scores are not well-defined in cases where labels for a task have one single
        class only (e.g. positive labels only or negative labels only). In this case, we will
        simply ignore this task and print a warning message.

        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks.

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'
        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'PR AUC score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                precision, recall, _ = precision_recall_curve(
                    y_true.long().numpy(), torch.sigmoid(y_pred).numpy())
                return auc(recall, precision)
        return self.multilabel_score(score, reduction)

    def r2(self, reduction='none'):
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'

        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'PR AUC score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                return r2_score(y_true.long().numpy(), y_pred.long().numpy())

        return self.multilabel_score(score, reduction)

    def plot_scatter(self):
        fig = plt.figure()
        fig.subplots()
        plt.scatter(self.y_true, self.y_pred, 10, label='prediction')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    def compute_metric(self, metric_name, reduction='none'):
        """Compute metric based on metric name.

        Parameters
        ----------
        metric_name : str

            * ``'r2'``: compute squared Pearson correlation coefficient
            * ``'mae'``: compute mean absolute error
            * ``'rmse'``: compute root mean square error
            * ``'roc_auc_score'``: compute roc-auc score
            * ``'pr_auc_score'``: compute pr-auc score

        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        if metric_name == 'r2':
            return self.r2(reduction)
        elif metric_name == 'pearson':
            return self.pearson(reduction)
        elif metric_name == 'mae':
            return self.mae(reduction)
        elif metric_name == 'mse':
            return self.mse(reduction)
        elif metric_name == 'roc_auc_score':
            return self.roc_auc_score(reduction)
        elif metric_name == 'pr_auc_score':
            return self.pr_auc_score(reduction)
        else:
            raise ValueError('Expect metric_name to be "r2" or "mae" or "mse" '
                             'or "roc_auc_score" or "pr_auc", got {}'.format(metric_name))


import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


class Meter_GPU(object):
    """Track and summarize model performance on a dataset for (multi-label) prediction.

    Parameters
    ----------
    mean : torch.float32 tensor of shape (T) or None.
        Mean of existing training labels across tasks if not ``None``. ``T`` for the number of tasks.
    std : torch.float32 tensor of shape (T) or None.
        Std of existing training labels across tasks if not ``None``.
    device : torch.device, optional
        The device to run the computations on (e.g., `cuda` or `cpu`). Defaults to `cuda` if available.
    """

    def __init__(self, mean=None, std=None, device=None):
        self.mask = []
        self.y_pred = []
        self.y_true = []

        # Default to GPU if available
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if (mean is not None) and (std is not None):
            self.mean = mean.to(self.device)
            self.std = std.to(self.device)
        else:
            self.mean = None
            self.std = None

    def update(self, y_pred, y_true, mask=None):
        """Update for the result of an iteration."""
        # Move tensors to device
        self.y_pred.append(y_pred.detach().to(self.device))
        self.y_true.append(y_true.detach().to(self.device))

        if mask is None:
            self.mask.append(torch.ones(self.y_pred[-1].shape, device=self.device))
        else:
            self.mask.append(mask.detach().to(self.device))

    def _finalize(self):
        """Prepare for evaluation and possibly undo normalization."""
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)

        if (self.mean is not None) and (self.std is not None):
            # Undo the normalization if mean and std are provided
            y_pred = y_pred * self.std + self.mean

        return mask, y_pred, y_true

    def _reduce_scores(self, scores, reduction='none'):
        """Finalize the scores to return."""
        if reduction == 'none':
            return scores
        elif reduction == 'mean':
            return np.mean(scores)
        elif reduction == 'sum':
            return np.sum(scores)
        else:
            raise ValueError(
                f"Expect reduction to be 'none', 'mean' or 'sum', got {reduction}")

    def multilabel_score(self, score_func, reduction='none'):
        """Evaluate multi-label predictions."""
        mask, y_pred, y_true = self._finalize()
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            task_score = score_func(task_y_true, task_y_pred)
            if task_score is not None:
                scores.append(task_score)
        return self._reduce_scores(scores, reduction)

    def pearson(self, reduction='none'):
        """Compute squared Pearson correlation coefficient."""

        def score(y_true, y_pred):
            return pearsonr(y_true.cpu().numpy(), y_pred.cpu().numpy())[0]

        return self.multilabel_score(score, reduction)

    def pearson_r2(self, reduction='none'):
        """Compute squared Pearson correlation coefficient."""

        def score(y_true, y_pred):
            return pearsonr(y_true.cpu().numpy(), y_pred.cpu().numpy())[0] ** 2

        return self.multilabel_score(score, reduction)

    def mae(self, reduction='none'):
        """Compute mean absolute error."""

        def score(y_true, y_pred):
            return F.l1_loss(y_true, y_pred).item()

        return self.multilabel_score(score, reduction)

    def mse(self, reduction='none'):
        """Compute root mean square error."""

        def score(y_true, y_pred):
            return torch.sqrt(F.mse_loss(y_pred, y_true)).item()

        return self.multilabel_score(score, reduction)

    def roc_auc_score(self, reduction='none'):
        """Compute the area under the receiver operating characteristic curve (ROC AUC score)"""
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'

        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class present for this task. ROC AUC score is not defined.')
                return None
            return roc_auc_score(y_true.long().cpu().numpy(), torch.sigmoid(y_pred).cpu().numpy())

        return self.multilabel_score(score, reduction)

    def pr_auc_score(self, reduction='none'):
        """Compute the area under the precision-recall curve (PR AUC score)."""
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'

        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class present for this task. PR AUC score is not defined.')
                return None
            precision, recall, _ = precision_recall_curve(
                y_true.long().cpu().numpy(), torch.sigmoid(y_pred).cpu().numpy())
            return auc(recall, precision)

        return self.multilabel_score(score, reduction)

    def r2(self, reduction='none'):
        """Compute R2 score (squared Pearson correlation coefficient)."""
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'

        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class present for this task. R2 score is not defined.')
                return None
            return r2_score(y_true.long().cpu().numpy(), y_pred.long().cpu().numpy())

        return self.multilabel_score(score, reduction)

    def plot_scatter(self):
        """Plot a scatter plot of predictions vs true labels."""
        fig = plt.figure()
        fig.subplots()
        plt.scatter(self.y_true, self.y_pred, 10, label='prediction')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    def compute_metric(self, metric_name, reduction='none'):
        """Compute metric based on the metric name."""
        if metric_name == 'r2':
            return self.r2(reduction)
        elif metric_name == 'pearson':
            return self.pearson_r2(reduction)
        elif metric_name == 'mae':
            return self.mae(reduction)
        elif metric_name == 'mse':
            return self.mse(reduction)
        elif metric_name == 'roc_auc_score':
            return self.roc_auc_score(reduction)
        elif metric_name == 'pr_auc_score':
            return self.pr_auc_score(reduction)
        else:
            raise ValueError(f'Expect metric_name to be "r2" or "mae" or "mse" '
                             f'or "roc_auc_score" or "pr_auc", got {metric_name}')

    def to(self, device):
        """Move all tensors to the specified device."""
        self.device = device
        self.mean = self.mean.to(device) if self.mean is not None else None
        self.std = self.std.to(device) if self.std is not None else None
        self.y_pred = [y_pred.to(device) for y_pred in self.y_pred]
        self.y_true = [y_true.to(device) for y_true in self.y_true]
        self.mask = [mask.to(device) for mask in self.mask]


def smiles_to_bigraph_(smiles, add_self_loop=False,
                      node_featurizer=None,
                      edge_featurizer=None,
                      canonical_atom_order=True,
                      explicit_hydrogens=False,
                      num_virtual_nodes=0):
    mol = Chem.MolFromSmiles(smiles)
    # Chem.SanitizeMol(mol)
    # Chem.Kekulize(mol, clearAromaticFlags=True)
    g = mol_to_bigraph(mol, add_self_loop, node_featurizer, edge_featurizer,
                          canonical_atom_order, explicit_hydrogens, num_virtual_nodes)
    # g_, wm = dgl.to_simple(g, writeback_mapping=True, return_counts=None, copy_ndata=True, copy_edata=True, aggregator='arbitrary')
    # g_ = dgl.to_bidirected(g, copy_ndata=True)
    # print(smiles, mol.GetNumBonds(), g_, wm)
    # print(mol.GetNumBonds(), g.num_edges(), g_.num_edges())
    return g

def mol_to_bigraph(mol, add_self_loop=False,
                   node_featurizer=None,
                   edge_featurizer=None,
                   canonical_atom_order=True,
                   explicit_hydrogens=False,
                   num_virtual_nodes=0):
    return mol_to_graph(mol, partial(construct_bigraph_from_mol, add_self_loop=add_self_loop),
                        node_featurizer, edge_featurizer,
                        canonical_atom_order, explicit_hydrogens, num_virtual_nodes)

def construct_bigraph_from_mol(mol, add_self_loop=False):
    g = dgl.graph(([], []), idtype=torch.int32)

    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    # Add edges
    src_list = []
    dst_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])

    if add_self_loop:
        nodes = g.nodes().tolist()
        src_list.extend(nodes)
        dst_list.extend(nodes)

    g.add_edges(torch.IntTensor(src_list), torch.IntTensor(dst_list))
    return g

def mol_to_graph(mol, graph_constructor, node_featurizer, edge_featurizer,
                 canonical_atom_order, explicit_hydrogens=False, num_virtual_nodes=0):
    """Convert an RDKit molecule object into a DGLGraph and featurize for it.

    This function can be used to construct any arbitrary ``DGLGraph`` from an
    RDKit molecule instance.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    graph_constructor : callable
        Takes an RDKit molecule as input and returns a DGLGraph
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to
        update ndata for a DGLGraph.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to
        update edata for a DGLGraph.
    canonical_atom_order : bool
        Whether to use a canonical order of atoms returned by RDKit. Setting it
        to true might change the order of atoms in the graph constructed.
    explicit_hydrogens : bool
        Whether to explicitly represent hydrogens as nodes in the graph. If True,
        it will call rdkit.Chem.AddHs(mol). If False, it will do nothing.
        Default to False.
    num_virtual_nodes : int
        The number of virtual nodes to add. The virtual nodes will be connected to
        all real nodes with virtual edges. If the returned graph has any node/edge
        feature, an additional column of binary values will be used for each feature
        to indicate the identity of virtual node/edges. The features of the virtual
        nodes/edges will be zero vectors except for the additional column. Default to 0.

    Returns
    -------
    DGLGraph or None
        Converted DGLGraph for the molecule if :attr:`mol` is valid and None otherwise.

    See Also
    --------
    mol_to_bigraph
    mol_to_complete_graph
    mol_to_nearest_neighbor_graph
    """
    if mol is None:
        print('Invalid mol found')
        return None

    # Whether to have hydrogen atoms as explicit nodes
    if explicit_hydrogens:
        mol = Chem.AddHs(mol)

    # if canonical_atom_order:
    #     new_order = rdmolfiles.CanonicalRankAtoms(mol)
    #     mol = rdmolops.RenumberAtoms(mol, new_order)
    g = graph_constructor(mol)

    if node_featurizer is not None:
        g.ndata.update(node_featurizer(mol))

    if edge_featurizer is not None:
        g.edata.update(edge_featurizer(mol))

    if num_virtual_nodes > 0:
        num_real_nodes = g.num_nodes()
        real_nodes = list(range(num_real_nodes))
        g.add_nodes(num_virtual_nodes)

        # Change Topology
        virtual_src = []
        virtual_dst = []
        for count in range(num_virtual_nodes):
            virtual_node = num_real_nodes + count
            virtual_node_copy = [virtual_node] * num_real_nodes
            virtual_src.extend(real_nodes)
            virtual_src.extend(virtual_node_copy)
            virtual_dst.extend(virtual_node_copy)
            virtual_dst.extend(real_nodes)
        g.add_edges(virtual_src, virtual_dst)

        for nk, nv in g.ndata.items():
            nv = torch.cat([nv, torch.zeros(g.num_nodes(), 1)], dim=1)
            nv[-num_virtual_nodes:, -1] = 1
            g.ndata[nk] = nv

        for ek, ev in g.edata.items():
            ev = torch.cat([ev, torch.zeros(g.num_edges(), 1)], dim=1)
            ev[-num_virtual_nodes * num_real_nodes * 2:, -1] = 1
            g.edata[ek] = ev
    return g


def show_mol_with_highlight(mol, highlight, save_path, name, length):
    img = Draw.MolToImage(mol, size=(1000, 1000), highlightAtoms=highlight)
    # img.show()
    img.save(save_path)

def moltosvg_highlight(smiles, atom_predictions, molSize=(1000, 1000), count = 0, save_path=''):
    mol = Chem.MolFromSmiles(smiles)
    cmap = cm.get_cmap('bwr')
    cmap = cm.get_cmap('Oranges')
    norm = matplotlib.colors.Normalize()
    Chem.Kekulize(mol)
    # Chem.SanitizeMol(mol)
    #
    # AllChem.UFFOptimizeMolecule(mol)
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    # atom_predictions = min_max_norm(atom_predictions)
    atom_colors = {}
    atom_colors_numpy = plt_colors.to_rgba(atom_predictions, norm=True)
    threshold = np.percentile(atom_predictions, 40)
    for i in range(len(atom_predictions)):
        if atom_predictions[i] >= threshold:
            atom_colors[i] = tuple(atom_colors_numpy[i][0][:3])
        else:
            atom_colors[i] = tuple([1, 1, 1])

    # img = Draw.MolToImage(mol, size=(1000, 1000), kekulize=True, wedgeBonds=False,
    #                       highlightAtoms=list(range(len(atom_predictions))))
    # rdDepictor.Compute2DCoords(mol)'
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0], molSize[1])
    drawer.DrawMolecule(mol,
                        highlightAtoms= range(len(atom_predictions)),
                        highlightAtomColors=atom_colors,
                        highlightBonds=[],
                        )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    with open(os.path.join(save_path, str(count) + '_important_nodes.png'), 'wb') as f:
        f.write(svg)

def mol_highlight_atoms_bonds(smiles, atom_predictions, edge_weight, starts, ends, molSize=(2000, 2000), count = 0, save_path=''):
    mol = Chem.MolFromSmiles(smiles)
    cmap = cm.get_cmap('bwr')
    cmap = cm.get_cmap('Oranges')
    norm = matplotlib.colors.Normalize()
    atom_colors = {}
    cmap = plt.cm.YlGn
    norm = matplotlib.colors.Normalize(vmin=atom_predictions.min(), vmax=atom_predictions.max())
    atom_colors_numpy = cmap(norm(atom_predictions))
    threshold = np.percentile(atom_predictions, 0)
    for i in range(len(atom_predictions)):
        if atom_predictions[i] >= threshold:
            atom_colors[i] = tuple(atom_colors_numpy[i][0][:3])
        else:
            atom_colors[i] = tuple([1, 1, 1])

    zipped_bonds = {(a, b) : c for a, b, c, in zip(starts, ends, edge_weight)}
    bonds = mol.GetBonds()
    bonds_weight = []
    for idx, bond in enumerate(bonds):
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bonds_weight.append(zipped_bonds[tuple((start, end))])
    bonds_weight = np.array(bonds_weight)
    cmap = plt.cm.Wistia
    cmap = plt.cm.YlGn
    # print(bonds_weight.min(), bonds_weight.max())
    # norm = matplotlib.colors.Normalize(vmin=bonds_weight.min(), vmax=bonds_weight.max())
    # normnorm = norm(bonds_weight)
    # print(normnorm.min(), normnorm.max())
    edge_colors_numpy = cmap(bonds_weight)
    edge_colors = {}
    threshold = np.percentile(bonds_weight, 0)
    for i in range(len(bonds_weight)):
        if bonds_weight[i] >= threshold:
            edge_colors[i] = tuple(edge_colors_numpy[i][0][:3])
        else:
            edge_colors[i] = tuple([1, 1, 1])

    # img = Draw.MolToImage(mol, size=(1000, 1000), kekulize=True, wedgeBonds=False,
    #                       highlightAtoms=list(range(len(atom_predictions))))
    # rdDepictor.Compute2DCoords(mol)'
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0], molSize[1])
    drawer.DrawMolecule(mol,
                        highlightAtoms= range(len(atom_predictions)),
                        highlightAtomColors=atom_colors,
                        # highlightBonds=[],
                        highlightBonds=range(len(bonds_weight)),
                        highlightBondColors=edge_colors,
                        )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    with open(os.path.join(save_path, str(count) + '_important_nodes.png'), 'wb') as f:
        f.write(svg)


def draw_molecule(smiles, save_path, file_name, size=(1000, 1000)):
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
    Chem.SanitizeMol(mol)
    AllChem.EmbedMolecule(mol)
    smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size)
    img.save(os.path.join(save_path, file_name + '.jpg'))

def addnote(mol):
    for atom in mol.GetAtoms():
        atom.SetProp('atomNote', str(atom.GetIdx())) #'molAtomMapNumber'
    return mol

def addnote_order(mol, order):
    idx = 0
    order = np.array(order)
    for atom in mol.GetAtoms():
        atom.SetProp('atomNote', str(list(np.where(order==idx))[0][0])) #'molAtomMapNumber'
        # print(order[idx])
        idx += 1

    return mol

def draw_molecules(smiles, save_path, file_name, legend, size=(1000, 1000)):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    length = int(len(mols))
    for i in range(length):
        mol = mols[i]
        try:
            Chem.Kekulize(mol)
            Chem.SanitizeMol(mol)
            AllChem.EmbedMolecule(mol)
            smile = Chem.MolToSmiles(mol)
            mols[i] = Chem.MolFromSmiles(smile)
        except:
            continue
        mols[i] = addnote(mols[i])
    img = Draw.MolsToGridImage(mols, molsPerRow=7, subImgSize=size, legends=[legend + str(i) for i in range(1, length + 1)], legendFontSize=50)
    img.save(os.path.join(save_path, file_name + '.jpg'))
