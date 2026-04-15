#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: Define DeepTPACS models and helper utilities for visualization.
Usage: Imported by training, prediction, heatmap, and embedding scripts.
"""

import os

import dgl.function as fn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout


__all__ = [
    'DeepTPACSPredictor_',
    'DeepTPACSPredictorFusion_',
    'DeepTPACSPredictor_heatmap_',
]


# pylint: disable=W0221
class DeepTPACSPredictor_(nn.Module):
    """
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge features.
    num_layers : int
        Number of GNN layers. Default to 2.
    num_timesteps : int
        Times of updating the graph representations with GRU. Default to 2.
    graph_feat_size : int
        Size for the learned graph representations. Default to 200.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    dropout : float
        Probability for performing the dropout. Default to 0.
    """

    def __init__(
        self,
        node_feat_size,
        edge_feat_size,
        num_layers=2,
        num_timesteps=2,
        graph_feat_size=200,
        n_tasks=1,
        dropout=0.0,
    ):
        super().__init__()

        self.gnn = AttentiveFPGNN(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            num_layers=num_layers,
            graph_feat_size=graph_feat_size,
            dropout=dropout,
        )
        self.readout = AttentiveFPReadout(
            feat_size=graph_feat_size,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, n_tasks),
        )

    def forward(
        self,
        graph,
        feat,
        edge_feats,
        get_node_weight=False,
        readout=False,
        embed=False,
        edge_weight=None,
    ):
        """Graph-level regression/soft classification."""
        node_feats = self.gnn(graph, feat, edge_feats)
        if embed is True:
            return node_feats

        if get_node_weight:
            g_feats, node_weights = self.readout(graph, node_feats, get_node_weight)
            if readout:
                return self.predict(g_feats), node_weights, g_feats
            return self.predict(g_feats), node_weights

        g_feats = self.readout(graph, node_feats, get_node_weight)
        if readout:
            return self.predict(g_feats), g_feats
        return self.predict(g_feats)


class DeepTPACSPredictorFusion_(nn.Module):
    """DeepTPACS with residual descriptor correction."""

    def __init__(
        self,
        node_feat_size,
        edge_feat_size,
        desc_dim,
        num_layers=2,
        num_timesteps=2,
        graph_feat_size=200,
        n_tasks=1,
        dropout=0.0,
        fusion_hidden_feats=256,
    ):
        super().__init__()

        self.gnn = AttentiveFPGNN(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            num_layers=num_layers,
            graph_feat_size=graph_feat_size,
            dropout=dropout,
        )
        self.readout = AttentiveFPReadout(
            feat_size=graph_feat_size,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, n_tasks),
        )
        self.delta = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(desc_dim, fusion_hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_feats, n_tasks),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, graph, feat, edge_feats, desc):
        node_feats = self.gnn(graph, feat, edge_feats)
        g_feats = self.readout(graph, node_feats, get_node_weight=False)
        y_gnn = self.predict(g_feats)
        delta = self.delta(desc)
        return y_gnn + self.alpha * delta


class DeepTPACSPredictor_heatmap_(nn.Module):
    """
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge features.
    num_layers : int
        Number of GNN layers. Default to 2.
    num_timesteps : int
        Times of updating the graph representations with GRU. Default to 2.
    graph_feat_size : int
        Size for the learned graph representations. Default to 200.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    dropout : float
        Probability for performing the dropout. Default to 0.
    """

    def __init__(
        self,
        node_feat_size,
        edge_feat_size,
        num_layers=2,
        num_timesteps=2,
        graph_feat_size=200,
        n_tasks=1,
        dropout=0.0,
    ):
        super().__init__()

        self.gnn = DeepTPACSFPGNN_(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            num_layers=num_layers,
            graph_feat_size=graph_feat_size,
            dropout=dropout,
        )
        self.readout = AttentiveFPReadout(
            feat_size=graph_feat_size,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, n_tasks),
        )

    def forward(
        self,
        graph,
        feat,
        edge_feats,
        get_node_weight=False,
        readout=False,
        heatmap=False,
        molecule_id=None,
        train_name=None,
        img_save_path='',
        order=[],
    ):
        """Graph-level regression/soft classification."""
        node_feats = self.gnn(graph, feat, edge_feats, heatmap, molecule_id, train_name, img_save_path, order)
        if get_node_weight:
            g_feats, node_weights = self.readout(graph, node_feats, get_node_weight)
            if readout:
                return self.predict(g_feats), node_weights, g_feats
            return self.predict(g_feats), node_weights

        g_feats = self.readout(graph, node_feats, get_node_weight)
        if readout:
            return self.predict(g_feats), g_feats
        return self.predict(g_feats)


class DeepTPACSGRU1(nn.Module):
    """
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """

    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super().__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size),
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.edge_transform[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, edge_logits, edge_feats, node_feats):
        """Update node representations."""
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))


class DeepTPACSGRU2(nn.Module):
    """Update node features with attention and GRU.

    This will be used in GNN layers for updating node representations.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """

    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super().__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size),
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, edge_logits, node_feats):
        """Update node representations."""
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.u_mul_e('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))


class GetContext(nn.Module):
    """
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    graph_feat_size : int
        Size of the learned graph representation (molecular fingerprint).
    dropout : float
        The probability for performing dropout.
    """

    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super().__init__()

        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU(),
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU(),
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU(),
        )
        self.attentive_gru = DeepTPACSGRU1(graph_feat_size, graph_feat_size, graph_feat_size, dropout)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node[0].reset_parameters()
        self.project_edge1[0].reset_parameters()
        self.project_edge2[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def apply_edges1(self, edges):
        """Edge feature update."""
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        """Edge feature update."""
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        """Incorporate edge features and update node representations."""
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        g.edata['he'] = edge_feats

        g.apply_edges(self.apply_edges1)
        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        g.apply_edges(self.apply_edges2)
        logits = self.project_edge2(g.edata['he2'])

        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])


class GNNLayer(nn.Module):
    """GNNLayer for updating node features.

    This layer performs message passing over node representations and update them.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    graph_feat_size : int
        Size for the graph representations to be computed.
    dropout : float
        The probability for performing dropout.
    """

    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super().__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU(),
        )
        self.attentive_gru = DeepTPACSGRU2(node_feat_size, graph_feat_size, dropout)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_edge[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def apply_edges(self, edges):
        """Edge feature generation."""
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        """Perform message passing and update node representations."""
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])

        return self.attentive_gru(g, logits, node_feats)


def get_map(confusion, file_name, img_save_path, annot=False, cbar_labelsize=20):
    fig = plt.figure(figsize=(20, 16), dpi=500)
    ax = fig.gca()
    sns.heatmap(
        confusion,
        cmap='YlGnBu', #Blues
        annot=annot,
        ax=ax,
        square=True,
        fmt='.1f',
        annot_kws={'size': 8},
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=cbar_labelsize)
    plt.savefig(os.path.join(img_save_path, file_name + '.png'), bbox_inches='tight')
    plt.clf()
    plt.close()


def heatmap_(node_feats, order, save_path, layer_count, molecule_id, train_name):
    corr = np.array(node_feats.cpu().detach().numpy())
    if int(len(order)) > 0:
        corr = corr[order, :]
    dt = pd.DataFrame(corr.T)
    corr = np.array(dt.corr(method='pearson'))
    corr = np.around(corr, decimals=2)
    layer_count += 1
    get_map(
        corr,
        str(molecule_id) + '_' + str(train_name) + '_layer_' + str(layer_count),
        save_path,
    )


class DeepTPACSFPGNN_(nn.Module):
    """
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge features.
    num_layers : int
        Number of GNN layers. Default to 2.
    graph_feat_size : int
        Size for the graph representations to be computed. Default to 200.
    dropout : float
        The probability for performing dropout. Default to 0.
    """

    def __init__(
        self,
        node_feat_size,
        edge_feat_size,
        num_layers=2,
        graph_feat_size=200,
        dropout=0.0,
    ):
        super().__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.init_context.reset_parameters()
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, node_feats, edge_feats, heatmap, molecule_id, train_name, img_save_path, order):
        """Performs message passing and updates node representations."""
        if not heatmap:
            node_feats = self.init_context(g, node_feats, edge_feats)
            for gnn in self.gnn_layers:
                node_feats = gnn(g, node_feats)
            return node_feats

        save_path = img_save_path
        layer_count = 0
        heatmap_(node_feats, order, save_path, layer_count, molecule_id, train_name)
        layer_count += 1

        node_feats = self.init_context(g, node_feats, edge_feats)
        heatmap_(node_feats, order, save_path, layer_count, molecule_id, train_name)
        layer_count += 1

        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
            if layer_count < 3:
                heatmap_(node_feats, order, save_path, layer_count, molecule_id, train_name)
                layer_count += 1
        return node_feats
