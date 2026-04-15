#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: Define the GCN-based graph neural network predictor.
Usage: Imported by training, prediction, and analysis scripts.
"""

import torch.nn as nn
from dgllife.model.model_zoo.mlp_predictor import MLPPredictor
from dgllife.model.gnn.gcn import GCN
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# pylint: disable=W0221
class GCNPredictor_(nn.Module):
    """GCN-based model for regression and classification on graphs.

    GCN is introduced in `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__. This model is based on GCN and can be used
    for regression and classification on graphs.

    After updating node representations, we perform a weighted sum with learnable
    weights and max pooling on them and concatenate the output of the two operations,
    which is then fed into an MLP for final prediction.

    For classification tasks, the output will be logits, i.e.
    values before sigmoid or softmax.

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of node representations after the i-th GCN layer.
        ``len(hidden_feats)`` equals the number of GCN layers. By default, we use
        ``[64, 64]``.
    gnn_norm : list of str
        ``gnn_norm[i]`` gives the message passing normalizer for the i-th GCN layer, which
        can be `'right'`, `'both'` or `'none'`. The `'right'` normalizer divides the aggregated
        messages by each node's in-degree. The `'both'` normalizer corresponds to the symmetric
        adjacency normalization in the original GCN paper. The `'none'` normalizer simply sums
        the messages. ``len(gnn_norm)`` equals the number of GCN layers. By default, we use
        ``['none', 'none']``.
    activation : list of activation functions or None
        If None, no activation will be applied. If not None, ``activation[i]`` gives the
        activation function to be used for the i-th GCN layer. ``len(activation)`` equals
        the number of GCN layers. By default, ReLU is applied for all GCN layers.
    residual : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GCN layer.
        ``len(residual)`` equals the number of GCN layers. By default, residual connection
        is performed for each GCN layer.
    batchnorm : list of bool
        ``batchnorm[i]`` decides if batch normalization is to be applied on the output of
        the i-th GCN layer. ``len(batchnorm)`` equals the number of GCN layers. By default,
        batch normalization is applied for all GCN layers.
    dropout : list of float
        ``dropout[i]`` decides the dropout probability on the output of the i-th GCN layer.
        ``len(dropout)`` equals the number of GCN layers. By default, no dropout is
        performed for all layers.
    classifier_hidden_feats : int
        (Deprecated, see ``predictor_hidden_feats``) Size of hidden graph representations
        in the classifier. Default to 128.
    classifier_dropout : float
        (Deprecated, see ``predictor_dropout``) The probability for dropout in the classifier.
        Default to 0.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    predictor_hidden_feats : int
        Size for hidden representations in the output MLP predictor. Default to 128.
    predictor_dropout : float
        The probability for dropout in the output MLP predictor. Default to 0.
    """
    def __init__(self, in_feats, hidden_feats=None, gnn_norm=None, activation=None,
                 residual=None, batchnorm=None, dropout=None, classifier_hidden_feats=128,
                 classifier_dropout=0., n_tasks=1, predictor_hidden_feats=128,
                 predictor_dropout=0.):
        super(GCNPredictor_, self).__init__()

        if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        if predictor_dropout == 0. and classifier_dropout != 0.:
            print('classifier_dropout is deprecated and will be removed in the future, '
                  'use predictor_dropout instead')
            predictor_dropout = classifier_dropout

        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       gnn_norm=gnn_norm,
                       activation=activation,
                       residual=residual,
                       batchnorm=batchnorm,
                       dropout=dropout)
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.predict = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
                                    n_tasks, predictor_dropout)

    def forward(self, graph, feat, eweight = None):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match
              in_feats in initialization

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            * Predictions on graphs
            * B for the number of graphs in the batch
        """
        node_feats = self.gnn(graph, feat)
        graph_feats = self.readout(graph, node_feats)
        return self.predict(graph_feats)

class GCNLayer_(nn.Module):
    r"""Single GCN layer from `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    out_feats : int
        Number of output node features.
    gnn_norm : str
        The message passing normalizer, which can be `'right'`, `'both'` or `'none'`. The
        `'right'` normalizer divides the aggregated messages by each node's in-degree.
        The `'both'` normalizer corresponds to the symmetric adjacency normalization in
        the original GCN paper. The `'none'` normalizer simply sums the messages.
        Default to be 'none'.
    activation : activation function
        Default to be None.
    residual : bool
        Whether to use residual connection, default to be True.
    batchnorm : bool
        Whether to use batch normalization on the output,
        default to be True.
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    allow_zero_in_degree: bool
        Whether to allow zero in degree nodes in graph. Defaults to False.
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        gnn_norm="none",
        activation=None,
        residual=True,
        batchnorm=True,
        dropout=0.0,
        allow_zero_in_degree=False,
    ):
        super(GCNLayer_, self).__init__()

        self.activation = activation
        self.graph_conv = GraphConv(
            in_feats=in_feats,
            out_feats=out_feats,
            norm=gnn_norm,
            activation=activation,
            allow_zero_in_degree=allow_zero_in_degree,
        )
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.graph_conv.reset_parameters()
        if self.residual:
            self.res_connection.reset_parameters()
        if self.bn:
            self.bn_layer.reset_parameters()

    def forward(self, g, feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match in_feats in initialization

        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output node feature size, which must match out_feats in initialization
        """
        new_feats = self.graph_conv(g, feats)
        if self.residual:
            res_feats = self.activation(self.res_connection(feats))
            new_feats = new_feats + res_feats
        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats


class GCN_(nn.Module):
    r"""GCN from `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of node representations after the i-th GCN layer.
        ``len(hidden_feats)`` equals the number of GCN layers.  By default, we use
        ``[64, 64]``.
    gnn_norm : list of str
        ``gnn_norm[i]`` gives the message passing normalizer for the i-th GCN layer, which
        can be `'right'`, `'both'` or `'none'`. The `'right'` normalizer divides the aggregated
        messages by each node's in-degree. The `'both'` normalizer corresponds to the symmetric
        adjacency normalization in the original GCN paper. The `'none'` normalizer simply sums
        the messages. ``len(gnn_norm)`` equals the number of GCN layers. By default, we use
        ``['none', 'none']``.
    activation : list of activation functions or None
        If not None, ``activation[i]`` gives the activation function to be used for
        the i-th GCN layer. ``len(activation)`` equals the number of GCN layers.
        By default, ReLU is applied for all GCN layers.
    residual : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GCN layer.
        ``len(residual)`` equals the number of GCN layers. By default, residual connection
        is performed for each GCN layer.
    batchnorm : list of bool
        ``batchnorm[i]`` decides if batch normalization is to be applied on the output of
        the i-th GCN layer. ``len(batchnorm)`` equals the number of GCN layers. By default,
        batch normalization is applied for all GCN layers.
    dropout : list of float
        ``dropout[i]`` decides the dropout probability on the output of the i-th GCN layer.
        ``len(dropout)`` equals the number of GCN layers. By default, no dropout is
        performed for all layers.
    allow_zero_in_degree: bool
        Whether to allow zero in degree nodes in graph for all layers. By default, will not
        allow zero in degree nodes.
    """

    def __init__(
        self,
        in_feats,
        hidden_feats=None,
        gnn_norm=None,
        activation=None,
        residual=None,
        batchnorm=None,
        dropout=None,
        allow_zero_in_degree=None,
    ):
        super(GCN_, self).__init__()

        if hidden_feats is None:
            hidden_feats = [64, 64]

        n_layers = len(hidden_feats)
        if gnn_norm is None:
            gnn_norm = ["none" for _ in range(n_layers)]
        if activation is None:
            activation = [F.relu for _ in range(n_layers)]
        if residual is None:
            residual = [True for _ in range(n_layers)]
        if batchnorm is None:
            batchnorm = [True for _ in range(n_layers)]
        if dropout is None:
            dropout = [0.0 for _ in range(n_layers)]
        lengths = [
            len(hidden_feats),
            len(gnn_norm),
            len(activation),
            len(residual),
            len(batchnorm),
            len(dropout),
        ]
        assert len(set(lengths)) == 1, (
            "Expect the lengths of hidden_feats, gnn_norm, "
            "activation, residual, batchnorm and dropout to "
            "be the same, got {}".format(lengths)
        )

        self.hidden_feats = hidden_feats
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(
                GCNLayer_(
                    in_feats,
                    hidden_feats[i],
                    gnn_norm[i],
                    activation[i],
                    residual[i],
                    batchnorm[i],
                    dropout[i],
                    allow_zero_in_degree=allow_zero_in_degree,
                )
            )
            in_feats = hidden_feats[i]

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, feats, molecule_id, train_name, save_path, order = []):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which equals in_feats in initialization

        Returns
        -------
        feats : FloatTensor of shape (N, M2)
            * N is the total number of nodes in the batch of graphs
            * M2 is the output node representation size, which equals
              hidden_sizes[-1] in initialization.
        """
        # save_path = r'C:\Users\zyb\Desktop\GNN_QY_TPAS\results\gcn_heatmap'
        layer_count = 0
        heatmap_(feats, order, save_path, layer_count, molecule_id, train_name)
        layer_count += 1
        for gnn in self.gnn_layers:
            feats = gnn(g, feats)
            heatmap_(feats, order, save_path, layer_count, molecule_id, train_name)
            layer_count += 1
        return feats

def heatmap_(node_feats, order, save_path, layer_count, molecule_id, train_name):
    corr = np.array(node_feats.cpu().detach().numpy())
    if int(len(order)) > 0:
        corr = corr[order, :]
    dt = pd.DataFrame(corr.T)
    corr = np.array(dt.corr(method='pearson'))
    corr = np.around(corr, decimals=2)
    layer_count += 1
    # df = pd.DataFrame(corr)
    # df.to_csv(os.path.join(save_path, str(molecule_id) + '_' + str(train_name) + '_layer_' + str(layer_count) + '.csv'), index=False)
    get_map(corr, str(molecule_id) + '_' + str(train_name) + '_layer_' + str(layer_count), save_path)


def get_map(confusion, file_name, img_save_path):
    # print(confusion.shape)
    #
    fig = plt.figure(figsize=(20, 16), dpi=500)
    ax = fig.gca()
    sns.heatmap(confusion, cmap="YlGnBu", annot=False, ax=ax, square=True,)
    # sns.heatmap(confusion, cmap="coolwarm", annot=False, ax=ax, square=True,)
                     # annot_kws={"size": -1})
    # plt.imshow(confusion, cmap=plt.cm.Blues)
    #
    #
    # indices = range(len(confusion[0]))
    # idx = range(len(confusion[0]))
    # plt.xticks(indices, idx, fontsize=15)
    # plt.yticks(indices, idx, fontsize=15)
    #
    # plt.colorbar()
    #
    # font = {'family' : 'Times New Roman',
    # 'weight' : 'normal',
    # 'size'   : 30,
    # }
    #
    # plt.xlabel('Predict', font)
    # plt.ylabel('True', font)
    # plt.title(file, font)
    #
    # plt.rcParams['font.sans-serif']=['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    #
    #
    #
    #
    # #         plt.text(first_index, second_index, confusion[second_index][first_index], fontsize=30)
    #
    plt.savefig(os.path.join(img_save_path, file_name + '.png'), bbox_inches = 'tight')
    plt.clf()
    plt.close()

class GCNPredictor_heatmap_(nn.Module):
    """GCN-based model for regression and classification on graphs.

    GCN is introduced in `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__. This model is based on GCN and can be used
    for regression and classification on graphs.

    After updating node representations, we perform a weighted sum with learnable
    weights and max pooling on them and concatenate the output of the two operations,
    which is then fed into an MLP for final prediction.

    For classification tasks, the output will be logits, i.e.
    values before sigmoid or softmax.

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of node representations after the i-th GCN layer.
        ``len(hidden_feats)`` equals the number of GCN layers. By default, we use
        ``[64, 64]``.
    gnn_norm : list of str
        ``gnn_norm[i]`` gives the message passing normalizer for the i-th GCN layer, which
        can be `'right'`, `'both'` or `'none'`. The `'right'` normalizer divides the aggregated
        messages by each node's in-degree. The `'both'` normalizer corresponds to the symmetric
        adjacency normalization in the original GCN paper. The `'none'` normalizer simply sums
        the messages. ``len(gnn_norm)`` equals the number of GCN layers. By default, we use
        ``['none', 'none']``.
    activation : list of activation functions or None
        If None, no activation will be applied. If not None, ``activation[i]`` gives the
        activation function to be used for the i-th GCN layer. ``len(activation)`` equals
        the number of GCN layers. By default, ReLU is applied for all GCN layers.
    residual : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GCN layer.
        ``len(residual)`` equals the number of GCN layers. By default, residual connection
        is performed for each GCN layer.
    batchnorm : list of bool
        ``batchnorm[i]`` decides if batch normalization is to be applied on the output of
        the i-th GCN layer. ``len(batchnorm)`` equals the number of GCN layers. By default,
        batch normalization is applied for all GCN layers.
    dropout : list of float
        ``dropout[i]`` decides the dropout probability on the output of the i-th GCN layer.
        ``len(dropout)`` equals the number of GCN layers. By default, no dropout is
        performed for all layers.
    classifier_hidden_feats : int
        (Deprecated, see ``predictor_hidden_feats``) Size of hidden graph representations
        in the classifier. Default to 128.
    classifier_dropout : float
        (Deprecated, see ``predictor_dropout``) The probability for dropout in the classifier.
        Default to 0.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    predictor_hidden_feats : int
        Size for hidden representations in the output MLP predictor. Default to 128.
    predictor_dropout : float
        The probability for dropout in the output MLP predictor. Default to 0.
    """
    def __init__(self, in_feats, hidden_feats=None, gnn_norm=None, activation=None,
                 residual=None, batchnorm=None, dropout=None, classifier_hidden_feats=128,
                 classifier_dropout=0., n_tasks=1, predictor_hidden_feats=128,
                 predictor_dropout=0.):
        super(GCNPredictor_heatmap_, self).__init__()

        if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        if predictor_dropout == 0. and classifier_dropout != 0.:
            print('classifier_dropout is deprecated and will be removed in the future, '
                  'use predictor_dropout instead')
            predictor_dropout = classifier_dropout

        self.gnn = GCN_(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       gnn_norm=gnn_norm,
                       activation=activation,
                       residual=residual,
                       batchnorm=batchnorm,
                       dropout=dropout)
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.predict = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
                                    n_tasks, predictor_dropout)

    def forward(self, graph, feat, molecule_id, train_name, save_path, order = [], eweight = None):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match
              in_feats in initialization

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            * Predictions on graphs
            * B for the number of graphs in the batch
        """
        node_feats = self.gnn(graph, feat, molecule_id, train_name, save_path, order)
        graph_feats = self.readout(graph, node_feats)
        return self.predict(graph_feats)
