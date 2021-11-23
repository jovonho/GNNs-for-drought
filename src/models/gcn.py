"""
    Adapted from
    https://github.com/salvaRC/Graphino/blob/master/graphino/structure_learner.py
    https://github.com/salvaRC/Graphino/blob/master/graphino/GCN/graph_conv_layer.py
    https://github.com/salvaRC/Graphino/blob/master/graphino/GCN/GCN_model.py
"""

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.models.mlp import MLP


class AdjacencyLearner(nn.Module):
    """
    Static features here refers to the feature matrix from which we will learn A.
    """

    def __init__(
        self,
        num_nodes: int,
        max_num_edges: int,
        dim: int,
        static_features: torch.Tensor,
        device: str = "cpu",
        alpha1: float = 0.1,
        alpha2: float = 2.0,
        self_loops: bool = True,
    ):
        super().__init__()
        if static_features is None:
            raise ValueError("Please give static node features (e.g. part of the timeseries)")
        self.num_nodes = num_nodes
        xd = static_features.shape[1]
        self.lin1 = nn.Linear(xd, dim)
        self.lin2 = nn.Linear(xd, dim)

        self.static_features = (
            static_features
            if isinstance(static_features, torch.Tensor)
            else torch.as_tensor(static_features)
        )
        self.static_features = self.static_features.float().to(device)

        self.device = device
        self.dim = dim
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.num_edges = max_num_edges
        self.self_loops = self_loops
        self.diag = torch.eye(self.num_nodes).bool().to(device)

    def forward(self):
        M1 = torch.tanh(self.alpha1 * self.lin1(self.static_features))
        M2 = torch.tanh(self.alpha1 * self.lin2(self.static_features))

        A = torch.sigmoid(self.alpha2 * M1 @ M2.T)
        # TODO: Could we simply add an A representing the local connectivity here?
        A = A.flatten()
        mask = torch.zeros(self.num_nodes * self.num_nodes).to(self.device)
        # Get the strongest weight's indices
        _, strongest_idxs = torch.topk(A, self.num_edges)
        mask[strongest_idxs] = 1
        A = A * mask
        A = A.reshape((self.num_nodes, self.num_nodes))
        if self.self_loops:
            A[self.diag] = A[self.diag].clamp(min=0.5)

        return A


class GraphConvolution(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        residual: bool = False,
        batch_norm: bool = False,
        activation: F = F.elu,
        dropout: float = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        self.residual = residual

        if self.in_features != self.out_features:
            self.residual = False

        self.batchnorm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self._norm = False

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, A):
        support = torch.matmul(input, self.weight)  # (batch-size, #nodes, #out-dim)
        node_repr = torch.matmul(A, support)  # (batch-size, #nodes, #out-dim)

        if self.bias is not None:
            node_repr = node_repr + self.bias

        if self.batchnorm is not None:
            node_repr = node_repr.transpose(1, 2)  # --> (batch-size, #out-dim, #nodes)
            # batch normalization over feature/node embedding dim.
            node_repr = self.batchnorm(node_repr)
            node_repr = node_repr.transpose(1, 2)

        node_repr = self.activation(node_repr)

        if self.residual:
            node_repr = input + node_repr  # residual connection

        node_repr = self.dropout(node_repr)
        return node_repr

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GCN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_dim: int,
        num_gcn_layers: int,
        adj_learn_features: torch.Tensor,
        adj_learn_dim: int,
        dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        num_nodes: int = 1575,
        A: np.ndarray = None,
        device: str = "cpu",
        verbose: bool = False,
    ):
        super().__init__()
        self.L = num_gcn_layers
        self.out_dim = self.MLP_input_dim = out_dim
        self.batch_norm = True
        self.graph_pooling = "mean"
        self.jumping_knowledge = True

        conv_kwargs = {
            "activation": F.elu,
            "batch_norm": True,
            "residual": True,
            "dropout": dropout,
        }

        layers = [GraphConvolution(in_features, hidden_dim, **conv_kwargs)]
        layers += [
            GraphConvolution(hidden_dim, hidden_dim, **conv_kwargs) for _ in range(self.L - 2)
        ]
        layers.append(GraphConvolution(hidden_dim, self.out_dim, **conv_kwargs))

        self.layers = nn.ModuleList(layers)

        # See Appendix A for details
        if self.jumping_knowledge:
            self.MLP_input_dim = self.MLP_input_dim + hidden_dim * (self.L - 1)

        # Set the output dimension of MLP to num_nodes to predict a value for each node
        self.MLP_layer = MLP(
            self.MLP_input_dim, num_nodes, [num_nodes], batch_norm=True, dropout_val=mlp_dropout
        )

        if A is None:
            if verbose:
                print("We will be learning the adjancency matrix")
            self.A, self.learn_adj = None, True
            # Cap the max number of edges.
            max_num_edges = 8 * num_nodes

            self.adj_learner = AdjacencyLearner(
                num_nodes,
                max_num_edges,
                dim=adj_learn_dim,
                device=device,
                static_features=adj_learn_features,
                alpha1=0.1,
                alpha2=2.0,
                self_loops=True,
            )
        else:
            print("Using a static Adjacency matrix")
            self.learn_adj = False
            self.A = torch.from_numpy(A).float().to(device)

        if verbose:
            print([x for x in self.layers])

    def forward(self, input, readout=True):
        if self.learn_adj:
            # Generate an adjacency matrix/connectivity structure for
            # the graph convolutional forward pass
            self.A = self.adj_learner.forward()

        # GCN forward pass --> Generate node embeddings
        embeddings = self.layers[0](input, self.A)  # shape (batch-size, #nodes, #features)
        X_all_embeddings = embeddings.clone()
        for conv in self.layers[1:]:
            embeddings = conv(embeddings, self.A)
            if self.jumping_knowledge:
                X_all_embeddings = torch.cat((X_all_embeddings, embeddings), dim=2)

        final_embs = X_all_embeddings if self.jumping_knowledge else embeddings

        # Graph pooling, e.g. take the mean over all node embeddings (dimension=1)
        # (batch-size, out-dim)
        # TODO: Experiment with other pooling functions like sum or max or combinations
        g_emb = torch.mean(final_embs, dim=1)

        # Pass the final graph embedding to an MLP to generate predictions
        out = self.MLP_layer.forward(g_emb).squeeze(1) if readout else g_emb
        return out
