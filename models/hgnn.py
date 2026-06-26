import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter


class HypergraphConv(MessagePassing):
    """Hypergraph convolution with optional real-valued incidence weights.

    Implements
    X' = D^{-1} H W B^{-1} H^T X Theta
  where H may contain non-negative real entries passed as ``edge_entry_weight``.
    When ``edge_entry_weight`` is None, each incidence is treated as weight 1.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.xavier_uniform_(self.lin.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: Tensor,
        hyperedge_index: Tensor,
        hyperedge_weight: Optional[Tensor] = None,
        edge_entry_weight: Optional[Tensor] = None,
        num_edges: Optional[int] = None,
    ) -> Tensor:
        num_nodes = x.size(0)
        if num_edges is None:
            num_edges = 0
            if hyperedge_index.numel() > 0:
                num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        vertex, edges = hyperedge_index[0], hyperedge_index[1]
        if edge_entry_weight is None:
            w = x.new_ones(hyperedge_index.size(1))
        else:
            w = edge_entry_weight.to(x.dtype).clamp_min(1e-12)

        x = self.lin(x)

        # B_e = sum_v H_{ve}; D_v = sum_e H_{ve} W_e
        B = scatter(w, edges, dim=0, dim_size=num_edges, reduce='sum').clamp_min(1e-12)
        D = scatter(hyperedge_weight[edges] * w, vertex, dim=0, dim_size=num_nodes, reduce='sum')
        D = D.clamp_min(1e-12)

        # PyG propagate 要求 norm 为 [num_edges] / [num_nodes]，逐条目权重在 message 中乘入。
        out = self.propagate(
            hyperedge_index, x=x, norm=1.0 / B, w=w, size=(num_nodes, num_edges),
        )
        out = self.propagate(
            hyperedge_index.flip([0]), x=out, norm=1.0 / D,
            w=hyperedge_weight[edges] * w, size=(num_edges, num_nodes),
        )

        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j: Tensor, norm_i: Tensor, w: Tensor) -> Tensor:
        return (w * norm_i).view(-1, 1) * x_j


class HGNNEncoder(nn.Module):
    def __init__(self, node_dim: int, emb_dim: int, num_layers: int, dropout: float = 0.4,
        activation: nn.Module = nn.LeakyReLU(), use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.node_encoder = nn.Linear(node_dim, emb_dim)
        self.layers = nn.ModuleList()
        self.activation = activation
        self.drop = nn.Dropout(dropout)
        for _ in range(num_layers):
            self.layers.append(HypergraphConv(emb_dim, emb_dim))
        self.norms = nn.ModuleList([
            nn.LayerNorm(emb_dim) for _ in range(num_layers)
        ])

    def forward(
        self,
        X: torch.Tensor,
        hyperedge_index: torch.Tensor,
        edge_entry_weight: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, None]:
        X = self.node_encoder(X)
        for layer, norm in zip(self.layers, self.norms):
            residual_input = X
            X = layer(X, hyperedge_index, edge_entry_weight=edge_entry_weight)
            X = norm(X)
            X = self.activation(X)
            if self.use_residual:
                X = X + residual_input
            X = self.drop(X)
        return X, None
