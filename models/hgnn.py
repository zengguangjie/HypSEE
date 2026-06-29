import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import NamedTuple, Optional

from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter


class BatchedDenseHypergraph(NamedTuple):
    """Sparse batched hypergraph built from dense incidence ``H`` (bs, n, e)."""

    hyperedge_index: Tensor
    edge_entry_weight: Tensor
    num_nodes: int
    num_edges: int
    batch_size: int
    num_nodes_per_graph: int
    num_edges_per_graph: int


def dense_incidence_to_sparse(
    H: Tensor,
    mask: Optional[Tensor] = None,
) -> BatchedDenseHypergraph:
    """Unfold dense incidence into sparse ``hyperedge_index`` and ``edge_entry_weight``.

    Each batch graph is offset-disjoint so a single sparse convolution can process
    the whole mini-batch. ``edge_entry_weight`` stores the node-hyperedge weights
    ``H[b, v, e]`` for every incidence entry.
    """
    if H.dim() != 3:
        raise ValueError(f"H must be 3-D (bs, num_v, num_e), got shape {tuple(H.shape)}")

    bs, n, e = H.shape
    device = H.device

    if mask is not None:
        node_mask = mask.view(bs, n, 1).to(dtype=H.dtype, device=device)
        H = H * node_mask

    v = torch.arange(n, device=device)
    edge = torch.arange(e, device=device)
    vv, ee = torch.meshgrid(v, edge, indexing='ij')
    v_local = vv.reshape(-1)
    e_local = ee.reshape(-1)

    v_idx = torch.cat([v_local + b * n for b in range(bs)], dim=0)
    e_idx = torch.cat([e_local + b * e for b in range(bs)], dim=0)
    hyperedge_index = torch.stack([v_idx, e_idx], dim=0)
    edge_entry_weight = H.reshape(-1)

    return BatchedDenseHypergraph(
        hyperedge_index=hyperedge_index,
        edge_entry_weight=edge_entry_weight,
        num_nodes=bs * n,
        num_edges=bs * e,
        batch_size=bs,
        num_nodes_per_graph=n,
        num_edges_per_graph=e,
    )


HYPERCONV_CHOICES = ('hypergraph', 'heal', 'unigcnii', 'unigat')


def build_hyperconv_layer(
    name: str,
    in_channels: int,
    out_channels: int,
    *,
    num_edges: Optional[int] = None,
    act2: bool = True,
    drop_rate: float = 0.5,
    eps: float = 1e-15,
    **kwargs,
) -> nn.Module:
    """Factory for dense hypergraph convolution layers used in HyperHierarchicalGRL."""
    name = name.lower()
    if name == 'hypergraph':
        return DenseHypergraphConv(
            in_channels, out_channels, act2=act2, drop_rate=drop_rate, eps=eps, **kwargs)
    if name == 'heal':
        if num_edges is None:
            raise ValueError("num_edges is required when hyper_conv='heal'")
        from models.hgnn_conv import HGNN_HEAL
        return HGNN_HEAL(
            num_edges, in_channels, act2=act2, drop_rate=drop_rate, eps=eps, **kwargs)
    if name == 'unigcnii':
        return DenseUniGCNIIConv(
            in_channels, out_channels, act2=act2, drop_rate=drop_rate, eps=eps, **kwargs)
    if name == 'unigat':
        gat_kwargs = dict(kwargs)
        gat_kwargs.setdefault('dropout', drop_rate)
        return DenseUniGATConv(
            in_channels, out_channels, act2=act2, drop_rate=drop_rate, eps=eps, **gat_kwargs)
    raise ValueError(f"hyper_conv must be one of {HYPERCONV_CHOICES}, got {name!r}")

class HypergraphConv(MessagePassing):
    """Hypergraph convolution with optional real-valued incidence weights.

    Implements
    X' = D^{-1} H W B^{-1} H^T X Theta
  where H may contain non-negative real entries passed as ``edge_entry_weight``.
    When ``edge_entry_weight`` is None, each incidence is treated as weight 1.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, eps: float = 1e-15, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.eps = eps
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
            w = edge_entry_weight.to(x.dtype).clamp_min(self.eps)

        x = self.lin(x)

        # B_e = sum_v H_{ve}; D_v = sum_e H_{ve} W_e
        B = scatter(w, edges, dim=0, dim_size=num_edges, reduce='sum').clamp_min(self.eps)
        D = scatter(hyperedge_weight[edges] * w, vertex, dim=0, dim_size=num_nodes, reduce='sum')
        D = D.clamp_min(self.eps)

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


class DenseHypergraphConv(nn.Module):
    """Dense batched wrapper around :class:`HypergraphConv`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
        act2: bool = False,
        eps: float = 1e-15,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act2 = nn.ReLU(inplace=True) if act2 else None
        self.drop = nn.Dropout(drop_rate)
        self.conv = HypergraphConv(in_channels, out_channels, bias=bias, eps=eps)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(
        self,
        X: Tensor,
        H: Tensor,
        W: Tensor,
        D: Optional[Tensor] = None,
        B: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        del D, B

        bs, n, _ = X.shape
        out_channels = self.conv.out_channels

        if mask is not None:
            node_mask = mask.view(bs, n, 1).to(dtype=X.dtype, device=X.device)
            X = X * node_mask

        sparse_hg = dense_incidence_to_sparse(H, mask=mask)
        x = X.reshape(sparse_hg.num_nodes, -1)
        z = self.conv(
            x,
            sparse_hg.hyperedge_index,
            hyperedge_weight=W.reshape(-1),
            edge_entry_weight=sparse_hg.edge_entry_weight,
            num_edges=sparse_hg.num_edges,
        )
        Z = z.view(bs, n, out_channels)

        if not self.is_last:
            if self.act2 is not None:
                Z = self.act2(Z)
            if self.bn is not None:
                Z = self.bn(Z.transpose(1, 2)).transpose(1, 2)
            Z = self.drop(Z)
        if mask is not None:
            Z = Z * mask.view(bs, n, 1).to(dtype=Z.dtype, device=Z.device)
        return Z


class _DenseHyperconvMixin:
    """Shared post-processing for dense hypergraph convolution wrappers."""

    is_last: bool
    bn: Optional[nn.BatchNorm1d]
    act2: Optional[nn.ReLU]
    drop: nn.Dropout

    def _postprocess(self, Z: Tensor, mask: Optional[Tensor]) -> Tensor:
        bs, n, _ = Z.shape
        if not self.is_last:
            if self.act2 is not None:
                Z = self.act2(Z)
            if self.bn is not None:
                Z = self.bn(Z.transpose(1, 2)).transpose(1, 2)
            Z = self.drop(Z)
        if mask is not None:
            Z = Z * mask.view(bs, n, 1).to(dtype=Z.dtype, device=Z.device)
        return Z

    @staticmethod
    def _prepare_dense_inputs(
        X: Tensor,
        H: Tensor,
        mask: Optional[Tensor],
    ) -> tuple[Tensor, BatchedDenseHypergraph]:
        bs, n, _ = X.shape
        if mask is not None:
            node_mask = mask.view(bs, n, 1).to(dtype=X.dtype, device=X.device)
            X = X * node_mask
        sparse_hg = dense_incidence_to_sparse(H, mask=mask)
        x = X.reshape(sparse_hg.num_nodes, -1)
        return x, sparse_hg


class DenseUniGCNIIConv(nn.Module, _DenseHyperconvMixin):
    """Dense batched wrapper around a UniGCNII block.

    UniGCNII layers in the same block share the initial representation ``x0``.
    This matches the residual semantics of :class:`UniGCNII.UniGCNIIEncoder`
    instead of resetting ``x0`` for every individual convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
        act2: bool = False,
        alpha: float = 0.1,
        lamda: float = 0.5,
        layer_idx: int = 1,
        num_layers: int = 2,
        eps: float = 1e-15,
    ):
        super().__init__()
        from models.UniGCNII import UniGCNIIConv

        if in_channels != out_channels:
            raise ValueError(
                "DenseUniGCNIIConv requires in_channels == out_channels because "
                "UniGCNII mixes each layer output with the shared x0 residual."
            )
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act2 = nn.ReLU(inplace=True) if act2 else None
        self.drop = nn.Dropout(drop_rate)
        self.inner_drop = nn.Dropout(drop_rate)
        self.inner_act = nn.ReLU(inplace=True)
        self.convs = nn.ModuleList([
            UniGCNIIConv(in_channels, out_channels, eps=eps)
            for _ in range(num_layers)
        ])
        self.alpha = alpha
        self.lamda = lamda
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.out_channels = out_channels

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(
        self,
        X: Tensor,
        H: Tensor,
        W: Tensor,
        D: Optional[Tensor] = None,
        B: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        del W, D, B

        bs, n, _ = X.shape
        x, sparse_hg = self._prepare_dense_inputs(X, H, mask)
        x0 = x
        degV = x.new_ones((sparse_hg.num_nodes, 1))
        degE = x.new_ones((sparse_hg.num_edges, 1))

        for i, conv in enumerate(self.convs):
            beta = math.log(self.lamda / (self.layer_idx + i) + 1)
            x = self.inner_drop(x)
            x = conv(
                x,
                sparse_hg.hyperedge_index,
                self.alpha,
                beta,
                x0,
                degV,
                degE,
                edge_entry_weight=sparse_hg.edge_entry_weight,
            )
            if i < self.num_layers - 1:
                x = self.inner_act(x)

        Z = x.view(bs, n, self.out_channels)
        return self._postprocess(Z, mask)


class DenseUniGATConv(nn.Module, _DenseHyperconvMixin):
    """Dense batched wrapper around :class:`UniGCNII.UniGATConv`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
        act2: bool = False,
        eps: float = 1e-15,
        **gat_kwargs,
    ):
        super().__init__()
        from models.UniGCNII import UniGATConv

        self.is_last = is_last
        self.out_channels = heads * out_channels
        self.bn = nn.BatchNorm1d(self.out_channels) if use_bn else None
        self.act2 = nn.ReLU(inplace=True) if act2 else None
        self.drop = nn.Dropout(drop_rate)
        self.conv = UniGATConv(in_channels, out_channels, heads=heads, eps=eps, **gat_kwargs)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(
        self,
        X: Tensor,
        H: Tensor,
        W: Tensor,
        D: Optional[Tensor] = None,
        B: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        del W, D, B

        bs, n, _ = X.shape
        x, sparse_hg = self._prepare_dense_inputs(X, H, mask)
        z = self.conv(
            x,
            sparse_hg.hyperedge_index,
            edge_entry_weight=sparse_hg.edge_entry_weight,
        )
        Z = z.view(bs, n, self.out_channels)
        return self._postprocess(Z, mask)
