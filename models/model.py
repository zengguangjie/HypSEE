import torch
from torch_geometric.nn import GINConv, MLP, DenseGINConv, GCNConv, DenseGCNConv, GraphConv, SAGEConv, GATConv, GATv2Conv, SGConv, ARMAConv, GCN2Conv
from torch_geometric.utils import to_dense_batch, is_undirected, to_dense_adj, add_self_loops
from torch.nn import Dropout, Linear, Parameter, ReLU, Sequential
import torch.nn.functional as F
# from hypergraph_conv import HypergraphConv
from models.hgnn_conv import HGNNConvDense, HGNNConv
from math import ceil
import math
from torch_geometric.nn.models import GCN
from torch_scatter import scatter_sum

from models.modules import HyperHierarchicalGRL, HyperStructLearning
from torch.nn import BatchNorm1d as BN

# EPS = 1e-15


class HypSEE(torch.nn.Module):
    # branch 1: handcrafted hyperedges; branch 2: hyperedges by GNN encoder.
    GNN_ARCH_CHOICES = (
        'GIN', 'GCN', 'GCN2Conv', 'GraphConv', 'SAGEConv', 'GATConv', 'GATv2Conv', 'SGConv', 'ARMAConv',
    )

    def __init__(self, in_channels, hidden_channels_gnn, hidden_channels, out_channels, num_layers_gnn,
                 num_edges1, num_edges2, avg_num_nodes, height=3, EPS=1e-15,
                 gnn_arch='GCN', decay_rate=0.5, hgsl_constraint='sigmoid', hgsl_topk=None,
                 use_gnn_encoder_S=True, shared_hyper_encoder=True, hyper_conv='hypergraph',
                 dropout=0.5, pool_type='clusternet'):
        super(HypSEE, self).__init__()

        self.EPS = EPS
        self.dropout = dropout
        self.gnn_arch = gnn_arch
        self.use_gnn_encoder_S = use_gnn_encoder_S
        self.shared_hyper_encoder = shared_hyper_encoder
        if shared_hyper_encoder and hyper_conv == 'heal' and num_edges1 != num_edges2:
            raise ValueError(
                f"shared_hyper_encoder with hyper_conv='heal' requires num_edges1 == num_edges2, "
                f"got num_edges1={num_edges1}, num_edges2={num_edges2}"
            )
        if gnn_arch not in self.GNN_ARCH_CHOICES:
            raise NotImplementedError(
                f"gnn_arch must be one of {self.GNN_ARCH_CHOICES}, got {gnn_arch!r}")

        if hidden_channels_gnn == 0:
            hidden_channels_gnn = hidden_channels

        self.gnn_encoder, self.batch_norm, self.gnn_drop, self.gnn_lin_in, self.gnn_lin_out = self._build_gnn_encoder(
            in_channels, hidden_channels_gnn, hidden_channels, num_layers_gnn, dropout)
        if use_gnn_encoder_S:
            self.gnn_encoder_S, self.batch_norm_S, self.gnn_drop_S, self.gnn_lin_in_S, self.gnn_lin_out_S = self._build_gnn_encoder(
                in_channels, hidden_channels_gnn, hidden_channels, num_layers_gnn, dropout)
        else:
            self.gnn_encoder_S = torch.nn.ModuleList()
            self.batch_norm_S = torch.nn.ModuleList()
            self.gnn_drop_S = torch.nn.ModuleList()
            self.gnn_lin_in_S = None
            self.gnn_lin_out_S = None

        if not use_gnn_encoder_S and in_channels != hidden_channels:
            self.view_S_proj = Linear(in_channels, hidden_channels)
        else:
            self.view_S_proj = None

        self.hyper_struct_learning = HyperStructLearning(
            hidden_channels, num_edges2, constraint=hgsl_constraint, topk=hgsl_topk)

        grl_common = dict(
            hidden_channels=hidden_channels,
            avg_num_nodes=avg_num_nodes,
            height=height,
            EPS=self.EPS,
            decay_rate=decay_rate,
            hyper_conv=hyper_conv,
            dropout=dropout,
            pool_type=pool_type,
        )
        if shared_hyper_encoder:
            self.hyper_hierarchical_GRL = HyperHierarchicalGRL(
                num_edges=num_edges2, **grl_common)
            self.hyper_hierarchical_GRL_S = None
            self.hyper_hierarchical_GRL_T = None
        else:
            self.hyper_hierarchical_GRL = None
            self.hyper_hierarchical_GRL_S = HyperHierarchicalGRL(
                num_edges=num_edges1, **grl_common)
            self.hyper_hierarchical_GRL_T = HyperHierarchicalGRL(
                num_edges=num_edges2, **grl_common)

        self.classifier = Sequential(
            Linear(hidden_channels * 2, hidden_channels * 2),
            ReLU(inplace=True),
            Dropout(dropout),
            Linear(hidden_channels * 2, out_channels),
        )

    def _make_gnn_conv(self, gnn_arch, in_channels, out_channels, dropout=0.5, layer_idx=None):
        if gnn_arch == 'GIN':
            return GINConv(
                Sequential(
                    Linear(in_channels, out_channels),
                    ReLU(),
                    Linear(out_channels, out_channels),
                    ReLU(),
                    BN(out_channels),
                ), train_eps=False)
        if gnn_arch == 'GCN':
            return GCNConv(in_channels, out_channels)
        if gnn_arch == 'GCN2Conv':
            assert layer_idx is not None
            return GCN2Conv(
                out_channels, alpha=0.1, theta=0.5, layer=layer_idx,
                add_self_loops=False)
        if gnn_arch == 'GraphConv':
            return GraphConv(in_channels, out_channels)
        if gnn_arch == 'SAGEConv':
            return SAGEConv(in_channels, out_channels)
        if gnn_arch == 'GATConv':
            return GATConv(in_channels, out_channels, heads=1, concat=False, dropout=dropout)
        if gnn_arch == 'GATv2Conv':
            return GATv2Conv(in_channels, out_channels, heads=1, concat=False, dropout=dropout)
        if gnn_arch == 'SGConv':
            return SGConv(in_channels, out_channels)
        if gnn_arch == 'ARMAConv':
            return ARMAConv(in_channels, out_channels)
        raise NotImplementedError(f"Unsupported gnn_arch: {gnn_arch!r}")

    def _build_gnn_encoder(self, in_channels, hidden_channels_gnn, hidden_channels, num_layers_gnn, dropout):
        convs = torch.nn.ModuleList()
        batch_norms = torch.nn.ModuleList()
        dropouts = torch.nn.ModuleList()
        lin_in = None
        lin_out = None

        if self.gnn_arch == 'GCN2Conv':
            lin_in = Linear(in_channels, hidden_channels_gnn)
            for layer_idx in range(1, num_layers_gnn + 1):
                convs.append(self._make_gnn_conv(
                    self.gnn_arch, hidden_channels_gnn, hidden_channels_gnn, dropout, layer_idx=layer_idx))
                dropouts.append(Dropout(dropout))
            if hidden_channels != hidden_channels_gnn:
                lin_out = Linear(hidden_channels_gnn, hidden_channels)
            return convs, batch_norms, dropouts, lin_in, lin_out

        use_batch_norm = self.gnn_arch == 'GCN'

        in_ch = in_channels
        for _ in range(num_layers_gnn - 1):
            convs.append(self._make_gnn_conv(self.gnn_arch, in_ch, hidden_channels_gnn, dropout))
            if use_batch_norm:
                batch_norms.append(BN(hidden_channels_gnn))
            dropouts.append(Dropout(dropout))
            in_ch = hidden_channels_gnn
        convs.append(self._make_gnn_conv(self.gnn_arch, in_ch, hidden_channels, dropout))
        if use_batch_norm:
            batch_norms.append(BN(hidden_channels))
        dropouts.append(Dropout(dropout))
        return convs, batch_norms, dropouts, lin_in, lin_out

    def _apply_gnn_encoder(self, x, edge_index, convs, batch_norms, dropouts, lin_in=None, lin_out=None):
        if len(convs) == 0:
            raise RuntimeError("gnn_encoder is empty; check num_layers_gnn")
        if self.gnn_arch == 'GCN2Conv':
            assert lin_in is not None
            x = F.relu(lin_in(x))
            x_0 = x
            for i, conv in enumerate(convs):
                x = dropouts[i](x)
                x = conv(x, x_0, edge_index)
                x = F.relu(x)
            if lin_out is not None:
                x = lin_out(x)
            return x
        for i, conv in enumerate(convs):
            x = conv(x, edge_index)
            if i < len(batch_norms):
                x = batch_norms[i](x).relu()
            elif self.gnn_arch != 'GIN':
                x = F.relu(x)
            x = dropouts[i](x)
        return x

    def reset_parameters(self):
        self.hyper_struct_learning.reset_parameters()

    # Hierarchical-Aware View: s
    # Embedding-Derived View: w
    # def forward(self, X_sparse, edge_index, H_S, batch):
    def forward(self, data_S, data_T, H_S):
        # view s:
        # print("----------------------view S--------------------")
        X_sparse_S, edge_index_S, batch_S = data_S.x, data_S.edge_index, data_S.batch
        embedding_gnn_S = X_sparse_S
        if self.use_gnn_encoder_S:
            edge_index_S, _ = add_self_loops(edge_index=edge_index_S)
            embedding_gnn_S = self._apply_gnn_encoder(
                embedding_gnn_S, edge_index_S, self.gnn_encoder_S, self.batch_norm_S, self.gnn_drop_S,
                lin_in=self.gnn_lin_in_S, lin_out=self.gnn_lin_out_S)
        elif self.view_S_proj is not None:
            embedding_gnn_S = self.view_S_proj(embedding_gnn_S)

        embedding_gnn_S, mask_S = to_dense_batch(embedding_gnn_S, batch_S)

        bs, _num_v, _num_e = H_S.size()
        W_S = torch.ones([bs, _num_e], dtype=H_S.dtype, device=H_S.device)
        if self.shared_hyper_encoder:
            assert self.hyper_hierarchical_GRL is not None
            grl_s = self.hyper_hierarchical_GRL
        else:
            assert self.hyper_hierarchical_GRL_S is not None
            grl_s = self.hyper_hierarchical_GRL_S
        Z_S, loss_hse_S = grl_s(embedding_gnn_S, H_S, W_S, mask_S)

        # print("----------------------view T--------------------")
        X_sparse_T, edge_index_T, batch_T = data_T.x, data_T.edge_index, data_T.batch
        embedding_gnn_T = X_sparse_T
        edge_index_T, _ = add_self_loops(edge_index=edge_index_T)
        embedding_gnn_T = self._apply_gnn_encoder(
            embedding_gnn_T, edge_index_T, self.gnn_encoder, self.batch_norm, self.gnn_drop,
            lin_in=self.gnn_lin_in, lin_out=self.gnn_lin_out)
        embedding_gnn_T, mask_T = to_dense_batch(embedding_gnn_T, batch_T)

        H_T = self.hyper_struct_learning(embedding_gnn_T)
        # W_T = torch.eye(H_T.shape[-1]).unsqueeze(0).repeat(X.shape[0], 1, 1).to(X.device)
        bs, _num_v, _num_e = H_T.size()
        W_T = torch.ones([bs, _num_e], dtype=H_T.dtype, device=H_T.device)
        if self.shared_hyper_encoder:
            assert self.hyper_hierarchical_GRL is not None
            grl_t = self.hyper_hierarchical_GRL
        else:
            assert self.hyper_hierarchical_GRL_T is not None
            grl_t = self.hyper_hierarchical_GRL_T
        Z_T, loss_hse_T = grl_t(embedding_gnn_T, H_T, W_T, mask_T)

        # print("----------------------classifier--------------------")
        Z = torch.cat((Z_S, Z_T), dim=-1)
        out = self.classifier(Z)
        return Z_S, Z_T, out, loss_hse_S, loss_hse_T


    def loss_con(self, Su, Sm, Tu, Tm):
        T = 0.5
        eps = self.EPS

        Su_abs = torch.linalg.vector_norm(Su, dim=1).clamp(min=eps)
        Sm_abs = torch.linalg.vector_norm(Sm, dim=1).clamp(min=eps)
        Tu_abs = torch.linalg.vector_norm(Tu, dim=1).clamp(min=eps)
        Tm_abs = torch.linalg.vector_norm(Tm, dim=1).clamp(min=eps)

        sim_matrix_P = (torch.einsum('ik,jk->ij', Su, Sm) / torch.einsum('i,j->ij', Su_abs, Sm_abs)) / T
        Log_Pu = torch.log_softmax(sim_matrix_P, dim=-1)
        sim_matrix_Q = (torch.einsum('ik,jk->ij', Tu, Tm) / torch.einsum('i,j->ij', Tu_abs, Tm_abs)) / T
        Log_Qu = torch.log_softmax(sim_matrix_Q, dim=-1)
        KL = F.kl_div(Log_Pu, Log_Qu, reduction="batchmean", log_target=True) + F.kl_div(Log_Qu, Log_Pu, reduction="batchmean", log_target=True)
        # print(KL)
        return KL/2