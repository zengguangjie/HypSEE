import torch
from torch_geometric.nn import GINConv, MLP, DenseGINConv, GCNConv, DenseGCNConv, GraphConv, SAGEConv, GATConv, GATv2Conv, SGConv, ARMAConv
from torch_geometric.utils import to_dense_batch, is_undirected, to_dense_adj, add_self_loops
from torch.nn import Linear, Parameter, ReLU, Sequential
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
        'GIN', 'GCN', 'GraphConv', 'SAGEConv', 'GATConv', 'GATv2Conv', 'SGConv', 'ARMAConv',
    )

    def __init__(self, in_channels, hidden_channels_gnn, hidden_channels, out_channels, num_layers_gnn, num_edges2, avg_num_nodes, height=3, EPS=1e-15,
                 gnn_arch='GCN', decay_rate=0.5, hgsl_constraint='sigmoid', hgsl_topk=None,
                 use_gnn_encoder_S=True):
        super(HypSEE, self).__init__()

        self.EPS = EPS
        self.gnn_arch = gnn_arch
        self.use_gnn_encoder_S = use_gnn_encoder_S
        if gnn_arch not in self.GNN_ARCH_CHOICES:
            raise NotImplementedError(
                f"gnn_arch must be one of {self.GNN_ARCH_CHOICES}, got {gnn_arch!r}")

        if hidden_channels_gnn == 0:
            hidden_channels_gnn = hidden_channels

        self.gnn_encoder, self.batch_norm = self._build_gnn_encoder(
            in_channels, hidden_channels_gnn, hidden_channels, num_layers_gnn)
        if use_gnn_encoder_S:
            self.gnn_encoder_S, self.batch_norm_S = self._build_gnn_encoder(
                in_channels, hidden_channels_gnn, hidden_channels, num_layers_gnn)
        else:
            self.gnn_encoder_S = torch.nn.ModuleList()
            self.batch_norm_S = torch.nn.ModuleList()

        if not use_gnn_encoder_S and in_channels != hidden_channels:
            self.view_S_proj = Linear(in_channels, hidden_channels)
        else:
            self.view_S_proj = None

        self.hyper_struct_learning = HyperStructLearning(
            hidden_channels, num_edges2, constraint=hgsl_constraint, topk=hgsl_topk)

        # Hierarchical Encoder.
        self.hyper_hierarchical_GRL = HyperHierarchicalGRL(hidden_channels, avg_num_nodes, num_edges2, height=height, EPS=self.EPS, decay_rate=decay_rate)

        self.classifier = Sequential(Linear(hidden_channels*2, hidden_channels*2),
                                    ReLU(inplace=True),
                                    Linear(hidden_channels*2, out_channels))

    @staticmethod
    def _make_gnn_conv(gnn_arch, in_channels, out_channels):
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
        if gnn_arch == 'GraphConv':
            return GraphConv(in_channels, out_channels)
        if gnn_arch == 'SAGEConv':
            return SAGEConv(in_channels, out_channels)
        if gnn_arch == 'GATConv':
            return GATConv(in_channels, out_channels, heads=1, concat=False)
        if gnn_arch == 'GATv2Conv':
            return GATv2Conv(in_channels, out_channels, heads=1, concat=False)
        if gnn_arch == 'SGConv':
            return SGConv(in_channels, out_channels)
        if gnn_arch == 'ARMAConv':
            return ARMAConv(in_channels, out_channels)
        raise NotImplementedError(f"Unsupported gnn_arch: {gnn_arch!r}")

    def _build_gnn_encoder(self, in_channels, hidden_channels_gnn, hidden_channels, num_layers_gnn):
        convs = torch.nn.ModuleList()
        batch_norms = torch.nn.ModuleList()
        use_batch_norm = self.gnn_arch == 'GCN'

        in_ch = in_channels
        for _ in range(num_layers_gnn - 1):
            convs.append(self._make_gnn_conv(self.gnn_arch, in_ch, hidden_channels_gnn))
            if use_batch_norm:
                batch_norms.append(BN(hidden_channels_gnn))
            in_ch = hidden_channels_gnn
        convs.append(self._make_gnn_conv(self.gnn_arch, in_ch, hidden_channels))
        if use_batch_norm:
            batch_norms.append(BN(hidden_channels))
        return convs, batch_norms

    def _apply_gnn_encoder(self, x, edge_index, convs, batch_norms):
        if len(convs) == 0:
            raise RuntimeError("gnn_encoder is empty; check num_layers_gnn")
        for i, conv in enumerate(convs):
            x = conv(x, edge_index)
            if i < len(batch_norms):
                x = batch_norms[i](x).relu()
            elif self.gnn_arch != 'GIN':
                x = F.relu(x)
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
                embedding_gnn_S, edge_index_S, self.gnn_encoder_S, self.batch_norm_S)
        elif self.view_S_proj is not None:
            embedding_gnn_S = self.view_S_proj(embedding_gnn_S)

        embedding_gnn_S, mask_S = to_dense_batch(embedding_gnn_S, batch_S)

        bs, _num_v, _num_e = H_S.size()
        W_S = torch.ones([bs, _num_e], dtype=H_S.dtype, device=H_S.device)
        Z_S, loss_hse_S = self.hyper_hierarchical_GRL(embedding_gnn_S, H_S, W_S, mask_S)

        # print("----------------------view T--------------------")
        X_sparse_T, edge_index_T, batch_T = data_T.x, data_T.edge_index, data_T.batch
        embedding_gnn_T = X_sparse_T
        edge_index_T, _ = add_self_loops(edge_index=edge_index_T)
        embedding_gnn_T = self._apply_gnn_encoder(
            embedding_gnn_T, edge_index_T, self.gnn_encoder, self.batch_norm)
        embedding_gnn_T, mask_T = to_dense_batch(embedding_gnn_T, batch_T)

        H_T = self.hyper_struct_learning(embedding_gnn_T)
        # W_T = torch.eye(H_T.shape[-1]).unsqueeze(0).repeat(X.shape[0], 1, 1).to(X.device)
        bs, _num_v, _num_e = H_T.size()
        W_T = torch.ones([bs, _num_e], dtype=H_T.dtype, device=H_T.device)
        Z_T, loss_hse_T = self.hyper_hierarchical_GRL(embedding_gnn_T, H_T, W_T, mask_T)

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