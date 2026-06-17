import torch
from torch_geometric.nn import GINConv, MLP, DenseGINConv, GCNConv, DenseGCNConv, DenseGraphConv
from torch_geometric.utils import to_dense_batch, is_undirected, to_dense_adj
from torch.nn import Linear, Parameter, ReLU, Sequential, BatchNorm1d as BN
import torch.nn.functional as F
# from hypergraph_conv import HypergraphConv
from models.hgnn_conv import HGNNConvDense, HGNNConv, HGNN_HEAL
from math import ceil
import math
from torch_geometric.nn.models import GCN
from torch_scatter import scatter_sum
# from torch.nn import BatchNorm1d as BatchNorm

# EPS = 1e-15



class HyperStructLearningSigmoid(torch.nn.Module):
    def __init__(self, out_channels_gnn, num_edges2):
        super(HyperStructLearningSigmoid, self).__init__()
        # self.out_channels_gnn = out_channels_gnn
        # self.num_edges2 = num_edges2
        self.lin = Linear(out_channels_gnn, num_edges2)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, embedding_gnn):
        embedding_gnn = embedding_gnn.unsqueeze(0) if embedding_gnn.dim() == 2 else embedding_gnn
        H = self.lin(embedding_gnn)
        # assert not torch.isnan(H).any()
        H = F.sigmoid(H)
        # assert torch.all(H >= 0)
        assert not (H<0).any()
        # assert not torch.isnan(H).any()
        return H



class HyperHierarchicalGRL(torch.nn.Module):
    def __init__(self, hidden_channels, avg_num_nodes, num_edges=None, height=3, decay_rate=0.5,
                 sym_D=False, EPS=1e-15, act2=True):
        super(HyperHierarchicalGRL, self).__init__()

        self.height = height
        self.EPS = EPS

        self.hyperconv_dict = {}
        self.pool_dict = {}
        num_nodes = avg_num_nodes

        for k in range(self.height, 1, -1):
            self.hyperconv_dict[k] = HGNN_HEAL(num_edges, hidden_channels, act2=act2)
            # print(next(self.hyperconv_dict[k].theta.parameters()).device)
            num_nodes = ceil(decay_rate * num_nodes)
            self.pool_dict[k] = Linear(hidden_channels, num_nodes)
        self.hyperconv_dict[1] = HGNN_HEAL(num_edges, hidden_channels, act2=act2)

        self.clu_mat = {}  # C
        self.vol_dict = {}
        self.sym_D = sym_D

    def forward(self, X, H, W, mask, temp=1.0):
        assert X.dim() == 3  # b*n*d
        assert H.dim() == 3  # b*n*k
        assert W.dim() == 2  # b*k
        assert mask.dim() == 2  # b*n
        H_input = H
        assert not torch.isnan(H).any()

        assert not torch.isnan(X).any()
        assert not torch.isinf(X).any()

        for k in range(self.height, 1, -1):
            bs, _num_v, _num_e = H.size()
            if k == self.height:
                mask = mask.view([bs, _num_v, 1]).to(H.dtype).to(H.device)
            else:
                mask = None
            if self.sym_D:
                HW = torch.einsum('bij,bj->bij', H, W)
                D = torch.einsum('bij,bjk->bik', HW, H.transpose(-1, -2))
                D = torch.einsum('bii->bi', D)
            else:
                D = torch.einsum('bij,bj->bi', H, W)
            Z = self.hyperconv_dict[k](X, H, W, D=D, mask=mask)
            C = self.pool_dict[k](Z)  # b*n_h*n_h-1
            C = torch.softmax(C / temp, dim=-1)
            assert not torch.isnan(Z).any()
            assert not torch.isnan(C).any()
            H = C.transpose(-1, -2).matmul(H)
            X = C.transpose(-1, -2).matmul(Z)

            assert not torch.isnan(X).any()
            assert not torch.isinf(X).any()

            self.clu_mat[k] = C
            self.vol_dict[k] = D

        k = 1
        bs, _num_v, _num_e = H.size()
        if k == self.height:
            mask = mask.view([bs, _num_v, 1]).to(H.dtype).to(H.device)
        else:
            mask = None
        assert not torch.isnan(H).any()
        assert not torch.isnan(W).any()
        if self.sym_D:
            HW = torch.einsum('bij,bj->bij', H, W)
            D = torch.einsum('bij,bjk->bik', HW, H.transpose(-1, -2))
            D = torch.einsum('bii->bi', D)
        else:
            D = torch.einsum('bij,bj->bi', H, W)
        # assert not (D<0).any()
        assert not torch.isnan(D).any()
        # assert (D>=0).all()
        Z = self.hyperconv_dict[k](X, H, W, D=D, mask=mask)
        C = torch.ones([bs, C.shape[-1], 1]).to(H.dtype).to(X.device)
        self.clu_mat[k] = C
        self.vol_dict[k] = D

        # Z = torch.sum(Z, dim=-2)
        Z = torch.mean(Z, dim=-2)
        loss_hse = self.hse_loss(H_input, W, 1e-10)
        return Z, loss_hse

    def hse_loss(self, H, W, EPS):
        bs, _num_v, _num_e = H.size()
        B = torch.einsum('bij->bj', H)  # b*k
        B_inv = 1.0 / (B + EPS)
        B_inv[B_inv == float("inf")] = 0

        ass_mat = {}
        ass_mat[self.height] = torch.eye(_num_v).unsqueeze(0).repeat(bs, 1, 1).to(H.device)
        vol_dict = self.vol_dict
        clu_mat = self.clu_mat

        ass_mat[self.height - 1] = self.clu_mat[self.height]
        for k in range(self.height - 2, 0, -1):
            ass_mat[k] = ass_mat[k + 1].matmul(clu_mat[k + 1])

        hse_loss = torch.zeros([bs], device=H.device)
        vol_H = torch.einsum('bi->b', vol_dict[self.height])  # b
        vol_dict[0] = vol_H.unsqueeze(-1)
        for k in range(1, self.height + 1):
            if self.sym_D:
                CV = torch.einsum('bij,bj->bij', clu_mat[k], vol_dict[k - 1])
                vol_parent = torch.einsum('bij->bjk', CV, clu_mat[k].transpose(-1, -2))
                vol_parent = torch.einsum('bii->bi', vol_parent)
            else:
                vol_parent = torch.einsum('bij,bj->bi', clu_mat[k], vol_dict[k - 1])
            log_vol_ratio_k = torch.log2((vol_dict[k] + EPS) / (vol_parent + EPS))
            # if k == self.height:
            #     SH = H
            SH = torch.einsum('bij,bjk->bik', ass_mat[k].transpose(-1, -2), H)  # b*n*k
            HS = torch.einsum('bij,bjk->bik', H.transpose(-1, -2), (1.0 - ass_mat[k]))  # b*k*n
            WB = W * B_inv  # b*k
            SHWB = torch.einsum('bij,bj->bij', SH, WB)  # b*n*k
            # cut_k = torch.einsum('bij,bjk->bik', SHWB, HS)  # b*n*n
            # cut_k = torch.einsum('bii->bi', cut_k)  # b*n
            cut_k = torch.einsum('bij,bji->bi', SHWB, HS)
            hse_loss_k = - cut_k * log_vol_ratio_k  # b*n
            hse_loss += hse_loss_k.sum(dim=-1) / vol_H
        return hse_loss.mean()




class HyperHierarchicalGRLDense(torch.nn.Module):
    def __init__height3(self, in_channels, hidden_channels, out_channels, avg_num_nodes, decay_rate=0.5, sym_D=False):
        super(HyperHierarchicalGRLDense, self).__init__()

        self.hyperconv3 = HGNNConvDense(in_channels, hidden_channels, sym_D=sym_D)
        num_nodes = ceil(decay_rate * avg_num_nodes)
        self.pool3 = Linear(hidden_channels, num_nodes)

        self.hyperconv2 = HGNNConvDense(hidden_channels, hidden_channels, sym_D=sym_D)
        num_nodes = ceil(decay_rate * num_nodes)
        self.pool2 = Linear(hidden_channels, num_nodes)

        self.hyperconv1 = HGNNConvDense(hidden_channels, out_channels, sym_D=sym_D)

        self.height = 3
        self.clu_mat = {}  # C
        self.vol_dict = {}
        self.sym_D = sym_D

    def __init__(self, in_channels, hidden_channels, out_channels, avg_num_nodes, height=3, decay_rate=0.5,
                 sym_D=False, EPS=1e-15):
        super(HyperHierarchicalGRLDense, self).__init__()

        # self.hyperconv3 = HGNNConvDense(in_channels, hidden_channels, sym_D=sym_D)
        # num_nodes = ceil(decay_rate * avg_num_nodes)
        # self.pool3 = Linear(hidden_channels, num_nodes)
        #
        # self.hyperconv2 = HGNNConvDense(hidden_channels, hidden_channels, sym_D=sym_D)
        # num_nodes = ceil(decay_rate * num_nodes)
        # self.pool2 = Linear(hidden_channels, num_nodes)
        #
        # self.hyperconv1 = HGNNConvDense(hidden_channels, out_channels, sym_D=sym_D)

        self.height = height
        self.EPS = EPS

        self.hyperconv_dict = {}
        self.pool_dict = {}
        num_nodes = avg_num_nodes
        for k in range(self.height, 1, -1):
            self.hyperconv_dict[k] = HGNNConvDense(in_channels, hidden_channels, sym_D=sym_D)
            # print(next(self.hyperconv_dict[k].theta.parameters()).device)
            num_nodes = ceil(decay_rate * num_nodes)
            self.pool_dict[k] = Linear(hidden_channels, num_nodes)
            in_channels = hidden_channels
        self.hyperconv_dict[1] = HGNNConvDense(hidden_channels, out_channels, sym_D=sym_D)

        self.clu_mat = {}  # C
        self.vol_dict = {}
        self.sym_D = sym_D

    def forward_hieght3(self, X, H, W, mask, temp=1.0):
        assert X.dim() == 3

        X3 = X
        H3 = H
        bs, _num_v, _num_e = H3.size()
        mask = mask.view([bs, _num_v, 1]).to(H.dtype).to(H.device)
        if self.sym_D:
            D3 = H3.matmul(W).matmul(H3.transpose(-1, -2))
        else:
            D3 = H3.matmul(W).matmul(torch.ones([bs, _num_e, _num_v]).to(H.dtype).to(H.device))
        eye = torch.eye(_num_v).unsqueeze(0).repeat(bs, 1, 1).to(H.dtype).to(H.device)
        D3 = D3 * eye
        Z3 = self.hyperconv3(X3, H3, W, D=D3, mask=mask)
        C3 = self.pool3(Z3)
        C3 = torch.softmax(C3 / temp, dim=-1)
        H2 = C3.transpose(-1, -2).matmul(H3)
        X2 = C3.transpose(-1, -2).matmul(Z3)
        # print(C3.shape, mask.shape)
        # mask = C3.transpose(-1, -2).matmul(mask).to(torch.bool).to(H.dtype)

        bs, _num_v, _num_e = H2.size()
        if self.sym_D:
            D2 = H2.matmul(W).matmul(H2.transpose(-1, -2))
        else:
            D2 = H2.matmul(W).matmul(torch.ones([bs, _num_e, _num_v]).to(H.dtype).to(H.device))
        eye = torch.eye(_num_v).unsqueeze(0).repeat(bs, 1, 1).to(H.dtype).to(H.device)
        D2 = D2 * eye
        Z2 = self.hyperconv2(X2, H2, W, D=D2)
        C2 = self.pool2(Z2)
        C2 = torch.softmax(C2 / temp, dim=-1)
        H1 = C2.transpose(-1, -2).matmul(H2)
        X1 = C2.transpose(-1, -2).matmul(Z2)

        bs, _num_v, _num_e = H1.size()
        if self.sym_D:
            D1 = H1.matmul(W).matmul(H1.transpose(-1, -2))
        else:
            D1 = H1.matmul(W).matmul(torch.ones([bs, _num_e, _num_v]).to(H.dtype).to(H.device))
        eye = torch.eye(_num_v).unsqueeze(0).repeat(bs, 1, 1).to(H.dtype).to(H.device)
        D1 = D1 * eye
        Z1 = self.hyperconv1(X1, H1, W, D=D1)
        # C1 is the assignment matrix between L1 nodes and root, thus a matrix full of 1s.
        C1 = torch.ones([bs, C2.shape[-1], 1]).to(H.dtype).to(X.device)

        self.clu_mat[3] = C3
        self.clu_mat[2] = C2
        self.clu_mat[1] = C1
        self.vol_dict[3] = D3
        self.vol_dict[2] = D2
        self.vol_dict[1] = D1

        # return Z3, Z2, Z1
        Z = torch.sum(Z1, dim=-2)
        loss_hse = self.hse_loss(H, W, self.EPS)
        return Z, loss_hse

    def forward(self, X, H, W, mask, temp=1.0):
        H_input = H
        assert X.dim() == 3
        # assert self.height >= 1
        for k in range(self.height, 1, -1):
            bs, _num_v, _num_e = H.size()
            if k == self.height:
                mask = mask.view([bs, _num_v, 1]).to(H.dtype).to(H.device)
            else:
                mask = None
            if self.sym_D:
                D = H.matmul(W).matmul(H.transpose(-1, -2)).to(H.dtype).to(H.device)
            else:
                D = H.matmul(W).matmul(torch.ones([bs, _num_e, _num_v]).to(H.dtype).to(H.device))
            eye = torch.eye(_num_v).unsqueeze(0).repeat(bs, 1, 1).to(H.dtype).to(H.device)
            D = D * eye
            Z = self.hyperconv_dict[k](X, H, W, D=D, mask=mask)
            C = self.pool_dict[k](Z)
            C = torch.softmax(C / temp, dim=-1)
            H = C.transpose(-1, -2).matmul(H)
            X = C.transpose(-1, -2).matmul(Z)
            self.clu_mat[k] = C
            self.vol_dict[k] = D

        k = 1
        bs, _num_v, _num_e = H.size()
        if k == self.height:
            mask = mask.view([bs, _num_v, 1]).to(H.dtype).to(H.device)
        else:
            mask = None
        if self.sym_D:
            D = H.matmul(W).matmul(H.transpose(-1, -2)).to(H.dtype).to(H.device)
        else:
            D = H.matmul(W).matmul(torch.ones([bs, _num_e, _num_v]).to(H.dtype).to(H.device))
        eye = torch.eye(_num_v).unsqueeze(0).repeat(bs, 1, 1).to(H.dtype).to(H.device)
        D = D * eye
        Z = self.hyperconv_dict[k](X, H, W, D=D, mask=mask)
        C = torch.ones([bs, C.shape[-1], 1]).to(H.dtype).to(X.device)
        self.clu_mat[k] = C
        self.vol_dict[k] = D

        Z = torch.sum(Z, dim=-2)
        loss_hse = self.hse_loss(H_input, W, self.EPS)
        return Z, loss_hse

    def hse_loss(self, H, W, EPS):
        assert len(self.clu_mat) > 0
        bs, _num_v, _num_e = H.size()
        B = torch.ones([bs, _num_e, _num_v]).to(H.device).matmul(H)
        eye = torch.eye(_num_e).unsqueeze(0).repeat(bs, 1, 1).to(H.dtype).to(H.device)
        B_inv = eye / (B + EPS)
        # print(torch.min(B_inv.diagonal(dim1=-2, dim2=-1)), torch.max(B_inv.diagonal(dim1=-2, dim2=-1)), "B_inv in hse_loss")
        B_inv[B_inv == float("inf")] = 0

        ass_mat = {self.height: torch.eye(_num_v).unsqueeze(0).repeat(bs, 1, 1).to(H.device)}
        vol_dict = self.vol_dict
        clu_mat = self.clu_mat
        for k in range(self.height - 1, 0, -1):
            ass_mat[k] = ass_mat[k + 1].matmul(self.clu_mat[k + 1])

        se_loss = torch.zeros([bs], device=H.device)
        vol_H = vol_dict[self.height].diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=False)
        # print(vol_H, "vol_H")
        vol_dict[0] = vol_H.unsqueeze(-1).unsqueeze(-1)
        for k in range(1, self.height + 1):
            # print(vol_dict[k].diagonal(dim1=-2, dim2=-1)[0], "vol_dict[k]")
            if self.sym_D:
                vol_parent = clu_mat[k].matmul(vol_dict[k - 1]).matmul(clu_mat[k].transpose(-1, -2))
            else:
                vol_parent = clu_mat[k].matmul(vol_dict[k - 1]).matmul(
                    torch.ones([bs, vol_dict[k - 1].shape[-1], clu_mat[k].shape[-2]]).to(H.dtype).to(H.device))
            # print(k)
            # print(vol_parent.diagonal(dim1=-2, dim2=-1)[0], "vol_parent")
            log_vol_ratio_k = torch.log2((vol_dict[k] + EPS) / (vol_parent + EPS))
            log_vol_ratio_k = log_vol_ratio_k.diagonal(dim1=-2, dim2=-1)
            # print(log_vol_ratio_k)
            # print(ass_mat[k].transpose(-1,-2).shape, H.shape, W.shape, B_inv.shape)
            cut_k = ass_mat[k].transpose(-1, -2).matmul(H).matmul(W).matmul(B_inv).matmul(H.transpose(-1, -2)).matmul(
                1.0 - ass_mat[k])
            cut_k = cut_k.diagonal(dim1=-2, dim2=-1)
            # print(cut_k)
            se_loss_k = - cut_k * log_vol_ratio_k
            se_loss += se_loss_k.sum(dim=-1) / vol_H
        return se_loss.mean()

# old one with adaptable choice of hgnn_arch
class HyperHierarchicalGRL_Adaptable(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, avg_num_nodes, num_edges=None, height=3, decay_rate=0.5,
                 sym_D=False, EPS=1e-15, hgnn_arch='HGNN', act2=True):
        super(HyperHierarchicalGRL_Adaptable, self).__init__()

        self.height = height
        self.EPS = EPS
        self.hgnn_arch = hgnn_arch

        self.hyperconv_dict = {}
        self.pool_dict = {}
        num_nodes = avg_num_nodes
        if hgnn_arch == 'HGNN':
            for k in range(self.height, 1, -1):
                self.hyperconv_dict[k] = HGNNConvDense(in_channels, hidden_channels, sym_D=sym_D)
                # print(next(self.hyperconv_dict[k].theta.parameters()).device)
                num_nodes = ceil(decay_rate * num_nodes)
                self.pool_dict[k] = Linear(hidden_channels, num_nodes)
                in_channels = hidden_channels
            self.hyperconv_dict[1] = HGNNConvDense(hidden_channels, out_channels, sym_D=sym_D)
        elif hgnn_arch == 'HEAL':
            for k in range(self.height, 1, -1):
                self.hyperconv_dict[k] = HGNN_HEAL(num_edges, in_channels, act2=act2)
                # print(next(self.hyperconv_dict[k].theta.parameters()).device)
                num_nodes = ceil(decay_rate * num_nodes)
                self.pool_dict[k] = Linear(in_channels, num_nodes)
                # in_channels = hidden_channels
            self.hyperconv_dict[1] = HGNN_HEAL(num_edges, in_channels, act2=act2)
        else:
            raise NotImplementedError

        self.clu_mat = {}  # C
        self.vol_dict = {}
        self.sym_D = sym_D

    def forward(self, X, H, W, mask, temp=1.0):
        assert X.dim() == 3  # b*n*d
        assert H.dim() == 3  # b*n*k
        assert W.dim() == 2  # b*k
        assert mask.dim() == 2  # b*n
        H_input = H
        assert not torch.isnan(H).any()
        for k in range(self.height, 1, -1):
            bs, _num_v, _num_e = H.size()
            if k == self.height:
                mask = mask.view([bs, _num_v, 1]).to(H.dtype).to(H.device)
            else:
                mask = None
            if self.sym_D:
                HW = torch.einsum('bij,bj->bij', H, W)
                D = torch.einsum('bij,bjk->bik', HW, H.transpose(-1, -2))
                D = torch.einsum('bii->bi', D)
            else:
                D = torch.einsum('bij,bj->bi', H, W)
            Z = self.hyperconv_dict[k](X, H, W, D=D, mask=mask)
            C = self.pool_dict[k](Z)  # b*n_h*n_h-1
            C = torch.softmax(C / temp, dim=-1)
            H = C.transpose(-1, -2).matmul(H)
            X = C.transpose(-1, -2).matmul(Z)
            self.clu_mat[k] = C
            self.vol_dict[k] = D

        k = 1
        bs, _num_v, _num_e = H.size()
        if k == self.height:
            mask = mask.view([bs, _num_v, 1]).to(H.dtype).to(H.device)
        else:
            mask = None
        if self.sym_D:
            HW = torch.einsum('bij,bj->bij', H, W)
            D = torch.einsum('bij,bjk->bik', HW, H.transpose(-1, -2))
            D = torch.einsum('bii->bi', D)
        else:
            D = torch.einsum('bij,bj->bi', H, W)
        # assert not (D<0).any()
        assert not torch.isnan(D).any()
        # assert (D>=0).all()
        Z = self.hyperconv_dict[k](X, H, W, D=D, mask=mask)
        C = torch.ones([bs, C.shape[-1], 1]).to(H.dtype).to(X.device)
        self.clu_mat[k] = C
        self.vol_dict[k] = D

        # Z = torch.sum(Z, dim=-2)
        Z = torch.mean(Z, dim=-2)
        loss_hse = self.hse_loss(H_input, W, self.EPS)
        return Z, loss_hse

    def hse_loss(self, H, W, EPS):
        bs, _num_v, _num_e = H.size()
        B = torch.einsum('bij->bj', H)  # b*k
        B_inv = 1.0 / (B + EPS)
        B_inv[B_inv == float("inf")] = 0

        ass_mat = {}
        ass_mat[self.height] = torch.eye(_num_v).unsqueeze(0).repeat(bs, 1, 1).to(H.device)
        vol_dict = self.vol_dict
        clu_mat = self.clu_mat

        ass_mat[self.height - 1] = self.clu_mat[self.height]
        for k in range(self.height - 2, 0, -1):
            ass_mat[k] = ass_mat[k + 1].matmul(clu_mat[k + 1])

        hse_loss = torch.zeros([bs], device=H.device)
        vol_H = torch.einsum('bi->b', vol_dict[self.height])  # b
        vol_dict[0] = vol_H.unsqueeze(-1)
        for k in range(1, self.height + 1):
            if self.sym_D:
                CV = torch.einsum('bij,bj->bij', clu_mat[k], vol_dict[k - 1])
                vol_parent = torch.einsum('bij->bjk', CV, clu_mat[k].transpose(-1, -2))
                vol_parent = torch.einsum('bii->bi', vol_parent)
            else:
                vol_parent = torch.einsum('bij,bj->bi', clu_mat[k], vol_dict[k - 1])
            log_vol_ratio_k = torch.log2((vol_dict[k] + EPS) / (vol_parent + EPS))
            # if k == self.height:
            #     SH = H
            SH = torch.einsum('bij,bjk->bik', ass_mat[k].transpose(-1, -2), H)  # b*n*k
            HS = torch.einsum('bij,bjk->bik', H.transpose(-1, -2), (1.0 - ass_mat[k]))  # b*k*n
            WB = W * B_inv  # b*k
            SHWB = torch.einsum('bij,bj->bij', SH, WB)  # b*n*k
            # cut_k = torch.einsum('bij,bjk->bik', SHWB, HS)  # b*n*n
            # cut_k = torch.einsum('bii->bi', cut_k)  # b*n
            cut_k = torch.einsum('bij,bji->bi', SHWB, HS)
            hse_loss_k = - cut_k * log_vol_ratio_k  # b*n
            hse_loss += hse_loss_k.sum(dim=-1) / vol_H
        return hse_loss.mean()


class HierarchicalGRL(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, avg_num_nodes, height=3, decay_rate=0.5,
                 sym_D=False, EPS=1e-15):
        super(HierarchicalGRL, self).__init__()

        self.height = height
        self.EPS = EPS

        self.conv_dict = {}
        self.pool_dict = {}
        num_nodes = avg_num_nodes
        # mlp = MLP([in_channels, hidden_channels])
        # mlp = MLP([in_channels, hidden_channels, hidden_channels])
        # mlp = Sequential(
        #     Linear(in_channels, 2 * hidden_channels),
        #     # BatchNorm(2 * hidden_channels),
        #     ReLU(),
        #     Linear(2 * hidden_channels, hidden_channels),
        # )
        """
        self.conv_dict[self.height] = GINConv(
            Sequential(
                Linear(in_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                BN(hidden_channels),
            ), train_eps=False)
        num_nodes = ceil(decay_rate * num_nodes)
        self.pool_dict[self.height] = Linear(hidden_channels, num_nodes)
        for k in range(self.height - 1, 1, -1):
            self.conv_dict[k] = DenseGINConv(
                Sequential(
                    Linear(hidden_channels, hidden_channels),
                    ReLU(),
                    Linear(hidden_channels, hidden_channels),
                    ReLU(),
                    BN(hidden_channels),
                ), train_eps=False)
            num_nodes = ceil(decay_rate * num_nodes)
            self.pool_dict[k] = Linear(hidden_channels, num_nodes)
        self.conv_dict[1] = DenseGINConv(
            Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, out_channels),
                ReLU(),
                # BN(out_channels),
            ), train_eps=False)
        """
        self.conv_dict[self.height] = GCNConv(in_channels, hidden_channels)
        num_nodes = ceil(decay_rate * num_nodes)
        self.pool_dict[self.height] = Linear(hidden_channels, num_nodes)
        for k in range(self.height-1, 1, -1):
            self.conv_dict[k] = DenseGraphConv(hidden_channels, hidden_channels)
            num_nodes = ceil(decay_rate * num_nodes)
            self.pool_dict[k] = Linear(hidden_channels, num_nodes)
        self.conv_dict[1] = DenseGraphConv(hidden_channels, out_channels)


        self.clu_mat = {}  # C
        self.vol_dict = {}
        self.sym_D = sym_D

    def forward(self, X, edge_index, batch, temp=1.0):
        assert X.dim() == 2  # n*d
        assert batch.dim() == 1  # n

        A = to_dense_adj(edge_index, batch)
        A_input = A

        k = self.height
        Z = self.conv_dict[k](X, edge_index)
        Z = Z.relu()
        Z, mask = to_dense_batch(Z, batch)
        C = self.pool_dict[k](Z)  # b*n*k
        C = torch.softmax(C / temp, dim=-1)
        A = C.transpose(-1, -2).matmul(A).matmul(C)
        # A = A - torch.einsum('bii->bi', A)
        ind = torch.arange(A.shape[1], device=A.device)
        A[:, ind, ind] = 0
        D = torch.sum(A, dim=-1, keepdim=False)
        # D_invsqrt = 1.0 / (torch.sqrt(D) + EPS)
        # D_invsqrt[D_invsqrt == float("inf")] = 0
        # A = torch.einsum('')
        D = torch.sqrt(D)[:, None] + self.EPS
        A = (A / D) / D.transpose(-1, -2)
        X = C.transpose(-1, -2).matmul(Z)
        self.clu_mat[k] = C

        for k in range(self.height - 1, 1, -1):
            # print(X.shape)
            Z = self.conv_dict[k](X, A)
            Z = Z.relu()
            C = self.pool_dict[k](Z)
            C = torch.softmax(C / temp, dim=-1)
            A = C.transpose(-1, -2).matmul(A).matmul(C)
            # A = A - torch.einsum('bii->bi', A)
            ind = torch.arange(A.shape[1], device=A.device)
            A[:, ind, ind] = 0
            D = torch.sum(A, dim=-1, keepdim=False)
            D = torch.sqrt(D)[:, None] + self.EPS
            A = (A / D) / D.transpose(-1, -2)
            X = C.transpose(-1, -2).matmul(Z)
            self.clu_mat[k] = C

        k = 1
        bs = X.shape[0]
        Z = self.conv_dict[k](X, A)
        Z = Z.relu()
        C = torch.ones([bs, C.shape[-1], 1]).to(X.dtype).to(X.device)
        self.clu_mat[k] = C

        Z = torch.mean(Z, dim=-2)
        loss_se = self.se_loss(A_input, self.EPS)
        return Z, loss_se

    def se_loss(self, A, EPS):
        bs, _num_v, _num_v = A.size()
        # bs = A.shape[0]
        clu_mat = self.clu_mat
        # assert is_undirected(edge_index)
        # weights = torch.ones(edge_index.shape[1]).to(edge_index.device)
        # degrees = scatter_sum(weights, edge_index[0])  # n
        # degrees = degrees.unsqueeze(-1)  # n*1
        # assert degrees.dim() == 2
        # degrees, mask = to_dense_batch(degrees, batch)  # b*n*1
        # degrees = degrees.squeeze(-1) * mask  # b*n
        degrees = torch.einsum('bij->bi', A) # b*n

        ass_mat = {}
        vol_dict = {}
        vol_dict[self.height] = degrees

        ass_mat[self.height] = torch.eye(_num_v).unsqueeze(0).repeat(bs, 1, 1).to(A.device)
        ass_mat[self.height - 1] = self.clu_mat[self.height]
        # print(degrees.shape, ass_mat[self.height-1].shape)
        vol_dict[self.height - 1] = torch.einsum('bij,bi->bj', ass_mat[self.height - 1], degrees)
        for k in range(self.height - 2, 0, -1):
            ass_mat[k] = ass_mat[k + 1].matmul(clu_mat[k + 1])  # b*n_k*n_k-1
            vol_dict[k] = torch.einsum('bij,bi->bj', ass_mat[k], degrees)  # b*n_k


        se_loss = torch.zeros([bs], device=A.device)
        vol_G = torch.einsum('bi->b', vol_dict[self.height])
        vol_dict[0] = vol_G.unsqueeze(-1)
        for k in range(1, self.height + 1):
            vol_parent = torch.einsum('bij,bj->bi', clu_mat[k], vol_dict[k - 1])  # b*n
            log_vol_ratio_k = torch.log2((vol_dict[k] + EPS) / (vol_parent + EPS))
            # SA = torch.einsum('bij,bjk->bik', ass_mat[k].transpose(-2,-1), A) # b*n_k*n
            # SAS = torch.einsum('bij,bjk->bik', SA, ass_mat[k]) # b*n_k*n_k
            SAS = ass_mat[k].transpose(-2, -1).matmul(A).matmul(ass_mat[k])  # b*n_k*n_k
            links = torch.einsum('bii->bi', SAS)  # b*n_k
            delta_vol = vol_dict[k] - links  # b*n_k
            se_loss_k = - delta_vol * log_vol_ratio_k
            se_loss += se_loss_k.sum(dim=-1) / vol_G
        return se_loss.mean()





class HyperStructLearning(torch.nn.Module):
    def __init__(self, out_channels_gnn, num_edges2):
        super(HyperStructLearning, self).__init__()
        self.out_channels_gnn = out_channels_gnn
        self.num_edges2 = num_edges2
        self.W = Parameter(torch.Tensor(self.out_channels_gnn, self.num_edges2),
                           requires_grad=True)  # should I just use a MLP instead?

    def reset_parameters(self):
        # super().reset_parameters()
        # self.W.reset_parameters()
        torch.nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    # the input embedding_gnn should be GNN Encoder output
    def forward(self, embedding_gnn):
        embedding_gnn = embedding_gnn.unsqueeze(0) if embedding_gnn.dim() == 2 else embedding_gnn
        H = torch.matmul(embedding_gnn, self.W / self.out_channels_gnn)
        exit(0)
        # sigmoid ??
        H = torch.clamp(H, 0)
        return H


class HypSEEDense(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_gnn, hidden_channels, out_channels, num_layers_gnn, num_edges2,
                 avg_num_nodes):
        super(HypSEEDense, self).__init__()
        if hidden_channels_gnn == 0:
            hidden_channels_gnn = in_channels
        self.gnn_encoder = torch.nn.ModuleList()
        in_channels_gnn = in_channels
        for _ in range(num_layers_gnn - 1):
            mlp = MLP([in_channels_gnn, hidden_channels_gnn, hidden_channels_gnn])
            self.gnn_encoder.append(DenseGINConv(nn=mlp, train_eps=False))
            in_channels_gnn = hidden_channels_gnn
        mlp = MLP([in_channels_gnn, hidden_channels_gnn, in_channels])
        self.gnn_encoder.append(DenseGINConv(nn=mlp, train_eps=False))

        self.hyper_struct_learning = HyperStructLearning(in_channels, num_edges2)

        self.hyper_hierarchical_GRL = HyperHierarchicalGRLDense(in_channels, hidden_channels, hidden_channels,
                                                                avg_num_nodes)

        self.classifier = Sequential(Linear(hidden_channels, hidden_channels),
                                     Linear(hidden_channels, out_channels))

    def reset_parameters(self):
        self.hyper_struct_learning.reset_parameters()

    def forward(self, X, adj, H_S, mask):
        assert X.dim() == 3
        bs, _num_v, _ = X.size()
        W_S = torch.eye(H_S.shape[-1]).unsqueeze(0).repeat(X.shape[0], 1, 1).to(X.device)
        Z_S = loss_hse_S = self.hyper_hierarchical_GRL(X, H_S, W_S)

        embedding_gnn_T = X
        for conv in self.gnn_encoder:
            embedding_gnn_T = conv(embedding_gnn_T, adj).relu()
        H_T = self.hyper_struct_learning(embedding_gnn_T)
        W_T = torch.eye(H_T.shape[-1]).unsqueeze(0).repeat(X.shape[0], 1, 1).to(X.device)
        Z_T, loss_hse_T = self.hyper_hierarchical_GRL(embedding_gnn_T, H_T, W_T)
        Z = torch.cat([Z_S, Z_T], dim=-1)
        out = self.classifier(Z)
        return Z_S, Z_T, out, loss_hse_S + loss_hse_T

    def loss_con(self, Su, Sm, Tu, Tm):
        EPS = 1e-8
        T = 0.5

        Su_abs = torch.clamp(torch.linalg.norm(Su, dim=1), min=EPS)
        Sm_abs = torch.clamp(torch.linalg.norm(Sm, dim=1), min=EPS)
        Tu_abs = torch.clamp(torch.linalg.norm(Tu, dim=1), min=EPS)
        Tm_abs = torch.clamp(torch.linalg.norm(Tm, dim=1), min=EPS)

        sim_matrix_P = (torch.einsum('ik,jk->ij', Su, Sm) / torch.einsum('i,j->ij', Su_abs, Sm_abs)) / T
        Log_Pu = torch.log_softmax(sim_matrix_P, dim=-1)
        sim_matrix_Q = (torch.einsum('ik,jk->ij', Tu, Tm) / torch.einsum('i,j->ij', Tu_abs, Tm_abs)) / T
        Log_Qu = torch.log_softmax(sim_matrix_Q, dim=-1)
        KL = F.kl_div(Log_Pu, Log_Qu, reduction="batchmean", log_target=True) + F.kl_div(Log_Qu, Log_Pu,
                                                                                         reduction="batchmean",
                                                                                         log_target=True)
        # print(KL)
        return KL / 2


# Sparse Implementation, not code reviewed.
# hierarchical representation learning (via hypergraph structural entropy pooling)
class HyperHierarchicalGRLSparse(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, avg_num_nodes, decay_rate=0.5, sym_D=True, EPS=1e-15):
        super(HyperHierarchicalGRLSparse, self).__init__()

        self.EPS = EPS

        self.hyperconv3 = HGNNConv(in_channels, hidden_channels, sym_D=sym_D)
        num_nodes = ceil(decay_rate * avg_num_nodes)
        self.pool3 = Linear(hidden_channels, num_nodes)

        self.hyperconv2 = HGNNConv(hidden_channels, hidden_channels, sym_D=sym_D)
        num_nodes = ceil(decay_rate * num_nodes)
        self.pool2 = Linear(hidden_channels, num_nodes)

        self.hyperconv1 = HGNNConv(hidden_channels, out_channels, sym_D=sym_D)

        self.height = 3
        self.clu_mat = {}  # C
        self.vol_dict = {}
        self.sym_D = sym_D

    def forward(self, X, H, W, temp=1.0):
        assert X.dim() == 3

        X3 = X
        H3 = H
        bs, _num_v, _num_e = H3.size()
        if self.sym_D:
            D3 = H3.matmul(W).matmul(H3.transpose(-1, -2))
        else:
            D3 = H3.matmul(W).matmul(torch.ones([bs, _num_e, _num_v]))
        Z3 = self.hyperconv3(X3, H3, W, D=D3)
        C3 = self.pool3(Z3)
        C3 = torch.softmax(C3 / temp, dim=-1)
        H2 = C3.transpose(-1, -2).matmul(H3)
        X2 = C3.transpose(-1, -2).matmul(Z3)

        bs, _num_v, _num_e = H2.size()
        if self.sym_D:
            D2 = H2.matmul(W).matmul(H2.transpose(-1, -2))
        else:
            D2 = H2.matmul(W).matmul(torch.ones([bs, _num_e, _num_v]))
        Z2 = self.hyperconv2(X2, H2, W, D=D2)
        C2 = self.pool2(Z2)
        C2 = torch.softmax(C2 / temp, dim=-1)
        H1 = C2.transpose(-1, -2).matmul(H2)
        X1 = C2.transpose(-1, -2).matmul(Z2)

        bs, _num_v, _num_e = H1.size()
        if self.sym_D:
            D1 = H1.matmul(W).matmul(H1.transpose(-1, -2))
        else:
            D1 = H1.matmul(W).matmul(torch.ones([bs, _num_e, _num_v]))
        Z1 = self.hyperconv1(X1, H1, W, D=D1)
        # C1 is the assignment matrix between L1 nodes and root, thus a matrix full of 1s.
        C1 = torch.ones([bs, C2.shape[-1], 1]).to(H.dtype).to(X.device)

        self.clu_mat[3] = C3
        self.clu_mat[2] = C2
        self.clu_mat[1] = C1
        self.vol_dict[3] = D3
        self.vol_dict[2] = D2
        self.vol_dict[1] = D1

        # return Z3, Z2, Z1
        Z = torch.sum(Z1, dim=-2)
        loss_hse = self.hse_loss(H, W)
        return Z, loss_hse

    def hse_loss(self, H, W):
        assert len(self.clu_mat) > 0
        bs, _num_v, _num_e = H.size()
        B = torch.ones([bs, _num_e, _num_v]).to(H.device).matmul(H)
        B_inv = 1.0 / B
        B_inv[B_inv == float("inf")] = 0

        ass_mat = {self.height: torch.eye(_num_v).unsqueeze(0).repeat(bs, 1, 1).to(H.device)}
        vol_dict = self.vol_dict
        clu_mat = self.clu_mat
        for k in range(self.height - 1, 0, -1):
            ass_mat[k] = ass_mat[k + 1].matmul(self.clu_mat[k + 1])

        se_loss = torch.zeros([bs], device=H.device)
        vol_H = vol_dict[self.height].diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=False)
        vol_dict[0] = vol_H.unsqueeze(-1).unsqueeze(-1)
        for k in range(1, self.height + 1):
            if self.sym_D:
                vol_parent = clu_mat[k].matmul(vol_dict[k - 1]).matmul(clu_mat[k].transpose(-1, -2))
            else:
                vol_parent = clu_mat[k].matmul(vol_dict[k - 1]).matmul(
                    torch.ones([bs, vol_dict[k - 1].shape[-1], clu_mat[k].shape[-2]]))
            log_vol_ratio_k = torch.log2((vol_dict[k] + self.EPS) / (vol_parent + self.EPS))
            log_vol_ratio_k = log_vol_ratio_k.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            # print(ass_mat[k].transpose(-1,-2).shape, H.shape, W.shape, B_inv.shape)
            cut_k = ass_mat[k].transpose(-1, -2).matmul(H).matmul(W).matmul(B_inv).matmul(H.transpose(-1, -2)).matmul(
                1.0 - ass_mat[k])
            cut_k = cut_k.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            se_loss_k = - cut_k * log_vol_ratio_k / vol_H
            se_loss += se_loss_k
        return se_loss.mean()
