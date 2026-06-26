import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if torch.isfinite(tensor).all():
        return
    nan_count = int(torch.isnan(tensor).sum())
    inf_count = int(torch.isinf(tensor).sum())
    finite = tensor[torch.isfinite(tensor)]
    stat = (
        f"min={finite.min().item():.4g}, max={finite.max().item():.4g}"
        if finite.numel() > 0 else "no finite values"
    )
    raise AssertionError(
        f"{name}: shape={tuple(tensor.shape)}, nan={nan_count}, inf={inf_count}, {stat}"
    )

class HGNNConvDense(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, use_bn=False, drop_rate=0.5, is_last=False, sym_D=False):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        self.sym_D = sym_D  # if sym_D, then D is calculated as D = HWH^T.diagonal()

    def forward_denseWDB(self, X, H, W, D=None, B=None, mask=None):
        assert torch.all(H>=0)

        assert X.dim() == 3
        bs, _num_v, _num_e = H.size()

        if mask is None:
            mask = torch.ones([bs, _num_v, 1]).to(H.dtype).to(H.device)
        else:
            mask = mask.view([bs, _num_v, 1]).to(H.dtype).to(H.device)
        # assert not torch.isinf(mask).any()


        if D is None:
            if self.sym_D:
                D = H.matmul(W).matmul(H.transpose(-1,-2))
            else:
                D = H.matmul(W).matmul(torch.ones([bs, _num_e, _num_v]).to(H.dtype).to(H.device))
        eye = torch.eye(_num_v).unsqueeze(0).repeat(bs, 1, 1).to(H.dtype).to(H.device)
        # D_inv = 1.0 / D.clamp(EPS)

        # D_invsqrt = eye / (torch.sqrt(D.clamp(1e-15)) + EPS)
        D_invsqrt = eye / (torch.sqrt(D) + EPS)
        D_invsqrt = D_invsqrt * mask
        D_invsqrt[D_invsqrt == float("inf")] = 0


        if B is None:
            B = torch.ones([bs, _num_e, _num_v]).to(X.device).matmul(H)
        eye = torch.eye(_num_e).unsqueeze(0).repeat(bs, 1, 1).to(H.dtype).to(H.device)
        # B_inv = 1.0 / B.clamp(EPS)
        B_inv = eye / (B + EPS)
        B_inv[B_inv == float("inf")] = 0
        # print(torch.min(B_inv), torch.max(B_inv), "B_inv")
        # X = X * mask
        # D_inv = D_inv * mask
        # H = H * mask
        # print(torch.min(D), torch.max(D))
        # print(torch.min(D_invsqrt), torch.max(D_invsqrt), "D_invsqrt")
        # print(X.device, self.theta.device())
        X = self.theta(X)

        L_HGNN_row = D_invsqrt.matmul(H.matmul(W.matmul(B_inv)).matmul(H.transpose(-1,-2))).matmul(D_invsqrt)
        # print(torch.min(L_HGNN_row), torch.max(L_HGNN_row), "L_HGNN_row")
        # print(torch.min(X), torch.max(X), "X")
        Z = L_HGNN_row.matmul(X)
        # print(torch.min(Z), torch.max(Z), "Z")
        if not self.is_last:
            Z = self.act(Z)
            if self.bn is not None:
                Z = self.bn(Z)
            Z = self.drop(Z)
        Z = Z * mask
        return Z

    def forward(self, X, H, W, D=None, B=None, mask=None):
        assert X.dim() == 3  # b*n*d
        assert H.dim() == 3  # b*n*k
        assert W.dim() == 2  # b*k
        bs, _num_v, _num_e = H.size()

        if mask is None:
            mask = torch.ones([bs, _num_v, 1]).to(H.dtype).to(H.device)
        else:
            mask = mask.view([bs, _num_v, 1]).to(H.dtype).to(H.device)

        if D is None:
            if self.sym_D:
                HW = torch.einsum('bij,bj->bij', H, W)
                D = torch.einsum('bij,bjk->bik', HW, H.transpose(-1,-2))
                D = torch.einsum('bii->bi', D)
            else:
                D = torch.einsum('bij,bj->bi', H, W)
        # print(D.shape, mask.shape)
        assert not (D<0).any()
        assert not torch.isnan(D).any()
        # assert (D>=0).all()
        D_invsqrt = torch.ones_like(D) / (torch.sqrt(D) + EPS)
        D_invsqrt = D_invsqrt * mask.squeeze(-1)
        D_invsqrt[D_invsqrt == float("inf")] = 0

        if B is None:
            B = torch.einsum('bij->bj', H)  # b*k
        B_inv = torch.ones_like(B) / (B + EPS)
        B_inv[B_inv == float("inf")] = 0

        X = self.theta(X) # b*n*d

        WB = W * B_inv # b*k
        HWB = torch.einsum('bij,bj->bij', H, WB) # b*n*k
        HWBH = torch.einsum('bij,bjk->bik', HWB, H.transpose(-1,-2)) # b*n*n
        DHWBH = torch.einsum('bi,bij->bij', D_invsqrt, HWBH) # b*n*n
        L_HGNN = torch.einsum('bij,bj->bij', DHWBH, D_invsqrt) # b*n*n
        Z = L_HGNN.matmul(X) # b*n*d
        if not self.is_last:
            Z = self.act(Z)
            if self.bn is not None:
                Z = self.bn(Z)
            Z = self.drop(Z)
        Z = Z * mask
        return Z

class HGNN_HEAL(nn.Module):
    def __init__(self, num_edges, in_channels, bias=True, use_bn=False, drop_rate=0.5, is_last=False, sym_D=False, act2=False):
        super().__init__()
        # out_channels = in_channels
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(in_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(num_edges, num_edges, bias=bias)
        self.sym_D = sym_D  # if sym_D, then D is calculated as D = HWH^T.diagonal()
        self.act2 = nn.ReLU(inplace=True) if act2 else None

    def forward(self, X, H, W=None, D=None, B=None, mask=None):

        _assert_finite("HGNN_HEAL.H", H)

        bs, _num_v, _num_e = H.size()
        if mask is None:
            mask = torch.ones([bs, _num_v, 1]).to(H.dtype).to(H.device)
        else:
            mask = mask.view([bs, _num_v, 1]).to(H.dtype).to(H.device)

        if D is None:
            if self.sym_D:
                HW = torch.einsum('bij,bj->bij', H, W)
                D = torch.einsum('bij,bjk->bik', HW, H.transpose(-1,-2))
                D = torch.einsum('bii->bi', D)
            else:
                D = torch.einsum('bij,bj->bi', H, W)
        D = D.clamp(min=EPS)
        D_invsqrt = torch.ones_like(D) / (torch.sqrt(D) + EPS)
        D_invsqrt = D_invsqrt * mask.squeeze(-1)
        D_invsqrt[D_invsqrt == float("inf")] = 0

        _assert_finite("HGNN_HEAL.D_invsqrt", D_invsqrt)

        if B is None:
            B = torch.einsum('bij->bj', H)  # b*k
        B = B.clamp(min=EPS)
        B_inv = torch.ones_like(B) / (B + EPS)
        B_inv[B_inv == float("inf")] = 0

        _assert_finite("HGNN_HEAL.B_inv", B_inv)

        WB = W * B_inv
        H = torch.einsum('bi,bij->bij', D_invsqrt, H)

        _assert_finite("HGNN_HEAL.H_scaled", H)
        _assert_finite("HGNN_HEAL.X", X)

        HX = H.transpose(-2,-1).matmul(X)

        _assert_finite("HGNN_HEAL.HX", HX)

        R = self.theta(HX.transpose(-2,-1))

        _assert_finite("HGNN_HEAL.R", R)

        R = self.act(R.transpose(-2,-1))
        R = R + HX
        R = torch.einsum('bi,bij->bij', WB, R)
        Z = H.matmul(R)
        Z = torch.einsum('bi,bij->bij', D_invsqrt, Z)
        if not self.is_last:
            if self.act2 is not None:
                Z = self.act2(Z)
            if self.bn is not None:
                Z = self.bn(Z)
            Z = self.drop(Z)
        Z = Z * mask
        return Z




# sparse HGNNConv
class HGNNConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, use_bn=False, drop_rate=0.5, is_last=False, sym_D=True):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        self.sym_D = sym_D  # if sym_D, then D is calculated as D = HWH^T

    def forward(self, X, H_sparse, W_sparse, D_sparse=None, B_sparse=None):
        if D_sparse is None:
            if self.sym_D:
                # D_sparse = torch.sparse.mm(torch.sparse.mm(H_sparse, W_sparse), H_sparse.transpose(-1, -2))
                D_sparse = H_sparse.matmul(W_sparse).matmul(H_sparse.transpose(-1, -2))
            else:
                ones = torch.ones(H_sparse.size()).transpose().to_sparse().to(H_sparse.device)
                # D_sparse = torch.sparse.mm(torch.sparse.mm(H_sparse, W_sparse), ones)
                D_sparse = H_sparse.matmul(W_sparse).matmul(ones)
        # D_inv = 1.0 / D_sparse
        # D_inv[D_inv == float("inf")] = 0
        values = 1.0 / D_sparse.values()
        values = torch.where(values == float("inf"), torch.tensor(0.0), values)
        D_inv = torch.sparse_coo_tensor(D_sparse.indices(), values, D_sparse.shape)

        if B_sparse is None:
            ones = torch.ones(H_sparse.size()).transpose().to_sparse().to(H_sparse.device)
            B_sparse = torch.sparse.mm(ones, H_sparse)
        # B_inv = 1.0 / B_sparse
        # B_inv[B_inv == float("inf")] = 0
        values = 1.0 / B_sparse.values()
        values = torch.where(values == float("inf"), torch.tensor(0.0), values)
        B_inv = torch.sparse_coo_tensor(B_sparse.indices(), values, B_sparse.shape)

        X = self.theta(X)

        L_HGNN_row = D_inv.matmul(H_sparse).matmul(W_sparse).matmul(B_inv).matmul(H_sparse.t())
        Z = L_HGNN_row.matmul(X)
        if not self.is_last:
            Z = self.act(Z)
            if self.bn is not None:
                Z = self.bn(Z)
            Z = self.drop(Z)
        return Z

