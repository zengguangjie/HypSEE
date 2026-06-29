"""ClusterNet assigner: two-step soft k-means on embedder outputs."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def cluster(
    data: torch.Tensor,
    k: int,
    num_iter: int,
    *,
    init: torch.Tensor,
    tau: float,
    eps: float = 1e-15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable soft k-means on the unit sphere. Returns cluster means, soft assignments, and distances."""
    data = F.normalize(data, p=2, dim=-1)
    mu = init
    for _ in range(num_iter):
        dist = data @ mu.t()
        r = torch.softmax(dist / tau, dim=-1)
        cluster_r = r.sum(dim=-2)
        cluster_mean = r.transpose(-1, -2) @ data
        mu = cluster_mean / cluster_r.unsqueeze(-1).clamp(min=eps)
    dist = data @ mu.t()
    r = torch.softmax(dist / tau, dim=-1)
    return mu, r, dist


def _batched_cluster_init(init: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Broadcast init to (batch, k, embed_dim). Accepts (k, d) or (batch, k, d)."""
    if init.dim() == 2:
        return init.unsqueeze(0).expand(batch_size, -1, -1)
    if init.dim() == 3:
        return init
    raise ValueError(
        f"init must have shape (k, embed_dim) or (batch, k, embed_dim), got {tuple(init.shape)}"
    )


def cluster_batched(
    data: torch.Tensor,
    k: int,
    num_iter: int,
    *,
    init: torch.Tensor,
    tau: float,
    eps: float = 1e-15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched soft k-means. data: (batch, num_nodes, embed_dim), init: (k, embed_dim)."""
    data = F.normalize(data, p=2, dim=-1)
    mu = _batched_cluster_init(init, data.size(0))
    for _ in range(num_iter):
        dist = torch.bmm(data, mu.transpose(1, 2))
        r = torch.softmax(dist / tau, dim=-1)
        cluster_r = r.sum(dim=1, keepdim=True).transpose(1, 2)
        cluster_mean = torch.bmm(r.transpose(1, 2), data)
        mu = cluster_mean / cluster_r.clamp(min=eps)
    dist = torch.bmm(data, mu.transpose(1, 2))
    r = torch.softmax(dist / tau, dim=-1)
    return mu, r, dist


class ClusterNetAssigner(torch.nn.Module):
    """ClusterNet assignment head: embedder output + two-step soft k-means."""

    def __init__(self, embed_dim: int, num_clusters: int, num_iter: int = 1, eps: float = 1e-15):
        super().__init__()
        self.K = num_clusters
        self.num_iter = num_iter
        self.eps = eps
        self.init = torch.nn.Parameter(torch.rand(num_clusters, embed_dim))

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        if x.dim() == 2:
            return self._forward_single(x, tau)
        if x.dim() == 3:
            return self._forward_batched(x, tau)
        raise ValueError(
            f"expected x with shape (num_nodes, embed_dim) or "
            f"(batch, num_nodes, embed_dim), got {tuple(x.shape)}"
        )

    def _forward_single(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        mu_init, _, _ = cluster(
            x, self.K, self.num_iter, init=self.init, tau=tau, eps=self.eps,
        )
        _, _, dist = cluster(
            x, self.K, 1, init=mu_init.detach().clone(), tau=tau, eps=self.eps,
        )
        return dist

    def _forward_batched(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        mu_init, _, _ = cluster_batched(
            x, self.K, self.num_iter, init=self.init, tau=tau, eps=self.eps,
        )
        _, _, dist = cluster_batched(
            x, self.K, 1, init=mu_init.detach().clone(), tau=tau, eps=self.eps,
        )
        return dist


class ClusterNetPoolLayer(torch.nn.Module):
    """Hierarchical pool layer: ClusterNet soft k-means assignment from nodes to clusters."""

    def __init__(self, hidden_channels: int, num_clusters: int, num_iter: int = 1, eps: float = 1e-15):
        super().__init__()
        self.assigner = ClusterNetAssigner(
            hidden_channels, num_clusters, num_iter=num_iter, eps=eps)

    def reset_parameters(self):
        torch.nn.init.uniform_(self.assigner.init, 0, 1)

    def forward(
        self,
        Z: torch.Tensor,
        H: torch.Tensor | None = None,
        W: torch.Tensor | None = None,
        D: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        tau: float = 1.0,
    ) -> torch.Tensor:
        """Return cosine-similarity logits, shape (batch, num_nodes, num_clusters)."""
        del H, W, D, mask
        return self.assigner(Z, tau=tau)


class LinearPoolLayer(torch.nn.Module):
    """Hierarchical pool layer: linear projection to cluster assignment logits."""

    def __init__(self, hidden_channels: int, num_clusters: int, num_iter: int = 1):
        super().__init__()
        del num_iter
        self.lin = torch.nn.Linear(hidden_channels, num_clusters)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(
        self,
        Z: torch.Tensor,
        H: torch.Tensor | None = None,
        W: torch.Tensor | None = None,
        D: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        tau: float = 1.0,
    ) -> torch.Tensor:
        """Return assignment logits, shape (batch, num_nodes, num_clusters)."""
        del H, W, D, mask, tau
        return self.lin(Z)


class UniGATPoolLayer(torch.nn.Module):
    """Hierarchical pool layer: UniGATConv on the current hypergraph incidence H."""

    def __init__(
        self,
        hidden_channels: int,
        num_clusters: int,
        num_iter: int = 1,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        eps: float = 1e-15,
    ):
        super().__init__()
        del num_iter
        from models.UniGCNII import UniGATConv

        self.num_clusters = num_clusters
        self.conv = UniGATConv(
            hidden_channels,
            num_clusters,
            heads=1,
            dropout=dropout,
            negative_slope=negative_slope,
            eps=eps,
        )

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(
        self,
        Z: torch.Tensor,
        H: torch.Tensor | None = None,
        W: torch.Tensor | None = None,
        D: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        tau: float = 1.0,
    ) -> torch.Tensor:
        """Return assignment logits, shape (batch, num_nodes, num_clusters)."""
        del W, D, tau
        if H is None:
            raise ValueError("UniGATPoolLayer requires hypergraph incidence H")

        from models.hgnn import dense_incidence_to_sparse

        bs, n, _ = Z.shape
        if mask is not None:
            node_mask = mask.view(bs, n, 1).to(dtype=Z.dtype, device=Z.device)
            Z = Z * node_mask

        sparse_hg = dense_incidence_to_sparse(H, mask=mask)
        x = Z.reshape(bs * n, -1)
        logits = self.conv(
            x,
            sparse_hg.hyperedge_index,
            edge_entry_weight=sparse_hg.edge_entry_weight,
        )
        logits = logits.view(bs, n, self.num_clusters)
        if mask is not None:
            logits = logits * mask.view(bs, n, 1).to(dtype=logits.dtype, device=logits.device)
        return logits
