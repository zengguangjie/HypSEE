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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable soft k-means on the unit sphere. Returns cluster means, soft assignments, and distances."""
    data = F.normalize(data, p=2, dim=1)
    mu = init
    for _ in range(num_iter):
        dist = data @ mu.t()
        r = torch.softmax(dist / tau, dim=1)
        cluster_r = r.sum(dim=0)
        cluster_mean = r.t() @ data
        mu = cluster_mean / cluster_r.unsqueeze(1).clamp(min=1e-8)
    dist = data @ mu.t()
    r = torch.softmax(dist / tau, dim=1)
    return mu, r, dist


class ClusterNetAssigner(torch.nn.Module):
    """ClusterNet assignment head: embedder output + two-step soft k-means."""

    def __init__(self, embed_dim: int, num_clusters: int, num_iter: int = 1):
        super().__init__()
        self.K = num_clusters
        self.num_iter = num_iter
        self.init = torch.nn.Parameter(torch.rand(num_clusters, embed_dim))

    def forward(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        mu_init, _, _ = cluster(
            x, self.K, self.num_iter, init=self.init, tau=tau,
        )
        _, _, dist = cluster(
            x, self.K, 1, init=mu_init.detach().clone(), tau=tau,
        )
        return dist
