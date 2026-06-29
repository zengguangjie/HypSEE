import copy
import re
from typing import Optional, Union, cast

import numpy as np
import torch
import torch_geometric.transforms as T
import torch_geometric.utils as tg_utils
from torch_geometric.data.data import BaseData
from torch_geometric.data.separate import separate
from torch_geometric.datasets import TUDataset

from datasets.feature_expansion import FeatureExpander


def _default_aug_device(device: Optional[Union[torch.device, str]] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _ensure_num_nodes(data: BaseData) -> BaseData:
    """Ensure num_nodes is stored explicitly for PyG batch collation."""
    if data.x is not None:
        data.num_nodes, _ = data.x.shape
    return data


def _cpu_adj_list(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Build an adjacency list on CPU for subgraph sampling (NetworkX-free)."""
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    ei = edge_index.detach().cpu().numpy()
    for u, v in zip(ei[0], ei[1]):
        u_i, v_i = int(u), int(v)
        adj[u_i].append(v_i)
    return adj


def _parse_feat_str(feat_str):
    tokens = set(feat_str.split("+")) if feat_str else set()
    degree = "deg" in tokens
    onehot_maxdeg = re.findall(r"odeg(\d+)", feat_str)
    onehot_maxdeg = int(onehot_maxdeg[0]) if onehot_maxdeg else None
    k = re.findall(r"an{0,1}k(\d+)", feat_str)
    k = int(k[0]) if k else 0
    centrality = "cent" in tokens
    return degree, onehot_maxdeg, k, centrality


def get_dataset(name, root=None, feat_str="deg") -> "TUDatasetExt":
    degree, onehot_maxdeg, k, centrality = _parse_feat_str(feat_str)
    pre_transform = FeatureExpander(
        degree=degree,
        onehot_maxdeg=onehot_maxdeg or 0,
        AK=k,
        centrality=centrality,
    ).transform
    return TUDatasetExt(
        root,
        name,
        pre_transform=pre_transform,
        use_node_attr=True,
        transform=None,
        pre_filter=None,
        processed_filename=f"data_{feat_str}.pt",
    )


def subset_dataset(dataset: "TUDatasetExt", index) -> "TUDatasetExt":
    """Return a dataset slice with a concrete type for static checkers."""
    return cast(TUDatasetExt, dataset[index])


def shuffle_dataset(dataset: "TUDatasetExt") -> "TUDatasetExt":
    """Return a shuffled dataset with a concrete type for static checkers."""
    return cast(TUDatasetExt, dataset.shuffle())


def get_dataset_addgraph(name, root="data"):
    return TUDataset(
        root,
        name,
        pre_transform=AddGraphIdTransform(),
        use_node_attr=True,
        transform=None,
        pre_filter=None,
    )


def get_dataset_dense(name, root="data", max_nodes=150):
    return TUDataset(
        root,
        name,
        pre_transform=None,
        use_node_attr=True,
        transform=T.ToDense(num_nodes=None),
        pre_filter=lambda data: data.num_nodes <= max_nodes,
    )


class AddGraphIdTransform:
    def __init__(self):
        self.graph_id = 0

    def __call__(self, data):
        data.graph_id = self.graph_id
        self.graph_id += 1
        return data


def node_drop(data, aug_ratio):
    node_num, _ = data.x.size()
    device = data.x.device
    drop_num = int(node_num * aug_ratio)

    idx_nondrop = torch.randperm(node_num, device=device)[drop_num:].sort().values

    edge_index, _ = tg_utils.subgraph(
        idx_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num
    )
    data.x = data.x[idx_nondrop]
    data.edge_index = edge_index
    data.num_nodes, _ = data.x.shape
    return data


def weighted_drop_nodes(data, aug_ratio, npower):
    node_num, _ = data.x.size()
    device = data.x.device
    drop_num = int(node_num * aug_ratio)
    if drop_num <= 0:
        return data
    if drop_num >= node_num:
        drop_num = node_num - 1

    row = data.edge_index[0]
    deg = torch.zeros(node_num, device=device, dtype=torch.float)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=device))
    deg = deg.clamp(min=0.1).pow(npower)
    idx_drop = torch.multinomial(deg, drop_num, replacement=False)
    keep_mask = torch.ones(node_num, dtype=torch.bool, device=device)
    keep_mask[idx_drop] = False
    idx_nondrop = keep_mask.nonzero(as_tuple=False).view(-1)

    edge_index, _ = tg_utils.subgraph(
        idx_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num
    )
    data.x = data.x[idx_nondrop]
    data.edge_index = edge_index
    data.num_nodes, _ = data.x.shape
    return data


def edge_pert(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    pert_num = int(edge_num * aug_ratio)
    device = data.x.device

    if pert_num <= 0 or edge_num <= 0:
        return data

    keep_count = max(edge_num - pert_num, 0)
    if keep_count > 0:
        keep_idx = torch.randperm(edge_num, device=device)[:keep_count]
        edge_index = data.edge_index[:, keep_idx]
    else:
        edge_index = data.edge_index.new_empty((2, 0))

    idx_add = torch.randint(0, node_num, (2, pert_num), device=device)
    adj = torch.zeros((node_num, node_num), device=device)
    if edge_index.numel() > 0:
        adj[edge_index[0], edge_index[1]] = 1
    adj[idx_add[0], idx_add[1]] = 1
    adj.fill_diagonal_(0)
    data.edge_index = adj.nonzero(as_tuple=False).t()
    data.num_nodes, _ = data.x.shape
    return data


def subgraph(data, aug_ratio):
    node_num, _ = data.x.size()
    device = data.x.device
    sub_num = max(int(node_num * (1 - aug_ratio)), 1)

    adj = _cpu_adj_list(data.edge_index, node_num)
    idx_sub = [int(torch.randint(0, node_num, (1,), device=device).item())]
    idx_neigh = set(adj[idx_sub[-1]])

    while len(idx_sub) < sub_num:
        if len(idx_neigh) == 0:
            idx_unsub = list(set(range(node_num)) - set(idx_sub))
            idx_neigh = {int(np.random.choice(idx_unsub))}
        sample_node = int(np.random.choice(list(idx_neigh)))
        idx_sub.append(sample_node)
        idx_neigh = idx_neigh.union(set(adj[idx_sub[-1]])).difference(set(idx_sub))

    idx_nondrop = torch.tensor(sorted(idx_sub), device=device, dtype=torch.long)
    edge_index, _ = tg_utils.subgraph(
        idx_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num
    )
    data.x = data.x[idx_nondrop]
    data.edge_index = edge_index
    data.num_nodes, _ = data.x.shape
    return data


def attr_mask(data, aug_ratio):
    node_num, _ = data.x.size()
    device = data.x.device
    mask_num = int(node_num * aug_ratio)
    if mask_num <= 0:
        return data

    _x = data.x.clone()
    token = data.x.mean(dim=0)
    idx_mask = torch.randperm(node_num, device=device)[:mask_num]
    _x[idx_mask] = token
    data.x = _x
    data.num_nodes, _ = data.x.shape
    return data


def maskN_permE(data, aug_ratio):
    data = attr_mask(data, aug_ratio)
    return edge_pert(data, aug_ratio)


_AUGMENTATIONS = {
    "dropN": node_drop,
    "permE": edge_pert,
    "subgraph": subgraph,
    "maskN": attr_mask,
    "maskN_permE": maskN_permE,
}

_RANDOM_AUGMENTATIONS = {
    "random2": (node_drop, subgraph),
    "random3": (node_drop, subgraph, edge_pert),
    "random4": (node_drop, subgraph, edge_pert, attr_mask),
    "random_mask_permE": (attr_mask, edge_pert),
}


def apply_augmentation(data, aug, aug_ratio, npower=1.0):
    if aug == "none":
        return data
    if aug == "wdropN":
        return _ensure_num_nodes(weighted_drop_nodes(data, aug_ratio, npower))
    if aug in _AUGMENTATIONS:
        return _ensure_num_nodes(_AUGMENTATIONS[aug](data, aug_ratio))
    if aug in _RANDOM_AUGMENTATIONS:
        choices = _RANDOM_AUGMENTATIONS[aug]
        fn = choices[np.random.randint(len(choices))]
        if fn is weighted_drop_nodes:
            return _ensure_num_nodes(weighted_drop_nodes(data, aug_ratio, npower))
        return _ensure_num_nodes(fn(data, aug_ratio))
    raise ValueError(f"unknown augmentation: {aug}")


def augment_batch(batch, aug, aug_ratio, npower=1.0, device=None):
    dev = _default_aug_device(device)
    if aug == "none":
        return batch.to(dev)
    from torch_geometric.data import Batch

    # deepcopy after .to(dev): apply_augmentation mutates in place; without a copy,
    # aug1/aug2 views or the DataLoader batch could share corrupted storage.
    # PyG Data.clone() only shallow-copies tensor fields; deepcopy remains safer here.
    aug_list = [
        apply_augmentation(copy.deepcopy(data.to(dev)), aug, aug_ratio, npower)
        for data in batch.to_data_list()
    ]
    return Batch.from_data_list(aug_list)


class TUDatasetExt(TUDataset):
    """TU Dortmund graph kernel benchmarks with custom caching and augmentation."""

    url = "https://www.chrsmrrs.com/graphkerneldatasets"

    def __init__(
        self,
        root,
        name,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        use_node_attr=False,
        processed_filename="data.pt",
        aug="none",
        aug_ratio=None,
        npower=1.0,
    ):
        self.processed_filename = processed_filename
        self.aug = aug
        self.aug_ratio = aug_ratio
        self.npower = npower
        self._base_data_list: Optional[list[Optional[BaseData]]] = None
        super().__init__(
            root, name, transform, pre_transform, pre_filter,
            use_node_attr=use_node_attr,
        )
        # TUDataset.process() calls get() before pre_transform, which can cache
        # raw graphs without graph_id/x. Drop that stale cache after init.
        self._base_data_list = None

    def process(self) -> None:
        super().process()
        self._base_data_list = None

    @property
    def processed_file_names(self):
        return self.processed_filename

    def _get_base_data(self, idx) -> BaseData:
        if self._base_data_list is None:
            self._base_data_list = cast(list[Optional[BaseData]], [None] * self.len())
        base_data_list = self._base_data_list
        if base_data_list[idx] is not None:
            return cast(BaseData, base_data_list[idx])

        data = cast(BaseData, separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        ))
        base_data_list[idx] = data
        return data

    def _apply_augmentation(self, data):
        return apply_augmentation(data, self.aug, self.aug_ratio, self.npower)

    def get(self, idx) -> BaseData:
        if self.len() == 1:
            if self._data is None:
                raise RuntimeError("dataset storage is not initialized")
            data = cast(BaseData, copy.copy(self._data))
            return self._apply_augmentation(data) if self.aug != "none" else data

        base = self._get_base_data(idx)
        if self.aug == "none":
            return copy.copy(base)
        data = copy.deepcopy(base)
        return self._apply_augmentation(data)
