import torch
from torch_cluster import random_walk
from torch_geometric.utils import add_remaining_self_loops, dense_to_sparse, remove_self_loops
from torch_geometric.data import Data
from torch_sparse import spspmm, coalesce
# from torch_geometric.utils import unbatch_edge_index
from datasets.tu_dataset import get_dataset
from torch_geometric.io import fs
import os
from torch_geometric.utils import k_hop_subgraph
import numpy as np
from sklearn.model_selection import StratifiedKFold


class TwoHopNeighbor(object):
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes

        value = edge_index.new_ones((edge_index.size(1), ), dtype=torch.float)

        index, value = spspmm(edge_index, value, edge_index, value, N, N, N, True)
        value.fill_(0)
        index, value = remove_self_loops(index, value)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, N, N)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)




def precompute_hypergraphs(data_name, data_root, mode, hyperedge_length, num_edges=32, seed=0):
    rng = torch.Generator()
    rng.manual_seed(seed)
    if mode == "RW":
        torch.manual_seed(seed)
        dataset = get_dataset(name=data_name, root=data_root)
        hypergraph_dict = {}
        store_path = os.path.join(data_root, data_name, mode, "len_{}.pt".format(int(hyperedge_length)))
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        for data in dataset:
            graph_id = data.graph_id.item()
            assert data.num_nodes is not None
            num_nodes = data.num_nodes
            sampled_nodes = torch.randint(0, num_nodes, (num_edges,), generator=rng)
            row, col = data.edge_index
            walk = random_walk(row, col, start=sampled_nodes, walk_length=hyperedge_length)
            if isinstance(walk, tuple):
                walk = walk[0]
            walk_vertex_indices = walk.flatten()
            walk_edge_indices = torch.arange(num_edges).unsqueeze(-1).repeat(1, hyperedge_length+1).flatten()
            walk_indices = torch.stack([walk_vertex_indices, walk_edge_indices], dim=0)
            # print(walk_indices)
            walk_values = torch.ones_like(walk_indices[0,:].squeeze(), dtype=torch.float)
            hypergraph = torch.sparse_coo_tensor(indices=walk_indices, values=walk_values, size=(num_nodes, num_edges))
            hypergraph = hypergraph.coalesce()
            # print(hypergraph.indices())
            walk_values = torch.ones_like(hypergraph.indices()[0,:].squeeze(), dtype=torch.float)
            hypergraph = torch.sparse_coo_tensor(indices=hypergraph.indices(), values=walk_values, size=(num_nodes, num_edges), is_coalesced=True)
            hypergraph_dict[graph_id] = hypergraph
        fs.torch_save(hypergraph_dict, store_path)
    elif mode == "HOP":
        dataset = get_dataset(name=data_name, root=data_root)
        hypergraph_dict = {}
        # hyperedge_length here is the number of neighbor hops
        store_path = os.path.join(data_root, data_name, mode, "len_{}.pt".format(int(hyperedge_length)))
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        for data in dataset:
            graph_id = data.graph_id.item()
            num_nodes = data.num_nodes
            assert num_nodes is not None
            indices_vertex = []
            indices_edge = []
            values = []
            for i in range(num_nodes):
                subset, _, _, _, = k_hop_subgraph(node_idx=i, num_hops=hyperedge_length, edge_index=data.edge_index, num_nodes=num_nodes)
                indices_vertex.append(subset)
                indices_edge.append(torch.ones_like(subset)*i)
                values.append(torch.ones_like(subset))
            indices_vertex = torch.cat(indices_vertex, dim=-1)
            indices_edge = torch.cat(indices_edge, dim=-1)
            indices = torch.stack([indices_vertex, indices_edge], dim=0)
            values = torch.cat(values, dim=-1)
            hypergraph = torch.sparse_coo_tensor(indices=indices, values=values, size=[num_nodes, num_nodes])
            hypergraph = hypergraph.coalesce()
            values = torch.ones_like(hypergraph.indices()[0,:].squeeze(), dtype=torch.float)
            hypergraph = torch.sparse_coo_tensor(indices=hypergraph.indices(), values=values, size=(num_nodes, num_nodes), is_coalesced=True)
            hypergraph_dict[graph_id] = hypergraph
        fs.torch_save(hypergraph_dict, store_path)
    # elif mode == "modularity":
    else:
        raise NotImplementedError





def load_hypergraphs(data_name, data_root, mode, hyperedge_length_list, num_edges, num_graphs, seed=0):
    rng = torch.Generator()
    rng.manual_seed(seed)
    indices_dict = {}
    values_dict = {}
    size_dict = {}
    for i in range(num_graphs):
        indices_dict[i] = []
        values_dict[i] = []
        size_dict[i] = (0,0)
    num_edges_list = []
    for i in range(len(hyperedge_length_list)-1):
        num_edges_list.append(int(num_edges / len(hyperedge_length_list)))
    num_edges_list.append(num_edges - np.sum(num_edges_list))
    for i, hyperedge_length in enumerate(hyperedge_length_list):
        num_edges_i = num_edges_list[i]
        store_path = os.path.join(data_root, data_name, mode, "len_{}.pt".format(int(hyperedge_length)))
        hypergraph_dict = fs.torch_load(store_path)
        for graph_id in hypergraph_dict.keys():
            indices = hypergraph_dict[graph_id].indices()
            values = hypergraph_dict[graph_id].values()
            size = hypergraph_dict[graph_id].size()
            # print(indices)
            # print(values)
            # print(size)
            # print(num_edges_i)
            # assert num_edges_i <= size[1]
            hyperedge_ids = torch.randint(0, size[1], (num_edges_i,), generator=rng)
            mask = torch.isin(indices[1,:], hyperedge_ids)
            selected_indices = indices[:,mask]
            unique_values, new_values = torch.unique(selected_indices[1,:], return_inverse=True)
            selected_indices[1,:] = new_values + size_dict[graph_id][1]
            indices_dict[graph_id].append(selected_indices)
            values_dict[graph_id].append(values[mask])
            size_dict[graph_id] = (size[0], size_dict[graph_id][1]+num_edges_i)
            ##
            # indices = hypergraph_dict[graph_id].indices()
            # indices[1,:] += size_dict[graph_id][1]
            # indices_dict[graph_id].append(indices)
            # # indices_dict[graph_id].append(hypergraph_dict[graph_id].indices())
            # values_dict[graph_id].append(hypergraph_dict[graph_id].values())
            # # size_i = hypergraph_dict[graph_id].size()
            # # size_dict[graph_id] = (size_i[0], size_dict[graph_id][1] + si)
            # size_dict[graph_id][0] = hypergraph_dict[graph_id].size()[0]
            # # size_dict[graph_id][1] += hypergraph_dict[graph_id].size()[1]
    result_hg_dict = {}
    for graph_id in indices_dict.keys():
        indices = torch.cat(indices_dict[graph_id], dim=-1)
        values = torch.cat(values_dict[graph_id], dim=-1)
        size = size_dict[graph_id]
        hypergraph = torch.sparse_coo_tensor(indices=indices, values=values, size=size, is_coalesced=True)
        result_hg_dict[graph_id] = hypergraph
    return result_hg_dict


def hypergraph_to_dense_batch(hypergraph_dict, graph_id_list, num_edges):
    H_batch = []
    max_num_nodes = 0
    for graph_id in graph_id_list:
        size_i = hypergraph_dict[graph_id].size()
        max_num_nodes = max(max_num_nodes, size_i[0])
    for graph_id in graph_id_list:
        hypergraph_i = hypergraph_dict[graph_id]
        hypergraph_i = torch.sparse_coo_tensor(indices=hypergraph_i.indices(), values=hypergraph_i.values(), size=(max_num_nodes, num_edges))
        H_i = hypergraph_i.to_dense()
        H_batch.append(H_i)
    H_batch = torch.stack(H_batch, dim=0)
    return H_batch


def _allocate_counts(n, ratios):
    """Largest-remainder allocation of n items across ratio buckets."""
    ratios = np.asarray(ratios, dtype=float)
    ratios = ratios / ratios.sum()
    raw = n * ratios
    counts = np.floor(raw).astype(int)
    remainder = int(n - counts.sum())
    if remainder > 0:
        fractional = raw - counts
        for i in np.argsort(-fractional)[:remainder]:
            counts[i] += 1
    return counts.tolist()


def stratified_semi_supervised_split(
    dataset,
    seed=0,
    labeled_ratio=0.1,
    gap_ratio=0.1,
    unlabeled_ratio=0.5,
    val_ratio=0.1,
    test_ratio=0.2,
):
    """Stratified split matching Exp.py slice protocol (including the 10% gap)."""
    ratios = [labeled_ratio, gap_ratio, unlabeled_ratio, val_ratio, test_ratio]
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {sum(ratios)}")

    y = dataset.y
    if y is None:
        raise RuntimeError("dataset labels are required for stratified split")
    y_np = y.detach().cpu().numpy()
    rng = np.random.RandomState(seed)

    bucket_names = ('labeled', 'gap', 'unlabeled', 'val', 'test')
    buckets = {name: [] for name in bucket_names}

    for cls in np.unique(y_np):
        cls_idx = np.where(y_np == cls)[0]
        rng.shuffle(cls_idx)
        counts = _allocate_counts(len(cls_idx), ratios)
        start = 0
        for name, count in zip(bucket_names, counts):
            buckets[name].extend(cls_idx[start:start + count].tolist())
            start += count

    for name in bucket_names:
        idx = np.asarray(buckets[name], dtype=np.int64)
        rng.shuffle(idx)
        buckets[name] = idx

    return buckets


def k_fold(dataset, folds, seed):
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    labeled_train_indices, unlabeled_train_indices, test_indices, val_indices = [], [], [], []
    split_indices = []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.y):
        # print(idx)
        split_indices.append(torch.from_numpy(idx))
    # print(len(test_indices))
    for i in range(folds):
        val_indices.append(split_indices[i-2])
        test_indices.append(torch.cat([split_indices[i-1], split_indices[i]], dim=-1))
    skf_semi = StratifiedKFold(folds-3, shuffle=True, random_state=seed)
    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        idx_train = train_mask.nonzero(as_tuple=False).view(-1)

        labeled_train_indices_i = []
        for _, idx in skf_semi.split(torch.zeros(idx_train.size()[0]), dataset[idx_train].y):
            idx_train_j = idx_train[idx]
            labeled_train_indices_i.append(idx_train_j)
            if len(labeled_train_indices_i) >= 2:
                break
        # assert labeled_train_indices_i == 2
        labeled_train_indices.append(labeled_train_indices_i[0])
        labeled_train_indices_i = torch.cat(labeled_train_indices_i, dim=-1)
        train_mask[labeled_train_indices_i.long()] = 0
        idx_train_unlabeled = train_mask.nonzero(as_tuple=False).view(-1)
        unlabeled_train_indices.append(idx_train_unlabeled)
    return labeled_train_indices, unlabeled_train_indices, val_indices, test_indices





if __name__=='__main__':
    data_name = "REDDIT-MULTI-5K"
    data_root = "/mnt/data/zengguangjie/HypSEE/data/REDDIT-MULTI-5K"
    mode = "RW"
    # hyperedge_length = 2
    for hyperedge_length in range(2, 11):
        print(hyperedge_length)
        precompute_hypergraphs(data_name, data_root, mode, hyperedge_length, num_edges=64, seed=0)

    store_path = os.path.join(data_root, data_name, mode, "len_{}.pt".format(int(hyperedge_length)))
    hypergraph_dict = fs.torch_load(store_path)
    print(hypergraph_dict)


