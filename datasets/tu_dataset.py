from torch_geometric.datasets import TUDataset
import torch
from itertools import repeat, product
import numpy as np
import os.path as osp
# from torch_cluster import random_walk
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
# from dataset.data_utils import hypergraph_construction
import torch_geometric.transforms as T
from datasets.feature_expansion import FeatureExpander
import re
import copy
from torch_geometric.data.separate import separate
import torch_geometric.utils as tg_utils

# def get_dataset(name, root=None, dense=True, max_nodes=150):
#     if root is None or root == '':
#         root = 'data'
#     if dense:
#         transform = T.ToDense(num_nodes=None)
#         # pre_filter=lambda data: data.num_nodes <= max_nodes
#     else:
#         transform = None
#         # path = osp.join('data', )
#     # else:path = osp.join(root, name)
#     dataset = TUDatasetExt(root, name, pre_transform=None, use_node_attr=True, transform=transform, pre_filter=None)
#     return dataset

def get_dataset(name, root=None, feat_str="deg"):
    degree = feat_str.find("deg") >= 0
    onehot_maxdeg = re.findall("odeg(\d+)", feat_str)
    onehot_maxdeg = int(onehot_maxdeg[0]) if onehot_maxdeg else None
    k = re.findall("an{0,1}k(\d+)", feat_str)
    # print(k)
    k = int(k[0]) if k else 0
    centrality = feat_str.find("cent") >= 0

    pre_transform = FeatureExpander(
        degree=degree, onehot_maxdeg=onehot_maxdeg, AK=k,
        centrality=centrality,
        # remove_edges=remove_edges,
        # edge_noises_add=edge_noises_add, edge_noises_delete=edge_noises_delete,
        # group_degree=groupd
        ).transform
    dataset = TUDatasetExt(root, name, pre_transform=pre_transform, use_node_attr=True, transform=None,
                        pre_filter=None, processed_filename="data_%s.pt" % feat_str)
    return dataset


def get_dataset_addgraph(name, root=None):
    add_graph_transform = AddGraphIdTransform()
    dataset = TUDataset(root, name, pre_transform=add_graph_transform, use_node_attr=True, transform=None, pre_filter=None)
    return dataset

def get_dataset_dense(name, root=None, max_nodes=150):
    dataset = TUDataset(root, name, pre_transform=None, use_node_attr=True, transform=T.ToDense(num_nodes=None), pre_filter=lambda data: data.num_nodes <= max_nodes)
    return dataset


class AddGraphIdTransform:
    def __init__(self):
        self.graph_id = 0
        # self.num_graphs = num_graphs

    def __call__(self, data):
        data.graph_id = self.graph_id
        self.graph_id += 1
        return data


class TUDatasetExt(TUDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <http://graphkernels.cs.tu-dortmund.de>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name <http://graphkernels.cs.tu-dortmund.de>`_ of
            the dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node features (if present).
            (default: :obj:`False`)
    """

    # url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/' \
    #       'graphkerneldatasets'
    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    # cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
    #                'graph_datasets/master/datasets')

    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 use_node_attr=False,
                 processed_filename='data.pt',
                 aug="none", aug_ratio=None):
        self.processed_filename = processed_filename
        self.aug = aug
        self.aug_ratio = aug_ratio
        super(TUDatasetExt, self).__init__(root, name, transform, pre_transform,
                                           pre_filter, use_node_attr)


    @property
    def processed_file_names(self):
        return self.processed_filename

    def get(self, idx):

        if self.len() == 1:
            return copy.copy(self._data)

        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        data = separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )
        # print(self.data)
        # if data.graph_id is not None:
        #     print(data)

        # self._data_list[idx] = copy.copy(data)

        if self.aug == 'dropN':
            # data = drop_nodes(data, self.aug_ratio)
            data = node_drop(data, self.aug_ratio)
        # elif self.aug == 'wdropN':
        #     data = weighted_drop_nodes(data, self.aug_ratio, self.npower)
        elif self.aug == 'permE':
            data = edge_pert(data, self.aug_ratio)
        elif self.aug == 'subgraph':
            data = subgraph(data, self.aug_ratio)
        elif self.aug == 'maskN':
            data = attr_mask(data, self.aug_ratio)
        elif self.aug == 'none':
            data = data
        elif self.aug == 'random4':
            ri = np.random.randint(4)
            if ri == 0:
                # data = drop_nodes(data, self.aug_ratio)
                data = node_drop(data, self.aug_ratio)
            elif ri == 1:
                data = subgraph(data, self.aug_ratio)
            elif ri == 2:
                data = edge_pert(data, self.aug_ratio)
            elif ri == 3:
                data = attr_mask(data, self.aug_ratio)
            else:
                print('sample augmentation error')
                assert False

        elif self.aug == 'random3':
            ri = np.random.randint(3)
            if ri == 0:
                # data = drop_nodes(data, self.aug_ratio)
                data = node_drop(data, self.aug_ratio)
            elif ri == 1:
                data = subgraph(data, self.aug_ratio)
            elif ri == 2:
                # data = permute_edges(data, self.aug_ratio)
                data = edge_pert(data, self.aug_ratio)
            else:
                print('sample augmentation error')
                assert False


        elif self.aug == 'random2':
            ri = np.random.randint(2)
            if ri == 0:
                # data = drop_nodes(data, self.aug_ratio)
                data = node_drop(data, self.aug_ratio)
            elif ri == 1:
                data = subgraph(data, self.aug_ratio)
            else:
                print('sample augmentation error')
                assert False


        else:
            print('augmentation error')
            assert False

        # print(data)
        # print(self.aug)
        # assert False

        self._data_list[idx] = copy.copy(data)
        # print(data, type(data))

        return data

# def drop_nodes(data, aug_ratio):
#
#     node_num, _ = data.x.size()
#     _, edge_num = data.edge_index.size()
#     drop_num = int(node_num  * aug_ratio)
#
#     idx_perm = np.random.permutation(node_num)
#
#     idx_drop = idx_perm[:drop_num]
#     idx_nondrop = idx_perm[drop_num:]
#     idx_nondrop.sort()
#     idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}
#
#     edge_index = data.edge_index.numpy()
#     adj = torch.zeros((node_num, node_num))
#     adj[edge_index[0], edge_index[1]] = 1
#     adj = adj[idx_nondrop, :][:, idx_nondrop]
#     edge_index = adj.nonzero().t()
#
#     try:
#         data.edge_index = edge_index
#         data.x = data.x[idx_nondrop]
#     except:
#         data = data
#     return data

def node_drop(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num  * aug_ratio)

    idx_perm = np.random.permutation(node_num)
    idx_nondrop = idx_perm[drop_num:].tolist()
    idx_nondrop.sort()

    edge_index, _ = tg_utils.subgraph(idx_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num)

    data.x = data.x[idx_nondrop]
    data.edge_index = edge_index
    # data.__num_nodes__, _ = data.x.shape
    data.num_nodes, _ = data.x.shape
    # print(data.__num_nodes__)
    # print(data.num_nodes)
    return data


def weighted_drop_nodes(data, aug_ratio, npower):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num  * aug_ratio)

    adj = np.zeros((node_num, node_num))
    adj[data.edge_index[0], data.edge_index[1]] = 1
    deg = adj.sum(axis=1)
    deg[deg==0] = 0.1
    # print(deg)
    # deg = deg ** (-1)
    deg = deg ** (npower)
    # print(deg)
    # print(deg / deg.sum())
    # assert False

    idx_drop = np.random.choice(node_num, drop_num, replace=False, p=deg / deg.sum())

    # idx_perm = np.random.permutation(node_num)
    # idx_drop = idx_perm[:drop_num]
    # idx_nondrop = idx_perm[drop_num:]

    idx_nondrop = np.array([n for n in range(node_num) if not n in idx_drop])

    # idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()
    ###
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    edge_index = adj.nonzero().t()

    ###
    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    try:
        data.edge_index = edge_index
        data.x = data.x[idx_nondrop]
    except:
        data = data
    return data


# def permute_edges(data, aug_ratio):
#
#     node_num, _ = data.x.size()
#     _, edge_num = data.edge_index.size()
#     permute_num = int(edge_num * aug_ratio)
#
#     edge_index = data.edge_index.numpy()
#
#     idx_add = np.random.choice(node_num, (2, permute_num))
#
#     # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]
#     # edge_index = [edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)] + idx_add
#
#     edge_index = np.concatenate((edge_index[:, np.random.choice(edge_num, (edge_num - permute_num), replace=False)], idx_add), axis=1)
#     data.edge_index = torch.tensor(edge_index)
#
#     return data

def edge_pert(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    pert_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index[:, np.random.choice(edge_num, (edge_num - pert_num), replace=False)]

    idx_add = np.random.choice(node_num, (2, pert_num))
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_add[0], idx_add[1]] = 1
    adj[np.arange(node_num), np.arange(node_num)] = 0
    edge_index = adj.nonzero(as_tuple=False).t()

    data.edge_index = edge_index
    data.num_nodes, _ = data.x.shape
    return data

# def subgraph(data, aug_ratio):
#
#     node_num, _ = data.x.size()
#     _, edge_num = data.edge_index.size()
#     sub_num = int(node_num * aug_ratio)
#
#     edge_index = data.edge_index.numpy()
#
#     idx_sub = [np.random.randint(node_num, size=1)[0]]
#     idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])
#
#     count = 0
#     while len(idx_sub) <= sub_num:
#         count = count + 1
#         if count > node_num:
#             break
#         if len(idx_neigh) == 0:
#             break
#         sample_node = np.random.choice(list(idx_neigh))
#         if sample_node in idx_sub:
#             continue
#         idx_sub.append(sample_node)
#         idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))
#
#     idx_drop = [n for n in range(node_num) if not n in idx_sub]
#     idx_nondrop = idx_sub
#     data.x = data.x[idx_nondrop]
#     idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}
#
#     edge_index = data.edge_index.numpy()
#     adj = torch.zeros((node_num, node_num))
#     adj[edge_index[0], edge_index[1]] = 1
#     adj[list(range(node_num)), list(range(node_num))] = 1
#     adj = adj[idx_nondrop, :][:, idx_nondrop]
#     edge_index = adj.nonzero().t()
#
#     # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
#     data.edge_index = edge_index
#
#     return data

def subgraph(data, aug_ratio):
    G = tg_utils.to_networkx(data)

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * (1-aug_ratio))

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in G.neighbors(idx_sub[-1])])

    while len(idx_sub) <= sub_num:
        if len(idx_neigh) == 0:
            idx_unsub = list(set([n for n in range(node_num)]).difference(set(idx_sub)))
            # if len(idx_unsub) == 0:
            #     idx_neigh = set()
            # else:
            # print(len(idx_unsub), node_num, len(idx_sub), sub_num, aug_ratio)
            idx_neigh = set([np.random.choice(idx_unsub)])

        sample_node = np.random.choice(list(idx_neigh))

        idx_sub.append(sample_node)
        idx_neigh = idx_neigh.union(set([n for n in G.neighbors(idx_sub[-1])])).difference(set(idx_sub))

    idx_nondrop = idx_sub
    idx_nondrop.sort()

    edge_index, _ = tg_utils.subgraph(idx_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num)

    data.x = data.x[idx_nondrop]
    data.edge_index = edge_index
    # data.__num_nodes__, _ = data.x.shape
    data.num_nodes, _ = data.x.shape
    # print(data.__num_nodes__)
    # print(data.num_nodes)
    return data


# def mask_nodes(data, aug_ratio):
#
#     node_num, feat_dim = data.x.size()
#     mask_num = int(node_num * aug_ratio)
#
#     token = data.x.mean(dim=0)
#     idx_mask = np.random.choice(node_num, mask_num, replace=False)
#     data.x[idx_mask] = torch.tensor(token, dtype=torch.float32)
#
#     return data

def attr_mask(data, aug_ratio):
    node_num, _ = data.x.size()
    mask_num = int(node_num * aug_ratio)
    _x = data.x.clone()

    token = data.x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)

    _x[idx_mask] = token
    data.x = _x
    data.num_nodes, _ = data.x.shape
    return data