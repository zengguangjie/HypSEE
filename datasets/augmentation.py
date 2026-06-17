import torch
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
import numpy as np

@functional_transform('augmentation_transform')
class AugmentationTransform(BaseTransform):
    def __init__(self, aug, aug_ratio):
        self.aug = aug
        self.aug_ratio = aug_ratio

    def __call__(self, data: Data) -> Data:
        if self.aug == 'dropN':
            data = drop_nodes(data, self.aug_ratio)
        elif self.aug == 'wdropN':
            data = weighted_drop_nodes(data, self.aug_ratio, self.npower)
        elif self.aug == 'permE':
            data = permute_edges(data, self.aug_ratio)
        elif self.aug == 'subgraph':
            data = subgraph(data, self.aug_ratio)
        elif self.aug == 'maskN':
            data = mask_nodes(data, self.aug_ratio)
        elif self.aug == 'none':
            data = data
        elif self.aug == 'random4':
            ri = np.random.randint(4)
            if ri == 0:
                data = drop_nodes(data, self.aug_ratio)
            elif ri == 1:
                data = subgraph(data, self.aug_ratio)
            elif ri == 2:
                data = permute_edges(data, self.aug_ratio)
            elif ri == 3:
                data = mask_nodes(data, self.aug_ratio)
            else:
                print('sample augmentation error')
                assert False

        elif self.aug == 'random3':
            ri = np.random.randint(3)
            if ri == 0:
                data = drop_nodes(data, self.aug_ratio)
            elif ri == 1:
                data = subgraph(data, self.aug_ratio)
            elif ri == 2:
                data = permute_edges(data, self.aug_ratio)
            else:
                print('sample augmentation error')
                assert False


        elif self.aug == 'random2':
            ri = np.random.randint(2)
            if ri == 0:
                data = drop_nodes(data, self.aug_ratio)
            elif ri == 1:
                data = subgraph(data, self.aug_ratio)
            else:
                print('sample augmentation error')
                assert False

        return data

def drop_nodes(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]: n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    edge_index = adj.nonzero().t()

    try:
        data.edge_index = edge_index
        data.x = data.x[idx_nondrop]
    except:
        data = data
    return data

def weighted_drop_nodes(data, aug_ratio, npower):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    adj = np.zeros((node_num, node_num))
    adj[data.edge_index[0], data.edge_index[1]] = 1
    deg = adj.sum(axis=1)
    deg[deg == 0] = 0.1
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
    idx_dict = {idx_nondrop[n]: n for n in list(range(idx_nondrop.shape[0]))}

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

def permute_edges(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    idx_add = np.random.choice(node_num, (2, permute_num))

    # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]
    # edge_index = [edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)] + idx_add

    edge_index = np.concatenate(
        (edge_index[:, np.random.choice(edge_num, (edge_num - permute_num), replace=False)], idx_add), axis=1)
    data.edge_index = torch.tensor(edge_index)

    return data

def subgraph(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0] == idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0] == idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    data.x = data.x[idx_nondrop]
    idx_dict = {idx_nondrop[n]: n for n in list(range(len(idx_nondrop)))}

    edge_index = data.edge_index.numpy()
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[list(range(node_num)), list(range(node_num))] = 1
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    edge_index = adj.nonzero().t()

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    data.edge_index = edge_index

    return data

def mask_nodes(data, aug_ratio):

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    token = data.x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(token, dtype=torch.float32)

    return data