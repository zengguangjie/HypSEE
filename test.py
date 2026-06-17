

from torch_geometric.datasets import TUDataset
import torch
from itertools import repeat, product
import numpy as np
import os.path as osp
from torch_cluster import random_walk
from torch_geometric.utils import dense_to_sparse, is_undirected
from torch_geometric.data import Data
from datasets.data_utils import hypergraph_construction
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, DenseDataLoader
import torch
import argparse
from datasets.tu_dataset import get_dataset, get_dataset_dense
import random
import os
from torch_geometric.loader import DataLoader, DenseDataLoader
from datasets.loaders import IterLoader
# from models.model import HypSEE, HypSEEDense
from datasets.data_utils import hypergraph_construction, hypergraph_construction_batch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected



if __name__=='__main__':
    dims = [8, 16, 32, 64, 128, 256]
    print(dims[1:])
    # def fix_seed(self, seed=0):
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.backends.cudnn.benchmark = False
    #     torch.cuda.manual_seed(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.cuda.manual_seed_all(seed)
    #     os.environ['PYTHONHASHSEED'] = str(seed)
    # fix_seed(0)
    # print(np.random.randint(100))
    # print(np.random.randint(100))
    # data_name = "MSRC_21"
    # data_root = "/mnt/data/zengguangjie/HypSEE/data/MSRC_21"
    # dataset = get_dataset(name=data_name, root=data_root)
    # length_list = 4
    # print(type(length_list) == int)
    # print([length_list])
    # H = torch.tensor([[0,1,2],
    #                  [3,4,5]])
    # HT = torch.tensor([[0,3],
    #                    [1,4],
    #                    [2,5]])
    # print(torch.matmul(H,HT))
    # print(torch.einsum('ij,ji->i', H, HT))
    # dataset = get_dataset(name='IMDB-BINARY', root='data/IMDB-BINARY')
    # val_loader_S = IterLoader(DataLoader(dataset[0.7:0.8], batch_size=2, shuffle=False))
    # val_loader_S.new_epoch()
    # batch = val_loader_S.next()
    # print(batch)
    # H_batch = hypergraph_construction_batch(batch, 3)
    # print(H_batch)
    # import torch
    # from torch_cluster import random_walk
    #
    # row = torch.tensor([0, 1, 1, 1, 2, 2, 3, 3, 4, 4])
    # col = torch.tensor([1, 0, 2, 3, 1, 4, 1, 4, 2, 3])
    # start = torch.tensor([0, 1, 2, 3, 4, 5])
    #
    # walk = random_walk(row, col, start, walk_length=3)
    # print(walk)
    # # walk = walk.view(2,3,4)
    # # print(walk)
    # batch_size = 2
    # num_nodes = 6
    # num_edges = 6
    # length_list = [2,3]
    #
    # mask = torch.ones(walk.size(), dtype=torch.bool, device=walk.device)
    # mask_indices = torch.randperm(num_edges)
    # length_list = sorted(length_list)
    # for index, length in enumerate(length_list[:-1]):
    #     segment_len = num_edges / len(length_list)
    #     cur_indices = mask_indices[int(segment_len * index): int(segment_len * (index + 1))]
    #     mask[cur_indices, length+1:] = False
    # print(mask)
    # walk = walk * mask
    # print(walk)

    # walk = walk.view(batch_size, num_edges, -1)
    # print(walk)
    # ptr = torch.from_numpy(np.array([0,3,6]))
    # size = (2,6,3)
    # # walk.view(size)
    # H_batch = torch.zeros(size)
    #
    # batch_indices = torch.arange(batch_size).view(-1,1,1).expand_as(walk).flatten()
    # print(batch_indices)
    # print(ptr)
    # print(ptr[batch_indices])
    # walk = walk - ptr[batch_indices].view(walk.size())
    # print(walk)
    # # batch_indices = batch_indices.flatten()
    # hyperedge_indices = torch.arange(num_edges).view(1,-1,1).expand_as(walk).flatten()
    # node_indices = walk.flatten()
    # H_batch[batch_indices, node_indices, hyperedge_indices] = 1
    # print(H_batch)


    # sim_mtx = [[1,2,3],
    #            [4,5,6],
    #            [7,8,9]]
    # sim_mtx = torch.tensor(sim_mtx)
    # print(sim_mtx[range(3),range(3)])
    # import torch
    #
    # # Example tensors
    # H = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32)  # Shape (2, 2, 2)
    # W = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)  # Shape (2, 2)
    #
    # # Step 1: Compute HW
    # HW = torch.einsum('bij,bj->bij', H, W)
    #
    # # Step 2: Compute D
    # D = torch.einsum('bij,bjk->bik', HW, H.transpose(-1, -2))
    #
    # # Step 3: Extract diagonal
    # D_diag = torch.einsum('bii->bi', D)
    #
    # # Print results
    # print("H:\n", H)
    # print("W:\n", W)
    # print("HW:\n", HW)
    # print("D:\n", D)
    # print("Diagonal of D:\n", D_diag)

    # dataset = get_dataset(name="IMDB-BINARY", root="data/IMDB-BINARY", feat_str="deg+odeg10")
    #
    # dataset.aug = 'none'
    # dataset_S = dataset.shuffle()
    # dataset_S.aug, dataset_S.aug_ratio = 'maskN', 0.8
    # labeled_loader_S = IterLoader(
    #     DataLoader(dataset_S[:0.1], batch_size=int(np.ceil(80 / 5)), shuffle=False))
    # labeled_loader_S.new_epoch()
    # labeled_batch_S = labeled_loader_S.next().to("cuda")
    # print(labeled_batch_S)
    # print(labeled_batch_S.graph_id)


    # list1 = np.array([1,2,3,4])
    # list2 = np.array([5,6,7,8])
    # print(list1+list2)
    # m = torch.tensor([0, 1, 1, 0, 1, 0], dtype=torch.bool)
    # B = torch.tensor([[1, 2, 3, 4, 5, 6],
    #                   [7, 8, 9, 10, 11, 12]])
    #
    # # Select elements along dimension 1 according to the mask
    # B_selected = B[:, m]
    #
    # print("Selected elements from B:")
    # print(B_selected)
    # a = (0,1)
    # b = (1,0)
    # print(max(a, b))

    # data_name = "PROTEINS"
    # data_root = "/data/zengguangjie/HypSEE/data/PROTEINS"
    # dataset = get_dataset(name=data_name, root=data_root)
    # print(is_undirected(dataset[0].edge_index))
    # edge_index = to_undirected(edge_index=dataset[0].edge_index)
    # print(edge_index)

    # for data in dataset:
    #     print(data.graph_id, data.num_nodes)



    # # Define the shape of the sparse matrices
    # shape_A = (4, 3)  # 4 rows, 3 columns
    # shape_B = (3, 5)  # 3 rows, 5 columns
    #
    # # Define the indices and values of A and B (non-zero elements)
    # # A is a sparse matrix of size (4, 3)
    # indices_A = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]])  # row, col indices
    # values_A = torch.tensor([1.0, 2.0, 3.0, 4.0])
    #
    # # B is a sparse matrix of size (3, 5)
    # indices_B = torch.tensor([[0, 1, 2], [0, 1, 2]])  # row, col indices
    # values_B = torch.tensor([5.0, 6.0, 7.0])
    #
    # # Create sparse COO tensors
    # A = torch.sparse_coo_tensor(indices_A, values_A, shape_A)
    # B = torch.sparse_coo_tensor(indices_B, values_B, shape_B)
    #
    # C = torch.ones(shape_B)
    #
    # # print(A.matmul(B))
    # # print(A.t())
    # print(A.matmul(C))

    # max_nodes = 150
    #
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
    #                 'PROTEINS_dense')
    # dataset = TUDataset(
    #     path,
    #     name='PROTEINS',
    #     transform=T.ToDense(max_nodes),
    #     pre_filter=lambda data: data.num_nodes <= max_nodes,
    # )
    # dataset = dataset.shuffle()
    # print(len(dataset))
    # exit(0)
    # n = (len(dataset) + 9) // 10
    # test_dataset = dataset[:n]
    # val_dataset = dataset[n:2 * n]
    # train_dataset = dataset[2 * n:]
    # test_loader = DenseDataLoader(test_dataset, batch_size=20)
    # val_loader = DenseDataLoader(val_dataset, batch_size=20)
    # train_loader = DenseDataLoader(train_dataset, batch_size=20)
    # # labeled_loader = DenseDataLoader(dataset, batch_size=20)
    # # print(labeled_loader)
    # # labeled_loader = iter(labeled_loader)
    # # data_batch = next(labeled_loader)
    # # data_batch = labeled_loader.next()
    # # next(iter(labeled_loader))
    # for batch in train_loader:
    #     print(batch)