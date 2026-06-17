
import sys
sys.path.append('.')

# import optuna
import torch
import argparse

# from ExpAug import ExpAug
# from datasets.tu_dataset import get_dataset, get_dataset_dense
# import random
import os
import numpy as np
# from torch_geometric.loader import DataLoader, DenseDataLoader
# from datasets.loaders import IterLoader
# from models.model import HypSEE2
# from datasets.data_utils import hypergraph_construction, hypergraph_construction_batch
# from datasets.data_utils import load_hypergraphs, hypergraph_to_dense_batch
# import torch.nn.functional as F
# from copy import deepcopy
import json
from Exp import Exp

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='IMDB-BINARY')
parser.add_argument('--data_root', type=str, default='data/')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=80)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
# parser.add_argument('--lr_decay_step_size', type=int, default=1)
parser.add_argument('--dim_embedding', type=int, default=32)
parser.add_argument('--dim_embedding_gnn', type=int, default=32)
parser.add_argument('--num_edges1', type=int, default=32)
parser.add_argument('--num_edges2', type=int, default=32)
parser.add_argument('--num_layers_gnn', type=int, default=2, help='number of layers in the first encoder embedding.')
parser.add_argument('--num_anchors', type=int, default=64, help='number of anchor graphs to be used')
parser.add_argument('--beta', type=float, default=1, help='beta balances two loss values.')
parser.add_argument('--weight_hse', type=float, default=0.001, help="weight of hierarchical se loss")
parser.add_argument('--warm_epochs', type=int, default=2)
# parser.add_argument('--num_edges1', type=int, default=32, help='number of hyperedges per handcrafted hypergraph')
# parser.add_argument('--mode', type=str, default='RW', choices=['RW', 'HOP'])
# parser.add_argument('--dense', type=bool, default=False, help='whether to use dense implementation for gnn and hgnn from the beginning')
parser.add_argument('--epoch_select', type=str, default='val_loss_sup', choices=['val_acc', 'val_loss_sup', 'val_loss_sup_hse'])
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--feat_str', type=str, default='')
parser.add_argument('--height', type=int, default=3)
parser.add_argument('--mode', type=str, default='RW', choices=['RW', 'HOP'])
# parser.add_argument('--hypergraph_length_list', type=int, default=None)
parser.add_argument('--T2', type=float, default=1.0, help='temperature for fix_match loss')
parser.add_argument('--threshold', type=float, default=0.95, help='threshold for fix_match loss')
parser.add_argument('--weight_fix', type=float, default=1)
parser.add_argument('--weight_simlr', type=float, default=1)
parser.add_argument('--aug1', type=str, default='none', choices=['dropN', 'wdropN', 'permE', 'subgraph', 'maskN', 'none', 'random4', 'random3', 'random2'])
parser.add_argument('--aug_ratio1', type=float, default=0.2)
parser.add_argument('--aug2', type=str, default='none', choices=['dropN', 'wdropN', 'permE', 'subgraph', 'maskN', 'none', 'random4', 'random3', 'random2'])
parser.add_argument('--aug_ratio2', type=float, default=0.2)
parser.add_argument('--EPS', type=float, default=1e-15)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--H1_update', type=str, default='epoch', choices=['epoch', 'run', 'exp'])
parser.add_argument('--use_config_file', type=int, default=1)
parser.add_argument('--hgsl_arch', type=str, default='GCN')
# parser.add_argument('--model', type=str, default='HypSEE')
args = parser.parse_args()


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        super().__init__()
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

configs = parser.parse_args()

# configs.dataset = dataset

configs_dict = vars(configs)
# json_file = open(f'./configs/{configs.data_name}.json', 'w')
# json.dump(configs_dict, json_file)
# exit(0)
# print(f'./sensitivity/{configs.data_name}.json')
# if os.path.exists(f'./sensitivity/{configs.data_name}.json') and configs.use_config_file == 1:
with open(f'./anchor/{configs.data_name}.json', 'rt') as f:
    configs_dict.update(json.load(f))
f.close()
# else:
#     print("using default configs")
configs = DotDict(configs_dict)

configs['data_root'] = './data/'
configs["data_root"] = os.path.join(configs["data_root"], configs["data_name"])

assert configs.data_root == configs['data_root']
print(configs.data_root)

# print(configs)

result_root = './anchor/results'
configs['result_root'] = result_root
if not os.path.exists(result_root):
    os.mkdir(result_root)
if not os.path.exists(f"{result_root}/{configs.data_name}"):
    os.mkdir(f"{result_root}/{configs.data_name}")
# if not os.path.exists(f"./checkpoints/{configs.data_name}"):
#     os.mkdir(f"./checkpoints/{configs.data_name}")
# result_path = f"{result_root}/{configs.data_name}_num_edges.txt"

accs = []
stds = []
# for num_edges in [8]:
# for num_edges in [16, 32, 64, 128, 256]:

    # configs['num_edges1'] = num_edges
    # configs['num_edges2'] = num_edges

# for warm_epochs in [0, 4, 6, 8, 10]:
for warm_epochs in [8, 6, 10, 4, 0]:
    configs['warm_epochs'] = warm_epochs

    exp = Exp(configs)
    acc = exp.exp()
    std = exp.configs['std']

    accs.append(acc)
    stds.append(std)

    torch.cuda.empty_cache()
# with open(result_path, 'w') as f:
#     f.write("num_edges\t")
#     # for num_edges in [16, 32, 64, 128, 256]:
#     for num_edges in [64, 128, 256]:
#         f.write('{}\t'.format(num_edges))
#     f.write("\n")
#     f.write("acc:\t")
#     for i in range(len(accs)):
#         f.write("{}\t".format(accs[i]))
#     f.write("\n")
#     f.write("std:\t")
#     for i in range(len(stds)):
#         f.write("{}\t".format(stds[i]))
    # f.write("average:\t{}\t".format(np.mean(accs)))
    # f.write("std:\t{}\n".format(np.std(accs)))






# study = optuna.create_study()
# # study.optimize(objective_loss, n_trials=50).py
# # study.optimize(objective_net, n_trials=50)
#
# study.optimize(objective_critical, n_trials=200)

# # exp = Exp(configs)
# exp = ExpAug(configs)
# # acc = exp.exp_fold()
# acc = exp.exp()
# torch.cuda.empty_cache()


#1.  using HGNN in HEAL
#2. take EPS as a hyperparameter
#3. take GNN as a hyperparameter.

# Hypergraph Augmentation?

