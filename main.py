import optuna
import torch
import argparse

from ExpAug import ExpAug
from datasets.tu_dataset import get_dataset, get_dataset_dense
import random
import os
import numpy as np
from torch_geometric.loader import DataLoader, DenseDataLoader
from datasets.loaders import IterLoader
# from models.model import HypSEE2
from datasets.data_utils import hypergraph_construction, hypergraph_construction_batch
from datasets.data_utils import load_hypergraphs, hypergraph_to_dense_batch
import torch.nn.functional as F
from copy import deepcopy
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
parser.add_argument('--feat_str', type=str, default='deg')
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
if os.path.exists(f'./configs/{configs.data_name}.json') and configs.use_config_file == 1:
    with open(f'./configs/{configs.data_name}.json', 'rt') as f:
        configs_dict.update(json.load(f))
    f.close()
else:
    print("using default configs")
configs = DotDict(configs_dict)


configs["data_root"] = os.path.join(configs["data_root"], configs["data_name"])

assert configs.data_root == configs['data_root']

# print(configs)

if not os.path.exists(f"./results"):
    os.mkdir("./results")
if not os.path.exists(f"./results/{configs.data_name}"):
    os.mkdir(f"./results/{configs.data_name}")
if not os.path.exists(f"./checkpoints/{configs.data_name}"):
    os.mkdir(f"./checkpoints/{configs.data_name}")


def objective_loss(trail):
    beta = trail.suggest_categorical('beta', [0.0, 1e-2, 1e-1, 1, 10])
    weight_hse = trail.suggest_categorical('weight_hse', [0, 1e-4, 1e-3, 1e-2])
    weight_fix = trail.suggest_categorical('weight_fix', [0, 1e-2, 1e-1, 1, 10])
    weight_simlr = trail.suggest_categorical('weight_simlr', [0, 1e-2, 1e-1, 1, 10])
    T2 = trail.suggest_categorical('T2', [0.5, 1, 2])
    threshold = trail.suggest_categorical('threshold', [0.9, 0.95, 0.975, 0.99])
    configs.beta = beta
    configs.weight_hse = weight_hse
    configs.weight_fix = weight_fix
    configs.weight_simlr = weight_simlr
    configs.T2 = T2
    configs.threshold = threshold

    exp = Exp(configs)
    acc = exp.exp()
    torch.cuda.empty_cache()
    return -acc

def objective_net(trail):
    lr = trail.suggest_categorical('lr', [1e-2, 2e-3, 1e-3, 1e-4])
    weight_decay = trail.suggest_categorical('weight_decay', [5e-3, 1e-3, 5e-4, 1e-4])
    dim_embedding = trail.suggest_categorical('dim_embedding', [32, 16, 64, 128])
    dim_embedding_gnn = trail.suggest_categorical('dim_embedding_gnn', [0, 32, 16, 64])
    num_edges2 = trail.suggest_categorical('num_edges2', [32, 64, 16, 128])
    num_layers_gnn = trail.suggest_categorical('num_layers_gnn', [1,2,3])
    epoch_select = trail.suggest_categorical('epoch_select', ['val_acc', 'val_loss_sup', 'val_loss_sup_hse', 'val_acc_eq'])
    configs.lr = lr
    configs.weight_decay = weight_decay
    configs.dim_embedding = dim_embedding
    configs.dim_embedding_gnn = dim_embedding_gnn
    configs.num_edges2 = num_edges2
    configs.layers_gnn = num_layers_gnn
    configs.epoch_select = epoch_select

    exp = Exp(configs)
    acc = exp.exp()
    torch.cuda.empty_cache()
    return -acc

def objective_net_fold(trail):
    lr = trail.suggest_categorical('lr', [1e-2, 2e-3, 1e-3, 1e-4])
    weight_decay = trail.suggest_categorical('weight_decay', [5e-3, 1e-3, 5e-4, 1e-4])
    dim_embedding = trail.suggest_categorical('dim_embedding', [32, 16, 64, 128])
    dim_embedding_gnn = trail.suggest_categorical('dim_embedding_gnn', [0, 32, 16, 64])
    num_edges2 = trail.suggest_categorical('num_edges2', [32, 64, 16, 128])
    num_layers_gnn = trail.suggest_categorical('num_layers_gnn', [1,2,3])
    epoch_select = trail.suggest_categorical('epoch_select', ['val_acc', 'val_loss_sup', 'val_loss_sup_hse', 'val_acc_eq'])
    configs.lr = lr
    configs.weight_decay = weight_decay
    configs.dim_embedding = dim_embedding
    configs.dim_embedding_gnn = dim_embedding_gnn
    configs.num_edges2 = num_edges2
    configs.layers_gnn = num_layers_gnn
    configs.epoch_select = epoch_select

    exp = Exp(configs)
    acc = exp.exp_fold()
    torch.cuda.empty_cache()
    return -acc

def objective_critical(trail):
    lr = trail.suggest_categorical('lr', [1e-2, 2e-3, 1e-3, 1e-4])
    weight_decay = trail.suggest_categorical('weight_decay', [5e-3, 1e-3, 5e-4, 1e-4])
    warm_epochs = trail.suggest_categorical('warm_epochs', [0,2])
    num_edges1 = trail.suggest_categorical('num_edges1', [16, 32, 64, 128, 256])
    dim_embedding = trail.suggest_categorical('dim_embedding', [16,32, 64, 128, 256])
    height = trail.suggest_categorical('height', [2,3,4,5])
    decay_rate = trail.suggest_categorical('decay_rate', [0.2, 0.5, 0.8])
    beta = trail.suggest_categorical('beta', [1, 0.1, 0.01])
    # aug1 = trail.suggest_categorical('aug1', ['dropN', 'permE', 'subgraph', 'maskN', 'none', 'random4', 'random3', 'random2'])
    # aug_ratio1 = trail.suggest_categorical('aug_ratio1', [0.2, 0.5, 0.8, 0.1])
    # aug2 = trail.suggest_categorical('aug2', ['dropN', 'permE', 'subgraph', 'maskN', 'none', 'random4', 'random3', 'random2'])
    # aug_ratio2 = trail.suggest_categorical('aug_ratio2', [0.2, 0.5, 0.8, 0.1])

    configs.lr = lr
    configs.weight_decay = weight_decay
    # configs.aug1 = aug1
    # configs.aug_ratio1 = aug_ratio1
    # configs.aug2 = aug2
    # configs.aug_ratio2 = aug_ratio2
    configs.warm_epochs = warm_epochs
    configs.num_edges1 = num_edges1
    configs.num_edges2 = num_edges1
    configs.dim_embedding = dim_embedding
    configs.dim_embedding_gnn = dim_embedding
    configs.height = height
    configs.decay_rate = decay_rate
    configs.beta = beta

    exp = Exp(configs)
    acc = exp.exp()
    torch.cuda.empty_cache()
    return -acc

def objective_IMDBM_HOP(trail):
    beta = trail.suggest_categorical('beta', [0.01, 0.1, 0.001, 1])
    print(beta)
    lr = trail.suggest_categorical('lr', [1e-2, 2e-3, 1e-3, 1e-4])
    decay_rate = trail.suggest_categorical('decay_rate', [0.2, 0.5, 0.8])
    num_edges1 = trail.suggest_categorical('num_edges1', [8, 16, 32, 64])
    # configs['height'] = 3
    height = trail.suggest_categorical('height', [2,3])
    feat_str = trail.suggest_categorical('feat_str', ["", "deg", "deg+odeg10", "deg+odeg100", "odeg100", "deg+odeg100+cent"])
    # mode = trail.suggest_categorical('mode', ['RW', 'HOP'])
    # configs['mode'] = 'RW'
    # configs['hypergraph_length_list'] = [4, 6]
    length1 = trail.suggest_categorical('length1', [1,2,3,4,5])
    length2 = trail.suggest_categorical('length2', [1,2,3,4,5])
    dim_embedding = trail.suggest_categorical('dim_embedding', [8, 16, 32, 64, 128])
    configs.height = height
    configs.decay_rate = decay_rate
    configs.beta = beta
    configs.num_edges1 = num_edges1
    configs.num_edges2 = num_edges1
    configs.lr = lr
    configs.dim_embedding = dim_embedding
    configs.dim_embedding_gnn = dim_embedding
    # if mode == 'HOP':
    #     length1 = int(max(1, length1-2))
    #     length2 = length2 - 2
    # configs.mode = mode
    configs['hypergraph_length_list'] = [length1, length2]
    # configs['hypergraph_length_list'] = [1,1]


    # configs['feat_str'] = 'deg+odeg100'
    configs['feat_str'] = feat_str

    exp = Exp(configs)
    acc = exp.exp()
    torch.cuda.empty_cache()
    return -acc







# study = optuna.create_study()
# # study.optimize(objective_loss, n_trials=50).py
# # study.optimize(objective_net, n_trials=50)
#
# # study.optimize(objective_critical, n_trials=200)
# study.optimize(objective_IMDBM_HOP, n_trials=200)

exp = Exp(configs)
# exp = ExpAug(configs)
# acc = exp.exp_fold()
acc = exp.exp()
torch.cuda.empty_cache()


#1.  using HGNN in HEAL
#2. take EPS as a hyperparameter
#3. take GNN as a hyperparameter.

# Hypergraph Augmentation?

