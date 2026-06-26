import torch
import argparse
import os
import json
from Exp import Exp

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='PROTEINS')
parser.add_argument('--data_root', type=str, default='data/')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=80)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--dim_embedding', type=int, default=32)
parser.add_argument('--dim_embedding_gnn', type=int, default=32)
parser.add_argument('--num_edges1', type=int, default=32)
parser.add_argument('--num_edges2', type=int, default=32)
parser.add_argument('--num_layers_gnn', type=int, default=2, help='number of layers in the first encoder embedding.')
parser.add_argument('--num_anchors', type=int, default=64, help='number of anchor graphs to be used')
parser.add_argument('--beta', type=float, default=1, help='beta balances two loss values.')
parser.add_argument('--weight_hse', type=float, default=0.001, help="weight of hierarchical se loss")
parser.add_argument('--warm_epochs', type=int, default=2)
parser.add_argument('--epoch_select', type=str, default='val_loss_sup', choices=['val_acc', 'val_loss_sup', 'val_loss_sup_hse'])
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--feat_str', type=str, default='deg')
parser.add_argument('--height', type=int, default=3)
parser.add_argument('--mode', type=str, default='RW', choices=['RW', 'HOP'])
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
parser.add_argument('--H1_update', type=str, default='exp', choices=['epoch', 'exp'])
parser.add_argument('--use_config_file', action='store_true',
                    help='load hyperparameters from configs/{data_name}.json')
parser.add_argument('--gnn_arch', type=str, default='GCN',
                    choices=['GIN', 'GCN', 'GraphConv', 'SAGEConv', 'GATConv', 'GATv2Conv', 'SGConv', 'ARMAConv'])
parser.add_argument('--hgsl_constraint', type=str, default='sigmoid',
                    choices=['sigmoid', 'softplus', 'relu', 'topk', 'none'],
                    help='constraint on learned hypergraph incidence matrix H_T')
parser.add_argument('--hgsl_topk', type=int, default=0,
                    help='number of hyperedges kept per node when hgsl_constraint=topk; '
                         '0 means auto (num_edges2 // 4)')
parser.add_argument('--use_gnn_encoder_S', action='store_true',
                    help='apply a separate GNN encoder on View S before hypergraph encoding')
parser.add_argument('--debug', type=int, default=0, choices=[0, 1],
                    help='on AssertionError: print traceback and context, then stop instead of skipping seed')
parser.add_argument('--grad_clip', type=float, default=5.0,
                    help='max gradient norm; 0 disables clipping')
parser.add_argument('--use_wandb', action='store_true',
                    help='log training metrics to Weights & Biases')
parser.add_argument('--wandb_project', type=str, default='HypSEE',
                    help='wandb project name')
parser.add_argument('--wandb_name', type=str, default=None,
                    help='wandb run name; defaults to data_name')
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

def _load_configs():
    configs_dict = vars(args)
    if os.path.exists(f'./configs/{args.data_name}.json') and args.use_config_file:
        with open(f'./configs/{args.data_name}.json', 'rt') as f:
            configs_dict.update(json.load(f))
    else:
        print("using default configs")
    configs = DotDict(configs_dict)
    configs.data_root = os.path.normpath(configs.data_root)
    return configs


def _wandb_config(configs):
    return {
        k: v for k, v in dict(configs).items()
        if not k.startswith(('wandb', 'sweep'))
    }


def main():
    configs = _load_configs()

    if not os.path.exists("./results"):
        os.mkdir("./results")
    if not os.path.exists(f"./results/{configs.data_name}"):
        os.mkdir(f"./results/{configs.data_name}")
    if not os.path.exists(f"./checkpoints/{configs.data_name}"):
        os.mkdir(f"./checkpoints/{configs.data_name}")

    use_wandb = configs.use_wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=configs.wandb_project,
            name=configs.wandb_name or configs.data_name,
            config=_wandb_config(configs),
        )

    try:
        exp = Exp(configs)
        acc = exp.exp()
    finally:
        torch.cuda.empty_cache()
        if use_wandb:
            import wandb
            if wandb.run is not None:
                wandb.finish()
    return acc


if __name__ == '__main__':
    main()

TODO：clusternet, hgcn, unignnII