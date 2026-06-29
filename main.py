import copy
import gc
import json
import os

import torch
import argparse

from Exp import Exp
from configs.sweep_profiles import (
    SWEEP_PROFILE_CHOICES,
    build_sweep_configuration,
    get_sweep_profile,
)
from utils.seed_utils import make_trial_seed

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='PROTEINS')
parser.add_argument('--data_root', type=str, default='data/')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=80)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay', type=float, default=0.01,
                    help='cosine annealing min lr ratio: eta_min = lr * lr_decay; 1.0 disables decay')
parser.add_argument('--weight_decay', type=float, default=0.0005,
                    help='L2 regularization coefficient in Adam (not learning rate decay)')
parser.add_argument('--dim_embedding', type=int, default=32)
parser.add_argument('--dim_embedding_gnn', type=int, default=32)
parser.add_argument('--num_edges2', type=int, default=32,
                    help='number of learned hyperedges in View T')
parser.add_argument('--num_edges1', type=int, default=None,
                    help='number of handcrafted hyperedges in View S; defaults to num_edges2 unless set via CLI, config, or sweep')
parser.add_argument('--num_layers_gnn', type=int, default=2, help='number of layers in the first encoder embedding.')
parser.add_argument('--num_anchors', type=int, default=64, help='number of anchor graphs to be used')
parser.add_argument('--beta', type=float, default=1, help='beta balances two loss values.')
parser.add_argument('--weight_hse', type=float, default=0.001, help="weight of hierarchical se loss")
parser.add_argument('--warm_epochs', type=int, default=2)
parser.add_argument('--epoch_select', type=str, default='val_acc_eq',
                    choices=['val_acc', 'val_acc_eq', 'val_loss_sup', 'val_loss_sup_hse', 'test_acc'])
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--seed', type=int, default=0, help='Base random seed')
parser.add_argument('--stratified_split', type=int, default=1, choices=[0, 1],
                    help='1 (default): stratified train/val/test split by class; '
                         '0: shuffle dataset and slice by index position')
parser.add_argument('--feat_str', type=str, default='cent')
parser.add_argument('--height', type=int, default=3)
parser.add_argument('--mode', type=str, default='RW', choices=['RW', 'HOP'])
parser.add_argument('--T2', type=float, default=1.0, help='temperature for fix_match loss')
parser.add_argument('--threshold', type=float, default=0.95, help='threshold for fix_match loss')
parser.add_argument('--weight_fix', type=float, default=1)
parser.add_argument('--weight_simlr', type=float, default=1)
parser.add_argument('--aug1', type=str, default='none',
                    choices=['permE', 'maskN', 'maskN_permE', 'none', 'random_mask_permE'])
parser.add_argument('--aug_ratio1', type=float, default=0.2)
parser.add_argument('--aug2', type=str, default='none',
                    choices=['dropN', 'wdropN', 'permE', 'subgraph', 'maskN', 'maskN_permE',
                             'none', 'random4', 'random3', 'random2', 'random_mask_permE'])
parser.add_argument('--aug_ratio2', type=float, default=0.2)
parser.add_argument('--EPS', type=float, default=1e-15,
                    help='numerical stability epsilon for model ops (hse_loss uses a separate fixed value)')
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--H1_update', type=str, default='exp', choices=['epoch', 'exp'])
parser.add_argument('--use_config_file', action='store_true',
                    help='load hyperparameters from configs/{data_name}.json')
parser.add_argument('--gnn_arch', type=str, default='GCN',
                    choices=['GIN', 'GCN', 'GCN2Conv', 'GraphConv', 'SAGEConv', 'GATConv', 'GATv2Conv', 'SGConv', 'ARMAConv'])
parser.add_argument('--hgsl_constraint', type=str, default='relu',
                    choices=['sigmoid', 'softplus', 'relu', 'topk'],
                    help='constraint on learned hypergraph incidence matrix H_T')
parser.add_argument('--hgsl_topk', type=int, default=0,
                    help='number of hyperedges kept per node when hgsl_constraint=topk; '
                         '0 means auto (num_edges2 // 4)')
parser.add_argument('--hyper_conv', type=str, default='heal',
                    choices=['hypergraph', 'heal', 'unigcnii', 'unigat'],
                    help='hypergraph convolution module in HyperHierarchicalGRL')
parser.add_argument('--shared_hyper_encoder', type=int, default=1, choices=[0, 1],
                    help='1 (default): one HyperHierarchicalGRL for View S and T; '
                         '0: separate encoders for num_edges1 / num_edges2 (CLI only, ignores config file)')
parser.add_argument('--no_gnn_encoder_S', dest='use_gnn_encoder_S', action='store_false',
                    help='disable the separate GNN encoder on View S before hypergraph encoding')
parser.add_argument('--debug', type=int, default=0, choices=[0, 1],
                    help='on AssertionError: print traceback and context, then stop instead of skipping seed')
parser.add_argument('--grad_clip', type=float, default=5.0,
                    help='max gradient norm; 0 disables clipping')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout rate for GNN encoder, classifier MLP, and hypergraph conv (0 disables)')
parser.add_argument('--pool_type', type=str, default='clusternet',
                    choices=['linear', 'clusternet', 'unigat'],
                    help='hierarchical pool layer in HyperHierarchicalGRL')
parser.add_argument('--use_wandb', action='store_true',
                    help='log training metrics to Weights & Biases')
parser.add_argument('--wandb_project', type=str, default='HypSEE',
                    help='wandb project name')
parser.add_argument('--wandb_name', type=str, default=None,
                    help='wandb run name; defaults to data_name')
parser.add_argument('--sweep', action='store_true',
                    help='run a wandb hyperparameter sweep (bayes search)')
parser.add_argument('--sweep_profile', type=str, default='base',
                    choices=list(SWEEP_PROFILE_CHOICES),
                    help='sweep parameter profile (see configs/sweep_*.py)')
parser.add_argument('--sweep_iters', type=int, default=500,
                    help='max number of wandb sweep runs')
parser.add_argument('--sweep_name', type=str, default=None,
                    help='display name for the wandb sweep in the UI (default: wandb auto-generated id)')
configs = parser.parse_args()


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


def _validate_hyper_encoder_config(configs_dict):
    if not configs_dict.get('shared_hyper_encoder', True):
        return
    if configs_dict.get('hyper_conv') != 'heal':
        return
    num_edges1 = configs_dict['num_edges1']
    num_edges2 = configs_dict['num_edges2']
    if num_edges1 != num_edges2:
        raise ValueError(
            f"shared_hyper_encoder with hyper_conv='heal' requires num_edges1 == num_edges2, "
            f"got num_edges1={num_edges1}, num_edges2={num_edges2}. "
            f"Use --shared_hyper_encoder 0 or set equal edge counts."
        )


def _resolve_num_edges1(configs_dict, num_edges1_cli):
    """Use num_edges2 unless num_edges1 is explicitly set (CLI, config file, or sweep)."""
    if num_edges1_cli is not None:
        configs_dict['num_edges1'] = num_edges1_cli
    elif configs_dict.get('num_edges1') is None:
        configs_dict['num_edges1'] = configs_dict['num_edges2']


def _load_configs(run_args, *, allow_config_file=True):
    configs_dict = vars(run_args) if not isinstance(run_args, dict) else dict(run_args)
    configs_dict = copy.deepcopy(configs_dict)
    use_gnn_encoder_S = configs_dict['use_gnn_encoder_S']
    shared_hyper_encoder = bool(configs_dict['shared_hyper_encoder'])
    num_edges1_cli = configs_dict.get('num_edges1')
    if (allow_config_file
            and os.path.exists(f'./configs/{configs_dict["data_name"]}.json')
            and configs_dict.get('use_config_file')):
        with open(f'./configs/{configs_dict["data_name"]}.json', 'rt') as f:
            configs_dict.update(json.load(f))
    elif not configs_dict.get('use_config_file'):
        print("using default configs")
    configs_dict['use_gnn_encoder_S'] = use_gnn_encoder_S
    configs_dict['shared_hyper_encoder'] = shared_hyper_encoder
    _resolve_num_edges1(configs_dict, num_edges1_cli)
    loaded = DotDict(configs_dict)
    loaded.data_root = os.path.normpath(loaded.data_root)
    _validate_hyper_encoder_config(loaded)
    return loaded


def _copy_run_configs(base_configs):
    """Per-run copy so sweep trials do not mutate the global argparse namespace."""
    return argparse.Namespace(**{k: copy.deepcopy(v) for k, v in vars(base_configs).items()})


def _merge_wandb_sweep_params(run_configs, sweep_profile, sweep_param_keys: tuple[str, ...]):
    """Apply sampled sweep hyperparameters from wandb.config onto run_configs."""
    import wandb
    if wandb.run is None:
        return run_configs
    for key in sweep_param_keys:
        if key in wandb.config:
            setattr(run_configs, key, sweep_profile.coerce_value(key, wandb.config[key]))
    if 'num_edges2' in wandb.config:
        run_configs.num_edges1 = run_configs.num_edges2
    sweep_profile.apply_fixed_run_defaults(run_configs)
    return run_configs


def _wandb_config(configs):
    return {
        k: v for k, v in dict(configs).items()
        if not k.startswith(('wandb', 'sweep'))
    }


def _release_cuda_memory():
    """Free Python-side references' GPU tensors and return cached blocks to the driver.

    ``empty_cache`` alone only releases blocks with no live references, so we run a
    full ``gc.collect`` first. This is critical in sweep mode where many trials run
    in the same long-lived process and any retained tensor accumulates across trials.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main(is_sweep=False):
    """Single training run.

    Three modes:
      - sweep : called by ``wandb.agent`` with ``is_sweep=True``; wandb injects
                sampled hyperparameters into ``wandb.config``.
      - wandb : ``--use_wandb`` without ``--sweep``; one wandb run logging CLI config.
      - plain : neither flag; no wandb logging.
    """
    run_args = _copy_run_configs(configs)
    sweep_profile = get_sweep_profile(configs.sweep_profile)
    sweep_cfg = build_sweep_configuration(configs.sweep_profile)
    sweep_param_keys = sweep_profile.param_keys(sweep_cfg)
    use_wandb = run_args.use_wandb or is_sweep

    if use_wandb:
        import wandb
        run_args.use_wandb = True
        if is_sweep:
            wandb.init()
            _merge_wandb_sweep_params(run_args, sweep_profile, sweep_param_keys)
            run_args.seed = make_trial_seed(configs.seed, wandb.run.id)
            loaded = _load_configs(run_args, allow_config_file=False)
            wandb.config.update({
                k: v for k, v in dict(loaded).items()
                if k not in sweep_param_keys and not k.startswith(('wandb', 'sweep'))
            }, allow_val_change=True)
            wandb.config.update({
                'base_seed': configs.seed,
                'sweep_profile': configs.sweep_profile,
            }, allow_val_change=True)
        else:
            loaded = _load_configs(run_args)
            wandb.init(
                project=loaded.wandb_project,
                name=loaded.wandb_name or loaded.data_name,
                config=_wandb_config(loaded),
            )
    else:
        loaded = _load_configs(run_args)

    os.makedirs(f"./results/{loaded.data_name}", exist_ok=True)
    os.makedirs(f"./checkpoints/{loaded.data_name}", exist_ok=True)

    acc = None
    exp = None
    try:
        exp = Exp(loaded)
        acc = exp.exp()
    finally:
        # Drop all references this trial holds before reclaiming CUDA memory, otherwise
        # cached model/optimizer/graph tensors survive into the next sweep trial.
        if exp is not None:
            exp.cleanup()
            del exp
        _release_cuda_memory()
        if use_wandb:
            import wandb
            if wandb.run is not None:
                wandb.finish()
    if acc is None:
        raise RuntimeError("experiment did not produce an accuracy result")
    return acc


def run_sweep():
    """Launch a wandb bayes sweep and run trials via the agent."""
    import wandb
    sweep_profile = get_sweep_profile(configs.sweep_profile)
    sweep_cfg = build_sweep_configuration(configs.sweep_profile)
    if configs.sweep_name:
        sweep_cfg['name'] = configs.sweep_name
    elif sweep_profile.default_sweep_name:
        sweep_cfg['name'] = f"{sweep_profile.default_sweep_name}_{configs.data_name}"
    print(f"Sweep profile={configs.sweep_profile!r}, data_name={configs.data_name!r}")
    print(f"  parameters: {list(sweep_cfg['parameters'].keys())}")
    sweep_id = wandb.sweep(sweep=sweep_cfg, project=configs.wandb_project)
    print(f"W&B sweep: {sweep_cfg.get('name', sweep_id)} (id={sweep_id})")
    wandb.agent(sweep_id, function=lambda: main(is_sweep=True), count=configs.sweep_iters)


if __name__ == '__main__':
    if configs.sweep:
        run_sweep()
    else:
        main(is_sweep=False)
