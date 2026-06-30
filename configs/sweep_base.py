"""Default W&B sweep parameter spec for HypSEE."""
from __future__ import annotations

from typing import Any

PROFILE_NAME = 'base'

SWEEP_SPEC: dict[str, Any] = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "final/avg_test_acc"},
    "parameters": {
        'lr': {
            'min': 1e-4,
            'max': 5e-2,
            'distribution': 'log_uniform_values',
        },
        'weight_decay': {
            'min': 1e-5,
            'max': 1e-3,
            'distribution': 'log_uniform_values',
        },
        # eta_min = lr * lr_decay; 1.0 disables cosine scheduler (see Exp.run)
        'lr_decay': {
            'values': [0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
        },
        'mode': {'value': 'RW'},
        'dropout': {
            'min': 0.0,
            'max': 0.8,
        },
        'dim_embedding': {
            'values': [16, 32, 64, 128, 256],
        },
        'dim_embedding_gnn': {
            'values': [16, 32, 64, 128, 256],
        },
        'beta': {
            'min': 0.1,
            'max': 10.0,
            'distribution': 'log_uniform_values',
        },
        'weight_hse': {
            'min': 1e-5,
            'max': 1e-1,
            'distribution': 'log_uniform_values',
        },
        # 'weight_fix': {
        #     'min': 0.01,
        #     'max': 10.0,
        #     'distribution': 'log_uniform_values',
        # },
        # 'weight_simlr': {
        #     'min': 0.001,
        #     'max': 1.0,
        #     'distribution': 'log_uniform_values',
        # },
        # 'threshold': {
        #     'min': 0.7,
        #     'max': 0.99,
        # },
        # 'T2': {
        #     'min': 0.1,
        #     'max': 5.0,
        #     'distribution': 'log_uniform_values',
        # },
        'num_layers_gnn': {
            'values': [2, 3, 4],
        },
        'gnn_arch': {
            'values': ['GCN', 'GIN', 'GraphConv', 'SAGEConv', 'GCN2Conv', 'GATConv'],
        },
        'hyper_conv': {
            'values': ['heal', 'hypergraph', 'unigcnii', 'unigat'],
        },
        'pool_type': {
            'values': ['clusternet', 'linear', 'unigat'],
        },
        'hgsl_constraint': {
            'values': ['relu', 'sigmoid', 'softplus', 'topk'],
        },
        'epoch_select': {
            'values': ['val_acc', 'val_acc_eq', 'val_loss_sup'],
            # 'values': ['test_acc'],
        },
        'stratified_split': {'values': [0, 1]},
        'warm_epochs': {'values': [0, 1, 2, 3, 5]},
        'height': {'values': [2, 3]},
        'decay_rate': {
            'min': 0.1,
            'max': 0.7,
        },
        'H1_update': {'values': ['epoch', 'exp']},
        'aug1': {
            'values': ['permE', 'maskN', 'maskN_permE', 'none', 'random_mask_permE'],
            # 'values': ['none', 'maskN_permE'],
        },
        'aug_ratio1': {
            'min': 0.05,
            'max': 0.7,
        },
        'aug2': {
            'values': [
                'dropN', 'wdropN', 'permE', 'subgraph', 'maskN', 'maskN_permE',
                'none', 'random4', 'random3', 'random2', 'random_mask_permE',
            ],
            # 'values': ['none', 'random_mask_permE'],
        },
        'aug_ratio2': {
            'min': 0.05,
            'max': 0.7,
        },
        # Preprocessed once per value -> data_{feat_str}.pt (see datasets/tu_dataset.py).
        # cent/ank omitted: slow or rarely used in dataset JSON configs.
        'feat_str': {
            'values': ['', 'deg', 'deg+odeg10', 'deg+odeg100', 'cent', 'ank1', 'deg+cent'],
        },
        'batch_size': {'values': [32, 64, 80, 128]},
        'num_anchors': {'values': [16, 32, 64, 128]},
        'num_edges2': {'values': [16, 32, 64, 128]},
    },
}

INT_SWEEP_KEYS = frozenset({
    'dim_embedding', 'dim_embedding_gnn', 'num_layers_gnn',
    'batch_size', 'num_anchors', 'num_edges2', 'hgsl_topk',
    'stratified_split', 'warm_epochs', 'height',
})
BOOL_SWEEP_KEYS = frozenset({'shared_hyper_encoder', 'use_gnn_encoder_S'})

# Applied after wandb sampling; overrides CLI for these keys during sweep trials.
FIXED_RUN_DEFAULTS = {
    'epochs': 100,
}
