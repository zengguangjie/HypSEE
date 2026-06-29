"""Minimal sweep profile for smoke tests and quick exploration."""
from __future__ import annotations

from typing import Any

PROFILE_NAME = 'quick'

SWEEP_SPEC: dict[str, Any] = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "final/avg_test_acc"},
    "parameters": {
        'lr': {
            'values': [0.001, 0.01],
        },
        'dropout': {
            'values': [0.3, 0.5],
        },
        'weight_hse': {
            'values': [0.0001, 0.001, 0.01],
        },
        'feat_str': {
            'values': ['deg', 'deg+odeg10'],
        },
    },
}

INT_SWEEP_KEYS = frozenset()
BOOL_SWEEP_KEYS = frozenset()

FIXED_RUN_DEFAULTS = {
    'epochs': 2,
    'runs': 1,
}
