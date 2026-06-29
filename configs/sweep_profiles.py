"""Sweep profile registry and builders."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from configs import sweep_base, sweep_quick


@dataclass(frozen=True)
class SweepProfile:
    name: str
    spec: dict[str, Any]
    int_keys: frozenset[str]
    bool_keys: frozenset[str]
    fixed_run_defaults: dict[str, Any]
    default_sweep_name: str | None = None

    def coerce_value(self, key: str, value: Any) -> Any:
        if key in self.bool_keys:
            return bool(value)
        if key in self.int_keys:
            return int(value)
        return value

    def param_keys(self, sweep_cfg: dict[str, Any] | None = None) -> tuple[str, ...]:
        cfg = sweep_cfg if sweep_cfg is not None else self.spec
        return tuple(cfg['parameters'].keys())

    def apply_fixed_run_defaults(self, run_configs) -> None:
        for key, value in self.fixed_run_defaults.items():
            setattr(run_configs, key, value)


SWEEP_PROFILES: dict[str, SweepProfile] = {
    sweep_base.PROFILE_NAME: SweepProfile(
        name=sweep_base.PROFILE_NAME,
        spec=sweep_base.SWEEP_SPEC,
        int_keys=sweep_base.INT_SWEEP_KEYS,
        bool_keys=sweep_base.BOOL_SWEEP_KEYS,
        fixed_run_defaults=sweep_base.FIXED_RUN_DEFAULTS,
        default_sweep_name='hypsee_base',
    ),
    sweep_quick.PROFILE_NAME: SweepProfile(
        name=sweep_quick.PROFILE_NAME,
        spec=sweep_quick.SWEEP_SPEC,
        int_keys=sweep_quick.INT_SWEEP_KEYS,
        bool_keys=sweep_quick.BOOL_SWEEP_KEYS,
        fixed_run_defaults=sweep_quick.FIXED_RUN_DEFAULTS,
        default_sweep_name='hypsee_quick',
    ),
}

SWEEP_PROFILE_CHOICES = tuple(SWEEP_PROFILES.keys())


def get_sweep_profile(profile: str) -> SweepProfile:
    if profile not in SWEEP_PROFILES:
        raise ValueError(
            f"Unknown sweep profile {profile!r}. "
            f"Choose from {SWEEP_PROFILE_CHOICES}."
        )
    return SWEEP_PROFILES[profile]


def build_sweep_configuration(profile: str = sweep_base.PROFILE_NAME) -> dict[str, Any]:
    """Return a deep copy of the sweep spec for the given profile."""
    sweep_profile = get_sweep_profile(profile)
    return copy.deepcopy(sweep_profile.spec)
