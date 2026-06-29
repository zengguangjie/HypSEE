import zlib


def make_trial_seed(base_seed: int, run_id: str) -> int:
    """Stable per-trial seed from CLI --seed and wandb run id."""
    return int(base_seed) + (zlib.crc32(run_id.encode()) % 1_000_000)
