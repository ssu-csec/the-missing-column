"""Config loading. All fixed thresholds and constants live in configs/*.yaml so they are auditable in one place.

Usage:
    cfg = load_configs()            # merged view of models/data/pools/beacon.yaml
    cfg.models["llama31_70"].n_layers
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_env(path: Path | None = None) -> list[str]:
    """Load REPO_ROOT/.env into os.environ without a hard python-dotenv dependency.

    Returns the key names found (never the values). Existing env vars are not overwritten,
    so a real shell export wins over the file. Secrets stay out of the transcript.
    """
    env_path = path or (REPO_ROOT / ".env")
    keys: list[str] = []
    if not env_path.exists():
        return keys
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))
        keys.append(key)
    return keys


@dataclass(frozen=True)
class Configs:
    models: dict[str, Any]
    data: dict[str, Any]
    pools: dict[str, Any]
    beacon: dict[str, Any]


def _read(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name) as fh:
        return yaml.safe_load(fh)


def load_configs(config_dir: Path | None = None) -> Configs:
    """Load and return the four YAML configs as one frozen object."""
    global CONFIG_DIR
    if config_dir is not None:
        CONFIG_DIR = Path(config_dir)
    return Configs(
        models=_read("models.yaml"),
        data=_read("data.yaml"),
        pools=_read("pools.yaml"),
        beacon=_read("beacon.yaml"),
    )
