"""Hydra/OmegaConf configuration loading without changing working directory."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def load_config(
    project_root: Path,
    config_name: str = "final",
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    """Compose one project configuration from the explicit config directory."""
    config_dir = (project_root / "configs").resolve()
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Configuration directory not found: {config_dir}")
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name=config_name, overrides=list(overrides or []))
    OmegaConf.resolve(config)
    return config


def resolved_config_dict(config: DictConfig) -> dict:
    """Convert a resolved DictConfig to plain containers for hashing/reporting."""
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("The root configuration must be a mapping")
    return value
