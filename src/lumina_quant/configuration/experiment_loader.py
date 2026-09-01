"""Loader for cost-aware framework experiment configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from lumina_quant.configuration.experiment_schema import ExperimentConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    return loaded


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _resolve_path(root: Path, path_like: str) -> Path:
    raw = Path(path_like)
    return raw if raw.is_absolute() else (root / raw).resolve()


def load_experiment_config(experiment_path: str | Path) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Load and resolve experiment config with base/strategy includes."""
    experiment_file = Path(experiment_path).resolve()
    payload = _load_yaml(experiment_file)
    config_root = experiment_file.parent

    base_cfg: dict[str, Any] = {}
    base_path_raw = str(payload.get("base_config") or "").strip()
    if base_path_raw:
        base_path = _resolve_path(config_root, base_path_raw)
        base_cfg = _load_yaml(base_path)

    merged = _deep_merge(base_cfg, payload)

    strategy_blocks: list[dict[str, Any]] = []
    for strategy in list(merged.get("strategies") or []):
        if not isinstance(strategy, dict):
            continue
        strategy_cfg = {}
        strategy_path_raw = str(strategy.get("config") or "").strip()
        if strategy_path_raw:
            strategy_path = _resolve_path(config_root, strategy_path_raw)
            strategy_cfg = _load_yaml(strategy_path)
        merged_strategy = _deep_merge(strategy_cfg, strategy)
        merged_strategy.pop("config", None)
        strategy_blocks.append(merged_strategy)

    merged["strategies"] = strategy_blocks
    merged.pop("base_config", None)

    config = ExperimentConfig.from_dict(merged)
    return config, config.to_dict()
