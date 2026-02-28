"""Plugin interface for timeframe-tunable research strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl


class StrategyPlugin(ABC):
    """Minimal contract for cost-aware framework strategy plugins."""

    @abstractmethod
    def compute_features(self, data: pl.DataFrame, params: dict) -> pl.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def compute_signal(self, features: pl.DataFrame, params: dict) -> pl.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def signal_to_targets(self, raw_signal: pl.DataFrame, params: dict) -> pl.DataFrame:
        raise NotImplementedError

    def expected_return_estimate(self, raw_signal: pl.DataFrame, params: dict) -> pl.DataFrame:
        _ = params
        if "signal" not in raw_signal.columns:
            return raw_signal
        return raw_signal.with_columns(pl.col("signal").alias("expected_return"))

    def confidence(self, raw_signal: pl.DataFrame, params: dict) -> pl.DataFrame:
        _ = params
        if "signal" not in raw_signal.columns:
            return raw_signal
        return raw_signal.with_columns(pl.col("signal").abs().alias("confidence"))


_PLUGIN_REGISTRY: dict[str, type[StrategyPlugin]] = {}


def register_plugin(name: str):
    def _decorator(cls: type[StrategyPlugin]) -> type[StrategyPlugin]:
        _PLUGIN_REGISTRY[str(name)] = cls
        return cls

    return _decorator


def get_plugin(name: str) -> StrategyPlugin:
    key = str(name)
    cls = _PLUGIN_REGISTRY.get(key)
    if cls is None:
        raise KeyError(f"Unknown strategy plugin: {name}")
    return cls()


def list_plugins() -> list[str]:
    return sorted(_PLUGIN_REGISTRY.keys())


__all__ = ["StrategyPlugin", "get_plugin", "list_plugins", "register_plugin"]
