"""Decorator-based plugin registry for strategies, indicators, and portfolios.

Usage
-----
Register a plugin with the ``@register`` decorator::

    from lumina_quant.core.plugin_registry import register

    @register("strategy", "MySMA", interface="event_driven")
    class MySMAStrategy(Strategy):
        ...

    @register("indicator", "RSI")
    class RSIIndicator:
        ...

    @register("portfolio", "EqualWeight")
    class EqualWeightPortfolio:
        ...

Retrieve via the global registry accessor::

    from lumina_quant.core.plugin_registry import GLOBAL_REGISTRY

    cls = GLOBAL_REGISTRY.get("strategy", "MySMA")
    names = GLOBAL_REGISTRY.list_names("indicator")

Backward-compat helpers (CLI entrypoints)
-----------------------------------------
``load_strategy_registry`` / ``import_private_strategy_registry`` /
``PublicStubStrategy`` / ``PublicStrategyRegistry`` are preserved here for
the CLI ``backtest``, ``optimize``, and ``live`` entrypoints.
``cli/_strategy_registry_fallback.py`` has been deleted; this module is its
sole successor.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from importlib import import_module
from typing import Any

from lumina_quant.strategy import Strategy

# ---------------------------------------------------------------------------
# Core registry internals
# ---------------------------------------------------------------------------

# Structure: {kind: {name: {"cls": <type>, "interface": str | None}}}
_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {}


def register(
    kind: str,
    name: str | None = None,
    *,
    interface: str | None = None,
) -> Callable:
    """Class decorator that registers a plugin in the global registry.

    Parameters
    ----------
    kind:
        Plugin category — one of ``"strategy"``, ``"indicator"``,
        ``"portfolio"``, or any other string key.
    name:
        Registry lookup name.  Defaults to ``cls.__name__`` when omitted.
    interface:
        Optional tag describing the plugin's execution contract.  Canonical
        values for strategies: ``"event_driven"`` (subclasses
        ``Strategy``) and ``"polars_batch"`` (subclasses
        ``StrategyPlugin``).  Phase 4 will unify the dual ABC; for now both
        are registered with explicit tags.
    """

    def decorator(cls: type) -> type:
        _name = name if name is not None else cls.__name__
        kind_bucket = _REGISTRY.setdefault(kind, {})
        kind_bucket[_name] = {"cls": cls, "interface": interface}
        return cls

    return decorator


class PluginRegistry:
    """Read-only accessor for the global plugin registry populated by ``@register``."""

    def get(self, kind: str, name: str) -> type | None:
        """Return the registered class for *kind*/*name*, or ``None``."""
        return _REGISTRY.get(kind, {}).get(name, {}).get("cls")

    def list_kinds(self) -> frozenset[str]:
        """Return all registered plugin kinds."""
        return frozenset(_REGISTRY.keys())

    def list_names(self, kind: str) -> frozenset[str]:
        """Return all registered names for *kind*."""
        return frozenset(_REGISTRY.get(kind, {}).keys())

    def get_interface(self, kind: str, name: str) -> str | None:
        """Return the ``interface`` tag for *kind*/*name*, or ``None``."""
        return _REGISTRY.get(kind, {}).get(name, {}).get("interface")

    def get_all(self, kind: str) -> dict[str, type]:
        """Return ``{name: cls}`` for every registered plugin of *kind*."""
        return {n: v["cls"] for n, v in _REGISTRY.get(kind, {}).items()}


#: Singleton accessor — use this in application code.
GLOBAL_REGISTRY = PluginRegistry()

# ---------------------------------------------------------------------------
# Backward-compat CLI helpers (formerly cli/_strategy_registry_fallback.py)
# ---------------------------------------------------------------------------


class PublicStubStrategy(Strategy):
    """Fallback strategy used when private strategy modules are unavailable."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Strategy modules are unavailable in this distribution.")

    def calculate_signals(self, event):
        _ = event
        return None


class PublicStrategyRegistry:
    """Minimal registry fallback used by public/lightweight installs."""

    DEFAULT_STRATEGY_NAME = "PublicStubStrategy"

    @staticmethod
    def get_strategy_map() -> dict[str, type[Strategy]]:
        return {"PublicStubStrategy": PublicStubStrategy}

    @staticmethod
    def resolve_strategy_class(
        name: str,
        default_name: str | None = None,
    ) -> type[Strategy]:
        _ = (name, default_name)
        return PublicStubStrategy

    @staticmethod
    def get_default_strategy_params(strategy_name: str) -> dict[str, Any]:
        _ = strategy_name
        return {}

    @staticmethod
    def resolve_strategy_params(
        strategy_name: str,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = strategy_name
        return dict(overrides or {})

    @staticmethod
    def get_default_optuna_config(strategy_name: str) -> dict[str, Any]:
        _ = strategy_name
        return {"n_trials": 20, "params": {}}

    @staticmethod
    def get_default_grid_config(strategy_name: str) -> dict[str, Any]:
        _ = strategy_name
        return {"params": {}}

    @staticmethod
    def resolve_optuna_config(
        strategy_name: str,
        override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = strategy_name
        cfg: dict[str, Any] = {"n_trials": 20, "params": {}}
        if isinstance(override, dict):
            cfg.update(override)
        return cfg

    @staticmethod
    def resolve_grid_config(
        strategy_name: str,
        override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = strategy_name
        cfg: dict[str, Any] = {"params": {}}
        if isinstance(override, dict):
            cfg.update(override)
        return cfg


def load_strategy_registry(importer: Callable[[], Any]) -> Any:
    """Load private strategy registry with a stable public fallback."""
    try:
        return importer()
    except Exception as exc:
        print(
            f"[WARN] Strategy registry import failed; using public fallback: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return PublicStrategyRegistry()


def import_private_strategy_registry() -> Any:
    """Import the private/public strategy registry module when available."""
    return import_module("lumina_quant.strategies.registry")


__all__ = [
    # Decorator + accessor
    "register",
    "PluginRegistry",
    "GLOBAL_REGISTRY",
    # Backward-compat CLI helpers
    "PublicStubStrategy",
    "PublicStrategyRegistry",
    "load_strategy_registry",
    "import_private_strategy_registry",
]
