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
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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


__all__ = [
    "GLOBAL_REGISTRY",
    "PluginRegistry",
    "register",
]
