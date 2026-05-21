"""Public strategy loading helpers.

External users can provide a strategy as ``module.path:ClassName``. The class
must be importable and implement ``on_bar(bar) -> Signal``. This module only
loads local Python classes; it does not contain private strategy registries or
production connectors.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

from lumina_quant.models import Bar, Signal
from lumina_quant.sample_strategy import MovingAverageCrossStrategy

DEFAULT_STRATEGY_REF = "lumina_quant.sample_strategy:MovingAverageCrossStrategy"
_SAMPLE_STRATEGY_REFS = {
    "sample_ma",
    "moving_average_cross",
    "MovingAverageCrossStrategy",
    DEFAULT_STRATEGY_REF,
    "lumina_quant.sample_strategy.MovingAverageCrossStrategy",
}
_ALIASES = {
    "sample_ma": DEFAULT_STRATEGY_REF,
    "moving_average_cross": DEFAULT_STRATEGY_REF,
    "MovingAverageCrossStrategy": DEFAULT_STRATEGY_REF,
}


@runtime_checkable
class StrategyProtocol(Protocol):
    def on_bar(self, bar: Bar) -> Signal: ...


def is_sample_strategy_ref(strategy_ref: str) -> bool:
    return (
        strategy_ref in _SAMPLE_STRATEGY_REFS
        or _ALIASES.get(strategy_ref) in _SAMPLE_STRATEGY_REFS
    )


def load_strategy_class(strategy_ref: str = DEFAULT_STRATEGY_REF) -> type[StrategyProtocol]:
    resolved = _ALIASES.get(strategy_ref, strategy_ref)
    if resolved == DEFAULT_STRATEGY_REF:
        return MovingAverageCrossStrategy
    module_name, class_name = _split_strategy_ref(resolved)
    module = importlib.import_module(module_name)
    strategy_cls = getattr(module, class_name)
    return strategy_cls


def build_strategy(
    strategy_ref: str = DEFAULT_STRATEGY_REF,
    strategy_params: Mapping[str, Any] | None = None,
) -> StrategyProtocol:
    strategy_cls = load_strategy_class(strategy_ref)
    instance = strategy_cls(**dict(strategy_params or {}))
    if not isinstance(instance, StrategyProtocol):
        raise TypeError(f"strategy '{strategy_ref}' must implement on_bar(bar)")
    return instance


def parse_strategy_param_assignments(items: Iterable[str] | None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"strategy parameter must use key=value form: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("strategy parameter key must not be empty")
        params[key] = _parse_scalar(raw_value.strip())
    return params


def _split_strategy_ref(strategy_ref: str) -> tuple[str, str]:
    if ":" in strategy_ref:
        module_name, class_name = strategy_ref.split(":", 1)
    else:
        module_name, _, class_name = strategy_ref.rpartition(".")
    if not module_name or not class_name:
        raise ValueError("strategy must be 'module:ClassName' or 'module.ClassName'")
    return module_name, class_name


def _parse_scalar(raw_value: str) -> Any:
    if not raw_value:
        return ""
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        pass
    lowered = raw_value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        return raw_value
