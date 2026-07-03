from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumina_quant.strategies import registry as strategy_registry


def test_registry_includes_rsi_and_moving_average_strategies():
    mapping = strategy_registry.get_strategy_map()
    assert "RsiStrategy" in mapping
    assert "MovingAverageCrossStrategy" in mapping
    assert "RareEventScoreStrategy" in mapping


def test_resolve_strategy_class_rejects_unknown_explicit_name():
    # 2026-07-03 audit fix: a typo'd strategy name must raise (with suggestions),
    # never silently substitute the default strategy.
    import pytest

    from lumina_quant.strategies.registry import resolve_strategy_class

    with pytest.raises(ValueError, match="Unknown strategy"):
        resolve_strategy_class("RsiStrategyy")
    # Empty/None keeps the default-name fallback; strict=False keeps legacy.
    assert resolve_strategy_class(None) is not None
    assert resolve_strategy_class("NopeStrategy", strict=False) is not None
