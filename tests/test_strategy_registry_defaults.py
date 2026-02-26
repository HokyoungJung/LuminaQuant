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
