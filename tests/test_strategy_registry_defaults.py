from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumina_quant.strategies import registry as strategy_registry


def test_registry_exposes_only_public_sample_strategy_by_default():
    mapping = strategy_registry.get_strategy_map()
    assert mapping == {
        "PublicSampleStrategy": strategy_registry.resolve_strategy_class("PublicSampleStrategy")
    }
