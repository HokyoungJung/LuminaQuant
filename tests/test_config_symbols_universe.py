from __future__ import annotations

from pathlib import Path

import yaml

from lumina_quant.configuration.schema import TradingConfig
from lumina_quant.research_universe import BINANCE_EXTENDED_RESEARCH_SYMBOLS_SLASHED


def test_config_omits_symbols_to_inherit_full_universe_default():
    # config.yaml intentionally does NOT pin trading.symbols: it inherits the
    # schema default, which is the full research universe and grows dynamically
    # as instruments are added to BINANCE_EXTENDED_RESEARCH_SYMBOLS_SLASHED.
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    trading = (payload or {}).get("trading") or {}
    assert "symbols" not in trading, (
        "config.yaml should omit trading.symbols so the universe stays dynamic; "
        "pin a list only for a deliberate custom live set."
    )


def test_schema_default_is_the_full_research_universe():
    assert list(TradingConfig().symbols) == list(BINANCE_EXTENDED_RESEARCH_SYMBOLS_SLASHED)
    assert len(TradingConfig().symbols) >= 100
