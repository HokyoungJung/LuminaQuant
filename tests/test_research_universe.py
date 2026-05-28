from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

from lumina_quant.research_universe import (
    BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS,
    BINANCE_EXTENDED_RESEARCH_SYMBOLS,
    BINANCE_EXTENDED_RESEARCH_SYMBOLS_SLASHED,
    BINANCE_TRADFI_COMMODITY_SYMBOLS,
    BINANCE_TRADFI_ENERGY_INDUSTRIAL_COMMODITY_SYMBOLS,
    BINANCE_TRADFI_EQUITY_SYMBOLS,
    BINANCE_TRADFI_ETF_INDEX_SYMBOLS,
    BINANCE_TRADFI_PERP_RESEARCH_SYMBOLS,
    BINANCE_TRADFI_PERP_RESEARCH_SYMBOLS_SLASHED,
    BINANCE_TRADFI_PRECIOUS_METAL_SYMBOLS,
    BINANCE_TRADFI_PREMARKET_SYMBOLS,
    compact_to_slashed_usdt,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, relative_path: str) -> ModuleType:
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_binance_extended_research_universe_is_unique_and_complete() -> None:
    assert len(BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS) == 10
    assert len(BINANCE_TRADFI_PRECIOUS_METAL_SYMBOLS) == 4
    assert len(BINANCE_TRADFI_ENERGY_INDUSTRIAL_COMMODITY_SYMBOLS) == 4
    assert len(BINANCE_TRADFI_COMMODITY_SYMBOLS) == 8
    assert len(BINANCE_TRADFI_ETF_INDEX_SYMBOLS) == 5
    assert len(BINANCE_TRADFI_EQUITY_SYMBOLS) == 43
    assert len(BINANCE_TRADFI_PREMARKET_SYMBOLS) == 2
    assert len(BINANCE_TRADFI_PERP_RESEARCH_SYMBOLS) == 58
    assert len(BINANCE_EXTENDED_RESEARCH_SYMBOLS) == 68

    counts = Counter(BINANCE_EXTENDED_RESEARCH_SYMBOLS)
    assert [symbol for symbol, count in counts.items() if count > 1] == []
    assert "OPENAIUSDT" in BINANCE_TRADFI_PREMARKET_SYMBOLS
    assert "NATGASUSDT" in BINANCE_TRADFI_COMMODITY_SYMBOLS
    assert "SOXLUSDT" in BINANCE_TRADFI_ETF_INDEX_SYMBOLS
    assert "NVDAUSDT" in BINANCE_TRADFI_EQUITY_SYMBOLS


def test_compact_to_slashed_usdt_handles_short_and_long_bases() -> None:
    assert compact_to_slashed_usdt("CLUSDT") == "CL/USDT"
    assert compact_to_slashed_usdt("COPPERUSDT") == "COPPER/USDT"
    assert compact_to_slashed_usdt("CRCLUSDT") == "CRCL/USDT"
    assert compact_to_slashed_usdt("OPENAIUSDT") == "OPENAI/USDT"
    assert BINANCE_TRADFI_PERP_RESEARCH_SYMBOLS_SLASHED[0] == "XAU/USDT"
    assert len(BINANCE_EXTENDED_RESEARCH_SYMBOLS_SLASHED) == len(BINANCE_EXTENDED_RESEARCH_SYMBOLS)


def test_research_universe_feeds_inventory_and_monitoring_defaults() -> None:
    inventory = _load_script_module(
        "build_multiasset_exchange_coverage_inventory_for_universe_test",
        "scripts/research/build_multiasset_exchange_coverage_inventory.py",
    )
    monitoring = _load_script_module(
        "run_alpha_zoo_multi_asset_monitoring_slate_for_universe_test",
        "scripts/research/run_alpha_zoo_multi_asset_monitoring_slate.py",
    )
    assert inventory.DEFAULT_SYMBOLS == BINANCE_EXTENDED_RESEARCH_SYMBOLS_SLASHED

    monitoring_symbols = {
        symbol for members in monitoring.ASSET_GROUPS.values() for symbol in members
    }
    missing = set(BINANCE_TRADFI_PERP_RESEARCH_SYMBOLS) - monitoring_symbols
    assert missing == set()

    for group in (
        "precious_metal_proxy",
        "tradfi_energy_industrial_commodity",
        "tradfi_etf_index",
        "tradfi_equity",
        "tradfi_premarket",
    ):
        assert group in monitoring.ASSET_GROUPS
