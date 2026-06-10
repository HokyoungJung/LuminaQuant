from __future__ import annotations

import sys
import textwrap

from lumina_quant.strategy_factory import runtime_settings


def test_current_research_market_data_settings_uses_explicit_runtime_mapping() -> None:
    settings = runtime_settings.current_research_market_data_settings(
        {
            "symbols": ["eth/usdt", "sol/usdt"],
            "market_data_parquet_path": "explicit/runtime/root",
            "market_data_exchange": "kraken",
        }
    )

    assert settings["symbols"] == ["ETH/USDT", "SOL/USDT"]
    assert settings["parquet_root"] == "explicit/runtime/root"
    assert settings["exchange"] == "kraken"


def test_default_research_symbol_universe_falls_back_when_config_import_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "lumina_quant.config", None)

    assert runtime_settings.default_research_symbol_universe() == (
        "BTC/USDT",
        "ETH/USDT",
        "XRP/USDT",
        "BNB/USDT",
        "SOL/USDT",
        "TRX/USDT",
        "DOGE/USDT",
        "ADA/USDT",
        "TON/USDT",
        "AVAX/USDT",
        "XAU/USDT",
        "XAG/USDT",
        "XPT/USDT",
        "XPD/USDT",
    )


def test_current_research_market_data_settings_reads_from_typed_config(
    tmp_path,
    monkeypatch,
) -> None:
    """current_research_market_data_settings() reads parquet_root and exchange from
    LQ_CONFIG_PATH YAML, replacing the old BaseConfig attribute-lookup approach."""
    cfg = textwrap.dedent(
        """
        storage:
          market_data_parquet_path: tmp/parquet
          market_data_exchange: bybit
        """
    ).strip()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    monkeypatch.setenv("LQ_CONFIG_PATH", str(cfg_path))

    assert runtime_settings.default_research_symbol_universe()[0] == "BTC/USDT"
    assert runtime_settings.current_research_market_data_settings() == {
        "symbols": list(runtime_settings._DEFAULT_SYMBOL_FALLBACK),
        "parquet_root": "tmp/parquet",
        "exchange": "bybit",
    }
