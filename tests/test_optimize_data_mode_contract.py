from __future__ import annotations

import polars as pl
import pytest
from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError
from lumina_quant.cli import optimize


def test_optimize_raw_first_rejects_csv_fallback():
    with pytest.raises(RawFirstDataMissingError):
        optimize.load_all_data(
            "data",
            ["BTC/USDT"],
            data_mode="raw-first",
            backtest_mode="windowed",
            data_source="csv",
            market_db_path="data/market_parquet",
            market_exchange="binance",
            timeframe="1s",
        )


def test_optimize_raw_first_passes_data_mode_to_owner_loader(monkeypatch):
    captured: dict[str, object] = {}
    frame = pl.DataFrame(
        {
            "datetime": [1_700_000_000_000],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    ).with_columns(pl.from_epoch("datetime", time_unit="ms").alias("datetime"))

    monkeypatch.setattr(optimize, "is_parquet_market_data_store", lambda *_args, **_kwargs: True)

    def _loader(*args, **kwargs):
        _ = args
        captured.update(kwargs)
        return {"BTC/USDT": frame}

    monkeypatch.setattr(optimize, "load_data_dict_from_parquet", _loader)
    loaded = optimize.load_all_data(
        "data",
        ["BTC/USDT"],
        data_mode="raw-first",
        backtest_mode="windowed",
        data_source="db",
        market_db_path="data/market_parquet",
        market_exchange="binance",
        timeframe="1s",
    )

    assert "BTC/USDT" in loaded
    assert captured.get("data_mode") == "raw-first"
