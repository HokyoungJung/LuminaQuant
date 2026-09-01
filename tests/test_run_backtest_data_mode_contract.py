from __future__ import annotations

import polars as pl
import pytest
from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError
from lumina_quant.cli import backtest as run_backtest


def test_raw_first_requires_parquet_store(monkeypatch):
    monkeypatch.setattr(
        run_backtest, "is_parquet_market_data_store", lambda *_args, **_kwargs: False
    )

    with pytest.raises(RawFirstDataMissingError):
        run_backtest._load_data_dict(
            "db",
            "data/market_parquet",
            "binance",
            base_timeframe="1s",
            data_mode="raw-first",
            backtest_mode="windowed",
            auto_collect_db=False,
        )


def test_raw_first_loader_passes_data_mode_to_owner_entrypoint(monkeypatch):
    called: dict[str, object] = {}
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

    monkeypatch.setattr(
        run_backtest, "is_parquet_market_data_store", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(run_backtest, "SYMBOL_LIST", ["BTC/USDT"])

    def _loader(*args, **kwargs):
        _ = args
        called.update(kwargs)
        return {run_backtest.SYMBOL_LIST[0]: frame}

    monkeypatch.setattr(run_backtest, "load_data_dict_from_parquet", _loader)
    loaded = run_backtest._load_data_dict(
        "db",
        "data/market_parquet",
        "binance",
        base_timeframe="1s",
        data_mode="raw-first",
        backtest_mode="windowed",
        auto_collect_db=False,
    )

    assert run_backtest.SYMBOL_LIST[0] in loaded
    assert called.get("data_mode") == "raw-first"


def test_raw_first_rejects_non_windowed_backtest_mode():
    with pytest.raises(RawFirstDataMissingError):
        run_backtest.resolve_data_contract(
            data_mode="raw-first",
            backtest_mode="legacy_batch",
            data_source="db",
            default_backtest_mode="windowed",
            default_data_source="auto",
        )


def test_external_source_uses_external_loader(monkeypatch):
    called: dict[str, object] = {}
    monkeypatch.setattr(run_backtest, "SYMBOL_LIST", ["BTC/USDT"])

    def _loader(root_path, *, symbol_list, symbol_map=None, start_date=None, end_date=None):
        called["root_path"] = root_path
        called["symbol_list"] = list(symbol_list)
        called["symbol_map"] = dict(symbol_map or {})
        called["start_date"] = start_date
        called["end_date"] = end_date
        return {"BTC/USDT": object()}

    monkeypatch.setattr(run_backtest, "load_data_dict_from_external_root", _loader)
    loaded = run_backtest._load_data_dict(
        "external",
        "data/market_parquet",
        "binance",
        base_timeframe="1s",
        external_data_root="var/data/external/backtest",
        data_mode="legacy",
        backtest_mode="windowed",
        auto_collect_db=False,
    )

    assert "BTC/USDT" in loaded
    assert called["root_path"] == "var/data/external/backtest"
    assert called["symbol_map"] == {}


def test_external_single_file_rejects_multi_symbol(tmp_path):
    path = tmp_path / "single.parquet"
    path.write_bytes(b"stub")
    with pytest.raises(RuntimeError):
        run_backtest.load_data_dict_from_external_root(
            str(path),
            symbol_list=["BTC/USDT", "ETH/USDT"],
        )


def test_external_source_passes_symbol_map(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setattr(run_backtest, "SYMBOL_LIST", ["BTC/USDT"])

    def _loader(root_path, *, symbol_list, symbol_map=None, start_date=None, end_date=None):
        captured["root_path"] = root_path
        captured["symbol_map"] = dict(symbol_map or {})
        return {"BTC/USDT": object()}

    monkeypatch.setattr(run_backtest, "load_data_dict_from_external_root", _loader)
    loaded = run_backtest._load_data_dict(
        "external",
        "data/market_parquet",
        "binance",
        base_timeframe="1s",
        external_data_root="var/data/external/backtest",
        external_symbol_map={"BTC/USDT": "custom.csv"},
        data_mode="legacy",
        backtest_mode="windowed",
        auto_collect_db=False,
    )

    assert "BTC/USDT" in loaded
    assert captured["symbol_map"] == {"BTC/USDT": "custom.csv"}
