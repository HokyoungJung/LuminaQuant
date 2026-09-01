from __future__ import annotations

from pathlib import Path

import polars as pl

from lumina_quant.market_data import load_data_dict_from_external_root


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime": [1_700_000_000_000, 1_700_000_001_000],
            "open": [1.0, 1.1],
            "high": [1.2, 1.3],
            "low": [0.9, 1.0],
            "close": [1.1, 1.2],
            "volume": [10.0, 11.0],
        }
    ).with_columns(pl.from_epoch("datetime", time_unit="ms").alias("datetime"))


def test_external_market_data_loader_reads_symbol_map_csv(tmp_path: Path):
    frame = _frame()
    csv_path = tmp_path / "custom_btc.csv"
    frame.write_csv(csv_path)

    loaded = load_data_dict_from_external_root(
        str(tmp_path),
        symbol_list=["BTC/USDT"],
        symbol_map={"BTC/USDT": "custom_btc.csv"},
    )

    assert "BTC/USDT" in loaded
    assert loaded["BTC/USDT"].height == 2


def test_external_market_data_loader_reads_single_parquet_file_for_one_symbol(tmp_path: Path):
    frame = _frame()
    parquet_path = tmp_path / "single.parquet"
    frame.write_parquet(parquet_path)

    loaded = load_data_dict_from_external_root(
        str(parquet_path),
        symbol_list=["BTC/USDT"],
    )

    assert "BTC/USDT" in loaded
    assert loaded["BTC/USDT"].height == 2
