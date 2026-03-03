from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError
from lumina_quant.live.data_materialized import MaterializedWindowReader


class _RepoStub:
    @staticmethod
    def load_committed_ohlcv_chunked(**kwargs):
        _ = kwargs
        return pl.DataFrame(
            schema={
                "datetime": pl.Datetime(time_unit="ms"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )


def test_live_reader_no_empty_window_degradation_on_missing_committed_data():
    reader = MaterializedWindowReader(
        root_path="data/market_parquet",
        exchange="binance",
        symbol_list=["BTC/USDT"],
        window_seconds=5,
        staleness_threshold_seconds=30,
    )
    reader.repo = _RepoStub()

    with pytest.raises(RawFirstDataMissingError) as exc:
        reader.read_snapshot()
    assert "Committed 1s data missing for binance:BTC/USDT." in str(exc.value)


def test_live_cli_fail_fast_exit_contract_present():
    source = Path("src/lumina_quant/cli/live.py").read_text(encoding="utf-8")
    assert "LiveDataFatalError" in source
    assert "SystemExit(2)" in source
