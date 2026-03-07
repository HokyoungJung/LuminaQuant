from __future__ import annotations

import queue
from datetime import UTC, datetime

from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.market_data import upsert_futures_feature_points_rows


def test_historic_handler_exposes_feature_points_from_sidecar_store(tmp_path):
    db_path = tmp_path / "market_parquet"
    upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {"timestamp_ms": 1_700_000_000_000, "funding_rate": 0.0002},
            {"timestamp_ms": 1_700_000_060_000, "open_interest": 2_000_000.0},
        ],
    )

    rows = [
        (datetime.fromtimestamp(1_700_000_000_000 / 1000, tz=UTC), 1.0, 1.0, 1.0, 1.0, 10.0),
        (datetime.fromtimestamp(1_700_000_060_000 / 1000, tz=UTC), 2.0, 2.0, 2.0, 2.0, 20.0),
    ]
    handler = HistoricCSVDataHandler(
        queue.Queue(),
        str(tmp_path),
        ["BTC/USDT"],
        data_dict={"BTC/USDT": rows},
        feature_db_path=str(db_path),
        feature_exchange="binance",
    )

    handler.update_bars()
    assert handler.get_latest_feature_value("BTC/USDT", "funding_rate") == 0.0002
    assert handler.get_latest_bar_value("BTC/USDT", "funding_rate") == 0.0002
    assert handler.get_latest_feature_value("BTC/USDT", "open_interest") is None

    handler.update_bars()
    assert handler.get_latest_feature_value("BTC/USDT", "open_interest") == 2_000_000.0
    assert handler.get_latest_bar_value("BTC/USDT", "open_interest") == 2_000_000.0
