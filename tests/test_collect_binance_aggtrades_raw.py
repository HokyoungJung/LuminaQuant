from __future__ import annotations

from lumina_quant.data_collector import collect_binance_aggtrades_raw
from lumina_quant.storage.parquet import ParquetMarketDataRepository


class _ExchangeStub:
    def close(self):
        return None


def test_collect_binance_aggtrades_raw_checkpoint_resume(tmp_path, monkeypatch):
    batches = {
        0: [
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_700_000_000_000,
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            },
            {
                "agg_trade_id": 2,
                "timestamp_ms": 1_700_000_001_000,
                "price": 100.5,
                "quantity": 0.2,
                "is_buyer_maker": True,
            },
        ],
        1_700_000_001_001: [
            {
                "agg_trade_id": 3,
                "timestamp_ms": 1_700_000_002_000,
                "price": 101.0,
                "quantity": 0.3,
                "is_buyer_maker": False,
            },
        ],
    }

    monkeypatch.setattr(
        "lumina_quant.data_collector.create_binance_exchange", lambda **_: _ExchangeStub()
    )

    def _fetch(*, exchange, symbol, since_ms, limit, retries, base_wait_sec):
        _ = exchange, symbol, limit, retries, base_wait_sec
        return list(batches.get(int(since_ms), []))

    monkeypatch.setattr("lumina_quant.data_collector.fetch_aggtrades_batch", _fetch)

    first = collect_binance_aggtrades_raw(
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        since_ms=0,
        until_ms=1_700_000_010_000,
        limit=1000,
        max_batches=10,
    )

    second = collect_binance_aggtrades_raw(
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        since_ms=None,
        until_ms=1_700_000_010_000,
        limit=1000,
        max_batches=10,
    )

    repo = ParquetMarketDataRepository(str(tmp_path))
    raw = repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT")

    assert int(first["fetched_rows"]) == 2
    assert int(second["fetched_rows"]) == 1
    assert raw.height == 3

    checkpoint = repo.read_raw_checkpoint(exchange="binance", symbol="BTC/USDT")
    last_id = checkpoint.get("last_trade_id", checkpoint.get("last_agg_trade_id"))
    assert int(last_id) == 3


def test_collect_binance_aggtrades_raw_bootstrap_lookback_used_without_checkpoint(
    tmp_path, monkeypatch
):
    observed_since: list[int] = []

    monkeypatch.setattr(
        "lumina_quant.data_collector.create_binance_exchange", lambda **_: _ExchangeStub()
    )

    def _fetch(*, exchange, symbol, since_ms, limit, retries, base_wait_sec):
        _ = exchange, symbol, limit, retries, base_wait_sec
        observed_since.append(int(since_ms))
        return []

    monkeypatch.setattr("lumina_quant.data_collector.fetch_aggtrades_batch", _fetch)

    until_ms = 1_700_000_010_000
    result = collect_binance_aggtrades_raw(
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        since_ms=None,
        until_ms=until_ms,
        bootstrap_lookback_hours=2,
        limit=1000,
        max_batches=10,
    )

    expected_since = until_ms - (2 * 60 * 60 * 1000)
    assert observed_since == [expected_since]
    assert int(result["start_cursor_ms"]) == expected_since
