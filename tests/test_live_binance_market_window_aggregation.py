from __future__ import annotations

from lumina_quant.live.binance_market_stream import BinanceMarketStreamClient
from lumina_quant.live.market_window_rolling import NormalizedTradeTick, RollingWindowAggregator


def test_rolling_window_aggregator_dedupes_duplicate_trade_events():
    agg = RollingWindowAggregator(
        symbol_list=["BTC/USDT"],
        window_seconds=3,
        max_lateness_ms=0,
    )

    first = NormalizedTradeTick(
        symbol="BTC/USDT",
        exchange_ts_ms=1_700_000_001_000,
        price=100.0,
        quantity=1.0,
        event_id="mkt:agg:BTC/USDT:1",
        receive_ts_ms=1_700_000_001_200,
    )
    dup = NormalizedTradeTick(
        symbol="BTC/USDT",
        exchange_ts_ms=1_700_000_001_000,
        price=100.0,
        quantity=1.0,
        event_id="mkt:agg:BTC/USDT:1",
        receive_ts_ms=1_700_000_001_300,
    )
    second = NormalizedTradeTick(
        symbol="BTC/USDT",
        exchange_ts_ms=1_700_000_002_000,
        price=101.0,
        quantity=2.0,
        event_id="mkt:agg:BTC/USDT:2",
        receive_ts_ms=1_700_000_002_200,
    )

    emitted = []
    emitted.extend(agg.ingest(first))
    emitted.extend(agg.ingest(dup))
    emitted.extend(agg.ingest(second))

    assert len(emitted) >= 2
    latest = emitted[-1]
    rows = latest.bars_1s["BTC/USDT"]
    assert rows[-2][5] == 1.0
    assert rows[-1][5] == 2.0
    assert rows[-1][4] == 101.0


def test_binance_market_stream_parser_normalizes_aggtrade_message():
    payload = {
        "stream": "btcusdt@aggTrade",
        "data": {
            "e": "aggTrade",
            "E": 1_700_000_003_000,
            "s": "BTCUSDT",
            "a": 42,
            "p": "102.5",
            "q": "0.75",
            "m": True,
        },
    }

    ticks = BinanceMarketStreamClient.parse_message(payload, receive_ts_ms=1_700_000_003_100)
    assert len(ticks) == 1
    tick = ticks[0]
    assert tick.symbol == "BTC/USDT"
    assert tick.exchange_ts_ms == 1_700_000_003_000
    assert tick.price == 102.5
    assert tick.quantity == 0.75
    assert tick.event_id == "mkt:agg:BTC/USDT:42"
