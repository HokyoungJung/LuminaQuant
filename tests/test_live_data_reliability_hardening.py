"""Tests for live-data reliability hardening (audit findings C5/D1/D2/D3/D4).

Covers:
  * C5 - market-stream recv-timeout handled inside the socket loop (no reconnect
    storm on idle), real errors routed through ``on_error`` with jittered
    exponential backoff, and gap recovery run after each successful reconnect.
  * D1 - paginated gap recovery from the per-symbol cursor, >1h chunking, and a
    truncation alert when the page budget is exhausted.
  * D2 - per-symbol last-real-trade tracking + stale-symbol detection.
  * D3 - ``listenKeyExpired`` passes through ``parse_message`` and forces a
    fresh-key reconnect; repeated keepalive failure forces a reconnect.
  * D4 - fail-closed bar-sanity gate at the MARKET_WINDOW seam, default OFF.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest
import websockets.sync.client as ws_sync

import lumina_quant.live.binance_market_stream as market_stream_mod
import lumina_quant.live.binance_user_stream as user_stream_mod
from lumina_quant.core.market_window_contract import (
    MarketWindowContractError,
    assert_bars_1s_sane,
    build_market_window_event,
)
from lumina_quant.live.binance_market_stream import (
    BinanceMarketStreamClient,
    BinanceMarketStreamConfig,
)
from lumina_quant.live.binance_user_stream import BinanceUserStreamClient
from lumina_quant.live.data_binance_live import BinanceLiveDataHandler
from lumina_quant.live.market_window_rolling import NormalizedTradeTick, RollingWindowAggregator


def _agg_trade_json(
    *, ts: int = 1_700_000_000_000, price: str = "100.0", qty: str = "1.0", a: int = 1
) -> str:
    return json.dumps(
        {
            "stream": "btcusdt@aggTrade",
            "data": {"e": "aggTrade", "E": ts, "s": "BTCUSDT", "a": a, "p": price, "q": qty},
        }
    )


class _FakeConn:
    """Minimal ``websockets.sync`` connection: scripted recv actions."""

    def __init__(self, actions, stop_event, *, stop_on_exhaust: bool):
        self._actions = list(actions)
        self._idx = 0
        self._stop_event = stop_event
        self._stop_on_exhaust = stop_on_exhaust

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def recv(self, timeout=None):
        if self._idx >= len(self._actions):
            if self._stop_on_exhaust:
                self._stop_event.set()
            raise TimeoutError()
        action = self._actions[self._idx]
        self._idx += 1
        if isinstance(action, BaseException):
            raise action
        if action == "timeout":
            raise TimeoutError()
        return action


def _install_fake_connect(monkeypatch, module, per_conn_actions, stop_event):
    counter = {"connect": 0}
    total = len(per_conn_actions)

    def _connect(url, **_kw):
        i = counter["connect"]
        counter["connect"] += 1
        actions = per_conn_actions[i] if i < total else []
        return _FakeConn(actions, stop_event, stop_on_exhaust=(i >= total - 1))

    monkeypatch.setattr(module, "connect", _connect)
    return counter


# --------------------------------------------------------------------------- C5


def test_idle_stream_does_not_reconnect(monkeypatch):
    stop_event = threading.Event()
    counter = _install_fake_connect(
        monkeypatch, ws_sync, [["timeout", "timeout", "timeout"]], stop_event
    )
    client = BinanceMarketStreamClient(BinanceMarketStreamConfig(symbols=["BTC/USDT"]))

    errors: list[Exception] = []
    reconnects: list[int] = []
    trades: list[NormalizedTradeTick] = []
    client.run_ws_loop(
        stop_event=stop_event,
        on_trade=trades.append,
        on_error=errors.append,
        on_reconnect=lambda: reconnects.append(1),
    )

    assert counter["connect"] == 1  # idle never dropped the socket
    assert errors == []
    assert reconnects == []
    assert trades == []


def test_idle_timeouts_keep_socket_open_for_later_trade(monkeypatch):
    stop_event = threading.Event()
    counter = _install_fake_connect(
        monkeypatch,
        ws_sync,
        [["timeout", "timeout", _agg_trade_json(a=7), "timeout"]],
        stop_event,
    )
    client = BinanceMarketStreamClient(BinanceMarketStreamConfig(symbols=["BTC/USDT"]))

    trades: list[NormalizedTradeTick] = []
    client.run_ws_loop(stop_event=stop_event, on_trade=trades.append)

    assert counter["connect"] == 1  # same connection survived the idle gaps
    assert len(trades) == 1
    assert trades[0].event_id == "mkt:agg:BTC/USDT:7"


def test_real_error_routes_on_error_and_recovers_on_reconnect(monkeypatch):
    stop_event = threading.Event()
    counter = _install_fake_connect(
        monkeypatch,
        ws_sync,
        [[ConnectionError("boom")], [_agg_trade_json(a=9)]],
        stop_event,
    )
    client = BinanceMarketStreamClient(BinanceMarketStreamConfig(symbols=["BTC/USDT"]))

    backoff_calls: list[int] = []
    client._reconnect_backoff_sleep = lambda **kw: backoff_calls.append(int(kw["attempt"]))

    errors: list[Exception] = []
    reconnects: list[int] = []
    trades: list[NormalizedTradeTick] = []
    client.run_ws_loop(
        stop_event=stop_event,
        on_trade=trades.append,
        on_error=errors.append,
        on_reconnect=lambda: reconnects.append(1),
    )

    assert counter["connect"] == 2
    assert len(errors) == 1 and isinstance(errors[0], ConnectionError)
    assert backoff_calls == [1]  # backed off once before the reconnect
    assert reconnects == [1]  # gap recovery fired after the reconnect
    assert len(trades) == 1


def test_reconnect_backoff_is_jittered_and_capped(monkeypatch):
    monkeypatch.setattr(market_stream_mod.random, "uniform", lambda _a, _b: 0.0)
    cfg = BinanceMarketStreamConfig(
        symbols=["BTC/USDT"], reconnect_delay_sec=1.0, max_reconnect_backoff_sec=10.0
    )
    client = BinanceMarketStreamClient(cfg)

    waited: list[float] = []
    fake_stop = SimpleNamespace(wait=lambda timeout=None: waited.append(float(timeout)))
    for attempt in (1, 2, 3, 4, 5):
        client._reconnect_backoff_sleep(attempt=attempt, stop_event=fake_stop)

    # 1, 2, 4, 8, then capped at 10 (jitter pinned to 0)
    assert waited == [1.0, 2.0, 4.0, 8.0, 10.0]


# --------------------------------------------------------------------------- D2


def _tick(symbol: str, ts_ms: int, price: float = 100.0, a: int = 1) -> NormalizedTradeTick:
    return NormalizedTradeTick(
        symbol=symbol,
        exchange_ts_ms=ts_ms,
        price=price,
        quantity=1.0,
        event_id=f"mkt:agg:{symbol}:{a}",
        receive_ts_ms=ts_ms + 50,
    )


def test_per_symbol_staleness_detects_dead_symbol():
    agg = RollingWindowAggregator(
        symbol_list=["BTC/USDT", "SOL/USDT", "ETH/USDT"],
        window_seconds=3,
        max_lateness_ms=0,
        stale_symbol_after_ms=5_000,
    )
    base = 1_700_000_000_000
    agg.ingest(_tick("SOL/USDT", base, a=1))  # last real trade at base
    agg.ingest(_tick("BTC/USDT", base + 10_000, a=2))  # watermark advances 10s

    # ETH never traded, SOL is 10s behind the watermark (> 5s threshold).
    assert set(agg.stale_symbols()) == {"SOL/USDT", "ETH/USDT"}
    assert agg.symbol_trade_age_ms("BTC/USDT") == 0
    assert agg.symbol_trade_age_ms("SOL/USDT") == 10_000
    assert agg.last_real_trade_ms("ETH/USDT") is None
    assert agg.symbol_trade_age_ms("ETH/USDT") is None


def test_per_symbol_staleness_disabled_by_default():
    agg = RollingWindowAggregator(symbol_list=["BTC/USDT", "SOL/USDT"], window_seconds=3)
    base = 1_700_000_000_000
    agg.ingest(_tick("BTC/USDT", base + 60_000, a=1))
    # Disabled (threshold 0) -> never reports stale regardless of age.
    assert agg.stale_symbols() == []
    # Explicit threshold still works on demand.
    assert set(agg.stale_symbols(threshold_ms=1_000)) == {"SOL/USDT"}


# --------------------------------------------------------------------------- D3


def test_parse_message_passes_through_listen_key_expired():
    parsed = BinanceUserStreamClient.parse_message(
        {"e": "listenKeyExpired", "E": 1_700_000_123_000, "listenKey": "abc"}
    )
    assert parsed is not None
    assert parsed["event_type"] == "listenKeyExpired"
    assert parsed["exchange_ts_ms"] == 1_700_000_123_000
    assert parsed["listen_key"] == "abc"


def test_parse_message_unknown_still_returns_none():
    assert BinanceUserStreamClient.parse_message({"e": "somethingElse"}) is None


def _make_user_stream_exchange(
    keys: list[str], *, keepalive_ok: bool, stop_after_keys: int, stop_event
):
    counters = {"create": 0, "keepalive": 0, "close": 0}

    def create_listen_key():
        idx = min(counters["create"], len(keys) - 1)
        counters["create"] += 1
        if counters["create"] >= stop_after_keys:
            stop_event.set()
        return keys[idx]

    def keepalive_listen_key(_key):
        counters["keepalive"] += 1
        return keepalive_ok

    def close_listen_key(_key):
        counters["close"] += 1
        return True

    exchange = SimpleNamespace(
        testnet=True,
        create_listen_key=create_listen_key,
        keepalive_listen_key=keepalive_listen_key,
        close_listen_key=close_listen_key,
    )
    return exchange, counters


def test_listen_key_expired_forces_fresh_key_reconnect(monkeypatch):
    client = BinanceUserStreamClient(exchange=None)
    stop_event = client._stop
    exchange, counters = _make_user_stream_exchange(
        ["key-1", "key-2"], keepalive_ok=True, stop_after_keys=2, stop_event=stop_event
    )
    client.exchange = exchange

    expiry_frame = json.dumps({"e": "listenKeyExpired", "E": 1_700_000_999_000})
    _install_fake_connect(monkeypatch, ws_sync, [[expiry_frame], []], stop_event)
    # Neutralize sleeps/clock in the module (safe: only rebinds the module ref).
    monkeypatch.setattr(
        user_stream_mod, "time", SimpleNamespace(monotonic=lambda: 0.0, sleep=lambda _s: None)
    )

    events: list[dict] = []
    client._run(on_event=events.append, on_error=None)

    assert any(evt.get("event_type") == "listenKeyExpired" for evt in events)
    assert counters["create"] >= 2  # a fresh listenKey was created for the reconnect


def test_repeated_keepalive_failure_forces_reconnect(monkeypatch):
    client = BinanceUserStreamClient(exchange=None, keepalive_interval_sec=60.0)
    stop_event = client._stop
    exchange, counters = _make_user_stream_exchange(
        ["key-1", "key-2"], keepalive_ok=False, stop_after_keys=2, stop_event=stop_event
    )
    client.exchange = exchange

    clock = {"t": 0.0}

    def _fake_monotonic():
        return clock["t"]

    fake_time = SimpleNamespace(monotonic=_fake_monotonic, sleep=lambda _s: None)
    monkeypatch.setattr(user_stream_mod, "time", fake_time)

    # Every recv advances the clock past the keepalive interval so keepalive
    # fires (and fails) on each loop turn.
    def _advancing_recv_actions():
        class _Conn(_FakeConn):
            def recv(self, timeout=None):
                clock["t"] += 100.0
                raise TimeoutError()

        return _Conn

    counter = {"connect": 0}

    def _connect(url, **_kw):
        counter["connect"] += 1

        class _AdvConn:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *_a):
                return False

            def recv(self_inner, timeout=None):
                clock["t"] += 100.0
                raise TimeoutError()

        return _AdvConn()

    monkeypatch.setattr(ws_sync, "connect", _connect)

    errors: list[Exception] = []
    client._run(on_event=lambda _e: None, on_error=errors.append)

    assert counters["keepalive"] >= 2  # retried, not silently swallowed
    assert errors  # repeated failure surfaced as an error
    assert counters["create"] >= 2  # forced a fresh-key reconnect


# --------------------------------------------------------------------------- D4


def test_assert_bars_1s_sane_rejects_bad_rows():
    good = {"BTC/USDT": ((1_000, 1.0, 2.0, 0.5, 1.5, 10.0),)}
    assert_bars_1s_sane(good)  # no raise

    for bad in (
        {"BTC/USDT": ((1_000, 1.0, float("nan"), 0.5, 1.5, 10.0),)},  # non-finite
        {"BTC/USDT": ((1_000, 1.0, 2.0, 0.5, -1.5, 10.0),)},  # non-positive close
        {"BTC/USDT": ((1_000, 1.0, 0.4, 0.9, 0.6, 10.0),)},  # high < low
        {"BTC/USDT": ((1_000, 1.0, 2.0, 0.5, 1.5, -1.0),)},  # negative volume
        {"BTC/USDT": ((1_000, 1.0, float("inf"), 0.5, 1.5, 10.0),)},  # inf
    ):
        with pytest.raises(MarketWindowContractError):
            assert_bars_1s_sane(bad)


def test_build_market_window_event_sanity_gate_is_opt_in():
    bad_bars = {"BTC/USDT": ((1_700_000_000_000, 1.0, 0.4, 0.9, 0.6, 10.0),)}  # high < low

    # Default OFF -> byte-identical (does not raise on the bad bar).
    event = build_market_window_event(time=1_700_000_000_000, window_seconds=20, bars_1s=bad_bars)
    assert event.type == "MARKET_WINDOW"

    # Opt-in -> fails closed.
    with pytest.raises(MarketWindowContractError):
        build_market_window_event(
            time=1_700_000_000_000, window_seconds=20, bars_1s=bad_bars, sanity_check=True
        )


def test_rolling_aggregator_sanity_flag_passes_healthy_stream():
    agg = RollingWindowAggregator(
        symbol_list=["BTC/USDT"], window_seconds=2, max_lateness_ms=0, bar_sanity_check=True
    )
    base = 1_700_000_000_000
    events = []
    events.extend(agg.ingest(_tick("BTC/USDT", base, price=100.0, a=1)))
    events.extend(agg.ingest(_tick("BTC/USDT", base + 2_000, price=101.0, a=2)))
    assert events  # healthy bars still emit windows with the gate enabled


# --------------------------------------------------------------------------- D1


class _FakeTradeExchange:
    """Exchange stub returning scripted agg-trade pages keyed by ``since``."""

    def __init__(self, trades_by_ts: dict[int, dict]):
        # trades_by_ts: exchange_ts_ms -> row
        self._rows = [trades_by_ts[k] for k in sorted(trades_by_ts)]
        self.calls: list[int | None] = []

    def fetch_trades(self, symbol, since=None, limit=None):
        self.calls.append(since)
        start = int(since) if since is not None else 0
        end = start + 3_599_999  # mirror the 1h cap
        page = [r for r in self._rows if start <= int(r["timestamp"]) <= end]
        return page[: int(limit or 500)]


def _row(ts: int, price: float = 100.0, a: int = 1) -> dict:
    return {
        "id": a,
        "symbol": "BTC/USDT",
        "timestamp": ts,
        "price": price,
        "amount": 1.0,
        "info": {"a": a},
    }


def _make_handler(exchange, *, symbols=("BTC/USDT",), config=None) -> BinanceLiveDataHandler:
    import queue

    cfg = config or SimpleNamespace()
    handler = object.__new__(BinanceLiveDataHandler)
    handler.events = queue.Queue()
    handler.symbol_list = [str(s) for s in symbols]
    handler.config = cfg
    handler.exchange = exchange
    handler.transport = "poll"
    handler.continue_backtest = True
    handler._shutdown = threading.Event()
    handler.lock = threading.Lock()
    handler.latest_symbol_data = {
        s: __import__("collections").deque(maxlen=500) for s in handler.symbol_list
    }
    handler.latest_book_ticker = {}
    handler.col_idx = {"datetime": 0, "open": 1, "high": 2, "low": 3, "close": 4, "volume": 5}
    handler._cursor_ms = dict.fromkeys(handler.symbol_list)
    handler._recovery_max_pages = 240
    handler._recovery_truncations = 0
    handler.aggregator = RollingWindowAggregator(
        symbol_list=list(handler.symbol_list), window_seconds=2, max_lateness_ms=0
    )
    return handler


def test_paginated_recovery_drains_multiple_pages():
    # 1200 dense trades across 3 pages of <=500 within one hour window.
    base = 1_700_000_000_000
    trades = {base + i * 100: _row(base + i * 100, a=i + 1) for i in range(1200)}
    exchange = _FakeTradeExchange(trades)
    handler = _make_handler(exchange)
    handler._cursor_ms["BTC/USDT"] = base

    handler._drain_symbol_ticks(symbol="BTC/USDT", until_ms=base + 1200 * 100)

    # More than one page fetched, and the cursor advanced past the last trade.
    assert len(exchange.calls) >= 3
    assert int(handler._cursor_ms["BTC/USDT"]) > base + 1199 * 100


def test_recovery_chunks_gap_longer_than_one_hour():
    # A trade at t0, then an empty >1h gap, then a trade ~2.5h later.
    base = 1_700_000_000_000
    later = base + int(2.5 * 3_600_000)
    trades = {base: _row(base, a=1), later: _row(later, a=2)}
    exchange = _FakeTradeExchange(trades)
    handler = _make_handler(exchange)
    handler._cursor_ms["BTC/USDT"] = base

    handler._drain_symbol_ticks(symbol="BTC/USDT", until_ms=later + 1)

    # Must have chunked forward across the empty hours (multiple windowed fetches)
    # and advanced the cursor past the far trade.
    assert len(exchange.calls) >= 3
    assert int(handler._cursor_ms["BTC/USDT"]) > later


def test_recovery_truncation_emits_alert():
    base = 1_700_000_000_000
    # Full pages forever: every fetch returns a full page so paging never catches up.
    dense = {base + i: _row(base + i, a=i + 1) for i in range(5000)}
    exchange = _FakeTradeExchange(dense)
    handler = _make_handler(exchange)
    handler._cursor_ms["BTC/USDT"] = base
    handler._recovery_max_pages = 3

    handler._drain_symbol_ticks(symbol="BTC/USDT", until_ms=base + 10_000_000)

    assert len(exchange.calls) == 3  # bounded by the page budget
    assert handler._recovery_truncations == 1
