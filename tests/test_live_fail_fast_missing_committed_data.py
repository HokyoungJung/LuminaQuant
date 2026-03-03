from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest
from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError
from lumina_quant.cli import live as live_cli
from lumina_quant.live.data_poll import LiveDataHandler


class _Config:
    MARKET_DATA_PARQUET_PATH = "data/market_parquet"
    MARKET_DATA_EXCHANGE = "binance"
    LIVE_POLL_SECONDS = 1
    INGEST_WINDOW_SECONDS = 5
    MATERIALIZED_STALENESS_THRESHOLD_SECONDS = 45
    MARKET_WINDOW_PARITY_V2_ENABLED = False
    MARKET_WINDOW_METRICS_LOG_PATH = "logs/live/market_window_metrics.ndjson"


class _ThreadStub:
    def __init__(self, target, daemon=True):
        self.target = target
        self.daemon = daemon

    def start(self):
        return None

    def is_alive(self):
        return False

    def join(self, timeout=None):
        _ = timeout
        return None


def test_data_handler_fail_fast_propagates_missing_committed_data(monkeypatch):
    class _ReaderStub:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

        @staticmethod
        def read_snapshot():
            raise RawFirstDataMissingError("committed manifest missing for BTC/USDT:1s")

    events = queue.Queue()
    monkeypatch.setattr("lumina_quant.live.data_materialized.threading.Thread", _ThreadStub)
    monkeypatch.setattr("lumina_quant.live.data_materialized.MaterializedWindowReader", _ReaderStub)
    handler = LiveDataHandler(events, ["BTC/USDT"], _Config, exchange=SimpleNamespace())
    handler._poll_market_data()

    fatal = handler.consume_fatal_error()
    assert isinstance(fatal, RawFirstDataMissingError)
    assert events.qsize() == 0


def _patch_entrypoint_env(monkeypatch, module, *, strategy_name: str):
    class _LiveConfig:
        SYMBOLS = ["BTC/USDT"]
        IS_TESTNET = True
        EXCHANGE = {"driver": "ccxt", "name": "binance", "market_type": "future"}
        TIMEFRAME = "1m"
        MATERIALIZED_STALENESS_THRESHOLD_SECONDS = 45
        MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS = 60

        @classmethod
        def validate(cls):
            return None

    class _Strategy:
        __name__ = strategy_name

    class _Trader:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs
            self.data_handler = SimpleNamespace(consume_fatal_error=lambda: None)

        @staticmethod
        def _ordered_shutdown():
            return None

        @staticmethod
        def _close_audit_store(status=None):
            _ = status
            return None

        @staticmethod
        def run():
            raise RawFirstDataMissingError("fatal committed data breach")

    monkeypatch.setattr(module, "LiveConfig", _LiveConfig)
    monkeypatch.setattr(module, "STRATEGY_MAP", {strategy_name: _Strategy})
    monkeypatch.setattr(module, "resolve_strategy_class", lambda *_args, **_kwargs: _Strategy)
    monkeypatch.setattr(module, "LiveTrader", _Trader)


def test_run_live_exits_with_code_2_on_fail_fast(monkeypatch):
    _patch_entrypoint_env(monkeypatch, live_cli, strategy_name="MovingAverageCrossStrategy")
    with pytest.raises(SystemExit) as exc:
        live_cli.main(["--no-selection"])
    assert int(exc.value.code) == 2


def test_run_live_ws_exits_with_code_2_on_fail_fast(monkeypatch):
    _patch_entrypoint_env(monkeypatch, live_cli, strategy_name="RsiStrategy")
    with pytest.raises(SystemExit) as exc:
        live_cli.main(["--transport", "ws", "--no-selection"])
    assert int(exc.value.code) == 2
