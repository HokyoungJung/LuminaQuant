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
        EXCHANGE = {"driver": "binance_futures", "name": "binance", "market_type": "future"}
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
    monkeypatch.setattr(
        module,
        "build_live_runtime_contract",
        lambda **_kwargs: SimpleNamespace(
            engine_cls=_Trader,
            data_handler_cls=object,
            execution_handler_cls=object,
            portfolio_cls=object,
            fatal_error_cls=RuntimeError,
            transport="poll",
        ),
    )


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


def test_committed_market_data_forces_poll_transport_even_when_ws_requested(monkeypatch, capsys):
    captured: dict[str, object] = {}

    class _LiveConfig:
        SYMBOLS = ["BTC/USDT"]
        IS_TESTNET = True
        EXCHANGE = {"driver": "binance_futures", "name": "binance", "market_type": "future"}
        TIMEFRAME = "1m"
        MARKET_DATA_SOURCE = "committed"
        ORDER_STATE_SOURCE = "polling"
        MATERIALIZED_STALENESS_THRESHOLD_SECONDS = 45
        MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS = 60

        @classmethod
        def validate(cls):
            return None

    class _Strategy:
        __name__ = "RsiStrategy"

    class _Trader:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs
            self.data_handler = SimpleNamespace(consume_fatal_error=lambda: None)

        @staticmethod
        def run():
            return None

    monkeypatch.setattr(live_cli, "LiveConfig", _LiveConfig)
    monkeypatch.setattr(
        live_cli,
        "_strategy_helpers",
        lambda: (
            "RsiStrategy",
            lambda include_opt_in=True: {"RsiStrategy": _Strategy},
            lambda *_args, **_kwargs: _Strategy,
        ),
    )

    def _capture_contract(*, transport="poll"):
        captured["transport"] = transport
        return SimpleNamespace(
            engine_cls=_Trader,
            data_handler_cls=object,
            execution_handler_cls=object,
            portfolio_cls=object,
            fatal_error_cls=RuntimeError,
            transport=transport,
        )

    monkeypatch.setattr(live_cli, "build_live_runtime_contract", _capture_contract)

    assert live_cli.main(["--transport", "ws", "--no-selection"]) == 0
    assert captured["transport"] == "poll"
    captured_output = capsys.readouterr().out
    assert "Requested Transport: ws" in captured_output
    assert "Effective Transport: poll" in captured_output


def test_selection_overrides_are_applied_before_live_config_validation(monkeypatch):
    observed: dict[str, object] = {}
    validate_calls: list[tuple[list[str], str]] = []

    class _LiveConfig:
        SYMBOLS = ["BTC/USDT"]
        IS_TESTNET = True
        EXCHANGE = {"driver": "binance_futures", "name": "binance", "market_type": "future"}
        TIMEFRAME = "1m"
        MARKET_DATA_SOURCE = "committed"
        ORDER_STATE_SOURCE = "polling"
        MATERIALIZED_STALENESS_THRESHOLD_SECONDS = 45
        MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS = 60

        @classmethod
        def validate(cls):
            validate_calls.append((list(cls.SYMBOLS), str(cls.TIMEFRAME)))
            return None

    class _Strategy:
        __name__ = "MovingAverageCrossStrategy"

    class _Trader:
        def __init__(self, *args, **kwargs):
            _ = args
            observed["kwargs"] = kwargs
            self.data_handler = SimpleNamespace(consume_fatal_error=lambda: None)

        @staticmethod
        def run():
            return None

    monkeypatch.setattr(live_cli, "LiveConfig", _LiveConfig)
    monkeypatch.setattr(
        live_cli,
        "_strategy_helpers",
        lambda: (
            "MovingAverageCrossStrategy",
            lambda include_opt_in=True: {"MovingAverageCrossStrategy": _Strategy},
            lambda *_args, **_kwargs: _Strategy,
        ),
    )
    monkeypatch.setattr(
        live_cli,
        "build_live_runtime_contract",
        lambda **_kwargs: SimpleNamespace(
            engine_cls=_Trader,
            data_handler_cls=object,
            execution_handler_cls=object,
            portfolio_cls=object,
            fatal_error_cls=RuntimeError,
            transport="poll",
        ),
    )
    monkeypatch.setattr(live_cli, "resolve_live_decision_file", lambda _path="": None)
    monkeypatch.setattr(live_cli, "resolve_selection_file", lambda _path="": "fake-selection.json")
    monkeypatch.setattr(live_cli, "load_selection_payload", lambda _path: {"ok": True})
    monkeypatch.setattr(
        live_cli,
        "extract_selection_config",
        lambda _payload: {
            "candidate_name": "MovingAverageCrossStrategy",
            "symbols": ["ETH/USDT", "SOL/USDT"],
            "strategy_timeframe": "5m",
            "params": {"fast": 3},
        },
    )

    assert live_cli.main([]) == 0
    assert validate_calls == [(["ETH/USDT", "SOL/USDT"], "5m")]
    assert observed["kwargs"]["symbol_list"] == ["ETH/USDT", "SOL/USDT"]
    assert observed["kwargs"]["strategy_params"] == {"fast": 3}


def test_live_cli_fails_closed_when_decision_points_to_unsupported_portfolio_mode(
    monkeypatch, capsys
):
    class _LiveConfig:
        SYMBOLS = ["BTC/USDT"]
        IS_TESTNET = True
        EXCHANGE = {"driver": "binance_futures", "name": "binance", "market_type": "future"}
        TIMEFRAME = "1m"
        MARKET_DATA_SOURCE = "committed"
        ORDER_STATE_SOURCE = "polling"
        MATERIALIZED_STALENESS_THRESHOLD_SECONDS = 45
        MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS = 60

        @classmethod
        def validate(cls):
            return None

    class _Strategy:
        __name__ = "MovingAverageCrossStrategy"

    monkeypatch.setattr(live_cli, "LiveConfig", _LiveConfig)
    monkeypatch.setattr(
        live_cli,
        "_strategy_helpers",
        lambda: (
            "MovingAverageCrossStrategy",
            lambda include_opt_in=True: {"MovingAverageCrossStrategy": _Strategy},
            lambda *_args, **_kwargs: _Strategy,
        ),
    )
    monkeypatch.setattr(live_cli, "resolve_live_decision_file", lambda _path="": "decision.json")
    monkeypatch.setattr(
        live_cli, "load_live_decision_payload", lambda _path: {"decision": "selected_live_mode"}
    )
    monkeypatch.setattr(
        live_cli,
        "extract_live_decision_config",
        lambda _payload: {
            "decision": "selected_live_mode",
            "reference": "unsupported_portfolio_mode",
            "target_kind": "portfolio_mode",
            "strategy_name": None,
        },
    )
    monkeypatch.setattr(
        live_cli,
        "build_live_runtime_contract",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("runtime should not build")),
    )

    assert live_cli.main([]) == 1
    captured = capsys.readouterr().out
    assert "portfolio mode" in captured
    assert "unsupported_portfolio_mode" in captured


def test_live_cli_uses_supported_portfolio_mode_strategy(monkeypatch):
    observed: dict[str, object] = {}

    class _LiveConfig:
        SYMBOLS = ["BTC/USDT"]
        IS_TESTNET = True
        EXCHANGE = {"driver": "binance_futures", "name": "binance", "market_type": "future"}
        TIMEFRAME = "1m"
        MARKET_DATA_SOURCE = "committed"
        ORDER_STATE_SOURCE = "polling"
        MATERIALIZED_STALENESS_THRESHOLD_SECONDS = 45
        MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS = 60

        @classmethod
        def validate(cls):
            return None

    class _FallbackStrategy:
        __name__ = "MovingAverageCrossStrategy"

    class _PortfolioModeStrategy:
        __name__ = "ArtifactPortfolioModeStrategy"

    class _Trader:
        def __init__(self, *args, **kwargs):
            _ = args
            observed["kwargs"] = kwargs
            self.data_handler = SimpleNamespace(consume_fatal_error=lambda: None)

        @staticmethod
        def run():
            return None

    monkeypatch.setattr(live_cli, "LiveConfig", _LiveConfig)
    monkeypatch.setattr(
        live_cli,
        "_strategy_helpers",
        lambda: (
            "MovingAverageCrossStrategy",
            lambda include_opt_in=True: {"MovingAverageCrossStrategy": _FallbackStrategy},
            lambda *_args, **_kwargs: _FallbackStrategy,
        ),
    )
    monkeypatch.setattr(live_cli, "resolve_live_decision_file", lambda _path="": "decision.json")
    monkeypatch.setattr(
        live_cli, "load_live_decision_payload", lambda _path: {"decision": "selected_live_mode"}
    )
    monkeypatch.setattr(
        live_cli,
        "extract_live_decision_config",
        lambda _payload: {
            "decision": "selected_live_mode",
            "reference": "hybrid_guarded_mode",
            "target_kind": "portfolio_mode",
            "strategy_name": None,
        },
    )
    monkeypatch.setattr(
        live_cli,
        "resolve_portfolio_mode_definition",
        lambda _reference: SimpleNamespace(symbols=["BNB/USDT", "TRX/USDT", "BTC/USDT"]),
    )
    monkeypatch.setattr(live_cli, "ArtifactPortfolioModeStrategy", _PortfolioModeStrategy)
    monkeypatch.setattr(
        live_cli,
        "build_live_runtime_contract",
        lambda **_kwargs: SimpleNamespace(
            engine_cls=_Trader,
            data_handler_cls=object,
            execution_handler_cls=object,
            portfolio_cls=object,
            fatal_error_cls=RuntimeError,
            transport="poll",
        ),
    )

    assert live_cli.main([]) == 0
    assert (
        observed["kwargs"]["strategy_name"] == "ArtifactPortfolioModeStrategy[hybrid_guarded_mode]"
    )
    assert observed["kwargs"]["strategy_params"] == {"portfolio_mode": "hybrid_guarded_mode"}
    assert observed["kwargs"]["symbol_list"] == ["BNB/USDT", "TRX/USDT", "BTC/USDT"]


def test_live_cli_uses_decision_strategy_target_without_falling_back_to_stale_selection(
    monkeypatch,
):
    observed: dict[str, object] = {}

    class _LiveConfig:
        SYMBOLS = ["BTC/USDT"]
        IS_TESTNET = True
        EXCHANGE = {"driver": "binance_futures", "name": "binance", "market_type": "future"}
        TIMEFRAME = "1m"
        MARKET_DATA_SOURCE = "committed"
        ORDER_STATE_SOURCE = "polling"
        MATERIALIZED_STALENESS_THRESHOLD_SECONDS = 45
        MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS = 60

        @classmethod
        def validate(cls):
            return None

    class _Moving:
        __name__ = "MovingAverageCrossStrategy"

    class _Rsi:
        __name__ = "RsiStrategy"

    class _Trader:
        def __init__(self, *args, **kwargs):
            _ = args
            observed["kwargs"] = kwargs
            self.data_handler = SimpleNamespace(consume_fatal_error=lambda: None)

        @staticmethod
        def run():
            return None

    monkeypatch.setattr(live_cli, "LiveConfig", _LiveConfig)
    monkeypatch.setattr(
        live_cli,
        "_strategy_helpers",
        lambda: (
            "MovingAverageCrossStrategy",
            lambda include_opt_in=True: {
                "MovingAverageCrossStrategy": _Moving,
                "RsiStrategy": _Rsi,
            },
            lambda name, **_kwargs: {"MovingAverageCrossStrategy": _Moving, "RsiStrategy": _Rsi}[
                name
            ],
        ),
    )
    monkeypatch.setattr(live_cli, "resolve_live_decision_file", lambda _path="": "decision.json")
    monkeypatch.setattr(
        live_cli, "load_live_decision_payload", lambda _path: {"decision": "selected_live_mode"}
    )
    monkeypatch.setattr(
        live_cli,
        "extract_live_decision_config",
        lambda _payload: {
            "decision": "selected_live_mode",
            "reference": "RsiStrategy",
            "target_kind": "strategy_class",
            "strategy_name": "RsiStrategy",
        },
    )
    monkeypatch.setattr(
        live_cli,
        "resolve_selection_file",
        lambda _path="": (_ for _ in ()).throw(
            AssertionError("stale selection should not be consulted")
        ),
    )
    monkeypatch.setattr(
        live_cli,
        "build_live_runtime_contract",
        lambda **_kwargs: SimpleNamespace(
            engine_cls=_Trader,
            data_handler_cls=object,
            execution_handler_cls=object,
            portfolio_cls=object,
            fatal_error_cls=RuntimeError,
            transport="poll",
        ),
    )

    assert live_cli.main([]) == 0
    assert observed["kwargs"]["strategy_name"] == "RsiStrategy"


def test_live_cli_fails_closed_when_decision_strategy_is_not_registered(monkeypatch, capsys):
    class _LiveConfig:
        SYMBOLS = ["BTC/USDT"]
        IS_TESTNET = True
        EXCHANGE = {"driver": "binance_futures", "name": "binance", "market_type": "future"}
        TIMEFRAME = "1m"
        MARKET_DATA_SOURCE = "committed"
        ORDER_STATE_SOURCE = "polling"
        MATERIALIZED_STALENESS_THRESHOLD_SECONDS = 45
        MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS = 60

        @classmethod
        def validate(cls):
            raise AssertionError("validate should not run for unsupported decision strategy")

    class _Moving:
        __name__ = "MovingAverageCrossStrategy"

    monkeypatch.setattr(live_cli, "LiveConfig", _LiveConfig)
    monkeypatch.setattr(
        live_cli,
        "_strategy_helpers",
        lambda: (
            "MovingAverageCrossStrategy",
            lambda include_opt_in=True: {"MovingAverageCrossStrategy": _Moving},
            lambda name, **_kwargs: {"MovingAverageCrossStrategy": _Moving}[name],
        ),
    )
    monkeypatch.setattr(live_cli, "resolve_live_decision_file", lambda _path="": "decision.json")
    monkeypatch.setattr(
        live_cli, "load_live_decision_payload", lambda _path: {"decision": "selected_live_mode"}
    )
    monkeypatch.setattr(
        live_cli,
        "extract_live_decision_config",
        lambda _payload: {
            "decision": "selected_live_mode",
            "reference": "GhostStrategy",
            "target_kind": "strategy_class",
            "strategy_name": "GhostStrategy",
        },
    )
    monkeypatch.setattr(
        live_cli,
        "resolve_selection_file",
        lambda _path="": (_ for _ in ()).throw(
            AssertionError("stale selection should not be consulted")
        ),
    )
    monkeypatch.setattr(
        live_cli,
        "build_live_runtime_contract",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("runtime should not build")),
    )

    assert live_cli.main([]) == 1
    captured = capsys.readouterr().out
    assert "unsupported strategy class" in captured
    assert "GhostStrategy" in captured


def test_live_cli_applies_alpha_zoo_decision_params_and_6x_runtime_overrides(monkeypatch):
    observed: dict[str, object] = {}
    validate_snapshot: dict[str, object] = {}

    class _LiveConfig:
        SYMBOLS = ["BTC/USDT"]
        IS_TESTNET = True
        EXCHANGE = {
            "driver": "binance_futures",
            "name": "binance",
            "market_type": "future",
            "position_mode": "HEDGE",
            "margin_mode": "isolated",
            "leverage": 3,
        }
        EXCHANGE_ID = "paper"
        MARKET_TYPE = "spot"
        POSITION_MODE = "ONEWAY"
        MARGIN_MODE = "cross"
        LEVERAGE = 3
        TARGET_ALLOCATION = 0.05
        TARGET_ALLOCATION_MODE = "legacy_notional_cap"
        MAX_ORDER_VALUE = 5000.0
        MAX_ORDER_NOTIONAL_PCT = 0.0
        MAX_SYMBOL_EXPOSURE_PCT = 0.25
        MAX_TOTAL_MARGIN_PCT = 0.5
        MAX_TOTAL_NOTIONAL_PCT = 0.0
        TIMEFRAME = "1m"
        WINDOW_SECONDS = 20
        INGEST_WINDOW_SECONDS = 20
        DECISION_CADENCE_SECONDS = 20
        MARKET_DATA_SOURCE = "committed"
        ORDER_STATE_SOURCE = "polling"
        MATERIALIZED_STALENESS_THRESHOLD_SECONDS = 45
        MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS = 60

        @classmethod
        def validate(cls):
            validate_snapshot["symbols"] = list(cls.SYMBOLS)
            validate_snapshot["timeframe"] = str(cls.TIMEFRAME)
            validate_snapshot["exchange"] = dict(cls.EXCHANGE)
            validate_snapshot["exchange_id"] = str(cls.EXCHANGE_ID)
            validate_snapshot["market_type"] = str(cls.MARKET_TYPE)
            validate_snapshot["position_mode"] = str(cls.POSITION_MODE)
            validate_snapshot["margin_mode"] = str(cls.MARGIN_MODE)
            validate_snapshot["leverage"] = int(cls.LEVERAGE)
            validate_snapshot["target_allocation"] = float(cls.TARGET_ALLOCATION)
            validate_snapshot["target_allocation_mode"] = str(cls.TARGET_ALLOCATION_MODE)
            validate_snapshot["max_order_value"] = float(cls.MAX_ORDER_VALUE)
            validate_snapshot["max_order_notional_pct"] = float(cls.MAX_ORDER_NOTIONAL_PCT)
            validate_snapshot["max_symbol_exposure_pct"] = float(cls.MAX_SYMBOL_EXPOSURE_PCT)
            validate_snapshot["max_total_notional_pct"] = float(cls.MAX_TOTAL_NOTIONAL_PCT)
            validate_snapshot["window_seconds"] = int(cls.WINDOW_SECONDS)
            validate_snapshot["ingest_window_seconds"] = int(cls.INGEST_WINDOW_SECONDS)
            validate_snapshot["decision_cadence_seconds"] = int(cls.DECISION_CADENCE_SECONDS)
            return None

    class _Fallback:
        __name__ = "MovingAverageCrossStrategy"

    class _AlphaZoo:
        __name__ = "CryptoFxAlphaZooStateStrategy"

    class _Trader:
        def __init__(self, *args, **kwargs):
            _ = args
            observed["kwargs"] = kwargs
            self.data_handler = SimpleNamespace(consume_fatal_error=lambda: None)

        @staticmethod
        def run():
            return None

    monkeypatch.setattr(live_cli, "LiveConfig", _LiveConfig)
    monkeypatch.setattr(
        live_cli,
        "_strategy_helpers",
        lambda: (
            "MovingAverageCrossStrategy",
            lambda include_opt_in=True: {
                "MovingAverageCrossStrategy": _Fallback,
                "CryptoFxAlphaZooStateStrategy": _AlphaZoo,
            },
            lambda name, **_kwargs: {
                "MovingAverageCrossStrategy": _Fallback,
                "CryptoFxAlphaZooStateStrategy": _AlphaZoo,
            }[name],
        ),
    )
    monkeypatch.setattr(live_cli, "resolve_live_decision_file", lambda _path="": "decision.json")
    monkeypatch.setattr(
        live_cli,
        "load_live_decision_payload",
        lambda _path: {
            "decision": "selected_live_mode",
            "selected_mode": "crypto_fx_alpha_zoo_state_calibrated",
            "strategy_name": "CryptoFxAlphaZooStateStrategy",
            "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "TRX/USDT"],
            "strategy_timeframe": "1h",
            "strategy_params": {
                "entry_threshold": 0.95,
                "calibrated_edges": {"default:LONG": 5.0, "default:SHORT": 5.0},
                "decision_cadence_seconds": 3600,
            },
            "exchange": {
                "driver": "binance_futures",
                "name": "binance",
                "market_type": "future",
                "position_mode": "HEDGE",
                "margin_mode": "isolated",
                "leverage": 6,
            },
            "sizing_mode": "isolated_margin_fraction",
            "target_allocation": 0.10,
            "risk_caps": {
                "max_order_value": 0.0,
                "max_order_notional_pct": 0.75,
                "max_symbol_exposure_pct": 0.75,
                "max_total_notional_pct": 1.5,
            },
            "window_seconds": 3600,
            "ingest_window_seconds": 3600,
            "decision_cadence_seconds": 3600,
        },
    )
    monkeypatch.setattr(
        live_cli,
        "resolve_selection_file",
        lambda _path="": (_ for _ in ()).throw(
            AssertionError("stale selection should not be consulted")
        ),
    )
    monkeypatch.setattr(
        live_cli,
        "build_live_runtime_contract",
        lambda **_kwargs: SimpleNamespace(
            engine_cls=_Trader,
            data_handler_cls=object,
            execution_handler_cls=object,
            portfolio_cls=object,
            fatal_error_cls=RuntimeError,
            transport="poll",
        ),
    )

    assert live_cli.main([]) == 0
    assert observed["kwargs"]["strategy_name"] == "CryptoFxAlphaZooStateStrategy"
    assert observed["kwargs"]["symbol_list"] == [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "BNB/USDT",
        "TRX/USDT",
    ]
    assert observed["kwargs"]["strategy_params"]["entry_threshold"] == 0.95
    assert observed["kwargs"]["strategy_params"]["calibrated_edges"] == {
        "default:LONG": 5.0,
        "default:SHORT": 5.0,
    }
    assert validate_snapshot["timeframe"] == "1h"
    assert validate_snapshot["exchange"]["leverage"] == 6
    assert validate_snapshot["exchange_id"] == "binance"
    assert validate_snapshot["market_type"] == "future"
    assert validate_snapshot["position_mode"] == "HEDGE"
    assert validate_snapshot["margin_mode"] == "isolated"
    assert validate_snapshot["leverage"] == 6
    assert validate_snapshot["target_allocation"] == 0.10
    assert validate_snapshot["target_allocation_mode"] == "isolated_margin_fraction"
    assert validate_snapshot["max_order_value"] == 0.0
    assert validate_snapshot["max_order_notional_pct"] == 0.75
    assert validate_snapshot["max_symbol_exposure_pct"] == 0.75
    assert validate_snapshot["max_total_notional_pct"] == 1.5
    assert validate_snapshot["window_seconds"] == 3600
    assert validate_snapshot["ingest_window_seconds"] == 3600
    assert validate_snapshot["decision_cadence_seconds"] == 3600


def test_strategy_helper_resolver_accepts_default_name_keyword():
    _, _, resolver = live_cli._strategy_helpers()

    strategy_cls = resolver(
        "MovingAverageCrossStrategy",
        default_name="MovingAverageCrossStrategy",
    )

    assert strategy_cls.__name__ == "MovingAverageCrossStrategy"
