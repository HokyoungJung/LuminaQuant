from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from lumina_quant.backtesting._config_view import BacktestConfigView
from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.configuration.schema import RuntimeConfig

SYMBOL = "BTC/USDT"
START = datetime(2026, 1, 1)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        TIMEFRAME="1m",
        DECISION_CADENCE_SECONDS=20,
        SKIP_AHEAD_ENABLED=True,
    )


def _make_strategy_class():
    class _Strategy:
        def __init__(self, bars, events, **kwargs):
            self.bars = bars
            self.events = events
            self.kwargs = dict(kwargs)
            self.decision_cadence_seconds = 0

    return _Strategy


def _make_strategy_rejecting_params_class():
    class _Strategy:
        def __init__(self, bars, events, **kwargs):
            if kwargs:
                raise TypeError("strategy params rejected")
            self.bars = bars
            self.events = events
            self.kwargs = dict(kwargs)
            self.decision_cadence_seconds = 0

    return _Strategy


def _make_data_handler_class(*, raise_on_kwargs: bool = False):
    calls: list[dict[str, object]] = []

    class _DataHandler:
        def __init__(self, events, csv_dir, symbol_list, start_date, end_date, data_dict, **kwargs):
            calls.append(dict(kwargs))
            if raise_on_kwargs and kwargs:
                raise TypeError("legacy data-handler kwargs rejected")
            self.events = events
            self.csv_dir = csv_dir
            self.symbol_list = symbol_list
            self.start_date = start_date
            self.end_date = end_date
            self.data_dict = data_dict
            self.kwargs = dict(kwargs)
            self.symbol_timestamps_ms = {}
            self._feature_lookup = None

    return _DataHandler, calls


def _make_portfolio_fallback_class():
    calls: list[dict[str, object]] = []

    class _Portfolio:
        def __init__(self, bars, events, start_date, config, **kwargs):
            calls.append(dict(kwargs))
            if kwargs:
                raise TypeError("compatibility fallback required")
            self.bars = bars
            self.events = events
            self.start_date = start_date
            self.config = config
            self.kwargs = dict(kwargs)
            self.trades = []
            self.all_holdings = []
            self.all_positions = []
            self.trade_count = 0
            self.current_holdings = {"total": 0}

    return _Portfolio, calls


def _make_portfolio_strict_class():
    calls: list[dict[str, object]] = []

    class _Portfolio:
        def __init__(self, bars, events, start_date, config, **kwargs):
            calls.append(dict(kwargs))
            self.bars = bars
            self.events = events
            self.start_date = start_date
            self.config = config
            self.kwargs = dict(kwargs)
            self.trades = []
            self.all_holdings = []
            self.all_positions = []
            self.trade_count = 0
            self.current_holdings = {"total": 0}

    return _Portfolio, calls


def _make_portfolio_rejecting_kwargs_class():
    calls: list[dict[str, object]] = []

    class _Portfolio:
        def __init__(self, bars, events, start_date, config, **kwargs):
            calls.append(dict(kwargs))
            if kwargs:
                raise TypeError("strict portfolio kwargs rejected")
            self.bars = bars
            self.events = events
            self.start_date = start_date
            self.config = config
            self.kwargs = dict(kwargs)
            self.trades = []
            self.all_holdings = []
            self.all_positions = []
            self.trade_count = 0
            self.current_holdings = {"total": 0}

    return _Portfolio, calls


def _make_execution_handler_class():
    calls: list[dict[str, object]] = []

    class _ExecutionHandler:
        def __init__(self, events, bars, config, **kwargs):
            calls.append(dict(kwargs))
            self.events = events
            self.bars = bars
            self.config = config
            self.kwargs = dict(kwargs)

    return _ExecutionHandler, calls


def _make_execution_handler_rejecting_kwargs_class():
    calls: list[dict[str, object]] = []

    class _ExecutionHandler:
        def __init__(self, events, bars, config, **kwargs):
            calls.append(dict(kwargs))
            if kwargs:
                raise TypeError("strict execution kwargs rejected")
            self.events = events
            self.bars = bars
            self.config = config
            self.kwargs = dict(kwargs)

    return _ExecutionHandler, calls


def _build_backtest(**kwargs) -> Backtest:
    data = {SYMBOL: []}
    return Backtest(
        csv_dir="data",
        symbol_list=[SYMBOL],
        start_date=START,
        end_date=None,
        data_handler_cls=kwargs.pop("data_handler_cls"),
        execution_handler_cls=kwargs.pop("execution_handler_cls"),
        portfolio_cls=kwargs.pop("portfolio_cls"),
        strategy_cls=kwargs.pop("strategy_cls"),
        data_dict=data,
        config=kwargs.pop("config", _config()),
        **kwargs,
    )


def test_default_empty_kwargs_preserve_legacy_portfolio_and_execution_paths():
    data_handler_cls, data_calls = _make_data_handler_class()
    portfolio_cls, portfolio_calls = _make_portfolio_fallback_class()
    execution_handler_cls, execution_calls = _make_execution_handler_class()

    backtest = _build_backtest(
        data_handler_cls=data_handler_cls,
        execution_handler_cls=execution_handler_cls,
        portfolio_cls=portfolio_cls,
        strategy_cls=_make_strategy_class(),
        data_handler_kwargs={},
    )

    assert backtest.data_handler.kwargs == {}
    assert data_calls == [{}]
    assert len(portfolio_calls) == 3
    assert portfolio_calls[0] == {
        "record_history": True,
        "track_metrics": True,
        "record_trades": True,
        "sampling_timeframe": "1m",
    }
    assert portfolio_calls[1] == {
        "record_history": True,
        "record_trades": True,
    }
    assert portfolio_calls[2] == {}
    assert execution_calls == [{}]
    assert backtest.execution_handler.kwargs == {}


def test_explicit_none_kwargs_preserve_legacy_portfolio_and_execution_paths():
    data_handler_cls, data_calls = _make_data_handler_class()
    portfolio_cls, portfolio_calls = _make_portfolio_fallback_class()
    execution_handler_cls, execution_calls = _make_execution_handler_class()

    backtest = _build_backtest(
        data_handler_cls=data_handler_cls,
        execution_handler_cls=execution_handler_cls,
        portfolio_cls=portfolio_cls,
        strategy_cls=_make_strategy_class(),
        portfolio_kwargs=None,
        execution_handler_kwargs=None,
    )

    assert backtest.portfolio_kwargs == {}
    assert backtest.execution_handler_kwargs == {}
    assert data_calls == [{}]
    assert len(portfolio_calls) == 3
    assert portfolio_calls[-1] == {}
    assert execution_calls == [{}]
    assert backtest.portfolio.kwargs == {}
    assert backtest.execution_handler.kwargs == {}


def test_legacy_nonempty_data_handler_kwargs_falls_back_when_not_strict():
    data_handler_cls, data_calls = _make_data_handler_class(raise_on_kwargs=True)
    portfolio_cls, _ = _make_portfolio_strict_class()
    execution_handler_cls, _ = _make_execution_handler_class()

    _build_backtest(
        data_handler_cls=data_handler_cls,
        execution_handler_cls=execution_handler_cls,
        portfolio_cls=portfolio_cls,
        strategy_cls=_make_strategy_class(),
        data_handler_kwargs={"legacy_flag": True},
    )

    assert data_calls == [{"legacy_flag": True}, {}]


def test_strict_data_handler_construction_requires_nonempty_kwargs():
    data_handler_cls, data_calls = _make_data_handler_class()
    portfolio_cls, _ = _make_portfolio_strict_class()
    execution_handler_cls, _ = _make_execution_handler_class()

    with pytest.raises(ValueError, match="strict_data_handler_construction requires nonempty"):
        _build_backtest(
            data_handler_cls=data_handler_cls,
            execution_handler_cls=execution_handler_cls,
            portfolio_cls=portfolio_cls,
            strategy_cls=_make_strategy_class(),
            strict_data_handler_construction=True,
            data_handler_kwargs={},
        )

    assert data_calls == []


def test_strict_data_handler_construction_makes_one_call_and_fails_loud():
    data_handler_cls, data_calls = _make_data_handler_class(raise_on_kwargs=True)
    portfolio_cls, _ = _make_portfolio_strict_class()
    execution_handler_cls, _ = _make_execution_handler_class()
    ordered_lookup = object()

    with pytest.raises(TypeError, match="legacy data-handler kwargs rejected"):
        _build_backtest(
            data_handler_cls=data_handler_cls,
            execution_handler_cls=execution_handler_cls,
            portfolio_cls=portfolio_cls,
            strategy_cls=_make_strategy_class(),
            strict_data_handler_construction=True,
            data_handler_kwargs={"ordered_lookup": ordered_lookup},
        )

    assert data_calls == [{"ordered_lookup": ordered_lookup}]


def test_nonempty_portfolio_and_execution_kwargs_use_single_strict_calls():
    data_handler_cls, data_calls = _make_data_handler_class()
    portfolio_cls, portfolio_calls = _make_portfolio_strict_class()
    execution_handler_cls, execution_calls = _make_execution_handler_class()
    sink = object()
    resolver = object()

    backtest = _build_backtest(
        data_handler_cls=data_handler_cls,
        execution_handler_cls=execution_handler_cls,
        portfolio_cls=portfolio_cls,
        strategy_cls=_make_strategy_class(),
        portfolio_kwargs={
            "fill_application_attribution_sink": sink,
            "funding_boundary_resolver": resolver,
        },
        execution_handler_kwargs={"record_cost_attribution": True},
    )

    assert data_calls == [{}]
    assert len(portfolio_calls) == 1
    assert portfolio_calls[0]["fill_application_attribution_sink"] is sink
    assert portfolio_calls[0]["funding_boundary_resolver"] is resolver
    assert portfolio_calls[0]["record_history"] is True
    assert portfolio_calls[0]["track_metrics"] is True
    assert portfolio_calls[0]["record_trades"] is True
    assert portfolio_calls[0]["sampling_timeframe"] == "1m"
    assert len(execution_calls) == 1
    assert execution_calls[0] == {"record_cost_attribution": True}
    assert backtest.portfolio.kwargs["fill_application_attribution_sink"] is sink
    assert backtest.portfolio.kwargs["funding_boundary_resolver"] is resolver
    assert backtest.execution_handler.kwargs == {"record_cost_attribution": True}


def test_nonempty_portfolio_kwargs_fail_loud_without_legacy_retry():
    data_handler_cls, _ = _make_data_handler_class()
    portfolio_cls, portfolio_calls = _make_portfolio_rejecting_kwargs_class()
    execution_handler_cls, execution_calls = _make_execution_handler_class()
    sink = object()

    with pytest.raises(TypeError, match="strict portfolio kwargs rejected"):
        _build_backtest(
            data_handler_cls=data_handler_cls,
            execution_handler_cls=execution_handler_cls,
            portfolio_cls=portfolio_cls,
            strategy_cls=_make_strategy_class(),
            portfolio_kwargs={"fill_application_attribution_sink": sink},
        )

    assert len(portfolio_calls) == 1
    assert portfolio_calls[0]["fill_application_attribution_sink"] is sink
    assert execution_calls == []


def test_nonempty_execution_kwargs_fail_loud_without_legacy_retry():
    data_handler_cls, _ = _make_data_handler_class()
    portfolio_cls, portfolio_calls = _make_portfolio_strict_class()
    execution_handler_cls, execution_calls = _make_execution_handler_rejecting_kwargs_class()

    with pytest.raises(TypeError, match="strict execution kwargs rejected"):
        _build_backtest(
            data_handler_cls=data_handler_cls,
            execution_handler_cls=execution_handler_cls,
            portfolio_cls=portfolio_cls,
            strategy_cls=_make_strategy_class(),
            execution_handler_kwargs={"record_cost_attribution": True},
        )

    assert len(portfolio_calls) == 1
    assert execution_calls == [{"record_cost_attribution": True}]


def test_nonempty_strategy_params_fail_loud_without_legacy_retry():
    data_handler_cls, _ = _make_data_handler_class()
    portfolio_cls, portfolio_calls = _make_portfolio_strict_class()
    execution_handler_cls, execution_calls = _make_execution_handler_class()

    with pytest.raises(TypeError, match="strategy params rejected"):
        _build_backtest(
            data_handler_cls=data_handler_cls,
            execution_handler_cls=execution_handler_cls,
            portfolio_cls=portfolio_cls,
            strategy_cls=_make_strategy_rejecting_params_class(),
            strategy_params={"portfolio_mode": "alpha_max", "decision_cadence_seconds": 1},
        )

    assert portfolio_calls == []
    assert execution_calls == []


def test_empty_strategy_params_fail_loud_without_retry():
    data_handler_cls, _ = _make_data_handler_class()
    portfolio_cls, portfolio_calls = _make_portfolio_strict_class()
    execution_handler_cls, execution_calls = _make_execution_handler_class()
    strategy_calls = []

    class _AlwaysRejectingStrategy:
        def __init__(self, bars, events, **kwargs):
            strategy_calls.append(dict(kwargs))
            raise TypeError("strategy rejected")

    with pytest.raises(TypeError, match="strategy rejected"):
        _build_backtest(
            data_handler_cls=data_handler_cls,
            execution_handler_cls=execution_handler_cls,
            portfolio_cls=portfolio_cls,
            strategy_cls=_AlwaysRejectingStrategy,
        )

    assert strategy_calls == [{}]
    assert portfolio_calls == []
    assert execution_calls == []


def test_runtime_config_is_wrapped_before_component_construction():
    data_handler_cls, _ = _make_data_handler_class()
    portfolio_cls, _ = _make_portfolio_strict_class()
    execution_handler_cls, _ = _make_execution_handler_class()
    runtime = RuntimeConfig()

    backtest = Backtest(
        csv_dir="data",
        symbol_list=[SYMBOL],
        start_date=START,
        data_handler_cls=data_handler_cls,
        execution_handler_cls=execution_handler_cls,
        portfolio_cls=portfolio_cls,
        strategy_cls=_make_strategy_class(),
        data_dict={SYMBOL: []},
        config=runtime,
    )

    assert isinstance(backtest.config, BacktestConfigView)
    assert backtest.config._rt is runtime
    assert backtest.portfolio.config is backtest.config
    assert backtest.execution_handler.config is backtest.config


def test_exact_alpha_max_config_bypasses_runtime_config_attribute_probes():
    unknown_reads: list[str] = []

    def reject_unknown_read(_self, name):
        unknown_reads.append(name)
        raise RuntimeError(f"unfrozen_runtime_field:{name}")

    config_type = type(
        "AlphaMaxBacktestConfig",
        (),
        {
            "__module__": "lumina_quant.research.alpha_max_engine_runner",
            "TIMEFRAME": "1s",
            "DECISION_CADENCE_SECONDS": 1,
            "SKIP_AHEAD_ENABLED": True,
            "__getattr__": reject_unknown_read,
        },
    )
    config = config_type()
    data_handler_cls, _ = _make_data_handler_class()
    portfolio_cls, _ = _make_portfolio_strict_class()
    execution_handler_cls, _ = _make_execution_handler_class()

    backtest = _build_backtest(
        data_handler_cls=data_handler_cls,
        execution_handler_cls=execution_handler_cls,
        portfolio_cls=portfolio_cls,
        strategy_cls=_make_strategy_class(),
        config=config,
        strategy_timeframe="1s",
    )

    assert backtest.config is config
    assert backtest.portfolio.config is config
    assert backtest.execution_handler.config is config
    assert unknown_reads == []
