from __future__ import annotations

from datetime import UTC, datetime
import numpy as np
import pytest

from lumina_quant.strategy_factory.strategy_signal_dispatch import (
    StrategySignalDispatchError,
    StrategySignalDispatcher,
)


def _strict_market_panel(*symbols: str, length: int = 2) -> dict[str, np.ndarray]:
    if not symbols:
        symbols = ("BTC/USDT",)
    aligned: dict[str, np.ndarray] = {
        "datetime": np.array(
            [datetime(2025, 1, 1, hour, tzinfo=UTC) for hour in range(length)],
            dtype=object,
        )
    }
    for symbol in symbols:
        aligned.update(
            {
                f"{symbol}:open": np.arange(99.0, 99.0 + length),
                f"{symbol}:high": np.arange(101.0, 101.0 + length),
                f"{symbol}:low": np.arange(98.0, 98.0 + length),
                f"{symbol}:close": np.arange(100.0, 100.0 + length),
                f"{symbol}:volume": np.arange(10.0, 10.0 + length),
            }
        )
    return aligned


def test_strategy_signal_dispatcher_routes_to_explicit_handler():
    calls: list[tuple[dict[str, object], list[str], int]] = []

    def _handler(params, aligned, symbols, n, exposures, meta):
        calls.append((dict(params), list(symbols), int(n)))
        exposures[:] = 1.0
        meta["handled"] = True

    dispatcher = StrategySignalDispatcher(handlers={"ExplicitStrategy": _handler})
    aligned = {"BTC/USDT:close": np.array([100.0, 101.0, 102.0], dtype=float)}

    portfolio_ret, turnover, exposure, meta = dispatcher.dispatch(
        {"strategy_class": "ExplicitStrategy", "params": {"alpha": 1}},
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert calls == [({"alpha": 1}, ["BTC/USDT"], 3)]
    assert portfolio_ret.shape == (3,)
    assert turnover.shape == (3,)
    assert np.allclose(exposure, 1.0)
    assert meta["handled"] is True


def test_strategy_signal_dispatcher_falls_back_when_handler_requires_more_symbols():
    dispatcher = StrategySignalDispatcher(
        handlers={
            "PairStrategy": lambda *args: (_ for _ in ()).throw(
                AssertionError("handler should not run")
            )
        },
        minimum_symbol_counts={"PairStrategy": 2},
    )
    aligned = {"BTC/USDT:close": np.array([100.0, 102.0, 104.0], dtype=float)}

    portfolio_ret, turnover, exposure, meta = dispatcher.dispatch(
        {"strategy_class": "PairStrategy"},
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert portfolio_ret.shape == (3,)
    assert turnover.shape == (3,)
    assert exposure.shape == (3,)
    assert meta == {"evaluation_mode": "generic_fallback_proxy"}


@pytest.mark.parametrize(
    ("dispatcher", "candidate", "aligned", "symbols", "router"),
    [
        (
            StrategySignalDispatcher(handlers={}),
            {"strategy_class": "Unknown"},
            {},
            ["BTC/USDT"],
            None,
        ),
        (
            StrategySignalDispatcher(handlers={}),
            {"strategy_class": "Known"},
            _strict_market_panel(),
            [],
            None,
        ),
        (
            StrategySignalDispatcher(handlers={}),
            {"strategy_class": "Known"},
            {**_strict_market_panel(), "datetime": np.array([1, 2])},
            ["BTC/USDT"],
            None,
        ),
        (
            StrategySignalDispatcher(handlers={}),
            {"strategy_class": "Known"},
            {**_strict_market_panel(), "BTC/USDT:close": np.array(["not", "prices"])},
            ["BTC/USDT"],
            None,
        ),
        (
            StrategySignalDispatcher(handlers={}),
            {"strategy_class": "Known"},
            {**_strict_market_panel(), "BTC/USDT:close": np.array([100.0, np.nan])},
            ["BTC/USDT"],
            None,
        ),
        (
            StrategySignalDispatcher(handlers={}),
            {},
            _strict_market_panel(),
            ["BTC/USDT"],
            None,
        ),
        (
            StrategySignalDispatcher(handlers={}),
            {"strategy_class": "Unknown"},
            _strict_market_panel(),
            ["BTC/USDT"],
            None,
        ),
        (
            StrategySignalDispatcher(
                handlers={"Pair": lambda *args: (_ for _ in ()).throw(AssertionError())},
                minimum_symbol_counts={"Pair": 2},
            ),
            {"strategy_class": "Pair"},
            _strict_market_panel(),
            ["BTC/USDT"],
            None,
        ),
        (
            StrategySignalDispatcher(handlers={"Known": lambda *args: None}),
            {"strategy_class": "Known"},
            {},
            ["BTC/USDT"],
            None,
        ),
        (
            StrategySignalDispatcher(handlers={"Known": lambda *args: None}),
            {"strategy_class": "Known"},
            {
                **_strict_market_panel("BTC/USDT", "ETH/USDT", length=3),
                "BTC/USDT:close": np.array([100.0, 101.0]),
            },
            ["BTC/USDT", "ETH/USDT"],
            None,
        ),
        (
            StrategySignalDispatcher(handlers={"Known": lambda *args: None}),
            {"strategy_class": "Known"},
            {
                **_strict_market_panel(),
                "BTC/USDT:feature": np.array([1.0]),
            },
            ["BTC/USDT"],
            None,
        ),
        (
            StrategySignalDispatcher(handlers={"Known": lambda *args: None}),
            {"strategy_class": "Known"},
            {**_strict_market_panel(), "BTC/USDT:close": np.array([100.0, -1.0])},
            ["BTC/USDT"],
            None,
        ),
        (
            StrategySignalDispatcher(handlers={"Known": lambda *args: None}),
            {"strategy_class": "Known"},
            _strict_market_panel(),
            ["BTC/USDT", "BTC/USDT"],
            None,
        ),
    ],
)
def test_strict_dispatch_rejects_invalid_input_without_generic_fallback(
    monkeypatch, dispatcher, candidate, aligned, symbols, router
):
    fallback_calls = 0

    def _fallback(**kwargs):
        nonlocal fallback_calls
        fallback_calls += 1

    monkeypatch.setattr(
        StrategySignalDispatcher, "_apply_generic_fallback", staticmethod(_fallback)
    )
    with pytest.raises(StrategySignalDispatchError):
        dispatcher.dispatch(
            candidate,
            aligned=aligned,
            symbols=symbols,
            unmapped_router=router,
            require_actual_engine=True,
        )
    assert fallback_calls == 0


@pytest.mark.parametrize(
    "router",
    [
        lambda *args: (_ for _ in ()).throw(ValueError("router failure")),
        lambda *args: None,
        lambda *args: np.zeros((1, 1)),
        lambda *args: np.array([[np.nan, 0.0]]),
    ],
)
def test_strict_dispatch_rejects_bad_registry_output_without_generic_fallback(monkeypatch, router):
    dispatcher = StrategySignalDispatcher(handlers={})
    fallback_calls = 0

    def _fallback(**kwargs):
        nonlocal fallback_calls
        fallback_calls += 1

    monkeypatch.setattr(
        StrategySignalDispatcher, "_apply_generic_fallback", staticmethod(_fallback)
    )
    with pytest.raises(StrategySignalDispatchError):
        dispatcher.dispatch(
            {"strategy_class": "Registered"},
            aligned=_strict_market_panel(),
            symbols=["BTC/USDT"],
            unmapped_router=router,
            require_actual_engine=True,
        )
    assert fallback_calls == 0


def test_strict_dispatch_rejects_handler_exception_nonfinite_and_invalid_mode(monkeypatch):
    def _handler(params, aligned, symbols, n, exposures, meta):
        if params["failure"] == "exception":
            raise ValueError("handler failure")
        if params["failure"] == "nonfinite":
            exposures[:] = np.nan
        elif params["failure"] == "registry_mode":
            meta["evaluation_mode"] = "registry_simulator"
        elif params["failure"] == "fallback_count":
            meta["generic_fallback_proxy_count"] = 1
        elif params["failure"] == "fallback_bool":
            meta["generic_fallback_proxy_count"] = False
        else:
            meta["evaluation_mode"] = "generic_fallback_proxy"

    fallback_calls = 0

    def _fallback(**kwargs):
        nonlocal fallback_calls
        fallback_calls += 1

    dispatcher = StrategySignalDispatcher(handlers={"Known": _handler})
    monkeypatch.setattr(
        StrategySignalDispatcher, "_apply_generic_fallback", staticmethod(_fallback)
    )
    for failure in (
        "exception",
        "nonfinite",
        "registry_mode",
        "fallback_count",
        "fallback_bool",
        "invalid_mode",
    ):
        with pytest.raises(StrategySignalDispatchError):
            dispatcher.dispatch(
                {"strategy_class": "Known", "params": {"failure": failure}},
                aligned=_strict_market_panel(),
                symbols=["BTC/USDT"],
                require_actual_engine=True,
            )
    assert fallback_calls == 0


def test_strict_dispatch_errors_preserve_handler_and_router_causes():
    def _handler(*args):
        raise ValueError("handler failure")

    dispatcher = StrategySignalDispatcher(handlers={"Known": _handler})
    with pytest.raises(StrategySignalDispatchError) as handler_error:
        dispatcher.dispatch(
            {"strategy_class": "Known"},
            aligned=_strict_market_panel(),
            symbols=["BTC/USDT"],
            require_actual_engine=True,
        )
    assert isinstance(handler_error.value.__cause__, ValueError)

    def _router(*args):
        raise LookupError("router failure")

    with pytest.raises(StrategySignalDispatchError) as router_error:
        StrategySignalDispatcher(handlers={}).dispatch(
            {"strategy_class": "Registered"},
            aligned=_strict_market_panel(),
            symbols=["BTC/USDT"],
            unmapped_router=_router,
            require_actual_engine=True,
        )
    assert isinstance(router_error.value.__cause__, LookupError)


def test_strict_dispatch_accepts_handler_and_registry_simulator_modes():
    def _handler(params, aligned, symbols, n, exposures, meta):
        exposures[:] = 0.5

    dispatcher = StrategySignalDispatcher(handlers={"Handled": _handler})
    handler_result = dispatcher.dispatch(
        {"strategy_class": "Handled"},
        aligned=_strict_market_panel(),
        symbols=["BTC/USDT"],
        require_actual_engine=True,
    )
    router_result = dispatcher.dispatch(
        {"strategy_class": "Registered"},
        aligned=_strict_market_panel(),
        symbols=["BTC/USDT"],
        unmapped_router=lambda *args: np.array([[0.5, 0.5]]),
        require_actual_engine=True,
    )

    for result, mode in ((handler_result, "handler"), (router_result, "registry_simulator")):
        assert result[3]["evaluation_mode"] == mode
        assert result[3]["generic_fallback_proxy_count"] == 0
        assert all(np.isfinite(values).all() for values in result[:3])


def test_strict_dispatch_wraps_malformed_candidate_and_params_causes():
    dispatcher = StrategySignalDispatcher(handlers={})

    with pytest.raises(StrategySignalDispatchError) as candidate_error:
        dispatcher.dispatch(
            [],
            aligned=_strict_market_panel(),
            symbols=["BTC/USDT"],
            require_actual_engine=True,
        )
    assert isinstance(candidate_error.value.__cause__, AttributeError)

    with pytest.raises(StrategySignalDispatchError) as params_error:
        dispatcher.dispatch(
            {"strategy_class": "Known", "params": object()},
            aligned=_strict_market_panel(),
            symbols=["BTC/USDT"],
            require_actual_engine=True,
        )
    assert isinstance(params_error.value.__cause__, TypeError)


def test_strict_dispatch_rejects_overflowed_final_portfolio_return():
    def _handler(params, aligned, symbols, n, exposures, meta):
        exposures[:] = np.finfo(float).max

    dispatcher = StrategySignalDispatcher(handlers={"Known": _handler})
    with pytest.raises(StrategySignalDispatchError) as error:
        dispatcher.dispatch(
            {"strategy_class": "Known"},
            aligned={**_strict_market_panel(), "BTC/USDT:close": np.array([1.0, 1e308])},
            symbols=["BTC/USDT"],
            require_actual_engine=True,
        )
    assert "derived portfolio outputs overflowed or became invalid" in str(error.value)
    assert isinstance(error.value.__cause__, FloatingPointError)


@pytest.mark.parametrize(
    "timestamps",
    [
        np.array([1, 2], dtype=np.int64),
        np.array(["2025-01-01", "2025-01-02"]),
    ],
)
def test_strict_dispatch_rejects_non_datetime_timestamp_carriers(timestamps):
    dispatcher = StrategySignalDispatcher(handlers={"Known": lambda *args: None})
    aligned = {**_strict_market_panel(), "datetime": timestamps}

    with pytest.raises(StrategySignalDispatchError, match="invalid datetime array"):
        dispatcher.dispatch(
            {"strategy_class": "Known"},
            aligned=aligned,
            symbols=["BTC/USDT"],
            require_actual_engine=True,
        )

    result = dispatcher.dispatch(
        {"strategy_class": "Known"},
        aligned=aligned,
        symbols=["BTC/USDT"],
    )
    assert result[3]["evaluation_mode"] == "handler"


@pytest.mark.parametrize(
    "timestamps",
    [
        np.array(["2025-01-01T00:00", "2025-01-01T01:00"], dtype="datetime64[m]"),
        np.array(
            [
                datetime(2025, 1, 1, tzinfo=UTC),
                datetime(2025, 1, 1, 1, tzinfo=UTC),
            ],
            dtype=object,
        ),
    ],
)
def test_strict_dispatch_accepts_supported_datetime_timestamp_carriers(timestamps):
    dispatcher = StrategySignalDispatcher(handlers={"Known": lambda *args: None})
    result = dispatcher.dispatch(
        {"strategy_class": "Known"},
        aligned={**_strict_market_panel(), "datetime": timestamps},
        symbols=["BTC/USDT"],
        require_actual_engine=True,
    )
    assert result[3]["evaluation_mode"] == "handler"


def test_strict_dispatch_allows_nonfinite_optional_support_data_for_handler():
    received: list[np.ndarray] = []

    def _handler(params, aligned, symbols, n, exposures, meta):
        received.append(aligned["BTC/USDT:feature"])
        exposures[:] = 0.25

    dispatcher = StrategySignalDispatcher(handlers={"Known": _handler})
    result = dispatcher.dispatch(
        {"strategy_class": "Known"},
        aligned={
            **_strict_market_panel(),
            "BTC/USDT:feature": np.array([np.nan, np.inf]),
        },
        symbols=["BTC/USDT"],
        require_actual_engine=True,
    )

    assert len(received) == 1
    assert np.isnan(received[0][0])
    assert np.isinf(received[0][1])
    assert result[3]["evaluation_mode"] == "handler"


@pytest.mark.parametrize(
    ("missing_key", "message"),
    [
        ("datetime", "missing datetime array"),
        ("BTC/USDT:open", "missing required bar array for BTC/USDT:open"),
        ("BTC/USDT:high", "missing required bar array for BTC/USDT:high"),
        ("BTC/USDT:low", "missing required bar array for BTC/USDT:low"),
        ("BTC/USDT:close", "missing required bar array for BTC/USDT:close"),
        ("BTC/USDT:volume", "missing required bar array for BTC/USDT:volume"),
    ],
)
def test_strict_dispatch_requires_complete_market_panel_before_handler(missing_key, message):
    calls = 0

    def _handler(*args):
        nonlocal calls
        calls += 1

    aligned = _strict_market_panel()
    del aligned[missing_key]

    with pytest.raises(StrategySignalDispatchError) as error:
        StrategySignalDispatcher(handlers={"Known": _handler}).dispatch(
            {"strategy_class": "Known"},
            aligned=aligned,
            symbols=["BTC/USDT"],
            require_actual_engine=True,
        )

    assert message in str(error.value)
    assert calls == 0


@pytest.mark.parametrize("bar_field", ("open", "high", "low", "close", "volume"))
def test_strict_dispatch_rejects_nonfinite_mandatory_bar_data(bar_field):
    aligned = _strict_market_panel()
    aligned[f"BTC/USDT:{bar_field}"][1] = np.nan

    with pytest.raises(StrategySignalDispatchError, match="nonfinite aligned array"):
        StrategySignalDispatcher(handlers={"Known": lambda *args: None}).dispatch(
            {"strategy_class": "Known"},
            aligned=aligned,
            symbols=["BTC/USDT"],
            require_actual_engine=True,
        )


@pytest.mark.parametrize(
    "missing_support",
    (
        {"missing_support_data": True},
        {"missing_support_symbols": ["BTC/USDT"]},
    ),
)
def test_strict_dispatch_rejects_missing_support_without_generic_fallback(
    monkeypatch, missing_support
):
    def _handler(params, aligned, symbols, n, exposures, meta):
        meta.update(missing_support)

    fallback_calls = 0

    def _fallback(**kwargs):
        nonlocal fallback_calls
        fallback_calls += 1

    monkeypatch.setattr(
        StrategySignalDispatcher, "_apply_generic_fallback", staticmethod(_fallback)
    )

    with pytest.raises(StrategySignalDispatchError, match="missing required support data"):
        StrategySignalDispatcher(handlers={"Known": _handler}).dispatch(
            {"strategy_class": "Known"},
            aligned=_strict_market_panel(),
            symbols=["BTC/USDT"],
            require_actual_engine=True,
        )

    assert fallback_calls == 0
