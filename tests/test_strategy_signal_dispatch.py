from __future__ import annotations

import numpy as np
import pytest

from lumina_quant.strategy_factory.strategy_signal_dispatch import (
    StrategySignalDispatchError,
    StrategySignalDispatcher,
)


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
    assert set(meta) == {"evaluation_mode"} and meta["evaluation_mode"] in (
        "handler",
        "generic_fallback_proxy",
    )


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
            {"BTC/USDT:close": np.array([100.0, 101.0])},
            [],
            None,
        ),
        (
            StrategySignalDispatcher(handlers={}),
            {"strategy_class": "Known"},
            {"datetime": np.array([1, 2])},
            ["BTC/USDT"],
            None,
        ),
        (
            StrategySignalDispatcher(handlers={}),
            {"strategy_class": "Known"},
            {"BTC/USDT:close": np.array(["not", "prices"])},
            ["BTC/USDT"],
            None,
        ),
        (
            StrategySignalDispatcher(handlers={}),
            {"strategy_class": "Known"},
            {"BTC/USDT:close": np.array([100.0, np.nan])},
            ["BTC/USDT"],
            None,
        ),
        (
            StrategySignalDispatcher(handlers={}),
            {},
            {"BTC/USDT:close": np.array([100.0, 101.0])},
            ["BTC/USDT"],
            None,
        ),
        (
            StrategySignalDispatcher(handlers={}),
            {"strategy_class": "Unknown"},
            {"BTC/USDT:close": np.array([100.0, 101.0])},
            ["BTC/USDT"],
            None,
        ),
        (
            StrategySignalDispatcher(
                handlers={"Pair": lambda *args: (_ for _ in ()).throw(AssertionError())},
                minimum_symbol_counts={"Pair": 2},
            ),
            {"strategy_class": "Pair"},
            {"BTC/USDT:close": np.array([100.0, 101.0])},
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
                "datetime": np.array([1, 2, 3]),
                "BTC/USDT:close": np.array([100.0, 101.0]),
                "ETH/USDT:close": np.array([100.0, 101.0, 102.0]),
            },
            ["BTC/USDT", "ETH/USDT"],
            None,
        ),
        (
            StrategySignalDispatcher(handlers={"Known": lambda *args: None}),
            {"strategy_class": "Known"},
            {
                "BTC/USDT:close": np.array([100.0, 101.0]),
                "BTC/USDT:feature": np.array([1.0, np.inf]),
            },
            ["BTC/USDT"],
            None,
        ),
        (
            StrategySignalDispatcher(handlers={"Known": lambda *args: None}),
            {"strategy_class": "Known"},
            {
                "BTC/USDT:close": np.array([100.0, 101.0]),
                "BTC/USDT:feature": np.array([1.0]),
            },
            ["BTC/USDT"],
            None,
        ),
        (
            StrategySignalDispatcher(handlers={"Known": lambda *args: None}),
            {"strategy_class": "Known"},
            {"BTC/USDT:close": np.array([100.0, -1.0])},
            ["BTC/USDT"],
            None,
        ),
        (
            StrategySignalDispatcher(handlers={"Known": lambda *args: None}),
            {"strategy_class": "Known"},
            {"BTC/USDT:close": np.array([100.0, 101.0])},
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
            aligned={"BTC/USDT:close": np.array([100.0, 101.0])},
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
                aligned={"BTC/USDT:close": np.array([100.0, 101.0])},
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
            aligned={"BTC/USDT:close": np.array([100.0, 101.0])},
            symbols=["BTC/USDT"],
            require_actual_engine=True,
        )
    assert isinstance(handler_error.value.__cause__, ValueError)

    def _router(*args):
        raise LookupError("router failure")

    with pytest.raises(StrategySignalDispatchError) as router_error:
        StrategySignalDispatcher(handlers={}).dispatch(
            {"strategy_class": "Registered"},
            aligned={"BTC/USDT:close": np.array([100.0, 101.0])},
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
        aligned={"BTC/USDT:close": np.array([100.0, 101.0])},
        symbols=["BTC/USDT"],
        require_actual_engine=True,
    )
    router_result = dispatcher.dispatch(
        {"strategy_class": "Registered"},
        aligned={"BTC/USDT:close": np.array([100.0, 101.0])},
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
            aligned={"BTC/USDT:close": np.array([100.0, 101.0])},
            symbols=["BTC/USDT"],
            require_actual_engine=True,
        )
    assert isinstance(candidate_error.value.__cause__, AttributeError)

    with pytest.raises(StrategySignalDispatchError) as params_error:
        dispatcher.dispatch(
            {"strategy_class": "Known", "params": object()},
            aligned={"BTC/USDT:close": np.array([100.0, 101.0])},
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
            aligned={"BTC/USDT:close": np.array([1.0, 1e308])},
            symbols=["BTC/USDT"],
            require_actual_engine=True,
        )
    assert "derived portfolio outputs overflowed or became invalid" in str(error.value)
    assert isinstance(error.value.__cause__, FloatingPointError)
