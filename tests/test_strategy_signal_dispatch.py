from __future__ import annotations

import numpy as np

from lumina_quant.strategy_factory.strategy_signal_dispatch import StrategySignalDispatcher


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
