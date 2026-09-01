"""Mode-based system assembly shared by backtest/live entrypoints."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any, ClassVar


def _resolve_live_portfolio_cls():
    from lumina_quant.live.portfolio import LivePortfolio as _LivePortfolio

    return _LivePortfolio


@dataclass(slots=True, frozen=True)
class RuntimeContract(Mapping[str, Any]):
    """Typed runtime contract for backtest and live engine assembly."""

    mode: str
    engine_cls: type[Any]
    data_handler_cls: type[Any]
    execution_handler_cls: type[Any]
    portfolio_cls: type[Any]
    transport: str | None = None
    fatal_error_cls: type[BaseException] | None = None

    _field_names: ClassVar[tuple[str, ...]] = (
        "mode",
        "engine_cls",
        "data_handler_cls",
        "execution_handler_cls",
        "portfolio_cls",
        "transport",
        "fatal_error_cls",
    )

    def __getitem__(self, key: str) -> Any:
        if key not in self._field_names:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._field_names)

    def __len__(self) -> int:
        return len(self._field_names)

    def to_dict(self) -> dict[str, Any]:
        return {key: self[key] for key in self}


def build_live_runtime_contract(*, transport: str = "poll") -> RuntimeContract:
    selected_transport = str(transport or "poll").strip().lower() or "poll"
    if selected_transport not in {"poll", "ws"}:
        raise ValueError(f"Unsupported live transport: {transport}")

    if selected_transport == "ws":
        from lumina_quant.live.data_ws import BinanceWebSocketDataHandler as LiveDataHandler
    else:
        from lumina_quant.live.data_poll import LiveDataHandler

    from lumina_quant.live.execution_live import LiveExecutionHandler
    from lumina_quant.live.trader import LiveDataFatalError, LiveTrader

    return RuntimeContract(
        mode="live",
        engine_cls=LiveTrader,
        data_handler_cls=LiveDataHandler,
        execution_handler_cls=LiveExecutionHandler,
        portfolio_cls=_resolve_live_portfolio_cls(),
        fatal_error_cls=LiveDataFatalError,
        transport=selected_transport,
    )


def build_system(mode: str, *, transport: str = "poll") -> RuntimeContract:
    selected = str(mode or "").strip().lower()
    if selected == "backtest":
        from lumina_quant.backtesting.backtest import Backtest
        from lumina_quant.backtesting.data import HistoricCSVDataHandler
        from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
        from lumina_quant.backtesting.portfolio_backtest import Portfolio

        return RuntimeContract(
            mode="backtest",
            engine_cls=Backtest,
            data_handler_cls=HistoricCSVDataHandler,
            execution_handler_cls=SimulatedExecutionHandler,
            portfolio_cls=Portfolio,
        )

    if selected in {"live", "sandbox"}:
        contract = build_live_runtime_contract(transport=transport)
        return replace(contract, mode=selected)

    raise ValueError(f"Unsupported system mode: {mode}")


def __getattr__(name: str):
    if name == "LivePortfolio":
        return _resolve_live_portfolio_cls()
    raise AttributeError(name)
