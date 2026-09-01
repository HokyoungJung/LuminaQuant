"""Cross-crypto slow-diffusion lead-lag strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.common import safe_float
from lumina_quant.strategy import Strategy
from lumina_quant.symbols import canonical_symbol
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@dataclass(slots=True)
class _PositionState:
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    last_completed_bar_key: str = ""


class CrossCryptoSlowDiffusionStrategy(Strategy):
    """Trade a lagging crypto after a large leader move survives raw-first screening."""

    uses_timeframe_aggregator = True
    required_timeframes = ("1h",)
    required_lookbacks = {"1h": 64}

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "leader_symbol": HyperParam.string("leader_symbol", default="BTC/USDT", tunable=False),
            "target_symbol": HyperParam.string("target_symbol", default="ETH/USDT", tunable=False),
            "timeframe": HyperParam.string("timeframe", default="1h", tunable=False),
            "lag_bars": HyperParam.integer("lag_bars", default=2, low=1, high=24),
            "leader_abs_ret_min": HyperParam.floating(
                "leader_abs_ret_min", default=0.015, low=0.0, high=1.0
            ),
            "target_underreaction_cap": HyperParam.floating(
                "target_underreaction_cap", default=999.0, low=0.0, high=999.0
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=8, low=1, high=240),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.008, low=0.0, high=1.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=175.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.0, low=0.0, high=1.0, tunable=False
            ),
            "take_profit_pct": HyperParam.floating(
                "take_profit_pct", default=0.0, low=0.0, high=1.0, tunable=False
            ),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
        }

    def __init__(
        self,
        bars: Any,
        events: Any,
        leader_symbol: str = "BTC/USDT",
        target_symbol: str = "ETH/USDT",
        timeframe: str = "1h",
        lag_bars: int = 2,
        leader_abs_ret_min: float = 0.015,
        target_underreaction_cap: float = 999.0,
        max_hold_bars: int = 8,
        target_allocation: float = 0.008,
        max_order_value: float = 175.0,
        stop_loss_pct: float = 0.0,
        take_profit_pct: float = 0.0,
        allow_short: bool = True,
    ) -> None:
        self.bars = bars
        self.events = events
        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "leader_symbol": leader_symbol,
                "target_symbol": target_symbol,
                "timeframe": timeframe,
                "lag_bars": lag_bars,
                "leader_abs_ret_min": leader_abs_ret_min,
                "target_underreaction_cap": target_underreaction_cap,
                "max_hold_bars": max_hold_bars,
                "target_allocation": target_allocation,
                "max_order_value": max_order_value,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "allow_short": allow_short,
            },
            keep_unknown=False,
        )
        self.leader_symbol = canonical_symbol(str(resolved["leader_symbol"]))
        self.target_symbol = canonical_symbol(str(resolved["target_symbol"]))
        self.timeframe = str(resolved["timeframe"] or "1h")
        self.lag_bars = max(1, int(resolved["lag_bars"]))
        self.leader_abs_ret_min = max(0.0, float(resolved["leader_abs_ret_min"]))
        self.target_underreaction_cap = max(0.0, float(resolved["target_underreaction_cap"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.take_profit_pct = max(0.0, float(resolved["take_profit_pct"]))
        self.allow_short = bool(resolved["allow_short"])
        self._state = _PositionState()

    def get_state(self) -> dict[str, Any]:
        return {
            "mode": self._state.mode,
            "entry_price": self._state.entry_price,
            "bars_held": int(self._state.bars_held),
            "last_completed_bar_key": self._state.last_completed_bar_key,
        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        mode = str(state.get("mode", "OUT")).upper()
        self._state.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
        self._state.entry_price = safe_float(state.get("entry_price"))
        try:
            self._state.bars_held = max(0, int(state.get("bars_held", 0)))
        except Exception:
            self._state.bars_held = 0
        self._state.last_completed_bar_key = str(state.get("last_completed_bar_key", ""))

    def calculate_signals(self, event: Any) -> None:
        _ = event
        return

    @staticmethod
    def _completed_bars(aggregator: Any, symbol: str, timeframe: str, lookback: int) -> list[Any]:
        getter = getattr(aggregator, "get_bars", None)
        if not callable(getter):
            return []
        bars = list(
            getter(symbol=str(symbol), timeframe=str(timeframe), n=max(lookback + 1, 2)) or []
        )
        # get_bars includes the currently forming bar; the screen used completed hourly marks.
        return bars[:-1] if len(bars) >= 2 else []

    @staticmethod
    def _close(bar: Any) -> float | None:
        if isinstance(bar, (tuple, list)) and len(bar) >= 5:
            return safe_float(bar[4])
        if isinstance(bar, dict):
            return safe_float(bar.get("close"))
        return None

    @staticmethod
    def _time_key(bar: Any) -> str:
        if isinstance(bar, (tuple, list)) and bar:
            return str(bar[0])
        if isinstance(bar, dict):
            return str(bar.get("time") or bar.get("datetime") or "")
        return ""

    def _metadata(self, *, leader_ret: float, target_ret: float, reason: str) -> dict[str, Any]:
        return {
            "strategy": "CrossCryptoSlowDiffusionStrategy",
            "reason": reason,
            "leader_symbol": self.leader_symbol,
            "target_symbol": self.target_symbol,
            "timeframe": self.timeframe,
            "lag_bars": int(self.lag_bars),
            "leader_return": float(leader_ret),
            "target_return": float(target_ret),
            "leader_abs_ret_min": float(self.leader_abs_ret_min),
            "target_underreaction_cap": float(self.target_underreaction_cap),
            "target_allocation": float(self.target_allocation),
            "max_symbol_exposure_pct": float(self.target_allocation),
            "max_order_value": float(self.max_order_value),
        }

    def _emit(
        self,
        *,
        event_time: Any,
        signal_type: str,
        price: float | None,
        metadata: dict[str, Any],
    ) -> None:
        stop_loss = None
        take_profit = None
        if price is not None and price > 0.0:
            if signal_type == "LONG":
                stop_loss = price * (1.0 - self.stop_loss_pct) if self.stop_loss_pct > 0.0 else None
                take_profit = (
                    price * (1.0 + self.take_profit_pct) if self.take_profit_pct > 0.0 else None
                )
            elif signal_type == "SHORT":
                stop_loss = price * (1.0 + self.stop_loss_pct) if self.stop_loss_pct > 0.0 else None
                take_profit = (
                    price * (1.0 - self.take_profit_pct) if self.take_profit_pct > 0.0 else None
                )
        self.events.put(
            SignalEvent(
                strategy_id="cross_crypto_slow_diffusion",
                symbol=self.target_symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=float(self.target_allocation if signal_type in {"LONG", "SHORT"} else 1.0),
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
            )
        )

    def _maybe_exit(
        self, *, event_time: Any, price: float, leader_ret: float, target_ret: float
    ) -> bool:
        if self._state.mode == "OUT":
            return False
        self._state.bars_held += 1
        stop_hit = False
        take_profit_hit = False
        if self._state.entry_price and self.stop_loss_pct > 0.0:
            if self._state.mode == "LONG":
                stop_hit = price <= self._state.entry_price * (1.0 - self.stop_loss_pct)
                take_profit_hit = (
                    self.take_profit_pct > 0.0
                    and price >= self._state.entry_price * (1.0 + self.take_profit_pct)
                )
            else:
                stop_hit = price >= self._state.entry_price * (1.0 + self.stop_loss_pct)
                take_profit_hit = (
                    self.take_profit_pct > 0.0
                    and price <= self._state.entry_price * (1.0 - self.take_profit_pct)
                )
        if not stop_hit and not take_profit_hit and self._state.bars_held < self.max_hold_bars:
            return True
        reason = "max_hold_exit"
        if stop_hit:
            reason = "stop_loss_exit"
        elif take_profit_hit:
            reason = "take_profit_exit"
        self._emit(
            event_time=event_time,
            signal_type="EXIT",
            price=price,
            metadata=self._metadata(leader_ret=leader_ret, target_ret=target_ret, reason=reason),
        )
        self._state.mode = "OUT"
        self._state.entry_price = None
        self._state.bars_held = 0
        return False

    def calculate_signals_window(self, event: Any, aggregator: Any) -> None:
        if aggregator is None:
            return
        lookback = self.lag_bars + 2
        leader_bars = self._completed_bars(aggregator, self.leader_symbol, self.timeframe, lookback)
        target_bars = self._completed_bars(aggregator, self.target_symbol, self.timeframe, lookback)
        if len(leader_bars) <= self.lag_bars or len(target_bars) <= self.lag_bars:
            return
        latest_target_bar = target_bars[-1]
        completed_key = self._time_key(latest_target_bar)
        if not completed_key or completed_key == self._state.last_completed_bar_key:
            return
        self._state.last_completed_bar_key = completed_key

        latest_leader_bar = leader_bars[-1]
        event_time = (
            latest_target_bar[0]
            if isinstance(latest_target_bar, (tuple, list))
            else getattr(event, "time", None)
        )
        leader_close = self._close(latest_leader_bar)
        target_close = self._close(latest_target_bar)
        leader_base = self._close(leader_bars[-1 - self.lag_bars])
        target_base = self._close(target_bars[-1 - self.lag_bars])
        if not all(
            value is not None and value > 0.0
            for value in (leader_close, target_close, leader_base, target_base)
        ):
            return

        leader_ret = float(leader_close / leader_base - 1.0)
        target_ret = float(target_close / target_base - 1.0)
        if self._maybe_exit(
            event_time=event_time,
            price=float(target_close),
            leader_ret=leader_ret,
            target_ret=target_ret,
        ):
            return
        if self._state.mode != "OUT":
            return

        if abs(leader_ret) < self.leader_abs_ret_min:
            return
        if (
            self.target_underreaction_cap < 999.0
            and abs(target_ret) > abs(leader_ret) * self.target_underreaction_cap
        ):
            return
        if leader_ret > 0.0:
            signal_type = "LONG"
            reason = "leader_up_target_lag_long"
        elif self.allow_short:
            signal_type = "SHORT"
            reason = "leader_down_target_lag_short"
        else:
            return

        self._emit(
            event_time=event_time,
            signal_type=signal_type,
            price=float(target_close),
            metadata=self._metadata(leader_ret=leader_ret, target_ret=target_ret, reason=reason),
        )
        self._state.mode = signal_type
        self._state.entry_price = float(target_close)
        self._state.bars_held = 0
