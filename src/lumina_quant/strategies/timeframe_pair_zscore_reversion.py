"""Completed-timeframe pair z-score reversion for precious-metal spreads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math
from statistics import mean
from typing import Any

from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.common import safe_float
from lumina_quant.indicators.rolling_stats import rolling_beta, rolling_corr, sample_std
from lumina_quant.strategy import Strategy
from lumina_quant.symbols import canonical_symbol
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@dataclass(slots=True)
class _PairPositionState:
    mode: str = "OUT"
    entry_x_price: float | None = None
    entry_y_price: float | None = None
    entry_beta: float | None = None
    bars_held: int = 0
    last_completed_bar_key: str = ""


class TimeframePairZScoreReversionStrategy(Strategy):
    """Fade completed-bar deviations in a hedge-adjusted pair spread.

    The original event-driven pair strategy is excellent for research frames,
    but live/backtest portfolio modes ingest 1s bars.  This strategy makes the
    pair alpha live-equivalent by only reading completed aggregator bars (for
    example 30m/1h), then emitting explicit two-leg LONG/SHORT/EXIT signals
    with target-allocation metadata for portfolio-level risk caps.
    """

    uses_timeframe_aggregator = True
    required_timeframes = ("1h",)
    required_lookbacks = {"1h": 256}

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "symbol_x": HyperParam.string("symbol_x", default="XAU/USDT", tunable=False),
            "symbol_y": HyperParam.string("symbol_y", default="XAG/USDT", tunable=False),
            "timeframe": HyperParam.string("timeframe", default="1h", tunable=False),
            "lookback_window": HyperParam.integer(
                "lookback_window",
                default=96,
                low=4,
                high=20000,
                optuna={"type": "int", "low": 24, "high": 240},
                grid=[48, 72, 96, 144],
            ),
            "hedge_window": HyperParam.integer(
                "hedge_window",
                default=192,
                low=4,
                high=40000,
                optuna={"type": "int", "low": 48, "high": 480},
                grid=[96, 144, 192, 288],
            ),
            "entry_z": HyperParam.floating(
                "entry_z",
                default=1.5,
                low=0.1,
                high=20.0,
                optuna={"type": "float", "low": 1.0, "high": 3.0, "step": 0.1},
                grid=[1.2, 1.5, 1.8, 2.2],
            ),
            "exit_z": HyperParam.floating(
                "exit_z",
                default=0.25,
                low=0.0,
                high=20.0,
                optuna={"type": "float", "low": 0.1, "high": 1.0, "step": 0.05},
                grid=[0.2, 0.4, 0.6],
            ),
            "stop_z": HyperParam.floating("stop_z", default=3.8, low=0.2, high=50.0),
            "min_correlation": HyperParam.floating(
                "min_correlation", default=0.10, low=-1.0, high=1.0
            ),
            "min_abs_beta": HyperParam.floating("min_abs_beta", default=0.02, low=0.0, high=100.0),
            "max_abs_beta": HyperParam.floating("max_abs_beta", default=8.0, low=0.01, high=100.0),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=72, low=1, high=100000),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.020, low=0.0, high=1.0, tunable=False
            ),
            "take_profit_pct": HyperParam.floating(
                "take_profit_pct", default=0.040, low=0.0, high=1.0, tunable=False
            ),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.020, low=0.0, high=1.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=350.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "allow_long_spread": HyperParam.boolean("allow_long_spread", default=True),
            "allow_short_spread": HyperParam.boolean("allow_short_spread", default=True),
            "entry_hours_utc": HyperParam.string("entry_hours_utc", default="", tunable=False),
            "excluded_entry_hours_utc": HyperParam.string(
                "excluded_entry_hours_utc", default="", tunable=False
            ),
            "min_entry_volume_x": HyperParam.floating(
                "min_entry_volume_x", default=0.0, low=0.0, high=1_000_000_000.0, tunable=False
            ),
            "min_entry_volume_y": HyperParam.floating(
                "min_entry_volume_y", default=0.0, low=0.0, high=1_000_000_000.0, tunable=False
            ),
        }

    def __init__(
        self,
        bars: Any,
        events: Any,
        symbol_x: str = "XAU/USDT",
        symbol_y: str = "XAG/USDT",
        timeframe: str = "1h",
        lookback_window: int = 96,
        hedge_window: int = 192,
        entry_z: float = 1.5,
        exit_z: float = 0.25,
        stop_z: float = 3.8,
        min_correlation: float = 0.10,
        min_abs_beta: float = 0.02,
        max_abs_beta: float = 8.0,
        max_hold_bars: int = 72,
        stop_loss_pct: float = 0.020,
        take_profit_pct: float = 0.040,
        target_allocation: float = 0.020,
        max_order_value: float = 350.0,
        allow_long_spread: bool = True,
        allow_short_spread: bool = True,
        entry_hours_utc: str | list[int] | tuple[int, ...] = "",
        excluded_entry_hours_utc: str | list[int] | tuple[int, ...] = "",
        min_entry_volume_x: float = 0.0,
        min_entry_volume_y: float = 0.0,
    ) -> None:
        self.bars = bars
        self.events = events
        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "symbol_x": symbol_x,
                "symbol_y": symbol_y,
                "timeframe": timeframe,
                "lookback_window": lookback_window,
                "hedge_window": hedge_window,
                "entry_z": entry_z,
                "exit_z": exit_z,
                "stop_z": stop_z,
                "min_correlation": min_correlation,
                "min_abs_beta": min_abs_beta,
                "max_abs_beta": max_abs_beta,
                "max_hold_bars": max_hold_bars,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "target_allocation": target_allocation,
                "max_order_value": max_order_value,
                "allow_long_spread": allow_long_spread,
                "allow_short_spread": allow_short_spread,
                "entry_hours_utc": entry_hours_utc,
                "excluded_entry_hours_utc": excluded_entry_hours_utc,
                "min_entry_volume_x": min_entry_volume_x,
                "min_entry_volume_y": min_entry_volume_y,
            },
            keep_unknown=False,
        )
        self.symbol_x = canonical_symbol(str(resolved["symbol_x"]))
        self.symbol_y = canonical_symbol(str(resolved["symbol_y"]))
        if not self.symbol_x or not self.symbol_y or self.symbol_x == self.symbol_y:
            raise ValueError("TimeframePairZScoreReversionStrategy requires two distinct symbols.")
        self.symbol_list = [self.symbol_x, self.symbol_y]
        self.timeframe = str(resolved["timeframe"] or "1h")
        self.lookback_window = max(4, int(resolved["lookback_window"]))
        self.hedge_window = max(self.lookback_window, int(resolved["hedge_window"]))
        self.entry_z = max(0.0, float(resolved["entry_z"]))
        self.exit_z = max(0.0, float(resolved["exit_z"]))
        self.stop_z = max(self.entry_z, float(resolved["stop_z"]))
        self.min_correlation = max(-1.0, min(1.0, float(resolved["min_correlation"])))
        self.min_abs_beta = max(0.0, float(resolved["min_abs_beta"]))
        self.max_abs_beta = max(self.min_abs_beta + 1e-9, float(resolved["max_abs_beta"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.take_profit_pct = max(0.0, float(resolved["take_profit_pct"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.allow_long_spread = bool(resolved["allow_long_spread"])
        self.allow_short_spread = bool(resolved["allow_short_spread"])
        self.entry_hours_utc = self._parse_hours(resolved["entry_hours_utc"])
        self.excluded_entry_hours_utc = self._parse_hours(resolved["excluded_entry_hours_utc"])
        self.min_entry_volume_x = max(0.0, float(resolved["min_entry_volume_x"]))
        self.min_entry_volume_y = max(0.0, float(resolved["min_entry_volume_y"]))
        self.required_timeframes = (self.timeframe,)
        self.required_lookbacks = {
            self.timeframe: max(self.hedge_window + self.lookback_window + 4, 64)
        }
        self._state = _PairPositionState()

    def get_state(self) -> dict[str, Any]:
        return {
            "mode": self._state.mode,
            "entry_x_price": self._state.entry_x_price,
            "entry_y_price": self._state.entry_y_price,
            "entry_beta": self._state.entry_beta,
            "bars_held": int(self._state.bars_held),
            "last_completed_bar_key": self._state.last_completed_bar_key,
        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        mode = str(state.get("mode", "OUT")).upper()
        self._state.mode = mode if mode in {"OUT", "LONG_SPREAD", "SHORT_SPREAD"} else "OUT"
        self._state.entry_x_price = safe_float(state.get("entry_x_price"))
        self._state.entry_y_price = safe_float(state.get("entry_y_price"))
        self._state.entry_beta = safe_float(state.get("entry_beta"))
        try:
            self._state.bars_held = max(0, int(state.get("bars_held", 0)))
        except Exception:
            self._state.bars_held = 0
        self._state.last_completed_bar_key = str(state.get("last_completed_bar_key", ""))

    @staticmethod
    def _parse_hours(raw: Any) -> frozenset[int]:
        if raw is None:
            return frozenset()
        if isinstance(raw, str):
            tokens = [item.strip() for item in raw.replace(";", ",").split(",")]
        else:
            try:
                tokens = list(raw)
            except TypeError:
                tokens = [raw]
        hours: set[int] = set()
        for token in tokens:
            if token == "" or token is None:
                continue
            try:
                hour = int(token)
            except Exception:
                continue
            if 0 <= hour <= 23:
                hours.add(hour)
        return frozenset(hours)

    @staticmethod
    def _completed_bars(aggregator: Any, symbol: str, timeframe: str, lookback: int) -> list[Any]:
        getter = getattr(aggregator, "get_bars", None)
        if not callable(getter):
            return []
        bars = list(
            getter(symbol=str(symbol), timeframe=str(timeframe), n=max(lookback + 1, 2)) or []
        )
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

    @staticmethod
    def _bar_hour_utc(bar: Any) -> int | None:
        raw = None
        if isinstance(bar, (tuple, list)) and bar:
            raw = bar[0]
        elif isinstance(bar, dict):
            raw = bar.get("time") or bar.get("datetime")
        if raw is None:
            return None
        if isinstance(raw, datetime):
            value = raw.astimezone(UTC) if raw.tzinfo is not None else raw.replace(tzinfo=UTC)
            return int(value.hour)
        if isinstance(raw, (int, float)):
            ts = float(raw)
            if abs(ts) > 100_000_000_000:
                ts /= 1000.0
            return int(datetime.fromtimestamp(ts, tz=UTC).hour)
        text = str(raw).strip()
        if not text:
            return None
        try:
            value = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None
        value = value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return int(value.hour)

    def _aligned_completed_pairs(self, aggregator: Any) -> list[tuple[Any, Any]]:
        lookback = max(self.hedge_window, self.lookback_window) + self.lookback_window + 4
        x_bars = self._completed_bars(aggregator, self.symbol_x, self.timeframe, lookback)
        y_bars = self._completed_bars(aggregator, self.symbol_y, self.timeframe, lookback)
        if not x_bars or not y_bars:
            return []
        y_by_key = {self._time_key(bar): bar for bar in y_bars if self._time_key(bar)}
        pairs: list[tuple[Any, Any]] = []
        for x_bar in x_bars:
            key = self._time_key(x_bar)
            y_bar = y_by_key.get(key)
            if key and y_bar is not None:
                pairs.append((x_bar, y_bar))
        return pairs

    def _spread_stats(
        self, pairs: list[tuple[Any, Any]]
    ) -> tuple[float, float, float, float, float, float] | None:
        min_pairs = max(self.hedge_window, self.lookback_window) + 1
        if len(pairs) < min_pairs:
            return None
        x_prices: list[float] = []
        y_prices: list[float] = []
        for x_bar, y_bar in pairs:
            close_x = self._close(x_bar)
            close_y = self._close(y_bar)
            if close_x is None or close_y is None or close_x <= 0.0 or close_y <= 0.0:
                return None
            x_prices.append(float(close_x))
            y_prices.append(float(close_y))
        x_values = [math.log(value) for value in x_prices]
        y_values = [math.log(value) for value in y_prices]
        beta = rolling_beta(
            x_values[-(self.hedge_window + 1) : -1],
            y_values[-(self.hedge_window + 1) : -1],
        )
        if beta is None or not (self.min_abs_beta <= abs(beta) <= self.max_abs_beta):
            return None
        corr = rolling_corr(
            x_values[-(self.lookback_window + 1) : -1],
            y_values[-(self.lookback_window + 1) : -1],
        )
        if corr is not None and corr < self.min_correlation:
            return None
        spread_window = [
            xv - (float(beta) * yv)
            for xv, yv in zip(
                x_values[-(self.lookback_window + 1) : -1],
                y_values[-(self.lookback_window + 1) : -1],
                strict=False,
            )
        ]
        spread_std = sample_std(spread_window)
        if spread_std is None or spread_std <= 1e-12:
            return None
        current_spread = x_values[-1] - (float(beta) * y_values[-1])
        zscore = (current_spread - mean(spread_window)) / spread_std
        return (
            float(zscore),
            float(beta),
            float(corr) if corr is not None else 0.0,
            float(x_prices[-1]),
            float(y_prices[-1]),
            float(current_spread),
        )

    def _entry_allowed(self, latest_bar: Any) -> bool:
        hour = self._bar_hour_utc(latest_bar)
        if self.entry_hours_utc and (hour is None or hour not in self.entry_hours_utc):
            return False
        if (
            self.excluded_entry_hours_utc
            and hour is not None
            and hour in self.excluded_entry_hours_utc
        ):
            return False
        if self.min_entry_volume_x > 0.0:
            volume_x = safe_float(self.bars.get_latest_bar_value(self.symbol_x, "volume"))
            if volume_x is None or volume_x < self.min_entry_volume_x:
                return False
        if self.min_entry_volume_y > 0.0:
            volume_y = safe_float(self.bars.get_latest_bar_value(self.symbol_y, "volume"))
            if volume_y is None or volume_y < self.min_entry_volume_y:
                return False
        return True

    def _metadata(
        self,
        *,
        mode: str,
        reason: str,
        zscore: float,
        beta: float,
        corr: float,
        include_sizing: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "strategy": "TimeframePairZScoreReversionStrategy",
            "pair_mode": mode,
            "reason": reason,
            "symbol_x": self.symbol_x,
            "symbol_y": self.symbol_y,
            "timeframe": self.timeframe,
            "lookback_window": int(self.lookback_window),
            "hedge_window": int(self.hedge_window),
            "entry_z": float(self.entry_z),
            "exit_z": float(self.exit_z),
            "min_entry_volume_x": float(self.min_entry_volume_x),
            "min_entry_volume_y": float(self.min_entry_volume_y),
            "zscore": float(zscore),
            "hedge_ratio": float(beta),
            "correlation": float(corr),
        }
        if include_sizing:
            payload.update(
                {
                    "target_allocation": float(self.target_allocation),
                    "max_symbol_exposure_pct": float(self.target_allocation),
                    "max_order_value": float(self.max_order_value),
                }
            )
        return payload

    def _leg_stop(self, signal_type: str, price: float) -> float | None:
        if self.stop_loss_pct <= 0.0 or price <= 0.0:
            return None
        if signal_type == "LONG":
            return price * (1.0 - self.stop_loss_pct)
        if signal_type == "SHORT":
            return price * (1.0 + self.stop_loss_pct)
        return None

    def _leg_take_profit(self, signal_type: str, price: float) -> float | None:
        if self.take_profit_pct <= 0.0 or price <= 0.0:
            return None
        if signal_type == "LONG":
            return price * (1.0 + self.take_profit_pct)
        if signal_type == "SHORT":
            return price * (1.0 - self.take_profit_pct)
        return None

    def _emit_leg(
        self,
        *,
        symbol: str,
        event_time: Any,
        signal_type: str,
        price: float,
        metadata: dict[str, Any],
    ) -> None:
        self.events.put(
            SignalEvent(
                strategy_id="timeframe_pair_zscore_reversion",
                symbol=symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=float(self.target_allocation if signal_type in {"LONG", "SHORT"} else 1.0),
                price=price,
                stop_loss=self._leg_stop(signal_type, price),
                take_profit=self._leg_take_profit(signal_type, price),
                metadata=metadata,
            )
        )

    def _emit_entry(
        self,
        *,
        event_time: Any,
        mode: str,
        zscore: float,
        beta: float,
        corr: float,
        close_x: float,
        close_y: float,
    ) -> None:
        metadata = self._metadata(
            mode=mode,
            reason="zscore_spread_reversion_entry",
            zscore=zscore,
            beta=beta,
            corr=corr,
            include_sizing=True,
        )
        if mode == "LONG_SPREAD":
            x_signal, y_signal = "LONG", "SHORT"
        else:
            x_signal, y_signal = "SHORT", "LONG"
        self._emit_leg(
            symbol=self.symbol_x,
            event_time=event_time,
            signal_type=x_signal,
            price=close_x,
            metadata=metadata,
        )
        self._emit_leg(
            symbol=self.symbol_y,
            event_time=event_time,
            signal_type=y_signal,
            price=close_y,
            metadata=metadata,
        )
        self._state.mode = mode
        self._state.entry_x_price = close_x
        self._state.entry_y_price = close_y
        self._state.entry_beta = beta
        self._state.bars_held = 0

    def _emit_exit(
        self,
        *,
        event_time: Any,
        reason: str,
        zscore: float,
        beta: float,
        corr: float,
        close_x: float,
        close_y: float,
    ) -> None:
        metadata = self._metadata(
            mode=self._state.mode,
            reason=reason,
            zscore=zscore,
            beta=beta,
            corr=corr,
            include_sizing=False,
        )
        self._emit_leg(
            symbol=self.symbol_x,
            event_time=event_time,
            signal_type="EXIT",
            price=close_x,
            metadata=metadata,
        )
        self._emit_leg(
            symbol=self.symbol_y,
            event_time=event_time,
            signal_type="EXIT",
            price=close_y,
            metadata=metadata,
        )
        self._state = _PairPositionState(last_completed_bar_key=self._state.last_completed_bar_key)

    def _pair_pnl_pct(self, close_x: float, close_y: float) -> float:
        if (
            self._state.entry_x_price is None
            or self._state.entry_y_price is None
            or self._state.entry_beta is None
            or self._state.entry_x_price <= 0.0
            or self._state.entry_y_price <= 0.0
            or close_x <= 0.0
            or close_y <= 0.0
        ):
            return 0.0
        raw = math.log(close_x / self._state.entry_x_price) - float(
            self._state.entry_beta
        ) * math.log(close_y / self._state.entry_y_price)
        normalized = raw / max(1e-9, 1.0 + abs(float(self._state.entry_beta)))
        return normalized if self._state.mode == "LONG_SPREAD" else -normalized

    def _maybe_exit(
        self,
        *,
        event_time: Any,
        zscore: float,
        beta: float,
        corr: float,
        close_x: float,
        close_y: float,
    ) -> bool:
        if self._state.mode == "OUT":
            return False
        self._state.bars_held += 1
        pair_pnl_pct = self._pair_pnl_pct(close_x, close_y)
        reason = ""
        if abs(zscore) <= self.exit_z:
            reason = "mean_reversion_exit"
        elif abs(zscore) >= self.stop_z:
            reason = "zscore_stop_exit"
        elif self.stop_loss_pct > 0.0 and pair_pnl_pct <= -self.stop_loss_pct:
            reason = "pair_stop_loss_exit"
        elif self.take_profit_pct > 0.0 and pair_pnl_pct >= self.take_profit_pct:
            reason = "pair_take_profit_exit"
        elif self._state.bars_held >= self.max_hold_bars:
            reason = "max_hold_exit"
        elif corr < (self.min_correlation * 0.5):
            reason = "correlation_break_exit"
        if not reason:
            return True
        self._emit_exit(
            event_time=event_time,
            reason=reason,
            zscore=zscore,
            beta=beta,
            corr=corr,
            close_x=close_x,
            close_y=close_y,
        )
        return True

    def calculate_signals(self, event: Any) -> None:
        _ = event
        return

    def calculate_signals_window(self, event: Any, aggregator: Any) -> None:
        if aggregator is None:
            return
        pairs = self._aligned_completed_pairs(aggregator)
        stats = self._spread_stats(pairs)
        if stats is None or not pairs:
            return
        latest_x_bar, _ = pairs[-1]
        completed_key = self._time_key(latest_x_bar)
        if not completed_key or completed_key == self._state.last_completed_bar_key:
            return
        self._state.last_completed_bar_key = completed_key

        zscore, beta, corr, close_x, close_y, _spread = stats
        event_time = (
            latest_x_bar[0]
            if isinstance(latest_x_bar, (tuple, list))
            else getattr(event, "time", None)
        )
        if self._maybe_exit(
            event_time=event_time,
            zscore=zscore,
            beta=beta,
            corr=corr,
            close_x=close_x,
            close_y=close_y,
        ):
            return
        if self._state.mode != "OUT" or not self._entry_allowed(latest_x_bar):
            return
        if zscore <= -self.entry_z and self.allow_long_spread:
            self._emit_entry(
                event_time=event_time,
                mode="LONG_SPREAD",
                zscore=zscore,
                beta=beta,
                corr=corr,
                close_x=close_x,
                close_y=close_y,
            )
        elif zscore >= self.entry_z and self.allow_short_spread:
            self._emit_entry(
                event_time=event_time,
                mode="SHORT_SPREAD",
                zscore=zscore,
                beta=beta,
                corr=corr,
                close_x=close_x,
                close_y=close_y,
            )
