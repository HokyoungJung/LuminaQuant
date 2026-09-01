"""Hourly single-asset shock mean-reversion strategy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math
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
    cooldown_remaining: int = 0


class HourlyShockReversionStrategy(Strategy):
    """Fade a large completed hourly move in one liquid crypto symbol."""

    uses_timeframe_aggregator = True
    preferred_contract = "context"
    required_timeframes = ("1h",)
    required_lookbacks = {"1h": 128}

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "target_symbol": HyperParam.string("target_symbol", default="ETH/USDT", tunable=False),
            "timeframe": HyperParam.string("timeframe", default="1h", tunable=False),
            "lookback_bars": HyperParam.integer("lookback_bars", default=4, low=1, high=240),
            "return_threshold": HyperParam.floating(
                "return_threshold", default=0.006, low=0.0, high=1.0
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=48, low=1, high=240),
            "cooldown_bars": HyperParam.integer(
                "cooldown_bars", default=0, low=0, high=240, tunable=False
            ),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.008, low=0.0, high=1.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=175.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.02, low=0.0, high=1.0, tunable=False
            ),
            "take_profit_pct": HyperParam.floating(
                "take_profit_pct", default=0.0, low=0.0, high=1.0, tunable=False
            ),
            "allow_long": HyperParam.boolean("allow_long", default=True, grid=[True, False]),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "entry_hours_utc": HyperParam.string("entry_hours_utc", default="", tunable=False),
            "excluded_entry_hours_utc": HyperParam.string(
                "excluded_entry_hours_utc", default="", tunable=False
            ),
            "regime_symbol": HyperParam.string("regime_symbol", default="", tunable=False),
            "regime_lookback_bars": HyperParam.integer(
                "regime_lookback_bars", default=24, low=1, high=240, tunable=False
            ),
            "counterguard_return_threshold": HyperParam.floating(
                "counterguard_return_threshold", default=0.0, low=0.0, high=1.0, tunable=False
            ),
            "volatility_lookback_bars": HyperParam.integer(
                "volatility_lookback_bars", default=0, low=0, high=240, tunable=False
            ),
            "max_realized_volatility": HyperParam.floating(
                "max_realized_volatility", default=0.0, low=0.0, high=1.0, tunable=False
            ),
            "flow_confirmation_lookback_bars": HyperParam.integer(
                "flow_confirmation_lookback_bars", default=0, low=0, high=24, tunable=False
            ),
            "flow_imbalance_min": HyperParam.floating(
                "flow_imbalance_min", default=0.0, low=0.0, high=1.0, tunable=False
            ),
        }

    def __init__(
        self,
        bars: Any,
        events: Any,
        target_symbol: str = "ETH/USDT",
        timeframe: str = "1h",
        lookback_bars: int = 4,
        return_threshold: float = 0.006,
        max_hold_bars: int = 48,
        cooldown_bars: int = 0,
        target_allocation: float = 0.008,
        max_order_value: float = 175.0,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.0,
        allow_long: bool = True,
        allow_short: bool = True,
        entry_hours_utc: str | list[int] | tuple[int, ...] = "",
        excluded_entry_hours_utc: str | list[int] | tuple[int, ...] = "",
        regime_symbol: str = "",
        regime_lookback_bars: int = 24,
        counterguard_return_threshold: float = 0.0,
        volatility_lookback_bars: int = 0,
        max_realized_volatility: float = 0.0,
        flow_confirmation_lookback_bars: int = 0,
        flow_imbalance_min: float = 0.0,
    ) -> None:
        self.bars = bars
        self.events = events
        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "target_symbol": target_symbol,
                "timeframe": timeframe,
                "lookback_bars": lookback_bars,
                "return_threshold": return_threshold,
                "max_hold_bars": max_hold_bars,
                "cooldown_bars": cooldown_bars,
                "target_allocation": target_allocation,
                "max_order_value": max_order_value,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "allow_long": allow_long,
                "allow_short": allow_short,
                "entry_hours_utc": entry_hours_utc,
                "excluded_entry_hours_utc": excluded_entry_hours_utc,
                "regime_symbol": regime_symbol,
                "regime_lookback_bars": regime_lookback_bars,
                "counterguard_return_threshold": counterguard_return_threshold,
                "volatility_lookback_bars": volatility_lookback_bars,
                "max_realized_volatility": max_realized_volatility,
                "flow_confirmation_lookback_bars": flow_confirmation_lookback_bars,
                "flow_imbalance_min": flow_imbalance_min,
            },
            keep_unknown=False,
        )
        self.target_symbol = canonical_symbol(str(resolved["target_symbol"]))
        self.timeframe = str(resolved["timeframe"] or "1h")
        self.lookback_bars = max(1, int(resolved["lookback_bars"]))
        self.return_threshold = max(0.0, float(resolved["return_threshold"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.cooldown_bars = max(0, int(resolved["cooldown_bars"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.take_profit_pct = max(0.0, float(resolved["take_profit_pct"]))
        self.allow_long = bool(resolved["allow_long"])
        self.allow_short = bool(resolved["allow_short"])
        self.entry_hours_utc = self._parse_hours(resolved["entry_hours_utc"])
        self.excluded_entry_hours_utc = self._parse_hours(resolved["excluded_entry_hours_utc"])
        raw_regime_symbol = str(resolved["regime_symbol"] or "").strip()
        self.regime_symbol = canonical_symbol(raw_regime_symbol) if raw_regime_symbol else ""
        self.regime_lookback_bars = max(1, int(resolved["regime_lookback_bars"]))
        self.counterguard_return_threshold = max(
            0.0, float(resolved["counterguard_return_threshold"])
        )
        self.volatility_lookback_bars = max(0, int(resolved["volatility_lookback_bars"]))
        self.max_realized_volatility = max(0.0, float(resolved["max_realized_volatility"]))
        self.flow_confirmation_lookback_bars = max(
            0, int(resolved["flow_confirmation_lookback_bars"])
        )
        self.flow_imbalance_min = max(0.0, float(resolved["flow_imbalance_min"]))
        self._state = _PositionState()

    @property
    def required_features(self) -> tuple[str, ...]:
        if self.flow_confirmation_lookback_bars > 0 and self.flow_imbalance_min > 0.0:
            return (
                "taker_buy_quote_volume",
                "taker_sell_quote_volume",
                "taker_buy_base_volume",
                "taker_sell_base_volume",
            )
        return ()

    def get_state(self) -> dict[str, Any]:
        return {
            "mode": self._state.mode,
            "entry_price": self._state.entry_price,
            "bars_held": int(self._state.bars_held),
            "last_completed_bar_key": self._state.last_completed_bar_key,
            "cooldown_remaining": int(self._state.cooldown_remaining),
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
        try:
            self._state.cooldown_remaining = max(0, int(state.get("cooldown_remaining", 0)))
        except Exception:
            self._state.cooldown_remaining = 0

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
    def _bar_hour_utc(bar: Any) -> int | None:
        raw = None
        if isinstance(bar, (tuple, list)) and bar:
            raw = bar[0]
        elif isinstance(bar, dict):
            raw = bar.get("time") or bar.get("datetime")
        if raw is None:
            return None
        if isinstance(raw, datetime):
            value = raw
            if value.tzinfo is not None:
                value = value.astimezone(UTC)
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
        if value.tzinfo is not None:
            value = value.astimezone(UTC)
        return int(value.hour)

    @staticmethod
    def _bar_start_ms(bar: Any) -> int | None:
        raw = None
        if isinstance(bar, (tuple, list)) and bar:
            raw = bar[0]
        elif isinstance(bar, dict):
            raw = bar.get("time") or bar.get("datetime")
        if raw is None:
            return None
        if isinstance(raw, datetime):
            value = raw.astimezone(UTC) if raw.tzinfo is not None else raw.replace(tzinfo=UTC)
            return int(value.timestamp() * 1000)
        if isinstance(raw, (int, float)):
            ts = float(raw)
            return int(ts if abs(ts) > 100_000_000_000 else ts * 1000)
        text = str(raw).strip()
        if not text:
            return None
        try:
            value = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None
        value = value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return int(value.timestamp() * 1000)

    @staticmethod
    def _realized_volatility(bars: list[Any], lookback: int) -> float:
        if lookback <= 1 or len(bars) < 3:
            return 0.0
        subset = bars[-max(2, lookback) :]
        returns: list[float] = []
        previous = HourlyShockReversionStrategy._close(subset[0])
        for bar in subset[1:]:
            current = HourlyShockReversionStrategy._close(bar)
            if previous is not None and current is not None and previous > 0.0 and current > 0.0:
                returns.append(math.log(float(current) / float(previous)))
            previous = current
        if not returns:
            return 0.0
        return math.sqrt(sum(value * value for value in returns) / len(returns))

    def _flow_confirmation_pass(
        self,
        *,
        feature_lookup: Any,
        latest_bar: Any,
        signal_type: str,
        latest_close: float,
    ) -> tuple[bool, dict[str, Any]]:
        if self.flow_confirmation_lookback_bars <= 0 or self.flow_imbalance_min <= 0.0:
            return True, {}
        metadata: dict[str, Any] = {
            "flow_confirmation_lookback_bars": int(self.flow_confirmation_lookback_bars),
            "flow_imbalance_min": float(self.flow_imbalance_min),
        }
        bar_start_ms = self._bar_start_ms(latest_bar)
        if bar_start_ms is None:
            metadata["filter_reject"] = "flow_bar_time_missing"
            return False, metadata
        window_start = int(bar_start_ms) - (self.flow_confirmation_lookback_bars - 1) * 3_600_000
        window_stop = int(bar_start_ms) + 3_600_000 - 1
        sum_between = getattr(feature_lookup, "sum_between", None)
        if not callable(sum_between):
            metadata["filter_reject"] = "flow_feature_lookup_missing"
            return False, metadata

        buy_quote = safe_float(
            sum_between(
                self.target_symbol,
                "taker_buy_quote_volume",
                start_timestamp_ms=window_start,
                end_timestamp_ms=window_stop,
            )
        )
        sell_quote = safe_float(
            sum_between(
                self.target_symbol,
                "taker_sell_quote_volume",
                start_timestamp_ms=window_start,
                end_timestamp_ms=window_stop,
            )
        )
        if buy_quote is None or sell_quote is None or buy_quote + sell_quote <= 0.0:
            buy_base = safe_float(
                sum_between(
                    self.target_symbol,
                    "taker_buy_base_volume",
                    start_timestamp_ms=window_start,
                    end_timestamp_ms=window_stop,
                )
            )
            sell_base = safe_float(
                sum_between(
                    self.target_symbol,
                    "taker_sell_base_volume",
                    start_timestamp_ms=window_start,
                    end_timestamp_ms=window_stop,
                )
            )
            if buy_base is not None and sell_base is not None and buy_base + sell_base > 0.0:
                buy_quote = buy_base * float(latest_close)
                sell_quote = sell_base * float(latest_close)
                metadata["flow_source"] = "base_volume"
        else:
            metadata["flow_source"] = "quote_volume"

        if buy_quote is None or sell_quote is None or buy_quote + sell_quote <= 0.0:
            metadata["filter_reject"] = "flow_feature_missing"
            return False, metadata

        imbalance = float((buy_quote - sell_quote) / (buy_quote + sell_quote))
        metadata["flow_imbalance"] = imbalance
        metadata["flow_window_start_ms"] = int(window_start)
        metadata["flow_window_end_ms"] = int(window_stop)
        if signal_type == "LONG" and imbalance > -self.flow_imbalance_min:
            metadata["filter_reject"] = "taker_sell_exhaustion_missing"
            return False, metadata
        if signal_type == "SHORT" and imbalance < self.flow_imbalance_min:
            metadata["filter_reject"] = "taker_buy_exhaustion_missing"
            return False, metadata
        return True, metadata

    def _entry_filters_pass(
        self,
        *,
        aggregator: Any,
        latest_bar: Any,
        signal_type: str,
        target_bars: list[Any],
        latest_close: float,
        feature_lookup: Any = None,
    ) -> tuple[bool, dict[str, Any]]:
        metadata: dict[str, Any] = {}
        hour = self._bar_hour_utc(latest_bar)
        if self.entry_hours_utc:
            metadata["entry_hours_utc"] = sorted(self.entry_hours_utc)
            if hour is None or hour not in self.entry_hours_utc:
                metadata["filter_reject"] = "entry_hour_not_allowed"
                metadata["entry_hour_utc"] = hour
                return False, metadata
        if self.excluded_entry_hours_utc:
            metadata["excluded_entry_hours_utc"] = sorted(self.excluded_entry_hours_utc)
            if hour is not None and hour in self.excluded_entry_hours_utc:
                metadata["filter_reject"] = "entry_hour_excluded"
                metadata["entry_hour_utc"] = hour
                return False, metadata
        if hour is not None:
            metadata["entry_hour_utc"] = hour

        if self.max_realized_volatility > 0.0 and self.volatility_lookback_bars > 1:
            realized_volatility = self._realized_volatility(
                target_bars, self.volatility_lookback_bars
            )
            metadata["realized_volatility"] = float(realized_volatility)
            metadata["max_realized_volatility"] = float(self.max_realized_volatility)
            if realized_volatility > self.max_realized_volatility:
                metadata["filter_reject"] = "realized_volatility_too_high"
                return False, metadata

        if self.regime_symbol and self.counterguard_return_threshold > 0.0:
            regime_bars = self._completed_bars(
                aggregator,
                self.regime_symbol,
                self.timeframe,
                self.regime_lookback_bars + 2,
            )
            if len(regime_bars) <= self.regime_lookback_bars:
                metadata["filter_reject"] = "regime_history_missing"
                return False, metadata
            latest = self._close(regime_bars[-1])
            base = self._close(regime_bars[-1 - self.regime_lookback_bars])
            if latest is None or base is None or latest <= 0.0 or base <= 0.0:
                metadata["filter_reject"] = "regime_price_missing"
                return False, metadata
            regime_return = float(latest / base - 1.0)
            metadata["regime_symbol"] = self.regime_symbol
            metadata["regime_lookback_bars"] = int(self.regime_lookback_bars)
            metadata["regime_return"] = regime_return
            metadata["counterguard_return_threshold"] = float(self.counterguard_return_threshold)
            if signal_type == "LONG" and regime_return <= -self.counterguard_return_threshold:
                metadata["filter_reject"] = "counterguard_downtrend"
                return False, metadata
            if signal_type == "SHORT" and regime_return >= self.counterguard_return_threshold:
                metadata["filter_reject"] = "counterguard_uptrend"
                return False, metadata

        if self.flow_confirmation_lookback_bars > 0 and self.flow_imbalance_min > 0.0:
            passed, flow_metadata = self._flow_confirmation_pass(
                feature_lookup=feature_lookup,
                latest_bar=latest_bar,
                signal_type=signal_type,
                latest_close=float(latest_close),
            )
            metadata.update(flow_metadata)
            if not passed:
                return False, metadata

        return True, metadata

    def _metadata(
        self,
        *,
        shock_return: float,
        reason: str,
        filter_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "strategy": "HourlyShockReversionStrategy",
            "reason": reason,
            "target_symbol": self.target_symbol,
            "timeframe": self.timeframe,
            "lookback_bars": int(self.lookback_bars),
            "shock_return": float(shock_return),
            "return_threshold": float(self.return_threshold),
            "target_allocation": float(self.target_allocation),
            "max_symbol_exposure_pct": float(self.target_allocation),
            "max_order_value": float(self.max_order_value),
            **dict(filter_metadata or {}),
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
                strategy_id="hourly_shock_reversion",
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

    def _maybe_exit(self, *, event_time: Any, price: float, shock_return: float) -> bool:
        if self._state.mode == "OUT":
            return False
        self._state.bars_held += 1
        stop_hit = False
        take_profit_hit = False
        if self._state.entry_price:
            if self._state.mode == "LONG":
                stop_hit = self.stop_loss_pct > 0.0 and price <= self._state.entry_price * (
                    1.0 - self.stop_loss_pct
                )
                take_profit_hit = (
                    self.take_profit_pct > 0.0
                    and price >= self._state.entry_price * (1.0 + self.take_profit_pct)
                )
            else:
                stop_hit = self.stop_loss_pct > 0.0 and price >= self._state.entry_price * (
                    1.0 + self.stop_loss_pct
                )
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
        metadata = self._metadata(shock_return=shock_return, reason=reason)
        metadata.pop("target_allocation", None)
        metadata.pop("max_symbol_exposure_pct", None)
        metadata.pop("max_order_value", None)
        self._emit(event_time=event_time, signal_type="EXIT", price=price, metadata=metadata)
        self._state.mode = "OUT"
        self._state.entry_price = None
        self._state.bars_held = 0
        self._state.cooldown_remaining = int(self.cooldown_bars)
        return True

    def calculate_signals(self, event: Any) -> None:
        _ = event
        return

    def calculate_signals_context(self, context: Any) -> None:
        self.calculate_signals_window(
            getattr(context, "event", None),
            getattr(context, "aggregator", None),
            feature_lookup=getattr(context, "feature_lookup", None),
        )

    def calculate_signals_window(
        self, event: Any, aggregator: Any, feature_lookup: Any = None
    ) -> None:
        if aggregator is None:
            return
        bars = self._completed_bars(
            aggregator, self.target_symbol, self.timeframe, self.lookback_bars + 2
        )
        if len(bars) <= self.lookback_bars:
            return
        latest_bar = bars[-1]
        completed_key = self._time_key(latest_bar)
        if not completed_key or completed_key == self._state.last_completed_bar_key:
            return
        self._state.last_completed_bar_key = completed_key

        latest_close = self._close(latest_bar)
        base_close = self._close(bars[-1 - self.lookback_bars])
        if latest_close is None or base_close is None or latest_close <= 0.0 or base_close <= 0.0:
            return

        event_time = (
            latest_bar[0] if isinstance(latest_bar, (tuple, list)) else getattr(event, "time", None)
        )
        shock_return = float(latest_close / base_close - 1.0)
        if self._maybe_exit(
            event_time=event_time, price=float(latest_close), shock_return=shock_return
        ):
            return
        if self._state.mode != "OUT":
            return
        if self._state.cooldown_remaining > 0:
            self._state.cooldown_remaining -= 1
            return

        if shock_return >= self.return_threshold:
            if not self.allow_short:
                return
            signal_type = "SHORT"
            reason = "positive_shock_reversion_short"
        elif shock_return <= -self.return_threshold:
            if not self.allow_long:
                return
            signal_type = "LONG"
            reason = "negative_shock_reversion_long"
        else:
            return

        filters_pass, filter_metadata = self._entry_filters_pass(
            aggregator=aggregator,
            latest_bar=latest_bar,
            signal_type=signal_type,
            target_bars=bars,
            latest_close=float(latest_close),
            feature_lookup=feature_lookup,
        )
        if not filters_pass:
            return

        self._emit(
            event_time=event_time,
            signal_type=signal_type,
            price=float(latest_close),
            metadata=self._metadata(
                shock_return=shock_return,
                reason=reason,
                filter_metadata=filter_metadata,
            ),
        )
        self._state.mode = signal_type
        self._state.entry_price = float(latest_close)
        self._state.bars_held = 0
