"""Taker-flow exhaustion reversal alpha with explicit risk filters.

This strategy is deliberately separate from the broader derivatives-flow squeeze
family.  It trades only when aggressive taker flow and price extension point in
the same direction, then fades that move under conservative funding, session,
and realized-volatility gates.  The thesis is order-flow exhaustion rather than
extra exposure.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
import math
from typing import Any

from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@dataclass(slots=True)
class _FlowState:
    closes: deque[float]
    taker_buy_quote_volume: deque[float]
    taker_sell_quote_volume: deque[float]
    funding_rate: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    high_watermark: float | None = None
    low_watermark: float | None = None
    bars_held: int = 0
    cooldown_remaining: int = 0
    ticks_seen: int = 0
    last_time_key: str = ""
    has_taker_flow: bool = False
    has_funding: bool = False
    last_flow_source: str = "missing"


class TakerFlowExhaustionReversalStrategy(Strategy):
    """Fade extreme taker-flow/price extensions with funding and vol guards."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False
    required_features = (
        "funding_rate",
        "taker_buy_quote_volume",
        "taker_sell_quote_volume",
    )

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "target_symbol": HyperParam.string("target_symbol", default="ETH/USDT", tunable=False),
            "flow_lookback_bars": HyperParam.integer(
                "flow_lookback_bars", default=90, low=4, high=2_880
            ),
            "momentum_lookback_bars": HyperParam.integer(
                "momentum_lookback_bars", default=180, low=4, high=2_880
            ),
            "volatility_lookback_bars": HyperParam.integer(
                "volatility_lookback_bars", default=180, low=4, high=2_880
            ),
            "evaluation_cadence_bars": HyperParam.integer(
                "evaluation_cadence_bars", default=180, low=1, high=10_080, tunable=False
            ),
            "flow_imbalance_min": HyperParam.floating(
                "flow_imbalance_min", default=0.14, low=0.0, high=1.0
            ),
            "price_extension_min": HyperParam.floating(
                "price_extension_min", default=0.006, low=0.0, high=0.50
            ),
            "funding_abs_cap": HyperParam.floating(
                "funding_abs_cap", default=0.00015, low=0.0, high=0.10
            ),
            "max_realized_volatility": HyperParam.floating(
                "max_realized_volatility", default=0.008, low=0.0, high=1.0
            ),
            "entry_hours_utc": HyperParam.string(
                "entry_hours_utc", default="13,14,15,16,17,18,19,20", tunable=False
            ),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.008, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=175.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "volatility_target_per_bar": HyperParam.floating(
                "volatility_target_per_bar", default=0.0011, low=0.0, high=0.10
            ),
            "min_volatility_multiplier": HyperParam.floating(
                "min_volatility_multiplier", default=0.35, low=0.0, high=5.0
            ),
            "max_volatility_multiplier": HyperParam.floating(
                "max_volatility_multiplier", default=1.0, low=0.0, high=5.0
            ),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.025, low=0.0, high=0.5),
            "take_profit_pct": HyperParam.floating(
                "take_profit_pct", default=0.050, low=0.0, high=1.0
            ),
            "trailing_exit_pct": HyperParam.floating(
                "trailing_exit_pct", default=0.018, low=0.0, high=0.5
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=2160, low=1, high=100_000),
            "cooldown_bars": HyperParam.integer(
                "cooldown_bars", default=0, low=0, high=100_000, tunable=False
            ),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(
        self,
        bars,
        events,
        target_symbol: str = "ETH/USDT",
        flow_lookback_bars: int = 90,
        momentum_lookback_bars: int = 180,
        volatility_lookback_bars: int = 180,
        evaluation_cadence_bars: int = 180,
        flow_imbalance_min: float = 0.14,
        price_extension_min: float = 0.006,
        funding_abs_cap: float = 0.00015,
        max_realized_volatility: float = 0.008,
        entry_hours_utc: str | list[int] | tuple[int, ...] = "13,14,15,16,17,18,19,20",
        target_allocation: float = 0.008,
        max_order_value: float = 175.0,
        volatility_target_per_bar: float = 0.0011,
        min_volatility_multiplier: float = 0.35,
        max_volatility_multiplier: float = 1.0,
        stop_loss_pct: float = 0.025,
        take_profit_pct: float = 0.050,
        trailing_exit_pct: float = 0.018,
        max_hold_bars: int = 2160,
        cooldown_bars: int = 0,
        allow_short: bool = True,
        min_price: float = 0.10,
    ) -> None:
        self.bars = bars
        self.events = events
        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "target_symbol": target_symbol,
                "flow_lookback_bars": flow_lookback_bars,
                "momentum_lookback_bars": momentum_lookback_bars,
                "volatility_lookback_bars": volatility_lookback_bars,
                "evaluation_cadence_bars": evaluation_cadence_bars,
                "flow_imbalance_min": flow_imbalance_min,
                "price_extension_min": price_extension_min,
                "funding_abs_cap": funding_abs_cap,
                "max_realized_volatility": max_realized_volatility,
                "entry_hours_utc": entry_hours_utc,
                "target_allocation": target_allocation,
                "max_order_value": max_order_value,
                "volatility_target_per_bar": volatility_target_per_bar,
                "min_volatility_multiplier": min_volatility_multiplier,
                "max_volatility_multiplier": max_volatility_multiplier,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "trailing_exit_pct": trailing_exit_pct,
                "max_hold_bars": max_hold_bars,
                "cooldown_bars": cooldown_bars,
                "allow_short": allow_short,
                "min_price": min_price,
            },
            keep_unknown=False,
        )
        self.target_symbol = str(resolved["target_symbol"] or "ETH/USDT")
        self.flow_lookback_bars = max(1, int(resolved["flow_lookback_bars"]))
        self.momentum_lookback_bars = max(1, int(resolved["momentum_lookback_bars"]))
        self.volatility_lookback_bars = max(2, int(resolved["volatility_lookback_bars"]))
        self.evaluation_cadence_bars = max(1, int(resolved["evaluation_cadence_bars"]))
        self.flow_imbalance_min = max(0.0, float(resolved["flow_imbalance_min"]))
        self.price_extension_min = max(0.0, float(resolved["price_extension_min"]))
        self.funding_abs_cap = max(0.0, float(resolved["funding_abs_cap"]))
        self.max_realized_volatility = max(0.0, float(resolved["max_realized_volatility"]))
        self.entry_hours_utc = self._parse_hours(resolved["entry_hours_utc"])
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.volatility_target_per_bar = max(0.0, float(resolved["volatility_target_per_bar"]))
        self.min_volatility_multiplier = max(0.0, float(resolved["min_volatility_multiplier"]))
        self.max_volatility_multiplier = max(
            self.min_volatility_multiplier, float(resolved["max_volatility_multiplier"])
        )
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.take_profit_pct = max(0.0, float(resolved["take_profit_pct"]))
        self.trailing_exit_pct = max(0.0, float(resolved["trailing_exit_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.cooldown_bars = max(0, int(resolved["cooldown_bars"]))
        self.allow_short = bool(resolved["allow_short"])
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = (
            max(
                self.flow_lookback_bars,
                self.momentum_lookback_bars,
                self.volatility_lookback_bars,
                self.max_hold_bars,
            )
            + 8
        )
        self._state = _FlowState(
            closes=deque(maxlen=size),
            taker_buy_quote_volume=deque(maxlen=size),
            taker_sell_quote_volume=deque(maxlen=size),
            funding_rate=deque(maxlen=size),
        )

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
    def _hour_utc(raw: Any) -> int | None:
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
        if value.tzinfo is not None:
            value = value.astimezone(UTC)
        return int(value.hour)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    @staticmethod
    def _as_row_dict(row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return dict(row)
        if isinstance(row, (tuple, list)):
            keys = ("timestamp_ms", "open", "high", "low", "close", "volume")
            return {key: row[idx] for idx, key in enumerate(keys) if idx < len(row)}
        return {}

    def _window_snapshot(self, event: Any) -> dict[str, float | None]:
        bars_1s = dict(getattr(event, "bars_1s", {}) or {})
        rows = list(bars_1s.get(self.target_symbol) or [])
        if not rows:
            return {}
        mapped = [self._as_row_dict(row) for row in rows]
        mapped = [row for row in mapped if row]
        if not mapped:
            return {}
        last = mapped[-1]

        def _sum_field(field: str) -> float | None:
            values = [value for row in mapped if (value := safe_float(row.get(field))) is not None]
            if not values:
                return None
            return float(sum(values))

        return {
            "close": safe_float(last.get("close")),
            "taker_buy_quote_volume": _sum_field("taker_buy_quote_volume"),
            "taker_sell_quote_volume": _sum_field("taker_sell_quote_volume"),
            "taker_buy_base_volume": _sum_field("taker_buy_base_volume"),
            "taker_sell_base_volume": _sum_field("taker_sell_base_volume"),
        }

    def _extract_feature(self, event: Any, field: str) -> float | None:
        direct = safe_float(getattr(event, field, None))
        if direct is not None:
            return direct
        getter = getattr(self.bars, "get_latest_feature_value", None)
        if callable(getter):
            try:
                return safe_float(getter(self.target_symbol, field))
            except Exception:
                return None
        getter = getattr(self.bars, "get_latest_bar_value", None)
        if callable(getter):
            try:
                return safe_float(getter(self.target_symbol, field))
            except Exception:
                return None
        return None

    def _resolve_flow_values(
        self, event: Any, snapshot: dict[str, float | None], close: float
    ) -> tuple[float, float, str]:
        buy_quote = self._extract_feature(event, "taker_buy_quote_volume")
        sell_quote = self._extract_feature(event, "taker_sell_quote_volume")
        if buy_quote is None:
            buy_quote = snapshot.get("taker_buy_quote_volume")
        if sell_quote is None:
            sell_quote = snapshot.get("taker_sell_quote_volume")
        if buy_quote is not None and sell_quote is not None and buy_quote + sell_quote > 0.0:
            return max(0.0, float(buy_quote)), max(0.0, float(sell_quote)), "feature_quote"
        buy_base = self._extract_feature(event, "taker_buy_base_volume")
        sell_base = self._extract_feature(event, "taker_sell_base_volume")
        if buy_base is None:
            buy_base = snapshot.get("taker_buy_base_volume")
        if sell_base is None:
            sell_base = snapshot.get("taker_sell_base_volume")
        if buy_base is not None and sell_base is not None and buy_base + sell_base > 0.0:
            return (
                max(0.0, float(buy_base) * close),
                max(0.0, float(sell_base) * close),
                "feature_base",
            )
        return 0.0, 0.0, "missing"

    @staticmethod
    def _pct_change(values: deque[float], lookback: int) -> float | None:
        if len(values) <= lookback:
            return None
        latest = float(values[-1])
        base = float(values[-1 - lookback])
        if abs(base) <= 1e-12:
            return None
        return latest / base - 1.0

    @staticmethod
    def _realized_vol(values: deque[float], lookback: int) -> float:
        if len(values) < 3:
            return 0.0
        subset = list(values)[-max(3, lookback) :]
        returns: list[float] = []
        previous = subset[0]
        for current in subset[1:]:
            if previous > 0.0 and current > 0.0:
                returns.append(math.log(current / previous))
            previous = current
        if not returns:
            return 0.0
        return math.sqrt(sum(value * value for value in returns) / len(returns)) * math.sqrt(
            len(returns)
        )

    def _flow_imbalance(self) -> float:
        buy = sum(list(self._state.taker_buy_quote_volume)[-self.flow_lookback_bars :])
        sell = sum(list(self._state.taker_sell_quote_volume)[-self.flow_lookback_bars :])
        total = buy + sell
        if total <= 1e-12:
            return 0.0
        return (buy - sell) / total

    def _volatility_multiplier(self, realized_vol: float) -> float:
        if realized_vol <= 1e-12 or self.volatility_target_per_bar <= 0.0:
            return self.max_volatility_multiplier
        return self._clamp(
            self.volatility_target_per_bar / realized_vol,
            self.min_volatility_multiplier,
            self.max_volatility_multiplier,
        )

    def _metadata(
        self,
        *,
        reason: str,
        price_extension: float,
        flow_imbalance: float,
        funding_rate: float,
        realized_volatility: float,
        target_allocation: float,
        vol_multiplier: float,
    ) -> dict[str, Any]:
        return {
            "strategy": "TakerFlowExhaustionReversalStrategy",
            "symbol": self.target_symbol,
            "reason": reason,
            "price_extension": float(price_extension),
            "flow_imbalance": float(flow_imbalance),
            "funding_rate": float(funding_rate),
            "realized_volatility": float(realized_volatility),
            "volatility_multiplier": float(vol_multiplier),
            "target_allocation": float(target_allocation),
            "max_symbol_exposure_pct": float(target_allocation),
            "max_order_value": float(self.max_order_value),
            "feature_coverage": {
                "taker_flow": bool(self._state.has_taker_flow),
                "funding_rate": bool(self._state.has_funding),
                "flow_source": self._state.last_flow_source,
            },
            "bars_held": int(self._state.bars_held),
            "cooldown_remaining": int(self._state.cooldown_remaining),
        }

    def _emit(
        self,
        event_time: Any,
        signal_type: str,
        *,
        strength: float = 1.0,
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        trailing_percent: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.events.put(
            SignalEvent(
                strategy_id="taker_flow_exhaustion_reversal",
                symbol=self.target_symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=float(strength),
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_percent=trailing_percent,
                metadata=dict(metadata or {}),
            )
        )

    def _maybe_exit(self, event: Any, close: float, metrics: dict[str, float]) -> bool:
        state = self._state
        if state.mode not in {"LONG", "SHORT"}:
            return False
        state.bars_held += 1
        if state.mode == "LONG":
            state.high_watermark = max(float(state.high_watermark or close), close)
            stop_hit = (
                state.entry_price is not None
                and self.stop_loss_pct > 0.0
                and close <= state.entry_price * (1.0 - self.stop_loss_pct)
            )
            take_hit = (
                state.entry_price is not None
                and self.take_profit_pct > 0.0
                and close >= state.entry_price * (1.0 + self.take_profit_pct)
            )
            trail_hit = (
                self.trailing_exit_pct > 0.0
                and state.high_watermark is not None
                and close <= state.high_watermark * (1.0 - self.trailing_exit_pct)
            )
            opposing_flow = (
                metrics["flow_imbalance"] >= self.flow_imbalance_min
                and metrics["price_extension"] <= -self.price_extension_min * 0.25
            )
        else:
            state.low_watermark = min(float(state.low_watermark or close), close)
            stop_hit = (
                state.entry_price is not None
                and self.stop_loss_pct > 0.0
                and close >= state.entry_price * (1.0 + self.stop_loss_pct)
            )
            take_hit = (
                state.entry_price is not None
                and self.take_profit_pct > 0.0
                and close <= state.entry_price * (1.0 - self.take_profit_pct)
            )
            trail_hit = (
                self.trailing_exit_pct > 0.0
                and state.low_watermark is not None
                and close >= state.low_watermark * (1.0 + self.trailing_exit_pct)
            )
            opposing_flow = (
                metrics["flow_imbalance"] <= -self.flow_imbalance_min
                and metrics["price_extension"] >= self.price_extension_min * 0.25
            )
        max_hold = state.bars_held >= self.max_hold_bars
        hard_vol = (
            self.max_realized_volatility > 0.0
            and metrics["realized_volatility"] > self.max_realized_volatility * 1.35
        )
        reasons = [
            name
            for name, flag in (
                ("stop", stop_hit),
                ("take_profit", take_hit),
                ("trailing", trail_hit),
                ("max_hold", max_hold),
                ("opposing_flow", opposing_flow),
                ("hard_volatility", hard_vol),
            )
            if flag
        ]
        if not reasons:
            return False
        metadata = self._metadata(
            reason=f"{state.mode.lower()}_exit:{'+'.join(reasons)}",
            price_extension=metrics["price_extension"],
            flow_imbalance=metrics["flow_imbalance"],
            funding_rate=metrics["funding_rate"],
            realized_volatility=metrics["realized_volatility"],
            target_allocation=0.0,
            vol_multiplier=metrics["volatility_multiplier"],
        )
        metadata.pop("target_allocation", None)
        metadata.pop("max_symbol_exposure_pct", None)
        metadata.pop("max_order_value", None)
        self._emit(getattr(event, "time", None), "EXIT", price=close, metadata=metadata)
        state.mode = "OUT"
        state.entry_price = None
        state.high_watermark = None
        state.low_watermark = None
        state.bars_held = 0
        state.cooldown_remaining = self.cooldown_bars
        return True

    def _enter(
        self, event: Any, signal_type: str, close: float, reason: str, metrics: dict[str, float]
    ) -> None:
        state = self._state
        vol_multiplier = metrics["volatility_multiplier"]
        target_allocation = self.target_allocation * vol_multiplier
        if target_allocation <= 0.0 or self.max_order_value <= 0.0:
            return
        metadata = self._metadata(
            reason=reason,
            price_extension=metrics["price_extension"],
            flow_imbalance=metrics["flow_imbalance"],
            funding_rate=metrics["funding_rate"],
            realized_volatility=metrics["realized_volatility"],
            target_allocation=target_allocation,
            vol_multiplier=vol_multiplier,
        )
        if signal_type == "LONG":
            stop_loss = close * (1.0 - self.stop_loss_pct) if self.stop_loss_pct > 0.0 else None
            take_profit = (
                close * (1.0 + self.take_profit_pct) if self.take_profit_pct > 0.0 else None
            )
            state.mode = "LONG"
            state.high_watermark = close
            state.low_watermark = None
        else:
            stop_loss = close * (1.0 + self.stop_loss_pct) if self.stop_loss_pct > 0.0 else None
            take_profit = (
                close * (1.0 - self.take_profit_pct) if self.take_profit_pct > 0.0 else None
            )
            state.mode = "SHORT"
            state.low_watermark = close
            state.high_watermark = None
        strength = (
            0.55
            + min(1.0, abs(metrics["price_extension"]) / max(self.price_extension_min, 1e-9)) * 0.35
        )
        strength += min(1.0, abs(metrics["flow_imbalance"])) * 0.35
        self._emit(
            getattr(event, "time", None),
            signal_type,
            strength=self._clamp(strength, 0.05, 1.5),
            price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_percent=self.trailing_exit_pct if self.trailing_exit_pct > 0.0 else None,
            metadata=metadata,
        )
        state.entry_price = close
        state.bars_held = 0

    def _process(self, event: Any, snapshot: dict[str, float | None] | None = None) -> None:
        event_time = getattr(event, "time", getattr(event, "datetime", None))
        key = time_key(event_time)
        if key and key == self._state.last_time_key:
            return
        self._state.last_time_key = key
        snapshot = dict(snapshot or {})
        close = snapshot.get("close")
        if close is None:
            close = self._extract_feature(event, "close")
        if close is None:
            close = safe_float(getattr(event, "close", None))
        if close is None or close <= self.min_price:
            return
        funding = self._extract_feature(event, "funding_rate")
        self._state.has_funding = self._state.has_funding or funding is not None
        buy_quote, sell_quote, flow_source = self._resolve_flow_values(event, snapshot, close)
        self._state.last_flow_source = flow_source
        self._state.has_taker_flow = self._state.has_taker_flow or flow_source.startswith("feature")
        self._state.closes.append(float(close))
        self._state.taker_buy_quote_volume.append(float(buy_quote))
        self._state.taker_sell_quote_volume.append(float(sell_quote))
        self._state.funding_rate.append(float(funding) if funding is not None else 0.0)
        self._state.ticks_seen += 1
        if (
            self.evaluation_cadence_bars > 1
            and self._state.ticks_seen % self.evaluation_cadence_bars
        ):
            return
        if len(self._state.closes) <= max(self.momentum_lookback_bars, self.flow_lookback_bars):
            return
        if not self._state.has_taker_flow or not self._state.has_funding:
            return
        price_extension = self._pct_change(self._state.closes, self.momentum_lookback_bars) or 0.0
        flow_imbalance = self._flow_imbalance()
        funding_rate = float(self._state.funding_rate[-1]) if self._state.funding_rate else 0.0
        realized_volatility = self._realized_vol(self._state.closes, self.volatility_lookback_bars)
        vol_multiplier = self._volatility_multiplier(realized_volatility)
        metrics = {
            "price_extension": price_extension,
            "flow_imbalance": flow_imbalance,
            "funding_rate": funding_rate,
            "realized_volatility": realized_volatility,
            "volatility_multiplier": vol_multiplier,
        }
        if self._maybe_exit(event, float(close), metrics):
            return
        if self._state.mode != "OUT":
            return
        if self._state.cooldown_remaining > 0:
            self._state.cooldown_remaining -= 1
            return
        hour = self._hour_utc(event_time)
        if self.entry_hours_utc and (hour is None or hour not in self.entry_hours_utc):
            return
        if self.funding_abs_cap > 0.0 and abs(funding_rate) > self.funding_abs_cap:
            return
        if (
            self.max_realized_volatility > 0.0
            and realized_volatility > self.max_realized_volatility
        ):
            return
        long_exhaustion = (
            flow_imbalance <= -self.flow_imbalance_min
            and price_extension <= -self.price_extension_min
        )
        short_exhaustion = (
            self.allow_short
            and flow_imbalance >= self.flow_imbalance_min
            and price_extension >= self.price_extension_min
        )
        if long_exhaustion:
            self._enter(event, "LONG", float(close), "taker_sell_exhaustion_reversal_long", metrics)
        elif short_exhaustion:
            self._enter(
                event, "SHORT", float(close), "taker_buy_exhaustion_reversal_short", metrics
            )

    def calculate_signals_window(self, event: Any, aggregator: Any) -> None:
        _ = aggregator
        if str(getattr(event, "type", "")).upper() != "MARKET_WINDOW":
            self.calculate_signals(event)
            return
        self._process(event, snapshot=self._window_snapshot(event))

    def calculate_signals(self, event: Any) -> None:
        event_type = str(getattr(event, "type", "")).upper()
        if event_type == "MARKET_WINDOW":
            self.calculate_signals_window(event, aggregator=None)
            return
        if event_type != "MARKET" or str(getattr(event, "symbol", "")) != self.target_symbol:
            return
        snapshot = {"close": safe_float(getattr(event, "close", None))}
        self._process(event, snapshot=snapshot)

    def get_state(self) -> dict[str, Any]:
        return {
            "state": {
                "closes": list(self._state.closes),
                "taker_buy_quote_volume": list(self._state.taker_buy_quote_volume),
                "taker_sell_quote_volume": list(self._state.taker_sell_quote_volume),
                "funding_rate": list(self._state.funding_rate),
                "mode": self._state.mode,
                "entry_price": self._state.entry_price,
                "high_watermark": self._state.high_watermark,
                "low_watermark": self._state.low_watermark,
                "bars_held": self._state.bars_held,
                "cooldown_remaining": self._state.cooldown_remaining,
                "ticks_seen": self._state.ticks_seen,
                "last_time_key": self._state.last_time_key,
                "has_taker_flow": self._state.has_taker_flow,
                "has_funding": self._state.has_funding,
                "last_flow_source": self._state.last_flow_source,
            }
        }

    def set_state(self, state: dict[str, Any]) -> None:
        payload = state.get("state") if isinstance(state, dict) else None
        if not isinstance(payload, dict):
            return
        for attr in ("closes", "taker_buy_quote_volume", "taker_sell_quote_volume", "funding_rate"):
            target = getattr(self._state, attr)
            target.clear()
            for value in list(payload.get(attr) or []):
                parsed = safe_float(value)
                if parsed is not None:
                    target.append(parsed)
        mode = str(payload.get("mode", "OUT")).upper()
        self._state.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
        self._state.entry_price = safe_float(payload.get("entry_price"))
        self._state.high_watermark = safe_float(payload.get("high_watermark"))
        self._state.low_watermark = safe_float(payload.get("low_watermark"))
        self._state.bars_held = max(0, int(payload.get("bars_held") or 0))
        self._state.cooldown_remaining = max(0, int(payload.get("cooldown_remaining") or 0))
        self._state.ticks_seen = max(0, int(payload.get("ticks_seen") or 0))
        self._state.last_time_key = str(payload.get("last_time_key") or "")
        self._state.has_taker_flow = bool(payload.get("has_taker_flow", False))
        self._state.has_funding = bool(payload.get("has_funding", False))
        self._state.last_flow_source = str(payload.get("last_flow_source") or "missing")
