"""Perpetual crowding/carry strategy (optional support-data path)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.advanced_alpha import perp_crowding_score
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@dataclass(slots=True)
class _State:
    funding_rate: deque[float]
    open_interest: deque[float]
    liquidation_long_notional: deque[float]
    liquidation_short_notional: deque[float]
    closes: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    last_time_key: str = ""


class PerpCrowdingCarryStrategy(Strategy):
    """Carry/crowding sleeve with risk-off fades on extreme positioning."""

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "window": HyperParam.integer("window", default=96, low=16, high=4096, grid=[64, 96, 144]),
            "mild_funding": HyperParam.floating("mild_funding", default=0.0002, low=0.0, high=0.02),
            "extreme_funding": HyperParam.floating("extreme_funding", default=0.0012, low=0.0001, high=0.05),
            "entry_threshold": HyperParam.floating(
                "entry_threshold",
                default=0.30,
                low=0.01,
                high=0.99,
                grid=[0.2, 0.3, 0.4],
            ),
            "exit_threshold": HyperParam.floating(
                "exit_threshold",
                default=0.10,
                low=0.0,
                high=0.8,
                grid=[0.05, 0.1, 0.2],
            ),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.02, low=0.001, high=0.5),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=72, low=1, high=100_000),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
        }

    def __init__(
        self,
        bars,
        events,
        window: int = 96,
        mild_funding: float = 0.0002,
        extreme_funding: float = 0.0012,
        entry_threshold: float = 0.30,
        exit_threshold: float = 0.10,
        stop_loss_pct: float = 0.02,
        max_hold_bars: int = 72,
        allow_short: bool = True,
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)

        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "window": window,
                "mild_funding": mild_funding,
                "extreme_funding": extreme_funding,
                "entry_threshold": entry_threshold,
                "exit_threshold": exit_threshold,
                "stop_loss_pct": stop_loss_pct,
                "max_hold_bars": max_hold_bars,
                "allow_short": allow_short,
            },
            keep_unknown=False,
        )

        self.window = int(resolved["window"])
        self.mild_funding = float(resolved["mild_funding"])
        self.extreme_funding = float(resolved["extreme_funding"])
        self.entry_threshold = float(resolved["entry_threshold"])
        self.exit_threshold = float(resolved["exit_threshold"])
        self.stop_loss_pct = float(resolved["stop_loss_pct"])
        self.max_hold_bars = int(resolved["max_hold_bars"])
        self.allow_short = bool(resolved["allow_short"])

        size = self.window + 8
        self._state = {
            symbol: _State(
                funding_rate=deque(maxlen=size),
                open_interest=deque(maxlen=size),
                liquidation_long_notional=deque(maxlen=size),
                liquidation_short_notional=deque(maxlen=size),
                closes=deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }

    def get_state(self):
        return {
            "symbol_state": {
                symbol: {
                    "funding_rate": list(item.funding_rate),
                    "open_interest": list(item.open_interest),
                    "liquidation_long_notional": list(item.liquidation_long_notional),
                    "liquidation_short_notional": list(item.liquidation_short_notional),
                    "closes": list(item.closes),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": int(item.bars_held),
                    "last_time_key": item.last_time_key,
                }
                for symbol, item in self._state.items()
            }
        }

    def set_state(self, state):
        if not isinstance(state, dict):
            return
        raw_state = state.get("symbol_state")
        if not isinstance(raw_state, dict):
            return

        for symbol, payload in raw_state.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            for attr in (
                "funding_rate",
                "open_interest",
                "liquidation_long_notional",
                "liquidation_short_notional",
                "closes",
            ):
                target = getattr(item, attr)
                target.clear()
                for value in list(payload.get(attr) or []):
                    parsed = safe_float(value)
                    if parsed is not None:
                        target.append(parsed)
            mode = str(payload.get("mode", "OUT")).upper()
            item.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
            item.entry_price = safe_float(payload.get("entry_price"))
            try:
                item.bars_held = max(0, int(payload.get("bars_held", 0)))
            except Exception:
                item.bars_held = 0
            item.last_time_key = str(payload.get("last_time_key", ""))

    def _extract_feature(self, event: Any, symbol: str, field: str) -> float | None:
        direct = safe_float(getattr(event, field, None))
        if direct is not None:
            return direct
        getter = getattr(self.bars, "get_latest_feature_value", None)
        if callable(getter):
            try:
                value = getter(symbol, field)
            except Exception:
                value = None
            parsed = safe_float(value)
            if parsed is not None:
                return parsed
        try:
            value = self.bars.get_latest_bar_value(symbol, field)
        except Exception:
            value = None
        return safe_float(value)

    def _emit(self, symbol, event_time, signal_type, *, strength=1.0, stop_loss=None, metadata=None):
        self.events.put(
            SignalEvent(
                strategy_id="perp_crowding_carry",
                symbol=symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=float(strength),
                stop_loss=stop_loss,
                metadata=metadata,
            )
        )

    def calculate_signals(self, event):
        if str(getattr(event, "type", "")).upper() != "MARKET":
            return

        symbol = str(getattr(event, "symbol", ""))
        if symbol not in self._state:
            return

        item = self._state[symbol]
        key = time_key(getattr(event, "time", getattr(event, "datetime", None)))
        if key and key == item.last_time_key:
            return
        item.last_time_key = key

        close = self._extract_feature(event, symbol, "close")
        if close is None:
            close = safe_float(getattr(event, "close", None))
        if close is not None:
            item.closes.append(float(close))

        feature_map = {
            "funding_rate": self._extract_feature(event, symbol, "funding_rate"),
            "open_interest": self._extract_feature(event, symbol, "open_interest"),
            "liquidation_long_notional": self._extract_feature(event, symbol, "liquidation_long_notional"),
            "liquidation_short_notional": self._extract_feature(event, symbol, "liquidation_short_notional"),
        }
        if any(value is None for value in feature_map.values()):
            return

        item.funding_rate.append(float(feature_map["funding_rate"]))
        item.open_interest.append(float(feature_map["open_interest"]))
        item.liquidation_long_notional.append(float(feature_map["liquidation_long_notional"]))
        item.liquidation_short_notional.append(float(feature_map["liquidation_short_notional"]))

        crowding = perp_crowding_score(
            funding_rate=item.funding_rate,
            open_interest=item.open_interest,
            liquidation_long_notional=item.liquidation_long_notional,
            liquidation_short_notional=item.liquidation_short_notional,
            window=self.window,
        )
        if not bool(crowding.get("available", False)):
            return

        score = float(crowding.get("score") or 0.0)
        comps = dict(crowding.get("components") or {})
        funding = float(item.funding_rate[-1]) if item.funding_rate else 0.0

        metadata = {
            "strategy": "PerpCrowdingCarryStrategy",
            "crowding_score": score,
            "funding_rate": funding,
            "components": comps,
            "bars_held": int(item.bars_held),
        }

        close_value = float(item.closes[-1]) if item.closes else 0.0

        if item.mode == "LONG":
            item.bars_held += 1
            should_exit = (
                (score <= self.exit_threshold)
                or (funding >= self.extreme_funding)
                or (item.bars_held >= self.max_hold_bars)
                or (close_value > 0.0 and item.entry_price is not None and close_value <= item.entry_price * (1.0 - self.stop_loss_pct))
            )
            if should_exit:
                self._emit(symbol, getattr(event, "time", None), "EXIT", metadata={**metadata, "reason": "long_exit"})
                item.mode = "OUT"
                item.entry_price = None
                item.bars_held = 0
            return

        if item.mode == "SHORT":
            item.bars_held += 1
            should_exit = (
                (score >= -self.exit_threshold)
                or (funding <= -self.extreme_funding)
                or (item.bars_held >= self.max_hold_bars)
                or (close_value > 0.0 and item.entry_price is not None and close_value >= item.entry_price * (1.0 + self.stop_loss_pct))
            )
            if should_exit:
                self._emit(symbol, getattr(event, "time", None), "EXIT", metadata={**metadata, "reason": "short_exit"})
                item.mode = "OUT"
                item.entry_price = None
                item.bars_held = 0
            return

        # Entry logic:
        # 1) Carry-aligned: mildly positive funding + low crowding => LONG
        # 2) Extreme positive funding + rising OI crowding => SHORT fade
        oi_delta_z = float(comps.get("oi_delta_z", 0.0))
        strength = min(2.0, max(0.2, abs(score)))

        carry_long = funding > 0.0 and funding <= self.mild_funding and score >= self.entry_threshold
        crowded_long = funding >= self.extreme_funding and oi_delta_z > 0.0 and score >= self.entry_threshold
        carry_short = funding < 0.0 and abs(funding) <= self.mild_funding and score <= -self.entry_threshold
        crowded_short = funding <= -self.extreme_funding and oi_delta_z < 0.0 and score <= -self.entry_threshold

        if carry_long and not crowded_long:
            self._emit(
                symbol,
                getattr(event, "time", None),
                "LONG",
                strength=strength,
                stop_loss=(close_value * (1.0 - self.stop_loss_pct) if close_value > 0 else None),
                metadata={**metadata, "reason": "carry_long_entry"},
            )
            item.mode = "LONG"
            item.entry_price = close_value if close_value > 0 else None
            item.bars_held = 0
            return

        if self.allow_short and crowded_long:
            self._emit(
                symbol,
                getattr(event, "time", None),
                "SHORT",
                strength=strength,
                stop_loss=(close_value * (1.0 + self.stop_loss_pct) if close_value > 0 else None),
                metadata={**metadata, "reason": "crowded_long_fade"},
            )
            item.mode = "SHORT"
            item.entry_price = close_value if close_value > 0 else None
            item.bars_held = 0
            return

        if self.allow_short and carry_short and not crowded_short:
            self._emit(
                symbol,
                getattr(event, "time", None),
                "SHORT",
                strength=strength,
                stop_loss=(close_value * (1.0 + self.stop_loss_pct) if close_value > 0 else None),
                metadata={**metadata, "reason": "carry_short_entry"},
            )
            item.mode = "SHORT"
            item.entry_price = close_value if close_value > 0 else None
            item.bars_held = 0
            return

        if crowded_short:
            self._emit(
                symbol,
                getattr(event, "time", None),
                "LONG",
                strength=strength,
                stop_loss=(close_value * (1.0 - self.stop_loss_pct) if close_value > 0 else None),
                metadata={**metadata, "reason": "crowded_short_fade"},
            )
            item.mode = "LONG"
            item.entry_price = close_value if close_value > 0 else None
            item.bars_held = 0
