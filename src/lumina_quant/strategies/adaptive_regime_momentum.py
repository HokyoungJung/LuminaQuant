"""Adaptive regime momentum strategy for live-equivalent portfolio sleeves.

The original top-cap momentum sleeves consume every 1-second row forwarded by
``MARKET_WINDOW`` and therefore make their lookbacks cadence-dependent in live
portfolio-mode backtests.  This strategy intentionally compresses each
``MARKET_WINDOW`` decision tick into exactly one bar per symbol, so lookbacks
mean decision bars (normally minutes) without requiring ``TimeframeAggregator``.
"""

from __future__ import annotations

import logging
from collections import deque
from itertools import pairwise
from statistics import mean, stdev
from typing import Any

from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.common import safe_float, safe_int, time_key
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

LOGGER = logging.getLogger(__name__)


class AdaptiveRegimeMomentumStrategy(Strategy):
    """Long/short momentum rotation with an explicit market-regime switch.

    Rules:
    - Use one close per MARKET_WINDOW decision tick; no timeframe aggregator.
    - Risk-on: long the strongest symbols with positive medium and short-term
      momentum.
    - Risk-off: short the weakest symbols with negative medium and short-term
      momentum.
    - Cash when the broad regime and per-symbol thresholds disagree.
    - Attach stop/take-profit/trailing controls and explicit target allocation
      metadata so portfolio sizing remains bounded.
    """

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "lookback_bars": HyperParam.integer(
                "lookback_bars", default=168, low=4, high=10080, tunable=False
            ),
            "short_lookback_bars": HyperParam.integer(
                "short_lookback_bars", default=12, low=1, high=1440, tunable=False
            ),
            "regime_lookback_bars": HyperParam.integer(
                "regime_lookback_bars", default=168, low=4, high=10080, tunable=False
            ),
            "volatility_lookback_bars": HyperParam.integer(
                "volatility_lookback_bars", default=60, low=2, high=1440, tunable=False
            ),
            "rebalance_bars": HyperParam.integer(
                "rebalance_bars", default=12, low=1, high=1440, tunable=False
            ),
            "signal_threshold": HyperParam.floating(
                "signal_threshold", default=0.010, low=0.0, high=1.0, tunable=False
            ),
            "broad_threshold": HyperParam.floating(
                "broad_threshold", default=0.0, low=0.0, high=1.0, tunable=False
            ),
            "max_longs": HyperParam.integer("max_longs", default=1, low=0, high=32, tunable=False),
            "max_shorts": HyperParam.integer(
                "max_shorts", default=1, low=0, high=32, tunable=False
            ),
            "gross_exposure": HyperParam.floating(
                "gross_exposure", default=0.70, low=0.0, high=2.0, tunable=False
            ),
            "long_exposure_multiplier": HyperParam.floating(
                "long_exposure_multiplier", default=1.0, low=0.0, high=5.0, tunable=False
            ),
            "short_exposure_multiplier": HyperParam.floating(
                "short_exposure_multiplier", default=1.0, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=1500.0, low=0.0, high=1000000.0, tunable=False
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.035, low=0.0, high=1.0, tunable=False
            ),
            "take_profit_pct": HyperParam.floating(
                "take_profit_pct", default=0.070, low=0.0, high=2.0, tunable=False
            ),
            "trailing_exit_pct": HyperParam.floating(
                "trailing_exit_pct", default=0.045, low=0.0, high=1.0, tunable=False
            ),
            "max_hold_bars": HyperParam.integer(
                "max_hold_bars", default=720, low=0, high=10080, tunable=False
            ),
            "max_realized_vol": HyperParam.floating(
                "max_realized_vol", default=0.0, low=0.0, high=1.0, tunable=False
            ),
            "volatility_target_per_bar": HyperParam.floating(
                "volatility_target_per_bar", default=0.0, low=0.0, high=1.0, tunable=False
            ),
            "min_volatility_exposure_multiplier": HyperParam.floating(
                "min_volatility_exposure_multiplier",
                default=0.0,
                low=0.0,
                high=5.0,
                tunable=False,
            ),
            "max_volatility_exposure_multiplier": HyperParam.floating(
                "max_volatility_exposure_multiplier",
                default=5.0,
                low=0.0,
                high=5.0,
                tunable=False,
            ),
            "volatility_trailing_multiplier": HyperParam.floating(
                "volatility_trailing_multiplier", default=0.0, low=0.0, high=100.0, tunable=False
            ),
            "min_dynamic_trailing_pct": HyperParam.floating(
                "min_dynamic_trailing_pct", default=0.0, low=0.0, high=1.0, tunable=False
            ),
            "max_dynamic_trailing_pct": HyperParam.floating(
                "max_dynamic_trailing_pct", default=1.0, low=0.0, high=1.0, tunable=False
            ),
            "volume_lookback_bars": HyperParam.integer(
                "volume_lookback_bars", default=60, low=2, high=1440, tunable=False
            ),
            "volume_weighted_broad": HyperParam.boolean(
                "volume_weighted_broad", default=False, tunable=False
            ),
            "min_entry_volume_z": HyperParam.floating(
                "min_entry_volume_z", default=-999.0, low=-999.0, high=20.0, tunable=False
            ),
            "min_price": HyperParam.floating(
                "min_price", default=0.10, low=0.0, high=1_000_000.0, tunable=False
            ),
            "btc_symbol": HyperParam.string("btc_symbol", default="BTC/USDT", tunable=False),
            "risk_off_exit": HyperParam.boolean("risk_off_exit", default=True, tunable=False),
        }

    def __init__(
        self,
        bars: Any,
        events: Any,
        lookback_bars: int = 168,
        short_lookback_bars: int = 12,
        regime_lookback_bars: int = 168,
        volatility_lookback_bars: int = 60,
        rebalance_bars: int = 12,
        signal_threshold: float = 0.010,
        broad_threshold: float = 0.0,
        max_longs: int = 1,
        max_shorts: int = 1,
        gross_exposure: float = 0.70,
        long_exposure_multiplier: float = 1.0,
        short_exposure_multiplier: float = 1.0,
        max_order_value: float = 1500.0,
        stop_loss_pct: float = 0.035,
        take_profit_pct: float = 0.070,
        trailing_exit_pct: float = 0.045,
        max_hold_bars: int = 720,
        max_realized_vol: float = 0.0,
        volatility_target_per_bar: float = 0.0,
        min_volatility_exposure_multiplier: float = 0.0,
        max_volatility_exposure_multiplier: float = 5.0,
        volatility_trailing_multiplier: float = 0.0,
        min_dynamic_trailing_pct: float = 0.0,
        max_dynamic_trailing_pct: float = 1.0,
        volume_lookback_bars: int = 60,
        volume_weighted_broad: bool = False,
        min_entry_volume_z: float = -999.0,
        min_price: float = 0.10,
        btc_symbol: str | None = None,
        risk_off_exit: bool = True,
    ) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        if not self.symbol_list:
            raise ValueError("AdaptiveRegimeMomentumStrategy requires at least one symbol.")

        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "lookback_bars": lookback_bars,
                "short_lookback_bars": short_lookback_bars,
                "regime_lookback_bars": regime_lookback_bars,
                "volatility_lookback_bars": volatility_lookback_bars,
                "rebalance_bars": rebalance_bars,
                "signal_threshold": signal_threshold,
                "broad_threshold": broad_threshold,
                "max_longs": max_longs,
                "max_shorts": max_shorts,
                "gross_exposure": gross_exposure,
                "long_exposure_multiplier": long_exposure_multiplier,
                "short_exposure_multiplier": short_exposure_multiplier,
                "max_order_value": max_order_value,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "trailing_exit_pct": trailing_exit_pct,
                "max_hold_bars": max_hold_bars,
                "max_realized_vol": max_realized_vol,
                "volatility_target_per_bar": volatility_target_per_bar,
                "min_volatility_exposure_multiplier": min_volatility_exposure_multiplier,
                "max_volatility_exposure_multiplier": max_volatility_exposure_multiplier,
                "volatility_trailing_multiplier": volatility_trailing_multiplier,
                "min_dynamic_trailing_pct": min_dynamic_trailing_pct,
                "max_dynamic_trailing_pct": max_dynamic_trailing_pct,
                "volume_lookback_bars": volume_lookback_bars,
                "volume_weighted_broad": volume_weighted_broad,
                "min_entry_volume_z": min_entry_volume_z,
                "min_price": min_price,
                "btc_symbol": btc_symbol,
                "risk_off_exit": risk_off_exit,
            },
            keep_unknown=False,
        )

        self.lookback_bars = max(1, int(resolved["lookback_bars"]))
        self.short_lookback_bars = max(1, int(resolved["short_lookback_bars"]))
        self.regime_lookback_bars = max(1, int(resolved["regime_lookback_bars"]))
        self.volatility_lookback_bars = max(2, int(resolved["volatility_lookback_bars"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.signal_threshold = max(0.0, float(resolved["signal_threshold"]))
        self.broad_threshold = max(0.0, float(resolved["broad_threshold"]))
        self.max_longs = max(0, int(resolved["max_longs"]))
        self.max_shorts = max(0, int(resolved["max_shorts"]))
        self.gross_exposure = max(0.0, float(resolved["gross_exposure"]))
        self.long_exposure_multiplier = max(0.0, float(resolved["long_exposure_multiplier"]))
        self.short_exposure_multiplier = max(0.0, float(resolved["short_exposure_multiplier"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.take_profit_pct = max(0.0, float(resolved["take_profit_pct"]))
        self.trailing_exit_pct = max(0.0, float(resolved["trailing_exit_pct"]))
        self.max_hold_bars = max(0, int(resolved["max_hold_bars"]))
        self.max_realized_vol = max(0.0, float(resolved["max_realized_vol"]))
        self.volatility_target_per_bar = max(0.0, float(resolved["volatility_target_per_bar"]))
        self.min_volatility_exposure_multiplier = max(
            0.0,
            float(resolved["min_volatility_exposure_multiplier"]),
        )
        self.max_volatility_exposure_multiplier = max(
            self.min_volatility_exposure_multiplier,
            float(resolved["max_volatility_exposure_multiplier"]),
        )
        self.volatility_trailing_multiplier = max(
            0.0,
            float(resolved["volatility_trailing_multiplier"]),
        )
        self.min_dynamic_trailing_pct = max(0.0, float(resolved["min_dynamic_trailing_pct"]))
        self.max_dynamic_trailing_pct = max(
            self.min_dynamic_trailing_pct,
            float(resolved["max_dynamic_trailing_pct"]),
        )
        self.volume_lookback_bars = max(2, int(resolved["volume_lookback_bars"]))
        self.volume_weighted_broad = bool(resolved["volume_weighted_broad"])
        self.min_entry_volume_z = float(resolved["min_entry_volume_z"])
        self.min_price = max(0.0, float(resolved["min_price"]))
        self.risk_off_exit = bool(resolved["risk_off_exit"])

        default_btc = "BTC/USDT" if "BTC/USDT" in self.symbol_list else self.symbol_list[0]
        candidate_btc = str(resolved["btc_symbol"] or "").strip()
        self.btc_symbol = candidate_btc if candidate_btc in self.symbol_list else default_btc

        history_len = (
            max(
                self.lookback_bars,
                self.short_lookback_bars,
                self.regime_lookback_bars,
                self.volatility_lookback_bars + 1,
                self.volume_lookback_bars + 1,
            )
            + 2
        )
        self._price_history: dict[str, deque[float]] = {
            symbol: deque(maxlen=history_len) for symbol in self.symbol_list
        }
        self._quote_volume_history: dict[str, deque[float]] = {
            symbol: deque(maxlen=history_len) for symbol in self.symbol_list
        }
        self._last_symbol_time_key = dict.fromkeys(self.symbol_list, "")
        self._position_state = dict.fromkeys(self.symbol_list, "OUT")
        self._entry_price = dict.fromkeys(self.symbol_list, 0.0)
        self._high_watermark = dict.fromkeys(self.symbol_list, 0.0)
        self._low_watermark = dict.fromkeys(self.symbol_list, 0.0)
        self._bars_held = dict.fromkeys(self.symbol_list, 0)
        self._last_eval_time_key = ""
        self._tick = 0

    def get_state(self) -> dict[str, Any]:
        return {
            "position_state": dict(self._position_state),
            "entry_price": dict(self._entry_price),
            "high_watermark": dict(self._high_watermark),
            "low_watermark": dict(self._low_watermark),
            "bars_held": dict(self._bars_held),
            "last_symbol_time_key": dict(self._last_symbol_time_key),
            "last_eval_time_key": str(self._last_eval_time_key),
            "tick": int(self._tick),
            "price_history": {
                symbol: list(history) for symbol, history in self._price_history.items()
            },
            "quote_volume_history": {
                symbol: list(history) for symbol, history in self._quote_volume_history.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        for attr_name, target in (
            ("position_state", self._position_state),
            ("entry_price", self._entry_price),
            ("high_watermark", self._high_watermark),
            ("low_watermark", self._low_watermark),
            ("bars_held", self._bars_held),
            ("last_symbol_time_key", self._last_symbol_time_key),
        ):
            raw = state.get(attr_name)
            if not isinstance(raw, dict):
                continue
            for symbol, value in raw.items():
                if symbol not in target:
                    continue
                if attr_name == "position_state":
                    if value in {"OUT", "LONG", "SHORT"}:
                        target[symbol] = str(value)
                elif attr_name == "bars_held":
                    target[symbol] = max(0, safe_int(value, 0))
                elif attr_name == "last_symbol_time_key":
                    target[symbol] = str(value)
                else:
                    target[symbol] = float(safe_float(value) or 0.0)

        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._tick = max(0, safe_int(state.get("tick", 0), 0))

        raw_history = state.get("price_history")
        if isinstance(raw_history, dict):
            for symbol, values in raw_history.items():
                if symbol not in self._price_history or not isinstance(values, list):
                    continue
                history = self._price_history[symbol]
                history.clear()
                keep = int(history.maxlen) if history.maxlen is not None else len(values)
                for value in values[-keep:]:
                    parsed = safe_float(value)
                    if parsed is not None and parsed > 0.0:
                        history.append(float(parsed))

        raw_volume_history = state.get("quote_volume_history")
        if not isinstance(raw_volume_history, dict):
            return
        for symbol, values in raw_volume_history.items():
            if symbol not in self._quote_volume_history or not isinstance(values, list):
                continue
            history = self._quote_volume_history[symbol]
            history.clear()
            keep = int(history.maxlen) if history.maxlen is not None else len(values)
            for value in values[-keep:]:
                parsed = safe_float(value)
                if parsed is not None and parsed >= 0.0:
                    history.append(float(parsed))

    def _latest_bar_close(self, symbol: str, row: Any | None = None) -> float | None:
        if isinstance(row, dict):
            parsed = safe_float(row.get("close"))
            if parsed is not None and parsed > 0.0:
                return float(parsed)
        if isinstance(row, (tuple, list)) and len(row) >= 5:
            parsed = safe_float(row[4])
            if parsed is not None and parsed > 0.0:
                return float(parsed)
        parsed = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
        if parsed is None or parsed <= 0.0:
            return None
        return float(parsed)

    @staticmethod
    def _row_close_and_quote_volume(row: Any) -> tuple[float | None, float | None]:
        close = None
        volume = None
        if isinstance(row, dict):
            close = safe_float(row.get("close"))
            volume = safe_float(row.get("volume"))
        elif isinstance(row, (tuple, list)):
            if len(row) >= 5:
                close = safe_float(row[4])
            if len(row) >= 6:
                volume = safe_float(row[5])
        if close is None or close <= 0.0:
            return None, None
        quote_volume = None
        if volume is not None and volume >= 0.0:
            quote_volume = float(close) * float(volume)
        return float(close), quote_volume

    def _window_close_and_quote_volume(self, rows: list[Any]) -> tuple[float | None, float | None]:
        close = None
        quote_volume = 0.0
        volume_seen = False
        for row in rows:
            row_close, row_quote_volume = self._row_close_and_quote_volume(row)
            if row_close is not None:
                close = float(row_close)
            if row_quote_volume is not None:
                quote_volume += float(row_quote_volume)
                volume_seen = True
        return close, (quote_volume if volume_seen else None)

    def _append_close(
        self,
        symbol: str,
        event_time_key: str,
        close_price: float,
        quote_volume: float | None = None,
    ) -> bool:
        if symbol not in self._price_history:
            return False
        if not event_time_key or self._last_symbol_time_key.get(symbol) == event_time_key:
            return False
        if close_price <= 0.0:
            return False
        self._last_symbol_time_key[symbol] = event_time_key
        self._price_history[symbol].append(float(close_price))
        parsed_volume = safe_float(quote_volume)
        self._quote_volume_history[symbol].append(
            float(parsed_volume) if parsed_volume is not None and parsed_volume >= 0.0 else 0.0
        )
        return True

    def _momentum(self, symbol: str, lookback: int) -> float | None:
        history = self._price_history.get(symbol)
        if history is None or len(history) <= lookback:
            return None
        latest = float(history[-1])
        base = float(history[-1 - lookback])
        if latest <= 0.0 or base <= 0.0:
            return None
        return (latest / base) - 1.0

    def _realized_vol(self, symbol: str) -> float:
        history = self._price_history.get(symbol)
        if history is None or len(history) <= self.volatility_lookback_bars:
            return 0.0
        values = list(history)[-(self.volatility_lookback_bars + 1) :]
        returns: list[float] = []
        for before, after in pairwise(values):
            if before > 0.0 and after > 0.0:
                returns.append((after / before) - 1.0)
        if len(returns) <= 1:
            return 0.0
        return float(stdev(returns))

    def _avg_quote_volume(self, symbol: str) -> float:
        history = self._quote_volume_history.get(symbol)
        if history is None or not history:
            return 0.0
        values = [
            float(value) for value in list(history)[-self.volume_lookback_bars :] if value >= 0.0
        ]
        if not values:
            return 0.0
        return float(mean(values))

    def _volume_z(self, symbol: str) -> float:
        history = self._quote_volume_history.get(symbol)
        if history is None or len(history) <= self.volume_lookback_bars:
            return 0.0
        values = [
            float(value)
            for value in list(history)[-(self.volume_lookback_bars + 1) :]
            if value >= 0.0
        ]
        if len(values) <= 2:
            return 0.0
        latest = values[-1]
        baseline = values[:-1]
        sigma = stdev(baseline) if len(baseline) > 1 else 0.0
        if sigma <= 1e-12:
            return 0.0
        return float((latest - mean(baseline)) / sigma)

    def _entry_volume_ok(self, symbol: str) -> bool:
        if self.min_entry_volume_z <= -998.0:
            return True
        return self._volume_z(symbol) >= self.min_entry_volume_z

    def _regime(self) -> tuple[str, float]:
        btc_momentum = self._momentum(self.btc_symbol, self.regime_lookback_bars)
        if btc_momentum is None:
            return "WARMUP", 0.0

        broad_momenta: list[float] = []
        weighted_sum = 0.0
        total_weight = 0.0
        for symbol in self.symbol_list:
            momentum = self._momentum(symbol, self.regime_lookback_bars)
            if momentum is None:
                continue
            broad_momenta.append(float(momentum))
            if self.volume_weighted_broad:
                weight = max(0.0, self._avg_quote_volume(symbol))
                if weight > 0.0:
                    weighted_sum += float(momentum) * weight
                    total_weight += weight
        if self.volume_weighted_broad and total_weight > 0.0:
            broad_mean = weighted_sum / total_weight
        else:
            broad_mean = mean(broad_momenta) if broad_momenta else btc_momentum
        broad_score = 0.70 * float(btc_momentum) + 0.30 * float(broad_mean)
        if broad_score >= self.broad_threshold:
            return "RISK_ON", broad_score
        if broad_score <= -self.broad_threshold:
            return "RISK_OFF", broad_score
        return "NEUTRAL", broad_score

    def _target_allocation(self, target_count: int, side: str) -> float:
        if target_count <= 0:
            return 0.0
        multiplier = (
            self.long_exposure_multiplier
            if str(side).upper() == "LONG"
            else self.short_exposure_multiplier
        )
        return max(0.0, float(self.gross_exposure) * float(multiplier) / float(target_count))

    def _volatility_exposure_multiplier(self, symbol: str) -> float:
        if self.volatility_target_per_bar <= 0.0:
            return 1.0
        realized = self._realized_vol(symbol)
        if realized <= 1e-12:
            return min(1.0, float(self.max_volatility_exposure_multiplier))
        multiplier = float(self.volatility_target_per_bar) / float(realized)
        multiplier = max(float(self.min_volatility_exposure_multiplier), multiplier)
        multiplier = min(float(self.max_volatility_exposure_multiplier), multiplier)
        return max(0.0, float(multiplier))

    def _target_allocation_for_symbol(self, symbol: str, target_count: int, side: str) -> float:
        base = self._target_allocation(target_count, side)
        if base <= 0.0:
            return 0.0
        return base * self._volatility_exposure_multiplier(symbol)

    def _build_targets(
        self,
    ) -> tuple[dict[str, str], dict[str, float], str, float, dict[str, float]]:
        regime, broad_score = self._regime()
        momentum_map: dict[str, float] = {}
        short_map: dict[str, float] = {}
        targets: dict[str, str] = {}
        target_allocations: dict[str, float] = {}

        for symbol in self.symbol_list:
            history = self._price_history.get(symbol)
            if history is None or not history or history[-1] < self.min_price:
                continue
            medium = self._momentum(symbol, self.lookback_bars)
            short = self._momentum(symbol, self.short_lookback_bars)
            if medium is None or short is None:
                continue
            if self.max_realized_vol > 0.0 and self._realized_vol(symbol) > self.max_realized_vol:
                continue
            momentum_map[symbol] = float(medium)
            short_map[symbol] = float(short)

        if regime == "RISK_ON" and self.max_longs > 0:
            longs = [
                symbol
                for symbol, momentum in sorted(
                    momentum_map.items(), key=lambda item: item[1], reverse=True
                )
                if momentum >= self.signal_threshold
                and short_map.get(symbol, 0.0) > 0.0
                and self._entry_volume_ok(symbol)
            ][: self.max_longs]
            for symbol in longs:
                targets[symbol] = "LONG"
                target_allocations[symbol] = self._target_allocation_for_symbol(
                    symbol,
                    len(longs),
                    "LONG",
                )
        elif regime == "RISK_OFF" and self.max_shorts > 0:
            shorts = [
                symbol
                for symbol, momentum in sorted(momentum_map.items(), key=lambda item: item[1])
                if momentum <= -self.signal_threshold
                and short_map.get(symbol, 0.0) < 0.0
                and self._entry_volume_ok(symbol)
            ][: self.max_shorts]
            for symbol in shorts:
                targets[symbol] = "SHORT"
                target_allocations[symbol] = self._target_allocation_for_symbol(
                    symbol,
                    len(shorts),
                    "SHORT",
                )

        return targets, momentum_map, regime, broad_score, target_allocations

    def _effective_trailing_pct(self, symbol: str) -> float:
        base = float(self.trailing_exit_pct)
        if self.volatility_trailing_multiplier <= 0.0:
            return max(0.0, base)
        realized = self._realized_vol(symbol)
        if realized <= 0.0:
            return max(0.0, base)
        dynamic = float(realized) * float(self.volatility_trailing_multiplier)
        dynamic = max(float(self.min_dynamic_trailing_pct), dynamic)
        dynamic = min(float(self.max_dynamic_trailing_pct), dynamic)
        return max(0.0, dynamic)

    def _emit_signal(
        self,
        *,
        symbol: str,
        event_time: Any,
        signal_type: str,
        price: float,
        regime: str,
        broad_score: float,
        momentum: float | None,
        target_allocation: float = 0.0,
        exit_reason: str = "",
    ) -> None:
        stop_loss = None
        take_profit = None
        trailing_percent = None
        effective_trailing_pct = self._effective_trailing_pct(symbol)
        if signal_type == "LONG":
            if self.stop_loss_pct > 0.0:
                stop_loss = price * (1.0 - self.stop_loss_pct)
            if self.take_profit_pct > 0.0:
                take_profit = price * (1.0 + self.take_profit_pct)
            if effective_trailing_pct > 0.0:
                trailing_percent = effective_trailing_pct
        elif signal_type == "SHORT":
            if self.stop_loss_pct > 0.0:
                stop_loss = price * (1.0 + self.stop_loss_pct)
            if self.take_profit_pct > 0.0:
                take_profit = price * (1.0 - self.take_profit_pct)
            if effective_trailing_pct > 0.0:
                trailing_percent = effective_trailing_pct

        metadata: dict[str, Any] = {
            "strategy": "AdaptiveRegimeMomentumStrategy",
            "regime": regime,
            "broad_score": float(broad_score),
            "momentum": float(momentum) if momentum is not None else None,
            "lookback_bars": int(self.lookback_bars),
            "short_lookback_bars": int(self.short_lookback_bars),
            "regime_lookback_bars": int(self.regime_lookback_bars),
            "signal_threshold": float(self.signal_threshold),
            "gross_exposure": float(self.gross_exposure),
            "long_exposure_multiplier": float(self.long_exposure_multiplier),
            "short_exposure_multiplier": float(self.short_exposure_multiplier),
            "max_order_value": float(self.max_order_value),
            "volatility_target_per_bar": float(self.volatility_target_per_bar),
            "volatility_exposure_multiplier": float(self._volatility_exposure_multiplier(symbol)),
            "effective_trailing_pct": float(effective_trailing_pct),
            "volume_z": float(self._volume_z(symbol)),
            "volume_weighted_broad": bool(self.volume_weighted_broad),
        }
        if target_allocation > 0.0:
            metadata["target_allocation"] = float(target_allocation)
            metadata["max_symbol_exposure_pct"] = float(target_allocation)
        if self.max_order_value > 0.0 and signal_type in {"LONG", "SHORT"}:
            metadata["max_order_value"] = float(self.max_order_value)
        if exit_reason:
            metadata["exit_reason"] = exit_reason

        self.events.put(
            SignalEvent(
                strategy_id="adaptive_regime_momentum",
                symbol=symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=float(target_allocation if target_allocation > 0.0 else 1.0),
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
                trailing_percent=trailing_percent,
            )
        )

    def _current_price(self, symbol: str) -> float | None:
        history = self._price_history.get(symbol)
        if history:
            return float(history[-1])
        return self._latest_bar_close(symbol)

    def _entry(self, symbol: str, side: str, price: float) -> None:
        self._position_state[symbol] = side
        self._entry_price[symbol] = float(price)
        self._high_watermark[symbol] = float(price)
        self._low_watermark[symbol] = float(price)
        self._bars_held[symbol] = 0

    def _exit(self, symbol: str) -> None:
        self._position_state[symbol] = "OUT"
        self._entry_price[symbol] = 0.0
        self._high_watermark[symbol] = 0.0
        self._low_watermark[symbol] = 0.0
        self._bars_held[symbol] = 0

    def _exit_reason(self, symbol: str, price: float, regime: str) -> str:
        state = self._position_state.get(symbol, "OUT")
        if state == "OUT":
            return ""

        self._bars_held[symbol] = int(self._bars_held.get(symbol, 0) or 0) + 1
        entry_price = float(self._entry_price.get(symbol, 0.0) or 0.0)
        if entry_price <= 0.0:
            return "missing_entry_price"

        self._high_watermark[symbol] = max(
            float(self._high_watermark.get(symbol, price) or price), price
        )
        self._low_watermark[symbol] = min(
            float(self._low_watermark.get(symbol, price) or price), price
        )

        if state == "LONG":
            if self.risk_off_exit and regime == "RISK_OFF":
                return "regime_flip"
            if self.stop_loss_pct > 0.0 and price <= entry_price * (1.0 - self.stop_loss_pct):
                return "stop_loss"
            if self.take_profit_pct > 0.0 and price >= entry_price * (1.0 + self.take_profit_pct):
                return "take_profit"
            if (
                (trailing_pct := self._effective_trailing_pct(symbol)) > 0.0
                and self._high_watermark[symbol] > entry_price
                and price <= self._high_watermark[symbol] * (1.0 - trailing_pct)
            ):
                return "trailing_exit"
        elif state == "SHORT":
            if self.risk_off_exit and regime == "RISK_ON":
                return "regime_flip"
            if self.stop_loss_pct > 0.0 and price >= entry_price * (1.0 + self.stop_loss_pct):
                return "stop_loss"
            if self.take_profit_pct > 0.0 and price <= entry_price * (1.0 - self.take_profit_pct):
                return "take_profit"
            if (
                (trailing_pct := self._effective_trailing_pct(symbol)) > 0.0
                and self._low_watermark[symbol] < entry_price
                and price >= self._low_watermark[symbol] * (1.0 + trailing_pct)
            ):
                return "trailing_exit"

        if (
            self.max_hold_bars > 0
            and int(self._bars_held.get(symbol, 0) or 0) >= self.max_hold_bars
        ):
            return "max_hold"
        return ""

    def _process_decision_bar(self, event_time: Any, event_time_key: str) -> None:
        if event_time_key == self._last_eval_time_key:
            return
        minimum_history = max(
            self.lookback_bars, self.short_lookback_bars, self.regime_lookback_bars
        )
        aligned_symbols = [
            symbol
            for symbol in self.symbol_list
            if self._last_symbol_time_key.get(symbol) == event_time_key
            and len(self._price_history[symbol]) > minimum_history
        ]
        if not aligned_symbols:
            return

        self._last_eval_time_key = event_time_key
        self._tick += 1
        targets, momentum_map, regime, broad_score, target_allocations = self._build_targets()

        # Risk controls are evaluated every decision bar; fresh entries only on
        # rebalance bars.
        for symbol in self.symbol_list:
            state = self._position_state.get(symbol, "OUT")
            if state == "OUT":
                continue
            price = self._current_price(symbol)
            if price is None:
                continue
            reason = self._exit_reason(symbol, price, regime)
            if reason:
                self._emit_signal(
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=price,
                    regime=regime,
                    broad_score=broad_score,
                    momentum=momentum_map.get(symbol),
                    exit_reason=reason,
                )
                self._exit(symbol)

        if self._tick % self.rebalance_bars != 0:
            return

        for symbol in self.symbol_list:
            state = self._position_state.get(symbol, "OUT")
            target_state = targets.get(symbol, "OUT")
            if state == target_state:
                continue
            price = self._current_price(symbol)
            if price is None:
                continue
            if state != "OUT":
                self._emit_signal(
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=price,
                    regime=regime,
                    broad_score=broad_score,
                    momentum=momentum_map.get(symbol),
                    exit_reason="rebalance",
                )
                self._exit(symbol)
            if target_state in {"LONG", "SHORT"}:
                self._emit_signal(
                    symbol=symbol,
                    event_time=event_time,
                    signal_type=target_state,
                    price=price,
                    regime=regime,
                    broad_score=broad_score,
                    momentum=momentum_map.get(symbol),
                    target_allocation=target_allocations.get(symbol, 0.0),
                )
                self._entry(symbol, target_state, price)

        LOGGER.info(
            "AdaptiveRegimeMomentum rebalance @%s regime=%s broad=%.4f long=%d short=%d",
            event_time_key,
            regime,
            broad_score,
            sum(1 for state in self._position_state.values() if state == "LONG"),
            sum(1 for state in self._position_state.values() if state == "SHORT"),
        )

    def calculate_signals_window(self, event: Any, aggregator: Any) -> None:
        _ = aggregator
        if str(getattr(event, "type", "")).upper() != "MARKET_WINDOW":
            self.calculate_signals(event)
            return

        event_time = getattr(event, "time", None)
        event_time_key = time_key(event_time)
        if not event_time_key:
            return

        bars_1s = dict(getattr(event, "bars_1s", {}) or {})
        for symbol in self.symbol_list:
            rows = list(bars_1s.get(symbol) or [])
            if not rows:
                continue
            close_price, quote_volume = self._window_close_and_quote_volume(rows)
            if close_price is None:
                close_price = self._latest_bar_close(symbol, rows[-1])
            if close_price is None:
                continue
            self._append_close(symbol, event_time_key, close_price, quote_volume)

        self._process_decision_bar(event_time, event_time_key)

    def calculate_signals(self, event: Any) -> None:
        if getattr(event, "type", None) != "MARKET":
            return
        event_symbol = str(getattr(event, "symbol", "") or "")
        if event_symbol not in self._price_history:
            return
        event_time = getattr(event, "time", None) or self.bars.get_latest_bar_datetime(event_symbol)
        event_time_key = time_key(event_time)
        close_price = self._latest_bar_close(event_symbol, event)
        if not event_time_key or close_price is None:
            return
        raw_volume = safe_float(getattr(event, "volume", None))
        quote_volume = (
            close_price * raw_volume if raw_volume is not None and raw_volume >= 0.0 else None
        )
        if not self._append_close(event_symbol, event_time_key, close_price, quote_volume):
            return
        self._process_decision_bar(event_time, event_time_key)


__all__ = ["AdaptiveRegimeMomentumStrategy"]
