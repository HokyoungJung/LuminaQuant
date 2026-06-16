"""Derivatives-flow squeeze/exhaustion alpha.

The strategy intentionally lives on the MARKET_WINDOW contract so live and
live-equivalent backtests consume the same cadence as the existing portfolio
mode runtime.  It combines price impulse, taker-flow imbalance, open-interest
expansion, funding crowding filters, liquidation exhaustion confirmation, and a
simple realized-volatility sizing governor.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from lumina_quant.core.events import SignalEvent
from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import simple_return
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.rolling_stats import rolling_beta
from lumina_quant.strategies.external_alpha_sleeves import _emit, _target_metadata
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@dataclass(slots=True)
class _SymbolState:
    opens: deque[float]
    highs: deque[float]
    lows: deque[float]
    closes: deque[float]
    quote_volume: deque[float]
    taker_buy_quote_volume: deque[float]
    taker_sell_quote_volume: deque[float]
    funding_rate: deque[float]
    open_interest: deque[float]
    liquidation_long_notional: deque[float]
    liquidation_short_notional: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    high_watermark: float | None = None
    low_watermark: float | None = None
    bars_held: int = 0
    ticks_seen: int = 0
    last_time_key: str = ""
    last_flow_source: str = "missing"
    has_funding: bool = False
    has_open_interest: bool = False
    has_liquidation: bool = False
    has_taker_flow: bool = False
    last_open_interest_source: str = "missing"


class DerivativesFlowSqueezeStrategy(Strategy):
    """Funding/OI/taker-flow/liquidation alpha with volatility-managed sizing."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False
    required_features = (
        "funding_rate",
        "open_interest",
        "liquidation_long_notional",
        "liquidation_short_notional",
        "taker_buy_quote_volume",
        "taker_sell_quote_volume",
    )

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "lookback_bars": HyperParam.integer("lookback_bars", default=192, low=16, high=10_080),
            "momentum_lookback_bars": HyperParam.integer(
                "momentum_lookback_bars", default=48, low=2, high=2_880
            ),
            "short_reclaim_bars": HyperParam.integer(
                "short_reclaim_bars", default=6, low=1, high=360
            ),
            "flow_lookback_bars": HyperParam.integer(
                "flow_lookback_bars", default=96, low=4, high=2_880
            ),
            "oi_lookback_bars": HyperParam.integer(
                "oi_lookback_bars", default=96, low=2, high=2_880
            ),
            "liquidation_window_bars": HyperParam.integer(
                "liquidation_window_bars", default=192, low=8, high=4_320
            ),
            "volatility_lookback_bars": HyperParam.integer(
                "volatility_lookback_bars", default=90, low=4, high=2_880
            ),
            "evaluation_cadence_bars": HyperParam.integer(
                "evaluation_cadence_bars", default=45, low=1, high=10_080, tunable=False
            ),
            "continuation_momentum_min": HyperParam.floating(
                "continuation_momentum_min", default=0.0015, low=0.0, high=0.20
            ),
            "flow_imbalance_min": HyperParam.floating(
                "flow_imbalance_min", default=0.08, low=0.0, high=1.0
            ),
            "oi_delta_min": HyperParam.floating(
                "oi_delta_min", default=0.0002, low=-0.20, high=0.20
            ),
            "oi_delta_z_min": HyperParam.floating(
                "oi_delta_z_min", default=-0.25, low=-5.0, high=5.0
            ),
            "max_abs_continuation_funding": HyperParam.floating(
                "max_abs_continuation_funding", default=0.0015, low=0.0, high=0.05
            ),
            "liquidation_z_min": HyperParam.floating(
                "liquidation_z_min", default=2.0, low=0.0, high=20.0
            ),
            "liquidation_notional_min": HyperParam.floating(
                "liquidation_notional_min", default=1.0, low=0.0, high=1_000_000_000.0
            ),
            "price_shock_min": HyperParam.floating(
                "price_shock_min", default=0.003, low=0.0, high=0.5
            ),
            "reclaim_min": HyperParam.floating("reclaim_min", default=0.0006, low=0.0, high=0.20),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.012, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=300.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "volatility_target_per_bar": HyperParam.floating(
                "volatility_target_per_bar", default=0.0012, low=0.0, high=0.10
            ),
            "min_volatility_multiplier": HyperParam.floating(
                "min_volatility_multiplier", default=0.30, low=0.0, high=5.0
            ),
            "max_volatility_multiplier": HyperParam.floating(
                "max_volatility_multiplier", default=1.0, low=0.0, high=5.0
            ),
            "volatility_hard_cap": HyperParam.floating(
                "volatility_hard_cap", default=0.020, low=0.0, high=1.0
            ),
            "funding_overheat_abs": HyperParam.floating(
                "funding_overheat_abs", default=0.004, low=0.0, high=0.10
            ),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.018, low=0.0, high=0.5),
            "take_profit_pct": HyperParam.floating(
                "take_profit_pct", default=0.040, low=0.0, high=1.0
            ),
            "trailing_exit_pct": HyperParam.floating(
                "trailing_exit_pct", default=0.018, low=0.0, high=0.5
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=288, low=1, high=100_000),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "enable_continuation": HyperParam.boolean(
                "enable_continuation", default=True, grid=[True, False]
            ),
            "enable_exhaustion": HyperParam.boolean(
                "enable_exhaustion", default=True, grid=[True, False]
            ),
            "allow_ohlcv_flow_proxy": HyperParam.boolean(
                "allow_ohlcv_flow_proxy", default=True, grid=[True, False], tunable=False
            ),
            "allow_volume_oi_proxy": HyperParam.boolean(
                "allow_volume_oi_proxy", default=True, grid=[True, False], tunable=False
            ),
        }

    def __init__(
        self,
        bars,
        events,
        lookback_bars: int = 192,
        momentum_lookback_bars: int = 48,
        short_reclaim_bars: int = 6,
        flow_lookback_bars: int = 96,
        oi_lookback_bars: int = 96,
        liquidation_window_bars: int = 192,
        volatility_lookback_bars: int = 90,
        evaluation_cadence_bars: int = 45,
        continuation_momentum_min: float = 0.0015,
        flow_imbalance_min: float = 0.08,
        oi_delta_min: float = 0.0002,
        oi_delta_z_min: float = -0.25,
        max_abs_continuation_funding: float = 0.0015,
        liquidation_z_min: float = 2.0,
        liquidation_notional_min: float = 1.0,
        price_shock_min: float = 0.003,
        reclaim_min: float = 0.0006,
        target_allocation: float = 0.012,
        max_order_value: float = 300.0,
        volatility_target_per_bar: float = 0.0012,
        min_volatility_multiplier: float = 0.30,
        max_volatility_multiplier: float = 1.0,
        volatility_hard_cap: float = 0.020,
        funding_overheat_abs: float = 0.004,
        stop_loss_pct: float = 0.018,
        take_profit_pct: float = 0.040,
        trailing_exit_pct: float = 0.018,
        max_hold_bars: int = 288,
        min_price: float = 0.10,
        allow_short: bool = True,
        enable_continuation: bool = True,
        enable_exhaustion: bool = True,
        allow_ohlcv_flow_proxy: bool = True,
        allow_volume_oi_proxy: bool = True,
    ) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])

        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "lookback_bars": lookback_bars,
                "momentum_lookback_bars": momentum_lookback_bars,
                "short_reclaim_bars": short_reclaim_bars,
                "flow_lookback_bars": flow_lookback_bars,
                "oi_lookback_bars": oi_lookback_bars,
                "liquidation_window_bars": liquidation_window_bars,
                "volatility_lookback_bars": volatility_lookback_bars,
                "evaluation_cadence_bars": evaluation_cadence_bars,
                "continuation_momentum_min": continuation_momentum_min,
                "flow_imbalance_min": flow_imbalance_min,
                "oi_delta_min": oi_delta_min,
                "oi_delta_z_min": oi_delta_z_min,
                "max_abs_continuation_funding": max_abs_continuation_funding,
                "liquidation_z_min": liquidation_z_min,
                "liquidation_notional_min": liquidation_notional_min,
                "price_shock_min": price_shock_min,
                "reclaim_min": reclaim_min,
                "target_allocation": target_allocation,
                "max_order_value": max_order_value,
                "volatility_target_per_bar": volatility_target_per_bar,
                "min_volatility_multiplier": min_volatility_multiplier,
                "max_volatility_multiplier": max_volatility_multiplier,
                "volatility_hard_cap": volatility_hard_cap,
                "funding_overheat_abs": funding_overheat_abs,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "trailing_exit_pct": trailing_exit_pct,
                "max_hold_bars": max_hold_bars,
                "min_price": min_price,
                "allow_short": allow_short,
                "enable_continuation": enable_continuation,
                "enable_exhaustion": enable_exhaustion,
                "allow_ohlcv_flow_proxy": allow_ohlcv_flow_proxy,
                "allow_volume_oi_proxy": allow_volume_oi_proxy,
            },
            keep_unknown=False,
        )
        self.lookback_bars = max(8, int(resolved["lookback_bars"]))
        self.momentum_lookback_bars = max(1, int(resolved["momentum_lookback_bars"]))
        self.short_reclaim_bars = max(1, int(resolved["short_reclaim_bars"]))
        self.flow_lookback_bars = max(1, int(resolved["flow_lookback_bars"]))
        self.oi_lookback_bars = max(1, int(resolved["oi_lookback_bars"]))
        self.liquidation_window_bars = max(2, int(resolved["liquidation_window_bars"]))
        self.volatility_lookback_bars = max(2, int(resolved["volatility_lookback_bars"]))
        self.evaluation_cadence_bars = max(1, int(resolved["evaluation_cadence_bars"]))
        self.continuation_momentum_min = max(0.0, float(resolved["continuation_momentum_min"]))
        self.flow_imbalance_min = max(0.0, float(resolved["flow_imbalance_min"]))
        self.oi_delta_min = float(resolved["oi_delta_min"])
        self.oi_delta_z_min = float(resolved["oi_delta_z_min"])
        self.max_abs_continuation_funding = max(
            0.0, float(resolved["max_abs_continuation_funding"])
        )
        self.liquidation_z_min = max(0.0, float(resolved["liquidation_z_min"]))
        self.liquidation_notional_min = max(0.0, float(resolved["liquidation_notional_min"]))
        self.price_shock_min = max(0.0, float(resolved["price_shock_min"]))
        self.reclaim_min = max(0.0, float(resolved["reclaim_min"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.volatility_target_per_bar = max(0.0, float(resolved["volatility_target_per_bar"]))
        self.min_volatility_multiplier = max(0.0, float(resolved["min_volatility_multiplier"]))
        self.max_volatility_multiplier = max(
            self.min_volatility_multiplier, float(resolved["max_volatility_multiplier"])
        )
        self.volatility_hard_cap = max(0.0, float(resolved["volatility_hard_cap"]))
        self.funding_overheat_abs = max(0.0, float(resolved["funding_overheat_abs"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.take_profit_pct = max(0.0, float(resolved["take_profit_pct"]))
        self.trailing_exit_pct = max(0.0, float(resolved["trailing_exit_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        self.allow_short = bool(resolved["allow_short"])
        self.enable_continuation = bool(resolved["enable_continuation"])
        self.enable_exhaustion = bool(resolved["enable_exhaustion"])
        self.allow_ohlcv_flow_proxy = bool(resolved["allow_ohlcv_flow_proxy"])
        self.allow_volume_oi_proxy = bool(resolved["allow_volume_oi_proxy"])

        size = (
            max(
                self.lookback_bars,
                self.momentum_lookback_bars,
                self.flow_lookback_bars,
                self.oi_lookback_bars,
                self.liquidation_window_bars,
                self.volatility_lookback_bars,
            )
            + 8
        )
        self._state = {
            symbol: _SymbolState(
                opens=deque(maxlen=size),
                highs=deque(maxlen=size),
                lows=deque(maxlen=size),
                closes=deque(maxlen=size),
                quote_volume=deque(maxlen=size),
                taker_buy_quote_volume=deque(maxlen=size),
                taker_sell_quote_volume=deque(maxlen=size),
                funding_rate=deque(maxlen=size),
                open_interest=deque(maxlen=size),
                liquidation_long_notional=deque(maxlen=size),
                liquidation_short_notional=deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "symbol_state": {
                symbol: {
                    "opens": list(item.opens),
                    "highs": list(item.highs),
                    "lows": list(item.lows),
                    "closes": list(item.closes),
                    "quote_volume": list(item.quote_volume),
                    "taker_buy_quote_volume": list(item.taker_buy_quote_volume),
                    "taker_sell_quote_volume": list(item.taker_sell_quote_volume),
                    "funding_rate": list(item.funding_rate),
                    "open_interest": list(item.open_interest),
                    "liquidation_long_notional": list(item.liquidation_long_notional),
                    "liquidation_short_notional": list(item.liquidation_short_notional),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "high_watermark": item.high_watermark,
                    "low_watermark": item.low_watermark,
                    "bars_held": int(item.bars_held),
                    "ticks_seen": int(item.ticks_seen),
                    "last_time_key": item.last_time_key,
                    "last_flow_source": item.last_flow_source,
                    "has_funding": bool(item.has_funding),
                    "has_open_interest": bool(item.has_open_interest),
                    "has_liquidation": bool(item.has_liquidation),
                    "has_taker_flow": bool(item.has_taker_flow),
                    "last_open_interest_source": item.last_open_interest_source,
                }
                for symbol, item in self._state.items()
            }
        }

    def set_state(self, state: dict[str, Any]) -> None:
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
                "opens",
                "highs",
                "lows",
                "closes",
                "quote_volume",
                "taker_buy_quote_volume",
                "taker_sell_quote_volume",
                "funding_rate",
                "open_interest",
                "liquidation_long_notional",
                "liquidation_short_notional",
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
            item.high_watermark = safe_float(payload.get("high_watermark"))
            item.low_watermark = safe_float(payload.get("low_watermark"))
            item.bars_held = self._safe_non_negative_int(payload.get("bars_held"))
            item.ticks_seen = self._safe_non_negative_int(payload.get("ticks_seen"))
            item.last_time_key = str(payload.get("last_time_key", ""))
            item.last_flow_source = str(payload.get("last_flow_source") or "missing")
            item.has_funding = bool(payload.get("has_funding", False))
            item.has_open_interest = bool(payload.get("has_open_interest", False))
            item.has_liquidation = bool(payload.get("has_liquidation", False))
            item.has_taker_flow = bool(payload.get("has_taker_flow", False))
            item.last_open_interest_source = str(
                payload.get("last_open_interest_source") or "missing"
            )

    @staticmethod
    def _safe_non_negative_int(value: Any) -> int:
        try:
            return max(0, int(value))
        except Exception:
            return 0

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

    def _window_snapshot(self, event: Any, symbol: str) -> dict[str, float | None]:
        bars_1s = dict(getattr(event, "bars_1s", {}) or {})
        rows = list(bars_1s.get(symbol) or [])
        if not rows:
            return {}
        mapped = [self._as_row_dict(row) for row in rows]
        mapped = [row for row in mapped if row]
        if not mapped:
            return {}
        first = mapped[0]
        last = mapped[-1]
        highs = [value for row in mapped if (value := safe_float(row.get("high"))) is not None]
        lows = [value for row in mapped if (value := safe_float(row.get("low"))) is not None]

        def _sum_field(field: str) -> float | None:
            values = [value for row in mapped if (value := safe_float(row.get(field))) is not None]
            if not values:
                return None
            return float(sum(values))

        return {
            "open": safe_float(first.get("open")),
            "high": max(highs) if highs else safe_float(last.get("high")),
            "low": min(lows) if lows else safe_float(last.get("low")),
            "close": safe_float(last.get("close")),
            "volume": _sum_field("volume"),
            "taker_buy_quote_volume": _sum_field("taker_buy_quote_volume"),
            "taker_sell_quote_volume": _sum_field("taker_sell_quote_volume"),
            "taker_buy_base_volume": _sum_field("taker_buy_base_volume"),
            "taker_sell_base_volume": _sum_field("taker_sell_base_volume"),
        }

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
        getter = getattr(self.bars, "get_latest_bar_value", None)
        if callable(getter):
            try:
                value = getter(symbol, field)
            except Exception:
                value = None
            return safe_float(value)
        return None

    def _resolve_flow_values(
        self,
        event: Any,
        symbol: str,
        snapshot: dict[str, float | None],
        close: float,
    ) -> tuple[float, float, str]:
        buy_quote = self._extract_feature(event, symbol, "taker_buy_quote_volume")
        sell_quote = self._extract_feature(event, symbol, "taker_sell_quote_volume")
        source = "feature_quote"
        if buy_quote is None:
            buy_quote = snapshot.get("taker_buy_quote_volume")
        if sell_quote is None:
            sell_quote = snapshot.get("taker_sell_quote_volume")
        if (
            buy_quote is not None
            and sell_quote is not None
            and (float(buy_quote) + float(sell_quote)) > 0.0
        ):
            return max(0.0, float(buy_quote)), max(0.0, float(sell_quote)), source

        buy_base = self._extract_feature(event, symbol, "taker_buy_base_volume")
        sell_base = self._extract_feature(event, symbol, "taker_sell_base_volume")
        source = "feature_base"
        if buy_base is None:
            buy_base = snapshot.get("taker_buy_base_volume")
        if sell_base is None:
            sell_base = snapshot.get("taker_sell_base_volume")
        if (
            buy_base is not None
            and sell_base is not None
            and (float(buy_base) + float(sell_base)) > 0.0
        ):
            return max(0.0, float(buy_base) * close), max(0.0, float(sell_base) * close), source

        if not self.allow_ohlcv_flow_proxy:
            return 0.0, 0.0, "missing"

        open_value = snapshot.get("open")
        high = snapshot.get("high")
        low = snapshot.get("low")
        volume = snapshot.get("volume")
        if open_value is None:
            open_value = self._extract_feature(event, symbol, "open")
        if high is None:
            high = self._extract_feature(event, symbol, "high")
        if low is None:
            low = self._extract_feature(event, symbol, "low")
        if volume is None:
            volume = self._extract_feature(event, symbol, "volume")
        if volume is None or close <= 0.0:
            return 0.0, 0.0, "missing"

        quote = max(0.0, float(volume) * close)
        if open_value is None or high is None or low is None:
            return quote * 0.5, quote * 0.5, "ohlcv_even_proxy"
        bar_range = max(abs(float(high) - float(low)), close * 1e-9, 1e-12)
        pressure = self._clamp((close - float(open_value)) / bar_range, -1.0, 1.0)
        buy_fraction = self._clamp(0.5 + 0.45 * pressure, 0.05, 0.95)
        return quote * buy_fraction, quote * (1.0 - buy_fraction), "ohlcv_directional_proxy"

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
        for prev, cur in pairwise(subset):
            if prev > 0.0 and cur > 0.0:
                returns.append(math.log(cur / prev))
        if not returns:
            return 0.0
        return math.sqrt(sum(value * value for value in returns) / len(returns))

    @staticmethod
    def _zscore_latest(values: deque[float], window: int) -> float:
        if len(values) < 8:
            return 0.0
        subset = list(values)[-max(8, window) :]
        if len(subset) < 8:
            return 0.0
        latest = float(subset[-1])
        history = subset[:-1]
        mean = sum(history) / len(history)
        variance = sum((value - mean) ** 2 for value in history) / max(1, len(history) - 1)
        sigma = math.sqrt(max(variance, 0.0))
        if sigma <= 1e-12:
            if latest > mean:
                return 10.0
            if latest < mean:
                return -10.0
            return 0.0
        return (latest - mean) / sigma

    @staticmethod
    def _oi_delta_z(values: deque[float], lookback: int, window: int) -> float:
        if len(values) <= lookback + 8:
            return 0.0
        series = list(values)
        deltas: list[float] = []
        start = max(lookback, len(series) - max(window, 8))
        for idx in range(start, len(series)):
            base = float(series[idx - lookback])
            if abs(base) <= 1e-12:
                continue
            deltas.append(float(series[idx]) / base - 1.0)
        if len(deltas) < 8:
            return 0.0
        latest = deltas[-1]
        history = deltas[:-1]
        mean = sum(history) / len(history)
        variance = sum((value - mean) ** 2 for value in history) / max(1, len(history) - 1)
        sigma = math.sqrt(max(variance, 0.0))
        if sigma <= 1e-12:
            return 0.0
        return (latest - mean) / sigma

    def _flow_imbalance(self, item: _SymbolState) -> float:
        if not item.taker_buy_quote_volume or not item.taker_sell_quote_volume:
            return 0.0
        buy = sum(list(item.taker_buy_quote_volume)[-self.flow_lookback_bars :])
        sell = sum(list(item.taker_sell_quote_volume)[-self.flow_lookback_bars :])
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
        item: _SymbolState,
        *,
        symbol: str,
        reason: str,
        price_momentum: float,
        short_momentum: float,
        flow_imbalance: float,
        oi_delta: float,
        oi_delta_z: float,
        funding: float,
        long_liq_z: float,
        short_liq_z: float,
        realized_vol: float,
        vol_multiplier: float,
        target_allocation: float,
    ) -> dict[str, Any]:
        return {
            "strategy": "DerivativesFlowSqueezeStrategy",
            "symbol": symbol,
            "reason": reason,
            "price_momentum": float(price_momentum),
            "short_momentum": float(short_momentum),
            "flow_imbalance": float(flow_imbalance),
            "oi_delta": float(oi_delta),
            "oi_delta_z": float(oi_delta_z),
            "funding_rate": float(funding),
            "liquidation_long_z": float(long_liq_z),
            "liquidation_short_z": float(short_liq_z),
            "realized_vol_per_bar": float(realized_vol),
            "volatility_multiplier": float(vol_multiplier),
            "target_allocation": float(target_allocation),
            "max_symbol_exposure_pct": float(target_allocation),
            "max_order_value": float(self.max_order_value),
            "feature_coverage": {
                "funding_rate": bool(item.has_funding),
                "open_interest": bool(item.has_open_interest),
                "liquidation": bool(item.has_liquidation),
                "taker_flow": bool(item.has_taker_flow),
                "flow_source": item.last_flow_source,
                "open_interest_source": item.last_open_interest_source,
            },
            "bars_held": int(item.bars_held),
        }

    def _emit(
        self,
        symbol: str,
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
                strategy_id="derivatives_flow_squeeze",
                symbol=symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=float(strength),
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=dict(metadata or {}),
                trailing_percent=trailing_percent,
            )
        )

    def _maybe_exit(
        self,
        event: Any,
        symbol: str,
        item: _SymbolState,
        close: float,
        metrics: dict[str, float],
    ) -> bool:
        if item.mode not in {"LONG", "SHORT"}:
            return False
        item.bars_held += 1
        if item.mode == "LONG":
            item.high_watermark = max(float(item.high_watermark or close), close)
            stop_hit = (
                item.entry_price is not None
                and self.stop_loss_pct > 0.0
                and close <= item.entry_price * (1.0 - self.stop_loss_pct)
            )
            take_hit = (
                item.entry_price is not None
                and self.take_profit_pct > 0.0
                and close >= item.entry_price * (1.0 + self.take_profit_pct)
            )
            trail_hit = (
                self.trailing_exit_pct > 0.0
                and item.high_watermark is not None
                and close <= item.high_watermark * (1.0 - self.trailing_exit_pct)
            )
            opposing_flow = (
                metrics["price_momentum"] < -self.continuation_momentum_min
                and metrics["flow_imbalance"] < -self.flow_imbalance_min
            )
            funding_overheat = (
                self.funding_overheat_abs > 0.0 and metrics["funding"] > self.funding_overheat_abs
            )
        else:
            item.low_watermark = min(float(item.low_watermark or close), close)
            stop_hit = (
                item.entry_price is not None
                and self.stop_loss_pct > 0.0
                and close >= item.entry_price * (1.0 + self.stop_loss_pct)
            )
            take_hit = (
                item.entry_price is not None
                and self.take_profit_pct > 0.0
                and close <= item.entry_price * (1.0 - self.take_profit_pct)
            )
            trail_hit = (
                self.trailing_exit_pct > 0.0
                and item.low_watermark is not None
                and close >= item.low_watermark * (1.0 + self.trailing_exit_pct)
            )
            opposing_flow = (
                metrics["price_momentum"] > self.continuation_momentum_min
                and metrics["flow_imbalance"] > self.flow_imbalance_min
            )
            funding_overheat = (
                self.funding_overheat_abs > 0.0 and metrics["funding"] < -self.funding_overheat_abs
            )

        max_hold = item.bars_held >= self.max_hold_bars
        hard_vol = (
            self.volatility_hard_cap > 0.0 and metrics["realized_vol"] > self.volatility_hard_cap
        )
        reasons = [
            name
            for name, flag in (
                ("stop", stop_hit),
                ("take_profit", take_hit),
                ("trailing", trail_hit),
                ("max_hold", max_hold),
                ("opposing_flow", opposing_flow),
                ("funding_overheat", funding_overheat),
                ("hard_volatility", hard_vol),
            )
            if flag
        ]
        if not reasons:
            return False

        metadata = self._metadata(
            item,
            symbol=symbol,
            reason=f"{item.mode.lower()}_exit:{'+'.join(reasons)}",
            price_momentum=metrics["price_momentum"],
            short_momentum=metrics["short_momentum"],
            flow_imbalance=metrics["flow_imbalance"],
            oi_delta=metrics["oi_delta"],
            oi_delta_z=metrics["oi_delta_z"],
            funding=metrics["funding"],
            long_liq_z=metrics["long_liq_z"],
            short_liq_z=metrics["short_liq_z"],
            realized_vol=metrics["realized_vol"],
            vol_multiplier=metrics["vol_multiplier"],
            target_allocation=0.0,
        )
        metadata.pop("target_allocation", None)
        metadata.pop("max_symbol_exposure_pct", None)
        metadata.pop("max_order_value", None)
        self._emit(symbol, getattr(event, "time", None), "EXIT", price=close, metadata=metadata)
        item.mode = "OUT"
        item.entry_price = None
        item.high_watermark = None
        item.low_watermark = None
        item.bars_held = 0
        return True

    def _enter(
        self,
        event: Any,
        symbol: str,
        item: _SymbolState,
        signal_type: str,
        close: float,
        reason: str,
        metrics: dict[str, float],
        raw_strength: float,
    ) -> None:
        vol_multiplier = metrics["vol_multiplier"]
        target_allocation = self.target_allocation * vol_multiplier
        if target_allocation <= 0.0 or self.max_order_value <= 0.0:
            return
        strength = self._clamp(raw_strength, 0.05, 2.0)
        metadata = self._metadata(
            item,
            symbol=symbol,
            reason=reason,
            price_momentum=metrics["price_momentum"],
            short_momentum=metrics["short_momentum"],
            flow_imbalance=metrics["flow_imbalance"],
            oi_delta=metrics["oi_delta"],
            oi_delta_z=metrics["oi_delta_z"],
            funding=metrics["funding"],
            long_liq_z=metrics["long_liq_z"],
            short_liq_z=metrics["short_liq_z"],
            realized_vol=metrics["realized_vol"],
            vol_multiplier=vol_multiplier,
            target_allocation=target_allocation,
        )
        if signal_type == "LONG":
            stop_loss = close * (1.0 - self.stop_loss_pct) if self.stop_loss_pct > 0.0 else None
            take_profit = (
                close * (1.0 + self.take_profit_pct) if self.take_profit_pct > 0.0 else None
            )
            item.mode = "LONG"
            item.high_watermark = close
            item.low_watermark = None
        else:
            stop_loss = close * (1.0 + self.stop_loss_pct) if self.stop_loss_pct > 0.0 else None
            take_profit = (
                close * (1.0 - self.take_profit_pct) if self.take_profit_pct > 0.0 else None
            )
            item.mode = "SHORT"
            item.low_watermark = close
            item.high_watermark = None
        self._emit(
            symbol,
            getattr(event, "time", None),
            signal_type,
            strength=strength,
            price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_percent=self.trailing_exit_pct if self.trailing_exit_pct > 0.0 else None,
            metadata=metadata,
        )
        item.entry_price = close
        item.bars_held = 0

    def _process_symbol(
        self, event: Any, symbol: str, *, snapshot: dict[str, float | None] | None = None
    ) -> None:
        item = self._state[symbol]
        key = time_key(getattr(event, "time", getattr(event, "datetime", None)))
        if key and key == item.last_time_key:
            return
        item.last_time_key = key

        snapshot = dict(snapshot or {})
        close = snapshot.get("close")
        if close is None:
            close = self._extract_feature(event, symbol, "close")
        if close is None:
            close = safe_float(getattr(event, "close", None))
        if close is None or close <= self.min_price:
            return
        open_value = snapshot.get("open")
        high = snapshot.get("high")
        low = snapshot.get("low")
        volume = snapshot.get("volume")
        if open_value is None:
            open_value = self._extract_feature(event, symbol, "open") or close
        if high is None:
            high = self._extract_feature(event, symbol, "high") or close
        if low is None:
            low = self._extract_feature(event, symbol, "low") or close
        if volume is None:
            volume = self._extract_feature(event, symbol, "volume") or 0.0

        funding = self._extract_feature(event, symbol, "funding_rate")
        open_interest = self._extract_feature(event, symbol, "open_interest")
        liq_long = self._extract_feature(event, symbol, "liquidation_long_notional")
        liq_short = self._extract_feature(event, symbol, "liquidation_short_notional")
        if open_interest is not None and open_interest <= 0.0:
            open_interest = None
        item.has_funding = item.has_funding or funding is not None
        item.has_open_interest = item.has_open_interest or open_interest is not None
        item.has_liquidation = item.has_liquidation or (
            (liq_long is not None and liq_long > 0.0) or (liq_short is not None and liq_short > 0.0)
        )

        buy_quote, sell_quote, flow_source = self._resolve_flow_values(
            event, symbol, snapshot, close
        )
        item.last_flow_source = flow_source
        item.has_taker_flow = item.has_taker_flow or flow_source.startswith("feature")
        quote_value = max(0.0, float(volume) * close)
        if open_interest is not None:
            open_interest_value = float(open_interest)
            item.last_open_interest_source = "feature"
        elif self.allow_volume_oi_proxy:
            open_interest_value = quote_value
            item.last_open_interest_source = "quote_volume_proxy"
        else:
            open_interest_value = 0.0
            item.last_open_interest_source = "missing"

        item.opens.append(float(open_value))
        item.highs.append(float(high))
        item.lows.append(float(low))
        item.closes.append(float(close))
        item.quote_volume.append(quote_value)
        item.taker_buy_quote_volume.append(float(buy_quote))
        item.taker_sell_quote_volume.append(float(sell_quote))
        item.funding_rate.append(float(funding) if funding is not None else 0.0)
        item.open_interest.append(open_interest_value)
        item.liquidation_long_notional.append(float(liq_long) if liq_long is not None else 0.0)
        item.liquidation_short_notional.append(float(liq_short) if liq_short is not None else 0.0)
        item.ticks_seen += 1

        if self.evaluation_cadence_bars > 1 and item.ticks_seen % self.evaluation_cadence_bars:
            return
        if len(item.closes) <= self.momentum_lookback_bars:
            return
        if (not item.has_open_interest and not self.allow_volume_oi_proxy) or not item.has_funding:
            return

        price_momentum = self._pct_change(item.closes, self.momentum_lookback_bars) or 0.0
        short_momentum = self._pct_change(item.closes, self.short_reclaim_bars) or 0.0
        flow_imbalance = self._flow_imbalance(item)
        oi_delta = self._pct_change(item.open_interest, self.oi_lookback_bars) or 0.0
        oi_delta_z = self._oi_delta_z(item.open_interest, self.oi_lookback_bars, self.lookback_bars)
        funding_value = float(item.funding_rate[-1]) if item.funding_rate else 0.0
        long_liq_z = self._zscore_latest(
            item.liquidation_long_notional, self.liquidation_window_bars
        )
        short_liq_z = self._zscore_latest(
            item.liquidation_short_notional, self.liquidation_window_bars
        )
        realized_vol = self._realized_vol(item.closes, self.volatility_lookback_bars)
        vol_multiplier = self._volatility_multiplier(realized_vol)
        metrics = {
            "price_momentum": float(price_momentum),
            "short_momentum": float(short_momentum),
            "flow_imbalance": float(flow_imbalance),
            "oi_delta": float(oi_delta),
            "oi_delta_z": float(oi_delta_z),
            "funding": float(funding_value),
            "long_liq_z": float(long_liq_z),
            "short_liq_z": float(short_liq_z),
            "realized_vol": float(realized_vol),
            "vol_multiplier": float(vol_multiplier),
        }
        if self._maybe_exit(event, symbol, item, close, metrics):
            return
        if item.mode != "OUT":
            return
        if self.volatility_hard_cap > 0.0 and realized_vol > self.volatility_hard_cap:
            return

        funding_ok = abs(funding_value) <= self.max_abs_continuation_funding
        oi_ok = oi_delta >= self.oi_delta_min and oi_delta_z >= self.oi_delta_z_min
        if self.enable_continuation and funding_ok and oi_ok:
            long_continuation = (
                price_momentum >= self.continuation_momentum_min
                and flow_imbalance >= self.flow_imbalance_min
            )
            short_continuation = (
                self.allow_short
                and price_momentum <= -self.continuation_momentum_min
                and flow_imbalance <= -self.flow_imbalance_min
            )
            if long_continuation:
                strength = (
                    0.35
                    + min(1.0, abs(price_momentum) / max(self.continuation_momentum_min, 1e-9))
                    * 0.35
                )
                strength += (
                    min(1.0, abs(flow_imbalance)) * 0.45 + max(0.0, min(1.0, oi_delta_z)) * 0.15
                )
                self._enter(
                    event, symbol, item, "LONG", close, "flow_continuation_long", metrics, strength
                )
                return
            if short_continuation:
                strength = (
                    0.35
                    + min(1.0, abs(price_momentum) / max(self.continuation_momentum_min, 1e-9))
                    * 0.35
                )
                strength += (
                    min(1.0, abs(flow_imbalance)) * 0.45 + max(0.0, min(1.0, oi_delta_z)) * 0.15
                )
                self._enter(
                    event,
                    symbol,
                    item,
                    "SHORT",
                    close,
                    "flow_continuation_short",
                    metrics,
                    strength,
                )
                return

        if not self.enable_exhaustion or not item.has_liquidation:
            return
        long_liq_latest = (
            float(item.liquidation_long_notional[-1]) if item.liquidation_long_notional else 0.0
        )
        short_liq_latest = (
            float(item.liquidation_short_notional[-1]) if item.liquidation_short_notional else 0.0
        )
        long_exhaustion = (
            long_liq_latest >= self.liquidation_notional_min
            and long_liq_z >= self.liquidation_z_min
            and price_momentum <= -self.price_shock_min
            and short_momentum >= self.reclaim_min
            and flow_imbalance >= -self.flow_imbalance_min * 0.5
        )
        short_exhaustion = (
            self.allow_short
            and short_liq_latest >= self.liquidation_notional_min
            and short_liq_z >= self.liquidation_z_min
            and price_momentum >= self.price_shock_min
            and short_momentum <= -self.reclaim_min
            and flow_imbalance <= self.flow_imbalance_min * 0.5
        )
        if long_exhaustion:
            strength = 0.55 + min(1.0, long_liq_z / max(self.liquidation_z_min, 1e-9)) * 0.65
            strength += min(1.0, max(0.0, short_momentum) / max(self.reclaim_min, 1e-9)) * 0.25
            self._enter(
                event, symbol, item, "LONG", close, "liquidation_exhaustion_long", metrics, strength
            )
            return
        if short_exhaustion:
            strength = 0.55 + min(1.0, short_liq_z / max(self.liquidation_z_min, 1e-9)) * 0.65
            strength += min(1.0, max(0.0, -short_momentum) / max(self.reclaim_min, 1e-9)) * 0.25
            self._enter(
                event,
                symbol,
                item,
                "SHORT",
                close,
                "liquidation_exhaustion_short",
                metrics,
                strength,
            )

    def calculate_signals_window(self, event: Any, aggregator: Any) -> None:
        _ = aggregator
        if str(getattr(event, "type", "")).upper() != "MARKET_WINDOW":
            self.calculate_signals(event)
            return
        for symbol in self.symbol_list:
            if symbol not in self._state:
                continue
            self._process_symbol(event, symbol, snapshot=self._window_snapshot(event, symbol))

    def calculate_signals(self, event: Any) -> None:
        event_type = str(getattr(event, "type", "")).upper()
        if event_type == "MARKET_WINDOW":
            self.calculate_signals_window(event, aggregator=None)
            return
        if event_type != "MARKET":
            return
        symbol = str(getattr(event, "symbol", ""))
        if symbol not in self._state:
            return
        snapshot = {
            "open": safe_float(getattr(event, "open", None)),
            "high": safe_float(getattr(event, "high", None)),
            "low": safe_float(getattr(event, "low", None)),
            "close": safe_float(getattr(event, "close", None)),
            "volume": safe_float(getattr(event, "volume", None)),
        }
        self._process_symbol(event, symbol, snapshot=snapshot)


_BTC_HINTS = ("BTCUSDT", "BTCUSD", "BTC")


@dataclass(slots=True)
class _CascadeSymbolState:
    closes: deque[float]
    volumes: deque[float]
    open_interest: deque[float]
    last_time_key: str = ""


@register("strategy", "LiquidationCascadeReversionStrategy", interface="event_driven")
class LiquidationCascadeReversionStrategy(Strategy):
    """Buy beaten-down alts into a BTC-led deleveraging cascade, BTC-beta-hedged.

    The sleeve is crisis-conditional and event-triggered: it only fires when BTC
    prints a sharp 1h-equivalent drop on a volume spike AND open interest
    contracts (a classic liquidation cascade).  On a trigger it goes long an
    equal-weight basket of the most beaten-down alts for a fixed ``hold_bars``
    window and shorts BTC sized by the basket's rolling beta to BTC so the book
    stays roughly market-neutral.  A no-re-layer lock prevents stacking entries
    during the hold window, keeping turnover intrinsically low (cascades are
    rare), well under the 2.5 hard cap.
    """

    decision_cadence_seconds = 3600
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False
    required_features = (
        "open_interest",
        "liquidation_long_notional",
        "liquidation_short_notional",
    )

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "btc_symbol": HyperParam.string("btc_symbol", default="", tunable=False),
            "btc_return_trigger": HyperParam.floating(
                "btc_return_trigger", default=-0.03, low=-0.50, high=0.0
            ),
            "volume_mult": HyperParam.floating("volume_mult", default=2.0, low=1.0, high=20.0),
            "oi_drop_trigger": HyperParam.floating(
                "oi_drop_trigger", default=0.05, low=0.0, high=0.90
            ),
            "hold_bars": HyperParam.integer("hold_bars", default=12, low=1, high=10_080),
            "basket_size": HyperParam.integer("basket_size", default=4, low=1, high=64),
            "beta_window": HyperParam.integer("beta_window", default=72, low=4, high=10_080),
            "return_lookback_bars": HyperParam.integer(
                "return_lookback_bars", default=1, low=1, high=2_880, tunable=False
            ),
            "alt_return_lookback_bars": HyperParam.integer(
                "alt_return_lookback_bars", default=4, low=1, high=2_880, tunable=False
            ),
            "volume_avg_bars": HyperParam.integer(
                "volume_avg_bars", default=24, low=2, high=10_080, tunable=False
            ),
            "oi_lookback_bars": HyperParam.integer(
                "oi_lookback_bars", default=4, low=1, high=2_880, tunable=False
            ),
            "min_basket": HyperParam.integer(
                "min_basket", default=2, low=1, high=64, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.0, low=0.0, high=1_000_000.0),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.020, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=400.0, low=0.0, high=1_000_000.0, tunable=False
            ),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.btc_symbol = self._resolve_btc_symbol(str(resolved["btc_symbol"]))
        self.alt_symbols = [symbol for symbol in self.symbol_list if symbol != self.btc_symbol]
        self.btc_return_trigger = min(0.0, float(resolved["btc_return_trigger"]))
        self.volume_mult = max(1.0, float(resolved["volume_mult"]))
        self.oi_drop_trigger = max(0.0, float(resolved["oi_drop_trigger"]))
        self.hold_bars = max(1, int(resolved["hold_bars"]))
        self.basket_size = max(1, int(resolved["basket_size"]))
        self.beta_window = max(2, int(resolved["beta_window"]))
        self.return_lookback_bars = max(1, int(resolved["return_lookback_bars"]))
        self.alt_return_lookback_bars = max(1, int(resolved["alt_return_lookback_bars"]))
        self.volume_avg_bars = max(2, int(resolved["volume_avg_bars"]))
        self.oi_lookback_bars = max(1, int(resolved["oi_lookback_bars"]))
        self.min_basket = max(1, int(resolved["min_basket"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        size = (
            max(
                self.beta_window,
                self.volume_avg_bars,
                self.oi_lookback_bars,
                self.alt_return_lookback_bars,
                self.return_lookback_bars,
            )
            + 8
        )
        self._state = {
            symbol: _CascadeSymbolState(
                closes=deque(maxlen=size),
                volumes=deque(maxlen=size),
                open_interest=deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }
        # Position bookkeeping for the no-re-layer lock + fixed-hold exit.
        self._in_position = False
        self._bars_held = 0
        self._active_symbols: list[str] = []
        self._last_eval_time_key = ""

    def _resolve_btc_symbol(self, explicit: str) -> str:
        if explicit and explicit in self.symbol_list:
            return explicit
        for symbol in self.symbol_list:
            upper = str(symbol).upper().replace("/", "").replace("-", "").replace("_", "")
            if any(hint in upper for hint in _BTC_HINTS):
                return symbol
        return ""

    def get_state(self) -> dict[str, Any]:
        return {
            "in_position": bool(self._in_position),
            "bars_held": int(self._bars_held),
            "active_symbols": list(self._active_symbols),
            "last_eval_time_key": self._last_eval_time_key,
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "volumes": list(item.volumes),
                    "open_interest": list(item.open_interest),
                    "last_time_key": item.last_time_key,
                }
                for symbol, item in self._state.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._in_position = bool(state.get("in_position", False))
        self._bars_held = self._safe_non_negative_int(state.get("bars_held"))
        self._active_symbols = [
            str(symbol)
            for symbol in list(state.get("active_symbols") or [])
            if str(symbol) in self._state
        ]
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        raw = state.get("symbol_state")
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            for attr in ("closes", "volumes", "open_interest"):
                target = getattr(item, attr)
                target.clear()
                for value in list(payload.get(attr) or [])[-int(target.maxlen or 0) :]:
                    parsed = safe_float(value)
                    if parsed is not None:
                        target.append(parsed)
            item.last_time_key = str(payload.get("last_time_key", ""))

    @staticmethod
    def _safe_non_negative_int(value: Any) -> int:
        try:
            return max(0, int(value))
        except Exception:
            return 0

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
        getter = getattr(self.bars, "get_latest_bar_value", None)
        if callable(getter):
            try:
                value = getter(symbol, field)
            except Exception:
                value = None
            return safe_float(value)
        return None

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        if str(getattr(event, "type", "")).upper() != "MARKET_WINDOW":
            self.calculate_signals(event)
            return
        event_key = time_key(getattr(event, "time", None))
        updated = False
        for symbol in self.symbol_list:
            if symbol not in self._state:
                continue
            snapshot = self._window_snapshot(event, symbol)
            if self._update(event, symbol, snapshot):
                updated = True
        if updated and event_key and event_key != self._last_eval_time_key:
            self._last_eval_time_key = event_key
            self._evaluate(getattr(event, "time", None))

    def calculate_signals(self, event: Any) -> None:
        event_type = str(getattr(event, "type", "")).upper()
        if event_type == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if event_type != "MARKET":
            return
        symbol = str(getattr(event, "symbol", ""))
        if symbol not in self._state:
            return
        snapshot = {
            "close": safe_float(getattr(event, "close", None)),
            "volume": safe_float(getattr(event, "volume", None)),
        }
        if self._update(event, symbol, snapshot):
            key = time_key(getattr(event, "time", None))
            if key and key != self._last_eval_time_key:
                self._last_eval_time_key = key
                self._evaluate(getattr(event, "time", None))

    def _window_snapshot(self, event: Any, symbol: str) -> dict[str, float | None]:
        bars_1s = dict(getattr(event, "bars_1s", {}) or {})
        rows = list(bars_1s.get(symbol) or [])
        if not rows:
            return {}
        closes: list[float] = []
        volumes: list[float] = []
        for row in rows:
            if isinstance(row, dict):
                close = safe_float(row.get("close"))
                volume = safe_float(row.get("volume"))
            elif isinstance(row, (tuple, list)):
                close = safe_float(row[4]) if len(row) > 4 else None
                volume = safe_float(row[5]) if len(row) > 5 else None
            else:
                close = None
                volume = None
            if close is not None:
                closes.append(close)
            if volume is not None:
                volumes.append(max(0.0, volume))
        return {
            "close": closes[-1] if closes else None,
            "volume": float(sum(volumes)) if volumes else None,
        }

    def _update(self, event: Any, symbol: str, snapshot: dict[str, float | None]) -> bool:
        item = self._state[symbol]
        key = time_key(getattr(event, "time", getattr(event, "datetime", None)))
        if key and key == item.last_time_key:
            return False
        close = snapshot.get("close")
        if close is None:
            close = self._extract_feature(event, symbol, "close")
        if close is None or close <= self.min_price or close <= 0.0:
            return False
        item.last_time_key = key
        volume = snapshot.get("volume")
        if volume is None:
            volume = self._extract_feature(event, symbol, "volume")
        open_interest = self._extract_feature(event, symbol, "open_interest")
        item.closes.append(float(close))
        item.volumes.append(max(0.0, float(volume)) if volume is not None else 0.0)
        item.open_interest.append(
            float(open_interest) if open_interest is not None and open_interest > 0.0 else 0.0
        )
        return True

    @staticmethod
    def _pct_change(values: deque[float], lookback: int) -> float | None:
        if len(values) <= lookback:
            return None
        latest = float(values[-1])
        base = float(values[-1 - lookback])
        if abs(base) <= 1e-12:
            return None
        return latest / base - 1.0

    def _volume_spike(self, item: _CascadeSymbolState) -> bool:
        if len(item.volumes) < self.volume_avg_bars + 1:
            return False
        latest = float(item.volumes[-1])
        history = list(item.volumes)[-(self.volume_avg_bars + 1) : -1]
        if not history:
            return False
        avg = sum(history) / float(len(history))
        if avg <= 1e-12:
            return False
        return latest >= self.volume_mult * avg

    def _btc_triggered(self) -> tuple[bool, dict[str, Any]]:
        item = self._state.get(self.btc_symbol)
        if item is None:
            return False, {}
        btc_return = self._pct_change(item.closes, self.return_lookback_bars)
        oi_change = self._pct_change(item.open_interest, self.oi_lookback_bars)
        if btc_return is None or oi_change is None:
            return False, {}
        return_ok = btc_return <= self.btc_return_trigger
        volume_ok = self._volume_spike(item)
        oi_ok = oi_change <= -self.oi_drop_trigger
        diagnostics = {
            "btc_return": float(btc_return),
            "btc_oi_change": float(oi_change),
            "btc_volume_spike": bool(volume_ok),
        }
        return bool(return_ok and volume_ok and oi_ok), diagnostics

    def _select_basket(self) -> list[tuple[str, float]]:
        rows: list[tuple[float, str]] = []
        for symbol in self.alt_symbols:
            item = self._state.get(symbol)
            if item is None:
                continue
            ret = simple_return(item.closes, lookback=self.alt_return_lookback_bars)
            if ret is None:
                continue
            rows.append((float(ret), symbol))
        # Most beaten-down (most negative return) alts lead the reversion basket.
        rows.sort(key=lambda entry: entry[0])
        return [(symbol, ret) for ret, symbol in rows[: self.basket_size]]

    def _basket_beta(self, symbols: list[str]) -> float | None:
        btc_item = self._state.get(self.btc_symbol)
        if btc_item is None or len(btc_item.closes) < 4:
            return None
        betas: list[float] = []
        for symbol in symbols:
            item = self._state.get(symbol)
            if item is None:
                continue
            beta = rolling_beta(list(item.closes), list(btc_item.closes))
            if beta is not None:
                betas.append(float(beta))
        if not betas:
            return None
        return float(sum(betas) / len(betas))

    def _evaluate(self, event_time: Any) -> None:
        # Manage an open position: fixed-hold exit + no-re-layer lock.
        if self._in_position:
            self._bars_held += 1
            if self._bars_held >= self.hold_bars:
                self._exit_all(event_time, "hold_window_elapsed")
            return
        if not self.btc_symbol or len(self.alt_symbols) < self.min_basket:
            return
        triggered, diagnostics = self._btc_triggered()
        if not triggered:
            return
        basket = self._select_basket()
        if len(basket) < self.min_basket:
            return
        basket_symbols = [symbol for symbol, _ in basket]
        beta = self._basket_beta(basket_symbols)
        self._enter_basket(event_time, basket, beta, diagnostics)

    def _enter_basket(
        self,
        event_time: Any,
        basket: list[tuple[str, float]],
        beta: float | None,
        diagnostics: dict[str, Any],
    ) -> None:
        if self.target_allocation <= 0.0 or self.max_order_value <= 0.0:
            return
        per_alt = self.target_allocation / float(len(basket))
        emitted: list[str] = []
        for symbol, ret in basket:
            item = self._state.get(symbol)
            price = float(item.closes[-1]) if item and item.closes else None
            metadata = _target_metadata(
                strategy="LiquidationCascadeReversionStrategy",
                target_allocation=per_alt,
                max_order_value=self.max_order_value,
                reason="liquidation_cascade_reversion_long",
                recent_return=float(ret),
                basket_beta=beta,
                **diagnostics,
            )
            _emit(
                self.events,
                strategy_id="liquidation_cascade_reversion",
                symbol=symbol,
                event_time=event_time,
                signal_type="LONG",
                price=price,
                metadata=metadata,
            )
            emitted.append(symbol)
        # BTC-beta hedge: short BTC sized by the basket's average beta.
        if self.btc_symbol and beta is not None and beta > 0.0:
            btc_item = self._state.get(self.btc_symbol)
            btc_price = float(btc_item.closes[-1]) if btc_item and btc_item.closes else None
            hedge_alloc = min(self.target_allocation, self.target_allocation * float(beta))
            if hedge_alloc > 0.0:
                metadata = _target_metadata(
                    strategy="LiquidationCascadeReversionStrategy",
                    target_allocation=hedge_alloc,
                    max_order_value=self.max_order_value,
                    reason="btc_beta_hedge",
                    basket_beta=beta,
                    **diagnostics,
                )
                _emit(
                    self.events,
                    strategy_id="liquidation_cascade_reversion",
                    symbol=self.btc_symbol,
                    event_time=event_time,
                    signal_type="SHORT",
                    price=btc_price,
                    metadata=metadata,
                )
                emitted.append(self.btc_symbol)
        self._in_position = True
        self._bars_held = 0
        self._active_symbols = emitted

    def _exit_all(self, event_time: Any, reason: str) -> None:
        for symbol in self._active_symbols:
            item = self._state.get(symbol)
            price = float(item.closes[-1]) if item and item.closes else None
            metadata = _target_metadata(
                strategy="LiquidationCascadeReversionStrategy",
                target_allocation=0.0,
                max_order_value=0.0,
                reason=reason,
            )
            _emit(
                self.events,
                strategy_id="liquidation_cascade_reversion",
                symbol=symbol,
                event_time=event_time,
                signal_type="EXIT",
                price=price,
                metadata=metadata,
            )
        self._in_position = False
        self._bars_held = 0
        self._active_symbols = []
