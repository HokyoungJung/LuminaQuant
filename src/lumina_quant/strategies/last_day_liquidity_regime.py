"""Last-day return strategy conditioned on liquidity regime.

Liquid assets follow short-horizon continuation while illiquid assets can
mean-revert. The implementation stays lightweight by using only bounded
close/volume history and simple cross-sectional ranking at rebalance time.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.common import safe_float, safe_int, time_key
from lumina_quant.indicators.momentum import lagged_momentum_return, rolling_mean_dollar_volume
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _SymbolState:
    closes: deque[float]
    volumes: deque[float]
    mode: str = "OUT"
    last_time_key: str = ""


class LastDayLiquidityRegimeStrategy(Strategy):
    """Trade 24h continuation for liquid assets and reversal for illiquid ones."""

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "momentum_lookback_bars": HyperParam.integer(
                "momentum_lookback_bars",
                default=24,
                low=4,
                high=2048,
                tunable=False,
            ),
            "signal_skip_bars": HyperParam.integer(
                "signal_skip_bars",
                default=1,
                low=0,
                high=128,
                tunable=False,
            ),
            "liquidity_window": HyperParam.integer(
                "liquidity_window",
                default=24,
                low=4,
                high=2048,
                tunable=False,
            ),
            "volatility_window": HyperParam.integer(
                "volatility_window",
                default=24,
                low=4,
                high=2048,
                tunable=False,
            ),
            "rebalance_bars": HyperParam.integer(
                "rebalance_bars",
                default=6,
                low=1,
                high=2048,
                tunable=False,
            ),
            "signal_threshold": HyperParam.floating(
                "signal_threshold",
                default=0.012,
                low=0.0,
                high=1.0,
                tunable=False,
            ),
            "liquidity_quantile": HyperParam.floating(
                "liquidity_quantile",
                default=0.60,
                low=0.0,
                high=1.0,
                tunable=False,
            ),
            "max_longs": HyperParam.integer(
                "max_longs",
                default=2,
                low=0,
                high=64,
                tunable=False,
            ),
            "max_shorts": HyperParam.integer(
                "max_shorts",
                default=1,
                low=0,
                high=64,
                tunable=False,
            ),
            "min_price": HyperParam.floating(
                "min_price",
                default=0.10,
                low=0.0,
                high=1_000_000.0,
                tunable=False,
            ),
            "max_realized_vol": HyperParam.floating(
                "max_realized_vol",
                default=0.09,
                low=0.0,
                high=5.0,
                tunable=False,
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct",
                default=0.05,
                low=0.001,
                high=0.5,
                tunable=False,
            ),
            "allow_short": HyperParam.boolean("allow_short", default=True, tunable=False),
            "illiquid_reversal": HyperParam.boolean(
                "illiquid_reversal",
                default=True,
                tunable=False,
            ),
        }

    def __init__(
        self,
        bars,
        events,
        momentum_lookback_bars: int = 24,
        signal_skip_bars: int = 1,
        liquidity_window: int = 24,
        volatility_window: int = 24,
        rebalance_bars: int = 6,
        signal_threshold: float = 0.012,
        liquidity_quantile: float = 0.60,
        max_longs: int = 2,
        max_shorts: int = 1,
        min_price: float = 0.10,
        max_realized_vol: float = 0.09,
        stop_loss_pct: float = 0.05,
        allow_short: bool = True,
        illiquid_reversal: bool = True,
    ):
        self.bars = bars
        self.events = events
        self.symbol_list = list(self.bars.symbol_list)
        if not self.symbol_list:
            raise ValueError("LastDayLiquidityRegimeStrategy requires at least one symbol.")

        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "momentum_lookback_bars": momentum_lookback_bars,
                "signal_skip_bars": signal_skip_bars,
                "liquidity_window": liquidity_window,
                "volatility_window": volatility_window,
                "rebalance_bars": rebalance_bars,
                "signal_threshold": signal_threshold,
                "liquidity_quantile": liquidity_quantile,
                "max_longs": max_longs,
                "max_shorts": max_shorts,
                "min_price": min_price,
                "max_realized_vol": max_realized_vol,
                "stop_loss_pct": stop_loss_pct,
                "allow_short": allow_short,
                "illiquid_reversal": illiquid_reversal,
            },
            keep_unknown=False,
        )

        self.momentum_lookback_bars = int(resolved["momentum_lookback_bars"])
        self.signal_skip_bars = int(resolved["signal_skip_bars"])
        self.liquidity_window = int(resolved["liquidity_window"])
        self.volatility_window = int(resolved["volatility_window"])
        self.rebalance_bars = int(resolved["rebalance_bars"])
        self.signal_threshold = float(resolved["signal_threshold"])
        self.liquidity_quantile = min(1.0, max(0.0, float(resolved["liquidity_quantile"])))
        self.max_longs = int(resolved["max_longs"])
        self.max_shorts = int(resolved["max_shorts"])
        self.min_price = float(resolved["min_price"])
        self.max_realized_vol = float(resolved["max_realized_vol"])
        self.stop_loss_pct = float(resolved["stop_loss_pct"])
        self.allow_short = bool(resolved["allow_short"])
        self.illiquid_reversal = bool(resolved["illiquid_reversal"])

        history_len = max(
            self.momentum_lookback_bars + self.signal_skip_bars + 2,
            self.liquidity_window + 2,
            self.volatility_window + 2,
        )
        self._state = {
            symbol: _SymbolState(
                closes=deque(maxlen=history_len),
                volumes=deque(maxlen=history_len),
            )
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0

    def get_state(self) -> dict[str, Any]:
        return {
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "volumes": list(item.volumes),
                    "mode": item.mode,
                    "last_time_key": item.last_time_key,
                }
                for symbol, item in self._state.items()
            },
            "last_eval_time_key": str(self._last_eval_time_key),
            "tick": int(self._tick),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        raw_symbol_state = state.get("symbol_state")
        if isinstance(raw_symbol_state, dict):
            for symbol, payload in raw_symbol_state.items():
                if symbol not in self._state or not isinstance(payload, dict):
                    continue
                item = self._state[symbol]
                item.closes.clear()
                item.volumes.clear()
                keep = item.closes.maxlen if item.closes.maxlen is not None else 0
                for value in list(payload.get("closes") or [])[-keep:]:
                    parsed = safe_float(value)
                    if parsed is not None and parsed > 0.0:
                        item.closes.append(parsed)
                for value in list(payload.get("volumes") or [])[-keep:]:
                    parsed = safe_float(value)
                    if parsed is not None and parsed >= 0.0:
                        item.volumes.append(parsed)
                mode = str(payload.get("mode", "OUT")).upper()
                item.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
                item.last_time_key = str(payload.get("last_time_key", ""))
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._tick = max(0, safe_int(state.get("tick", 0), 0))

    def _resolve_close(self, symbol: str, event: Any) -> float | None:
        if getattr(event, "symbol", None) == symbol:
            close = safe_float(getattr(event, "close", None))
            if close is not None and close > 0.0:
                return close
        close = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
        if close is None or close <= 0.0:
            return None
        return close

    def _resolve_volume(self, symbol: str, event: Any) -> float | None:
        if getattr(event, "symbol", None) == symbol:
            volume = safe_float(getattr(event, "volume", None))
            if volume is not None and volume >= 0.0:
                return volume
        volume = safe_float(self.bars.get_latest_bar_value(symbol, "volume"))
        if volume is None or volume < 0.0:
            return None
        return volume

    @staticmethod
    def _realized_vol(closes: deque[float], *, window: int) -> float | None:
        values = list(closes)
        win = max(4, int(window))
        if len(values) <= win:
            return None
        returns: list[float] = []
        for prev, current in zip(values[-(win + 1) : -1], values[-win:], strict=True):
            if prev <= 0.0 or current <= 0.0:
                continue
            returns.append(math.log(current / prev))
        if len(returns) < win:
            return None
        mean_ret = sum(returns) / float(len(returns))
        variance = sum((ret - mean_ret) ** 2 for ret in returns) / float(max(1, len(returns) - 1))
        if variance < 0.0:
            return None
        value = math.sqrt(variance)
        return value if math.isfinite(value) else None

    def _liquidity_threshold(self, rows: list[dict[str, Any]]) -> float | None:
        liquidities = sorted(
            float(row["liquidity"]) for row in rows if float(row["liquidity"]) > 0.0
        )
        if not liquidities:
            return None
        idx = min(
            len(liquidities) - 1,
            max(0, math.floor((len(liquidities) - 1) * self.liquidity_quantile)),
        )
        return float(liquidities[idx])

    def _build_targets(
        self, aligned_symbols: list[str]
    ) -> tuple[dict[str, str], dict[str, float], dict[str, str], float | None]:
        signal_rows: list[dict[str, Any]] = []
        for symbol in aligned_symbols:
            item = self._state[symbol]
            latest = item.closes[-1]
            if latest < self.min_price:
                continue
            momentum = lagged_momentum_return(
                item.closes,
                lookback=self.momentum_lookback_bars,
                skip_bars=self.signal_skip_bars,
            )
            if momentum is None:
                continue
            liquidity = rolling_mean_dollar_volume(
                item.closes,
                item.volumes,
                window=self.liquidity_window,
            )
            if liquidity is None:
                continue
            realized_vol = self._realized_vol(item.closes, window=self.volatility_window)
            if realized_vol is None or not math.isfinite(realized_vol):
                realized_vol = 0.0
            if realized_vol > self.max_realized_vol:
                continue
            signal_rows.append(
                {
                    "symbol": symbol,
                    "momentum": float(momentum),
                    "liquidity": float(liquidity),
                    "realized_vol": float(realized_vol),
                }
            )

        if not signal_rows:
            return {}, {}, {}, None

        liquidity_threshold = self._liquidity_threshold(signal_rows)
        adjusted_scores: dict[str, float] = {}
        liquidity_regimes: dict[str, str] = {}
        for row in signal_rows:
            symbol = str(row["symbol"])
            momentum = float(row["momentum"])
            is_liquid = (
                liquidity_threshold is None or float(row["liquidity"]) >= liquidity_threshold
            )
            if is_liquid:
                adjusted_scores[symbol] = momentum
                liquidity_regimes[symbol] = "liquid_momentum"
            elif self.illiquid_reversal:
                adjusted_scores[symbol] = -momentum
                liquidity_regimes[symbol] = "illiquid_reversal"
            else:
                adjusted_scores[symbol] = 0.0
                liquidity_regimes[symbol] = "filtered_illiquid"

        ordered = sorted(adjusted_scores.items(), key=lambda item: item[1])
        longs = [symbol for symbol, score in reversed(ordered) if score >= self.signal_threshold][
            : self.max_longs
        ]
        shorts = []
        if self.allow_short and self.max_shorts > 0:
            shorts = [symbol for symbol, score in ordered if score <= -self.signal_threshold][
                : self.max_shorts
            ]
        long_set = set(longs)
        shorts = [symbol for symbol in shorts if symbol not in long_set]

        targets = dict.fromkeys(longs, "LONG")
        targets.update(dict.fromkeys(shorts, "SHORT"))
        return targets, adjusted_scores, liquidity_regimes, liquidity_threshold

    def _emit_signal(
        self,
        symbol: str,
        event_time: Any,
        signal_type: str,
        *,
        adjusted_score: float | None,
        liquidity_regime: str,
        liquidity_threshold: float | None,
    ) -> None:
        close_price = safe_float(self.bars.get_latest_bar_value(symbol, "close"))
        stop_loss = None
        if close_price is not None and close_price > 0.0:
            if signal_type == "LONG":
                stop_loss = close_price * (1.0 - self.stop_loss_pct)
            elif signal_type == "SHORT":
                stop_loss = close_price * (1.0 + self.stop_loss_pct)
        self.events.put(
            SignalEvent(
                strategy_id="last_day_liquidity_regime",
                symbol=symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=1.0,
                stop_loss=stop_loss,
                metadata={
                    "strategy": "LastDayLiquidityRegimeStrategy",
                    "adjusted_score": adjusted_score,
                    "liquidity_regime": liquidity_regime,
                    "liquidity_threshold": liquidity_threshold,
                    "momentum_lookback_bars": int(self.momentum_lookback_bars),
                    "signal_skip_bars": int(self.signal_skip_bars),
                    "liquidity_window": int(self.liquidity_window),
                    "rebalance_bars": int(self.rebalance_bars),
                    "signal_threshold": float(self.signal_threshold),
                    "illiquid_reversal": bool(self.illiquid_reversal),
                },
            )
        )

    def calculate_signals(self, event: Any) -> None:
        if getattr(event, "type", None) != "MARKET":
            return

        event_symbol = getattr(event, "symbol", None)
        if event_symbol not in self._state:
            return

        event_time = self.bars.get_latest_bar_datetime(event_symbol)
        event_time_key = time_key(event_time)
        if not event_time_key:
            return
        item = self._state[event_symbol]
        if item.last_time_key == event_time_key:
            return
        item.last_time_key = event_time_key

        close_price = self._resolve_close(event_symbol, event)
        volume = self._resolve_volume(event_symbol, event)
        if close_price is None or volume is None:
            return
        item.closes.append(close_price)
        item.volumes.append(volume)

        if event_time_key == self._last_eval_time_key:
            return

        aligned_symbols: list[str] = []
        minimum_history = max(
            self.momentum_lookback_bars + self.signal_skip_bars + 1,
            self.liquidity_window,
            self.volatility_window + 1,
        )
        for symbol in self.symbol_list:
            state = self._state[symbol]
            if time_key(self.bars.get_latest_bar_datetime(symbol)) != event_time_key:
                continue
            if len(state.closes) < minimum_history or len(state.volumes) < self.liquidity_window:
                continue
            aligned_symbols.append(symbol)

        minimum_symbols = max(2, min(len(self.symbol_list), self.max_longs + self.max_shorts + 1))
        if len(aligned_symbols) < minimum_symbols:
            return

        self._last_eval_time_key = event_time_key
        self._tick += 1
        if self._tick % self.rebalance_bars != 0:
            return

        targets, adjusted_scores, liquidity_regimes, liquidity_threshold = self._build_targets(
            aligned_symbols
        )

        for symbol, state in [(key, value.mode) for key, value in self._state.items()]:
            target_state = targets.get(symbol, "OUT")
            adjusted_score = adjusted_scores.get(symbol)
            liquidity_regime = liquidity_regimes.get(symbol, "inactive")
            if state == target_state:
                continue
            if state != "OUT" and target_state == "OUT":
                self._emit_signal(
                    symbol,
                    event_time,
                    "EXIT",
                    adjusted_score=adjusted_score,
                    liquidity_regime=liquidity_regime,
                    liquidity_threshold=liquidity_threshold,
                )
                self._state[symbol].mode = "OUT"
            elif state == "LONG" and target_state == "SHORT":
                self._emit_signal(
                    symbol,
                    event_time,
                    "EXIT",
                    adjusted_score=adjusted_score,
                    liquidity_regime=liquidity_regime,
                    liquidity_threshold=liquidity_threshold,
                )
                self._emit_signal(
                    symbol,
                    event_time,
                    "SHORT",
                    adjusted_score=adjusted_score,
                    liquidity_regime=liquidity_regime,
                    liquidity_threshold=liquidity_threshold,
                )
                self._state[symbol].mode = "SHORT"
            elif state == "SHORT" and target_state == "LONG":
                self._emit_signal(
                    symbol,
                    event_time,
                    "EXIT",
                    adjusted_score=adjusted_score,
                    liquidity_regime=liquidity_regime,
                    liquidity_threshold=liquidity_threshold,
                )
                self._emit_signal(
                    symbol,
                    event_time,
                    "LONG",
                    adjusted_score=adjusted_score,
                    liquidity_regime=liquidity_regime,
                    liquidity_threshold=liquidity_threshold,
                )
                self._state[symbol].mode = "LONG"

        for symbol, target_state in targets.items():
            if self._state[symbol].mode != "OUT":
                continue
            self._emit_signal(
                symbol,
                event_time,
                "LONG" if target_state == "LONG" else "SHORT",
                adjusted_score=adjusted_scores.get(symbol),
                liquidity_regime=liquidity_regimes.get(symbol, "inactive"),
                liquidity_threshold=liquidity_threshold,
            )
            self._state[symbol].mode = target_state

        LOGGER.info(
            "LastDayLiquidityRegime rebalance @%s longs=%d shorts=%d",
            event_time_key,
            sum(1 for state in self._state.values() if state.mode == "LONG"),
            sum(1 for state in self._state.values() if state.mode == "SHORT"),
        )
