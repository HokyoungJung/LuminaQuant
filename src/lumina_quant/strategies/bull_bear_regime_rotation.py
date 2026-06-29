"""Bull/bear breadth-regime rotation strategy.

A crypto-basket directional router designed for the repo's current performance
weakness: the incumbent leaf/top-cap stack can produce large upside in a few
months but has low monthly hit rate and weak explicit downside capture.  This
strategy keeps the implementation OHLCV-only and no-lookahead while adding a
single top-down regime decision:

- BULL: broad positive momentum + benchmark confirmation -> long the strongest
  names.
- BEAR: broad negative momentum + benchmark confirmation -> short the weakest
  names.
- NEUTRAL: flatten, rather than forcing churn in chop.

It is intentionally a research sleeve, not a live default.  Promotion still
requires cost-realistic walk-forward/shadow validation on the data-bearing
machine.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import simple_return
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.moving_average import simple_moving_average
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import (
    _age_cross_positions,
    _emit_rebalance_targets,
    _state_size,
)
from lumina_quant.strategies.external_alpha_sleeves import (
    _Snapshot,
    _event_symbols,
    _market_snapshot,
    _safe_non_negative_int,
    _window_snapshot,
)
from lumina_quant.strategies.robust_alpha_sleeves import (
    _CrossSectionalState,
    _mode,
    _restore_deque,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


def _pack_cross(item: _CrossSectionalState) -> dict[str, Any]:
    return {
        "closes": list(item.closes),
        "volumes": list(item.volumes),
        "mode": item.mode,
        "entry_price": item.entry_price,
        "bars_held": int(item.bars_held),
        "last_time_key": item.last_time_key,
    }


def _restore_cross(item: _CrossSectionalState, payload: dict[str, Any]) -> None:
    _restore_deque(item.closes, payload.get("closes"))
    _restore_deque(item.volumes, payload.get("volumes"))
    item.mode = _mode(payload.get("mode"))
    item.entry_price = safe_float(payload.get("entry_price"))
    item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
    item.last_time_key = str(payload.get("last_time_key", ""))


@register("strategy", "BullBearRegimeRotationStrategy", interface="event_driven")
class BullBearRegimeRotationStrategy(Strategy):
    """Rotate a crypto basket long in broad uptrends and short in broad downtrends.

    The strategy computes each symbol's simple return over ``momentum_lookback``
    and its position relative to ``trend_ma_window``.  Cross-sectional breadth is
    the fraction of eligible names with positive/negative trend-confirmed
    momentum.  A benchmark (BTC by default, falling back to the first symbol) must
    confirm the same direction before exposure is opened.

    - BULL when up-breadth and benchmark return clear their gates: long strongest
      ``max_longs`` symbols.
    - BEAR when down-breadth and benchmark return clear their gates: short weakest
      ``max_shorts`` symbols, if ``allow_short`` is true.
    - NEUTRAL otherwise: flatten existing positions to avoid chop leakage.

    This differs from ``BreadthRegimeTrendTimerStrategy`` by explicitly trading
    both sides of the regime (long in bull, short in bear) and from TopCap TSMOM
    by gating total exposure on market-wide breadth and benchmark confirmation.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False
    strategy_name = "BullBearRegimeRotationStrategy"
    strategy_id = "bull_bear_regime_rotation"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "momentum_lookback": HyperParam.integer(
                "momentum_lookback", default=48, low=3, high=20000
            ),
            "trend_ma_window": HyperParam.integer(
                "trend_ma_window", default=48, low=3, high=20000
            ),
            "signal_threshold": HyperParam.floating(
                "signal_threshold", default=0.015, low=0.0, high=1.0
            ),
            "bull_breadth": HyperParam.floating("bull_breadth", default=0.58, low=0.0, high=1.0),
            "bear_breadth": HyperParam.floating("bear_breadth", default=0.55, low=0.0, high=1.0),
            "exit_breadth": HyperParam.floating("exit_breadth", default=0.42, low=0.0, high=1.0),
            "benchmark_symbol": HyperParam.string(
                "benchmark_symbol", default="BTC/USDT", tunable=False
            ),
            "benchmark_lookback": HyperParam.integer(
                "benchmark_lookback", default=48, low=3, high=20000
            ),
            "benchmark_bull_threshold": HyperParam.floating(
                "benchmark_bull_threshold", default=0.005, low=0.0, high=1.0
            ),
            "benchmark_bear_threshold": HyperParam.floating(
                "benchmark_bear_threshold", default=0.005, low=0.0, high=1.0
            ),
            "max_longs": HyperParam.integer("max_longs", default=8, low=0, high=256),
            "max_shorts": HyperParam.integer("max_shorts", default=6, low=0, high=256),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "max_gross": HyperParam.floating("max_gross", default=1.00, low=0.0, high=5.0),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=3, low=1, high=10080),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.10, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=180, low=1, high=200000),
            "min_symbols": HyperParam.integer("min_symbols", default=5, low=2, high=512),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.90, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=750.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.momentum_lookback = max(1, int(resolved["momentum_lookback"]))
        self.trend_ma_window = max(2, int(resolved["trend_ma_window"]))
        self.signal_threshold = max(0.0, float(resolved["signal_threshold"]))
        self.bull_breadth = max(0.0, min(1.0, float(resolved["bull_breadth"])))
        self.bear_breadth = max(0.0, min(1.0, float(resolved["bear_breadth"])))
        self.exit_breadth = max(0.0, min(1.0, float(resolved["exit_breadth"])))
        self.benchmark_symbol = self._resolve_benchmark(str(resolved["benchmark_symbol"]))
        self.benchmark_lookback = max(1, int(resolved["benchmark_lookback"]))
        self.benchmark_bull_threshold = max(0.0, float(resolved["benchmark_bull_threshold"]))
        self.benchmark_bear_threshold = max(0.0, float(resolved["benchmark_bear_threshold"]))
        self.max_longs = max(0, int(resolved["max_longs"]))
        self.max_shorts = max(0, int(resolved["max_shorts"]))
        self.allow_short = bool(resolved["allow_short"])
        self.max_gross = max(0.0, float(resolved["max_gross"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(
            self.momentum_lookback,
            self.trend_ma_window,
            self.benchmark_lookback,
            self.max_hold_bars,
        )
        self._state = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0
        self._regime = "NEUTRAL"
        self._last_up_breadth = 0.0
        self._last_down_breadth = 0.0
        self._last_benchmark_return: float | None = None

    def _resolve_benchmark(self, preferred: str) -> str:
        if preferred in self.symbol_list:
            return preferred
        for candidate in ("BTC/USDT", "BTCUSDT", "ETH/USDT", "ETHUSDT"):
            if candidate in self.symbol_list:
                return candidate
        return self.symbol_list[0] if self.symbol_list else preferred

    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
            "regime": self._regime,
            "last_up_breadth": float(self._last_up_breadth),
            "last_down_breadth": float(self._last_down_breadth),
            "last_benchmark_return": self._last_benchmark_return,
            "symbol_state": {symbol: _pack_cross(item) for symbol, item in self._state.items()},
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._tick = _safe_non_negative_int(state.get("tick"))
        raw_regime = str(state.get("regime", "NEUTRAL")).upper()
        self._regime = raw_regime if raw_regime in {"BULL", "BEAR", "NEUTRAL"} else "NEUTRAL"
        up = safe_float(state.get("last_up_breadth"))
        down = safe_float(state.get("last_down_breadth"))
        bench = safe_float(state.get("last_benchmark_return"))
        if up is not None:
            self._last_up_breadth = max(0.0, min(1.0, float(up)))
        if down is not None:
            self._last_down_breadth = max(0.0, min(1.0, float(down)))
        self._last_benchmark_return = bench
        raw = state.get("symbol_state")
        if isinstance(raw, dict):
            for symbol, payload in raw.items():
                if symbol in self._state and isinstance(payload, dict):
                    _restore_cross(self._state[symbol], payload)

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        event_key = time_key(getattr(event, "time", None))
        updated = False
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None and self._update(symbol, snapshot):
                updated = True
        if updated and event_key and event_key != self._last_eval_time_key:
            self._last_eval_time_key = event_key
            self._tick += 1
            self._rebalance(getattr(event, "time", None))

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = getattr(event, "symbol", None)
        if symbol in self._state:
            snapshot = _market_snapshot(event)
            if snapshot is not None and self._update(str(symbol), snapshot):
                key = time_key(snapshot.time)
                if key and key != self._last_eval_time_key:
                    self._last_eval_time_key = key
                    self._tick += 1
                    self._rebalance(snapshot.time)

    def _update(self, symbol: str, snapshot: _Snapshot) -> bool:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return False
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return False
        item.last_time_key = key
        item.closes.append(close)
        item.volumes.append(max(0.0, float(snapshot.volume or 0.0)))
        return True

    def _symbol_score(self, symbol: str, item: _CrossSectionalState) -> tuple[float | None, dict[str, Any]]:
        closes = list(item.closes)
        ret = simple_return(closes, lookback=self.momentum_lookback)
        ma = simple_moving_average(closes, self.trend_ma_window)
        if ret is None or ma is None or not closes:
            return None, {}
        close = closes[-1]
        if close <= self.min_price:
            return None, {}
        ma_gap = (close / ma) - 1.0 if ma > 0.0 else 0.0
        # Momentum is the primary ranking variable; MA agreement is a gate and a
        # small score stabilizer so pure one-bar spikes do not dominate ranking.
        score = float(ret + 0.25 * ma_gap)
        return score, {
            "raw_momentum_return": float(ret),
            "ma_gap": float(ma_gap),
            "symbol_scope": symbol,
        }

    def _benchmark_return(self) -> float | None:
        item = self._state.get(self.benchmark_symbol)
        if item is None:
            return None
        return simple_return(list(item.closes), lookback=self.benchmark_lookback)

    def _breadth_rows(
        self,
    ) -> tuple[float, float, list[tuple[float, str, dict[str, Any]]], list[tuple[float, str, dict[str, Any]]]]:
        eligible = 0
        up_count = 0
        down_count = 0
        up_rows: list[tuple[float, str, dict[str, Any]]] = []
        down_rows: list[tuple[float, str, dict[str, Any]]] = []
        for symbol, item in self._state.items():
            score, meta = self._symbol_score(symbol, item)
            if score is None:
                continue
            eligible += 1
            if score >= self.signal_threshold:
                up_count += 1
                up_rows.append((score, symbol, meta))
            elif score <= -self.signal_threshold:
                down_count += 1
                down_rows.append((score, symbol, meta))
        up_breadth = (up_count / eligible) if eligible else 0.0
        down_breadth = (down_count / eligible) if eligible else 0.0
        up_rows.sort(key=lambda row: row[0], reverse=True)
        down_rows.sort(key=lambda row: row[0])
        return float(up_breadth), float(down_breadth), up_rows, down_rows

    def _classify_regime(self, up_breadth: float, down_breadth: float, benchmark_ret: float | None) -> str:
        bench = 0.0 if benchmark_ret is None else float(benchmark_ret)
        if up_breadth >= self.bull_breadth and bench >= self.benchmark_bull_threshold:
            return "BULL"
        if (
            self.allow_short
            and down_breadth >= self.bear_breadth
            and bench <= -self.benchmark_bear_threshold
        ):
            return "BEAR"
        # Hysteresis: once in a directional regime, do not flip to neutral until
        # that side's breadth materially decays or the benchmark disagrees.
        if self._regime == "BULL" and up_breadth > self.exit_breadth and bench >= 0.0:
            return "BULL"
        if self._regime == "BEAR" and down_breadth > self.exit_breadth and bench <= 0.0:
            return "BEAR"
        return "NEUTRAL"

    def _flatten(self, event_time: Any, reason: str) -> None:
        _emit_rebalance_targets(
            self.events,
            self._state,
            {},
            event_time=event_time,
            strategy_id=self.strategy_id,
            strategy_name=self.strategy_name,
            target_gross_exposure=0.0,
            max_order_value=self.max_order_value,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
            threshold=max(self.signal_threshold, 1e-12),
        )
        # _emit_rebalance_targets emits reason=rebalance_removed.  Keep the
        # explicit method argument in state for observability through regime.
        _ = reason

    def _rebalance(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        up_breadth, down_breadth, up_rows, down_rows = self._breadth_rows()
        benchmark_ret = self._benchmark_return()
        regime = self._classify_regime(up_breadth, down_breadth, benchmark_ret)
        self._last_up_breadth = up_breadth
        self._last_down_breadth = down_breadth
        self._last_benchmark_return = benchmark_ret
        self._regime = regime

        if regime == "NEUTRAL":
            self._flatten(event_time, "neutral_regime")
            return
        if self._tick % self.rebalance_bars:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id=self.strategy_id,
                strategy_name=self.strategy_name,
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return

        selected_rows = up_rows if regime == "BULL" else down_rows
        if not selected_rows:
            self._flatten(event_time, "no_regime_targets")
            return

        if regime == "BULL":
            targets = {
                symbol: (
                    "LONG",
                    score,
                    {
                        **meta,
                        "regime": regime,
                        "up_breadth": up_breadth,
                        "down_breadth": down_breadth,
                        "benchmark_return": benchmark_ret,
                    },
                )
                for score, symbol, meta in selected_rows[: self.max_longs]
                if score >= self.signal_threshold
            }
            scale = min(1.0, max(0.0, up_breadth))
        else:
            targets = {
                symbol: (
                    "SHORT",
                    score,
                    {
                        **meta,
                        "regime": regime,
                        "up_breadth": up_breadth,
                        "down_breadth": down_breadth,
                        "benchmark_return": benchmark_ret,
                    },
                )
                for score, symbol, meta in selected_rows[: self.max_shorts]
                if score <= -self.signal_threshold
            }
            scale = min(1.0, max(0.0, down_breadth))
        if not targets:
            self._flatten(event_time, "empty_target_set")
            return
        gross = self.target_allocation * min(self.max_gross, 1.0) * max(self.exit_breadth, scale)
        _emit_rebalance_targets(
            self.events,
            self._state,
            targets,
            event_time=event_time,
            strategy_id=self.strategy_id,
            strategy_name=self.strategy_name,
            target_gross_exposure=gross,
            max_order_value=self.max_order_value,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
            threshold=max(self.signal_threshold, 1e-12),
        )


__all__ = ["BullBearRegimeRotationStrategy"]
