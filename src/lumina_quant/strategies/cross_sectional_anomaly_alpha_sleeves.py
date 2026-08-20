"""Cross-sectional anomaly alpha sleeves (decorrelated factor families).

This module adds four theory-grounded, cross-sectional anomaly sleeves that are
implementable directly from OHLCV plus a benchmark proxy.  Each class is a
multi-symbol basket rebalancer expressed entirely through the local event/bar
contract; they reuse the shared cross-sectional rebalancer, snapshot, and emit
primitives so they stay byte-for-byte consistent with the existing sleeves:

- ``IdiosyncraticVolatilityStrategy``: ranks symbols by the volatility of their
  benchmark-residual returns (``ret - beta * benchmark_ret``); long the low
  idiosyncratic-volatility quantile, short the high.  Captures the
  idiosyncratic-volatility anomaly (Ang-Hodrick-Xing-Zhang), which is distinct
  from the *systematic* beta of ``BettingAgainstBetaStrategy`` and from the
  *total* volatility plus momentum blend of ``LowVolatilityMomentumStrategy``.
- ``LotterySkewnessStrategy``: ranks symbols by a lottery score that blends
  trailing return skewness and the maximum single-bar return (Bali-Cakici-
  Whitelaw MAX); short the high-skew/lottery names, long the low-skew ones.
  Captures the lottery-preference premium, which no existing sleeve models.
- ``TrendEfficiencyMomentumStrategy``: scores each symbol by Kaufman efficiency
  ratio multiplied by the sign of its trailing trend; long clean high-efficiency
  uptrends, short low-efficiency/downtrends.  Captures trend *quality* (smooth
  vs choppy), distinct from raw time-series / cross-sectional momentum that
  ignore path efficiency.
- ``DispersionConditionedReversionStrategy``: only acts when cross-sectional
  return dispersion exceeds a regime threshold, then fades extreme movers toward
  the basket mean (long the bottom-return quantile, short the top).  A
  regime-gated cross-sectional reversion, distinct from the *unconditional*
  ``CrossSectionalShortTermReversalStrategy``.

All sleeves self-skip until at least ``min_symbols`` (default 4) names carry
sufficient history, guard against ``None`` / short history / divide-by-zero, use
bounded deques, and never raise from the hot path.  They auto-discover through
the plugin registry; validate scoring on the data-bearing research machine.
"""

from __future__ import annotations

import math
from collections import deque
from statistics import mean
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import (
    finite_floats,
    simple_return,
    trend_efficiency,
)
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.momentum import kaufman_efficiency_ratio
from lumina_quant.indicators.rolling_stats import rolling_beta, sample_std
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import (
    _age_cross_positions,
    _emit_rebalance_targets,
    _ranked_targets,
)
from lumina_quant.strategies.equity_xs_factor_alpha_sleeves import (
    _default_benchmark,
    _state_size,
)
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
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


# DELIBERATE private copy (NOT an alias of ``rolling_stats.rolling_skewness``):
# this module is a REGISTERED strategy module, so its default numerics must stay
# byte-identical.  The canonical indicator accumulates moments with ``math.fsum``
# (compensated summation) while this original recipe uses ``statistics.mean``
# plus plain ``sum`` moments -- a last-ULP drift on the LotterySkewness scoring
# path that can flip cross-sectional rank ties and therefore signals.  Keep the
# verbatim original here; ``skew_innovation`` consumers alias the canonical copy
# where the fsum parity is bit-exact.
def _skewness(values: list[float]) -> float | None:
    """Return the sample (Fisher-Pearson, ddof-free) skewness of ``values``.

    ``None`` is returned when there are fewer than three finite samples or the
    dispersion is degenerate (zero standard deviation).  Pure and ``None``-safe;
    never raises.
    """
    cleaned = finite_floats(values)
    count = len(cleaned)
    if count < 3:
        return None
    avg = mean(cleaned)
    variance = sum((value - avg) ** 2 for value in cleaned) / float(count)
    if variance <= _EPS:
        return None
    std = variance**0.5
    third = sum((value - avg) ** 3 for value in cleaned) / float(count)
    skew = third / (std**3)
    return float(skew) if math.isfinite(skew) else None


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


def _bar_simple_returns(closes: list[float]) -> list[float]:
    """Return the bar-to-bar simple returns of a price path (drops bad bars)."""
    out: list[float] = []
    prev: float | None = None
    for value in closes:
        if prev is not None and abs(prev) > _EPS and value == value:
            out.append(value / prev - 1.0)
        prev = value
    return out


class _CrossUpdateMixin:
    """Shared per-symbol close ingestion / state serialization for XS sleeves."""

    min_price: float
    symbol_list: list[str]
    _state: dict[str, _CrossSectionalState]
    _last_eval_time_key: str
    _tick: int

    def _update_symbol(self, symbol: str, snapshot: _Snapshot) -> bool:
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

    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
            "symbol_state": {symbol: _pack_cross(item) for symbol, item in self._state.items()},
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._tick = _safe_non_negative_int(state.get("tick"))
        raw = state.get("symbol_state")
        if isinstance(raw, dict):
            for symbol, payload in raw.items():
                if symbol in self._state and isinstance(payload, dict):
                    _restore_cross(self._state[symbol], payload)

    def _rebalance(self, event_time: Any) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        event_key = time_key(getattr(event, "time", None))
        updated = False
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None and self._update_symbol(symbol, snapshot):
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
            if snapshot is not None and self._update_symbol(str(symbol), snapshot):
                key = time_key(snapshot.time)
                if key and key != self._last_eval_time_key:
                    self._last_eval_time_key = key
                    self._tick += 1
                    self._rebalance(snapshot.time)


@register("strategy", "IdiosyncraticVolatilityStrategy", interface="event_driven")
class IdiosyncraticVolatilityStrategy(_CrossUpdateMixin, Strategy):
    """Long low / short high idiosyncratic (benchmark-residual) volatility."""

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "benchmark_symbol": HyperParam.string(
                "benchmark_symbol", default="BTC/USDT", tunable=False
            ),
            "beta_window": HyperParam.integer("beta_window", default=120, low=8, high=20000),
            "vol_window": HyperParam.integer("vol_window", default=60, low=4, high=4096),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=5, low=1, high=10080),
            "quantile_pct": HyperParam.floating("quantile_pct", default=0.25, low=0.05, high=0.50),
            "min_symbols": HyperParam.integer("min_symbols", default=4, low=2, high=512),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.08, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=120, low=1, high=200000),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=0.36, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=400.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.benchmark_symbol = _default_benchmark(
            self.symbol_list, str(resolved["benchmark_symbol"])
        )
        self.beta_window = max(4, int(resolved["beta_window"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.quantile_pct = max(0.01, min(0.50, float(resolved["quantile_pct"])))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.allow_short = bool(resolved["allow_short"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(self.beta_window, self.vol_window, self.max_hold_bars)
        self._state = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0

    def _rebalance(self, event_time: Any) -> None:
        if self._tick % self.rebalance_bars:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id="idiosyncratic_volatility",
                strategy_name="IdiosyncraticVolatilityStrategy",
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        benchmark = self._state.get(self.benchmark_symbol)
        need = max(self.beta_window, self.vol_window) + 1
        if benchmark is None or len(benchmark.closes) < need:
            return
        bench_returns = _bar_simple_returns(list(benchmark.closes))
        if len(bench_returns) < max(4, self.vol_window):
            return
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for symbol, item in self._state.items():
            if symbol == self.benchmark_symbol or len(item.closes) < need:
                continue
            sym_returns = _bar_simple_returns(list(item.closes))
            window = min(len(sym_returns), len(bench_returns), self.beta_window)
            if window < max(4, self.vol_window):
                continue
            x_tail = bench_returns[-window:]
            y_tail = sym_returns[-window:]
            # rolling_beta(a, b) = cov(a, b) / var(b): pass the SYMBOL series first
            # and the BENCHMARK second so beta is the asset-on-market loading
            # (cov(sym, bench) / var(bench)); the residual below then removes the
            # systematic component, leaving the idiosyncratic return.
            beta = rolling_beta(y_tail, x_tail)
            if beta is None:
                beta = 0.0
            residuals = [ry - beta * rx for ry, rx in zip(y_tail, x_tail, strict=False)]
            idio_vol = sample_std(residuals[-self.vol_window :])
            if idio_vol is None or idio_vol <= _EPS:
                continue
            # Low idio-vol -> high score (long); high idio-vol -> low score (short).
            score = -float(idio_vol)
            rows.append(
                (
                    score,
                    symbol,
                    {"idiosyncratic_vol": float(idio_vol), "beta": float(beta)},
                )
            )
        if len(rows) < self.min_symbols:
            return
        max_side = max(1, int(len(rows) * self.quantile_pct))
        targets = _ranked_targets(
            rows,
            threshold=-1.0e18,
            max_longs=max_side,
            max_shorts=max_side if self.allow_short else 0,
            allow_short=self.allow_short,
        )
        _emit_rebalance_targets(
            self.events,
            self._state,
            targets,
            event_time=event_time,
            strategy_id="idiosyncratic_volatility",
            strategy_name="IdiosyncraticVolatilityStrategy",
            target_gross_exposure=self.target_gross_exposure,
            max_order_value=self.max_order_value,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
            threshold=_EPS,
        )


@register("strategy", "LotterySkewnessStrategy", interface="event_driven")
class LotterySkewnessStrategy(_CrossUpdateMixin, Strategy):
    """Short high lottery (skew/MAX) names, long low-skew names."""

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "skew_window": HyperParam.integer("skew_window", default=60, low=4, high=4096),
            "max_window": HyperParam.integer("max_window", default=20, low=2, high=4096),
            "max_weight": HyperParam.floating("max_weight", default=0.50, low=0.0, high=1.0),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=5, low=1, high=10080),
            "quantile_pct": HyperParam.floating("quantile_pct", default=0.25, low=0.05, high=0.50),
            "min_symbols": HyperParam.integer("min_symbols", default=4, low=2, high=512),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.08, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=120, low=1, high=200000),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=0.36, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=400.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.skew_window = max(3, int(resolved["skew_window"]))
        self.max_window = max(2, int(resolved["max_window"]))
        self.max_weight = max(0.0, min(1.0, float(resolved["max_weight"])))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.quantile_pct = max(0.01, min(0.50, float(resolved["quantile_pct"])))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.allow_short = bool(resolved["allow_short"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(self.skew_window, self.max_window, self.max_hold_bars)
        self._state = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0

    def _lottery_score(self, closes: list[float]) -> tuple[float, dict[str, Any]] | None:
        returns = _bar_simple_returns(closes)
        if len(returns) < max(3, self.skew_window // 2):
            return None
        skew = _skewness(returns[-self.skew_window :])
        max_tail = returns[-self.max_window :]
        max_ret = max(max_tail) if max_tail else None
        if skew is None and max_ret is None:
            return None
        skew_component = float(skew) if skew is not None else 0.0
        max_component = float(max_ret) if max_ret is not None else 0.0
        # Higher lottery score -> more lottery-like (short candidate).
        lottery = (1.0 - self.max_weight) * skew_component + self.max_weight * max_component
        return float(lottery), {
            "skewness": skew_component,
            "max_return": max_component,
            "lottery_score": float(lottery),
        }

    def _rebalance(self, event_time: Any) -> None:
        if self._tick % self.rebalance_bars:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id="lottery_skewness",
                strategy_name="LotterySkewnessStrategy",
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        need = max(self.skew_window, self.max_window) + 1
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for symbol, item in self._state.items():
            if len(item.closes) < need:
                continue
            scored = self._lottery_score(list(item.closes))
            if scored is None:
                continue
            lottery, meta = scored
            # Long low-lottery (negate so low lottery -> high rank score).
            rows.append((-lottery, symbol, meta))
        if len(rows) < self.min_symbols:
            return
        max_side = max(1, int(len(rows) * self.quantile_pct))
        targets = _ranked_targets(
            rows,
            threshold=-1.0e18,
            max_longs=max_side,
            max_shorts=max_side if self.allow_short else 0,
            allow_short=self.allow_short,
        )
        _emit_rebalance_targets(
            self.events,
            self._state,
            targets,
            event_time=event_time,
            strategy_id="lottery_skewness",
            strategy_name="LotterySkewnessStrategy",
            target_gross_exposure=self.target_gross_exposure,
            max_order_value=self.max_order_value,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
            threshold=_EPS,
        )


@register("strategy", "TrendEfficiencyMomentumStrategy", interface="event_driven")
class TrendEfficiencyMomentumStrategy(_CrossUpdateMixin, Strategy):
    """Long clean high-efficiency uptrends, short low-efficiency/downtrends."""

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "efficiency_period": HyperParam.integer(
                "efficiency_period", default=20, low=2, high=4096
            ),
            "trend_lookback_bars": HyperParam.integer(
                "trend_lookback_bars", default=20, low=2, high=4096
            ),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=5, low=1, high=10080),
            "quantile_pct": HyperParam.floating("quantile_pct", default=0.25, low=0.05, high=0.50),
            "signal_threshold": HyperParam.floating(
                "signal_threshold", default=0.10, low=0.0, high=20.0
            ),
            "min_symbols": HyperParam.integer("min_symbols", default=4, low=2, high=512),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.08, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=120, low=1, high=200000),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=0.36, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=400.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.efficiency_period = max(2, int(resolved["efficiency_period"]))
        self.trend_lookback_bars = max(2, int(resolved["trend_lookback_bars"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.quantile_pct = max(0.01, min(0.50, float(resolved["quantile_pct"])))
        self.signal_threshold = max(0.0, float(resolved["signal_threshold"]))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.allow_short = bool(resolved["allow_short"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(self.efficiency_period, self.trend_lookback_bars, self.max_hold_bars)
        self._state = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0

    def _rebalance(self, event_time: Any) -> None:
        if self._tick % self.rebalance_bars:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id="trend_efficiency_momentum",
                strategy_name="TrendEfficiencyMomentumStrategy",
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        need = max(self.efficiency_period, self.trend_lookback_bars) + 1
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for symbol, item in self._state.items():
            if len(item.closes) < need:
                continue
            closes = list(item.closes)
            efficiency = kaufman_efficiency_ratio(closes, period=self.efficiency_period)
            if efficiency is None:
                efficiency = trend_efficiency(closes, window=self.efficiency_period)
            trend = simple_return(closes, lookback=self.trend_lookback_bars)
            if efficiency is None or trend is None:
                continue
            direction = 1.0 if trend > 0.0 else (-1.0 if trend < 0.0 else 0.0)
            # Trend QUALITY: clean (high efficiency) trends score larger in the
            # direction of the trailing move; choppy ones collapse toward zero.
            score = float(efficiency) * direction
            rows.append(
                (
                    score,
                    symbol,
                    {
                        "efficiency_ratio": float(efficiency),
                        "trailing_return": float(trend),
                        "trend_sign": direction,
                    },
                )
            )
        if len(rows) < self.min_symbols:
            return
        max_side = max(1, int(len(rows) * self.quantile_pct))
        targets = _ranked_targets(
            rows,
            threshold=self.signal_threshold,
            max_longs=max_side,
            max_shorts=max_side if self.allow_short else 0,
            allow_short=self.allow_short,
        )
        _emit_rebalance_targets(
            self.events,
            self._state,
            targets,
            event_time=event_time,
            strategy_id="trend_efficiency_momentum",
            strategy_name="TrendEfficiencyMomentumStrategy",
            target_gross_exposure=self.target_gross_exposure,
            max_order_value=self.max_order_value,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
            threshold=max(self.signal_threshold, _EPS),
        )


@register("strategy", "DispersionConditionedReversionStrategy", interface="event_driven")
class DispersionConditionedReversionStrategy(_CrossUpdateMixin, Strategy):
    """Regime-gated cross-sectional reversion: fade extremes only in high dispersion."""

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "reversion_lookback_bars": HyperParam.integer(
                "reversion_lookback_bars", default=5, low=1, high=4096
            ),
            "dispersion_threshold": HyperParam.floating(
                "dispersion_threshold", default=0.02, low=0.0, high=5.0
            ),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=3, low=1, high=10080),
            "quantile_pct": HyperParam.floating("quantile_pct", default=0.25, low=0.05, high=0.50),
            "min_symbols": HyperParam.integer("min_symbols", default=4, low=2, high=512),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.06, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=48, low=1, high=200000),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=0.36, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=400.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.reversion_lookback_bars = max(1, int(resolved["reversion_lookback_bars"]))
        self.dispersion_threshold = max(0.0, float(resolved["dispersion_threshold"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.quantile_pct = max(0.01, min(0.50, float(resolved["quantile_pct"])))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.allow_short = bool(resolved["allow_short"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(self.reversion_lookback_bars, self.max_hold_bars)
        self._state = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0

    def _rebalance(self, event_time: Any) -> None:
        if self._tick % self.rebalance_bars:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id="dispersion_conditioned_reversion",
                strategy_name="DispersionConditionedReversionStrategy",
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        need = self.reversion_lookback_bars + 1
        returns: list[tuple[float, str]] = []
        for symbol, item in self._state.items():
            if len(item.closes) < need:
                continue
            ret = simple_return(list(item.closes), lookback=self.reversion_lookback_bars)
            if ret is None:
                continue
            returns.append((float(ret), symbol))
        if len(returns) < self.min_symbols:
            return
        ret_values = [value for value, _symbol in returns]
        dispersion = sample_std(ret_values)
        if dispersion is None or dispersion < self.dispersion_threshold:
            # Regime gate closed: do not open new cross-sectional reversion bets.
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id="dispersion_conditioned_reversion",
                strategy_name="DispersionConditionedReversionStrategy",
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        basket_mean = mean(ret_values)
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for ret, symbol in returns:
            # Fade toward the basket mean: bottom-return names (ret << mean) score
            # high (long), top-return names (ret >> mean) score low (short).
            score = -(ret - basket_mean)
            rows.append(
                (
                    float(score),
                    symbol,
                    {
                        "reversion_return": float(ret),
                        "basket_mean_return": float(basket_mean),
                        "cross_sectional_dispersion": float(dispersion),
                    },
                )
            )
        max_side = max(1, int(len(rows) * self.quantile_pct))
        targets = _ranked_targets(
            rows,
            threshold=_EPS,
            max_longs=max_side,
            max_shorts=max_side if self.allow_short else 0,
            allow_short=self.allow_short,
        )
        _emit_rebalance_targets(
            self.events,
            self._state,
            targets,
            event_time=event_time,
            strategy_id="dispersion_conditioned_reversion",
            strategy_name="DispersionConditionedReversionStrategy",
            target_gross_exposure=self.target_gross_exposure,
            max_order_value=self.max_order_value,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
            threshold=_EPS,
        )


__all__ = [
    "DispersionConditionedReversionStrategy",
    "IdiosyncraticVolatilityStrategy",
    "LotterySkewnessStrategy",
    "TrendEfficiencyMomentumStrategy",
]
