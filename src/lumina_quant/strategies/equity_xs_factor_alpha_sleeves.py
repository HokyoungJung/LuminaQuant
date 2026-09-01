"""Dormant equity cross-sectional factor alpha sleeves (S1, S2, S3, S8).

These classes implement classic US-equity / tradfi factor families on the
equity-perp universe, expressed entirely through the local event/bar contract:

- ``CrossSectionalEquityMomentumStrategy`` (S1): vol-scaled 12-1 momentum
  (lookback 252, skip 21), long top-quintile / short bottom-quintile, with a
  short suppression filter when the basket median sits below its 200-bar SMA.
- ``ResidualEquityMomentumStrategy`` (S2, IMOM): rolling 252-bar beta + OLS
  regression-slope residual cumulative return (t-252 -> t-21), cross-sectionally
  ranked, inverse-residual-vol sized.
- ``BettingAgainstBetaStrategy`` (S3): rolling 252-bar beta vs the benchmark,
  long low-beta quintile / short high-beta quintile, beta-neutralized.
- ``SemisLeadLagRotationStrategy`` (S8): the semis ETF leader (SOXL) leads the
  single-name semis; on a >1 sigma leader move, tilt toward the under-reacted
  laggards.

The sleeves are coded but DORMANT: they only emit when at least
``min_symbols`` (default 4) equity perps carry sufficient history, and their
candidate wiring intersects with the materialized universe.  They auto-discover
through the plugin registry; validate scoring on the data-bearing machine.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.advanced_alpha import cross_leadlag_spillover
from lumina_quant.indicators.alpha_features import (
    log_return,
    realized_volatility,
    rolling_zscore,
    simple_return,
)
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.research_factors import leadlag_spillover
from lumina_quant.indicators.rolling_stats import (
    RollingZScoreWindow,
    rolling_beta,
    ts_regression_slope,
)
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import (
    _age_cross_positions,
    _emit_rebalance_targets,
    _ranked_targets,
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

_DEFAULT_BENCHMARKS = ("SPYUSDT", "SPY/USDT", "QQQUSDT", "QQQ/USDT", "BTC/USDT", "BTCUSDT")
_DEFAULT_SEMIS_LEADER = ("SOXLUSDT", "SOXL/USDT")


def _default_benchmark(symbols: list[str], preferred: str) -> str:
    if preferred in symbols:
        return preferred
    for candidate in _DEFAULT_BENCHMARKS:
        if candidate in symbols:
            return candidate
    return symbols[0] if symbols else preferred


def _state_size(*values: int) -> int:
    return max(8, max(int(value) for value in values) + 8)


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


class _CrossUpdateMixin:
    """Shared per-symbol close ingestion / state serialization for XS sleeves."""

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


@register("strategy", "CrossSectionalEquityMomentumStrategy", interface="event_driven")
class CrossSectionalEquityMomentumStrategy(_CrossUpdateMixin, Strategy):
    """Vol-scaled 12-1 cross-sectional equity momentum with a regime short gate."""

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "lookback_bars": HyperParam.integer("lookback_bars", default=252, low=8, high=20000),
            "skip_bars": HyperParam.integer("skip_bars", default=21, low=0, high=4096),
            "vol_window": HyperParam.integer("vol_window", default=63, low=4, high=4096),
            "regime_sma_bars": HyperParam.integer(
                "regime_sma_bars", default=200, low=8, high=20000
            ),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=21, low=1, high=10080),
            "signal_threshold": HyperParam.floating(
                "signal_threshold", default=0.0, low=0.0, high=20.0
            ),
            "quintile_pct": HyperParam.floating("quintile_pct", default=0.20, low=0.05, high=0.50),
            "min_symbols": HyperParam.integer("min_symbols", default=4, low=2, high=512),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.12, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=252, low=1, high=200000),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=0.40, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=500.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.lookback_bars = max(2, int(resolved["lookback_bars"]))
        self.skip_bars = max(0, int(resolved["skip_bars"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.regime_sma_bars = max(2, int(resolved["regime_sma_bars"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.signal_threshold = max(0.0, float(resolved["signal_threshold"]))
        self.quintile_pct = max(0.01, min(0.50, float(resolved["quintile_pct"])))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.allow_short = bool(resolved["allow_short"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(
            self.lookback_bars + self.skip_bars,
            self.regime_sma_bars,
            self.vol_window,
            self.max_hold_bars,
        )
        self._state = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0

    def _suppress_shorts(self) -> bool:
        below: list[bool] = []
        for item in self._state.values():
            if len(item.closes) < self.regime_sma_bars:
                continue
            close = item.closes[-1]
            sma = sum(list(item.closes)[-self.regime_sma_bars :]) / float(self.regime_sma_bars)
            below.append(close < sma)
        if not below:
            return False
        # Suppress shorts when the basket-median symbol trades above its SMA
        # (broad uptrend); allow them when the median is below its SMA.
        below_fraction = sum(1 for flag in below if flag) / float(len(below))
        return below_fraction < 0.5

    def _rebalance(self, event_time: Any) -> None:
        if self._tick % self.rebalance_bars:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id="cross_sectional_equity_momentum",
                strategy_name="CrossSectionalEquityMomentumStrategy",
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        rows: list[tuple[float, str, dict[str, Any]]] = []
        required = self.lookback_bars + self.skip_bars + 1
        for symbol, item in self._state.items():
            if len(item.closes) < required:
                continue
            # 12-1 momentum: total return over lookback, skipping the most
            # recent skip_bars to avoid short-term reversal.
            closes = list(item.closes)
            recent = closes[-(self.skip_bars + 1)] if self.skip_bars > 0 else closes[-1]
            base = closes[-required]
            if base <= 0.0 or recent <= 0.0:
                continue
            mom = recent / base - 1.0
            vol = realized_volatility(item.closes, window=self.vol_window)
            if vol is None or vol <= _EPS:
                continue
            score = mom / max(vol, _EPS)
            rows.append(
                (
                    float(score),
                    symbol,
                    {"momentum_12_1": mom, "realized_vol": vol},
                )
            )
        if len(rows) < self.min_symbols:
            return
        max_side = max(1, int(len(rows) * self.quintile_pct))
        allow_short = self.allow_short and not self._suppress_shorts()
        targets = _ranked_targets(
            rows,
            threshold=self.signal_threshold,
            max_longs=max_side,
            max_shorts=max_side if allow_short else 0,
            allow_short=allow_short,
        )
        _emit_rebalance_targets(
            self.events,
            self._state,
            targets,
            event_time=event_time,
            strategy_id="cross_sectional_equity_momentum",
            strategy_name="CrossSectionalEquityMomentumStrategy",
            target_gross_exposure=self.target_gross_exposure,
            max_order_value=self.max_order_value,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
            threshold=max(self.signal_threshold, _EPS),
        )


@register("strategy", "ResidualEquityMomentumStrategy", interface="event_driven")
class ResidualEquityMomentumStrategy(_CrossUpdateMixin, Strategy):
    """Idiosyncratic (residual) momentum vs a benchmark, inverse-residual-vol sized."""

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "benchmark_symbol": HyperParam.string(
                "benchmark_symbol", default="SPYUSDT", tunable=False
            ),
            "lookback_bars": HyperParam.integer("lookback_bars", default=252, low=8, high=20000),
            "skip_bars": HyperParam.integer("skip_bars", default=21, low=0, high=4096),
            "beta_window": HyperParam.integer("beta_window", default=252, low=8, high=20000),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=21, low=1, high=10080),
            "signal_threshold": HyperParam.floating(
                "signal_threshold", default=0.0, low=0.0, high=20.0
            ),
            "quintile_pct": HyperParam.floating("quintile_pct", default=0.20, low=0.05, high=0.50),
            "min_symbols": HyperParam.integer("min_symbols", default=4, low=2, high=512),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.12, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=252, low=1, high=200000),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=0.40, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=500.0, low=0.0, high=1_000_000.0, tunable=False
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
        self.lookback_bars = max(2, int(resolved["lookback_bars"]))
        self.skip_bars = max(0, int(resolved["skip_bars"]))
        self.beta_window = max(2, int(resolved["beta_window"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.signal_threshold = max(0.0, float(resolved["signal_threshold"]))
        self.quintile_pct = max(0.01, min(0.50, float(resolved["quintile_pct"])))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.allow_short = bool(resolved["allow_short"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(
            self.lookback_bars + self.skip_bars,
            self.beta_window,
            self.max_hold_bars,
        )
        self._state = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0

    @staticmethod
    def _bar_log_returns(closes: list[float]) -> list[float]:
        out: list[float] = []
        prev = None
        for value in closes:
            if prev is not None and prev > 0.0 and value > 0.0:
                out.append(math.log(value / prev))
            prev = value
        return out

    def _rebalance(self, event_time: Any) -> None:
        if self._tick % self.rebalance_bars:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id="residual_equity_momentum",
                strategy_name="ResidualEquityMomentumStrategy",
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        benchmark = self._state.get(self.benchmark_symbol)
        required = self.lookback_bars + self.skip_bars + 1
        if benchmark is None or len(benchmark.closes) < required:
            return
        bench_returns = self._bar_log_returns(list(benchmark.closes))
        if len(bench_returns) < max(4, self.beta_window // 4):
            return
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for symbol, item in self._state.items():
            if symbol == self.benchmark_symbol or len(item.closes) < required:
                continue
            sym_returns = self._bar_log_returns(list(item.closes))
            window = min(len(sym_returns), len(bench_returns), self.beta_window)
            if window < 4:
                continue
            x_tail = bench_returns[-window:]
            y_tail = sym_returns[-window:]
            beta = rolling_beta(y_tail, x_tail)
            slope = ts_regression_slope(x_tail, y_tail)
            beta_used = slope if slope is not None else beta
            if beta_used is None:
                continue
            # Residual cumulative return over the t-252 -> t-21 window.
            residuals: list[float] = []
            span = window - max(0, self.skip_bars)
            if span < 2:
                continue
            for ry, rx in zip(y_tail[:span], x_tail[:span], strict=False):
                residuals.append(ry - beta_used * rx)
            if len(residuals) < 2:
                continue
            residual_cumret = sum(residuals)
            mean_res = residual_cumret / float(len(residuals))
            var_res = sum((r - mean_res) ** 2 for r in residuals) / float(len(residuals) - 1)
            residual_vol = var_res**0.5
            if residual_vol <= _EPS:
                continue
            score = residual_cumret / max(residual_vol, _EPS)
            rows.append(
                (
                    float(score),
                    symbol,
                    {
                        "residual_cumret": residual_cumret,
                        "residual_vol": residual_vol,
                        "beta": beta_used,
                    },
                )
            )
        if len(rows) < self.min_symbols:
            return
        max_side = max(1, int(len(rows) * self.quintile_pct))
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
            strategy_id="residual_equity_momentum",
            strategy_name="ResidualEquityMomentumStrategy",
            target_gross_exposure=self.target_gross_exposure,
            max_order_value=self.max_order_value,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
            threshold=max(self.signal_threshold, _EPS),
        )


@register("strategy", "BettingAgainstBetaStrategy", interface="event_driven")
class BettingAgainstBetaStrategy(_CrossUpdateMixin, Strategy):
    """Long low-beta / short high-beta quintiles, beta-neutralized."""

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "benchmark_symbol": HyperParam.string(
                "benchmark_symbol", default="SPYUSDT", tunable=False
            ),
            "beta_window": HyperParam.integer("beta_window", default=252, low=8, high=20000),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=21, low=1, high=10080),
            "quintile_pct": HyperParam.floating("quintile_pct", default=0.20, low=0.05, high=0.50),
            "beta_z_window": HyperParam.integer("beta_z_window", default=63, low=4, high=20000),
            "min_symbols": HyperParam.integer("min_symbols", default=4, low=2, high=512),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.12, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=252, low=1, high=200000),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=0.40, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=500.0, low=0.0, high=1_000_000.0, tunable=False
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
        self.beta_window = max(2, int(resolved["beta_window"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.quintile_pct = max(0.01, min(0.50, float(resolved["quintile_pct"])))
        self.beta_z_window = max(2, int(resolved["beta_z_window"]))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.allow_short = bool(resolved["allow_short"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(self.beta_window, self.beta_z_window, self.max_hold_bars)
        self._state = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._beta_windows: dict[str, RollingZScoreWindow] = {
            symbol: RollingZScoreWindow(self.beta_z_window) for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0

    @staticmethod
    def _bar_log_returns(closes: list[float]) -> list[float]:
        out: list[float] = []
        prev = None
        for value in closes:
            if prev is not None and prev > 0.0 and value > 0.0:
                out.append(math.log(value / prev))
            prev = value
        return out

    def _rebalance(self, event_time: Any) -> None:
        if self._tick % self.rebalance_bars:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id="betting_against_beta",
                strategy_name="BettingAgainstBetaStrategy",
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        benchmark = self._state.get(self.benchmark_symbol)
        if benchmark is None or len(benchmark.closes) < self.beta_window:
            return
        bench_returns = self._bar_log_returns(list(benchmark.closes))
        if len(bench_returns) < 4:
            return
        betas: list[tuple[float, str]] = []
        for symbol, item in self._state.items():
            if symbol == self.benchmark_symbol or len(item.closes) < self.beta_window:
                continue
            sym_returns = self._bar_log_returns(list(item.closes))
            window = min(len(sym_returns), len(bench_returns), self.beta_window)
            if window < 4:
                continue
            beta = rolling_beta(sym_returns[-window:], bench_returns[-window:])
            if beta is None:
                continue
            betas.append((float(beta), symbol))
        if len(betas) < self.min_symbols:
            return
        # Long low-beta, short high-beta: score is inverse-rank of beta.
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for beta, symbol in betas:
            # Lower beta -> higher score (long); higher beta -> negative score.
            score = -beta
            rows.append((float(score), symbol, {"beta": beta}))
        max_side = max(1, int(len(rows) * self.quintile_pct))
        targets = _ranked_targets(
            rows,
            threshold=0.0,
            max_longs=max_side,
            max_shorts=max_side if self.allow_short else 0,
            allow_short=self.allow_short,
        )
        _emit_rebalance_targets(
            self.events,
            self._state,
            targets,
            event_time=event_time,
            strategy_id="betting_against_beta",
            strategy_name="BettingAgainstBetaStrategy",
            target_gross_exposure=self.target_gross_exposure,
            max_order_value=self.max_order_value,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
            threshold=_EPS,
        )


@register("strategy", "SemisLeadLagRotationStrategy", interface="event_driven")
class SemisLeadLagRotationStrategy(_CrossUpdateMixin, Strategy):
    """Tilt toward under-reacted semis laggards after a >1 sigma SOXL leader move."""

    decision_cadence_seconds = 3600
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "leader_symbol": HyperParam.string("leader_symbol", default="SOXLUSDT", tunable=False),
            "leader_lookback_bars": HyperParam.integer(
                "leader_lookback_bars", default=4, low=1, high=4096
            ),
            "leader_z_window": HyperParam.integer(
                "leader_z_window", default=120, low=4, high=20000
            ),
            "follower_lookback_bars": HyperParam.integer(
                "follower_lookback_bars", default=2, low=1, high=4096
            ),
            "leader_sigma": HyperParam.floating("leader_sigma", default=1.0, low=0.0, high=10.0),
            "spillover_window": HyperParam.integer(
                "spillover_window", default=240, low=32, high=20000
            ),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=2, low=1, high=10080),
            "max_longs": HyperParam.integer("max_longs", default=3, low=0, high=128),
            "min_symbols": HyperParam.integer("min_symbols", default=4, low=2, high=512),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.045, low=0.0, high=0.50
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=96, low=1, high=200000),
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
        preferred_leader = str(resolved["leader_symbol"])
        self.leader_symbol = preferred_leader
        if preferred_leader not in self.symbol_list:
            for candidate in _DEFAULT_SEMIS_LEADER:
                if candidate in self.symbol_list:
                    self.leader_symbol = candidate
                    break
        self.leader_lookback_bars = max(1, int(resolved["leader_lookback_bars"]))
        self.leader_z_window = max(3, int(resolved["leader_z_window"]))
        self.follower_lookback_bars = max(1, int(resolved["follower_lookback_bars"]))
        self.leader_sigma = max(0.0, float(resolved["leader_sigma"]))
        self.spillover_window = max(32, int(resolved["spillover_window"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.max_longs = max(0, int(resolved["max_longs"]))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.allow_short = bool(resolved["allow_short"])
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = _state_size(
            self.leader_z_window,
            self.spillover_window,
            self.max_hold_bars,
        )
        self._state = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._leader_returns: deque[float] = deque(maxlen=size)
        self._last_eval_time_key = ""
        self._tick = 0

    def _rebalance(self, event_time: Any) -> None:
        if self._tick % self.rebalance_bars:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id="semis_lead_lag_rotation",
                strategy_name="SemisLeadLagRotationStrategy",
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        leader = self._state.get(self.leader_symbol)
        if leader is None or len(leader.closes) <= self.leader_lookback_bars:
            return
        leader_ret = log_return(leader.closes, lookback=self.leader_lookback_bars)
        if leader_ret is None:
            return
        self._leader_returns.append(float(leader_ret))
        leader_z = rolling_zscore(self._leader_returns, window=self.leader_z_window)
        if leader_z is None or abs(leader_z) < self.leader_sigma:
            _age_cross_positions(
                self.events,
                self._state,
                event_time=event_time,
                strategy_id="semis_lead_lag_rotation",
                strategy_name="SemisLeadLagRotationStrategy",
                stop_loss_pct=self.stop_loss_pct,
                max_hold_bars=self.max_hold_bars,
            )
            return
        # Estimate spillover predictions to confirm the leader/lagger structure.
        price_by_symbol = {
            symbol: list(item.closes)
            for symbol, item in self._state.items()
            if len(item.closes) >= 32
        }
        predictions: dict[str, Any] = {}
        if len(price_by_symbol) >= 3:
            try:
                spill = cross_leadlag_spillover(
                    price_by_symbol,
                    leaders=(self.leader_symbol,),
                    window=self.spillover_window,
                )
                predictions = dict(spill.get("predictions") or {})
                if not predictions:
                    legacy = leadlag_spillover(price_by_symbol, window=self.spillover_window)
                    predictions = dict(legacy.get("predictions") or {})
            except Exception:
                predictions = {}
        leader_up = leader_z > 0.0
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for symbol, item in self._state.items():
            if symbol == self.leader_symbol or len(item.closes) <= self.follower_lookback_bars:
                continue
            follower_ret = simple_return(item.closes, lookback=self.follower_lookback_bars)
            if follower_ret is None:
                continue
            # Under-reaction gap: leader moved, follower has not yet caught up.
            gap = leader_ret - follower_ret
            score = gap if leader_up else -gap
            pred = predictions.get(symbol) or predictions.get(symbol.replace("USDT", "/USDT"))
            spill_score = float((pred or {}).get("score", 0.0)) if isinstance(pred, dict) else 0.0
            rows.append(
                (
                    float(score),
                    symbol,
                    {
                        "leader_z": leader_z,
                        "leader_return": leader_ret,
                        "follower_return": follower_ret,
                        "spillover_score": spill_score,
                    },
                )
            )
        if len(rows) < self.min_symbols:
            return
        targets = _ranked_targets(
            rows,
            threshold=0.0,
            max_longs=self.max_longs,
            max_shorts=self.max_longs if self.allow_short else 0,
            allow_short=self.allow_short,
        )
        _emit_rebalance_targets(
            self.events,
            self._state,
            targets,
            event_time=event_time,
            strategy_id="semis_lead_lag_rotation",
            strategy_name="SemisLeadLagRotationStrategy",
            target_gross_exposure=self.target_gross_exposure,
            max_order_value=self.max_order_value,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
            threshold=_EPS,
        )


__all__ = [
    "BettingAgainstBetaStrategy",
    "CrossSectionalEquityMomentumStrategy",
    "ResidualEquityMomentumStrategy",
    "SemisLeadLagRotationStrategy",
]
