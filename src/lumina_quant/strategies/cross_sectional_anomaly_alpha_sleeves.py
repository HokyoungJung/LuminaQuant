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
from datetime import timedelta
from itertools import pairwise
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
    _event_datetime_utc,
    _window_snapshot,
)
from lumina_quant.strategies.robust_alpha_sleeves import (
    _CrossSectionalState,
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


def _valid_float(value: Any, *, positive: bool = False) -> float | None:
    """Return a finite numeric checkpoint value without coercing wire data."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or (positive and number <= _EPS):
        return None
    return number


def _validated_cross_state(
    state: Any, symbols: list[str], size: int
) -> tuple[str, int, dict[str, _CrossSectionalState], dict[str, deque[str]]] | None:
    """Validate the complete base checkpoint before constructing replacement state."""
    if not isinstance(state, dict) or set(state) != {
        "last_eval_time_key",
        "tick",
        "symbol_state",
        "close_times",
    }:
        return None
    last_eval_key = state["last_eval_time_key"]
    tick = state["tick"]
    raw_items = state["symbol_state"]
    raw_times = state["close_times"]
    if (
        not isinstance(last_eval_key, str)
        or (last_eval_key and _event_datetime_from_key(last_eval_key) is None)
        or isinstance(tick, bool)
        or not isinstance(tick, int)
        or tick < 0
        or not isinstance(raw_items, dict)
        or not isinstance(raw_times, dict)
        or set(raw_items) != set(symbols)
        or set(raw_times) != set(symbols)
    ):
        return None
    items: dict[str, _CrossSectionalState] = {}
    close_times: dict[str, deque[str]] = {}
    common_times: list[str] | None = None
    for symbol in symbols:
        payload = raw_items[symbol]
        times = raw_times[symbol]
        if (
            not isinstance(payload, dict)
            or set(payload)
            != {"closes", "volumes", "mode", "entry_price", "bars_held", "last_time_key"}
            or not isinstance(payload["closes"], list)
            or not isinstance(payload["volumes"], list)
            or not isinstance(times, list)
            or not (len(payload["closes"]) == len(payload["volumes"]) == len(times))
            or len(times) > size
            or not isinstance(payload["mode"], str)
            or payload["mode"] not in {"OUT", "LONG", "SHORT"}
            or isinstance(payload["bars_held"], bool)
            or not isinstance(payload["bars_held"], int)
            or payload["bars_held"] < 0
        ):
            return None
        closes = [_valid_float(value, positive=True) for value in payload["closes"]]
        volumes = [_valid_float(value) for value in payload["volumes"]]
        entry_price = payload["entry_price"]
        if any(value is None for value in closes) or any(value is None for value in volumes):
            return None
        if entry_price is not None and _valid_float(entry_price, positive=True) is None:
            return None
        if not all(
            isinstance(key, str) and _event_datetime_from_key(key) is not None for key in times
        ):
            return None
        dts = [_event_datetime_from_key(key) for key in times]
        if any(current <= previous for previous, current in pairwise(dts)):
            return None
        if common_times is None:
            common_times = list(times)
        elif times != common_times:
            return None
        item_last_key = payload["last_time_key"]
        if (
            not isinstance(item_last_key, str)
            or (times and item_last_key != times[-1])
            or (not times and item_last_key)
            or (times and last_eval_key != times[-1])
            or (not times and last_eval_key)
        ):
            return None
        if (payload["mode"] == "OUT") != (entry_price is None):
            return None
        items[symbol] = _CrossSectionalState(
            deque((float(value) for value in closes), maxlen=size),
            deque((float(value) for value in volumes), maxlen=size),
            payload["mode"],
            _valid_float(entry_price, positive=True) if entry_price is not None else None,
            payload["bars_held"],
            item_last_key,
        )
        close_times[symbol] = deque(times, maxlen=size)
    return last_eval_key, tick, items, close_times


def _bar_simple_returns(closes: list[float]) -> list[float]:
    """Return the bar-to-bar simple returns of a price path (drops bad bars)."""
    out: list[float] = []
    prev: float | None = None
    for value in closes:
        if prev is not None and abs(prev) > _EPS and value == value:
            out.append(value / prev - 1.0)
        prev = value
    return out


def _event_datetime_from_key(key: str) -> Any:
    """Parse a normalized key, including numeric keys made by ``time_key``."""
    dt = _event_datetime_utc(key)
    if dt is not None:
        return dt
    try:
        return _event_datetime_utc(float(key))
    except TypeError, ValueError:
        return None


def _daily_simple_returns(
    closes: list[float], keys: list[str], window: int | None = None
) -> list[float] | None:
    """Return completed UTC-day returns; never substitute bar returns for MAX."""
    if len(closes) != len(keys):
        return None
    by_day: dict[str, tuple[Any, str, float]] = {}
    for close, key in zip(closes, keys, strict=True):
        dt = _event_datetime_from_key(key)
        if dt is None or close <= _EPS:
            return None
        day = dt.date().isoformat()
        previous = by_day.get(day)
        if previous is None or dt > previous[0]:
            by_day[day] = (dt, key, close)
    completed_days = sorted(by_day)
    # The newest observed UTC date is still open. Its close must not affect
    # MAX, regardless of the bar cadence.
    if len(completed_days) < 3:
        return None
    completed_days = completed_days[:-1]
    if window is not None:
        entries = [(by_day[day][1], by_day[day][2]) for day in completed_days[-(window + 1) :]]
        entries = _consecutive_completed_daily_closes(entries, window + 1)
        if entries is None:
            return None
        return _bar_simple_returns([close for _, close in entries])
    return _bar_simple_returns([by_day[day][2] for day in completed_days])


def _consecutive_completed_daily_closes(entries: Any, count: int) -> list[tuple[str, float]] | None:
    """Validate the exact UTC-day close tail required for Lottery MAX."""
    if not isinstance(entries, list) or len(entries) != count:
        return None
    restored: list[tuple[str, float]] = []
    dates: list[Any] = []
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2 or not isinstance(entry[0], str):
            return None
        dt = _event_datetime_from_key(entry[0])
        close = _valid_float(entry[1], positive=True)
        if dt is None or close is None:
            return None
        restored.append((entry[0], close))
        dates.append(dt.date())
    if any(current != previous + timedelta(days=1) for previous, current in pairwise(dates)):
        return None
    return restored


class _CrossUpdateMixin:
    """Shared per-symbol close ingestion / state serialization for XS sleeves."""

    min_price: float
    symbol_list: list[str]
    _state: dict[str, _CrossSectionalState]
    _last_eval_time_key: str
    _tick: int

    def _append_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        """Append a snapshot already validated as part of a complete panel."""
        item = self._state[symbol]
        key = time_key(snapshot.time)
        item.last_time_key = key
        item.closes.append(float(snapshot.close))
        item.volumes.append(max(0.0, float(snapshot.volume or 0.0)))
        self._close_times[symbol].append(key)

    def _window_updates(self, event: Any) -> list[tuple[str, _Snapshot]] | None:
        """Return one exact-time snapshot per configured name, or reject the batch."""
        event_key = time_key(getattr(event, "time", None))
        event_dt = _event_datetime_from_key(event_key)
        last_eval_dt = _event_datetime_from_key(self._last_eval_time_key)
        if event_dt is None or (last_eval_dt is not None and event_dt <= last_eval_dt):
            return None
        try:
            payload_symbols = list(dict(getattr(event, "bars_1s", {}) or {}))
        except TypeError, ValueError:
            return None
        configured_symbols = set(self.symbol_list)
        if (
            len(configured_symbols) != len(self.symbol_list)
            or len(payload_symbols) != len(set(payload_symbols))
            or set(payload_symbols) != configured_symbols
        ):
            return None
        updates: list[tuple[str, _Snapshot]] = []
        for symbol in self.symbol_list:
            raw_rows = list((getattr(event, "bars_1s", {}) or {}).get(symbol) or [])
            row_keys = {
                time_key(
                    row.get("time")
                    if isinstance(row, dict)
                    else row[0]
                    if isinstance(row, (tuple, list)) and row
                    else None
                )
                for row in raw_rows
            }
            snapshot = _window_snapshot(event, symbol)
            last_symbol_dt = _event_datetime_from_key(self._state[symbol].last_time_key)
            if (
                snapshot is None
                or "" in row_keys
                or len(row_keys) != len(raw_rows)
                or time_key(snapshot.time) != event_key
                or safe_float(snapshot.close) is None
                or safe_float(snapshot.close) <= self.min_price
                or (last_symbol_dt is not None and event_dt <= last_symbol_dt)
            ):
                return None
            updates.append((symbol, snapshot))
        return updates

    def _after_window_commit(self, updates: list[tuple[str, _Snapshot]]) -> None:
        """Optional strategy-specific processing after an atomic panel append."""
        _ = updates

    def _synchronized_returns(
        self, symbol: str, benchmark_symbol: str, window: int
    ) -> tuple[list[float], list[float]] | None:
        """Return a complete, regular common-grid return tail, or abstain."""
        item = self._state[symbol]
        benchmark = self._state[benchmark_symbol]
        symbol_times = list(self._close_times[symbol])
        benchmark_times = list(self._close_times[benchmark_symbol])
        count = min(
            len(item.closes), len(benchmark.closes), len(symbol_times), len(benchmark_times)
        )
        if count < window + 1:
            return None
        symbol_closes = list(item.closes)[-count:]
        benchmark_closes = list(benchmark.closes)[-count:]
        if symbol_times[-count:] != benchmark_times[-count:]:
            return None
        tail_times = symbol_times[-(window + 1) :]
        tail_dts = [_event_datetime_from_key(key) for key in tail_times]
        if len(tail_dts) != window + 1 or any(dt is None for dt in tail_dts):
            return None
        gaps = [tail_dts[index + 1] - tail_dts[index] for index in range(len(tail_dts) - 1)]
        if not gaps or gaps[0].total_seconds() <= 0 or any(gap != gaps[0] for gap in gaps[1:]):
            return None
        return (
            _bar_simple_returns(symbol_closes[-(window + 1) :]),
            _bar_simple_returns(benchmark_closes[-(window + 1) :]),
        )

    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
            "symbol_state": {symbol: _pack_cross(item) for symbol, item in self._state.items()},
            "close_times": {symbol: list(times) for symbol, times in self._close_times.items()},
        }

    def set_state(self, state: dict[str, Any]) -> None:
        size = next(iter(self._state.values())).closes.maxlen or 0
        validated = _validated_cross_state(state, self.symbol_list, size)
        if validated is None:
            return
        self._last_eval_time_key, self._tick, self._state, self._close_times = validated

    def _rebalance(self, event_time: Any) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        updates = self._window_updates(event)
        if updates is None:
            return
        for symbol, snapshot in updates:
            self._append_symbol(symbol, snapshot)
        self._after_window_commit(updates)
        self._last_eval_time_key = time_key(getattr(event, "time", None))
        self._tick += 1
        self._rebalance(getattr(event, "time", None))

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if getattr(event, "type", None) != "MARKET":
            return
        # A single-symbol callback cannot establish a common-time panel and
        # therefore must not mutate cross-sectional history.


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
        self._close_times = {symbol: deque(maxlen=size) for symbol in self.symbol_list}
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
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for symbol, item in self._state.items():
            if symbol == self.benchmark_symbol or len(item.closes) < need:
                continue
            aligned = self._synchronized_returns(
                symbol, self.benchmark_symbol, max(self.beta_window, self.vol_window)
            )
            if aligned is None:
                continue
            y_tail, x_tail = aligned
            if len(y_tail) < max(self.beta_window, self.vol_window):
                continue
            # rolling_beta(a, b) = cov(a, b) / var(b): pass the SYMBOL series first
            # and the BENCHMARK second so beta is the asset-on-market loading
            # (cov(sym, bench) / var(bench)); the residual below then removes the
            # systematic component, leaving the idiosyncratic return.
            beta = rolling_beta(y_tail[-self.beta_window :], x_tail[-self.beta_window :])
            if beta is None:
                continue
            vol_returns = y_tail[-self.vol_window :]
            vol_benchmark_returns = x_tail[-self.vol_window :]
            residuals = [
                ry - beta * rx for ry, rx in zip(vol_returns, vol_benchmark_returns, strict=True)
            ]
            idio_vol = sample_std(residuals)
            total_vol = sample_std(vol_returns)
            if idio_vol is None or total_vol is None or idio_vol <= _EPS or total_vol <= _EPS:
                continue
            # Low idio-vol -> high score (long); high idio-vol -> low score (short).
            standardized_idio_vol = float(idio_vol) / float(total_vol)
            score = -standardized_idio_vol
            rows.append(
                (
                    score,
                    symbol,
                    {
                        "idiosyncratic_vol": float(idio_vol),
                        "standardized_idiosyncratic_vol": standardized_idio_vol,
                        "beta": float(beta),
                    },
                )
            )
        if len(rows) < self.min_symbols:
            return
        if max(score for score, _, _ in rows) - min(score for score, _, _ in rows) <= _EPS:
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
        self._close_times = {symbol: deque(maxlen=size) for symbol in self.symbol_list}
        # MAX needs max_window daily returns, hence max_window + 1 completed
        # daily closes. Keep this separately from bar history so sub-daily
        # streams cannot evict the required daily observations.
        self._completed_daily_closes = {
            symbol: deque(maxlen=self.max_window + 1) for symbol in self.symbol_list
        }
        self._open_daily_closes: dict[str, tuple[str, float] | None] = dict.fromkeys(
            self.symbol_list
        )
        self._last_eval_time_key = ""
        self._tick = 0

    def _after_window_commit(self, updates: list[tuple[str, _Snapshot]]) -> None:
        """Advance daily-close state only after every panel symbol committed."""
        for symbol, snapshot in updates:
            key = time_key(snapshot.time)
            dt = _event_datetime_from_key(key)
            close = safe_float(snapshot.close)
            if dt is None or close is None:
                continue
            open_close = self._open_daily_closes[symbol]
            open_dt = _event_datetime_from_key(open_close[0]) if open_close is not None else None
            if open_close is not None and open_dt is not None and open_dt.date() != dt.date():
                self._completed_daily_closes[symbol].append(open_close)
            # This timestamp is strictly newer by window validation.
            self._open_daily_closes[symbol] = (key, float(close))

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["completed_daily_closes"] = {
            symbol: [[key, close] for key, close in closes]
            for symbol, closes in self._completed_daily_closes.items()
        }
        state["open_daily_closes"] = {
            symbol: [entry[0], entry[1]] if entry is not None else None
            for symbol, entry in self._open_daily_closes.items()
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or set(state) != {
            "last_eval_time_key",
            "tick",
            "symbol_state",
            "close_times",
            "completed_daily_closes",
            "open_daily_closes",
        }:
            return
        size = next(iter(self._state.values())).closes.maxlen or 0
        base = _validated_cross_state(
            {
                key: state[key]
                for key in ("last_eval_time_key", "tick", "symbol_state", "close_times")
            },
            self.symbol_list,
            size,
        )
        completed = state["completed_daily_closes"]
        open_closes = state["open_daily_closes"]
        if (
            base is None
            or not isinstance(completed, dict)
            or not isinstance(open_closes, dict)
            or set(completed) != set(self.symbol_list)
            or set(open_closes) != set(self.symbol_list)
        ):
            return
        restored_completed: dict[str, deque[tuple[str, float]]] = {}
        restored_open: dict[str, tuple[str, float] | None] = {}
        for symbol in self.symbol_list:
            raw_entries = completed[symbol]
            entry = open_closes[symbol]
            if not isinstance(raw_entries, list) or len(raw_entries) > self.max_window + 1:
                return
            entries = _consecutive_completed_daily_closes(raw_entries, len(raw_entries))
            if entries is None:
                return
            if entry is None:
                if entries or base[2][symbol].closes or base[2][symbol].last_time_key:
                    return
                restored_completed[symbol] = deque(maxlen=self.max_window + 1)
                restored_open[symbol] = None
                continue
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                return
            key, close = entry
            if not isinstance(key, str):
                return
            dt = _event_datetime_from_key(key)
            number = _valid_float(close, positive=True)
            last_dt = _event_datetime_from_key(entries[-1][0]) if entries else None
            latest_close = float(base[2][symbol].closes[-1]) if base[2][symbol].closes else None
            if (
                dt is None
                or number is None
                or (last_dt is not None and dt.date() != last_dt.date() + timedelta(days=1))
                or key != base[2][symbol].last_time_key
                or latest_close is None
                or number != latest_close
            ):
                return
            restored_completed[symbol] = deque(entries, maxlen=self.max_window + 1)
            restored_open[symbol] = (key, number)
        (
            self._last_eval_time_key,
            self._tick,
            self._state,
            self._close_times,
        ) = base
        self._completed_daily_closes = restored_completed
        self._open_daily_closes = restored_open

    def _lottery_score(
        self,
        closes: list[float],
        keys: list[str] | None = None,
        completed_daily_closes: list[tuple[str, float]] | None = None,
    ) -> tuple[float, dict[str, Any]] | None:
        returns = _bar_simple_returns(closes)
        if len(returns) < max(3, self.skew_window // 2):
            return None
        skew = _skewness(returns[-self.skew_window :])
        daily_returns = (
            _bar_simple_returns(
                [close for _, close in completed_daily_closes]
                if _consecutive_completed_daily_closes(completed_daily_closes, self.max_window + 1)
                is not None
                else []
            )
            if completed_daily_closes is not None
            else _daily_simple_returns(closes, keys, self.max_window)
            if keys is not None
            else None
        )
        if daily_returns is None or len(daily_returns) < self.max_window:
            return None
        max_tail = daily_returns[-self.max_window :]
        max_ret = max(max_tail)
        max_scale = sample_std(max_tail)
        if max_scale is None or max_scale <= _EPS:
            return None
        if skew is None and max_ret is None:
            return None
        skew_component = float(skew) if skew is not None else 0.0
        max_component = float(max_ret) / float(max_scale)
        # Higher lottery score -> more lottery-like (short candidate).
        lottery = (1.0 - self.max_weight) * skew_component + self.max_weight * max_component
        return float(lottery), {
            "skewness": skew_component,
            "max_daily_return": float(max_ret),
            "standardized_max_daily_return": max_component,
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
            scored = self._lottery_score(
                list(item.closes),
                list(self._close_times[symbol]),
                list(self._completed_daily_closes[symbol]),
            )
            if scored is None:
                continue
            lottery, meta = scored
            # Long low-lottery (negate so low lottery -> high rank score).
            rows.append((-lottery, symbol, meta))
        if len(rows) < self.min_symbols:
            return
        if max(score for score, _, _ in rows) - min(score for score, _, _ in rows) <= _EPS:
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
        self._close_times = {symbol: deque(maxlen=size) for symbol in self.symbol_list}
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
        self._close_times = {symbol: deque(maxlen=size) for symbol in self.symbol_list}
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
