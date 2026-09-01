"""Directional, long-biased EQUITY/ETF return-rider and rotation alpha sleeves.

These sleeves target the equity/ETF tradfi universe (cleanly, never the crypto
perp set) and are built to MAXIMIZE compound return by riding winners.  They
reuse the proven per-symbol return-rider ride machinery
(:class:`~lumina_quant.strategies.return_rider_alpha_sleeves._ReturnRiderBase`:
ATR trailing stop, pyramiding, vol-targeted sizing) and the dual-momentum index
rotation machinery, with equity-appropriate, long-only behaviour.

- :class:`LeveragedTrendTimingRiderStrategy`: a long-only leveraged-ETF trend
  timer (S-EQ2).  It enters LONG only when price is above a long regime SMA AND a
  golden-cross (fast SMA above slow SMA) is confirmed for a buffer of bars, then
  rides the move with the inherited ATR trailing stop / pyramiding and exits to
  flat (cash) the moment the trend filter is lost.  Its sizing override adds a
  decay penalty that shrinks the allocation as realized volatility rises so the
  notorious leveraged-ETF volatility decay is throttled.  ``allow_short`` is
  hard-disabled (held through a downtrend, leveraged ETFs decay catastrophically;
  the regime filter is the entire edge).
- :class:`DualMomentumDefensiveRotationStrategy`: a long-only dual-momentum index
  rotation with a DEFENSIVE leg (S-EQ3).  An absolute-momentum gate on the
  benchmark's blended return decides the regime: risk-on -> rank the equity/index
  ETFs and hold the top-N above their long SMA (relative momentum); risk-off ->
  rotate 100% LONG into the single best defensive instrument by its own momentum
  (precious-metal / energy / long-vol proxy) instead of going flat to cash.

Both remain data-local (no fetching, no artifact writes, no hidden environment
state), emit through the shared helpers, keep bounded deques, guard ``None``/short
history (skip, never raise), and must still be validated on the data-bearing
machine before any live eligibility.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import realized_volatility, simple_return
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.moving_average import simple_moving_average
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
from lumina_quant.strategies.return_rider_alpha_sleeves import (
    _ReturnRiderBase,
    _RiderCommon,
    _RiderState,
)
from lumina_quant.strategies.robust_alpha_sleeves import (
    _CrossSectionalState,
    _mode,
    _restore_deque,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@register("strategy", "LeveragedTrendTimingRiderStrategy", interface="event_driven")
class LeveragedTrendTimingRiderStrategy(_ReturnRiderBase):
    """Long-only leveraged-ETF trend timer with decay-aware sizing.

    Entry (LONG only): the latest close must sit above the long regime SMA
    (``regime_sma_bars``) AND a golden cross must hold — the fast SMA
    (``fast_sma_bars``) above the slow SMA (``slow_sma_bars``) — confirmed for the
    last ``confirm_bars`` bars to cut whipsaw.  The position is then ridden with
    the inherited ATR trailing stop and pyramided into continuation; the trend
    filter loss flips the entry decision off so the inherited ride/stop logic
    flattens to cash.  ``allow_short`` is hard-disabled.

    Sizing: :meth:`_vol_scaled_allocation` is overridden to apply a decay penalty
    on top of the inherited vol-target — the allocation is shrunk toward
    ``decay_floor`` as realized volatility rises above ``decay_vol_ref``, so the
    leveraged-ETF volatility decay is throttled in choppy regimes.
    """

    strategy_name = "LeveragedTrendTimingRiderStrategy"
    strategy_id = "leveraged_trend_timing_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "regime_sma_bars": HyperParam.integer(
                "regime_sma_bars", default=200, low=8, high=20000
            ),
            "fast_sma_bars": HyperParam.integer("fast_sma_bars", default=50, low=2, high=20000),
            "slow_sma_bars": HyperParam.integer("slow_sma_bars", default=200, low=4, high=20000),
            "confirm_bars": HyperParam.integer("confirm_bars", default=3, low=1, high=512),
            "trend_buffer": HyperParam.floating("trend_buffer", default=0.0, low=0.0, high=0.50),
            "decay_vol_ref": HyperParam.floating("decay_vol_ref", default=0.03, low=0.0, high=1.0),
            "decay_floor": HyperParam.floating("decay_floor", default=0.25, low=0.0, high=1.0),
            "trail_atr_mult": HyperParam.floating(
                "trail_atr_mult", default=3.0, low=0.1, high=20.0
            ),
            "atr_period": HyperParam.integer("atr_period", default=14, low=2, high=1024),
            "max_adds": HyperParam.integer("max_adds", default=3, low=0, high=32),
            "add_step_atr": HyperParam.floating("add_step_atr", default=1.0, low=0.0, high=20.0),
            "vol_window": HyperParam.integer("vol_window", default=20, low=2, high=4096),
            "target_vol": HyperParam.floating("target_vol", default=0.02, low=0.0, high=1.0),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=252, low=1, high=200000),
            "allow_short": HyperParam.boolean("allow_short", default=False, grid=[False]),
            "add_alloc_fraction": HyperParam.floating(
                "add_alloc_fraction", default=0.5, low=0.0, high=1.0
            ),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.30, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=5000.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.regime_sma_bars = max(2, int(resolved["regime_sma_bars"]))
        self.fast_sma_bars = max(2, int(resolved["fast_sma_bars"]))
        self.slow_sma_bars = max(self.fast_sma_bars + 1, int(resolved["slow_sma_bars"]))
        self.confirm_bars = max(1, int(resolved["confirm_bars"]))
        self.trend_buffer = max(0.0, float(resolved["trend_buffer"]))
        self.decay_vol_ref = max(0.0, float(resolved["decay_vol_ref"]))
        self.decay_floor = max(0.0, min(1.0, float(resolved["decay_floor"])))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        # ``allow_short`` is hard-disabled regardless of any supplied value: a
        # leveraged ETF held through a downtrend decays catastrophically.
        resolved = dict(resolved)
        resolved["allow_short"] = False
        return self._resolve_common(
            resolved,
            extra_window=max(self.regime_sma_bars, self.slow_sma_bars) + self.confirm_bars,
        )

    def _trend_ok_at(self, closes: list[float], *, offset: int) -> bool:
        """Return whether the long-only trend filter holds ``offset`` bars back.

        ``offset==0`` checks the latest bar.  Both the long regime SMA gate (close
        above SMA, with a buffer) and the fast/slow golden-cross must hold.
        """
        end = len(closes) - int(offset)
        if end <= 0:
            return False
        window = closes[:end]
        regime = simple_moving_average(window, self.regime_sma_bars)
        fast = simple_moving_average(window, self.fast_sma_bars)
        slow = simple_moving_average(window, self.slow_sma_bars)
        if regime is None or fast is None or slow is None:
            return False
        close = window[-1]
        above_regime = close > regime * (1.0 + self.trend_buffer)
        golden_cross = fast > slow * (1.0 + self.trend_buffer)
        return bool(above_regime and golden_cross)

    def _trend_confirmed(self, closes: list[float]) -> bool:
        """Return whether the trend filter has held for ``confirm_bars`` bars."""
        if len(closes) <= self.slow_sma_bars + self.confirm_bars:
            return False
        return all(self._trend_ok_at(closes, offset=offset) for offset in range(self.confirm_bars))

    def _entry_decision(self, item: _RiderState) -> str:
        # LONG only, gated on the confirmed regime SMA + golden cross.
        return "LONG" if self._trend_confirmed(list(item.closes)) else ""

    def _should_pyramid(self, item: _RiderState) -> bool:
        # Only compound while the live (latest-bar) trend filter still holds.
        return self._trend_ok_at(list(item.closes), offset=0)

    def _ride(self, symbol: str, item: _RiderState, snapshot: _Snapshot, close: float) -> None:
        # Exit to flat the moment the trend filter is lost (decay protection),
        # otherwise defer to the inherited ATR trailing-stop / max-hold ride.
        if item.mode == "LONG" and not self._trend_ok_at(list(item.closes), offset=0):
            item.bars_held += 1
            self._emit_exit(symbol, item, snapshot.time, close, "trend_filter_lost")
            return
        super()._ride(symbol, item, snapshot, close)

    def _vol_scaled_allocation(self, closes: list[float]) -> float:
        """Vol-target the base allocation, then apply a decay penalty.

        On top of the inherited ``target_vol/realized_vol`` scaling, the
        allocation is multiplied by a decay factor that shrinks toward
        ``decay_floor`` as realized volatility rises above ``decay_vol_ref`` —
        the leveraged-ETF decay is worst precisely in high-volatility chop.
        """
        alloc = super()._vol_scaled_allocation(closes)
        if self.decay_vol_ref > 0.0:
            vol = realized_volatility(closes, window=self.vol_window)
            if vol is not None and vol > self.decay_vol_ref:
                penalty = self.decay_vol_ref / vol
                penalty = max(self.decay_floor, min(1.0, penalty))
                alloc *= penalty
        return max(0.0, min(self.target_allocation * 2.0, alloc))

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        closes = list(item.closes)
        regime = simple_moving_average(closes, self.regime_sma_bars)
        return {
            "regime_sma": float(regime) if regime is not None else None,
            "allow_short": False,
        }


def _state_size(*values: int) -> int:
    return max(8, max(int(value) for value in values) + 8)


def _parse_symbol_csv(raw: str, available: list[str]) -> list[str]:
    """Parse a comma-separated symbol list, keeping only available symbols."""
    out: list[str] = []
    available_set = set(available)
    for chunk in str(raw or "").split(","):
        text = chunk.strip()
        if text and text in available_set and text not in out:
            out.append(text)
    return out


@register("strategy", "DualMomentumDefensiveRotationStrategy", interface="event_driven")
class DualMomentumDefensiveRotationStrategy(Strategy):
    """Long-only dual-momentum index rotation with a defensive risk-off leg.

    Absolute gate: the benchmark's blended (1/3/6/12-bar) return must be positive
    to be risk-on.  Risk-on -> rank the index/ETF universe by the same blended
    momentum and hold the top ``max_holdings`` confirmed above their long SMA
    (relative momentum, long-only).  Risk-off -> rotate 100% LONG into the single
    best ``defensive_symbols`` instrument by its own blended momentum (e.g.
    precious-metal / energy / long-vol proxy) instead of going flat to cash, so
    the sleeve keeps earning in equity-bear / commodity-bull regimes.
    """

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "DualMomentumDefensiveRotationStrategy"
    strategy_id = "dual_momentum_defensive_rotation"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "benchmark_symbol": HyperParam.string(
                "benchmark_symbol", default="SPYUSDT", tunable=False
            ),
            "defensive_symbols": HyperParam.string("defensive_symbols", default="", tunable=False),
            "blend_lookbacks": HyperParam.string(
                "blend_lookbacks", default="1,3,6,12", tunable=False
            ),
            "absolute_lookback_bars": HyperParam.integer(
                "absolute_lookback_bars", default=12, low=1, high=10080
            ),
            "sma_bars": HyperParam.integer("sma_bars", default=200, low=8, high=20000),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=21, low=1, high=10080),
            "max_holdings": HyperParam.integer("max_holdings", default=3, low=1, high=64),
            "min_symbols": HyperParam.integer("min_symbols", default=4, low=2, high=512),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.12, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=252, low=1, high=200000),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=0.45, low=0.0, high=5.0, tunable=False
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
        self.benchmark_symbol = self._resolve_benchmark(str(resolved["benchmark_symbol"]))
        self.defensive_symbols = _parse_symbol_csv(
            str(resolved["defensive_symbols"]), self.symbol_list
        )
        self.blend_lookbacks = self._parse_lookbacks(str(resolved["blend_lookbacks"] or ""))
        self.absolute_lookback_bars = max(1, int(resolved["absolute_lookback_bars"]))
        self.sma_bars = max(2, int(resolved["sma_bars"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.max_holdings = max(1, int(resolved["max_holdings"]))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        # Defensive symbols form their own selection pool; they should not be
        # double-counted in the risk-on equity ranking pool.
        defensive_set = set(self.defensive_symbols)
        self.rotation_symbols = [s for s in self.symbol_list if s not in defensive_set]
        size = _state_size(
            max(self.blend_lookbacks, default=12),
            self.absolute_lookback_bars,
            self.sma_bars,
            self.max_hold_bars,
        )
        self._state = {
            symbol: _CrossSectionalState(deque(maxlen=size), deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0

    def _resolve_benchmark(self, preferred: str) -> str:
        if preferred in self.symbol_list:
            return preferred
        for candidate in ("SPYUSDT", "SPY/USDT", "QQQUSDT", "QQQ/USDT"):
            if candidate in self.symbol_list:
                return candidate
        return self.symbol_list[0] if self.symbol_list else preferred

    @staticmethod
    def _parse_lookbacks(raw: str) -> list[int]:
        out: list[int] = []
        for chunk in raw.split(","):
            text = chunk.strip()
            if not text:
                continue
            try:
                value = int(float(text))
            except Exception:
                continue
            if value >= 1:
                out.append(value)
        return out or [1, 3, 6, 12]

    def _blended_momentum(self, closes: deque[float]) -> float | None:
        blended: list[float] = []
        for lb in self.blend_lookbacks:
            ret = simple_return(closes, lookback=lb)
            if ret is not None:
                blended.append(ret)
        if not blended:
            return None
        return sum(blended) / float(len(blended))

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
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "volumes": list(item.volumes),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": int(item.bars_held),
                    "last_time_key": item.last_time_key,
                }
                for symbol, item in self._state.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._tick = _safe_non_negative_int(state.get("tick"))
        raw = state.get("symbol_state")
        if isinstance(raw, dict):
            for symbol, payload in raw.items():
                if symbol not in self._state or not isinstance(payload, dict):
                    continue
                item = self._state[symbol]
                _restore_deque(item.closes, payload.get("closes"))
                _restore_deque(item.volumes, payload.get("volumes"))
                item.mode = _mode(payload.get("mode"))
                item.entry_price = safe_float(payload.get("entry_price"))
                item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
                item.last_time_key = str(payload.get("last_time_key", ""))

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

    def _defensive_targets(self) -> dict[str, tuple[str, float, dict[str, Any]]]:
        """Pick the single best defensive instrument by its own momentum (LONG)."""
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for symbol in self.defensive_symbols:
            item = self._state.get(symbol)
            if item is None or len(item.closes) <= max(self.blend_lookbacks, default=12):
                continue
            score = self._blended_momentum(item.closes)
            if score is None:
                continue
            rows.append((float(score), symbol, {"regime": "risk_off", "blended_momentum": score}))
        return _ranked_targets(
            rows,
            threshold=0.0,
            max_longs=1,
            max_shorts=0,
            allow_short=False,
        )

    def _rebalance(self, event_time: Any) -> None:
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
        max_lookback = max(self.blend_lookbacks, default=12)
        benchmark = self._state.get(self.benchmark_symbol)
        bench_ret: float | None = None
        if benchmark is not None and len(benchmark.closes) > self.absolute_lookback_bars:
            bench_ret = simple_return(benchmark.closes, lookback=self.absolute_lookback_bars)
        # Absolute-momentum gate: non-positive benchmark return => risk-off.
        if bench_ret is None or bench_ret <= 0.0:
            targets = self._defensive_targets()
            if not targets:
                # No usable defensive leg yet: flatten rather than over-hold.
                _age_cross_positions(
                    self.events,
                    self._state,
                    event_time=event_time,
                    strategy_id=self.strategy_id,
                    strategy_name=self.strategy_name,
                    stop_loss_pct=0.0,
                    max_hold_bars=1,
                )
                return
            self._emit(targets, event_time)
            return
        # Risk-on: rank the rotation pool by blended momentum, hold top-N above SMA.
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for symbol in self.rotation_symbols:
            item = self._state.get(symbol)
            if item is None or len(item.closes) <= max_lookback or len(item.closes) < self.sma_bars:
                continue
            score = self._blended_momentum(item.closes)
            if score is None:
                continue
            close = item.closes[-1]
            sma = simple_moving_average(item.closes, self.sma_bars)
            if sma is None or close < sma or score <= 0.0:
                continue
            rows.append(
                (
                    float(score),
                    symbol,
                    {"regime": "risk_on", "blended_momentum": score, "sma": sma},
                )
            )
        if len(rows) < min(self.min_symbols, max(2, self.max_holdings)):
            # Not enough qualifying equity ETFs: fall back to the defensive leg.
            targets = self._defensive_targets()
            if not targets:
                _age_cross_positions(
                    self.events,
                    self._state,
                    event_time=event_time,
                    strategy_id=self.strategy_id,
                    strategy_name=self.strategy_name,
                    stop_loss_pct=0.0,
                    max_hold_bars=1,
                )
                return
            self._emit(targets, event_time)
            return
        targets = _ranked_targets(
            rows,
            threshold=0.0,
            max_longs=self.max_holdings,
            max_shorts=0,
            allow_short=False,
        )
        self._emit(targets, event_time)

    def _emit(
        self,
        targets: dict[str, tuple[str, float, dict[str, Any]]],
        event_time: Any,
    ) -> None:
        _emit_rebalance_targets(
            self.events,
            self._state,
            targets,
            event_time=event_time,
            strategy_id=self.strategy_id,
            strategy_name=self.strategy_name,
            target_gross_exposure=self.target_gross_exposure,
            max_order_value=self.max_order_value,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=self.max_hold_bars,
            threshold=_EPS,
        )


__all__ = [
    "DualMomentumDefensiveRotationStrategy",
    "LeveragedTrendTimingRiderStrategy",
]
