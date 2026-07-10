"""Externally motivated alpha sleeves for forward testing.

No data is collected here.  These are drop-in, event-driven strategy classes
built from documented alpha families the user can forward-test on another
machine with local market/feature data:

- Time-series momentum with volatility scaling and crash gates.
- Liquidity/volume shock mean reversion.
- Perpetual-futures funding/basis carry plus trend rotation.
- Opening-range/session momentum continuation.

All classes register through the plugin registry, so they become selectable by
``optimization.strategy`` without editing ``strategies.registry``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from lumina_quant.core.events import SignalEvent
from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.annualization import (
    annualize_per_bar_vol as _canon_annualize_per_bar_vol,
    bars_per_year_from_spacing as _canon_bars_per_year_from_spacing,
    median_bar_spacing_seconds as _canon_median_bar_spacing_seconds,
)
from lumina_quant.indicators.alpha_features import (
    basis_bps,
    clipped_tanh_score,
    drawdown_from_peak,
    log_return,
    order_flow_imbalance,
    range_zscore,
    realized_volatility,
    simple_return,
    trend_efficiency,
    volatility_ratio,
    volume_zscore,
)
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@dataclass(slots=True)
class _SingleAssetState:
    opens: deque[float]
    highs: deque[float]
    lows: deque[float]
    closes: deque[float]
    volumes: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    target_price: float | None = None
    high_watermark: float | None = None
    low_watermark: float | None = None
    bars_held: int = 0
    ticks_seen: int = 0
    last_time_key: str = ""
    last_signal_session: str = ""
    session_key: str = ""
    session_open: float | None = None
    session_high: float | None = None
    session_low: float | None = None
    session_bars: int = 0


@dataclass(slots=True)
class _RotationState:
    closes: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    last_time_key: str = ""


@dataclass(slots=True)
class _CarryState:
    closes: deque[float]
    funding_rate: deque[float]
    basis_bps: deque[float]
    open_interest: deque[float]
    taker_buy_quote_volume: deque[float]
    taker_sell_quote_volume: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    last_time_key: str = ""


@dataclass(frozen=True, slots=True)
class _Snapshot:
    time: Any
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None


_EPS = 1e-12


def _safe_non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return 0


def _row_dict(row: Any, fallback_time: Any = None) -> dict[str, Any]:
    if isinstance(row, dict):
        out = dict(row)
        if "time" not in out and fallback_time is not None:
            out["time"] = fallback_time
        return out
    if isinstance(row, (tuple, list)):
        keys = ("time", "open", "high", "low", "close", "volume")
        return {key: row[idx] for idx, key in enumerate(keys) if idx < len(row)}
    return {}


def _window_snapshot(event: Any, symbol: str) -> _Snapshot | None:
    bars_1s = dict(getattr(event, "bars_1s", {}) or {})
    rows = [_row_dict(row, getattr(event, "time", None)) for row in list(bars_1s.get(symbol) or [])]
    rows = [row for row in rows if row]
    if not rows:
        return None
    first = rows[0]
    last = rows[-1]
    highs = [value for row in rows if (value := safe_float(row.get("high"))) is not None]
    lows = [value for row in rows if (value := safe_float(row.get("low"))) is not None]
    volumes = [value for row in rows if (value := safe_float(row.get("volume"))) is not None]
    return _Snapshot(
        time=last.get("time") or first.get("time") or getattr(event, "time", None),
        open=safe_float(first.get("open")),
        high=max(highs) if highs else safe_float(last.get("high")),
        low=min(lows) if lows else safe_float(last.get("low")),
        close=safe_float(last.get("close")),
        volume=float(sum(max(0.0, value) for value in volumes)) if volumes else None,
    )


def _market_snapshot(event: Any) -> _Snapshot | None:
    close = safe_float(getattr(event, "close", None))
    if close is None:
        return None
    return _Snapshot(
        time=getattr(event, "time", None),
        open=safe_float(getattr(event, "open", None)),
        high=safe_float(getattr(event, "high", None)),
        low=safe_float(getattr(event, "low", None)),
        close=close,
        volume=safe_float(getattr(event, "volume", None)),
    )


def _event_symbols(event: Any, configured: list[str]) -> list[str]:
    if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
        bars_1s = dict(getattr(event, "bars_1s", {}) or {})
        return [symbol for symbol in configured if symbol in bars_1s]
    symbol = getattr(event, "symbol", None)
    return [str(symbol)] if symbol in configured else []


def _event_datetime_utc(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.astimezone(UTC) if raw.tzinfo is not None else raw.replace(tzinfo=UTC)
    if isinstance(raw, (int, float)):
        ts = float(raw)
        if abs(ts) > 100_000_000_000:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=UTC)
        except Exception:
            return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


# The vol-target annualization MATH lives once in ``indicators/annualization``
# (canonical home, epoch-float inputs).  The wrappers below are a thin RAW-
# timestamp adapter: they parse each raw event time (datetime / ISO string /
# epoch) into epoch-seconds via ``_event_datetime_utc`` and then delegate, so
# sleeves can keep passing whatever timestamp form their events carry while the
# median-spacing / sqrt-annualize logic has a single source of truth.


def _times_to_epochs(times: Any) -> list[float]:
    """Parse raw timestamp-like values into epoch-seconds (unparseable dropped)."""
    epochs: list[float] = []
    for raw in list(times or []):
        dt = _event_datetime_utc(raw)
        if dt is not None:
            epochs.append(dt.timestamp())
    return epochs


def _median_bar_spacing_seconds(times: Any) -> float | None:
    """Median positive gap (seconds) between consecutive parsed timestamps.

    ``times`` is any iterable of raw timestamp-like values (datetimes, epochs,
    ISO strings); unparseable entries are dropped, then the canonical
    ``median_bar_spacing_seconds`` computes the median positive spacing (``None``
    when fewer than two usable timestamps or no positive spacing exists).
    """
    return _canon_median_bar_spacing_seconds(_times_to_epochs(times))


def _bars_per_year_from_spacing(times: Any) -> float | None:
    """Deterministic bars-per-year inferred from the median observed bar spacing.

    Returns ``None`` when the spacing cannot be inferred (fewer than two usable
    timestamps).  Callers must then pass through WITHOUT annualization rather
    than guess a horizon.
    """
    return _canon_bars_per_year_from_spacing(_times_to_epochs(times))


def _annualize_per_bar_vol(per_bar_vol: float, times: Any) -> float | None:
    """Annualize a per-bar vol via ``sqrt(bars_per_year)`` from observed spacing.

    This is the horizon bridge for vol-target sizing: a per-bar realized-vol
    estimate (e.g. ~0.03 on a 1d bar) must be scaled to the same annual horizon
    as an annual-scale ``target_vol`` (e.g. 0.20) before the Moreira-Muir
    ``min(1, target_vol / vol)`` clamp; comparing the two directly leaves the
    throttle INERT.  Returns ``None`` when spacing is unavailable -- the caller
    must then leave sizing a pass-through rather than compare mismatched
    horizons.
    """
    return _canon_annualize_per_bar_vol(per_bar_vol, _bars_per_year_from_spacing(times))


def _session_key(raw_time: Any, *, start_minute_utc: int) -> str:
    dt = _event_datetime_utc(raw_time)
    if dt is None:
        return ""
    session_start = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        minutes=max(0, min(1439, int(start_minute_utc)))
    )
    session_day = session_start.date()
    if dt < session_start:
        session_day = (session_start - timedelta(days=1)).date()
    return str(session_day)


def _target_metadata(
    *,
    strategy: str,
    target_allocation: float,
    max_order_value: float,
    **extra: Any,
) -> dict[str, Any]:
    metadata = {"strategy": strategy, **extra}
    if target_allocation > 0.0:
        metadata["target_allocation"] = float(target_allocation)
        metadata["max_symbol_exposure_pct"] = float(target_allocation)
    if max_order_value > 0.0:
        metadata["max_order_value"] = float(max_order_value)
    return metadata


def _emit(
    events: Any,
    *,
    strategy_id: str,
    symbol: str,
    event_time: Any,
    signal_type: str,
    strength: float = 1.0,
    price: float | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    trailing_percent: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    events.put(
        SignalEvent(
            strategy_id=strategy_id,
            symbol=symbol,
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


def _extract_feature(bars: Any, event: Any, symbol: str, field: str) -> float | None:
    """Resolve a feature value via the 3-tier cascade.

    Tries the event attribute, then ``bars.get_latest_feature_value``, then
    ``bars.get_latest_bar_value``; returns ``None`` (skip, never raise) when no
    source provides a finite value.
    """
    direct = safe_float(getattr(event, field, None))
    if direct is not None:
        return direct
    getter = getattr(bars, "get_latest_feature_value", None)
    if callable(getter):
        try:
            parsed = safe_float(getter(symbol, field))
        except Exception:
            parsed = None
        if parsed is not None:
            return parsed
    getter = getattr(bars, "get_latest_bar_value", None)
    if callable(getter):
        try:
            return safe_float(getter(symbol, field))
        except Exception:
            return None
    return None


@register("strategy", "LiquidityShockReversionStrategy", interface="event_driven")
class LiquidityShockReversionStrategy(Strategy):
    """Fade high-volume, high-range shocks after the shock bar closes."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "volume_window": HyperParam.integer("volume_window", default=96, low=8, high=4096),
            "range_window": HyperParam.integer("range_window", default=96, low=8, high=4096),
            "volume_shock_z": HyperParam.floating(
                "volume_shock_z", default=2.0, low=0.0, high=20.0
            ),
            "range_shock_z": HyperParam.floating("range_shock_z", default=1.5, low=0.0, high=20.0),
            "return_shock_pct": HyperParam.floating(
                "return_shock_pct", default=0.012, low=0.0, high=0.50
            ),
            "revert_fraction": HyperParam.floating(
                "revert_fraction", default=0.50, low=0.05, high=1.0
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=96, low=1, high=100000),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.018, low=0.0, high=0.50
            ),
            "take_profit_pct": HyperParam.floating(
                "take_profit_pct", default=0.0, low=0.0, high=1.0
            ),
            "trailing_exit_pct": HyperParam.floating(
                "trailing_exit_pct", default=0.012, low=0.0, high=0.50
            ),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.015, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=300.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "evaluation_cadence_bars": HyperParam.integer(
                "evaluation_cadence_bars", default=1, low=1, high=10_080, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.volume_window = max(3, int(resolved["volume_window"]))
        self.range_window = max(3, int(resolved["range_window"]))
        self.volume_shock_z = max(0.0, float(resolved["volume_shock_z"]))
        self.range_shock_z = max(0.0, float(resolved["range_shock_z"]))
        self.return_shock_pct = max(0.0, float(resolved["return_shock_pct"]))
        self.revert_fraction = max(0.01, min(1.0, float(resolved["revert_fraction"])))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.take_profit_pct = max(0.0, float(resolved["take_profit_pct"]))
        self.trailing_exit_pct = max(0.0, float(resolved["trailing_exit_pct"]))
        self.allow_short = bool(resolved["allow_short"])
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.evaluation_cadence_bars = max(1, int(resolved["evaluation_cadence_bars"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = max(self.volume_window, self.range_window, self.max_hold_bars) + 8
        self._state = {
            symbol: _SingleAssetState(
                opens=deque(maxlen=size),
                highs=deque(maxlen=size),
                lows=deque(maxlen=size),
                closes=deque(maxlen=size),
                volumes=deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "symbol_state": {symbol: self._pack_state(item) for symbol, item in self._state.items()}
        }

    @staticmethod
    def _pack_state(item: _SingleAssetState) -> dict[str, Any]:
        return {
            "opens": list(item.opens),
            "highs": list(item.highs),
            "lows": list(item.lows),
            "closes": list(item.closes),
            "volumes": list(item.volumes),
            "mode": item.mode,
            "entry_price": item.entry_price,
            "target_price": item.target_price,
            "high_watermark": item.high_watermark,
            "low_watermark": item.low_watermark,
            "bars_held": int(item.bars_held),
            "ticks_seen": int(item.ticks_seen),
            "last_time_key": item.last_time_key,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        raw = state.get("symbol_state") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            for attr in ("opens", "highs", "lows", "closes", "volumes"):
                target = getattr(item, attr)
                target.clear()
                for value in list(payload.get(attr) or [])[-int(target.maxlen or 0) :]:
                    parsed = safe_float(value)
                    if parsed is not None:
                        target.append(parsed)
            mode = str(payload.get("mode", "OUT")).upper()
            item.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
            item.entry_price = safe_float(payload.get("entry_price"))
            item.target_price = safe_float(payload.get("target_price"))
            item.high_watermark = safe_float(payload.get("high_watermark"))
            item.low_watermark = safe_float(payload.get("low_watermark"))
            item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
            item.ticks_seen = _safe_non_negative_int(payload.get("ticks_seen"))
            item.last_time_key = str(payload.get("last_time_key", ""))

    def _snapshot_for(self, event: Any, symbol: str) -> _Snapshot | None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            return _window_snapshot(event, symbol)
        return _market_snapshot(event)

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = self._snapshot_for(event, symbol)
            if snapshot is not None:
                self._process_symbol(symbol, snapshot)

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = getattr(event, "symbol", None)
        if symbol in self._state:
            snapshot = self._snapshot_for(event, str(symbol))
            if snapshot is not None:
                self._process_symbol(str(symbol), snapshot)

    def _append_snapshot(self, item: _SingleAssetState, snapshot: _Snapshot) -> None:
        item.opens.append(float(snapshot.open if snapshot.open is not None else snapshot.close))
        item.highs.append(float(snapshot.high if snapshot.high is not None else snapshot.close))
        item.lows.append(float(snapshot.low if snapshot.low is not None else snapshot.close))
        item.closes.append(float(snapshot.close))
        item.volumes.append(max(0.0, float(snapshot.volume or 0.0)))

    def _maybe_exit(self, symbol: str, item: _SingleAssetState, snapshot: _Snapshot) -> bool:
        close = float(snapshot.close or 0.0)
        if item.mode not in {"LONG", "SHORT"} or item.entry_price is None or close <= 0.0:
            return False
        item.bars_held += 1
        reason = ""
        if item.mode == "LONG":
            item.high_watermark = max(float(item.high_watermark or close), close)
            if self.stop_loss_pct > 0.0 and close <= item.entry_price * (1.0 - self.stop_loss_pct):
                reason = "stop_loss"
            elif item.target_price is not None and close >= item.target_price:
                reason = "reversion_target"
            elif self.take_profit_pct > 0.0 and close >= item.entry_price * (
                1.0 + self.take_profit_pct
            ):
                reason = "take_profit"
            elif (
                self.trailing_exit_pct > 0.0
                and item.high_watermark is not None
                and close <= item.high_watermark * (1.0 - self.trailing_exit_pct)
            ):
                reason = "trailing_exit"
        else:
            item.low_watermark = min(float(item.low_watermark or close), close)
            if self.stop_loss_pct > 0.0 and close >= item.entry_price * (1.0 + self.stop_loss_pct):
                reason = "stop_loss"
            elif item.target_price is not None and close <= item.target_price:
                reason = "reversion_target"
            elif self.take_profit_pct > 0.0 and close <= item.entry_price * (
                1.0 - self.take_profit_pct
            ):
                reason = "take_profit"
            elif (
                self.trailing_exit_pct > 0.0
                and item.low_watermark is not None
                and close >= item.low_watermark * (1.0 + self.trailing_exit_pct)
            ):
                reason = "trailing_exit"
        if not reason and item.bars_held >= self.max_hold_bars:
            reason = "max_hold"
        if not reason:
            return False
        _emit(
            self.events,
            strategy_id="liquidity_shock_reversion",
            symbol=symbol,
            event_time=snapshot.time,
            signal_type="EXIT",
            price=close,
            metadata={"strategy": "LiquidityShockReversionStrategy", "reason": reason},
        )
        item.mode = "OUT"
        item.entry_price = None
        item.target_price = None
        item.high_watermark = None
        item.low_watermark = None
        item.bars_held = 0
        return True

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return
        item.last_time_key = key
        if snapshot.close is None or snapshot.close <= self.min_price:
            return
        self._append_snapshot(item, snapshot)
        item.ticks_seen += 1
        self._maybe_exit(symbol, item, snapshot)
        if item.ticks_seen % self.evaluation_cadence_bars:
            return
        if item.mode != "OUT" or len(item.closes) < max(self.volume_window, self.range_window) + 1:
            return
        close = float(item.closes[-1])
        prev_close = float(item.closes[-2])
        if prev_close <= 0.0:
            return
        ret = close / prev_close - 1.0
        vol_z = volume_zscore(item.volumes, window=self.volume_window)
        rng_z = range_zscore(item.highs, item.lows, item.closes, window=self.range_window)
        if vol_z is None or rng_z is None:
            return
        if (
            abs(ret) < self.return_shock_pct
            or vol_z < self.volume_shock_z
            or rng_z < self.range_shock_z
        ):
            return
        if ret < 0.0:
            signal_type = "LONG"
            target = close + (prev_close - close) * self.revert_fraction
            stop_loss = close * (1.0 - self.stop_loss_pct) if self.stop_loss_pct > 0.0 else None
        elif self.allow_short:
            signal_type = "SHORT"
            target = close - (close - prev_close) * self.revert_fraction
            stop_loss = close * (1.0 + self.stop_loss_pct) if self.stop_loss_pct > 0.0 else None
        else:
            return
        strength = min(3.0, max(0.25, abs(ret) / max(self.return_shock_pct, _EPS)))
        metadata = _target_metadata(
            strategy="LiquidityShockReversionStrategy",
            target_allocation=self.target_allocation,
            max_order_value=self.max_order_value,
            reason="downside_liquidity_shock"
            if signal_type == "LONG"
            else "upside_liquidity_shock",
            shock_return=float(ret),
            volume_z=float(vol_z),
            range_z=float(rng_z),
            reversion_target=float(target),
            revert_fraction=float(self.revert_fraction),
        )
        _emit(
            self.events,
            strategy_id="liquidity_shock_reversion",
            symbol=symbol,
            event_time=snapshot.time,
            signal_type=signal_type,
            strength=strength,
            price=close,
            stop_loss=stop_loss,
            metadata=metadata,
            trailing_percent=self.trailing_exit_pct if self.trailing_exit_pct > 0.0 else None,
        )
        item.mode = signal_type
        item.entry_price = close
        item.target_price = target
        item.high_watermark = close
        item.low_watermark = close
        item.bars_held = 0


@register("strategy", "VolManagedMomentumCrashGateStrategy", interface="event_driven")
class VolManagedMomentumCrashGateStrategy(Strategy):
    """Cross-sectional time-series momentum with volatility targeting and crash gate."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "momentum_lookback_bars": HyperParam.integer(
                "momentum_lookback_bars", default=168, low=4, high=10080
            ),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=24, low=1, high=10080),
            "vol_window": HyperParam.integer("vol_window", default=96, low=4, high=4096),
            "target_vol": HyperParam.floating("target_vol", default=0.003, low=0.0, high=1.0),
            "max_leverage": HyperParam.floating("max_leverage", default=1.2, low=0.0, high=10.0),
            "signal_threshold": HyperParam.floating(
                "signal_threshold", default=0.35, low=0.0, high=20.0
            ),
            "max_longs": HyperParam.integer("max_longs", default=3, low=0, high=128),
            "max_shorts": HyperParam.integer("max_shorts", default=3, low=0, high=128),
            "crash_window_bars": HyperParam.integer(
                "crash_window_bars", default=48, low=2, high=4096
            ),
            "crash_return_pct": HyperParam.floating(
                "crash_return_pct", default=-0.08, low=-1.0, high=0.0
            ),
            "vol_ratio_window": HyperParam.integer(
                "vol_ratio_window", default=192, low=16, high=8192
            ),
            "vol_ratio_max": HyperParam.floating("vol_ratio_max", default=2.5, low=0.0, high=20.0),
            "stress_reduce": HyperParam.floating("stress_reduce", default=0.35, low=0.0, high=1.0),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=0.45, low=0.0, high=5.0, tunable=False
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.045, low=0.0, high=0.50
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=1440, low=1, high=200000),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=1000.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "btc_symbol": HyperParam.string("btc_symbol", default="BTC/USDT", tunable=False),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        if not self.symbol_list:
            raise ValueError("VolManagedMomentumCrashGateStrategy requires at least one symbol.")
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.momentum_lookback_bars = max(1, int(resolved["momentum_lookback_bars"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.target_vol = max(0.0, float(resolved["target_vol"]))
        self.max_leverage = max(0.0, float(resolved["max_leverage"]))
        self.signal_threshold = max(0.0, float(resolved["signal_threshold"]))
        self.max_longs = max(0, int(resolved["max_longs"]))
        self.max_shorts = max(0, int(resolved["max_shorts"]))
        self.crash_window_bars = max(2, int(resolved["crash_window_bars"]))
        self.crash_return_pct = min(0.0, float(resolved["crash_return_pct"]))
        self.vol_ratio_window = max(16, int(resolved["vol_ratio_window"]))
        self.vol_ratio_max = max(0.0, float(resolved["vol_ratio_max"]))
        self.stress_reduce = max(0.0, min(1.0, float(resolved["stress_reduce"])))
        self.allow_short = bool(resolved["allow_short"])
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        default_btc = "BTC/USDT" if "BTC/USDT" in self.symbol_list else self.symbol_list[0]
        self.btc_symbol = str(resolved["btc_symbol"] or default_btc)
        if self.btc_symbol not in self.symbol_list:
            self.btc_symbol = default_btc
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = (
            max(
                self.momentum_lookback_bars,
                self.vol_window,
                self.crash_window_bars,
                self.vol_ratio_window,
                self.max_hold_bars,
            )
            + 8
        )
        self._state = {
            symbol: _RotationState(closes=deque(maxlen=size)) for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0

    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
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
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            item.closes.clear()
            for value in list(payload.get("closes") or [])[-int(item.closes.maxlen or 0) :]:
                parsed = safe_float(value)
                if parsed is not None and parsed > 0.0:
                    item.closes.append(parsed)
            mode = str(payload.get("mode", "OUT")).upper()
            item.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
            item.entry_price = safe_float(payload.get("entry_price"))
            item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
            item.last_time_key = str(payload.get("last_time_key", ""))

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        event_key = time_key(getattr(event, "time", None))
        updated: list[str] = []
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is None or snapshot.close is None or snapshot.close <= self.min_price:
                continue
            item = self._state[symbol]
            key = time_key(snapshot.time)
            if key and key == item.last_time_key:
                continue
            item.last_time_key = key
            item.closes.append(float(snapshot.close))
            updated.append(symbol)
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
        if symbol not in self._state:
            return
        snapshot = _market_snapshot(event)
        if snapshot is None or snapshot.close is None or snapshot.close <= self.min_price:
            return
        item = self._state[str(symbol)]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return
        item.last_time_key = key
        item.closes.append(float(snapshot.close))
        if key and key != self._last_eval_time_key:
            self._last_eval_time_key = key
            self._tick += 1
            self._rebalance(snapshot.time)

    def _score_symbol(self, symbol: str) -> tuple[float, float, float] | None:
        closes = self._state[symbol].closes
        if len(closes) <= max(self.momentum_lookback_bars, self.vol_window):
            return None
        mom = log_return(closes, lookback=self.momentum_lookback_bars)
        vol = realized_volatility(closes, window=self.vol_window)
        eff = trend_efficiency(closes, window=min(self.momentum_lookback_bars, self.vol_window))
        if mom is None or vol is None or vol <= _EPS:
            return None
        score = float(mom / vol) * max(0.25, float(eff if eff is not None else 0.5))
        leverage = self.max_leverage
        if self.target_vol > 0.0:
            leverage = min(self.max_leverage, self.target_vol / max(vol, _EPS))
        return float(score), float(vol), float(leverage)

    def _crash_gate(self) -> tuple[bool, float, dict[str, Any]]:
        btc = self._state.get(self.btc_symbol)
        if btc is None:
            return False, 1.0, {"benchmark_symbol": self.btc_symbol, "reason": "missing_benchmark"}
        drawdown = drawdown_from_peak(btc.closes, window=self.crash_window_bars)
        vr = volatility_ratio(
            btc.closes,
            fast_window=max(2, self.vol_window // 2),
            slow_window=self.vol_ratio_window,
        )
        ret = simple_return(btc.closes, lookback=self.crash_window_bars)
        crash = False
        if ret is not None and ret <= self.crash_return_pct:
            crash = True
        if drawdown is not None and drawdown <= self.crash_return_pct:
            crash = True
        if vr is not None and self.vol_ratio_max > 0.0 and vr >= self.vol_ratio_max:
            crash = True
        multiplier = self.stress_reduce if crash else 1.0
        return (
            crash,
            multiplier,
            {
                "benchmark_symbol": self.btc_symbol,
                "benchmark_return": ret,
                "benchmark_drawdown": drawdown,
                "benchmark_vol_ratio": vr,
                "stress_multiplier": multiplier,
            },
        )

    def _rebalance(self, event_time: Any) -> None:
        if self._tick % self.rebalance_bars:
            self._age_and_stop(event_time)
            return
        rows: list[tuple[float, str, float, float]] = []
        for symbol in self.symbol_list:
            scored = self._score_symbol(symbol)
            if scored is None:
                continue
            score, vol, leverage = scored
            rows.append((score, symbol, vol, leverage))
        if not rows:
            self._age_and_stop(event_time)
            return
        rows.sort(key=lambda row: row[0])
        crash, stress_multiplier, crash_meta = self._crash_gate()
        long_rows = [row for row in reversed(rows) if row[0] >= self.signal_threshold]
        short_rows = [row for row in rows if row[0] <= -self.signal_threshold]
        if crash:
            long_rows = []
        if not self.allow_short:
            short_rows = []
        long_rows = long_rows[: self.max_longs]
        short_rows = short_rows[: self.max_shorts]
        targets: dict[str, tuple[str, float, float, float]] = {}
        selected_count = len(long_rows) + len(short_rows)
        base_alloc = 0.0
        if selected_count > 0:
            base_alloc = self.target_gross_exposure * stress_multiplier / float(selected_count)
        for score, symbol, vol, leverage in long_rows:
            targets[symbol] = ("LONG", score, vol, min(base_alloc * leverage, self.max_leverage))
        for score, symbol, vol, leverage in short_rows:
            targets[symbol] = ("SHORT", score, vol, min(base_alloc * leverage, self.max_leverage))
        self._apply_targets(event_time, targets, crash_meta)

    def _age_and_stop(self, event_time: Any) -> None:
        for symbol, item in self._state.items():
            if item.mode == "OUT":
                continue
            item.bars_held += 1
            close = item.closes[-1] if item.closes else None
            stop = False
            if close is not None and item.entry_price is not None and self.stop_loss_pct > 0.0:
                if item.mode == "LONG" and close <= item.entry_price * (1.0 - self.stop_loss_pct):
                    stop = True
                if item.mode == "SHORT" and close >= item.entry_price * (1.0 + self.stop_loss_pct):
                    stop = True
            if stop or item.bars_held >= self.max_hold_bars:
                _emit(
                    self.events,
                    strategy_id="vol_managed_momentum_crash_gate",
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=close,
                    metadata={
                        "strategy": "VolManagedMomentumCrashGateStrategy",
                        "reason": "stop_loss" if stop else "max_hold",
                    },
                )
                item.mode = "OUT"
                item.entry_price = None
                item.bars_held = 0

    def _apply_targets(
        self,
        event_time: Any,
        targets: dict[str, tuple[str, float, float, float]],
        crash_meta: dict[str, Any],
    ) -> None:
        for symbol, item in self._state.items():
            target = targets.get(symbol)
            close = item.closes[-1] if item.closes else None
            if target is None:
                if item.mode != "OUT":
                    _emit(
                        self.events,
                        strategy_id="vol_managed_momentum_crash_gate",
                        symbol=symbol,
                        event_time=event_time,
                        signal_type="EXIT",
                        price=close,
                        metadata={
                            "strategy": "VolManagedMomentumCrashGateStrategy",
                            "reason": "rebalance_removed",
                            **crash_meta,
                        },
                    )
                    item.mode = "OUT"
                    item.entry_price = None
                    item.bars_held = 0
                continue
            target_mode, score, vol, alloc = target
            if item.mode == target_mode:
                item.bars_held += 1
                continue
            if item.mode != "OUT":
                _emit(
                    self.events,
                    strategy_id="vol_managed_momentum_crash_gate",
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=close,
                    metadata={
                        "strategy": "VolManagedMomentumCrashGateStrategy",
                        "reason": "side_flip",
                        **crash_meta,
                    },
                )
            stop_loss = None
            if close is not None and self.stop_loss_pct > 0.0:
                stop_loss = close * (
                    1.0 - self.stop_loss_pct if target_mode == "LONG" else 1.0 + self.stop_loss_pct
                )
            metadata = _target_metadata(
                strategy="VolManagedMomentumCrashGateStrategy",
                target_allocation=max(0.0, float(alloc)),
                max_order_value=self.max_order_value,
                score=float(score),
                realized_vol=float(vol),
                target_mode=target_mode,
                **crash_meta,
            )
            _emit(
                self.events,
                strategy_id="vol_managed_momentum_crash_gate",
                symbol=symbol,
                event_time=event_time,
                signal_type=target_mode,
                strength=max(0.1, min(3.0, abs(score))),
                price=close,
                stop_loss=stop_loss,
                metadata=metadata,
            )
            item.mode = target_mode
            item.entry_price = close
            item.bars_held = 0


@register("strategy", "FundingDislocationTrendCarryStrategy", interface="event_driven")
class FundingDislocationTrendCarryStrategy(Strategy):
    """Perp funding/basis carry blended with multi-horizon trend rotation."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False
    required_features = (
        "funding_rate",
        "mark_price",
        "index_price",
        "open_interest",
        "taker_buy_quote_volume",
        "taker_sell_quote_volume",
    )

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "fast_lookback_bars": HyperParam.integer(
                "fast_lookback_bars", default=24, low=2, high=4096
            ),
            "mid_lookback_bars": HyperParam.integer(
                "mid_lookback_bars", default=96, low=4, high=8192
            ),
            "slow_lookback_bars": HyperParam.integer(
                "slow_lookback_bars", default=336, low=8, high=20000
            ),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=24, low=1, high=10080),
            "signal_threshold": HyperParam.floating(
                "signal_threshold", default=0.45, low=0.0, high=20.0
            ),
            "max_longs": HyperParam.integer("max_longs", default=3, low=0, high=128),
            "max_shorts": HyperParam.integer("max_shorts", default=3, low=0, high=128),
            "vol_window": HyperParam.integer("vol_window", default=96, low=4, high=4096),
            "crowding_window": HyperParam.integer("crowding_window", default=96, low=4, high=4096),
            "trend_weight": HyperParam.floating("trend_weight", default=0.55, low=0.0, high=5.0),
            "carry_weight": HyperParam.floating("carry_weight", default=0.30, low=0.0, high=5.0),
            "basis_weight": HyperParam.floating("basis_weight", default=0.20, low=0.0, high=5.0),
            "crowding_penalty_weight": HyperParam.floating(
                "crowding_penalty_weight", default=0.20, low=0.0, high=5.0
            ),
            "funding_scale": HyperParam.floating(
                "funding_scale", default=0.0006, low=0.000001, high=0.10
            ),
            "basis_scale_bps": HyperParam.floating(
                "basis_scale_bps", default=35.0, low=0.1, high=10000.0
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.035, low=0.0, high=0.50
            ),
            "max_abs_exposure": HyperParam.floating(
                "max_abs_exposure", default=0.08, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=750.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=1440, low=1, high=200000),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        if not self.symbol_list:
            raise ValueError("FundingDislocationTrendCarryStrategy requires at least one symbol.")
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.fast_lookback_bars = max(1, int(resolved["fast_lookback_bars"]))
        self.mid_lookback_bars = max(1, int(resolved["mid_lookback_bars"]))
        self.slow_lookback_bars = max(1, int(resolved["slow_lookback_bars"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.signal_threshold = max(0.0, float(resolved["signal_threshold"]))
        self.max_longs = max(0, int(resolved["max_longs"]))
        self.max_shorts = max(0, int(resolved["max_shorts"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.crowding_window = max(2, int(resolved["crowding_window"]))
        self.trend_weight = float(resolved["trend_weight"])
        self.carry_weight = float(resolved["carry_weight"])
        self.basis_weight = float(resolved["basis_weight"])
        self.crowding_penalty_weight = float(resolved["crowding_penalty_weight"])
        self.funding_scale = max(_EPS, float(resolved["funding_scale"]))
        self.basis_scale_bps = max(_EPS, float(resolved["basis_scale_bps"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_abs_exposure = max(0.0, float(resolved["max_abs_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.allow_short = bool(resolved["allow_short"])
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = (
            max(
                self.fast_lookback_bars,
                self.mid_lookback_bars,
                self.slow_lookback_bars,
                self.vol_window,
                self.crowding_window,
                self.max_hold_bars,
            )
            + 8
        )
        self._state = {
            symbol: _CarryState(
                closes=deque(maxlen=size),
                funding_rate=deque(maxlen=size),
                basis_bps=deque(maxlen=size),
                open_interest=deque(maxlen=size),
                taker_buy_quote_volume=deque(maxlen=size),
                taker_sell_quote_volume=deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0

    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "funding_rate": list(item.funding_rate),
                    "basis_bps": list(item.basis_bps),
                    "open_interest": list(item.open_interest),
                    "taker_buy_quote_volume": list(item.taker_buy_quote_volume),
                    "taker_sell_quote_volume": list(item.taker_sell_quote_volume),
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
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            for attr in (
                "closes",
                "funding_rate",
                "basis_bps",
                "open_interest",
                "taker_buy_quote_volume",
                "taker_sell_quote_volume",
            ):
                target = getattr(item, attr)
                target.clear()
                for value in list(payload.get(attr) or [])[-int(target.maxlen or 0) :]:
                    parsed = safe_float(value)
                    if parsed is not None:
                        target.append(parsed)
            mode = str(payload.get("mode", "OUT")).upper()
            item.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
            item.entry_price = safe_float(payload.get("entry_price"))
            item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
            item.last_time_key = str(payload.get("last_time_key", ""))

    def _extract_feature(self, event: Any, symbol: str, field: str) -> float | None:
        return _extract_feature(self.bars, event, symbol, field)

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        event_key = time_key(getattr(event, "time", None))
        updated = False
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is None or snapshot.close is None or snapshot.close <= self.min_price:
                continue
            if self._update_symbol(event, symbol, snapshot):
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
        if symbol not in self._state:
            return
        snapshot = _market_snapshot(event)
        if snapshot is None or snapshot.close is None or snapshot.close <= self.min_price:
            return
        if self._update_symbol(event, str(symbol), snapshot):
            key = time_key(snapshot.time)
            if key and key != self._last_eval_time_key:
                self._last_eval_time_key = key
                self._tick += 1
                self._rebalance(snapshot.time)

    def _update_symbol(self, event: Any, symbol: str, snapshot: _Snapshot) -> bool:
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return False
        item.last_time_key = key
        funding = self._extract_feature(event, symbol, "funding_rate")
        mark = self._extract_feature(event, symbol, "mark_price")
        index = self._extract_feature(event, symbol, "index_price")
        oi = self._extract_feature(event, symbol, "open_interest")
        buy = self._extract_feature(event, symbol, "taker_buy_quote_volume")
        sell = self._extract_feature(event, symbol, "taker_sell_quote_volume")
        basis = basis_bps(mark, index)
        if funding is None or basis is None or oi is None or buy is None or sell is None:
            return False
        item.closes.append(float(snapshot.close))
        item.funding_rate.append(float(funding))
        item.basis_bps.append(float(basis))
        item.open_interest.append(float(oi))
        item.taker_buy_quote_volume.append(max(0.0, float(buy)))
        item.taker_sell_quote_volume.append(max(0.0, float(sell)))
        return True

    def _score_symbol(self, symbol: str) -> tuple[float, dict[str, Any]] | None:
        item = self._state[symbol]
        need = max(self.slow_lookback_bars, self.vol_window, self.crowding_window)
        if len(item.closes) <= need:
            return None
        vol = realized_volatility(item.closes, window=self.vol_window)
        if vol is None or vol <= _EPS:
            return None
        legs = []
        for lookback, weight in (
            (self.fast_lookback_bars, 0.25),
            (self.mid_lookback_bars, 0.45),
            (self.slow_lookback_bars, 0.30),
        ):
            ret = log_return(item.closes, lookback=lookback)
            if ret is not None:
                legs.append((float(ret / vol), weight, lookback))
        if not legs:
            return None
        trend = sum(value * weight for value, weight, _lookback in legs) / sum(
            weight for _value, weight, _lookback in legs
        )
        avg_funding = sum(list(item.funding_rate)[-self.crowding_window :]) / float(
            min(len(item.funding_rate), self.crowding_window)
        )
        avg_basis = sum(list(item.basis_bps)[-self.crowding_window :]) / float(
            min(len(item.basis_bps), self.crowding_window)
        )
        carry_component = -clipped_tanh_score(avg_funding, scale=self.funding_scale)
        basis_component = -clipped_tanh_score(avg_basis, scale=self.basis_scale_bps)
        flow = order_flow_imbalance(
            item.taker_buy_quote_volume,
            item.taker_sell_quote_volume,
            window=self.crowding_window,
        )
        oi_delta = simple_return(item.open_interest, lookback=max(2, self.crowding_window // 4))
        crowding = abs(float(flow or 0.0)) + max(0.0, float(oi_delta or 0.0))
        raw_score = (
            self.trend_weight * float(trend)
            + self.carry_weight * carry_component
            + self.basis_weight * basis_component
        )
        if abs(raw_score) > _EPS and crowding > 0.35:
            raw_score -= math.copysign(self.crowding_penalty_weight * (crowding - 0.35), raw_score)
        meta = {
            "trend_component": float(trend),
            "carry_component": float(carry_component),
            "basis_component": float(basis_component),
            "avg_funding_rate": float(avg_funding),
            "avg_basis_bps": float(avg_basis),
            "flow_imbalance": flow,
            "open_interest_delta": oi_delta,
            "crowding_penalty_input": float(crowding),
            "realized_vol": float(vol),
        }
        return float(raw_score), meta

    def _rebalance(self, event_time: Any) -> None:
        if self._tick % self.rebalance_bars:
            self._age_and_stop(event_time)
            return
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for symbol in self.symbol_list:
            scored = self._score_symbol(symbol)
            if scored is not None:
                score, meta = scored
                rows.append((score, symbol, meta))
        if not rows:
            self._age_and_stop(event_time)
            return
        rows.sort(key=lambda row: row[0])
        longs = [row for row in reversed(rows) if row[0] >= self.signal_threshold][: self.max_longs]
        shorts = [row for row in rows if row[0] <= -self.signal_threshold][: self.max_shorts]
        if not self.allow_short:
            shorts = []
        selected = len(longs) + len(shorts)
        base_alloc = self.max_abs_exposure / float(max(1, selected)) if selected else 0.0
        targets: dict[str, tuple[str, float, dict[str, Any]]] = {}
        for score, symbol, meta in longs:
            targets[symbol] = ("LONG", score, meta)
        for score, symbol, meta in shorts:
            targets[symbol] = ("SHORT", score, meta)
        self._apply_targets(event_time, targets, base_alloc)

    def _age_and_stop(self, event_time: Any) -> None:
        for symbol, item in self._state.items():
            if item.mode == "OUT":
                continue
            item.bars_held += 1
            close = item.closes[-1] if item.closes else None
            stop = False
            if close is not None and item.entry_price is not None and self.stop_loss_pct > 0.0:
                if item.mode == "LONG" and close <= item.entry_price * (1.0 - self.stop_loss_pct):
                    stop = True
                if item.mode == "SHORT" and close >= item.entry_price * (1.0 + self.stop_loss_pct):
                    stop = True
            if stop or item.bars_held >= self.max_hold_bars:
                _emit(
                    self.events,
                    strategy_id="funding_dislocation_trend_carry",
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=close,
                    metadata={
                        "strategy": "FundingDislocationTrendCarryStrategy",
                        "reason": "stop_loss" if stop else "max_hold",
                    },
                )
                item.mode = "OUT"
                item.entry_price = None
                item.bars_held = 0

    def _apply_targets(
        self,
        event_time: Any,
        targets: dict[str, tuple[str, float, dict[str, Any]]],
        base_alloc: float,
    ) -> None:
        for symbol, item in self._state.items():
            target = targets.get(symbol)
            close = item.closes[-1] if item.closes else None
            if target is None:
                if item.mode != "OUT":
                    _emit(
                        self.events,
                        strategy_id="funding_dislocation_trend_carry",
                        symbol=symbol,
                        event_time=event_time,
                        signal_type="EXIT",
                        price=close,
                        metadata={
                            "strategy": "FundingDislocationTrendCarryStrategy",
                            "reason": "rebalance_removed",
                        },
                    )
                    item.mode = "OUT"
                    item.entry_price = None
                    item.bars_held = 0
                continue
            target_mode, score, score_meta = target
            if item.mode == target_mode:
                item.bars_held += 1
                continue
            if item.mode != "OUT":
                _emit(
                    self.events,
                    strategy_id="funding_dislocation_trend_carry",
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=close,
                    metadata={
                        "strategy": "FundingDislocationTrendCarryStrategy",
                        "reason": "side_flip",
                    },
                )
            stop_loss = None
            if close is not None and self.stop_loss_pct > 0.0:
                stop_loss = close * (
                    1.0 - self.stop_loss_pct if target_mode == "LONG" else 1.0 + self.stop_loss_pct
                )
            alloc = min(self.max_abs_exposure, max(0.0, base_alloc * min(2.0, abs(score))))
            metadata = _target_metadata(
                strategy="FundingDislocationTrendCarryStrategy",
                target_allocation=alloc,
                max_order_value=self.max_order_value,
                score=float(score),
                target_mode=target_mode,
                **score_meta,
            )
            _emit(
                self.events,
                strategy_id="funding_dislocation_trend_carry",
                symbol=symbol,
                event_time=event_time,
                signal_type=target_mode,
                strength=max(0.1, min(3.0, abs(score))),
                price=close,
                stop_loss=stop_loss,
                metadata=metadata,
            )
            item.mode = target_mode
            item.entry_price = close
            item.bars_held = 0


@register("strategy", "OpeningRangeContinuationStrategy", interface="event_driven")
class OpeningRangeContinuationStrategy(Strategy):
    """Session opening-range momentum sleeve with volatility/volume confirmation."""

    decision_cadence_seconds = 60
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "session_start_minute_utc": HyperParam.integer(
                "session_start_minute_utc", default=0, low=0, high=1439, tunable=False
            ),
            "opening_range_bars": HyperParam.integer(
                "opening_range_bars", default=30, low=2, high=1440
            ),
            "entry_delay_bars": HyperParam.integer("entry_delay_bars", default=0, low=0, high=1440),
            "opening_return_threshold": HyperParam.floating(
                "opening_return_threshold", default=0.006, low=0.0, high=0.50
            ),
            "breakout_buffer_pct": HyperParam.floating(
                "breakout_buffer_pct", default=0.0, low=0.0, high=0.05
            ),
            "min_volume_z": HyperParam.floating("min_volume_z", default=0.0, low=-5.0, high=20.0),
            "volume_window": HyperParam.integer("volume_window", default=96, low=8, high=4096),
            "max_realized_vol": HyperParam.floating(
                "max_realized_vol", default=0.0, low=0.0, high=1.0
            ),
            "vol_window": HyperParam.integer("vol_window", default=96, low=4, high=4096),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=360, low=1, high=100000),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.025, low=0.0, high=0.50
            ),
            "take_profit_pct": HyperParam.floating(
                "take_profit_pct", default=0.060, low=0.0, high=1.0
            ),
            "trailing_exit_pct": HyperParam.floating(
                "trailing_exit_pct", default=0.025, low=0.0, high=0.50
            ),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.012, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=250.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.session_start_minute_utc = max(0, min(1439, int(resolved["session_start_minute_utc"])))
        self.opening_range_bars = max(2, int(resolved["opening_range_bars"]))
        self.entry_delay_bars = max(0, int(resolved["entry_delay_bars"]))
        self.opening_return_threshold = max(0.0, float(resolved["opening_return_threshold"]))
        self.breakout_buffer_pct = max(0.0, float(resolved["breakout_buffer_pct"]))
        self.min_volume_z = float(resolved["min_volume_z"])
        self.volume_window = max(3, int(resolved["volume_window"]))
        self.max_realized_vol = max(0.0, float(resolved["max_realized_vol"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.take_profit_pct = max(0.0, float(resolved["take_profit_pct"]))
        self.trailing_exit_pct = max(0.0, float(resolved["trailing_exit_pct"]))
        self.allow_short = bool(resolved["allow_short"])
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = (
            max(self.volume_window, self.vol_window, self.opening_range_bars, self.max_hold_bars)
            + 8
        )
        self._state = {
            symbol: _SingleAssetState(
                opens=deque(maxlen=size),
                highs=deque(maxlen=size),
                lows=deque(maxlen=size),
                closes=deque(maxlen=size),
                volumes=deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "symbol_state": {
                symbol: {
                    **LiquidityShockReversionStrategy._pack_state(item),
                    "last_signal_session": item.last_signal_session,
                    "session_key": item.session_key,
                    "session_open": item.session_open,
                    "session_high": item.session_high,
                    "session_low": item.session_low,
                    "session_bars": int(item.session_bars),
                }
                for symbol, item in self._state.items()
            }
        }

    def set_state(self, state: dict[str, Any]) -> None:
        raw = state.get("symbol_state") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            for attr in ("opens", "highs", "lows", "closes", "volumes"):
                target = getattr(item, attr)
                target.clear()
                for value in list(payload.get(attr) or [])[-int(target.maxlen or 0) :]:
                    parsed = safe_float(value)
                    if parsed is not None:
                        target.append(parsed)
            mode = str(payload.get("mode", "OUT")).upper()
            item.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
            item.entry_price = safe_float(payload.get("entry_price"))
            item.high_watermark = safe_float(payload.get("high_watermark"))
            item.low_watermark = safe_float(payload.get("low_watermark"))
            item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
            item.ticks_seen = _safe_non_negative_int(payload.get("ticks_seen"))
            item.last_time_key = str(payload.get("last_time_key", ""))
            item.last_signal_session = str(payload.get("last_signal_session", ""))
            item.session_key = str(payload.get("session_key", ""))
            item.session_open = safe_float(payload.get("session_open"))
            item.session_high = safe_float(payload.get("session_high"))
            item.session_low = safe_float(payload.get("session_low"))
            item.session_bars = _safe_non_negative_int(payload.get("session_bars"))

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None:
                self._process_symbol(symbol, snapshot)

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = getattr(event, "symbol", None)
        if symbol in self._state:
            snapshot = _market_snapshot(event)
            if snapshot is not None:
                self._process_symbol(str(symbol), snapshot)

    def _reset_session_if_needed(self, item: _SingleAssetState, snapshot: _Snapshot) -> bool:
        key = _session_key(snapshot.time, start_minute_utc=self.session_start_minute_utc)
        if not key:
            return False
        new_session = key != item.session_key
        high = snapshot.high if snapshot.high is not None else snapshot.close
        low = snapshot.low if snapshot.low is not None else snapshot.close
        if new_session:
            item.session_key = key
            item.session_open = snapshot.open if snapshot.open is not None else snapshot.close
            item.session_high = high
            item.session_low = low
            item.session_bars = 0
            item.last_signal_session = ""
        elif item.session_bars < self.opening_range_bars:
            if high is not None:
                item.session_high = max(float(item.session_high or high), float(high))
            if low is not None:
                item.session_low = min(float(item.session_low or low), float(low))
        item.session_bars += 1
        return new_session

    def _exit_position(
        self,
        symbol: str,
        item: _SingleAssetState,
        snapshot: _Snapshot,
        reason: str,
    ) -> None:
        _emit(
            self.events,
            strategy_id="opening_range_continuation",
            symbol=symbol,
            event_time=snapshot.time,
            signal_type="EXIT",
            price=safe_float(snapshot.close),
            metadata={"strategy": "OpeningRangeContinuationStrategy", "reason": reason},
        )
        item.mode = "OUT"
        item.entry_price = None
        item.high_watermark = None
        item.low_watermark = None
        item.bars_held = 0

    def _maybe_exit(self, symbol: str, item: _SingleAssetState, snapshot: _Snapshot) -> None:
        close = float(snapshot.close or 0.0)
        if item.mode not in {"LONG", "SHORT"} or item.entry_price is None or close <= 0.0:
            return
        item.bars_held += 1
        reason = ""
        if item.mode == "LONG":
            item.high_watermark = max(float(item.high_watermark or close), close)
            if self.stop_loss_pct > 0.0 and close <= item.entry_price * (1.0 - self.stop_loss_pct):
                reason = "stop_loss"
            elif self.take_profit_pct > 0.0 and close >= item.entry_price * (
                1.0 + self.take_profit_pct
            ):
                reason = "take_profit"
            elif (
                self.trailing_exit_pct > 0.0
                and item.high_watermark is not None
                and close <= item.high_watermark * (1.0 - self.trailing_exit_pct)
            ):
                reason = "trailing_exit"
        else:
            item.low_watermark = min(float(item.low_watermark or close), close)
            if self.stop_loss_pct > 0.0 and close >= item.entry_price * (1.0 + self.stop_loss_pct):
                reason = "stop_loss"
            elif self.take_profit_pct > 0.0 and close <= item.entry_price * (
                1.0 - self.take_profit_pct
            ):
                reason = "take_profit"
            elif (
                self.trailing_exit_pct > 0.0
                and item.low_watermark is not None
                and close >= item.low_watermark * (1.0 + self.trailing_exit_pct)
            ):
                reason = "trailing_exit"
        if not reason and item.bars_held >= self.max_hold_bars:
            reason = "max_hold"
        if not reason:
            return
        self._exit_position(symbol, item, snapshot, reason)

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return
        item.last_time_key = key
        if snapshot.close is None or snapshot.close <= self.min_price:
            return
        item.opens.append(float(snapshot.open if snapshot.open is not None else snapshot.close))
        item.highs.append(float(snapshot.high if snapshot.high is not None else snapshot.close))
        item.lows.append(float(snapshot.low if snapshot.low is not None else snapshot.close))
        item.closes.append(float(snapshot.close))
        item.volumes.append(max(0.0, float(snapshot.volume or 0.0)))
        new_session = self._reset_session_if_needed(item, snapshot)
        if new_session and item.mode != "OUT":
            self._exit_position(symbol, item, snapshot, "session_roll")
        self._maybe_exit(symbol, item, snapshot)
        if item.mode != "OUT" or item.last_signal_session == item.session_key:
            return
        trigger_bar = self.opening_range_bars + self.entry_delay_bars + 1
        if item.session_bars < trigger_bar or item.session_open is None or item.session_open <= 0.0:
            return
        if item.session_high is None or item.session_low is None:
            return
        opening_ret = float(snapshot.close / item.session_open - 1.0)
        if abs(opening_ret) < self.opening_return_threshold:
            return
        vol_z = volume_zscore(item.volumes, window=self.volume_window)
        if vol_z is not None and vol_z < self.min_volume_z:
            return
        rv = realized_volatility(item.closes, window=self.vol_window)
        if self.max_realized_vol > 0.0 and rv is not None and rv > self.max_realized_vol:
            return
        upper_break = float(item.session_high) * (1.0 + self.breakout_buffer_pct)
        lower_break = float(item.session_low) * (1.0 - self.breakout_buffer_pct)
        if opening_ret > 0.0 and float(snapshot.close) > upper_break:
            signal_type = "LONG"
            stop_loss = (
                snapshot.close * (1.0 - self.stop_loss_pct) if self.stop_loss_pct > 0.0 else None
            )
            take_profit = (
                snapshot.close * (1.0 + self.take_profit_pct)
                if self.take_profit_pct > 0.0
                else None
            )
        elif self.allow_short and float(snapshot.close) < lower_break:
            signal_type = "SHORT"
            stop_loss = (
                snapshot.close * (1.0 + self.stop_loss_pct) if self.stop_loss_pct > 0.0 else None
            )
            take_profit = (
                snapshot.close * (1.0 - self.take_profit_pct)
                if self.take_profit_pct > 0.0
                else None
            )
        else:
            return
        metadata = _target_metadata(
            strategy="OpeningRangeContinuationStrategy",
            target_allocation=self.target_allocation,
            max_order_value=self.max_order_value,
            session_key=item.session_key,
            opening_return=float(opening_ret),
            opening_range_bars=int(self.opening_range_bars),
            opening_range_high=float(item.session_high),
            opening_range_low=float(item.session_low),
            breakout_buffer_pct=float(self.breakout_buffer_pct),
            volume_z=vol_z,
            realized_vol=rv,
        )
        _emit(
            self.events,
            strategy_id="opening_range_continuation",
            symbol=symbol,
            event_time=snapshot.time,
            signal_type=signal_type,
            strength=min(
                3.0, max(0.25, abs(opening_ret) / max(self.opening_return_threshold, _EPS))
            ),
            price=snapshot.close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_percent=self.trailing_exit_pct if self.trailing_exit_pct > 0.0 else None,
            metadata=metadata,
        )
        item.mode = signal_type
        item.entry_price = float(snapshot.close)
        item.high_watermark = float(snapshot.close)
        item.low_watermark = float(snapshot.close)
        item.bars_held = 0
        item.last_signal_session = item.session_key


__all__ = [
    "FundingDislocationTrendCarryStrategy",
    "LiquidityShockReversionStrategy",
    "OpeningRangeContinuationStrategy",
    "VolManagedMomentumCrashGateStrategy",
]
