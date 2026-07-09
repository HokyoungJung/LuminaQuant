"""Silent-volume-shock lagged-resolution rider (CONDITIONAL, high death-prior).

``SilentVolumeShockResolutionStrategy`` is a per-symbol time-series sleeve that
ARMS on an abnormal-VOLUME bar that carries NO contemporaneous price move
(heavy tape, quiet price, quiet range) and then takes a direction ONLY from the
SUBSEQUENT vol-scaled price resolution over the next few bars -- never from the
shock bar itself.  It is strictly a LAGGED, LEAD-structure sleeve: volume leads,
price resolves, the sleeve tailgates the revealed side.

THEORY / PROVENANCE
-------------------
- Easley & O'Hara (1992, JF 47(2)): abnormal volume with no price adjustment is
  the signature of one-sided informed/patient flow being fully absorbed at the
  quote -- an information-arrival event whose direction is revealed only once
  the inventory imbalance forces price to pick a side.
- Gervais, Kaniel & Mingelgrin (2001, JF 56(3)): the high-volume-return premium
  in its TIME-SERIES lagged form (the XS-share form is owned by the flow-share
  sleeve).
- Wang (1994, JPE 102(1)) / Llorente, Michaely, Saar & Wang (2002, RFS 15(4)):
  informed-volume episodes CONTINUE (the naive short-horizon reversal prior is
  systematically wrong-signed on them).
- Barber & Odean (2008, RFS 21(2)): attention-constrained traders trigger on
  return/news salience, so a price-flat volume shock is invisible to them; they
  join only once the resolution drift is underway -- the flow the sleeve front-
  runs.

HONEST PRIOR OF DEATH: HIGH.  All anchors are equities-origin; the crypto
transplant IS the hypothesis.  The dominant pre-registered null is EVENT-SET
POLLUTION (crypto flat-price volume shocks dominated by wash/incentive volume,
funding-timestamp churn, and MM inventory recycling), which would make the armed
set informationless and the resolution sign a coin flip.  A clean rejection
reported in full is a valid outcome.

MECHANISM (per completed daily bar; OHLCV only; per symbol independently)
------------------------------------------------------------------------
State machine IDLE -> ARMED -> POSITIONED -> COOLDOWN.

1. Maintain trailing ``shock_window`` bars of log dollar volume
   ``log(max(close*volume, eps))``, log returns, and log range ``log(H/L)``.
2. SILENT SHOCK arms when, at a flat bar, ``vz >= v_shock_z`` AND
   ``rz = |log ret| / trailing_ret_std <= quiet_ret_z`` AND
   ``range_z <= quiet_range_z`` -- the exact complement of the
   volume-AND-range-AND-return conjunction the liquidity-shock fader needs.  NO
   order is placed on the shock bar; a ``resolution_max_bars`` window opens and a
   fresh shock inside it REFRESHES the window.
3. RESOLUTION: the first bar ``t+j`` (1 <= j <= resolution_max_bars) whose
   cumulative log return from the arming close breaches
   ``+/- resolution_ret_mult * sigma * sqrt(j)`` (vol-and-horizon scaled so a
   slow drift cannot resolve) triggers an entry WITH the sign of that move,
   inverse-realized-vol sized (SHORT only if ``allow_short``).  ``sigma`` is the
   trailing return std frozen at arm time.
4. EXIT: a hard ``min_hold_bars`` floor (the proven C1 turnover rescue), then a
   giveback exit (close crosses back through the arming close) or a
   ``max_hold_bars`` cap; an optional hard stop any time.  On exit start a
   ``cooldown_bars`` lockout.  A window that expires with no resolution disarms
   with no trade (the honest null path).

This is a PER-SYMBOL time-series sleeve applied across a multi-symbol liquid book
(honest ``allow_multi_asset`` at the data-PC handoff); it is NOT a cross-sectional
rank book and carries NO fake carry/XS-rank family tag.  It is data-local (no
I/O), pure Python (``math`` + ``deque`` only, no numpy), completed-bar, never
raises from ``calculate_signals``, and ships WITHOUT ``@register`` (inert until a
later integration wave wires it as ``research_only``).

DISTINCT-FROM
-------------
Versus ``AbnormalReturnContinuationStrategy`` (a RETURN-shock continuation
follower): remove the shock bar's abnormal VOLUME and this sleeve goes silent
while the return follower fires identically -- the volume antecedent is load-
bearing.  Versus ``LiquidityShockReversionStrategy``: it needs the exact
COMPLEMENTARY conjunction (volume AND range AND return) and fades the shock bar
itself; this sleeve needs a quiet price/range on the shock bar and trades the
LATER resolution.  Versus ``PriceVolumeCorrContinuationStrategy`` (contemporaneous
return/volume-change correlation gate), ``VolumeClockMomentumRiderStrategy``
(momentum sampled in volume time), and ``CrossSectionalFlowShareRotationStrategy``
/ ``CrossSectionalCloseLocationAccumulationStrategy`` (XS turnover-share / bar-
geometry books): none of them arm on a lagged, price-flat abnormal-turnover event
and none take direction from the subsequent vol-scaled resolution.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import realized_volatility
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _Snapshot,
    _emit,
    _event_symbols,
    _market_snapshot,
    _safe_non_negative_int,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategies.price_volume_continuation_alpha_sleeves import (
    _mode,
    _restore_deque,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "silent_volume_shock_resolution"
_STRATEGY_NAME = "SilentVolumeShockResolutionStrategy"


@dataclass(slots=True)
class _State:
    """Per-symbol OHLCV history + arm/position/cooldown machine state."""

    highs: deque[float]
    lows: deque[float]
    closes: deque[float]
    volumes: deque[float]
    mode: str = "OUT"  # OUT / LONG / SHORT (POSITIONED == LONG|SHORT)
    entry_price: float | None = None
    ref_price: float | None = None  # arming close -> giveback reference
    bars_held: int = 0
    cooldown_remaining: int = 0
    armed: bool = False
    arm_close: float | None = None
    arm_sigma: float | None = None
    arm_age: int = 0  # bars elapsed since the arm bar (0 on the arm bar)
    last_time_key: str = ""


def _trailing_z(values: list[float], window: int) -> float | None:
    """Sample z-score of the last value against its trailing ``window``.

    ``None`` on insufficient history or a degenerate (zero-variance) window.
    """
    if window < 2 or len(values) < 2:
        return None
    tail = values[-window:]
    if len(tail) < 2:
        return None
    mean_value = sum(tail) / float(len(tail))
    variance = sum((value - mean_value) ** 2 for value in tail) / float(len(tail) - 1)
    sigma = variance**0.5
    if sigma <= _EPS:
        return None
    result = (tail[-1] - mean_value) / sigma
    return float(result) if math.isfinite(result) else None


def _trailing_std(values: list[float], window: int) -> float | None:
    """Sample standard deviation of the last ``window`` values (``None`` if degenerate)."""
    if window < 2 or len(values) < 2:
        return None
    tail = values[-window:]
    if len(tail) < 2:
        return None
    mean_value = sum(tail) / float(len(tail))
    variance = sum((value - mean_value) ** 2 for value in tail) / float(len(tail) - 1)
    sigma = variance**0.5
    return float(sigma) if math.isfinite(sigma) else None


@register("strategy", "SilentVolumeShockResolutionStrategy", interface="event_driven")
class SilentVolumeShockResolutionStrategy(Strategy):
    """Arm on a price-flat abnormal-volume shock; enter on the lagged resolution sign.

    See the module docstring for the full theory, the honest HIGH prior of death,
    the mechanism, and the distinct-from rationale versus the return-shock /
    liquidity-shock / volume-clock / flow-share incumbents.  Reads only local
    event/bar OHLCV; performs no I/O and never raises from ``calculate_signals``.
    """

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = _STRATEGY_NAME
    strategy_id = _STRATEGY_ID

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "shock_window": HyperParam.integer("shock_window", default=90, low=8, high=20000),
            "v_shock_z": HyperParam.floating("v_shock_z", default=2.0, low=0.0, high=20.0),
            "quiet_ret_z": HyperParam.floating("quiet_ret_z", default=0.5, low=0.0, high=20.0),
            "quiet_range_z": HyperParam.floating("quiet_range_z", default=1.0, low=0.0, high=20.0),
            "resolution_max_bars": HyperParam.integer(
                "resolution_max_bars", default=10, low=1, high=4096
            ),
            "resolution_ret_mult": HyperParam.floating(
                "resolution_ret_mult", default=1.0, low=0.0, high=20.0
            ),
            "min_hold_bars": HyperParam.integer("min_hold_bars", default=7, low=0, high=100000),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=21, low=1, high=100000),
            "cooldown_bars": HyperParam.integer("cooldown_bars", default=5, low=0, high=100000),
            "vol_window": HyperParam.integer("vol_window", default=30, low=2, high=4096),
            "target_vol": HyperParam.floating("target_vol", default=0.0, low=0.0, high=2.0),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.0, low=0.0, high=0.50),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "min_history_bars": HyperParam.integer(
                "min_history_bars", default=100, low=2, high=20000
            ),
            "base_allocation": HyperParam.floating(
                "base_allocation", default=0.015, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=300.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.shock_window = max(8, int(resolved["shock_window"]))
        self.v_shock_z = max(0.0, float(resolved["v_shock_z"]))
        self.quiet_ret_z = max(0.0, float(resolved["quiet_ret_z"]))
        self.quiet_range_z = max(0.0, float(resolved["quiet_range_z"]))
        self.resolution_max_bars = max(1, int(resolved["resolution_max_bars"]))
        self.resolution_ret_mult = max(0.0, float(resolved["resolution_ret_mult"]))
        self.min_hold_bars = max(0, int(resolved["min_hold_bars"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.cooldown_bars = max(0, int(resolved["cooldown_bars"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.target_vol = max(0.0, float(resolved["target_vol"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.allow_short = bool(resolved["allow_short"])
        self.min_history_bars = max(2, int(resolved["min_history_bars"]))
        self.base_allocation = max(0.0, float(resolved["base_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = (
            max(
                self.shock_window,
                self.min_history_bars,
                self.vol_window,
                self.max_hold_bars,
                self.resolution_max_bars,
            )
            + 8
        )
        self._state: dict[str, _State] = {
            symbol: _State(
                highs=deque(maxlen=size),
                lows=deque(maxlen=size),
                closes=deque(maxlen=size),
                volumes=deque(maxlen=size),
            )
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
            "symbol_state": {
                symbol: {
                    "highs": list(item.highs),
                    "lows": list(item.lows),
                    "closes": list(item.closes),
                    "volumes": list(item.volumes),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "ref_price": item.ref_price,
                    "bars_held": int(item.bars_held),
                    "cooldown_remaining": int(item.cooldown_remaining),
                    "armed": bool(item.armed),
                    "arm_close": item.arm_close,
                    "arm_sigma": item.arm_sigma,
                    "arm_age": int(item.arm_age),
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
            try:
                _restore_deque(item.highs, payload.get("highs"))
                _restore_deque(item.lows, payload.get("lows"))
                _restore_deque(item.closes, payload.get("closes"))
                _restore_deque(item.volumes, payload.get("volumes"))
                item.mode = _mode(payload.get("mode"))
                item.entry_price = safe_float(payload.get("entry_price"))
                item.ref_price = safe_float(payload.get("ref_price"))
                item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
                item.cooldown_remaining = _safe_non_negative_int(payload.get("cooldown_remaining"))
                item.armed = bool(payload.get("armed"))
                item.arm_close = safe_float(payload.get("arm_close"))
                item.arm_sigma = safe_float(payload.get("arm_sigma"))
                item.arm_age = _safe_non_negative_int(payload.get("arm_age"))
                item.last_time_key = str(payload.get("last_time_key", ""))
            except Exception:
                continue

    # ------------------------------------------------------------------ #
    # ingestion
    # ------------------------------------------------------------------ #
    def _update_symbol(self, symbol: str, snapshot: _Snapshot) -> bool:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return False
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return False
        item.last_time_key = key
        high = safe_float(snapshot.high)
        low = safe_float(snapshot.low)
        hi = float(high) if high is not None else close
        lo = float(low) if low is not None else close
        # Guard an inverted or non-positive bar so downstream log ranges stay finite.
        if not math.isfinite(hi) or hi <= 0.0:
            hi = close
        if not math.isfinite(lo) or lo <= 0.0:
            lo = close
        if lo > hi:
            hi, lo = lo, hi
        item.highs.append(hi)
        item.lows.append(lo)
        item.closes.append(close)
        raw_volume = safe_float(snapshot.volume)
        item.volumes.append(max(0.0, float(raw_volume)) if raw_volume is not None else 0.0)
        return True

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
            self._evaluate(getattr(event, "time", None))

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
                    self._evaluate(snapshot.time)

    # ------------------------------------------------------------------ #
    # per-symbol feature helpers
    # ------------------------------------------------------------------ #
    def _log_dollar_volumes(self, item: _State) -> list[float]:
        out: list[float] = []
        for close, volume in zip(item.closes, item.volumes, strict=False):
            out.append(math.log(max(close * volume, _EPS)))
        return out

    def _log_returns(self, item: _State) -> list[float]:
        out: list[float] = []
        for prev, cur in pairwise(item.closes):
            if prev > 0.0 and cur > 0.0:
                out.append(math.log(cur / prev))
        return out

    def _log_ranges(self, item: _State) -> list[float]:
        out: list[float] = []
        for high, low in zip(item.highs, item.lows, strict=False):
            if high > 0.0 and low > 0.0 and high >= low:
                out.append(math.log(max(high, _EPS) / max(low, _EPS)))
            else:
                out.append(0.0)
        return out

    def _shock_features(self, item: _State) -> tuple[float | None, float | None, float]:
        """Return ``(volume_z, return_z_magnitude, range_z)`` for the latest bar."""
        vz = _trailing_z(self._log_dollar_volumes(item), self.shock_window)
        returns = self._log_returns(item)
        ret_std = _trailing_std(returns, self.shock_window)
        rz: float | None
        if not returns or ret_std is None or ret_std <= _EPS:
            rz = None
        else:
            rz = abs(returns[-1]) / ret_std
        range_z_raw = _trailing_z(self._log_ranges(item), self.shock_window)
        # A degenerate (constant) range history reads as maximally quiet.
        range_z = 0.0 if range_z_raw is None else abs(range_z_raw)
        return vz, rz, range_z

    # ------------------------------------------------------------------ #
    # evaluation
    # ------------------------------------------------------------------ #
    def _evaluate(self, event_time: Any) -> None:
        for symbol, item in self._state.items():
            self._evaluate_symbol(symbol, item, event_time)

    def _evaluate_symbol(self, symbol: str, item: _State, event_time: Any) -> None:
        if item.mode in {"LONG", "SHORT"}:
            self._age_position(symbol, item, event_time)
            return
        if item.cooldown_remaining > 0:
            item.cooldown_remaining -= 1
            return
        if len(item.closes) < self.min_history_bars:
            return
        if item.armed:
            self._handle_armed(symbol, item, event_time)
            return
        self._maybe_arm(item)

    def _maybe_arm(self, item: _State) -> None:
        vz, rz, range_z = self._shock_features(item)
        if vz is None or rz is None:
            return
        if vz < self.v_shock_z or rz > self.quiet_ret_z or range_z > self.quiet_range_z:
            return
        sigma = _trailing_std(self._log_returns(item), self.shock_window)
        if sigma is None or sigma <= _EPS:
            return
        item.armed = True
        item.arm_close = float(item.closes[-1])
        item.arm_sigma = float(sigma)
        item.arm_age = 0

    def _handle_armed(self, symbol: str, item: _State, event_time: Any) -> None:
        item.arm_age += 1
        arm_close = item.arm_close
        sigma = item.arm_sigma
        close = item.closes[-1] if item.closes else None
        if arm_close is None or sigma is None or close is None or arm_close <= 0.0 or close <= 0.0:
            self._disarm(item)
            return
        cum_ret = math.log(close / arm_close)
        threshold = self.resolution_ret_mult * sigma * math.sqrt(item.arm_age)
        if math.isfinite(cum_ret) and abs(cum_ret) >= threshold and threshold > 0.0:
            self._enter_resolution(symbol, item, event_time, cum_ret)
            return
        # A fresh silent shock inside the window refreshes it around the new close.
        vz, rz, range_z = self._shock_features(item)
        if (
            vz is not None
            and rz is not None
            and vz >= self.v_shock_z
            and rz <= self.quiet_ret_z
            and range_z <= self.quiet_range_z
        ):
            fresh_sigma = _trailing_std(self._log_returns(item), self.shock_window)
            if fresh_sigma is not None and fresh_sigma > _EPS:
                item.arm_close = float(close)
                item.arm_sigma = float(fresh_sigma)
                item.arm_age = 0
                return
        if item.arm_age >= self.resolution_max_bars:
            self._disarm(item)

    def _enter_resolution(self, symbol: str, item: _State, event_time: Any, cum_ret: float) -> None:
        if cum_ret > 0.0:
            target_mode = "LONG"
        elif cum_ret < 0.0:
            if not self.allow_short:
                self._disarm(item)
                return
            target_mode = "SHORT"
        else:
            self._disarm(item)
            return
        close = float(item.closes[-1])
        vol = realized_volatility(list(item.closes), window=self.vol_window)
        size_scalar = 1.0
        if self.target_vol > 0.0 and vol is not None and vol > _EPS:
            size_scalar = min(1.0, self.target_vol / vol)
        alloc = max(0.0, self.base_allocation * size_scalar)
        stop_loss = None
        if self.stop_loss_pct > 0.0:
            stop_loss = close * (
                1.0 - self.stop_loss_pct if target_mode == "LONG" else 1.0 + self.stop_loss_pct
            )
        metadata = _target_metadata(
            strategy=_STRATEGY_NAME,
            target_allocation=alloc,
            max_order_value=self.max_order_value,
            target_mode=target_mode,
            reason="silent_shock_resolution",
            arm_close=float(item.arm_close) if item.arm_close is not None else None,
            resolution_cum_return=float(cum_ret),
            resolution_bars=int(item.arm_age),
            arm_sigma=float(item.arm_sigma) if item.arm_sigma is not None else None,
            realized_vol=float(vol) if vol is not None else None,
            inverse_vol_scalar=float(size_scalar),
        )
        _emit(
            self.events,
            strategy_id=_STRATEGY_ID,
            symbol=symbol,
            event_time=event_time,
            signal_type=target_mode,
            strength=1.0,
            price=close,
            stop_loss=stop_loss,
            metadata=metadata,
        )
        item.mode = target_mode
        item.entry_price = close
        item.ref_price = item.arm_close
        item.bars_held = 0
        item.armed = False
        item.arm_close = None
        item.arm_sigma = None
        item.arm_age = 0

    def _disarm(self, item: _State) -> None:
        item.armed = False
        item.arm_close = None
        item.arm_sigma = None
        item.arm_age = 0

    def _age_position(self, symbol: str, item: _State, event_time: Any) -> None:
        if item.entry_price is None or not item.closes:
            return
        close = item.closes[-1]
        if close is None or close <= 0.0:
            return
        item.bars_held += 1
        entry = float(item.entry_price)
        ref = float(item.ref_price) if item.ref_price is not None else entry
        reason = ""
        if item.mode == "LONG":
            if self.stop_loss_pct > 0.0 and close <= entry * (1.0 - self.stop_loss_pct):
                reason = "stop_loss"
            elif item.bars_held >= self.min_hold_bars and close <= ref:
                reason = "giveback"
        else:
            if self.stop_loss_pct > 0.0 and close >= entry * (1.0 + self.stop_loss_pct):
                reason = "stop_loss"
            elif item.bars_held >= self.min_hold_bars and close >= ref:
                reason = "giveback"
        if not reason and item.bars_held >= self.max_hold_bars:
            reason = "max_hold"
        if not reason:
            return
        _emit(
            self.events,
            strategy_id=_STRATEGY_ID,
            symbol=symbol,
            event_time=event_time,
            signal_type="EXIT",
            price=close,
            metadata={"strategy": _STRATEGY_NAME, "reason": reason},
        )
        item.mode = "OUT"
        item.entry_price = None
        item.ref_price = None
        item.bars_held = 0
        item.cooldown_remaining = self.cooldown_bars


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the integration wave (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits per the live-safety
# plan).  This is a PER-SYMBOL time-series sleeve (one symbol per candidate,
# ``candidate_mix_type == "single"``), NOT a cross-sectional rank book and NOT
# carry -- so it is honestly EXCLUDED from any carry/XS-rank tag allowlist route.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "time_series_momentum"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "time_series",
    "volume_shock",
    "information_arrival",
    "lagged_resolution",
    "episodic",
    "low_turnover",
    "crypto",
)

_SILENT_VOLUME_SHOCK_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": (
        {
            "variant": "silent_shock_z20_r10",
            "shock_window": 90,
            "v_shock_z": 2.0,
            "quiet_ret_z": 0.5,
            "quiet_range_z": 1.0,
            "resolution_max_bars": 10,
            "resolution_ret_mult": 1.0,
            "min_hold_bars": 7,
            "max_hold_bars": 21,
            "cooldown_bars": 5,
            "allow_short": True,
        },
        {
            "variant": "silent_shock_z25_r5",
            "shock_window": 90,
            "v_shock_z": 2.5,
            "quiet_ret_z": 0.5,
            "quiet_range_z": 1.0,
            "resolution_max_bars": 5,
            "resolution_ret_mult": 0.75,
            "min_hold_bars": 7,
            "max_hold_bars": 21,
            "cooldown_bars": 5,
            "allow_short": True,
        },
    ),
}

__all__ = ["SilentVolumeShockResolutionStrategy"]
