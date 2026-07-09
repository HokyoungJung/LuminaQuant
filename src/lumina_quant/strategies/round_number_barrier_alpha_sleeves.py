"""Round-number psychological-barrier episodic sleeve (CONDITIONAL, HIGH death-prior).

``RoundNumberBarrierStrategy`` trades short-horizon episodic behaviour around
psychological round-number price levels on an EX-ANTE-FROZEN power-of-ten grid:
one proximity transform + one approach-direction interaction, bounce vs
breakout-acceleration, with a half-grid profit target.

HONEST PRIOR OF DEATH: HIGH -- the LOWEST-confidence lane in the wave, priced in
via the ``research_only`` tier and a BINDING half-shifted PLACEBO-GRID falsifier
(``round_number_grid(..., half_shift=True)``).  The published evidence (Osler
2003; Urquhart 2017) is ORDER/TRADE clustering at round numbers, NOT proven
crypto forward-return predictability; the primary pre-registered null is that
marginal crypto execution flow is bots/MMs without left-digit bias, so the
barrier-conditional forward return equals the unconditional drift (IC ~ 0, sweep
cells flip sign).  A clean rejection reported in full is a valid outcome.

THEORY / PROVENANCE
-------------------
- Osler (2003, JF 58(5)) / Osler (2000, FRBNY EPR): in real FX dealer order books
  take-profit orders cluster AT round numbers (the bounce leg) and stop-losses
  just BEYOND them (the breakout leg rides the forced stop/liquidation flow).
- Donaldson & Kim (1993, JFQA 28(3)): round-number support/resistance in the DJIA
  (covers the TradFi legs).
- Bhattacharya, Holden & Jacobsen (2012, Mgmt Sci 58(2)): left-digit / clustering
  effects around thresholds.
- Urquhart (2017, Economics Letters 159): heavy BTC round-number CLUSTERING
  (clustering only -- return predictability is NOT established, flagged).

MECHANISM (per completed daily bar; OHLCV only; per symbol independently)
------------------------------------------------------------------------
GRID (frozen ex-ante, zero fitted salience): ``k = floor(log10(close))``,
spacing ``g = 10^(k-1)`` (BTC@60000 -> $1000 levels; coin@0.75 -> $0.01), nearest
level ``L = round(close/g)*g``.  ONE PROXIMITY TRANSFORM: signed normalized
distance ``d = (close - L)/g in [-0.5, +0.5]``; the barrier is engaged iff
``|d| <= prox_band``.  ONE DIRECTION INTERACTION: the trailing approach return
``r_app`` over ``approach_bars`` must satisfy ``|r_app| >= 0.5*g/close`` (the
approach must have traversed at least half a grid cell -- kills idle-drift noise);
``mode = 'breakout'`` when ``sign(r_app) == sign(d)`` (the level was cleared -->
ride the stop cascade away from it) else ``'bounce'`` (the level held ahead -->
fade back off it).  Position DIRECTION = ``sign(d)`` in BOTH modes -- the unified
Osler hypothesis that the level repels price away from whichever side it occupies;
``d == 0`` exactly -> abstain.  ``mode`` is metadata plus a pre-registered
``{both, bounce_only, breakout_only}`` filter, NEVER a fitted sign flip.

Execution: one entry per ``(level, side)`` episode (re-armed only after price
leaves the band); inverse-realized-vol sized; hard ``min_hold_bars``; exit at
``|close - L| >= 0.5*g`` (mid-grid / next-level target) after the min-hold or at
``max_hold_bars``; a ``cooldown_bars`` lockout on exit; min-history + min-price
floors; never-raise on ``close <= 0`` (``log10`` undefined -> skip).  Deliberately
minimal: exactly one transform + one interaction, no volume leg, no regime gate,
no per-symbol tuning.

This is a PER-SYMBOL episodic leaf applied with IDENTICAL frozen params across the
whole liquid book; it is NOT a cross-sectional rank book and carries NO fake
carry/XS-rank family tag.  It is data-local (no I/O), pure Python (``math`` +
``deque`` only, no numpy), completed-bar, never raises from ``calculate_signals``,
and ships WITHOUT ``@register`` (inert until a later integration wave wires it as
``research_only``).

DISTINCT-FROM
-------------
Versus ``DonchianAtrTrendStrategy`` / ``FalseBreakoutReversalStrategy`` (path-
extremum channel breakouts): those need a genuine new channel high/low; this
sleeve engages on an EXOGENOUS grid level that need not be a path extremum.
Versus ``HourlyShockReversionStrategy`` (magnitude-only shock fade): two identical
shocks ending at different prices act identically for the shock fader but
differently here, purely by grid proximity.  Versus
``CrossSectionalNearHighAnchoringStrategy`` (XS trailing-high anchor): its book is
set by distance-from-52wk-high, independent of the absolute round-number grid.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.indicators.alpha_features import realized_volatility, simple_return
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

_STRATEGY_ID = "round_number_barrier"
_STRATEGY_NAME = "RoundNumberBarrierStrategy"


def round_number_grid(close: Any, *, half_shift: bool = False) -> tuple[float, float, float] | None:
    """Return ``(level, spacing, signed_distance)`` for ``close`` on the frozen grid.

    The grid is a pure power-of-ten decade lattice: ``k = floor(log10(close))``,
    spacing ``g = 10^(k-1)``, nearest level ``L = round(close/g)*g``, and signed
    normalized distance ``d = (close - L)/g in [-0.5, +0.5]``.  ``half_shift=True``
    is the falsifier: the lattice is offset by half a cell (levels at
    ``(m + 0.5)*g``) so a true salience effect must VANISH on it by construction.
    Returns ``None`` on a non-finite or non-positive close (``log10`` undefined).
    """
    value = safe_float(close)
    if value is None or not math.isfinite(value) or value <= 0.0:
        return None
    k = math.floor(math.log10(value))
    spacing = 10.0 ** (k - 1)
    if not math.isfinite(spacing) or spacing <= 0.0:
        return None
    if half_shift:
        base = math.floor(value / spacing)
        candidates = ((base - 0.5) * spacing, (base + 0.5) * spacing)
        level = min(candidates, key=lambda lvl: abs(value - lvl))
    else:
        level = round(value / spacing) * spacing
    distance = (value - level) / spacing
    if not math.isfinite(distance):
        return None
    return float(level), float(spacing), float(distance)


@dataclass(slots=True)
class _State:
    """Per-symbol OHLC history + episodic position/cooldown machine state."""

    closes: deque[float]
    mode: str = "OUT"  # OUT / LONG / SHORT
    entry_price: float | None = None
    ref_level: float | None = None
    ref_spacing: float | None = None
    bars_held: int = 0
    cooldown_remaining: int = 0
    last_episode_sign: int = 0  # sign of the last-traded level side (0 == none)
    last_episode_level: float | None = None
    last_time_key: str = ""


class RoundNumberBarrierStrategy(Strategy):
    """Trade episodic bounce/breakout around a frozen power-of-ten round-number grid.

    See the module docstring for the full theory, the honest HIGH prior of death,
    the frozen-grid mechanism, and the distinct-from rationale versus the channel-
    breakout / shock-fade / near-high incumbents.  Reads only local event/bar
    OHLCV; performs no I/O and never raises from ``calculate_signals``.
    """

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = _STRATEGY_NAME
    strategy_id = _STRATEGY_ID

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "prox_band": HyperParam.floating("prox_band", default=0.15, low=0.0, high=0.5),
            "approach_bars": HyperParam.integer("approach_bars", default=5, low=1, high=512),
            # The pre-registered {both, bounce_only, breakout_only} sweep lives in
            # the candidate slice below; the schema exposes it as a tunable string.
            "mode_filter": HyperParam.string("mode_filter", default="both", tunable=True),
            "min_hold_bars": HyperParam.integer("min_hold_bars", default=3, low=0, high=100000),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=10, low=1, high=100000),
            "cooldown_bars": HyperParam.integer("cooldown_bars", default=5, low=0, high=100000),
            "vol_window": HyperParam.integer("vol_window", default=20, low=2, high=4096),
            "target_vol": HyperParam.floating("target_vol", default=0.0, low=0.0, high=2.0),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "min_history_bars": HyperParam.integer(
                "min_history_bars", default=20, low=2, high=20000
            ),
            "base_allocation": HyperParam.floating(
                "base_allocation", default=0.015, low=0.0, high=2.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=300.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.01, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.prox_band = max(0.0, min(0.5, float(resolved["prox_band"])))
        self.approach_bars = max(1, int(resolved["approach_bars"]))
        mode_filter = str(resolved["mode_filter"] or "both").lower()
        self.mode_filter = (
            mode_filter if mode_filter in {"both", "bounce_only", "breakout_only"} else "both"
        )
        self.min_hold_bars = max(0, int(resolved["min_hold_bars"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.cooldown_bars = max(0, int(resolved["cooldown_bars"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.target_vol = max(0.0, float(resolved["target_vol"]))
        self.allow_short = bool(resolved["allow_short"])
        self.min_history_bars = max(2, int(resolved["min_history_bars"]))
        self.base_allocation = max(0.0, float(resolved["base_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        size = (
            max(self.min_history_bars, self.approach_bars + 1, self.vol_window, self.max_hold_bars)
            + 8
        )
        self._state: dict[str, _State] = {
            symbol: _State(closes=deque(maxlen=size)) for symbol in self.symbol_list
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
                    "closes": list(item.closes),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "ref_level": item.ref_level,
                    "ref_spacing": item.ref_spacing,
                    "bars_held": int(item.bars_held),
                    "cooldown_remaining": int(item.cooldown_remaining),
                    "last_episode_sign": int(item.last_episode_sign),
                    "last_episode_level": item.last_episode_level,
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
                _restore_deque(item.closes, payload.get("closes"))
                item.mode = _mode(payload.get("mode"))
                item.entry_price = safe_float(payload.get("entry_price"))
                item.ref_level = safe_float(payload.get("ref_level"))
                item.ref_spacing = safe_float(payload.get("ref_spacing"))
                item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
                item.cooldown_remaining = _safe_non_negative_int(payload.get("cooldown_remaining"))
                raw_sign = safe_float(payload.get("last_episode_sign"))
                item.last_episode_sign = int(raw_sign) if raw_sign in (-1.0, 1.0) else 0
                item.last_episode_level = safe_float(payload.get("last_episode_level"))
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
        item.closes.append(close)
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
        self._maybe_enter(symbol, item, event_time)

    def _maybe_enter(self, symbol: str, item: _State, event_time: Any) -> None:
        close = item.closes[-1]
        grid = round_number_grid(close)
        if grid is None:
            return
        level, spacing, distance = grid
        # Price outside the proximity band -> the episode resets (re-armable).
        if abs(distance) > self.prox_band:
            item.last_episode_sign = 0
            item.last_episode_level = None
            return
        if distance == 0.0:
            return  # exactly on the level -> abstain (undefined side)
        r_app = simple_return(list(item.closes), lookback=self.approach_bars)
        if r_app is None:
            return
        # The approach must have traversed at least half a grid cell.
        if abs(r_app) < 0.5 * spacing / close:
            return
        side = 1 if distance > 0.0 else -1
        mode = "breakout" if (r_app > 0.0) == (distance > 0.0) else "bounce"
        if self.mode_filter == "bounce_only" and mode != "bounce":
            return
        if self.mode_filter == "breakout_only" and mode != "breakout":
            return
        if side < 0 and not self.allow_short:
            return
        # One entry per (level, side) episode until price leaves the band.
        if item.last_episode_sign == side and item.last_episode_level == level:
            return
        self._enter(symbol, item, event_time, level, spacing, distance, r_app, mode, side)

    def _enter(
        self,
        symbol: str,
        item: _State,
        event_time: Any,
        level: float,
        spacing: float,
        distance: float,
        r_app: float,
        mode: str,
        side: int,
    ) -> None:
        target_mode = "LONG" if side > 0 else "SHORT"
        close = item.closes[-1]
        vol = realized_volatility(list(item.closes), window=self.vol_window)
        size_scalar = 1.0
        if self.target_vol > 0.0 and vol is not None and vol > _EPS:
            size_scalar = min(1.0, self.target_vol / vol)
        alloc = max(0.0, self.base_allocation * size_scalar)
        metadata = _target_metadata(
            strategy=_STRATEGY_NAME,
            target_allocation=alloc,
            max_order_value=self.max_order_value,
            target_mode=target_mode,
            reason="round_number_barrier",
            grid_level=float(level),
            grid_spacing=float(spacing),
            proximity=float(distance),
            approach_return=float(r_app),
            barrier_mode=mode,
            realized_vol=float(vol) if vol is not None else None,
            inverse_vol_scalar=float(size_scalar),
        )
        _emit(
            self.events,
            strategy_id=_STRATEGY_ID,
            symbol=symbol,
            event_time=event_time,
            signal_type=target_mode,
            strength=max(0.25, min(3.0, 1.0 - abs(distance) / max(self.prox_band, _EPS) + 1.0)),
            price=close,
            metadata=metadata,
        )
        item.mode = target_mode
        item.entry_price = close
        item.ref_level = level
        item.ref_spacing = spacing
        item.bars_held = 0
        item.last_episode_sign = side
        item.last_episode_level = level

    def _age_position(self, symbol: str, item: _State, event_time: Any) -> None:
        if item.entry_price is None or item.ref_level is None or item.ref_spacing is None:
            return
        if not item.closes:
            return
        close = item.closes[-1]
        if close is None or close <= 0.0:
            return
        item.bars_held += 1
        target = 0.5 * float(item.ref_spacing)
        reason = ""
        if item.bars_held >= self.min_hold_bars and abs(close - float(item.ref_level)) >= target:
            reason = "half_grid_target"
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
        item.ref_level = None
        item.ref_spacing = None
        item.bars_held = 0
        item.cooldown_remaining = self.cooldown_bars


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the integration wave (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits per the live-safety
# plan).  This is a PER-SYMBOL episodic leaf (one symbol per candidate,
# ``candidate_mix_type == "single"``) with IDENTICAL frozen params across the
# book -- NOT a cross-sectional rank book and NOT carry, so it is honestly
# EXCLUDED from any carry/XS-rank tag allowlist route.  The default mode "both"
# is bounce-dominant per Osler 2000; the breakout_only cells carry a "breakout"
# tag for re-tag at integration.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "mean_reversion"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "round_number",
    "psychological_barrier",
    "support_resistance",
    "episodic",
    "left_digit_bias",
    "crypto",
)

_ROUND_NUMBER_BARRIER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": (
        {
            "variant": "rn_band15_both",
            "prox_band": 0.15,
            "approach_bars": 5,
            "mode_filter": "both",
            "min_hold_bars": 3,
            "max_hold_bars": 10,
            "cooldown_bars": 5,
            "allow_short": True,
        },
        {
            "variant": "rn_band10_bounce",
            "prox_band": 0.10,
            "approach_bars": 3,
            "mode_filter": "bounce_only",
            "min_hold_bars": 3,
            "max_hold_bars": 10,
            "cooldown_bars": 5,
            "allow_short": True,
        },
    ),
}

__all__ = ["RoundNumberBarrierStrategy", "round_number_grid"]
