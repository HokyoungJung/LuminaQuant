"""Spread-stress-gated liquidity-provision reversal (CONDITIONAL, high death-prior).

``SpreadStressLiquidityReversionStrategy`` fades the trailing 2-3 bar return of a
liquid major ONLY inside a spread-STRESS episode -- and stays structurally flat
otherwise.  The episode gate is an OHLC-estimated effective spread (the
Corwin-Schultz 2012 high-low estimator, :mod:`lumina_quant.indicators.hl_spread`)
z-scored against its own trailing distribution: an entry is permitted only when
that z clears ``z_entry`` (a rare ~2-5% tail of bars), one entry per contiguous
episode, with a hard min-hold, a reversion-target / max-hold exit, and a
cooldown.

THEORY / PROVENANCE
-------------------
- Nagel (2012, RFS 25(7)): short-horizon reversal profits ARE the expected
  return to liquidity provision, and they spike precisely when market-making
  capital withdraws -- observable as a widening effective spread.
- Corwin & Schultz (2012, JF 67(2)): the effective spread is recoverable from
  OHLC alone (the high-low range's variance component scales with interval
  length while the spread component does not).
- Brunnermeier & Pedersen (2009, RFS 22(6)) / Grossman-Miller (1988, JF 43(3)):
  the counterparty is forced immediacy demand (mechanical margin liquidations,
  panic sellers) whose payment is the structural price of immediacy, not a
  correctable mistake -- so the edge is risk-bearing income that draws down
  exactly in the worst regimes.

HONEST PRIOR OF DEATH: HIGH.  This is the highest base-rate-of-death family in
the repo (short-horizon reversal), transplanted to crypto where UNCONDITIONAL
daily reversal on liquid majors is WRONG-SIGNED (liquid coins show daily
MOMENTUM, external verdict 9).  The entire design IS the conditioning: supply
liquidity only when the OHLC-estimated spread says immediacy is being paid for.
The lane exists to MEASURE whether that conditioning flips the wrong sign; the
expected null (liquidation-cascade continuation dominates gated-episode
reversal) is pre-registered, and a clean rejection is a valid outcome.

MECHANISM (per completed decision bar; OHLCV only; per symbol independently)
----------------------------------------------------------------------------
1. Maintain per-symbol OHLCV deques; compute the smoothed Corwin-Schultz spread
   over the trailing ``cs_smooth_window + 1`` bars and append it to a per-symbol
   spread history.
2. ``spread_stress_z`` = trailing z-score of that spread history over
   ``z_window_bars``.  A contiguous run of bars with ``z >= z_entry`` is an
   EPISODE; ``one_entry_per_episode`` permits a single entry per episode.
3. Inside an open episode, when flat and past cooldown, fade the trailing
   ``fade_lookback_bars`` return: a NEGATIVE trailing return -> LONG (buy the
   stressed sell-off), a POSITIVE trailing return -> SHORT.  Size inverse-vol
   (``target_vol / realized_vol`` capped at 1x).  Calm regime -> no signal (no
   bolted-on momentum leg: cheap and orthogonal).
4. Exits: hard stop-loss any time; reversion-target (``take_profit_pct``, only
   after ``min_hold_bars``); ``max_hold_bars`` cap.  On exit start a
   ``cooldown_bars`` lockout.  Liquid-book restriction: a name below
   ``min_history_bars`` or ``min_dollar_volume`` is skipped.

This is a PER-SYMBOL time-series sleeve applied across a multi-symbol liquid
book (honest ``allow_multi_asset``); it is NOT a cross-sectional rank book and
carries NO fake carry/momentum family tag.  It is data-local (no I/O), pure
Python (``math`` + ``deque`` + the pure ``corwin_schultz_spread`` primitive, no
numpy), never raises from ``calculate_signals``, and ships WITHOUT ``@register``
(inert until a later integration wave wires it as ``research_only``).

DISTINCT-FROM
-------------
Versus ``LiquidityShockReversionStrategy`` (a per-bar conjunctive volume/range/
return SHOCK fade): this sleeve is EPISODE- not shock-driven -- it can fire on a
multi-bar cumulative drift with flat volume and small per-bar returns purely
because the OHLC spread is stressed, where the shock incumbent's conjunction
stays silent.  Versus ``HourlyShockReversionStrategy``: an isolated price shock
in a CALM (low-spread) regime leaves this gate CLOSED (no trade) while the shock
fader trades.  Versus ``StationarityGatedResidualReversionStrategy``: a lockstep
asset+benchmark crash has ~0 residual (that sleeve abstains) yet a stressed
spread + negative trailing return here -> LONG; conversely an idiosyncratic
residual dislocation in a calm-spread market has this gate closed while the
residual sleeve trades.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import realized_volatility, simple_return
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.hl_spread import corwin_schultz_spread
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
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "spread_stress_reversion"
_STRATEGY_NAME = "SpreadStressLiquidityReversionStrategy"


@dataclass(slots=True)
class _SpreadState:
    highs: deque[float]
    lows: deque[float]
    closes: deque[float]
    volumes: deque[float]
    spreads: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    cooldown_remaining: int = 0
    in_episode: bool = False
    episode_entered: bool = False
    last_time_key: str = ""


def _rolling_z(values: list[float]) -> float | None:
    """Sample z-score of the last value vs its trailing window (``None`` if degenerate)."""
    if len(values) < 2:
        return None
    mean_value = sum(values) / float(len(values))
    variance = sum((value - mean_value) ** 2 for value in values) / float(len(values) - 1)
    sigma = variance**0.5
    if sigma <= _EPS:
        return None
    result = (values[-1] - mean_value) / sigma
    return float(result) if math.isfinite(result) else None


def _median(values: list[float]) -> float | None:
    cleaned = [float(value) for value in values if math.isfinite(value)]
    if not cleaned:
        return None
    ordered = sorted(cleaned)
    count = len(ordered)
    mid = count // 2
    if count % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _coerce_float_list(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[float] = []
    for item in value:
        parsed = safe_float(item)
        if parsed is not None:
            out.append(parsed)
    return out


@register("strategy", "SpreadStressLiquidityReversionStrategy", interface="event_driven")
class SpreadStressLiquidityReversionStrategy(Strategy):
    """Fade trailing returns only inside OHLC-spread-stress episodes on liquid majors.

    See the module docstring for the full theory, the honest HIGH prior of
    death, the mechanism, and the distinct-from rationale versus the shock /
    residual reversion incumbents.  Reads only local event/bar OHLCV; performs
    no I/O and never raises from ``calculate_signals``.
    """

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "cs_smooth_window": HyperParam.integer("cs_smooth_window", default=5, low=1, high=512),
            "z_window_bars": HyperParam.integer("z_window_bars", default=120, low=8, high=20000),
            "z_entry": HyperParam.floating("z_entry", default=2.0, low=0.0, high=20.0),
            "fade_lookback_bars": HyperParam.integer(
                "fade_lookback_bars", default=3, low=1, high=512
            ),
            "min_hold_bars": HyperParam.integer("min_hold_bars", default=5, low=0, high=100000),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=10, low=1, high=100000),
            "cooldown_bars": HyperParam.integer("cooldown_bars", default=5, low=0, high=100000),
            "one_entry_per_episode": HyperParam.boolean(
                "one_entry_per_episode", default=True, grid=[True, False]
            ),
            "vol_window_bars": HyperParam.integer("vol_window_bars", default=30, low=2, high=4096),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "min_history_bars": HyperParam.integer(
                "min_history_bars", default=150, low=2, high=20000
            ),
            "min_dollar_volume": HyperParam.floating(
                "min_dollar_volume", default=0.0, low=0.0, high=1e15
            ),
            "dollar_volume_window": HyperParam.integer(
                "dollar_volume_window", default=20, low=1, high=4096
            ),
            "target_vol": HyperParam.floating("target_vol", default=0.0, low=0.0, high=2.0),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.06, low=0.0, high=0.50),
            "take_profit_pct": HyperParam.floating(
                "take_profit_pct", default=0.0, low=0.0, high=1.0
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
        self.cs_smooth_window = max(1, int(resolved["cs_smooth_window"]))
        self.z_window_bars = max(8, int(resolved["z_window_bars"]))
        self.z_entry = max(0.0, float(resolved["z_entry"]))
        self.fade_lookback_bars = max(1, int(resolved["fade_lookback_bars"]))
        self.min_hold_bars = max(0, int(resolved["min_hold_bars"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.cooldown_bars = max(0, int(resolved["cooldown_bars"]))
        self.one_entry_per_episode = bool(resolved["one_entry_per_episode"])
        self.vol_window_bars = max(2, int(resolved["vol_window_bars"]))
        self.allow_short = bool(resolved["allow_short"])
        self.min_history_bars = max(2, int(resolved["min_history_bars"]))
        self.min_dollar_volume = max(0.0, float(resolved["min_dollar_volume"]))
        self.dollar_volume_window = max(1, int(resolved["dollar_volume_window"]))
        self.target_vol = max(0.0, float(resolved["target_vol"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.take_profit_pct = max(0.0, float(resolved["take_profit_pct"]))
        self.base_allocation = max(0.0, float(resolved["base_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        # Minimum trailing spread observations before a z-score is trusted for
        # the episode gate (a fraction of the configured window).
        self._z_history_min = max(8, self.z_window_bars // 4)
        bar_size = (
            max(
                self.min_history_bars,
                self.z_window_bars,
                self.vol_window_bars,
                self.dollar_volume_window,
                self.fade_lookback_bars,
                self.max_hold_bars,
                self.cs_smooth_window,
            )
            + 8
        )
        self._state: dict[str, _SpreadState] = {
            symbol: _SpreadState(
                highs=deque(maxlen=bar_size),
                lows=deque(maxlen=bar_size),
                closes=deque(maxlen=bar_size),
                volumes=deque(maxlen=bar_size),
                spreads=deque(maxlen=self.z_window_bars),
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
                    "spreads": list(item.spreads),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": int(item.bars_held),
                    "cooldown_remaining": int(item.cooldown_remaining),
                    "in_episode": bool(item.in_episode),
                    "episode_entered": bool(item.episode_entered),
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
                for attr in ("highs", "lows", "closes", "volumes", "spreads"):
                    target = getattr(item, attr)
                    target.clear()
                    maxlen = int(target.maxlen or 0)
                    values = _coerce_float_list(payload.get(attr))
                    for value in values[-maxlen:] if maxlen else values:
                        target.append(value)
                mode = str(payload.get("mode", "OUT")).upper()
                item.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
                item.entry_price = safe_float(payload.get("entry_price"))
                item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
                item.cooldown_remaining = _safe_non_negative_int(payload.get("cooldown_remaining"))
                item.in_episode = bool(payload.get("in_episode"))
                item.episode_entered = bool(payload.get("episode_entered"))
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
        item.highs.append(float(high if high is not None else close))
        item.lows.append(float(low if low is not None else close))
        item.closes.append(close)
        item.volumes.append(max(0.0, float(snapshot.volume or 0.0)))
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
    # per-symbol evaluation
    # ------------------------------------------------------------------ #
    def _evaluate(self, event_time: Any) -> None:
        for symbol, item in self._state.items():
            self._evaluate_symbol(symbol, item, event_time)

    def _update_spread(self, item: _SpreadState) -> None:
        if len(item.highs) < self.cs_smooth_window + 1:
            return
        spread = corwin_schultz_spread(
            list(item.highs), list(item.lows), smooth_window=self.cs_smooth_window
        )
        if spread is not None and math.isfinite(spread):
            item.spreads.append(float(spread))

    def _spread_stress_z(self, item: _SpreadState) -> float | None:
        if len(item.spreads) < self._z_history_min:
            return None
        return _rolling_z(list(item.spreads)[-self.z_window_bars :])

    def _liquid(self, item: _SpreadState) -> bool:
        if len(item.closes) < self.min_history_bars:
            return False
        if self.min_dollar_volume <= 0.0:
            return True
        closes = list(item.closes)[-self.dollar_volume_window :]
        volumes = list(item.volumes)[-self.dollar_volume_window :]
        dollar_volumes = [
            close * volume for close, volume in zip(closes, volumes, strict=False) if close > 0.0
        ]
        median_dv = _median(dollar_volumes)
        return median_dv is not None and median_dv >= self.min_dollar_volume

    def _age_position(self, symbol: str, item: _SpreadState, event_time: Any) -> bool:
        if item.mode not in {"LONG", "SHORT"} or item.entry_price is None:
            return False
        close = item.closes[-1] if item.closes else None
        if close is None or close <= 0.0:
            return False
        item.bars_held += 1
        entry = float(item.entry_price)
        reason = ""
        if item.mode == "LONG":
            if self.stop_loss_pct > 0.0 and close <= entry * (1.0 - self.stop_loss_pct):
                reason = "stop_loss"
            elif (
                self.take_profit_pct > 0.0
                and item.bars_held >= self.min_hold_bars
                and close >= entry * (1.0 + self.take_profit_pct)
            ):
                reason = "reversion_target"
        else:
            if self.stop_loss_pct > 0.0 and close >= entry * (1.0 + self.stop_loss_pct):
                reason = "stop_loss"
            elif (
                self.take_profit_pct > 0.0
                and item.bars_held >= self.min_hold_bars
                and close <= entry * (1.0 - self.take_profit_pct)
            ):
                reason = "reversion_target"
        if not reason and item.bars_held >= self.max_hold_bars:
            reason = "max_hold"
        if not reason:
            return False
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
        item.bars_held = 0
        item.cooldown_remaining = self.cooldown_bars
        return True

    def _update_episode(self, item: _SpreadState, z: float | None) -> None:
        stressed = z is not None and z >= self.z_entry
        if stressed:
            if not item.in_episode:
                # A fresh spread-stress episode opens: reset the one-entry latch.
                item.episode_entered = False
            item.in_episode = True
        else:
            item.in_episode = False

    def _evaluate_symbol(self, symbol: str, item: _SpreadState, event_time: Any) -> None:
        self._update_spread(item)
        self._age_position(symbol, item, event_time)
        z = self._spread_stress_z(item)
        self._update_episode(item, z)
        if item.mode != "OUT":
            return
        if item.cooldown_remaining > 0:
            item.cooldown_remaining -= 1
            return
        if not item.in_episode:
            return
        if self.one_entry_per_episode and item.episode_entered:
            return
        self._maybe_enter(symbol, item, event_time, z)

    def _maybe_enter(
        self, symbol: str, item: _SpreadState, event_time: Any, z: float | None
    ) -> None:
        if z is None or not self._liquid(item):
            return
        closes = list(item.closes)
        fade_return = simple_return(closes, lookback=self.fade_lookback_bars)
        vol = realized_volatility(closes, window=self.vol_window_bars)
        if fade_return is None or vol is None or vol <= _EPS:
            return
        if fade_return < 0.0:
            target_mode = "LONG"
        elif fade_return > 0.0:
            if not self.allow_short:
                return
            target_mode = "SHORT"
        else:
            return
        # Mark the episode consumed even if sizing collapses, so one_entry holds.
        item.episode_entered = True
        size_scalar = 1.0
        if self.target_vol > 0.0:
            size_scalar = min(1.0, self.target_vol / max(vol, _EPS))
        alloc = max(0.0, self.base_allocation * size_scalar)
        close = closes[-1]
        stop_loss = None
        if self.stop_loss_pct > 0.0:
            stop_loss = close * (
                1.0 - self.stop_loss_pct if target_mode == "LONG" else 1.0 + self.stop_loss_pct
            )
        reversion_target = None
        if self.take_profit_pct > 0.0:
            reversion_target = close * (
                1.0 + self.take_profit_pct if target_mode == "LONG" else 1.0 - self.take_profit_pct
            )
        metadata = _target_metadata(
            strategy=_STRATEGY_NAME,
            target_allocation=alloc,
            max_order_value=self.max_order_value,
            target_mode=target_mode,
            reason="spread_stress_liquidity_provision",
            spread_stress_z=float(z),
            z_entry=float(self.z_entry),
            corwin_schultz_spread=float(item.spreads[-1]) if item.spreads else None,
            fade_return=float(fade_return),
            fade_lookback_bars=int(self.fade_lookback_bars),
            realized_vol=float(vol),
            inverse_vol_scalar=float(size_scalar),
            reversion_target=reversion_target,
        )
        _emit(
            self.events,
            strategy_id=_STRATEGY_ID,
            symbol=symbol,
            event_time=event_time,
            signal_type=target_mode,
            strength=max(0.25, min(3.0, abs(z) / max(self.z_entry, _EPS))),
            price=close,
            stop_loss=stop_loss,
            metadata=metadata,
        )
        item.mode = target_mode
        item.entry_price = close
        item.bars_held = 0


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for a later integration wave (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits per the live-safety
# plan).  Admission route is `allow_multi_asset=True` at the data-PC handoff:
# this is a PER-SYMBOL time-series liquidity-provision reversal on a multi-symbol
# liquid book, NOT a cross-sectional rank book and NOT carry -- so it is honestly
# EXCLUDED from any carry/XS-rank tag allowlist route (no fake family tag added).
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "mean_reversion"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "mean_reversion",
    "liquidity_provision",
    "spread_stress",
    "episodic",
    "per_symbol",
    "corwin_schultz",
    "crypto",
)

_SPREAD_STRESS_REVERSION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": (
        {
            "variant": "strict_gate",
            "cs_smooth_window": 5,
            "z_window_bars": 120,
            "z_entry": 2.0,
            "fade_lookback_bars": 3,
            "min_hold_bars": 5,
            "max_hold_bars": 10,
            "cooldown_bars": 5,
            "vol_window_bars": 30,
            "min_history_bars": 150,
            "allow_short": True,
        },
        {
            "variant": "lenient_gate",
            "cs_smooth_window": 5,
            "z_window_bars": 90,
            "z_entry": 1.75,
            "fade_lookback_bars": 2,
            "min_hold_bars": 4,
            "max_hold_bars": 8,
            "cooldown_bars": 4,
            "vol_window_bars": 24,
            "min_history_bars": 120,
            "allow_short": True,
        },
    ),
}


__all__ = ["SpreadStressLiquidityReversionStrategy"]
