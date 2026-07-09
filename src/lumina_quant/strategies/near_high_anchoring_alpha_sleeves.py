"""Cross-sectional 52-week-high ANCHORING sleeve (pure nearness, decoupled from momentum).

``CrossSectionalNearHighAnchoringStrategy`` ranks the cross-section on the
*pure nearness* of each symbol's completed close to its own trailing high --
``close / rolling_max(high)`` -- and takes a LONG-SHORT book: long the
highest-nearness quantile, short the lowest.  The nearness score carries NO
momentum term and is gated by NO minimum-momentum filter, which is exactly what
separates this sleeve from the incumbent long-only
``NearHighMomentumStrategy`` (whose score BLENDS momentum with distance-to-high
and is gated on ``min_momentum``).

THEORY / PROVENANCE
--------------------
- George & Hwang (2004), *Journal of Finance* -- the 52-week-high ratio
  (``price / 52-week high``) predicts the equity cross-section; traders anchor
  on the salient high and under-react as price approaches it.
- Jia, Simkins, Yan, Zhang & Zhao (2025), *Journal of Banking & Finance*
  (S0378426625002122 / SSRN 5386180) -- the "Nearness 52" measure predicts the
  CRYPTO cross-section, long-short ~ +130bps/week GROSS, and is explicitly NOT
  subsumed by momentum.  Cost-survival of that gross effect is the single thing
  the data-PC must verify; this module only builds the tradeable book.

The counterparty is the anchoring/disposition seller who caps a winner early:
by fading them (going long names pinned to their high, short names stranded far
below), the sleeve harvests the documented under-reaction.

SIGNAL SPEC
-----------
Per completed decision bar (OHLC), per symbol:

1. Append the completed ``close`` and ``high`` (``high`` floored at ``close`` so
   nearness stays in ``(0, 1]``; when ``high`` is missing the ``close`` is used).
2. ``nearness = close / max(high over the effective lookback window)`` where the
   **effective lookback = ``min(high_lookback_bars, bars_available)``** -- a young
   symbol with fewer than the full 52-week window is ADMITTED through its own
   ``max_available`` window rather than dropped, provided it clears the
   per-symbol ``min_history_bars`` floor.  Below that floor the symbol is skipped
   (never-raise), never admitted with a degenerate window.
3. ``nearness_z`` = the CROSS-SECTIONAL z-score of ``nearness`` across all
   eligible symbols this bar (decoupled from any time-series momentum).
4. Rank by nearness; the top ``quantile_pct`` fraction are LONG candidates and
   the bottom ``quantile_pct`` fraction are SHORT candidates.  Sizing is
   inverse-realized-vol risk parity normalised to ``target_gross_exposure`` and
   clamped by a ``target_vol`` portfolio scalar (same convention as the sibling
   cross-sectional books).

Cadence is a slow ``rebalance_bars`` clock (weekly-equivalent) plus a hard
``min_hold_bars`` floor: a would-be flip or exit inside the min-hold window is
suppressed, keeping turnover -- and therefore realised cost -- low.  Stops and
``max_hold_bars`` age every bar independent of the rebalance clock.  The book
self-skips (emits nothing) whenever fewer than ``min_symbols`` names carry
enough history/vol to score.

DISTINCT-FROM (the incumbent this sleeve was built to diverge from)
-------------------------------------------------------------------
``NearHighMomentumStrategy`` (``adaptive_crypto_alpha_sleeves.py``) is
cross-sectional but (i) LONG-ONLY (``max_shorts=0``), (ii) scores
``momentum + max(-near_high_pct, distance)`` and GATES on ``min_momentum`` -- so
a name near its high with weak/negative recent momentum is EXCLUDED there, and
the low-nearness tail is never shorted.  This sleeve ranks on PURE nearness with
no momentum term and no momentum gate, and it SHORTS the low-nearness tail.
Those are the load-bearing divergences the build-gate test pins.

This module is data-local (no I/O, no hidden configuration bus), pure
Python/numpy-free (``math`` only), and never raises from ``calculate_signals``.
It ships WITHOUT ``@register`` (inert until the integration wave wires it).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.indicators.alpha_features import realized_volatility
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import (
    _age_cross_positions,
    _state_size,
)
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

_STRATEGY_ID = "near_high_anchoring"
_STRATEGY_NAME = "CrossSectionalNearHighAnchoringStrategy"


@dataclass(slots=True)
class _State:
    closes: deque[float]
    highs: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    last_time_key: str = ""
    score: float | None = None


def _coerce_float_list(value: Any) -> list[float]:
    """Best-effort ``list[float]`` coercion that never raises on adversarial input."""
    if not isinstance(value, (list, tuple)):
        return []
    out: list[float] = []
    for item in value:
        parsed = safe_float(item)
        if parsed is not None:
            out.append(parsed)
    return out


class CrossSectionalNearHighAnchoringStrategy(Strategy):
    """Long-short cross-sectional 52-week-high anchoring on PURE nearness.

    See the module docstring for the full theory, signal spec, and the
    distinct-from rationale versus the long-only, momentum-gated
    ``NearHighMomentumStrategy`` incumbent.  This class only reads local
    event/bar OHLC; it performs no I/O and never raises from
    ``calculate_signals``.
    """

    # Weekly-cadence cross-sectional book; live-applicable on >= 30-minute bars.
    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            # 52-week-equivalent trailing-high window (bars).  Default ~= 52 weeks
            # of 30-minute bars (52 * 7 * 48); the data-PC sweeps 10/20/30/52wk.
            "high_lookback_bars": HyperParam.integer(
                "high_lookback_bars", default=17472, low=20, high=200000
            ),
            # Per-symbol history floor: below this a symbol is skipped; between
            # this and ``high_lookback_bars`` it is admitted via max_available.
            "min_history_bars": HyperParam.integer(
                "min_history_bars", default=1000, low=5, high=100000
            ),
            "vol_window": HyperParam.integer("vol_window", default=20, low=2, high=2000),
            "quantile_pct": HyperParam.floating("quantile_pct", default=0.25, low=0.02, high=0.50),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=336, low=1, high=100000),
            "min_hold_bars": HyperParam.integer("min_hold_bars", default=168, low=0, high=100000),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "min_symbols": HyperParam.integer("min_symbols", default=4, low=2, high=512),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=1.0, low=0.0, high=3.0
            ),
            "target_vol": HyperParam.floating("target_vol", default=0.20, low=0.0, high=2.0),
            "stop_loss_pct": HyperParam.floating("stop_loss_pct", default=0.10, low=0.0, high=0.50),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=0, low=0, high=200000),
            "base_allocation": HyperParam.floating(
                "base_allocation", default=0.20, low=0.0, high=2.0, tunable=False
            ),
            "max_symbol_exposure_pct": HyperParam.floating(
                "max_symbol_exposure_pct", default=0.40, low=0.0, high=2.0, tunable=False
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
        self.high_lookback_bars = max(3, int(resolved["high_lookback_bars"]))
        self.min_history_bars = max(2, int(resolved["min_history_bars"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.quantile_pct = min(0.5, max(0.0, float(resolved["quantile_pct"])))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.min_hold_bars = max(0, int(resolved["min_hold_bars"]))
        self.allow_short = bool(resolved["allow_short"])
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.target_vol = max(0.0, float(resolved["target_vol"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.max_hold_bars = max(0, int(resolved["max_hold_bars"]))
        self.base_allocation = max(0.0, float(resolved["base_allocation"]))
        self.max_symbol_exposure_pct = max(0.0, float(resolved["max_symbol_exposure_pct"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        # A young symbol never needs more retained bars than the full 52-week
        # window; the vol window and max-hold floors keep short-history symbols
        # scoreable.
        size = _state_size(self.high_lookback_bars, self.vol_window + 1, self.max_hold_bars)
        self._state: dict[str, _State] = {
            symbol: _State(
                closes=deque(maxlen=size),
                highs=deque(maxlen=size),
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
                    "closes": list(item.closes),
                    "highs": list(item.highs),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": int(item.bars_held),
                    "last_time_key": item.last_time_key,
                    "score": item.score,
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
                for attr in ("closes", "highs"):
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
                item.last_time_key = str(payload.get("last_time_key", ""))
                item.score = safe_float(payload.get("score"))
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
        if high is None or high < close:
            high = close
        item.closes.append(close)
        item.highs.append(high)
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
    # scoring / selection
    # ------------------------------------------------------------------ #
    def _score_and_select(
        self,
    ) -> tuple[dict[str, tuple[str, float, dict[str, Any]]], dict[str, float]]:
        nearness: dict[str, float] = {}
        vols: dict[str, float] = {}
        metas: dict[str, dict[str, Any]] = {}
        for symbol, item in self._state.items():
            closes = list(item.closes)
            highs = list(item.highs)
            # Per-symbol min-history floor: below it the symbol is skipped, never
            # admitted with a degenerate window.
            if len(closes) < self.min_history_bars or len(highs) < self.min_history_bars:
                continue
            close = closes[-1]
            if close is None or close <= 0.0:
                continue
            # Effective lookback = min(52wk, max_available): a young symbol is
            # admitted through its own available window rather than dropped.
            eff_lookback = min(self.high_lookback_bars, len(highs))
            trailing_high = max(highs[-eff_lookback:])
            if trailing_high <= _EPS:
                continue
            near = close / trailing_high
            if not math.isfinite(near):
                continue
            vol = realized_volatility(closes, window=self.vol_window)
            if vol is None or vol <= _EPS:
                continue
            nearness[symbol] = float(near)
            vols[symbol] = float(vol)
            metas[symbol] = {
                "nearness": float(near),
                "trailing_high": float(trailing_high),
                "lookback_used": int(eff_lookback),
                "full_lookback": bool(eff_lookback >= self.high_lookback_bars),
            }

        if len(nearness) < self.min_symbols:
            return {}, {}

        # Cross-sectional z-score of nearness (decoupled from time-series momentum).
        values = list(nearness.values())
        count = len(values)
        mean_value = sum(values) / float(count)
        variance = sum((value - mean_value) ** 2 for value in values) / float(max(1, count - 1))
        sigma = variance**0.5
        for symbol, near in nearness.items():
            z = 0.0 if sigma <= _EPS else (near - mean_value) / sigma
            metas[symbol]["nearness_z"] = float(z)

        # Deterministic ascending order by nearness (symbol tiebreak): the top
        # quantile is long, the bottom quantile is short.
        ordered = sorted(nearness, key=lambda symbol: (nearness[symbol], symbol))
        n_side = max(1, int(self.quantile_pct * count))
        if 2 * n_side > count:
            n_side = count // 2
        if n_side < 1:
            return {}, {}
        short_syms = ordered[:n_side]
        long_syms = ordered[-n_side:]

        targets: dict[str, tuple[str, float, dict[str, Any]]] = {}
        for symbol in long_syms:
            targets[symbol] = ("LONG", float(metas[symbol]["nearness_z"]), metas[symbol])
        if self.allow_short:
            for symbol in short_syms:
                if symbol in targets:
                    continue
                targets[symbol] = ("SHORT", float(metas[symbol]["nearness_z"]), metas[symbol])
        return targets, vols

    def _inverse_vol_weights(
        self,
        targets: dict[str, tuple[str, float, dict[str, Any]]],
        vols: dict[str, float],
    ) -> tuple[dict[str, float], float]:
        inv = {
            symbol: 1.0 / max(vols.get(symbol, 0.0), _EPS)
            for symbol in targets
            if vols.get(symbol, 0.0) > _EPS
        }
        total_inv = sum(inv.values())
        if total_inv <= _EPS:
            return {}, 1.0
        portfolio_vol = sum((inv[symbol] / total_inv) * vols[symbol] for symbol in inv)
        scalar = 1.0
        if self.target_vol > 0.0 and portfolio_vol > _EPS:
            scalar = min(1.0, self.target_vol / portfolio_vol)
        weights = {
            symbol: (inv[symbol] / total_inv) * self.target_gross_exposure * scalar
            for symbol in inv
        }
        return weights, float(scalar)

    # ------------------------------------------------------------------ #
    # aging / emission
    # ------------------------------------------------------------------ #
    def _age(self, event_time: Any) -> None:
        max_hold = self.max_hold_bars if self.max_hold_bars > 0 else (1 << 62)
        _age_cross_positions(
            self.events,
            self._state,  # type: ignore[arg-type]
            event_time=event_time,
            strategy_id=_STRATEGY_ID,
            strategy_name=_STRATEGY_NAME,
            stop_loss_pct=self.stop_loss_pct,
            max_hold_bars=max_hold,
        )

    def _evaluate(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        # Stops / max-hold age EVERY bar so a held name is always protected,
        # independent of the slow rebalance clock.
        self._age(event_time)
        if self._tick % self.rebalance_bars:
            return
        targets, vols = self._score_and_select()
        weights, scalar = self._inverse_vol_weights(targets, vols)
        self._emit_targets(targets, weights, scalar, event_time)

    def _emit_targets(
        self,
        targets: dict[str, tuple[str, float, dict[str, Any]]],
        weights: dict[str, float],
        scalar: float,
        event_time: Any,
    ) -> None:
        for symbol, item in self._state.items():
            target = targets.get(symbol)
            price = item.closes[-1] if item.closes else None
            if target is None:
                if item.mode != "OUT":
                    # Min-hold floor: a would-be exit inside the hold window is
                    # suppressed (turnover discipline).
                    if item.bars_held < self.min_hold_bars:
                        continue
                    _emit(
                        self.events,
                        strategy_id=_STRATEGY_ID,
                        symbol=symbol,
                        event_time=event_time,
                        signal_type="EXIT",
                        price=price,
                        metadata={"strategy": _STRATEGY_NAME, "reason": "rank_lapsed"},
                    )
                    item.mode = "OUT"
                    item.entry_price = None
                    item.bars_held = 0
                    item.score = None
                continue
            target_mode, score, meta = target
            if item.mode == target_mode:
                item.score = float(score)
                continue
            # Min-hold floor: a would-be side-flip inside the hold window is
            # suppressed; the current position is kept until the hold clears.
            if item.mode != "OUT" and item.bars_held < self.min_hold_bars:
                continue
            if item.mode != "OUT":
                _emit(
                    self.events,
                    strategy_id=_STRATEGY_ID,
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=price,
                    metadata={"strategy": _STRATEGY_NAME, "reason": "side_flip"},
                )
            weight = float(weights.get(symbol, 0.0))
            alloc = max(0.0, self.base_allocation * weight)
            stop_loss = None
            if price is not None and self.stop_loss_pct > 0.0:
                stop_loss = price * (
                    1.0 - self.stop_loss_pct if target_mode == "LONG" else 1.0 + self.stop_loss_pct
                )
            metadata = _target_metadata(
                strategy=_STRATEGY_NAME,
                target_allocation=alloc,
                max_order_value=self.max_order_value,
                score=float(score),
                target_mode=target_mode,
                inverse_vol_weight=weight,
                vol_target_scalar=float(scalar),
                **meta,
            )
            if self.max_symbol_exposure_pct > 0.0:
                metadata["max_symbol_exposure_pct"] = min(
                    float(metadata.get("max_symbol_exposure_pct", self.max_symbol_exposure_pct)),
                    self.max_symbol_exposure_pct,
                )
            _emit(
                self.events,
                strategy_id=_STRATEGY_ID,
                symbol=symbol,
                event_time=event_time,
                signal_type=target_mode,
                strength=max(0.25, min(3.0, abs(score))),
                price=price,
                stop_loss=stop_loss,
                metadata=metadata,
            )
            item.mode = target_mode
            item.entry_price = price
            item.bars_held = 0
            item.score = float(score)


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the W3 integrator (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits per the live-safety
# plan).  Admission route is `allow_multi_asset=True` at the data-PC handoff:
# this book is a pure cross-sectional long-short (NOT carry, NOT momentum), so
# it is honestly EXCLUDED from any carry/momentum tag-superset allowlist -- no
# fake carry tag is added to game that path.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "near_high",
    "anchoring",
    "long_short",
    "zscore",
    "crypto",
)

__all__ = ["CrossSectionalNearHighAnchoringStrategy"]
