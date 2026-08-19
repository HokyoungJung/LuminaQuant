"""Long-only daily rotation: MA-score exposure x inverse-vol parity x vol target.

LINEAGE / ATTRIBUTION.  This module is an INDEPENDENT ADAPTATION of the publicly
described systrader79-style dynamic-asset-allocation template that circulates in
Korean retail-quant educational material as "ipyeongseon score bijung jojeol +
byeondongseong risk parity + mokpyo byeondongseong" (moving-average score
exposure scaling + volatility risk parity + target volatility).  It is NOT a
reproduction of that author's live rules, NOT an endorsement, and carries NO
performance claim: no code, parameters or backtest of the original were
available, only the three-stage structure stated in public writing.  research_only.

WHAT THE PUBLIC SOURCE STATES (the structure adapted here):
  1. a market-timing "score" per asset = the fraction of a set of trailing
     moving averages that price sits above (1.0 = fully risk-on, 0.0 = risk-off),
     used to SCALE that asset's exposure rather than to flip it on/off;
  2. inverse-volatility risk parity to split capital ACROSS assets;
  3. a target-volatility clamp that shrinks any asset whose realized volatility
     exceeds the target.

WHAT IS THE AUTHOR'S (this file's) CHOICE, not the public source's:
  * the exact MA set ``ma_score_windows`` (default 3/5/10/20 daily bars) -- the
    public description gives no canonical window set;
  * ``target_vol_per_bar`` expressed as a PER-BAR (daily) fraction (0.02) rather
    than an annualized number, so no bar-spacing inference is needed;
  * the ``min_score`` floor (0.25) that zeroes a name whose timing score is weak
    instead of holding a token sliver;
  * ``max_weight`` (0.35) per-name concentration cap and ``gross_cap`` (1.0);
  * the ``min_weight_change`` (0.02) hysteresis band and the ``rebalance_bars``
    (5) evaluation stride, both pure turnover control;
  * long-only: the source's score is a de-risking dial, so a zero score means
    FLAT here, never short.

HYPOTHESIS.  Trend persistence (the MA score) and the volatility-scaling effect
are close to orthogonal: multiplying a per-asset timing score by an inverse-vol
risk-parity weight and a per-asset vol-target clamp should deliver a smoother
equity path than either sizing rule alone, because the score de-risks in
drawdowns while risk parity keeps a single loud asset from dominating.

MECHANICS.  Daily bars (``decision_cadence_seconds = 86400``) over the whole
``bars.symbol_list`` book.  Every ``rebalance_bars`` evaluations each symbol with
enough LAGGED history (indicators are computed over ``closes[:-1]``, so the
bar being acted on never feeds its own signal) gets

    score_i = moving_average_score(closes[:-1], windows=ma_score_windows)
    vol_i   = realized per-bar log-return vol over ``vol_window``
    rp_i    = (1/vol_i) / sum_j (1/vol_j)          over ELIGIBLE symbols
    c_i     = min(1, target_vol_per_bar / vol_i)   (1.0 when the target is off)
    w_i     = min(max_weight, gross_cap * rp_i * score_i * c_i)

and ``w_i = 0`` when ``score_i < min_score``.  Because ``sum_j rp_j = 1`` and
both ``score_i`` and ``c_i`` live on ``[0, 1]``, the book's gross exposure can
never exceed ``gross_cap``.  Fewer than ``min_symbols`` eligible names means the
book holds whatever it has rather than rebalancing on a thin cross-section.

The module is data-local (no I/O, no configuration bus), never raises from
``calculate_signals``, and round-trips its full state.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators import moving_average_score, realized_volatility
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
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "ma_score_vol_target_rotation"
_STRATEGY_NAME = "MaScoreVolTargetRotationStrategy"
_DEFAULT_MA_SCORE_WINDOWS = "3,5,10,20"


@dataclass(slots=True)
class _State:
    closes: deque[float]
    weight: float = 0.0
    last_time_key: str = ""


def _parse_windows(raw: Any) -> tuple[int, ...]:
    """Parse a comma-separated MA-window spec into a sorted unique tuple.

    Unparseable / non-positive chunks are dropped; an empty result falls back to
    the module default so the score is never silently disabled by a typo.
    """
    out: set[int] = set()
    for chunk in str(raw or "").split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            value = int(float(text))
        except TypeError, ValueError:
            continue
        if value >= 1:
            out.add(value)
    if not out:
        return tuple(int(x) for x in _DEFAULT_MA_SCORE_WINDOWS.split(","))
    return tuple(sorted(out))


@register("strategy", _STRATEGY_NAME, interface="event_driven")
class MaScoreVolTargetRotationStrategy(Strategy):
    """Long-only rotation sized by MA score x inverse-vol parity x vol target.

    Each rebalance the per-asset target weight is the product of a market-timing
    score (fraction of trailing SMAs below price), an inverse-volatility risk
    parity share of the eligible cross-section, and a per-asset target-volatility
    clamp, capped at ``max_weight``.  Weight changes smaller than
    ``min_weight_change`` are suppressed, so a stable cross-section trades once
    and then holds.
    """

    decision_cadence_seconds = 86400
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "ma_score_windows": HyperParam.string(
                "ma_score_windows",
                default=_DEFAULT_MA_SCORE_WINDOWS,
                description="Comma-separated trailing SMA windows for the timing score.",
            ),
            "vol_window": HyperParam.integer("vol_window", default=20, low=2, high=400),
            "rebalance_bars": HyperParam.integer("rebalance_bars", default=5, low=1, high=250),
            "target_vol_per_bar": HyperParam.floating(
                "target_vol_per_bar", default=0.02, low=0.0, high=1.0
            ),
            "gross_cap": HyperParam.floating("gross_cap", default=1.0, low=0.0, high=3.0),
            "max_weight": HyperParam.floating("max_weight", default=0.35, low=0.0, high=1.0),
            "min_score": HyperParam.floating("min_score", default=0.25, low=0.0, high=1.0),
            "min_weight_change": HyperParam.floating(
                "min_weight_change", default=0.02, low=0.0, high=1.0
            ),
            "min_symbols": HyperParam.integer("min_symbols", default=2, low=1, high=512),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=500.0, low=0.0, high=1_000_000.0, tunable=False
            ),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.ma_score_windows = _parse_windows(resolved["ma_score_windows"])
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.rebalance_bars = max(1, int(resolved["rebalance_bars"]))
        self.target_vol_per_bar = max(0.0, float(resolved["target_vol_per_bar"]))
        self.gross_cap = max(0.0, float(resolved["gross_cap"]))
        self.max_weight = max(0.0, min(1.0, float(resolved["max_weight"])))
        self.min_score = max(0.0, min(1.0, float(resolved["min_score"])))
        self.min_weight_change = max(0.0, float(resolved["min_weight_change"]))
        self.min_symbols = max(1, int(resolved["min_symbols"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        # One extra slot for the lagged read (indicators run on ``closes[:-1]``)
        # and one for the incoming bar, so a FULL deque still satisfies both
        # ``realized_volatility`` (needs vol_window+1 lagged closes) and
        # ``moving_average_score`` (needs the longest MA window).
        size = max(self.vol_window, self.ma_score_windows[-1]) + 2
        self._state = {symbol: _State(closes=deque(maxlen=size)) for symbol in self.symbol_list}
        self._last_eval_time_key = ""
        self._pending_time_key = ""
        self._pending_time: Any = None
        self._pending_count = 0
        self._tick = 0

    # ------------------------------------------------------------------ state

    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "pending_time_key": self._pending_time_key,
            "pending_count": int(self._pending_count),
            "tick": int(self._tick),
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "weight": float(item.weight),
                    "last_time_key": item.last_time_key,
                }
                for symbol, item in self._state.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_eval_time_key = str(state.get("last_eval_time_key", ""))
        self._pending_time_key = str(state.get("pending_time_key", ""))
        self._pending_count = _safe_non_negative_int(state.get("pending_count"))
        self._tick = _safe_non_negative_int(state.get("tick"))
        # ``_pending_time`` is a raw event timestamp and is deliberately NOT
        # persisted (it is not reliably JSON-able); a restored session simply
        # re-learns it from the next print.
        self._pending_time = None
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
                if parsed is not None:
                    item.closes.append(parsed)
            weight = safe_float(payload.get("weight"))
            item.weight = max(0.0, float(weight)) if weight is not None else 0.0
            item.last_time_key = str(payload.get("last_time_key", ""))

    # ------------------------------------------------------------- ingestion

    def _update_symbol(self, symbol: str, snapshot: _Snapshot, key: str) -> bool:
        """Append one close under the DECISION key (same key the quorum counts).

        Per-symbol dedupe and the bar-completion quorum must agree on what "one
        bar" is: a window event stamps every symbol with the window's own time,
        not the last 1s row's time, so they can never drift apart.
        """
        close = safe_float(snapshot.close)
        if close is None or close <= 0.0:
            return False
        item = self._state[symbol]
        if key and key == item.last_time_key:
            return False
        item.last_time_key = key
        item.closes.append(close)
        return True

    def _quorum(self) -> int:
        """Symbols that have ever printed -- the bar is complete once all report.

        Warming symbols with no history do not block the first evaluations, and
        the count grows as the book fills in.
        """
        return max(1, sum(1 for item in self._state.values() if item.closes))

    def _note_update(self, key: str, event_time: Any) -> None:
        """Advance the per-bar quorum and evaluate once the bar is complete."""
        if key and key != self._pending_time_key:
            # A strictly newer bar arrived while the previous one never reached
            # quorum (a symbol stalled or was delisted).  Evaluate the stale bar
            # now so one missing print cannot freeze the book forever.
            if self._pending_time_key and self._pending_time_key != self._last_eval_time_key:
                self._last_eval_time_key = self._pending_time_key
                self._tick += 1
                self._evaluate(self._pending_time)
            self._pending_time_key = key
            self._pending_count = 0
        self._pending_time = event_time
        self._pending_count += 1
        if not key or key == self._last_eval_time_key:
            return
        if self._pending_count >= self._quorum():
            self._last_eval_time_key = key
            self._tick += 1
            self._evaluate(event_time)

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        event_time = getattr(event, "time", None)
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is None:
                continue
            key = time_key(event_time) or time_key(snapshot.time)
            if self._update_symbol(symbol, snapshot, key):
                self._note_update(key, event_time)

    def calculate_signals(self, event: Any) -> None:
        event_type = str(getattr(event, "type", "")).upper()
        if event_type == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if event_type != "MARKET":
            return
        symbol = str(getattr(event, "symbol", ""))
        if symbol not in self._state:
            return
        snapshot = _market_snapshot(event)
        if snapshot is None:
            return
        key = time_key(snapshot.time)
        if self._update_symbol(symbol, snapshot, key):
            self._note_update(key, snapshot.time)

    # ------------------------------------------------------------- rebalance

    def _features(self, item: _State) -> tuple[float, float] | None:
        """Return LAGGED ``(ma_score, per_bar_vol)`` for a symbol, or ``None``.

        Both indicators read ``closes[:-1]``: the bar currently being acted on
        never contributes to the signal that trades it.
        """
        lagged = list(item.closes)[:-1]
        if len(lagged) < max(self.vol_window + 1, self.ma_score_windows[-1]):
            return None
        score = moving_average_score(lagged, windows=self.ma_score_windows)
        if score is None:
            return None
        vol = realized_volatility(lagged, window=self.vol_window)
        if vol is None or vol <= _EPS or not math.isfinite(vol):
            return None
        return float(score), float(vol)

    def _target_weights(self) -> dict[str, tuple[float, float, float, float, float]]:
        """Return ``{symbol: (weight, score, vol, rp_weight, vol_clamp)}``.

        Empty when fewer than ``min_symbols`` names are eligible (the book then
        holds whatever it has rather than rebalancing on a thin cross-section).
        """
        eligible: dict[str, tuple[float, float]] = {}
        for symbol, item in self._state.items():
            features = self._features(item)
            if features is not None:
                eligible[symbol] = features
        if len(eligible) < self.min_symbols:
            return {}
        inv = {symbol: 1.0 / vol for symbol, (_score, vol) in eligible.items()}
        total_inv = sum(inv.values())
        if total_inv <= _EPS:
            return {}
        out: dict[str, tuple[float, float, float, float, float]] = {}
        for symbol, (score, vol) in eligible.items():
            rp = inv[symbol] / total_inv
            clamp = 1.0
            if self.target_vol_per_bar > 0.0:
                clamp = min(1.0, self.target_vol_per_bar / vol)
            weight = 0.0
            if score >= self.min_score:
                weight = min(self.max_weight, self.gross_cap * rp * score * clamp)
            out[symbol] = (max(0.0, weight), score, vol, rp, clamp)
        return out

    def _evaluate(self, event_time: Any) -> None:
        if self.rebalance_bars > 1 and self._tick % self.rebalance_bars != 0:
            return
        targets = self._target_weights()
        if not targets:
            return
        # ponytail: symbols missing from ``targets`` (history gap, degenerate
        # zero vol) keep whatever weight they hold instead of being flattened --
        # a DATA outage is not a trading signal, and force-liquidating on a
        # missing print would be the more destructive default.
        for symbol, (weight, score, vol, rp, clamp) in targets.items():
            item = self._state[symbol]
            current = float(item.weight)
            price = item.closes[-1] if item.closes else None
            if weight <= 0.0:
                if current > 0.0:
                    _emit(
                        self.events,
                        strategy_id=_STRATEGY_ID,
                        symbol=symbol,
                        event_time=event_time,
                        signal_type="EXIT",
                        price=price,
                        metadata={
                            "strategy": _STRATEGY_NAME,
                            "reason": "score_below_floor"
                            if score < self.min_score
                            else "zero_weight",
                            "ma_score": float(score),
                        },
                    )
                    item.weight = 0.0
                continue
            if current > 0.0 and abs(weight - current) < self.min_weight_change:
                continue
            if current > 0.0:
                # The portfolio contract has no partial resize: a second LONG
                # ADDS to the open position and EXIT closes it in full, so a
                # size CHANGE is expressed as EXIT-then-LONG at the new weight.
                # ponytail: this round-trips the whole position for a small
                # resize; the ``min_weight_change`` band is what keeps that from
                # happening on noise.
                _emit(
                    self.events,
                    strategy_id=_STRATEGY_ID,
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=price,
                    metadata={
                        "strategy": _STRATEGY_NAME,
                        "reason": "resize",
                        "previous_weight": current,
                        "target_weight": float(weight),
                    },
                )
            metadata = _target_metadata(
                strategy=_STRATEGY_NAME,
                target_allocation=float(weight),
                max_order_value=self.max_order_value,
                ma_score=float(score),
                vol=float(vol),
                rp_weight=float(rp),
                vol_clamp=float(clamp),
            )
            _emit(
                self.events,
                strategy_id=_STRATEGY_ID,
                symbol=symbol,
                event_time=event_time,
                signal_type="LONG",
                strength=max(0.25, min(3.0, float(score))),
                price=price,
                metadata=metadata,
            )
            item.weight = float(weight)


__all__ = ["MaScoreVolTargetRotationStrategy"]
