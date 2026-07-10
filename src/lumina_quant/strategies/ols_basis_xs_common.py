"""Shared cross-sectional book engine for the OLS-basis alpha sleeves.

The trend-quality and path-convexity sleeves differ ONLY in the trailing
log-price statistic they rank on (signed-R^2 trend quality vs. orthonormal
curvature); every other moving part -- weekly ISO decision clock, per-symbol
close ingestion, quantile long/short book with an entry/exit hysteresis band, a
hard minimum-hold, inverse-realized-vol sizing, emission, and state
serialization -- is identical.  That common apparatus lives here (built once)
and both lanes subclass :class:`OLSBasisCrossSectionalBook`, supplying a single
``_score_symbol`` hook and their own parameter schema.

Design contract (shared with the merged wave-1 cross-sectional lanes):

- Completed-bar, weekly effective cadence via an internal ``_week_key`` clock
  (decides at most once per ISO week even when fed higher-frequency bars).
- LOW turnover by construction: quantile entry band (e.g. top/bottom 20%) with a
  WIDER hysteresis retention band (e.g. 40%/60%) plus a hard minimum-hold in
  weekly decisions -- a rank flip inside the hold window is suppressed.
- Never raises from ``calculate_signals``; self-skips (emits nothing) when fewer
  than ``min_symbols`` names carry a valid score + realized-vol history.
- Pure Python (``math`` / ``deque`` only); no I/O, no hidden configuration bus.

This module ships WITHOUT ``@register`` and defines no registered strategy on
its own -- it is an inert shared base.  Registration / tier hints / candidate
wiring land atomically in a later integration wave on the concrete lanes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.indicators.alpha_features import realized_volatility
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _emit,
    _event_datetime_utc,
    _event_symbols,
    _market_snapshot,
    _safe_non_negative_int,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy


@dataclass(slots=True)
class _XSBasisState:
    """Per-symbol trailing closes + position / min-hold bookkeeping."""

    closes: deque[float]
    mode: str = "OUT"  # OUT / LONG / SHORT
    entry_price: float | None = None
    bars_held: int = 0  # decision (weekly) bars in the CURRENT position
    last_time_key: str = ""  # dedup identical ingested bars
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


class OLSBasisCrossSectionalBook(Strategy):
    """Weekly cross-sectional long/short book over a trailing log-price statistic.

    Subclasses set the shared configuration attributes and call
    :meth:`_finalize_state` at the end of ``__init__``, then implement
    :meth:`_score_symbol`.  See the module docstring for the full contract.
    """

    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    # --- attributes every subclass must set before _finalize_state() ---
    _strategy_id: str = "ols_basis_xs"
    _strategy_name: str = "OLSBasisCrossSectionalBook"
    _window: int = 56
    min_symbols: int = 5
    quantile_entry: float = 0.20
    quantile_exit: float = 0.40
    min_hold_decisions: int = 4
    max_hold_decisions: int = 10_000
    vol_window: int = 30
    target_gross_exposure: float = 1.0
    max_order_value: float = 400.0
    min_price: float = 0.10
    allow_short: bool = True
    require_sign: bool = False

    # ------------------------------------------------------------------ #
    # scoring hook
    # ------------------------------------------------------------------ #
    def _score_symbol(self, closes: list[float]) -> tuple[float, bool] | None:
        """Return ``(score, tradeable)`` for a symbol, or ``None`` to skip it.

        ``score`` is the cross-sectional ranking statistic; ``tradeable`` gates
        whether the symbol may enter either book (e.g. a quality floor) while it
        still counts toward ``min_symbols``.  Must never raise.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def _finalize_state(self) -> None:
        """Build per-symbol state; call at the end of the subclass ``__init__``."""
        history = max(self._window, self.vol_window) + 8
        self._state: dict[str, _XSBasisState] = {
            symbol: _XSBasisState(closes=deque(maxlen=history)) for symbol in self.symbol_list
        }
        self._last_eval_week = ""

    # ------------------------------------------------------------------ #
    # cadence
    # ------------------------------------------------------------------ #
    def _week_key(self, raw_time: Any) -> str:
        """Bucket a bar timestamp into an ISO ``YYYY-Wnn`` weekly decision key."""
        dt = _event_datetime_utc(raw_time)
        if dt is None:
            return time_key(raw_time)
        iso = dt.isocalendar()
        return f"{int(iso[0]):04d}-W{int(iso[1]):02d}"

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_week": self._last_eval_week,
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
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
        self._last_eval_week = str(state.get("last_eval_week", ""))
        raw = state.get("symbol_state")
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            try:
                target = item.closes
                target.clear()
                maxlen = int(target.maxlen or 0)
                values = _coerce_float_list(payload.get("closes"))
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
    def _update_symbol(self, symbol: str, snapshot: Any) -> bool:
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
        updated = False
        for symbol in _event_symbols(event, self.symbol_list):
            snapshot = _window_snapshot(event, symbol)
            if snapshot is not None and self._update_symbol(symbol, snapshot):
                updated = True
        if updated:
            self._maybe_decide(getattr(event, "time", None))

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
                self._maybe_decide(snapshot.time)

    def _maybe_decide(self, event_time: Any) -> None:
        week = self._week_key(event_time)
        if week and week != self._last_eval_week:
            self._last_eval_week = week
            self._evaluate(event_time)

    # ------------------------------------------------------------------ #
    # decision
    # ------------------------------------------------------------------ #
    def _evaluate(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        # Age every held position once per weekly decision BEFORE selection so the
        # hard min-hold / max-hold windows are measured in decision bars.
        for item in self._state.values():
            if item.mode in {"LONG", "SHORT"}:
                item.bars_held += 1

        scored: list[tuple[float, str, bool, float]] = []
        for symbol, item in self._state.items():
            closes = list(item.closes)
            if len(closes) < self._window:
                continue
            result = self._score_symbol(closes)
            if result is None:
                continue
            score, tradeable = result
            if score is None:
                continue
            vol = realized_volatility(closes, window=self.vol_window)
            if vol is None or vol <= _EPS:
                continue
            scored.append((float(score), symbol, bool(tradeable), float(vol)))

        if len(scored) < self.min_symbols:
            self._enforce_max_hold(event_time)
            return

        desired, vols = self._select_book(scored)
        self._emit_targets(desired, vols, event_time)

    def _select_book(
        self, scored: list[tuple[float, str, bool, float]]
    ) -> tuple[dict[str, str | None], dict[str, float]]:
        ranked = sorted(scored, key=lambda row: row[0], reverse=True)
        n = len(ranked)
        n_enter = max(1, int(n * self.quantile_entry))
        n_hold = max(1, int(n * self.quantile_exit))

        def _long_ok(row: tuple[float, str, bool, float]) -> bool:
            return row[2] and (not self.require_sign or row[0] > 0.0)

        def _short_ok(row: tuple[float, str, bool, float]) -> bool:
            return row[2] and (not self.require_sign or row[0] < 0.0)

        long_entry = {row[1] for row in ranked[:n_enter] if _long_ok(row)}
        long_hold = {row[1] for row in ranked[:n_hold] if _long_ok(row)}
        short_entry: set[str] = set()
        short_hold: set[str] = set()
        if self.allow_short:
            short_entry = {row[1] for row in ranked[n - n_enter :] if _short_ok(row)}
            short_hold = {row[1] for row in ranked[n - n_hold :] if _short_ok(row)}

        score_by = {row[1]: row[0] for row in ranked}
        vols = {row[1]: row[3] for row in ranked}
        desired: dict[str, str | None] = {}
        for symbol, item in self._state.items():
            cur = item.mode
            if cur in {"LONG", "SHORT"} and item.bars_held >= self.max_hold_decisions:
                desired[symbol] = None
                continue
            if cur in {"LONG", "SHORT"} and item.bars_held < self.min_hold_decisions:
                # Hard min-hold: a would-be flip inside the window is suppressed.
                desired[symbol] = cur
                continue
            if cur == "LONG":
                desired[symbol] = "LONG" if symbol in long_hold else None
            elif cur == "SHORT":
                desired[symbol] = "SHORT" if symbol in short_hold else None
            elif symbol in long_entry:
                desired[symbol] = "LONG"
            elif symbol in short_entry:
                desired[symbol] = "SHORT"
            else:
                desired[symbol] = None
            if desired[symbol] is not None and symbol in score_by:
                item.score = float(score_by[symbol])
        return desired, vols

    def _enforce_max_hold(self, event_time: Any) -> None:
        """Exit only positions past ``max_hold_decisions`` when the book self-skips."""
        for symbol, item in self._state.items():
            if item.mode in {"LONG", "SHORT"} and item.bars_held >= self.max_hold_decisions:
                self._emit_exit(symbol, item, event_time, reason="max_hold")

    # ------------------------------------------------------------------ #
    # sizing / emission
    # ------------------------------------------------------------------ #
    def _inverse_vol_weights(
        self, book: dict[str, str], vols: dict[str, float]
    ) -> dict[str, float]:
        inv = {
            symbol: 1.0 / max(vols.get(symbol, 0.0), _EPS)
            for symbol in book
            if vols.get(symbol, 0.0) > _EPS
        }
        total = sum(inv.values())
        if total <= _EPS:
            return {}
        return {symbol: (inv[symbol] / total) * self.target_gross_exposure for symbol in inv}

    def _emit_targets(
        self,
        desired: dict[str, str | None],
        vols: dict[str, float],
        event_time: Any,
    ) -> None:
        book = {symbol: mode for symbol, mode in desired.items() if mode in {"LONG", "SHORT"}}
        weights = self._inverse_vol_weights(book, vols)
        for symbol, item in self._state.items():
            target_mode = desired.get(symbol)
            price = item.closes[-1] if item.closes else None
            if target_mode not in {"LONG", "SHORT"}:
                if item.mode != "OUT":
                    self._emit_exit(symbol, item, event_time, reason="rebalance_removed")
                continue
            if item.mode == target_mode:
                continue
            if item.mode != "OUT":
                _emit(
                    self.events,
                    strategy_id=self._strategy_id,
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=price,
                    metadata={"strategy": self._strategy_name, "reason": "side_flip"},
                )
            weight = float(weights.get(symbol, 0.0))
            score = float(item.score) if item.score is not None else 0.0
            alloc = max(0.0, weight)
            if alloc <= 0.0:
                # Zero-alloc entries omit ``target_allocation`` from metadata and
                # the engine resizes them to its DEFAULT allocation -- an unsized,
                # un-vol-gated position. Skip the entry; if a side-flip EXIT was
                # just emitted, drop to OUT (mirroring ``_emit_exit``) so state
                # matches it.
                if item.mode != "OUT":
                    item.mode = "OUT"
                    item.entry_price = None
                    item.bars_held = 0
                    item.score = None
                continue
            metadata = _target_metadata(
                strategy=self._strategy_name,
                target_allocation=alloc,
                max_order_value=self.max_order_value,
                score=score,
                target_mode=target_mode,
                inverse_vol_weight=weight,
            )
            _emit(
                self.events,
                strategy_id=self._strategy_id,
                symbol=symbol,
                event_time=event_time,
                signal_type=target_mode,
                strength=max(0.25, min(3.0, abs(score))),
                price=price,
                metadata=metadata,
            )
            item.mode = target_mode
            item.entry_price = price
            item.bars_held = 0

    def _emit_exit(self, symbol: str, item: _XSBasisState, event_time: Any, *, reason: str) -> None:
        price = item.closes[-1] if item.closes else None
        _emit(
            self.events,
            strategy_id=self._strategy_id,
            symbol=symbol,
            event_time=event_time,
            signal_type="EXIT",
            price=price,
            metadata={"strategy": self._strategy_name, "reason": reason},
        )
        item.mode = "OUT"
        item.entry_price = None
        item.bars_held = 0
        item.score = None


__all__ = ["OLSBasisCrossSectionalBook", "_XSBasisState"]
