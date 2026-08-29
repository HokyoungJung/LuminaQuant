"""Basis-Funding Gap Convergence sleeve (front-running the next funding print).

``BasisFundingGapConvergenceStrategy`` trades the SPREAD between the
instantaneous perp premium and the standing funding print.  Binance funding =
clamp(premium-index TWAP) + interest, settled every 8h -- funding is a LAGGED
TWAP of the instantaneous mark-index premium.  Per symbol the sleeve computes

    ``gap = basis_bps(mark, index) - funding_implied_basis_bps(funding)``

via ``indicators/funding_structure.basis_funding_gap_bps``.  A large POSITIVE
gap means the instantaneous basis is richer than the standing funding: the
NEXT funding print mechanically catches up AND the basis itself mean-reverts,
so a SHORT collects rising funding accrual plus basis convergence; symmetric
on discounts.  At each 8h funding boundary (the ``_FUNDING_CADENCE_SECONDS =
28800`` discipline mirrored from ``CrossSectionalFundingMomentumCarry``), the
gap is CROSS-SECTIONALLY z-scored; a FLAT name enters only when ``|z| >
entry_z`` (pre-registered 2.0), FADING the gap (z > +2 => SHORT, z < -2 =>
LONG).  Entries are inverse-vol sized (risk parity normalized to
``target_gross_exposure``, scaled by an annualized vol-target clamp).  A hard
min-hold of ``min_hold_intervals = 2`` funding intervals (16h) applies before
ANY exit; after it the position exits on a gap SIGN FLIP, and unconditionally
at ``max_hold_intervals = 9`` funding intervals (72h).  Entry metadata stamps
``intended_hold_seconds`` (>= 2 funding intervals) so the sanctioned L-D
funding-entry guard (skip entry when remaining intended hold is shorter than
one funding interval) composes on top.  Exits are close-confirmed EXIT
signals; no engine intrabar brackets are used, and a zero/negative computed
allocation emits NOTHING (v5 zero-alloc resize trap).

External prior (pre-registered): Ackerer, Hugonnier & Jermann, "Perpetual
Futures Pricing" (2023 wp/SSRN): perp premium equals the expectation of future
funding flows, making premium-vs-funding disagreement a convergence object.
Plus the Binance funding specification itself: premium-TWAP construction makes
the gap a near-mechanical predictor of the next funding print (the
deterministic leg of the trade).

EXPECTED NULL (verbatim, pre-registered): "The gap is already priced:
rich-basis names keep drifting up (crowded momentum), so mark-to-market loss
exceeds funding collected + convergence; or post-2025 gaps on liquid perps are
only a few bps wide -- structurally below the 10bps RPT floor."

FALSIFYING MEASUREMENT (verbatim, pre-registered): "Event study on the
data-PC conditioned on |gap z|>2 at funding boundaries: median net PnL per
event of holding the fade for 3 funding intervals at the 20bps cost grid.
Kill if median < 10bps. Sanity pre-check: corr(gap, next-funding minus
current-funding) must be strongly positive -- if not, the feature pipeline is
broken, not the thesis."

Nearest graveyard: canonical killer (ii) cost/turnover death at 20-30bps --
pre-empted ex-ante: entries only at 8h funding boundaries AND only on |z|>2
(~1-2 events/week/name), hard min-hold of 2 funding intervals, and the stamped
``intended_hold_seconds`` composing with the L-D funding-interval entry guard.
Graveyard #16 metals-RV negative evidence does not apply: this is
own-instrument basis-vs-funding mechanical convergence, not a cross-asset RV
pair.

Nearest incumbent: ``FundingDislocationTrendCarry`` uses the basis_bps LEVEL
as a 0.20-weighted component blended with trend, funding level, and crowding.
Orthogonality: this book trades the SPREAD basis-minus-funding (the
funding-expectation error), which is zero whenever basis and funding agree no
matter how extreme both are -- a variable no incumbent computes; and it
carries no trend blend, so it is a pure convergence fade roughly orthogonal to
momentum by construction.  ``PerpCrowdingCarry`` fades a
funding+OI+liquidation crowding composite, again a levels object.

All params are PRE-REGISTERED constants (no tunable grids in the candidate
slices).  Data-local (no I/O), never raises from ``calculate_signals``,
``get_state``/``set_state`` are lossless and adversarially safe.  Ships
research_only (tier hint wired by the integrator, not this module).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import realized_volatility
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.funding_structure import basis_funding_gap_bps
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import _state_size
from lumina_quant.strategies.diversified_multifactor_ensemble import _zscore_across
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _Snapshot,
    _annualize_per_bar_vol,
    _emit,
    _event_datetime_utc,
    _extract_feature,
    _market_snapshot,
    _safe_non_negative_int,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "basis_funding_gap_convergence"
_STRATEGY_NAME = "BasisFundingGapConvergenceStrategy"


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


# 8h funding cadence floor (seconds).  decision_cadence_seconds must be >= this
# (the boundary discipline mirrored from CrossSectionalFundingMomentumCarry).
_FUNDING_CADENCE_SECONDS = 28800


def _is_funding_boundary(dt: datetime | None) -> bool:
    """Whether ``dt`` is an exact UTC 00:00/08:00/16:00 funding boundary."""
    return (
        dt is not None
        and dt.minute == 0
        and dt.second == 0
        and dt.microsecond == 0
        and dt.hour % 8 == 0
    )


@dataclass(slots=True)
class _State:
    closes: deque[float]
    gaps: deque[float]
    mode: str = "OUT"
    entry_price: float | None = None
    entry_gap_sign: int = 0
    intervals_held: int = 0
    last_time_key: str = ""


@register("strategy", _STRATEGY_NAME, interface="event_driven")
class BasisFundingGapConvergenceStrategy(Strategy):
    """Fade the cross-sectional basis-minus-funding gap at 8h funding boundaries.

    Per closed funding interval the per-symbol gap (instantaneous mark-index
    basis minus the funding-implied basis, in bps) is captured via the shared
    3-tier no-lookahead ``_extract_feature`` cascade (all-or-nothing: when any
    of funding/mark/index is absent the symbol simply contributes no entries).
    Gaps are cross-sectionally z-scored; FLAT names enter only on ``|z| >
    entry_z``, fading the gap (rich basis => SHORT, discount => LONG),
    inverse-vol sized with an annualized vol-target clamp.  A hard min-hold of
    ``min_hold_intervals`` funding intervals precedes ANY exit; afterwards the
    position exits on a gap sign flip, and unconditionally at
    ``max_hold_intervals`` (72h at the defaults).  Entry metadata stamps
    ``intended_hold_seconds`` so the L-D funding-entry guard composes.
    """

    decision_cadence_seconds = _FUNDING_CADENCE_SECONDS
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False
    required_features = ("funding_rate", "mark_price", "index_price")

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "entry_z": HyperParam.floating("entry_z", default=2.0, low=0.5, high=10.0),
            "min_hold_intervals": HyperParam.integer(
                "min_hold_intervals", default=2, low=1, high=100
            ),
            "max_hold_intervals": HyperParam.integer(
                "max_hold_intervals", default=9, low=1, high=1000
            ),
            "min_symbols": HyperParam.integer("min_symbols", default=6, low=4, high=512),
            "funding_rate_horizon_seconds": HyperParam.integer(
                "funding_rate_horizon_seconds",
                default=_FUNDING_CADENCE_SECONDS,
                low=1,
                high=31_536_000,
                tunable=False,
            ),
            "funding_rate_unit": HyperParam.categorical(
                "funding_rate_unit", default="decimal", choices=("decimal", "bps")
            ),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "vol_window": HyperParam.integer("vol_window", default=24, low=2, high=400),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=1.0, low=0.0, high=3.0
            ),
            "target_vol": HyperParam.floating("target_vol", default=0.20, low=0.0, high=2.0),
            "base_allocation": HyperParam.floating(
                "base_allocation", default=0.20, low=0.0, high=2.0, tunable=False
            ),
            "max_symbol_exposure_pct": HyperParam.floating(
                "max_symbol_exposure_pct", default=0.40, low=0.0, high=2.0, tunable=False
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
        self.entry_z = max(0.0, float(resolved["entry_z"]))
        self.min_hold_intervals = max(1, int(resolved["min_hold_intervals"]))
        self.max_hold_intervals = max(self.min_hold_intervals, int(resolved["max_hold_intervals"]))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.funding_rate_horizon_seconds = max(1, int(resolved["funding_rate_horizon_seconds"]))
        self.funding_rate_unit = str(resolved["funding_rate_unit"]).lower()
        self.allow_short = bool(resolved["allow_short"])
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.target_vol = max(0.0, float(resolved["target_vol"]))
        self.base_allocation = max(0.0, float(resolved["base_allocation"]))
        self.max_symbol_exposure_pct = max(0.0, float(resolved["max_symbol_exposure_pct"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        # Enforce the 8h funding cadence floor in code (instances cannot decide
        # faster than the underlying funding accrual interval).
        self.decision_cadence_seconds = max(
            int(type(self).decision_cadence_seconds), _FUNDING_CADENCE_SECONDS
        )
        size = _state_size(self.vol_window, self.max_hold_intervals)
        self._state = {
            symbol: _State(closes=deque(maxlen=size), gaps=deque(maxlen=size))
            for symbol in self.symbol_list
        }
        self._last_eval_time_key = ""
        self._tick = 0
        # Recent decision-interval epochs (seconds) for deterministic bar-spacing
        # inference (annualizes the per-interval portfolio vol for the clamp).
        self._recent_times: deque[float] = deque(maxlen=16)

    # ------------------------------------------------------------------ state
    def get_state(self) -> dict[str, Any]:
        return {
            "last_eval_time_key": self._last_eval_time_key,
            "tick": int(self._tick),
            "recent_times": list(self._recent_times),
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "gaps": list(item.gaps),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "entry_gap_sign": int(item.entry_gap_sign),
                    "intervals_held": int(item.intervals_held),
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
        self._recent_times.clear()
        for value in _coerce_float_list(state.get("recent_times"))[
            -int(self._recent_times.maxlen or 0) :
        ]:
            self._recent_times.append(value)
        raw = state.get("symbol_state")
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            for attr in ("closes", "gaps"):
                target = getattr(item, attr)
                target.clear()
                maxlen = int(target.maxlen or 0)
                values = _coerce_float_list(payload.get(attr))
                for value in values[-maxlen:] if maxlen else values:
                    target.append(value)
            mode = str(payload.get("mode", "OUT")).upper()
            item.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
            item.entry_price = safe_float(payload.get("entry_price"))
            sign = safe_float(payload.get("entry_gap_sign"))
            item.entry_gap_sign = (
                0 if sign is None else (1 if sign > 0 else (-1 if sign < 0 else 0))
            )
            item.intervals_held = _safe_non_negative_int(payload.get("intervals_held"))
            item.last_time_key = str(payload.get("last_time_key", ""))

    def on_warmup_end(self) -> None:
        """One-shot engine hook fired at the warmup -> live boundary.

        The engine suppresses signals emitted during warmup, so any position
        this sleeve believes it entered during warmup never existed in the
        portfolio (ghost book).  Reset the per-symbol position/hold-interval
        state to OUT while preserving the accrued closes/gaps history and the
        ``last_time_key`` freshness markers.
        """
        for item in self._state.values():
            item.mode = "OUT"
            item.entry_price = None
            item.entry_gap_sign = 0
            item.intervals_held = 0

    # ------------------------------------------------------------- data intake
    def _extract_feature(self, event: Any, symbol: str, field: str) -> float | None:
        return _extract_feature(self.bars, event, symbol, field)

    def _update_symbol(self, event: Any, symbol: str, snapshot: _Snapshot) -> bool:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return False
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return False
        # All-or-nothing gap capture at the closed interval (3-tier cascade,
        # None-tolerant).  When ANY of funding/mark/index is absent we DROP the
        # bar entirely so this symbol simply contributes no entries.
        funding = self._extract_feature(event, symbol, "funding_rate")
        mark = self._extract_feature(event, symbol, "mark_price")
        index = self._extract_feature(event, symbol, "index_price")
        gap = basis_funding_gap_bps(mark, index, self._normalize_funding_rate(funding))
        if gap is None:
            return False
        item.last_time_key = key
        item.closes.append(close)
        item.gaps.append(float(gap))
        return True

    def _normalize_funding_rate(self, value: Any) -> float | None:
        """Convert a declared funding print into a decimal rate per 8h interval."""
        rate = safe_float(value)
        if rate is None:
            return None
        if self.funding_rate_unit == "bps":
            rate /= 10_000.0
        elif self.funding_rate_unit != "decimal":
            return None
        return rate * (_FUNDING_CADENCE_SECONDS / self.funding_rate_horizon_seconds)

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        event_time = getattr(event, "time", None)
        event_key = time_key(event_time)
        event_dt = _event_datetime_utc(event_time)
        if not event_key or not _is_funding_boundary(event_dt):
            return
        # A cross-sectional decision is valid only for one complete, common
        # timestamped panel.  Never blend cached per-symbol bars into a rebalance.
        snapshots: dict[str, _Snapshot] = {}
        for symbol in self.symbol_list:
            snapshot = _window_snapshot(event, symbol)
            if snapshot is None or _event_datetime_utc(snapshot.time) != event_dt:
                return
            snapshots[symbol] = snapshot
        updated = False
        for symbol, snapshot in snapshots.items():
            updated = self._update_symbol(event, symbol, snapshot) or updated
            # Persist the event's canonical identity after UTC-equivalence
            # admission, so strict freshness cannot depend on ISO spelling.
            self._state[symbol].last_time_key = event_key
        if updated and event_key != self._last_eval_time_key:
            self._last_eval_time_key = event_key
            self._tick += 1
            self._evaluate(event_time, strict_freshness=True)

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
                # MARKET events may extend individual histories, but cannot
                # produce a cross-sectional rebalance: peers are not a common
                # timestamped snapshot.
                self._update_symbol(event, str(symbol), snapshot)

    # -------------------------------------------------------------- evaluation
    def _symbol_is_fresh(
        self, item: _State, eval_key: str, eval_dt: datetime | None, *, strict: bool
    ) -> bool:
        """True when the symbol's last ACCEPTED update is current for this evaluation.

        Batch/window flow (``strict=True``): every live symbol is updated inside
        the very event that triggers the evaluation, so the last accepted key
        must EQUAL the evaluation key.  Per-symbol MARKET flow: the evaluation
        fires on the FIRST symbol of a new interval while the rest still carry
        the previous interval's key, so one decision cadence of tolerance
        applies.  Anything older is a dead feed: frozen closes/gaps must not
        enter the cross-sectional panel (stale-symbol cross-section trap).
        """
        if eval_key and item.last_time_key == eval_key:
            return True
        if strict:
            return False
        if eval_dt is None:
            return False
        last_dt = _event_datetime_utc(item.last_time_key)
        if last_dt is None:
            return False
        return abs((eval_dt - last_dt).total_seconds()) <= float(self.decision_cadence_seconds)

    def _collect(
        self, event_time: Any, *, strict: bool
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Latest gap + realized vol per FRESH symbol (stale/short-history self-skip)."""
        gaps: dict[str, float] = {}
        vols: dict[str, float] = {}
        eval_key = time_key(event_time)
        eval_dt = _event_datetime_utc(event_time)
        for symbol, item in self._state.items():
            if not item.gaps:
                continue
            if not self._symbol_is_fresh(item, eval_key, eval_dt, strict=strict):
                continue
            vol = realized_volatility(item.closes, window=self.vol_window)
            if vol is None or vol <= _EPS:
                continue
            gaps[symbol] = float(item.gaps[-1])
            vols[symbol] = float(vol)
        return gaps, vols

    def _inverse_vol_weights(
        self, symbols: Any, vols: dict[str, float]
    ) -> tuple[dict[str, float], float]:
        """Inverse-vol risk parity weights + annualized portfolio vol-target scalar."""
        inv = {
            symbol: 1.0 / max(vols.get(symbol, 0.0), _EPS)
            for symbol in symbols
            if vols.get(symbol, 0.0) > _EPS
        }
        total_inv = sum(inv.values())
        if total_inv <= _EPS:
            return {}, 1.0
        portfolio_vol = sum((inv[symbol] / total_inv) * vols[symbol] for symbol in inv)
        scalar = 1.0
        if self.target_vol > 0.0 and portfolio_vol > _EPS:
            portfolio_vol_ann = _annualize_per_bar_vol(portfolio_vol, self._recent_times)
            if portfolio_vol_ann is not None and portfolio_vol_ann > _EPS:
                scalar = min(1.0, self.target_vol / portfolio_vol_ann)
        weights = {
            symbol: (inv[symbol] / total_inv) * self.target_gross_exposure * scalar
            for symbol in inv
        }
        return weights, float(scalar)

    def _evaluate(self, event_time: Any, *, strict_freshness: bool) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        dt = _event_datetime_utc(event_time)
        if dt is not None:
            self._recent_times.append(dt.timestamp())
        # Exits run on EVERY funding-interval evaluation BEFORE fresh entries:
        # hard min-hold first, then gap sign flip, then unconditional max-hold.
        self._age(event_time)
        gaps, vols = self._collect(event_time, strict=strict_freshness)
        if len(gaps) < self.min_symbols:
            return
        z = _zscore_across(gaps)
        candidates: dict[str, tuple[str, float]] = {}
        for symbol, score in z.items():
            item = self._state.get(symbol)
            if item is None or item.mode != "OUT":
                # Event-conditional book: held names ride to their exit rules,
                # they are never re-ranked or resized.
                continue
            if score > self.entry_z:
                if self.allow_short:
                    candidates[symbol] = ("SHORT", float(score))
            elif score < -self.entry_z:
                candidates[symbol] = ("LONG", float(score))
        if not candidates:
            return
        weights, scalar = self._inverse_vol_weights(candidates.keys(), vols)
        self._emit_entries(candidates, weights, scalar, event_time)

    def _age(self, event_time: Any) -> None:
        for symbol, item in self._state.items():
            if item.mode == "OUT":
                continue
            price = item.closes[-1] if item.closes else None
            item.intervals_held += 1
            reason = ""
            if item.intervals_held >= self.max_hold_intervals:
                reason = "max_hold"
            elif item.intervals_held >= self.min_hold_intervals:
                # Gap sign flip exit -- only AFTER the hard min-hold.
                gap = item.gaps[-1] if item.gaps else None
                if gap is not None and item.entry_gap_sign != 0 and gap * item.entry_gap_sign < 0.0:
                    reason = "gap_sign_flip"
            if not reason:
                continue
            _emit(
                self.events,
                strategy_id=_STRATEGY_ID,
                symbol=symbol,
                event_time=event_time,
                signal_type="EXIT",
                price=price,
                metadata={"strategy": _STRATEGY_NAME, "reason": reason},
            )
            item.mode = "OUT"
            item.entry_price = None
            item.entry_gap_sign = 0
            item.intervals_held = 0

    def _emit_entries(
        self,
        candidates: dict[str, tuple[str, float]],
        weights: dict[str, float],
        scalar: float,
        event_time: Any,
    ) -> None:
        for symbol, (target_mode, score) in candidates.items():
            item = self._state[symbol]
            weight = float(weights.get(symbol, 0.0))
            alloc = max(0.0, self.base_allocation * weight)
            if alloc <= 0.0:
                # v5 zero-alloc resize trap: a zero/negative computed allocation
                # must emit NOTHING (an unsized entry would be resized to the
                # engine DEFAULT allocation).
                continue
            price = item.closes[-1] if item.closes else None
            gap = item.gaps[-1] if item.gaps else None
            metadata = _target_metadata(
                strategy=_STRATEGY_NAME,
                target_allocation=alloc,
                max_order_value=self.max_order_value,
                gap_bps=float(gap) if gap is not None else None,
                gap_z=float(score),
                target_mode=target_mode,
                inverse_vol_weight=weight,
                vol_target_scalar=float(scalar),
                # >= 2 funding intervals by construction (min_hold_intervals >= 1
                # is clamped and pre-registered at 2): lets the sanctioned L-D
                # funding-entry guard compose on top of this sleeve.
                intended_hold_seconds=int(self.min_hold_intervals * _FUNDING_CADENCE_SECONDS),
                min_hold_intervals=int(self.min_hold_intervals),
                max_hold_intervals=int(self.max_hold_intervals),
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
                strength=max(0.25, min(3.0, abs(score) / max(self.entry_z, _EPS))),
                price=price,
                metadata=metadata,
            )
            item.mode = target_mode
            item.entry_price = price
            item.entry_gap_sign = (
                0 if gap is None else (1 if gap > 0.0 else (-1 if gap < 0.0 else 0))
            )
            item.intervals_held = 0


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the integrator (this lane does NOT wire candidates
# itself -- the orchestrator edits candidate_library/registry afterwards).
# Multi-symbol admission route: `cross_sectional` tag + allow_multi_asset.
# The "carry" tag is honest: the fade side COLLECTS the funding catch-up leg.
# All variants are PRE-REGISTERED constants -- no tunable grids.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "carry",
    "funding",
    "basis",
    "convergence",
    "crypto",
)

_BASIS_FUNDING_GAP_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    # Bar feed timeframes; the in-code cadence floor keeps DECISIONS at the 8h
    # funding boundary regardless of the feed cadence.
    "4h": (
        {
            "variant": "boundary_fade",
            "entry_z": 2.0,
            "min_hold_intervals": 2,
            "max_hold_intervals": 9,
            "min_symbols": 8,
            "allow_short": True,
            "vol_window": 24,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
        },
        {
            "variant": "patient_fade",
            "entry_z": 2.5,
            "min_hold_intervals": 3,
            "max_hold_intervals": 9,
            "min_symbols": 8,
            "allow_short": True,
            "vol_window": 24,
            "target_gross_exposure": 1.0,
            "target_vol": 0.18,
        },
    ),
    "1h": (
        {
            "variant": "boundary_fade",
            "entry_z": 2.0,
            "min_hold_intervals": 2,
            "max_hold_intervals": 9,
            "min_symbols": 8,
            "allow_short": True,
            "vol_window": 48,
            "target_gross_exposure": 1.0,
            "target_vol": 0.20,
        },
    ),
}

__all__ = ["BasisFundingGapConvergenceStrategy"]
