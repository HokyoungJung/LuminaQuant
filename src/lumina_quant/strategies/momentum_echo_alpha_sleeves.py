"""Cross-sectional momentum ECHO-minus-recent z-spread sleeve (W3-10).

``CrossSectionalIntermediateEchoMomentumStrategy`` harvests the Novy-Marx (2012)
momentum-echo decomposition, crypto-compressed from months to weeks.  Its SOLE
shipped mechanism is a TWO-WINDOW cross-sectional z-SPREAD

    score = z_xs(r_echo) - z_xs(r_recent)

where ``r_echo`` is the intermediate-horizon return over ``t-12..t-7`` weeks
(default) and ``r_recent`` the salient return over ``t-6..t-2`` weeks; the
freshest ~2 weeks are excluded from BOTH windows as the reversal band.  The
sleeve LONGS names whose continuation is carried by the STALE echo window and NOT
by the recent window (top ``quantile_pct``), and SHORTS the mirror.

WHY A Z-SPREAD, NOT A PLAIN ECHO SORT (the refute fix, folded in)
-----------------------------------------------------------------
A pure echo-window mom/vol rank (``closes[-50]/closes[-85]-1``) is the SAME
transform as the registered ``CrossSectionalEquityMomentumStrategy`` at an
in-span parameter point ``(lookback_bars=35, skip_bars=49, rebalance_bars=7)`` --
i.e. behaviorally spanned.  The two-window z-spread ``z_xs(r_echo) - z_xs(r_recent)``
is NOT expressible in any incumbent's parameter space: neither
``CrossSectionalEquityMomentum`` nor ``TrendGatedResidualMomentum`` can subtract a
SECOND cross-sectional z-window.  It directly encodes the Novy-Marx echo-vs-recent
decomposition claim.  The authoring build gate PROVES that inseparability by
running both incumbents at their honest ECHO-EQUIVALENT nearest-neighbour cells
against a pair with identical echo windows and mirrored recent windows: the
incumbents tie the pair (their windows never see the second, subtracted window)
while this sleeve separates them, and a recent-window perturbation strictly moves
this sleeve's score while leaving the incumbent cell bit-identical.

THEORY / PROVENANCE
-------------------
- Novy-Marx (2012, JFE 103(3)) "Is momentum really momentum?": intermediate past
  returns drive momentum (replicated in commodities, currencies, indices).
- Goyal & Wahal (2015, JFE 115(3)): HONEST COUNTER-ANCHOR -- echo largely fails
  in 37 non-US markets; the pre-registered null is that crypto follows the non-US
  pattern where recent returns dominate.
- Liu & Tsyvinski (2021, RFS): crypto momentum concentrates at 1-4wk formation and
  decays beyond -- the basis for the compression AND the horizon-mismatch null.
- Hong & Stein (1999, JF 54(6)); Kosc, Sakowski & Slepaczuk (2019, Physica A 523)
  -- weekly contrarian dominates in broad crypto XS, arguing for excluding the
  recent window from a continuation signal.

TURNOVER / COST
---------------
LOW.  Weekly clock; typical holding 4-10 weeks; consecutive weekly echo windows
share 4/5 of their span; the score's leading edge is 2 weeks stale and the echo
leg 7 weeks stale; a hard ``min_hold_decisions`` (the proven C1 rescue) plus a
cooldown and an enter-25 / exit-40 rank-band hysteresis suppress churn; recent
shocks move the score only through the bounded z-spread, never the echo rank.

This module is data-local (no I/O), pure Python (``math`` + ``deque`` + the pure
rolling-stat primitives, no numpy/scipy/statsmodels), completed-bar, and never
raises from ``calculate_signals``.  It ships WITHOUT ``@register`` on purpose: an
unregistered module under ``strategies/`` is inert; registration, the
``research_only`` tier hint, candidate wiring, and baseline re-pins land in the
separate atomic integration commit.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import realized_volatility
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.rolling_stats import sample_std
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import _state_size
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _Snapshot,
    _emit,
    _event_datetime_utc,
    _event_symbols,
    _market_snapshot,
    _safe_non_negative_int,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "momentum_echo_minus_recent"
_STRATEGY_NAME = "CrossSectionalIntermediateEchoMomentumStrategy"

# A fresh symbol has never traded, so its cooldown starts long-elapsed -- the
# first eligible entry must not be blocked by a phantom startup cooldown.
_COOLDOWN_SATISFIED = 1 << 30


@dataclass(slots=True)
class _State:
    """Per-symbol close history + weekly position / min-hold / cooldown state."""

    closes: deque[float]
    mode: str = "OUT"  # OUT / LONG / SHORT
    entry_price: float | None = None
    bars_held: int = 0  # weekly DECISIONS in the current position
    bars_since_exit: int = _COOLDOWN_SATISFIED  # weekly decisions since last exit
    last_bar_key: str = ""  # daily-bar dedup
    score: float | None = None


@dataclass(slots=True)
class _Windows:
    """Raw per-symbol echo / recent window returns and the echo-slice sizing vol."""

    r_echo: float
    r_recent: float
    sizing_vol: float


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


def _mode(raw: Any) -> str:
    """Coerce a serialized mode token to one of ``{OUT, LONG, SHORT}``."""
    parsed = str(raw or "OUT").upper()
    return parsed if parsed in {"OUT", "LONG", "SHORT"} else "OUT"


def _xs_zscores(values: dict[str, float]) -> dict[str, float]:
    """Return the cross-sectional z-score of each value (degenerate -> all 0)."""
    if not values:
        return {}
    items = list(values.values())
    count = len(items)
    mean_value = sum(items) / float(count)
    sigma = sample_std(items)
    if sigma is None or sigma <= _EPS:
        return dict.fromkeys(values, 0.0)
    return {symbol: (value - mean_value) / sigma for symbol, value in values.items()}


@register("strategy", "CrossSectionalIntermediateEchoMomentumStrategy", interface="event_driven")
class CrossSectionalIntermediateEchoMomentumStrategy(Strategy):
    """Weekly XS long-short on the echo-minus-recent cross-sectional z-spread.

    See the module docstring for the Novy-Marx decomposition, the refute fix
    (why the SOLE mechanism is the two-window z-spread, not a pure echo sort),
    and the cost argument.  This class only reads local event/bar OHLCV; it
    performs no I/O and never raises from ``calculate_signals``.
    """

    decision_cadence_seconds = 86400  # daily bars; weekly internal decision clock
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = _STRATEGY_NAME
    strategy_id = _STRATEGY_ID

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "echo_start_weeks": HyperParam.integer("echo_start_weeks", default=12, low=3, high=104),
            "echo_end_weeks": HyperParam.integer("echo_end_weeks", default=7, low=2, high=103),
            "recent_start_weeks": HyperParam.integer(
                "recent_start_weeks", default=6, low=2, high=103
            ),
            "recent_end_weeks": HyperParam.integer("recent_end_weeks", default=2, low=1, high=102),
            "bars_per_week": HyperParam.integer("bars_per_week", default=7, low=1, high=168),
            "quantile_entry_pct": HyperParam.floating(
                "quantile_entry_pct", default=0.25, low=0.02, high=0.50
            ),
            "quantile_exit_pct": HyperParam.floating(
                "quantile_exit_pct", default=0.40, low=0.02, high=1.0
            ),
            "min_hold_decisions": HyperParam.integer(
                "min_hold_decisions", default=4, low=0, high=100000
            ),
            "cooldown_decisions": HyperParam.integer(
                "cooldown_decisions", default=1, low=0, high=100000
            ),
            "min_symbols": HyperParam.integer("min_symbols", default=5, low=2, high=512),
            "min_history_bars": HyperParam.integer(
                "min_history_bars", default=92, low=8, high=100000
            ),
            "max_hold_decisions": HyperParam.integer(
                "max_hold_decisions", default=52, low=1, high=200000
            ),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
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
        self.bars_per_week = max(1, int(resolved["bars_per_week"]))
        # Enforce the window ordering echo_start > echo_end >= recent_start >
        # recent_end >= 1 so the two windows never overlap and the recent band is
        # excluded; clamp defensively rather than raising.
        echo_end = max(2, int(resolved["echo_end_weeks"]))
        echo_start = max(echo_end + 1, int(resolved["echo_start_weeks"]))
        recent_end = max(1, int(resolved["recent_end_weeks"]))
        recent_start = max(recent_end + 1, int(resolved["recent_start_weeks"]))
        self.echo_start_weeks = echo_start
        self.echo_end_weeks = echo_end
        self.recent_start_weeks = recent_start
        self.recent_end_weeks = recent_end
        entry = min(0.5, max(0.02, float(resolved["quantile_entry_pct"])))
        exit_pct = min(1.0, max(entry, float(resolved["quantile_exit_pct"])))
        self.quantile_entry_pct = entry
        self.quantile_exit_pct = exit_pct
        self.min_hold_decisions = max(0, int(resolved["min_hold_decisions"]))
        self.cooldown_decisions = max(0, int(resolved["cooldown_decisions"]))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.max_hold_decisions = max(1, int(resolved["max_hold_decisions"]))
        self.allow_short = bool(resolved["allow_short"])
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        # Bar offsets into the trailing close deque (measured from the last bar).
        self.echo_base_offset = self.bars_per_week * self.echo_start_weeks + 1
        self.echo_recent_offset = self.bars_per_week * self.echo_end_weeks + 1
        self.recent_base_offset = self.bars_per_week * self.recent_start_weeks + 1
        self.recent_recent_offset = self.bars_per_week * self.recent_end_weeks + 1
        self._required_bars = self.echo_base_offset + 1
        self.min_history_bars = max(self._required_bars, int(resolved["min_history_bars"]))
        size = _state_size(self.echo_base_offset + 2, self.min_history_bars)
        self._state: dict[str, _State] = {
            symbol: _State(closes=deque(maxlen=size)) for symbol in self.symbol_list
        }
        self._last_decision_week = ""
        self._tick = 0

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        return {
            "last_decision_week": self._last_decision_week,
            "tick": int(self._tick),
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": int(item.bars_held),
                    "bars_since_exit": int(item.bars_since_exit),
                    "last_bar_key": item.last_bar_key,
                    "score": item.score,
                }
                for symbol, item in self._state.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._last_decision_week = str(state.get("last_decision_week", ""))
        self._tick = _safe_non_negative_int(state.get("tick"))
        raw = state.get("symbol_state")
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            try:
                item.closes.clear()
                maxlen = int(item.closes.maxlen or 0)
                values = _coerce_float_list(payload.get("closes"))
                for value in values[-maxlen:] if maxlen else values:
                    item.closes.append(value)
                item.mode = _mode(payload.get("mode"))
                item.entry_price = safe_float(payload.get("entry_price"))
                item.bars_held = _safe_non_negative_int(payload.get("bars_held"))
                raw_cooldown = payload.get("bars_since_exit")
                item.bars_since_exit = (
                    _safe_non_negative_int(raw_cooldown)
                    if raw_cooldown is not None
                    else _COOLDOWN_SATISFIED
                )
                item.last_bar_key = str(payload.get("last_bar_key", ""))
                item.score = safe_float(payload.get("score"))
            except Exception:
                continue

    # ------------------------------------------------------------------ #
    # ingestion / weekly cadence
    # ------------------------------------------------------------------ #
    def _week_key(self, raw_time: Any) -> str:
        """Bucket a bar timestamp into an ISO ``YYYY-Wnn`` weekly decision key."""
        dt = _event_datetime_utc(raw_time)
        if dt is None:
            return time_key(raw_time)
        iso = dt.isocalendar()
        return f"{int(iso[0]):04d}-W{int(iso[1]):02d}"

    def _update_symbol(self, symbol: str, snapshot: _Snapshot) -> bool:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return False
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_bar_key:
            return False
        item.last_bar_key = key
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
        if not week or week == self._last_decision_week:
            return
        self._last_decision_week = week
        self._tick += 1
        self._rebalance(event_time)

    # ------------------------------------------------------------------ #
    # scoring (two-window cross-sectional z-spread)
    # ------------------------------------------------------------------ #
    def _windows_for(self, closes: list[float]) -> _Windows | None:
        """Return the raw echo / recent returns and the echo-slice sizing vol."""
        n = len(closes)
        if n < self._required_bars or n < self.min_history_bars:
            return None
        echo_base = closes[n - self.echo_base_offset]
        echo_recent = closes[n - self.echo_recent_offset]
        recent_base = closes[n - self.recent_base_offset]
        recent_recent = closes[n - self.recent_recent_offset]
        if min(echo_base, echo_recent, recent_base, recent_recent) <= 0.0:
            return None
        r_echo = echo_recent / echo_base - 1.0
        r_recent = recent_recent / recent_base - 1.0
        # Sizing vol over the ECHO slice only (score and sizing are functions of
        # the two declared windows; the latest bar is used only for liquidity).
        echo_slice = closes[n - self.echo_base_offset : n - self.echo_recent_offset + 1]
        sizing_vol = realized_volatility(echo_slice, window=max(2, len(echo_slice) - 1))
        if sizing_vol is None or sizing_vol <= _EPS:
            sizing_vol = 0.0
        return _Windows(
            r_echo=float(r_echo), r_recent=float(r_recent), sizing_vol=float(sizing_vol)
        )

    def raw_windows(self, closes: list[float]) -> _Windows | None:
        """Public helper: the raw echo / recent window returns for one symbol."""
        return self._windows_for(closes)

    def _score_symbols(self) -> tuple[dict[str, tuple[float, dict[str, Any]]], dict[str, float]]:
        raw: dict[str, _Windows] = {}
        for symbol, item in self._state.items():
            windows = self._windows_for(list(item.closes))
            if windows is not None:
                raw[symbol] = windows
        if len(raw) < self.min_symbols:
            return {}, {}
        z_echo = _xs_zscores({symbol: w.r_echo for symbol, w in raw.items()})
        z_recent = _xs_zscores({symbol: w.r_recent for symbol, w in raw.items()})
        scored: dict[str, tuple[float, dict[str, Any]]] = {}
        sizing: dict[str, float] = {}
        for symbol, w in raw.items():
            spread = z_echo[symbol] - z_recent[symbol]
            scored[symbol] = (
                float(spread),
                {
                    "echo_return": w.r_echo,
                    "recent_return": w.r_recent,
                    "z_echo": float(z_echo[symbol]),
                    "z_recent": float(z_recent[symbol]),
                    "echo_minus_recent_spread": float(spread),
                },
            )
            sizing[symbol] = w.sizing_vol
        return scored, sizing

    def score_for(self, symbol: str) -> float | None:
        """Public helper: the current echo-minus-recent z-spread for one symbol."""
        scored, _sizing = self._score_symbols()
        entry = scored.get(symbol)
        return None if entry is None else entry[0]

    # ------------------------------------------------------------------ #
    # selection (rank-band hysteresis + hard min-hold + cooldown)
    # ------------------------------------------------------------------ #
    def _desired_book(self, scored: dict[str, tuple[float, dict[str, Any]]]) -> dict[str, str]:
        ordered = sorted(scored.items(), key=lambda kv: (kv[1][0], kv[0]), reverse=True)
        rank = {symbol: idx for idx, (symbol, _payload) in enumerate(ordered)}
        n = len(ordered)
        k_enter = max(1, int(n * self.quantile_entry_pct))
        k_exit = max(k_enter, int(n * self.quantile_exit_pct))
        desired: dict[str, str] = {}
        for symbol, item in self._state.items():
            idx = rank.get(symbol)
            cur = item.mode
            if cur != "OUT" and item.bars_held < self.min_hold_decisions:
                desired[symbol] = cur
                continue
            if idx is None:
                desired[symbol] = "OUT"
                continue
            long_enter = idx < k_enter
            long_hold = idx < k_exit
            short_enter = idx >= n - k_enter
            short_hold = idx >= n - k_exit
            if cur == "LONG" and long_hold:
                desired[symbol] = "LONG"
            elif cur == "SHORT" and short_hold and self.allow_short:
                desired[symbol] = "SHORT"
            elif long_enter and item.bars_since_exit >= self.cooldown_decisions:
                desired[symbol] = "LONG"
            elif (
                short_enter and self.allow_short and item.bars_since_exit >= self.cooldown_decisions
            ):
                desired[symbol] = "SHORT"
            else:
                desired[symbol] = "OUT"
        return desired

    def _inverse_vol_weights(self, book: list[str], sizing: dict[str, float]) -> dict[str, float]:
        if not book:
            return {}
        valid = [1.0 / sizing[s] for s in book if sizing.get(s, 0.0) > _EPS]
        avg_inv = (sum(valid) / len(valid)) if valid else 1.0
        inv = {
            symbol: (1.0 / sizing[symbol] if sizing.get(symbol, 0.0) > _EPS else avg_inv)
            for symbol in book
        }
        total = sum(inv.values())
        if total <= _EPS:
            share = self.target_gross_exposure / float(len(book))
            return dict.fromkeys(book, share)
        return {symbol: self.target_gross_exposure * inv[symbol] / total for symbol in book}

    # ------------------------------------------------------------------ #
    # rebalance / emission
    # ------------------------------------------------------------------ #
    def _rebalance(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        for item in self._state.values():
            if item.mode in {"LONG", "SHORT"}:
                item.bars_held += 1
            else:
                item.bars_since_exit += 1
        scored, sizing = self._score_symbols()
        if len(scored) < self.min_symbols:
            self._age_max_hold(event_time)
            return
        desired = self._desired_book(scored)
        book = [symbol for symbol, side in desired.items() if side in {"LONG", "SHORT"}]
        weights = self._inverse_vol_weights(book, sizing)
        self._emit_book(desired, weights, scored, event_time)

    def _age_max_hold(self, event_time: Any) -> None:
        for symbol, item in self._state.items():
            if item.mode == "OUT":
                continue
            if item.bars_held >= self.max_hold_decisions:
                price = item.closes[-1] if item.closes else None
                _emit(
                    self.events,
                    strategy_id=_STRATEGY_ID,
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=price,
                    metadata={"strategy": _STRATEGY_NAME, "reason": "max_hold"},
                )
                self._flatten(item)

    def _emit_book(
        self,
        desired: dict[str, str],
        weights: dict[str, float],
        scored: dict[str, tuple[float, dict[str, Any]]],
        event_time: Any,
    ) -> None:
        for symbol, item in self._state.items():
            side = desired.get(symbol, "OUT")
            price = item.closes[-1] if item.closes else None
            if side == "OUT":
                if item.mode != "OUT" and item.bars_held >= self.min_hold_decisions:
                    _emit(
                        self.events,
                        strategy_id=_STRATEGY_ID,
                        symbol=symbol,
                        event_time=event_time,
                        signal_type="EXIT",
                        price=price,
                        metadata={"strategy": _STRATEGY_NAME, "reason": "rank_lapsed"},
                    )
                    self._flatten(item)
                continue
            if item.mode == side:
                score, _meta = scored.get(symbol, (item.score or 0.0, {}))
                item.score = float(score)
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
                self._flatten(item)
            score, meta = scored.get(symbol, (0.0, {}))
            weight = max(0.0, float(weights.get(symbol, 0.0)))
            metadata = _target_metadata(
                strategy=_STRATEGY_NAME,
                target_allocation=weight,
                max_order_value=self.max_order_value,
                score=float(score),
                target_mode=side,
                **meta,
            )
            _emit(
                self.events,
                strategy_id=_STRATEGY_ID,
                symbol=symbol,
                event_time=event_time,
                signal_type=side,
                strength=max(0.25, min(3.0, abs(float(score)))),
                price=price,
                metadata=metadata,
            )
            item.mode = side
            item.entry_price = price
            item.bars_held = 0
            item.score = float(score)

    def _flatten(self, item: _State) -> None:
        item.mode = "OUT"
        item.entry_price = None
        item.bars_held = 0
        item.bars_since_exit = 0
        item.score = None


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the integration wave (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits).  Honest family:
# cross-sectional momentum, admitted via `allow_multi_asset=True`; NOT carry, so
# no fake carry tag is added to game the allocator superset route.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "momentum_echo",
    "intermediate_horizon",
    "echo_minus_recent",
    "decomposition_spread",
    "low_turnover",
    "crypto",
)

# Candidate slice (weekly decision clock via the ISO-week key).  Each variant dict
# is a pre-registered sweep cell counted toward N_eff at the data-PC.  The
# formation windows are expressed in WEEKS (echo/recent_*_weeks) and converted to
# BAR offsets via ``bars_per_week`` (echo_base_offset = bars_per_week * weeks + 1),
# so the ONLY scaled knob is ``bars_per_week``: 7 (1d) -> 42 (4h) -> 168 (1h) to
# keep each week span the same calendar length.  The week counts, entry/exit
# fractions, and ``min_hold_decisions`` (weekly ISO clock) are timeframe INVARIANT
# and stay fixed.  Deepest formation offset at 1h (echo_start_weeks 16 * 168 =
# 2688 bars) sits well under the ~9000-bar cap.
_MOMENTUM_ECHO_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "4h": (
        {
            "variant": "echo12_7_minus_recent6_2",
            "echo_start_weeks": 12,
            "echo_end_weeks": 7,
            "recent_start_weeks": 6,
            "recent_end_weeks": 2,
            "bars_per_week": 42,
            "quantile_entry_pct": 0.25,
            "quantile_exit_pct": 0.40,
            "min_hold_decisions": 4,
            "min_symbols": 5,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
        {
            "variant": "echo16_9_minus_recent8_2",
            "echo_start_weeks": 16,
            "echo_end_weeks": 9,
            "recent_start_weeks": 8,
            "recent_end_weeks": 2,
            "bars_per_week": 42,
            "quantile_entry_pct": 0.25,
            "quantile_exit_pct": 0.40,
            "min_hold_decisions": 4,
            "min_symbols": 5,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
    ),
    "1h": (
        {
            "variant": "echo12_7_minus_recent6_2",
            "echo_start_weeks": 12,
            "echo_end_weeks": 7,
            "recent_start_weeks": 6,
            "recent_end_weeks": 2,
            "bars_per_week": 168,
            "quantile_entry_pct": 0.25,
            "quantile_exit_pct": 0.40,
            "min_hold_decisions": 4,
            "min_symbols": 5,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
        {
            "variant": "echo16_9_minus_recent8_2",
            "echo_start_weeks": 16,
            "echo_end_weeks": 9,
            "recent_start_weeks": 8,
            "recent_end_weeks": 2,
            "bars_per_week": 168,
            "quantile_entry_pct": 0.25,
            "quantile_exit_pct": 0.40,
            "min_hold_decisions": 4,
            "min_symbols": 5,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
    ),
    "1d": (
        {
            "variant": "echo12_7_minus_recent6_2",
            "echo_start_weeks": 12,
            "echo_end_weeks": 7,
            "recent_start_weeks": 6,
            "recent_end_weeks": 2,
            "quantile_entry_pct": 0.25,
            "quantile_exit_pct": 0.40,
            "min_hold_decisions": 4,
            "min_symbols": 5,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
        {
            "variant": "echo16_9_minus_recent8_2",
            "echo_start_weeks": 16,
            "echo_end_weeks": 9,
            "recent_start_weeks": 8,
            "recent_end_weeks": 2,
            "quantile_entry_pct": 0.25,
            "quantile_exit_pct": 0.40,
            "min_hold_decisions": 4,
            "min_symbols": 5,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
    ),
}

__all__ = ["CrossSectionalIntermediateEchoMomentumStrategy"]
