"""Off-session basis dislocation fade sleeve (TradFi perp re-anchoring at cash reopen).

``OffSessionBasisDislocationStrategy`` implements pre-registered proposal
``offsession-basis-dislocation-fade``.  TradFi perps trade 24/7 while the
underlying cash market closes, and the ``index_price`` goes stale off-session --
so mark-index basis accumulated during closed hours is a direct print of
unanchored 24/7 flow pushing the perp off its last anchored price (the
ADR-premium object, computed from data the incumbents ignore).  Per TradFi
symbol the sleeve accumulates the off-session mark-index basis CHANGE (in bps;
off-session = outside UTC ``[14, 20)`` Mon-Fri).  At the last decision bar
before the cash reopen (the last non-ambiguous pre-open hour, UTC hour
``cash_start_hour_utc - 2`` = 12 on Mon-Fri; the flagship window is the
weekend, Friday close -> Sunday night -> Monday pre-open) the accumulated
dislocation is cross-sectionally z-scored; the sleeve **SHORTS** the top
quintile (pumped off-anchor) and **LONGS** the bottom quintile, only when
``|z| > 2``, holds through re-anchoring with a hard 36h min-hold and a fixed
72h max exit, inverse-realized-vol sized with an annualized vol-target clamp.

SESSION-BOUNDARY CONVENTION (mirrored helper).  The DST-safe boundary-hour-drop
convention is copied verbatim from
``strategies/offsession_tugofwar_alpha_sleeves.py`` --
``CrossSectionalOffSessionTugOfWarStrategy._classify_hour`` and its
``_ambiguous_hours = {cash_start-1, cash_start, cash_end}`` frozenset
construction (= ``{13, 14, 20}``): the NYSE open sits at :30, so the
open-straddling hour is 13 under EDT but 14 under EST -- both are dropped, as
is the close hour 20; weekends are fully off-session.  Basis changes across
AMBIGUOUS hours re-anchor without accruing (dropped from the accumulator, the
same symmetric-loss treatment the tug-of-war sleeve applies to its return
components); CASH hours reset the accumulator (each episode measures only the
CURRENT off-session dislocation).

THEORY / PROVENANCE (external prior, exact references from the proposal)
------------------------------------------------------------------------
- Chalmers, Edelen & Kadlec (2001, *JF*): NAV-timing on stale prices -- stale
  anchors are systematically exploitable by trading against the un-anchored leg.
- Gagnon & Karolyi (2010, *JFE*): multi-market arbitrage -- cross-listed price
  gaps to the anchored market close mostly within a day of the home-market open.
- Repo ledger: OffSessionTugOfWar's pivot note documents that the
  session-structure axis is computable on this feed and that off-session prices
  lack a discovery anchor.

EXPECTED NULL (pre-registered, verbatim).  "The mechanical null: off-session
moves in globally traded names (TSM/ASML/BABA class) are genuine
Asia/Europe-hours information, so at reopen the INDEX re-anchors up to the mark
-- the gap closes through the index leg while the perp price never retraces,
and the fade collects nothing (this is OffSessionTugOfWar's own declared
wrong-sign alternative, sharpened to basis space)."

FALSIFYING MEASUREMENT (pre-registered, verbatim).  "Reopen decomposition event
study on the data-PC: for |z|>2 dislocation events, split gap closure over the
first N cash-session hours into the delta-mark vs delta-index contributions.
Kill if the mark-leg share of closure is <= ~30%, or if mark-leg net move after
the 20bps cost grid is < 10bps per event."

NEAREST GRAVEYARD.  Item 1 calendar-primary rules -- pre-empted: the session
calendar only defines the MEASUREMENT window; the trade requires a measured,
thresholded price dislocation (``|z| > 2``) and stays flat when no dislocation
printed, so calendar time alone never generates a position.  Canonical killer
(ii) cost death -- pre-empted by the naturally weekly event cadence, the hard
36h min-hold and the fixed 72h max exit (``intended_hold_seconds`` is stamped
in every entry's metadata for the L-D funding-entry guard).

NEAREST INCUMBENT.  ``CrossSectionalOffSessionTugOfWarStrategy`` -- same
session-decomposition axis, but it is a slow multi-week CHARACTERISTIC on
hourly returns holding a continuous weekly book, and it never touches
mark/index.  This sleeve trades the CURRENT episode's realized dislocation
using the basis against a stale anchor, is event-conditional, and is flat most
of the time.  ``SessionFilteredPairCarry`` only hour-gates pair entries -- a
different object entirely.

UNIVERSE.  TradFi symbols only.  A symbol-suffix filter is NOT reliable, so the
constructor accepts ``tradfi_symbols`` (tuple, default ``()`` = "use all
configured symbols"); **candidate wiring passes the TradFi set** (the
pre-registered ``_DEFAULT_TRADFI_UNIVERSE`` below, equity + ETF/index legs in
compact and slashed forms).  With the TradFi set supplied, a crypto-only
universe intersects to zero eligible symbols and the sleeve deterministically
self-skips; ``min_symbols`` is floored at 4 regardless of configuration.

ADDED PRE-REGISTERED CONSTANT (documented deviation).  Because a cross-sectional
z-score is scale-invariant, a pure ``|z| > 2`` gate could trigger on
spread-noise-sized dislocations; ``min_abs_dislocation_bps`` (10bps, fixed,
non-tunable -- below the 15-25bps round-trip cost floor of brief section 2)
additionally requires the accumulated dislocation itself to exceed
microstructure noise before an entry is allowed.

This module is data-local (no I/O, no hidden configuration bus), pure
Python/stdlib (``math`` only), and never raises from ``calculate_signals``.
Signals fire at completed bar closes; the fixed max-hold exit is a
close-confirmed EXIT signal (never an engine intrabar bracket).  A computed
zero/negative allocation emits NOTHING (v5 zero-alloc resize trap).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import basis_bps, realized_volatility
from lumina_quant.indicators.annualization import (
    annualize_per_bar_vol,
    bars_per_year_from_spacing,
    median_bar_spacing_seconds,
)
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.research_universe import (
    BINANCE_TRADFI_EQUITY_SYMBOLS,
    BINANCE_TRADFI_ETF_INDEX_SYMBOLS,
    compact_to_slashed_usdt,
)
from lumina_quant.strategies.external_alpha_sleeves import (
    _EPS,
    _Snapshot,
    _emit,
    _event_datetime_utc,
    _extract_feature,
    _market_snapshot,
    _safe_non_negative_int,
    _target_metadata,
    _window_snapshot,
)
from lumina_quant.strategies.offsession_tugofwar_alpha_sleeves import (
    _coerce_float_list,
    _cross_z,
)
from lumina_quant.strategy import Strategy
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "offsession_basis_dislocation"
_STRATEGY_NAME = "OffSessionBasisDislocationStrategy"

# PRE-REGISTERED STALENESS MULTIPLE (fixed, non-tunable, documented deviation).
# A symbol whose last accepted bar predates the decision time by more than
# 3x the inferred bar spacing (median spacing of recent decision-path epochs)
# is a halted/frozen leg: its accumulator is a stale photograph of a past
# episode, so it is excluded from cross-sectional scoring and can never be
# entered.  3x tolerates a couple of missed prints on the hourly cadence while
# rejecting anything halted for a session-scale gap.
_STALE_BAR_SPACING_MULTIPLE = 3.0


def _default_tradfi_universe() -> tuple[str, ...]:
    """TradFi equity + ETF/index perp legs, compact AND slashed forms, sorted.

    This is the pre-registered set candidate wiring passes as ``tradfi_symbols``.
    Premarket-only symbols (no cash session) and commodities (~23h futures
    sessions blur the split) are structurally excluded, exactly as in the
    tug-of-war incumbent's universe.
    """
    compact = (*BINANCE_TRADFI_EQUITY_SYMBOLS, *BINANCE_TRADFI_ETF_INDEX_SYMBOLS)
    members: set[str] = set()
    for symbol in compact:
        members.add(symbol)
        try:
            members.add(compact_to_slashed_usdt(symbol))
        except ValueError:
            continue
    return tuple(sorted(members))


_DEFAULT_TRADFI_UNIVERSE: tuple[str, ...] = _default_tradfi_universe()


def _coerce_symbol_tuple(value: Any) -> tuple[str, ...]:
    """Best-effort tuple[str, ...] coercion that never raises on adversarial input."""
    if not isinstance(value, (list, tuple, set, frozenset)):
        return ()
    out: list[str] = []
    for item in value:
        try:
            text = str(item).strip()
        except Exception:
            continue
        if text:
            out.append(text)
    return tuple(out)


@dataclass(slots=True)
class _State:
    # Trailing closes (realized-vol sizing + last price for emission).
    closes: deque[float]
    # Last observed mark-index basis (bps) used as the delta anchor.
    prev_basis_bps: float | None = None
    # Accumulated off-session basis change (bps) since the last CASH reset.
    off_accum_bps: float = 0.0
    # Count of OFF-session deltas accrued since the last CASH reset.
    off_obs: int = 0
    mode: str = "OUT"
    entry_price: float | None = None
    entry_epoch: float | None = None
    last_time_key: str = ""
    score: float | None = None
    # Epoch (seconds) of the last ACCEPTED bar -- staleness gate input.
    last_bar_epoch: float | None = None
    # The most recent cash mark, retained until that UTC cash session completes.
    cash_session_date: str | None = None
    cash_session_basis_bps: float | None = None
    # Session date whose closing cash mark is the active off-session baseline.
    baseline_session_date: str | None = None


@register("strategy", "OffSessionBasisDislocationStrategy", interface="event_driven")
class OffSessionBasisDislocationStrategy(Strategy):
    """Event-conditional XS fade of off-session mark-index basis dislocations.

    See the module docstring for the full theory, the mirrored DST-safe session
    classifier, the pre-registered null/falsifier, and the universe contract.
    Reads only local event/bar data plus the ``mark_price`` / ``index_price``
    feature columns (via the None-tolerant ``_extract_feature`` cascade),
    performs no I/O, and never raises from ``calculate_signals``.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False
    required_features = ("mark_price", "index_price")

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            # Structurally pinned to NYSE cash hours (external market-structure
            # fact); the DST-ambiguous flanking hours {start-1, start, end} are
            # dropped -- mirrored from the tug-of-war incumbent.
            "cash_start_hour_utc": HyperParam.integer(
                "cash_start_hour_utc", default=14, low=0, high=23, tunable=False
            ),
            "cash_end_hour_utc": HyperParam.integer(
                "cash_end_hour_utc", default=20, low=1, high=24, tunable=False
            ),
            "entry_z": HyperParam.floating(
                "entry_z", default=2.0, low=0.0, high=10.0, tunable=False
            ),
            "quantile_pct": HyperParam.floating("quantile_pct", default=0.20, low=0.02, high=0.50),
            "min_symbols": HyperParam.integer("min_symbols", default=4, low=4, high=512),
            "min_off_observations": HyperParam.integer(
                "min_off_observations", default=3, low=1, high=10000, tunable=False
            ),
            "min_abs_dislocation_bps": HyperParam.floating(
                "min_abs_dislocation_bps", default=10.0, low=0.0, high=10000.0, tunable=False
            ),
            "min_hold_hours": HyperParam.integer(
                "min_hold_hours", default=36, low=0, high=100000, tunable=False
            ),
            "max_hold_hours": HyperParam.integer(
                "max_hold_hours", default=72, low=1, high=100000, tunable=False
            ),
            "vol_window": HyperParam.integer("vol_window", default=168, low=2, high=20000),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
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
                "max_order_value", default=400.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        configured = list(getattr(self.bars, "symbol_list", []) or [])
        # Universe contract: ``tradfi_symbols`` (tuple) restricts the tradable
        # set; ``()`` means "use all configured symbols".  Candidate wiring
        # passes the TradFi set (``_DEFAULT_TRADFI_UNIVERSE``), so a crypto-only
        # universe intersects to zero members and the sleeve self-skips.  Both
        # compact and slashed forms of every supplied symbol are accepted.
        self.tradfi_symbols = _coerce_symbol_tuple(params.get("tradfi_symbols", ()))
        if self.tradfi_symbols:
            allowed: set[str] = set()
            for symbol in self.tradfi_symbols:
                allowed.add(symbol)
                try:
                    allowed.add(compact_to_slashed_usdt(symbol))
                except ValueError:
                    continue
            self.symbol_list = [s for s in configured if s in allowed]
        else:
            self.symbol_list = list(configured)
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.cash_start_hour_utc = max(0, min(23, int(resolved["cash_start_hour_utc"])))
        self.cash_end_hour_utc = max(1, min(24, int(resolved["cash_end_hour_utc"])))
        self.entry_z = max(0.0, float(resolved["entry_z"]))
        self.quantile_pct = min(0.5, max(0.0, float(resolved["quantile_pct"])))
        # HARD FLOOR: this cross-sectional book never scores fewer than 4 legs.
        self.min_symbols = max(4, int(resolved["min_symbols"]))
        self.min_off_observations = max(1, int(resolved["min_off_observations"]))
        self.min_abs_dislocation_bps = max(0.0, float(resolved["min_abs_dislocation_bps"]))
        self.min_hold_hours = max(0, int(resolved["min_hold_hours"]))
        self.max_hold_hours = max(self.min_hold_hours, max(1, int(resolved["max_hold_hours"])))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.allow_short = bool(resolved["allow_short"])
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.target_vol = max(0.0, float(resolved["target_vol"]))
        self.base_allocation = max(0.0, float(resolved["base_allocation"]))
        self.max_symbol_exposure_pct = max(0.0, float(resolved["max_symbol_exposure_pct"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        # Ambiguous (DST) hours dropped: {start-1, start, end} -- the exact
        # frozenset construction mirrored from the tug-of-war incumbent.
        self._ambiguous_hours = frozenset(
            {
                (self.cash_start_hour_utc - 1) % 24,
                self.cash_start_hour_utc % 24,
                self.cash_end_hour_utc % 24,
            }
        )
        # Last decision bar before cash reopen: the last NON-AMBIGUOUS pre-open
        # hour (start-1 and start are both dropped by the DST convention).
        self._decision_hour_utc = (self.cash_start_hour_utc - 2) % 24
        closes_size = max(8, self.vol_window + 8)
        self._state: dict[str, _State] = {
            symbol: _State(closes=deque(maxlen=closes_size)) for symbol in self.symbol_list
        }
        self._last_decision_date: str | None = None
        # Recent decision-bar epochs (seconds) for deterministic bar-spacing
        # inference feeding the annualized vol-target clamp.
        self._recent_times: deque[float] = deque(maxlen=16)

    # ------------------------------------------------------------------ #
    # session classification (mirrored from the tug-of-war incumbent)
    # ------------------------------------------------------------------ #
    def _classify_hour(self, dt: Any) -> str:
        """Return ``"CASH"`` / ``"OFF"`` / ``"AMBIGUOUS"`` for a bar's UTC hour."""
        if dt.weekday() >= 5:  # weekend -> fully off-session
            return "OFF"
        hour = dt.hour
        if hour in self._ambiguous_hours:
            return "AMBIGUOUS"
        if self.cash_start_hour_utc <= hour < self.cash_end_hour_utc:
            return "CASH"
        return "OFF"

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        return {
            "last_decision_date": self._last_decision_date,
            "recent_times": list(self._recent_times),
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "prev_basis_bps": item.prev_basis_bps,
                    "off_accum_bps": item.off_accum_bps,
                    "off_obs": int(item.off_obs),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "entry_epoch": item.entry_epoch,
                    "last_time_key": item.last_time_key,
                    "score": item.score,
                    "last_bar_epoch": item.last_bar_epoch,
                    "cash_session_date": item.cash_session_date,
                    "cash_session_basis_bps": item.cash_session_basis_bps,
                    "baseline_session_date": item.baseline_session_date,
                }
                for symbol, item in self._state.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self._recent_times.clear()
        for value in _coerce_float_list(state.get("recent_times"))[
            -int(self._recent_times.maxlen or 0) :
        ]:
            self._recent_times.append(value)
        decision_date = state.get("last_decision_date")
        self._last_decision_date = str(decision_date) if decision_date is not None else None
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
                closes = _coerce_float_list(payload.get("closes"))
                for value in closes[-maxlen:] if maxlen else closes:
                    item.closes.append(value)
                item.prev_basis_bps = safe_float(payload.get("prev_basis_bps"))
                accum = safe_float(payload.get("off_accum_bps"))
                item.off_accum_bps = accum if accum is not None else 0.0
                item.off_obs = _safe_non_negative_int(payload.get("off_obs"))
                mode = str(payload.get("mode", "OUT")).upper()
                item.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
                item.entry_price = safe_float(payload.get("entry_price"))
                item.entry_epoch = safe_float(payload.get("entry_epoch"))
                item.last_time_key = str(payload.get("last_time_key", ""))
                item.score = safe_float(payload.get("score"))
                item.last_bar_epoch = safe_float(payload.get("last_bar_epoch"))
                cash_date = payload.get("cash_session_date")
                item.cash_session_date = str(cash_date) if cash_date is not None else None
                item.cash_session_basis_bps = safe_float(payload.get("cash_session_basis_bps"))
                baseline_date = payload.get("baseline_session_date")
                item.baseline_session_date = (
                    str(baseline_date) if baseline_date is not None else None
                )
            except Exception:
                continue

    def on_warmup_end(self) -> None:
        """One-shot engine hook: drop warmup ghost positions, keep learning.

        The engine suppresses every signal emitted during the warmup region,
        so any position this sleeve believes it entered during warmup never
        existed in the portfolio (ghost-position desynchronization).  Flatten
        the position fields WITHOUT emitting EXITs (there is nothing to exit);
        the learned accumulators and deques -- trailing closes, the basis
        anchor, the off-session accumulator/counter, bar identity, the decision
        clock, and the bar-spacing history -- are preserved verbatim.
        """
        for item in self._state.values():
            item.mode = "OUT"
            item.entry_price = None
            item.entry_epoch = None
            item.score = None

    # ------------------------------------------------------------------ #
    # ingestion
    # ------------------------------------------------------------------ #
    def _age_book(self, event_time: Any) -> None:
        """Panel-wide max-hold aging pass: ANY symbol's bar ages the WHOLE book.

        Runs at the top of the evaluation path, BEFORE the triggering bar is
        ingested.  Every held symbol whose entry age has reached the fixed
        ``max_hold_hours`` ceiling relative to the CURRENT event time gets its
        close-confirmed max-hold EXIT (price = last known close, None-safe) --
        including HALTED symbols that print no further bars of their own.
        """
        dt = _event_datetime_utc(event_time)
        if dt is None:
            return
        now = dt.timestamp()
        max_hold_seconds = self.max_hold_hours * 3600.0
        for symbol, item in self._state.items():
            if item.mode == "OUT" or item.entry_epoch is None:
                continue
            if now - item.entry_epoch < max_hold_seconds:
                continue
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
            item.mode = "OUT"
            item.entry_price = None
            item.entry_epoch = None
            item.score = None

    def _update_symbol(self, symbol: str, snapshot: _Snapshot, event: Any) -> bool:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return False
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_time_key:
            return False
        dt = _event_datetime_utc(snapshot.time)
        if dt is None:
            return False
        item.last_time_key = key
        item.last_bar_epoch = dt.timestamp()
        item.closes.append(close)
        # None-tolerant feature access (8h staleness): the 3-tier cascade from
        # external_alpha_sleeves resolves event attr -> feature store -> bar
        # value and yields None (skip, never raise) when a column is stale.
        mark = _extract_feature(self.bars, event, symbol, "mark_price")
        index = _extract_feature(self.bars, event, symbol, "index_price")
        basis = basis_bps(mark, index)
        cls = self._classify_hour(dt)
        if cls == "CASH":
            # Cache marks while cash is open.  The baseline is deliberately
            # committed once, at the following OFF bar, rather than repeatedly
            # treating every cached cash mark as a new session close.
            session_date = dt.date().isoformat()
            if item.cash_session_date != session_date:
                item.cash_session_date = session_date
                item.cash_session_basis_bps = None
                item.off_accum_bps = 0.0
                item.off_obs = 0
            if basis is not None:
                item.cash_session_basis_bps = basis
            item.prev_basis_bps = basis
        elif basis is None:
            # Stale/missing mark or index: invalidate the anchor so the gap
            # across the outage is DROPPED, never misattributed to one bar.
            item.prev_basis_bps = None
        elif cls == "AMBIGUOUS":
            # Boundary-hour drop: re-anchor without accruing.
            item.prev_basis_bps = basis
        else:  # OFF
            # Promote the final cash mark only after its UTC cash session has
            # completed.  This prevents a pre-open OFF print from assigning a
            # same-date (still incomplete) cash mark as its anchor.
            cash_date = item.cash_session_date
            if (
                cash_date is not None
                and item.cash_session_basis_bps is not None
                and item.baseline_session_date != cash_date
                and (
                    dt.date().isoformat() > cash_date
                    or (dt.date().isoformat() == cash_date and dt.hour >= self.cash_end_hour_utc)
                )
            ):
                item.baseline_session_date = cash_date
                item.prev_basis_bps = item.cash_session_basis_bps
            if item.prev_basis_bps is not None:
                item.off_accum_bps += basis - item.prev_basis_bps
                item.off_obs += 1
            item.prev_basis_bps = basis
        # NOTE: the fixed max-hold exit lives in the panel-wide ``_age_book``
        # pass (top of the evaluation path), NOT here -- a per-symbol check
        # would let a HALTED symbol's position outlive its 72h ceiling.
        return True

    def calculate_signals_window(self, event: Any, aggregator: Any = None) -> None:
        _ = aggregator
        event_time = getattr(event, "time", None)
        event_key = time_key(event_time)
        event_dt = _event_datetime_utc(event_time)
        if not event_key or event_dt is None:
            return
        # Rebalances require a complete panel from this one event timestamp;
        # a cached leg is not an observation at the decision time.
        snapshots: dict[str, _Snapshot] = {}
        for symbol in self.symbol_list:
            snapshot = _window_snapshot(event, symbol)
            if snapshot is None or _event_datetime_utc(snapshot.time) != event_dt:
                return
            snapshots[symbol] = snapshot
        self._age_book(event_time)
        updated = False
        for symbol, snapshot in snapshots.items():
            updated = self._update_symbol(symbol, snapshot, event) or updated
            self._state[symbol].last_time_key = event_key
        if updated:
            self._maybe_evaluate(event_time)

    def calculate_signals(self, event: Any) -> None:
        if str(getattr(event, "type", "")).upper() == "MARKET_WINDOW":
            self.calculate_signals_window(event, None)
            return
        if getattr(event, "type", None) != "MARKET":
            return
        symbol = getattr(event, "symbol", None)
        if symbol in self._state:
            snapshot = _market_snapshot(event)
            if snapshot is None:
                return
            self._age_book(snapshot.time)
            if self._update_symbol(str(symbol), snapshot, event):
                # Individual MARKET updates can learn state but never form a
                # cross-sectional rebalance snapshot.
                return

    # ------------------------------------------------------------------ #
    # pre-reopen decision clock
    # ------------------------------------------------------------------ #
    def _maybe_evaluate(self, event_time: Any) -> None:
        dt = _event_datetime_utc(event_time)
        if dt is None:
            return
        epoch = dt.timestamp()
        if not self._recent_times or epoch > self._recent_times[-1]:
            self._recent_times.append(epoch)
        # One decision per weekday, at the last non-ambiguous pre-open hour.
        if dt.weekday() >= 5 or dt.hour != self._decision_hour_utc:
            return
        date_key = dt.date().isoformat()
        if date_key == self._last_decision_date:
            return
        self._last_decision_date = date_key
        self._evaluate(event_time)

    # ------------------------------------------------------------------ #
    # scoring / selection
    # ------------------------------------------------------------------ #
    def _dislocation_scores(
        self, decision_epoch: float | None = None
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return ``(accumulated_dislocation_bps, vols)`` for eligible symbols.

        Symbols whose last ACCEPTED bar predates ``decision_epoch`` by more
        than ``_STALE_BAR_SPACING_MULTIPLE`` x the inferred bar spacing are
        halted/frozen legs: their accumulator is a stale photograph, so they
        are excluded from scoring (and therefore can never be entered).
        Unknown spacing or an unknown decision time passes through unfiltered.
        """
        stale_after: float | None = None
        if decision_epoch is not None:
            spacing = median_bar_spacing_seconds(self._recent_times)
            if spacing is not None and spacing > 0.0:
                stale_after = _STALE_BAR_SPACING_MULTIPLE * spacing
        scores: dict[str, float] = {}
        vols: dict[str, float] = {}
        for symbol, item in self._state.items():
            if item.off_obs < self.min_off_observations:
                continue
            if (
                stale_after is not None
                and decision_epoch is not None
                and item.last_bar_epoch is not None
                and (decision_epoch - item.last_bar_epoch) > stale_after
            ):
                continue
            vol = realized_volatility(item.closes, window=self.vol_window)
            if vol is None or vol <= _EPS:
                continue
            scores[symbol] = float(item.off_accum_bps)
            vols[symbol] = float(vol)
        return scores, vols

    def _score_and_select(
        self,
        decision_epoch: float | None = None,
    ) -> tuple[dict[str, tuple[str, float, dict[str, Any]]], dict[str, float]]:
        scores, vols = self._dislocation_scores(decision_epoch)
        if len(scores) < self.min_symbols:
            return {}, {}
        z_by_symbol = _cross_z(scores)
        ordered = sorted(z_by_symbol, key=lambda symbol: (z_by_symbol[symbol], symbol))
        count = len(ordered)
        n_side = max(1, int(self.quantile_pct * count))
        if 2 * n_side > count:
            n_side = count // 2
        if n_side < 1:
            return {}, {}
        long_core = ordered[:n_side]
        short_core = ordered[count - n_side :]
        targets: dict[str, tuple[str, float, dict[str, Any]]] = {}
        for symbol in ordered:
            z_value = float(z_by_symbol[symbol])
            dislocation = float(scores[symbol])
            # Event gate: quintile membership AND |z| > entry_z AND an absolute
            # dislocation beyond microstructure noise.  No dislocation -> flat.
            if abs(z_value) <= self.entry_z:
                continue
            if abs(dislocation) < self.min_abs_dislocation_bps:
                continue
            meta = {
                "off_basis_accum_bps": dislocation,
                "dislocation_z": z_value,
                "off_obs": int(self._state[symbol].off_obs),
            }
            if symbol in long_core and z_value < -self.entry_z:
                targets[symbol] = ("LONG", z_value, meta)
            elif self.allow_short and symbol in short_core and z_value > self.entry_z:
                targets[symbol] = ("SHORT", z_value, meta)
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
        # Per-bar portfolio vol annualized via sqrt(bars_per_year) inferred from
        # observed bar spacing BEFORE the Moreira-Muir clamp; unknown spacing ->
        # pass-through (scalar=1.0) rather than throttle on mismatched horizons.
        portfolio_vol = sum((inv[symbol] / total_inv) * vols[symbol] for symbol in inv)
        scalar = 1.0
        if self.target_vol > 0.0 and portfolio_vol > _EPS:
            bars_per_year = bars_per_year_from_spacing(self._recent_times)
            portfolio_vol_ann = annualize_per_bar_vol(portfolio_vol, bars_per_year)
            if portfolio_vol_ann is not None and portfolio_vol_ann > _EPS:
                scalar = min(1.0, self.target_vol / portfolio_vol_ann)
        weights = {
            symbol: (inv[symbol] / total_inv) * self.target_gross_exposure * scalar
            for symbol in inv
        }
        return weights, float(scalar)

    # ------------------------------------------------------------------ #
    # emission
    # ------------------------------------------------------------------ #
    def _evaluate(self, event_time: Any) -> None:
        if len(self.symbol_list) < self.min_symbols:
            return
        decision_dt = _event_datetime_utc(event_time)
        decision_epoch = decision_dt.timestamp() if decision_dt is not None else None
        targets, vols = self._score_and_select(decision_epoch)
        if not targets:
            return
        weights, scalar = self._inverse_vol_weights(targets, vols)
        self._emit_targets(targets, weights, scalar, event_time)

    def _emit_targets(
        self,
        targets: dict[str, tuple[str, float, dict[str, Any]]],
        weights: dict[str, float],
        scalar: float,
        event_time: Any,
    ) -> None:
        decision_dt = _event_datetime_utc(event_time)
        decision_epoch = decision_dt.timestamp() if decision_dt is not None else None
        min_hold_seconds = self.min_hold_hours * 3600.0
        intended_hold_seconds = float(self.max_hold_hours * 3600.0)
        for symbol, item in self._state.items():
            target = targets.get(symbol)
            if target is None:
                # Untargeted holds ride to the fixed max-hold exit -- the book
                # is event-conditional, not continuously rebalanced.
                continue
            target_mode, score, meta = target
            price = item.closes[-1] if item.closes else None
            if item.mode == target_mode:
                # Same side re-selected: keep the position AND its entry clock
                # (refreshing would extend the fixed 72h window indefinitely).
                item.score = float(score)
                continue
            if item.mode != "OUT":
                # Hard 36h min-hold: a side flip inside the hold window is
                # suppressed; the position ages toward its fixed exit instead.
                if (
                    decision_epoch is None
                    or item.entry_epoch is None
                    or (decision_epoch - item.entry_epoch) < min_hold_seconds
                ):
                    continue
                _emit(
                    self.events,
                    strategy_id=_STRATEGY_ID,
                    symbol=symbol,
                    event_time=event_time,
                    signal_type="EXIT",
                    price=price,
                    metadata={"strategy": _STRATEGY_NAME, "reason": "side_flip"},
                )
                item.mode = "OUT"
                item.entry_price = None
                item.entry_epoch = None
                item.score = None
            weight = float(weights.get(symbol, 0.0))
            alloc = max(0.0, self.base_allocation * weight)
            if alloc <= 0.0 or decision_epoch is None:
                # v5 zero-alloc resize trap: a zero/negative computed allocation
                # (or an unusable hold clock) must emit NOTHING -- the engine
                # would otherwise resize the entry to its DEFAULT allocation.
                continue
            metadata = _target_metadata(
                strategy=_STRATEGY_NAME,
                target_allocation=alloc,
                max_order_value=self.max_order_value,
                score=float(score),
                target_mode=target_mode,
                inverse_vol_weight=weight,
                vol_target_scalar=float(scalar),
                # L-D funding-entry guard input: the intended hold (the fixed
                # 72h exit window) is stamped on every entry; >= the hard 36h
                # min-hold by construction.
                intended_hold_seconds=intended_hold_seconds,
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
                strength=max(0.25, min(3.0, abs(float(score)))),
                price=price,
                metadata=metadata,
            )
            item.mode = target_mode
            item.entry_price = price
            item.entry_epoch = decision_epoch
            item.score = float(score)


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the integrator (this lane does NOT wire candidates
# itself -- new-file-only, no shared-file edits).  Admission route is
# ``allow_multi_asset=True`` at the data-PC handoff: this is a pure
# cross-sectional event-conditional long-short book.  Candidate wiring MUST pass
# ``tradfi_symbols`` (done below via the pre-registered default universe) --
# ``()`` would fall back to "all configured symbols" and score crypto legs.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "session_decomposition",
    "basis_dislocation",
    "stale_anchor_fade",
    "tradfi_perp",
    "event_conditional",
    "low_turnover",
)

# Candidate slice keyed by the 1h bar cadence this lane consumes; each variant
# dict is a pre-registered sweep cell counted toward N_eff at the data-PC.
# Constants are FIXED per the proposal: z=2.0, min-hold 36h, max-hold 72h.
_OFFSESSION_BASIS_DISLOCATION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "reopen_fade_z2_q20_36h72h",
            "tradfi_symbols": _DEFAULT_TRADFI_UNIVERSE,
            "entry_z": 2.0,
            "quantile_pct": 0.20,
            "min_symbols": 6,
            "min_off_observations": 3,
            "min_abs_dislocation_bps": 10.0,
            "min_hold_hours": 36,
            "max_hold_hours": 72,
            "vol_window": 168,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
        {
            "variant": "reopen_fade_z2_q25_36h72h",
            "tradfi_symbols": _DEFAULT_TRADFI_UNIVERSE,
            "entry_z": 2.0,
            "quantile_pct": 0.25,
            "min_symbols": 6,
            "min_off_observations": 3,
            "min_abs_dislocation_bps": 10.0,
            "min_hold_hours": 36,
            "max_hold_hours": 72,
            "vol_window": 168,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
    ),
}

__all__ = ["OffSessionBasisDislocationStrategy"]
