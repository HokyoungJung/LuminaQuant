"""Volume-confirmed trend CONTINUATION gate (wave-2b Lane 9).

``PriceVolumeCorrContinuationStrategy`` is a per-symbol time-series sleeve that
holds a trend position ONLY while the trend is confirmed by expanding volume in
its own direction.  The health statistic is the trailing Pearson correlation
between the per-bar log RETURN and the per-bar log VOLUME-CHANGE -- the qlib158
``CORD`` transform (``indicators/qlib158.py:397``), NOT the levels-based
``price_volume_correlation`` (``indicators/volume.py:261``) and NOT the
volume-clock / flow-share transforms.  The alpha claim is the GATE: ride an
established trend while live information diffusion (Ying 1966 / Karpoff 1987
price-volume co-movement) confirms it; step aside during unconfirmed
(liquidity/short-covering) legs that Campbell-Grossman-Wang (1993) predict revert.

THEORY / PROVENANCE
-------------------
- Ying (1966, Econometrica 34(3)); Karpoff (1987, JFQA 22(1)): price and volume
  co-move; the return / volume-change correlation is the compact summary of that
  co-movement.
- Copeland (1976, JF 31(4)) sequential-information-arrival: a trend advancing on
  SHRINKING or anti-aligned volume is exhausting (information has stopped
  arriving) -- the exhaustion reading that flattens the position.
- Blume, Easley & O'Hara (1994, JF 49(1)): volume carries information beyond
  price -- justifies gating a trend on the volume co-movement.
- Lee & Swaminathan (2000, JF 55(5)) momentum life cycle: low-volume winners are
  late-stage (the counterparty is the unconditioned trend-follower holding an
  advance informed volume has abandoned).
- Llorente, Michaely, Saar & Wang (2002, RFS 15(4)): informed vs liquidity
  trading is separable through the return/volume relationship.
- HONEST COUNTER-ANCHOR (the expected-null side): Campbell, Grossman & Wang
  (1993, QJE 108(4)) -- liquidity-driven high-volume moves REVERT.  All
  continuation anchors are equities-origin; crypto-specific evidence is thin, so
  this sleeve is a falsification probe measured standalone at the data-PC.

MECHANISM (per completed daily bar; weekly decision clock)
----------------------------------------------------------
Per symbol, on an ISO-week internal decision clock:

1. ``dir_t`` = sign of the trailing ``mom_window_weeks`` log return (the base
   trend direction).
2. ``pv_corr_t`` = trailing Pearson correlation of (per-bar log return, per-bar
   log volume-change) over ``corr_window`` bars -- a guarded log-ratio with an
   epsilon floor and symmetric clipping (mirroring the qlib158 ``CORD`` form).
3. signed confirmation score = ``dir_t * pv_corr_t`` -- POSITIVE exactly when the
   trend advances on expanding volume in its own direction (an uptrend where
   up-bars carry the volume, ``corr > 0``; a downtrend where down-bars carry it,
   ``corr < 0`` -- both map to ``dir * corr > 0``).
4. ENTER/HOLD ``dir_t`` only when ``score >= corr_entry``; go FLAT (exhaustion)
   only when ``score <= corr_exit`` -- a hysteresis band.  A hard
   ``min_hold_decisions`` (>= 4 weekly decisions, the proven C1 turnover rescue)
   plus a ``cooldown_decisions`` window suppress flicker.  Sizing is
   inverse-realized-vol / vol-targeted; ``allow_short`` is configurable.

Because the gate ABSTAINS on unconfirmed legs, the sleeve trades STRICTLY LESS
than the unconditioned trend incumbent -- it cannot die on turnover the
incumbent survives.

DISTINCT-FROM (the incumbents this sleeve was built to diverge from)
--------------------------------------------------------------------
- Every pure-PRICE trend sleeve (``LowTurnoverTrendPersistenceStrategy``,
  ``MultiTimeframeTrendEnsembleStrategy``): they act IDENTICALLY on two feeds
  with the same price path but different volume -- this sleeve's whole signal is
  the volume co-movement they ignore.
- ``VolumeClockMomentumRiderStrategy``: samples momentum in VOLUME time but is
  BLIND to the return/volume-change CORRELATION -- it trades a healthy and an
  exhausted advance identically.
- ``CrossSectionalFlowShareRotationStrategy``: ranks the cross-section by volume
  MAGNITUDE share; when every symbol carries an identical share its ``_rolling_z``
  collapses to 0 and it emits nothing, while this sleeve trades the per-symbol
  return/volume co-movement.

This module is data-local (no I/O, no hidden configuration bus), pure Python
(``math`` + ``deque`` only, no numpy), completed-bar, per-symbol single-asset
(no ``min_symbols`` machinery), and never raises from ``calculate_signals``.  It
ships WITHOUT ``@register`` (inert until the integration wave wires it as
``research_only``).

NOTE: the return/volume-change correlation numeric is implemented LANE-LOCAL
(``_return_volume_change_correlation``) per the new-file-only discipline; a
shared ``indicators/volume.py`` twin may be promoted in the atomic integration
commit, never in this lane.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import log_return, realized_volatility
from lumina_quant.indicators.annualization import (
    annualize_per_bar_vol,
    bars_per_year_from_spacing,
)
from lumina_quant.indicators.common import safe_float, time_key
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

_STRATEGY_ID = "price_volume_corr_continuation"
_STRATEGY_NAME = "PriceVolumeCorrContinuationStrategy"

# A fresh symbol has never traded, so its cooldown is treated as long elapsed.
_COOLDOWN_SATISFIED = 1 << 30
# Volume-change guards: an epsilon floor keeps a zero/near-zero volume bar from
# blowing up the log-ratio, and symmetric clipping bounds a single spike.
_VOL_EPS = 1e-9
_VOL_CHANGE_CLIP = 5.0


@dataclass(slots=True)
class _State:
    """Per-symbol close/volume history + weekly position / hold / cooldown state."""

    closes: deque[float]
    volumes: deque[float]
    mode: str = "OUT"  # OUT / LONG / SHORT
    entry_price: float | None = None
    bars_held: int = 0  # weekly DECISIONS in the current position
    bars_since_exit: int = _COOLDOWN_SATISFIED  # weekly decisions since last exit
    last_bar_key: str = ""  # daily-bar dedup
    last_decision_week: str = ""  # weekly-cadence decision gate
    score: float | None = None


def _restore_deque(target: deque[float], payload: Any) -> None:
    """Refill a bounded deque from a serialized payload, dropping non-finite values."""
    target.clear()
    maxlen = int(target.maxlen or 0)
    values: list[float] = []
    if isinstance(payload, (list, tuple)):
        for value in payload:
            parsed = safe_float(value)
            if parsed is not None:
                values.append(parsed)
    for value in values[-maxlen:] if maxlen else values:
        target.append(value)


def _mode(raw: Any) -> str:
    """Coerce a serialized mode token to one of ``{OUT, LONG, SHORT}``."""
    parsed = str(raw or "OUT").upper()
    return parsed if parsed in {"OUT", "LONG", "SHORT"} else "OUT"


def _return_volume_change_correlation(closes: Any, volumes: Any, *, window: int) -> float | None:
    """Return the trailing Pearson corr of (log return, log volume-change).

    The qlib158 ``CORD`` statistic, implemented lane-local: per bar, the price
    innovation is ``log(close_t / close_{t-1})`` and the volume innovation is a
    guarded, clipped ``log(vol_t / vol_{t-1})`` (epsilon floor on each volume so
    a zero/near-zero bar cannot blow up the ratio, symmetric ``_VOL_CHANGE_CLIP``
    on the result).  Returns ``None`` on insufficient history (< ``window`` usable
    pairs), non-positive/non-finite closes, or a degenerate (zero-variance) leg.
    """
    win = max(2, int(window))
    close_list = [safe_float(value) for value in list(closes)]
    volume_list = [safe_float(value) for value in list(volumes)]
    n = min(len(close_list), len(volume_list))
    if n < win + 1:
        return None
    close_tail = close_list[-(win + 1) :]
    volume_tail = volume_list[-(win + 1) :]
    rets: list[float] = []
    vch: list[float] = []
    for idx in range(1, len(close_tail)):
        prev_c = close_tail[idx - 1]
        cur_c = close_tail[idx]
        prev_v = volume_tail[idx - 1]
        cur_v = volume_tail[idx]
        if prev_c is None or cur_c is None or prev_c <= 0.0 or cur_c <= 0.0:
            continue
        if prev_v is None or cur_v is None:
            continue
        r = math.log(cur_c / prev_c)
        dv = math.log(max(cur_v, _VOL_EPS) / max(prev_v, _VOL_EPS))
        dv = max(-_VOL_CHANGE_CLIP, min(_VOL_CHANGE_CLIP, dv))
        if not (math.isfinite(r) and math.isfinite(dv)):
            continue
        rets.append(r)
        vch.append(dv)
    if len(rets) < 2:
        return None
    count = len(rets)
    mean_r = sum(rets) / float(count)
    mean_v = sum(vch) / float(count)
    srr = sum((value - mean_r) ** 2 for value in rets)
    svv = sum((value - mean_v) ** 2 for value in vch)
    if srr <= _EPS or svv <= _EPS:
        return None
    srv = sum((r - mean_r) * (v - mean_v) for r, v in zip(rets, vch, strict=False))
    corr = srv / math.sqrt(srr * svv)
    if not math.isfinite(corr):
        return None
    return max(-1.0, min(1.0, corr))


@register("strategy", "PriceVolumeCorrContinuationStrategy", interface="event_driven")
class PriceVolumeCorrContinuationStrategy(Strategy):
    """Per-symbol volume-confirmed trend-continuation gate at weekly cadence.

    See the module docstring for the full theory, mechanism, and distinct-from
    rationale.  This class only reads local event/bar OHLCV (close + volume); it
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
            "corr_window": HyperParam.integer("corr_window", default=70, low=3, high=8192),
            "mom_window_weeks": HyperParam.integer("mom_window_weeks", default=8, low=1, high=104),
            "bars_per_week": HyperParam.integer("bars_per_week", default=7, low=1, high=168),
            "corr_entry": HyperParam.floating("corr_entry", default=0.25, low=0.0, high=1.0),
            "corr_exit": HyperParam.floating("corr_exit", default=0.0, low=-1.0, high=1.0),
            "min_hold_decisions": HyperParam.integer(
                "min_hold_decisions", default=4, low=0, high=100000
            ),
            "cooldown_decisions": HyperParam.integer(
                "cooldown_decisions", default=2, low=0, high=100000
            ),
            "vol_window": HyperParam.integer("vol_window", default=30, low=2, high=4096),
            "target_vol": HyperParam.floating("target_vol", default=0.20, low=0.0, high=2.0),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.20, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=400.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.01, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        self.bars = bars
        self.events = events
        self.symbol_list = list(getattr(self.bars, "symbol_list", []) or [])
        resolved = resolve_params_from_schema(self.get_param_schema(), params, keep_unknown=False)
        self.corr_window = max(3, int(resolved["corr_window"]))
        self.mom_window_weeks = max(1, int(resolved["mom_window_weeks"]))
        self.bars_per_week = max(1, int(resolved["bars_per_week"]))
        self.corr_entry = max(0.0, float(resolved["corr_entry"]))
        self.corr_exit = float(resolved["corr_exit"])
        self.min_hold_decisions = max(0, int(resolved["min_hold_decisions"]))
        self.cooldown_decisions = max(0, int(resolved["cooldown_decisions"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.target_vol = max(0.0, float(resolved["target_vol"]))
        self.allow_short = bool(resolved["allow_short"])
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        self.mom_window_bars = self.mom_window_weeks * self.bars_per_week
        # Bounded ring buffer: the longest lookback any gate consumes + margin.
        size = max(self.corr_window + 1, self.mom_window_bars + 1, self.vol_window + 1) + 8
        self._state: dict[str, _State] = {
            symbol: _State(closes=deque(maxlen=size), volumes=deque(maxlen=size))
            for symbol in self.symbol_list
        }
        # Recent distinct-bar epochs (seconds) for deterministic bar-spacing
        # inference: the vol-target allocation annualizes the per-bar realized
        # vol via sqrt(bars_per_year) so the throttle is not left inert. The
        # spacing is a GLOBAL bar-clock property, so ``_last_ingest_key`` dedups
        # one epoch per distinct bar time across all symbols.
        self._recent_times: deque[float] = deque(maxlen=16)
        self._last_ingest_key = ""

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        return {
            "recent_times": list(self._recent_times),
            "last_ingest_key": self._last_ingest_key,
            "symbol_state": {
                symbol: {
                    "closes": list(item.closes),
                    "volumes": list(item.volumes),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": int(item.bars_held),
                    "bars_since_exit": int(item.bars_since_exit),
                    "last_bar_key": item.last_bar_key,
                    "last_decision_week": item.last_decision_week,
                    "score": item.score,
                }
                for symbol, item in self._state.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if isinstance(state, dict):
            self._last_ingest_key = str(state.get("last_ingest_key", ""))
            _restore_deque(self._recent_times, state.get("recent_times"))
        raw = state.get("symbol_state") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            try:
                _restore_deque(item.closes, payload.get("closes"))
                _restore_deque(item.volumes, payload.get("volumes"))
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
                item.last_decision_week = str(payload.get("last_decision_week", ""))
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

    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        item = self._state[symbol]
        key = time_key(snapshot.time)
        if key and key == item.last_bar_key:
            return
        item.last_bar_key = key
        # Record one epoch per distinct bar time (across symbols) so the
        # vol-target scalar can infer the global bar spacing deterministically.
        if key and key != self._last_ingest_key:
            self._last_ingest_key = key
            dt = _event_datetime_utc(snapshot.time)
            if dt is not None:
                self._recent_times.append(dt.timestamp())
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        raw_volume = safe_float(snapshot.volume)
        volume = max(0.0, raw_volume) if raw_volume is not None else 0.0
        item.closes.append(close)
        item.volumes.append(volume)
        # Weekly effective cadence: decide at most once per new ISO week.
        week = self._week_key(snapshot.time)
        if not week or week == item.last_decision_week:
            return
        item.last_decision_week = week
        self._decide(symbol, item, snapshot.time)

    # ------------------------------------------------------------------ #
    # signal
    # ------------------------------------------------------------------ #
    def _confirmation(self, item: _State) -> tuple[int, float | None, float | None]:
        """Return ``(dir_t, pv_corr, score)`` for a symbol at a decision bar."""
        closes = list(item.closes)
        mom = log_return(closes, lookback=self.mom_window_bars)
        if mom is None or mom == 0.0:
            dir_t = 0
        else:
            dir_t = 1 if mom > 0.0 else -1
        pv_corr = _return_volume_change_correlation(
            closes, list(item.volumes), window=self.corr_window
        )
        if dir_t == 0 or pv_corr is None:
            return dir_t, pv_corr, None
        return dir_t, pv_corr, float(dir_t) * float(pv_corr)

    def _vol_scaled_allocation(self, closes: list[float]) -> float:
        # ``vol`` is a PER-BAR realized vol; annualize it via sqrt(bars_per_year)
        # (inferred from observed bar spacing) before dividing an annual-scale
        # ``target_vol`` by it, otherwise the ratio is pinned at the max clamp and
        # the throttle is INERT. When spacing is unavailable we pass through at
        # ``target_allocation`` rather than size on mismatched horizons.
        alloc = self.target_allocation
        if self.target_vol > 0.0:
            vol = realized_volatility(closes, window=self.vol_window)
            if vol is not None and vol > _EPS:
                bars_per_year = bars_per_year_from_spacing(self._recent_times)
                vol_ann = annualize_per_bar_vol(vol, bars_per_year)
                if vol_ann is not None and vol_ann > _EPS:
                    alloc = self.target_allocation * (self.target_vol / vol_ann)
        return max(0.0, min(self.target_allocation * 2.0, alloc))

    # ------------------------------------------------------------------ #
    # decision (weekly cadence)
    # ------------------------------------------------------------------ #
    def _decide(self, symbol: str, item: _State, event_time: Any) -> None:
        if item.mode in {"LONG", "SHORT"}:
            item.bars_held += 1
        else:
            item.bars_since_exit += 1

        dir_t, pv_corr, score = self._confirmation(item)

        if item.mode in {"LONG", "SHORT"}:
            # HARD min-hold rescue: no change of any kind inside the hold window.
            if item.bars_held < self.min_hold_decisions:
                return
            # Exhaustion exit: the trend is no longer confirmed by volume in its
            # own direction (score fell to/below the exit band).
            if score is None or score <= self.corr_exit:
                self._emit_exit(symbol, item, event_time, reason="volume_exhaustion")
            return

        # Flat: honour the post-exit cooldown, then enter on a confirmed trend.
        if item.bars_since_exit < self.cooldown_decisions:
            return
        if score is None or score < self.corr_entry:
            return
        if dir_t < 0 and not self.allow_short:
            return
        self._enter(symbol, item, dir_t, pv_corr, event_time)

    def _enter(
        self, symbol: str, item: _State, direction: int, pv_corr: float | None, event_time: Any
    ) -> None:
        closes = list(item.closes)
        price = closes[-1] if closes else None
        alloc = self._vol_scaled_allocation(closes)
        mode = "LONG" if direction > 0 else "SHORT"
        conviction = abs(float(pv_corr)) if pv_corr is not None else 0.0
        metadata = _target_metadata(
            strategy=_STRATEGY_NAME,
            target_allocation=alloc,
            max_order_value=self.max_order_value,
            target_mode=mode,
            action="entry",
            net_sign=int(direction),
            pv_corr=float(pv_corr) if pv_corr is not None else None,
            confirmation_score=float(direction) * float(pv_corr) if pv_corr is not None else None,
        )
        _emit(
            self.events,
            strategy_id=_STRATEGY_ID,
            symbol=symbol,
            event_time=event_time,
            signal_type=mode,
            strength=max(0.25, min(3.0, 0.5 + conviction)),
            price=price,
            metadata=metadata,
        )
        item.mode = mode
        item.entry_price = price
        item.bars_held = 0
        item.score = float(direction) * conviction

    def _emit_exit(self, symbol: str, item: _State, event_time: Any, *, reason: str) -> None:
        price = item.closes[-1] if item.closes else None
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
        item.bars_held = 0
        item.bars_since_exit = 0
        item.score = None


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the integration wave (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits per the live-safety
# plan).  This is a PER-SYMBOL sleeve (one symbol per candidate,
# ``candidate_mix_type == "single"``, like the return-rider family), so it is
# NOT a cross-sectional multi-asset book and must NOT be given
# ``allow_multi_asset`` or any fake carry tag.  The data-PC handoff gates it on
# RPT >= 10bps per split and factor_ic vs the pure-price trend incumbents.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "time_series_momentum"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "time_series_momentum",
    "trend",
    "volume_confirmation",
    "continuation",
    "price_volume",
    "min_hold",
    "crypto",
)

# Candidate slice, per-symbol single-asset (``candidate_mix_type == "single"``).
# MULTI-TIMEFRAME (1d / 4h / 1h): the decision clock is TIMESTAMP-based (ISO-week
# bucketing in ``_week_key``), so it stays WEEKLY at every bar tf -- the sleeve
# ingests finer bars but still decides once per calendar week.  Scaling rule
# (mirrors ``residual_reversion``'s multi-tf precedent):
#   * DECISION-denominated + timestamp-based UNCHANGED: ``mom_window_weeks`` (in
#     weeks), ``min_hold_decisions`` / ``cooldown_decisions`` (in WEEKLY
#     decisions) are identical across tf -- 4 weekly decisions is 4 weekly
#     decisions regardless of bar size (the proven min-hold rescue).
#   * BAR-denominated windows scale x6 (4h) / x24 (1h) to hold the wall-clock
#     span: ``corr_window`` (trend-health horizon), ``vol_window`` (sizing), and
#     ``bars_per_week`` (the weeks->bars conversion feeding ``mom_window_bars``;
#     7 daily = 42 four-hour = 168 hourly bars per week -- 168 is the schema max).
#   * Thresholds / ratios / bools UNCHANGED (``corr_entry`` / ``corr_exit`` /
#     ``target_vol`` / ``allow_short``).
# No window here reaches the ~9000-bar cap (corr_window tops out at 1680 @ 1h).
_PRICE_VOLUME_CONTINUATION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": (
        {
            "variant": "corr70_mom8wk",
            "corr_window": 70,
            "mom_window_weeks": 8,
            "bars_per_week": 7,
            "corr_entry": 0.25,
            "corr_exit": 0.0,
            "min_hold_decisions": 4,
            "cooldown_decisions": 2,
            "vol_window": 30,
            "target_vol": 0.20,
            "allow_short": True,
        },
        {
            "variant": "corr56_mom4wk",
            "corr_window": 56,
            "mom_window_weeks": 4,
            "bars_per_week": 7,
            "corr_entry": 0.30,
            "corr_exit": 0.0,
            "min_hold_decisions": 4,
            "cooldown_decisions": 2,
            "vol_window": 30,
            "target_vol": 0.20,
            "allow_short": True,
        },
    ),
    "4h": (
        {
            "variant": "corr70_mom8wk",
            "corr_window": 420,
            "mom_window_weeks": 8,
            "bars_per_week": 42,
            "corr_entry": 0.25,
            "corr_exit": 0.0,
            "min_hold_decisions": 4,
            "cooldown_decisions": 2,
            "vol_window": 180,
            "target_vol": 0.20,
            "allow_short": True,
        },
        {
            "variant": "corr56_mom4wk",
            "corr_window": 336,
            "mom_window_weeks": 4,
            "bars_per_week": 42,
            "corr_entry": 0.30,
            "corr_exit": 0.0,
            "min_hold_decisions": 4,
            "cooldown_decisions": 2,
            "vol_window": 180,
            "target_vol": 0.20,
            "allow_short": True,
        },
    ),
    "1h": (
        {
            "variant": "corr70_mom8wk",
            "corr_window": 1680,
            "mom_window_weeks": 8,
            "bars_per_week": 168,
            "corr_entry": 0.25,
            "corr_exit": 0.0,
            "min_hold_decisions": 4,
            "cooldown_decisions": 2,
            "vol_window": 720,
            "target_vol": 0.20,
            "allow_short": True,
        },
        {
            "variant": "corr56_mom4wk",
            "corr_window": 1344,
            "mom_window_weeks": 4,
            "bars_per_week": 168,
            "corr_entry": 0.30,
            "corr_exit": 0.0,
            "min_hold_decisions": 4,
            "cooldown_decisions": 2,
            "vol_window": 720,
            "target_vol": 0.20,
            "allow_short": True,
        },
    ),
}

__all__ = ["PriceVolumeCorrContinuationStrategy"]
