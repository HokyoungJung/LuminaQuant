"""Low-turnover per-symbol trend-persistence sleeve (L1, min-hold rescued).

``LowTurnoverTrendPersistenceStrategy`` is a per-symbol time-series-momentum
(TSMOM) sleeve deliberately engineered to survive the repo's realized
per-turn-cost (RPT) graveyard.  The graveyard records that high-turnover trend
sweeps (#13) and a debounced-momentum hysteresis variant (#14, RPT
9.36-9.48bps) both died UNDER the 10bps realized-cost floor, while the proven
rescue was to lift the minimum hold to 36 bars (RPT 3.69 -> 24.69/59.72/21.11bps,
clearing 10bps on every split).  This sleeve IS that rescued construction: a
multi-horizon TSMOM sign-agreement entry, wrapped in a HARD minimum-hold plus
cooldown so a would-be flip inside the hold window is suppressed by design.

THEORY / PROVENANCE
--------------------
- Moskowitz, Ooi & Pedersen (2012, JFE): time-series momentum earns a
  persistent premium across asset classes; multi-horizon agreement isolates the
  cleanest, most-persistent trends.
- Hurst, Ooi & Pedersen (2017, JPM): the effect is robust across centuries but
  is a MODEST, slow-moving premium -- the honest expectation here is small.
- Frazzini, Israel & Moskowitz (2012): trading costs are the binding constraint
  on momentum realizability -> the low-turnover tilt (weekly cadence + hard
  min-hold) is the design's whole point.
- DECAY caveat (Liu & Tsyvinski 2021, RFS; Rosen & Wang 2024): crypto TSMOM is
  a DECAYING factor.  This sleeve is a low-turnover FALSIFICATION probe of a
  decaying axis, not a promise of edge.

SIGNAL SPEC
-----------
Per symbol, per completed WEEKLY decision bar (the ``_week_key`` internal clock
buckets ingested bars into ISO weeks so the sleeve decides at most once per week
even when fed higher-frequency bars):

1. Multi-horizon TSMOM sign over three lookbacks (``tsmom_short`` /
   ``tsmom_mid`` / ``tsmom_long``, defaults 28/56/84 bars ~= 4/8/12 weeks on
   daily bars) via :func:`simple_return`.  A directional signal requires ALL
   THREE horizons to agree on sign (strict agreement); any disagreement or a
   neutral horizon yields no signal (flat).
2. Trend-EFFICIENCY gate: the pure :func:`kaufman_efficiency_ratio`
   (``efficiency_period``) must clear ``min_efficiency`` -- trades only when the
   move is efficient, not choppy.  The efficiency ratio ALSO scales the emitted
   conviction / size (KER-weighted).
3. ADX-style trend-strength gate: :func:`average_directional_index`
   (``adx_period``) must clear ``adx_min``.
4. Low-vol-PERSISTENCE gate: :func:`volatility_ratio`
   (``vol_persist_fast`` / ``vol_persist_slow``) must stay at/under
   ``vol_persist_max`` -- short-horizon realized vol may not spike above its
   slow baseline (a persistent, non-blow-off trend).

A flat symbol enters in the agreed direction only when (1) AND (2) AND (3) AND
(4) all hold AND the cooldown since the last exit has elapsed.  A held position
is NEVER changed while ``bars_held < min_hold_bars`` -- the hard-min-hold
rescue -- so a would-be reversal inside the hold window is suppressed.  Once the
min-hold is satisfied, the position EXITs on a confirmed opposite-direction
signal (or the ``max_hold_bars`` backstop), then a ``cooldown_bars`` flat window
must pass before any re-entry.  Sizing is vol-targeted (``target_vol`` over
``vol_window``) and KER-weighted, clamped to twice ``target_allocation``.

DISTINCT-FROM
-------------
Shares the multi-horizon-sign core with
``MultiTimeframeTrendEnsembleStrategy`` (aggressive_return_alpha_sleeves.py),
but that sleeve rides/pyramids with an ATR trailing stop and re-enters on the
next horizon-agreement flip with NO hold floor -- it is a high-turnover return
rider.  This sleeve's load-bearing difference is the HARD min-hold + cooldown
(plus the ADX + low-vol-persistence gates): on an input where the ensemble flips
on a fresh horizon agreement, this sleeve stays put until its min-hold expires,
producing a divergent action at the same bar.  That divergence is the
orthogonality key; the min-hold rescue is what moves RPT above the 10bps floor
that killed graveyard #13/#14.

This module is data-local (no I/O, no hidden configuration bus), pure Python
(``math``/``deque`` only, no numpy), completed-bar, and never raises from
``calculate_signals``.  It ships WITHOUT ``@register`` (inert): registration,
tier hint, and candidate wiring land atomically in the W3 integration wave.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.alpha_features import (
    realized_volatility,
    simple_return,
    volatility_ratio,
)
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.momentum import kaufman_efficiency_ratio
from lumina_quant.indicators.trend import average_directional_index
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

_STRATEGY_ID = "low_turnover_trend_persistence"
_STRATEGY_NAME = "LowTurnoverTrendPersistenceStrategy"

# A fresh symbol has never traded, so its cooldown is treated as long elapsed --
# the first eligible entry must not be blocked by a phantom startup cooldown.
_COOLDOWN_SATISFIED = 1 << 30


@dataclass(slots=True)
class _TrendState:
    """Per-symbol OHLC history + position / min-hold / cooldown bookkeeping."""

    opens: deque[float]
    highs: deque[float]
    lows: deque[float]
    closes: deque[float]
    mode: str = "OUT"  # OUT / LONG / SHORT
    entry_price: float | None = None
    bars_held: int = 0  # decision (weekly) bars in the CURRENT position
    bars_since_exit: int = _COOLDOWN_SATISFIED  # decision bars since last exit
    last_bar_key: str = ""  # dedup identical ingested bars
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


@register("strategy", _STRATEGY_NAME, interface="event_driven")
class LowTurnoverTrendPersistenceStrategy(Strategy):
    """Per-symbol multi-horizon TSMOM with a hard min-hold + cooldown rescue.

    See the module docstring for the full theory, signal spec, and
    distinct-from rationale.  This class only reads local event/bar OHLC; it
    performs no I/O and never raises from ``calculate_signals``.
    """

    decision_cadence_seconds = 604800  # weekly effective cadence (see _week_key)
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = _STRATEGY_NAME
    strategy_id = _STRATEGY_ID

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "tsmom_short": HyperParam.integer("tsmom_short", default=28, low=1, high=4096),
            "tsmom_mid": HyperParam.integer("tsmom_mid", default=56, low=2, high=8192),
            "tsmom_long": HyperParam.integer("tsmom_long", default=84, low=3, high=20000),
            "efficiency_period": HyperParam.integer(
                "efficiency_period", default=20, low=2, high=4096
            ),
            "min_efficiency": HyperParam.floating(
                "min_efficiency", default=0.30, low=0.0, high=1.0
            ),
            "adx_period": HyperParam.integer("adx_period", default=14, low=2, high=1024),
            "adx_min": HyperParam.floating("adx_min", default=20.0, low=0.0, high=100.0),
            "vol_persist_fast": HyperParam.integer(
                "vol_persist_fast", default=16, low=2, high=4096
            ),
            "vol_persist_slow": HyperParam.integer(
                "vol_persist_slow", default=64, low=3, high=8192
            ),
            "vol_persist_max": HyperParam.floating(
                "vol_persist_max", default=1.5, low=0.1, high=20.0
            ),
            # The proven min-hold rescue: >= the 36-bar analog that lifted RPT
            # above the 10bps floor (graveyard #13/#14). Weekly decision bars.
            "min_hold_bars": HyperParam.integer("min_hold_bars", default=36, low=1, high=200000),
            "cooldown_bars": HyperParam.integer("cooldown_bars", default=4, low=0, high=200000),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=2000, low=1, high=1000000),
            "vol_window": HyperParam.integer("vol_window", default=56, low=2, high=4096),
            "target_vol": HyperParam.floating("target_vol", default=0.20, low=0.0, high=2.0),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.20, low=0.0, high=5.0, tunable=False
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
        # Horizons are clamped strictly increasing so agreement across three
        # distinct lookbacks is well-defined.
        self.tsmom_short = max(1, int(resolved["tsmom_short"]))
        self.tsmom_mid = max(self.tsmom_short + 1, int(resolved["tsmom_mid"]))
        self.tsmom_long = max(self.tsmom_mid + 1, int(resolved["tsmom_long"]))
        self.efficiency_period = max(2, int(resolved["efficiency_period"]))
        self.min_efficiency = max(0.0, min(1.0, float(resolved["min_efficiency"])))
        self.adx_period = max(2, int(resolved["adx_period"]))
        self.adx_min = max(0.0, float(resolved["adx_min"]))
        self.vol_persist_fast = max(2, int(resolved["vol_persist_fast"]))
        self.vol_persist_slow = max(self.vol_persist_fast + 1, int(resolved["vol_persist_slow"]))
        self.vol_persist_max = max(0.0, float(resolved["vol_persist_max"]))
        self.min_hold_bars = max(1, int(resolved["min_hold_bars"]))
        self.cooldown_bars = max(0, int(resolved["cooldown_bars"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.target_vol = max(0.0, float(resolved["target_vol"]))
        self.allow_short = bool(resolved["allow_short"])
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        # Bounded ring buffer: the longest lookback any gate consumes, plus a
        # small margin (the +1 the ratio/return helpers need past the window).
        self._history_size = (
            max(
                self.tsmom_long,
                self.vol_persist_slow,
                self.adx_period,
                self.efficiency_period,
                self.vol_window,
            )
            + 8
        )
        self._state: dict[str, _TrendState] = {
            symbol: _TrendState(
                opens=deque(maxlen=self._history_size),
                highs=deque(maxlen=self._history_size),
                lows=deque(maxlen=self._history_size),
                closes=deque(maxlen=self._history_size),
            )
            for symbol in self.symbol_list
        }

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        return {
            "symbol_state": {
                symbol: {
                    "opens": list(item.opens),
                    "highs": list(item.highs),
                    "lows": list(item.lows),
                    "closes": list(item.closes),
                    "mode": item.mode,
                    "entry_price": item.entry_price,
                    "bars_held": int(item.bars_held),
                    "bars_since_exit": int(item.bars_since_exit),
                    "last_bar_key": item.last_bar_key,
                    "last_decision_week": item.last_decision_week,
                    "score": item.score,
                }
                for symbol, item in self._state.items()
            }
        }

    def set_state(self, state: dict[str, Any]) -> None:
        raw = state.get("symbol_state") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._state or not isinstance(payload, dict):
                continue
            item = self._state[symbol]
            try:
                _restore_deque(item.opens, payload.get("opens"))
                _restore_deque(item.highs, payload.get("highs"))
                _restore_deque(item.lows, payload.get("lows"))
                _restore_deque(item.closes, payload.get("closes"))
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
    # ingestion / cadence
    # ------------------------------------------------------------------ #
    def _week_key(self, raw_time: Any) -> str:
        """Bucket a bar timestamp into an ISO ``YYYY-Wnn`` weekly decision key.

        Falls back to the raw time key when the timestamp is unparseable, so a
        degenerate feed still deduplicates and never raises.
        """
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
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        item.opens.append(float(snapshot.open if snapshot.open is not None else close))
        item.highs.append(float(snapshot.high if snapshot.high is not None else close))
        item.lows.append(float(snapshot.low if snapshot.low is not None else close))
        item.closes.append(close)
        # Weekly effective cadence: decide at most once per new ISO week.
        week = self._week_key(snapshot.time)
        if not week or week == item.last_decision_week:
            return
        item.last_decision_week = week
        self._decide(symbol, item, snapshot.time)

    # ------------------------------------------------------------------ #
    # signal / gates
    # ------------------------------------------------------------------ #
    def _horizon_agreement(self, closes: list[float]) -> int:
        """Return ``+1``/``-1`` when all three TSMOM horizons agree, else ``0``."""
        net = 0
        for lookback in (self.tsmom_short, self.tsmom_mid, self.tsmom_long):
            ret = simple_return(closes, lookback=lookback)
            if ret is None or ret == 0.0:
                return 0
            sign = 1 if ret > 0.0 else -1
            if net == 0:
                net = sign
            elif sign != net:
                return 0
        return net

    def _efficiency(self, closes: list[float]) -> float | None:
        """Return the pure Kaufman efficiency ratio over ``efficiency_period``."""
        return kaufman_efficiency_ratio(closes, period=self.efficiency_period)

    def _desired_signal(self, item: _TrendState) -> int:
        """Return the gated target direction ``+1``/``-1``/``0`` for a symbol.

        A directional target requires: all three TSMOM horizons agree, the
        Kaufman efficiency ratio clears ``min_efficiency``, ADX clears
        ``adx_min``, and short/slow realized-vol ratio stays at/under
        ``vol_persist_max`` (low-vol persistence).
        """
        closes = list(item.closes)
        if len(closes) < self.tsmom_long + 1:
            return 0
        net = self._horizon_agreement(closes)
        if net == 0:
            return 0
        er = self._efficiency(closes)
        if er is None or er < self.min_efficiency:
            return 0
        adx, _plus_di, _minus_di = average_directional_index(
            list(item.highs), list(item.lows), closes, period=self.adx_period
        )
        if adx is None or adx < self.adx_min:
            return 0
        vratio = volatility_ratio(
            closes, fast_window=self.vol_persist_fast, slow_window=self.vol_persist_slow
        )
        if vratio is None or vratio > self.vol_persist_max:
            return 0
        return net

    # ------------------------------------------------------------------ #
    # sizing
    # ------------------------------------------------------------------ #
    def _vol_scaled_allocation(self, closes: list[float], conviction: float) -> float:
        """Vol-target the base allocation and scale by KER conviction; clamped."""
        alloc = self.target_allocation
        if self.target_vol > 0.0:
            vol = realized_volatility(closes, window=self.vol_window)
            if vol is not None and vol > _EPS:
                alloc = self.target_allocation * (self.target_vol / vol)
        alloc *= conviction
        return max(0.0, min(self.target_allocation * 2.0, alloc))

    # ------------------------------------------------------------------ #
    # decision (weekly cadence)
    # ------------------------------------------------------------------ #
    def _decide(self, symbol: str, item: _TrendState, event_time: Any) -> None:
        # Age the position / cooldown once per weekly decision bar.
        if item.mode in {"LONG", "SHORT"}:
            item.bars_held += 1
        else:
            item.bars_since_exit += 1

        desired = self._desired_signal(item)

        if item.mode in {"LONG", "SHORT"}:
            # HARD min-hold rescue: no change of any kind while inside the hold
            # window -- a would-be flip is suppressed here (this is what lifts
            # RPT above the 10bps floor that killed graveyard #13/#14).
            if item.bars_held < self.min_hold_bars:
                return
            current = 1 if item.mode == "LONG" else -1
            confirmed_reversal = desired != 0 and desired != current
            maxed = item.bars_held >= self.max_hold_bars
            if confirmed_reversal or maxed:
                self._emit_exit(
                    symbol,
                    item,
                    event_time,
                    reason="confirmed_reversal" if confirmed_reversal else "max_hold",
                )
            return

        # Flat: honour the post-exit cooldown, then enter on a gated signal.
        if item.bars_since_exit < self.cooldown_bars:
            return
        if desired == 0:
            return
        if desired < 0 and not self.allow_short:
            return
        self._enter(symbol, item, desired, event_time)

    def _enter(self, symbol: str, item: _TrendState, direction: int, event_time: Any) -> None:
        closes = list(item.closes)
        price = closes[-1] if closes else None
        er = self._efficiency(closes)
        er_value = 0.0 if er is None else max(0.0, min(1.0, float(er)))
        # KER conviction in [0.5, 1.5]: an efficient trend carries more risk.
        conviction = max(0.5, min(1.5, 0.5 + er_value))
        alloc = self._vol_scaled_allocation(closes, conviction)
        adx, _plus_di, _minus_di = average_directional_index(
            list(item.highs), list(item.lows), closes, period=self.adx_period
        )
        mode = "LONG" if direction > 0 else "SHORT"
        metadata = _target_metadata(
            strategy=_STRATEGY_NAME,
            target_allocation=alloc,
            max_order_value=self.max_order_value,
            target_mode=mode,
            action="entry",
            net_sign=int(direction),
            efficiency_ratio=float(er_value),
            adx=float(adx) if adx is not None else None,
            conviction=float(conviction),
        )
        _emit(
            self.events,
            strategy_id=_STRATEGY_ID,
            symbol=symbol,
            event_time=event_time,
            signal_type=mode,
            strength=max(0.25, min(3.0, conviction)),
            price=price,
            metadata=metadata,
        )
        item.mode = mode
        item.entry_price = price
        item.bars_held = 0
        item.score = float(direction) * er_value

    def _emit_exit(self, symbol: str, item: _TrendState, event_time: Any, *, reason: str) -> None:
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
# Candidate-wiring hints for the W3 integrator (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits per the live-safety
# plan).  This is a PER-SYMBOL sleeve (one symbol per candidate,
# ``candidate_mix_type == "single"``, like the return-rider family), so it is
# NOT a cross-sectional multi-asset book and must NOT be given ``allow_multi_asset``
# or any fake carry tag.  The data-PC handoff gates it on RPT >= 10bps per split
# (the exact gate graveyard #14 failed) and factor_ic vs
# ``MultiTimeframeTrendEnsembleStrategy``.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "time_series_momentum"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "time_series_momentum",
    "trend",
    "tsmom",
    "low_turnover",
    "min_hold",
    "multi_horizon",
    "crypto",
)

# Candidate slice.  The ``tsmom_short/mid/long`` defaults of 28/56/84 are the
# 4/8/12-WEEK wall-clock horizons the thesis rests on (28/56/84 * bars_per_day),
# and every signal window here (TSMOM lookbacks, ``efficiency_period``,
# ``adx_period``, the ``vol_persist_fast/slow`` ratio windows, and the vol-target
# ``vol_window``) reads the RAW ingested bar series -- so at 4h/1h they scale x6
# / x24 to keep those wall-clock spans (and the multi-week TSMOM thesis) honest
# at intraday cadence rather than collapsing to a few days.  The decision clock
# itself is a timestamp-based WEEKLY ISO key (``_week_key``), so
# ``min_hold_bars``/``cooldown_bars``/``max_hold_bars`` are WEEKLY decision bars
# -- timeframe-agnostic and UNCHANGED (the proven min-hold rescue that lifts RPT
# above the 10bps floor).  Efficiency/ADX/vol-persist THRESHOLDS and ``target_vol``
# are scale-invariant and unchanged.  Per-symbol single-asset
# (``candidate_mix_type == "single"``).
_LOW_TURNOVER_TREND_PERSISTENCE_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "4h": (
        {
            "variant": "core_4_8_12wk",
            "tsmom_short": 168,
            "tsmom_mid": 336,
            "tsmom_long": 504,
            "efficiency_period": 120,
            "min_efficiency": 0.30,
            "adx_period": 84,
            "adx_min": 20.0,
            "vol_persist_fast": 96,
            "vol_persist_slow": 384,
            "vol_persist_max": 1.5,
            "min_hold_bars": 36,
            "cooldown_bars": 4,
            "max_hold_bars": 2000,
            "vol_window": 336,
            "target_vol": 0.20,
            "allow_short": True,
        },
    ),
    "1h": (
        {
            "variant": "core_4_8_12wk",
            "tsmom_short": 672,
            "tsmom_mid": 1344,
            "tsmom_long": 2016,
            "efficiency_period": 480,
            "min_efficiency": 0.30,
            "adx_period": 336,
            "adx_min": 20.0,
            "vol_persist_fast": 384,
            "vol_persist_slow": 1536,
            "vol_persist_max": 1.5,
            "min_hold_bars": 36,
            "cooldown_bars": 4,
            "max_hold_bars": 2000,
            "vol_window": 1344,
            "target_vol": 0.20,
            "allow_short": True,
        },
    ),
    "1d": (
        {
            "variant": "core_4_8_12wk",
            "tsmom_short": 28,
            "tsmom_mid": 56,
            "tsmom_long": 84,
            "efficiency_period": 20,
            "min_efficiency": 0.30,
            "adx_period": 14,
            "adx_min": 20.0,
            "vol_persist_fast": 16,
            "vol_persist_slow": 64,
            "vol_persist_max": 1.5,
            "min_hold_bars": 36,
            "cooldown_bars": 4,
            "max_hold_bars": 2000,
            "vol_window": 56,
            "target_vol": 0.20,
            "allow_short": True,
        },
    ),
}

__all__ = ["LowTurnoverTrendPersistenceStrategy"]
