"""Volume-clock (subordinated-clock) momentum rider.

THEORY / PROVENANCE.  Clark (1973, "A Subordinated Stochastic Process Model
with Finite Variance for Speculative Prices," Econometrica) showed that
speculative price changes behave like a stochastic process evaluated in an
internal "business time" driven by the rate of information arrival, not
wall-clock time, with cumulative volume acting as a natural proxy for that
internal clock (the subordinator).  Ane & Geman (2000, "Order Flow,
Transaction Clock, and Normality of Asset Returns," Journal of Finance) show
empirically that returns sampled in transaction/volume time are markedly
closer to Gaussian (less fat-tailed, less heteroskedastic) than the same
returns sampled at fixed wall-clock intervals, because volume time
compresses the quiet, uninformative stretches of the tape and expands the
bursts where information is actually being impounded into price.  Easley,
Lopez de Prado & O'Hara (2012, "The Volume Clock: Insights into the
High-Frequency Paradigm," Journal of Portfolio Management) formalize volume
bars (fixed-size buckets of traded value rather than fixed wall-clock
intervals) as the discretization of this idea and show they produce more
stable statistical properties for downstream signals.

This sleeve does not invent a new momentum formula -- it takes the same
"log return over N look-back units" momentum every trend sleeve in this
registry already uses, and reparameterizes the sampling axis: N is counted
in VOLUME BARS (buckets of accumulated dollar volume), not wall-clock bars.

MECHANICS.  Per accepted wall-clock decision bar the sleeve:

1. Computes this bar's dollar volume as ``close * volume`` (an OHLCV-only
   approximation to true traded value; honest limitation -- see below).
2. Appends it to a rolling window (``vbar_ref_window`` wall-clock bars) and
   takes the ROLLING MEDIAN as the reference per-bar dollar volume.  The
   median (rather than the mean) is deliberately robust to the very outlier
   bars this sleeve is designed to react to -- a single abnormally heavy bar
   does not itself blow out the reference volume-bar size.
3. Sets the volume-bar size ``V = reference * vbar_mult`` and accumulates
   this bar's dollar volume into a running bucket.  Whenever the bucket
   reaches ``>= V`` a volume bar CLOSES at this wall-clock bar's close price;
   the excess remainder carries into the next bucket.  A single unusually
   heavy wall-clock bar can close MULTIPLE volume bars in one step (handled
   with an explicit loop): this is a deliberate, honestly-documented
   OHLCV-granularity approximation.  With only bar-level OHLCV (no tick/trade
   data) there is no way to know the true intra-bar sequencing of trades, so
   every volume bar closed out of one heavy wall-clock bar is stamped with
   that SAME close price.  This slightly understates intra-bar price
   dispersion but preserves the core subordination property (heavy-volume
   bars advance the volume clock faster) without fabricating price paths
   that were never observed.
4. Maintains a bounded history of volume-bar close prices.  The entry signal
   is the volume-clock momentum ``log(vclose[-1] / vclose[-1 - vmom_bars])``:
   LONG when ``>= vmom_entry``, SHORT when ``<= -vmom_entry`` (short only if
   ``allow_short``).
5. LIVENESS GUARD: if the most recent volume bar closed more than
   ``max_stale_bars`` wall-clock bars ago, the volume clock is considered
   STALE (a dead/near-zero-volume market has stopped advancing its internal
   clock) and no entry/pyramid signal is produced regardless of price
   action.  This is the honest reading of "no signal": a subordinated clock
   that has stopped ticking has nothing new to say about direction.

Ride/exit is entirely inherited from :class:`_ReturnRiderBase` (ATR trailing
stop, pyramiding, vol-targeted sizing, ``max_hold_bars`` cap); this sleeve
only decides WHETHER and WHICH WAY to enter/add.

DISTINCT FROM the registry's many wall-clock momentum/trend sleeves
(``TopCapTimeSeriesMomentumStrategy``, ``AdaptiveTrendRiderStrategy``,
``TrendEfficiencyMomentumStrategy``, and similar): those compute log-return
or efficiency-ratio momentum over a fixed COUNT of wall-clock bars, so their
signal timing is a deterministic function of calendar time alone.  This
sleeve computes the IDENTICAL momentum arithmetic but over a fixed amount of
TRADED VALUE -- its sampling clock runs fast when the tape is busy and slow
when it is quiet.  Two markets (or the same market in two regimes) with an
identical wall-clock price path but different volume distribution across
that path will therefore produce DIFFERENT entry timing under this sleeve
while producing IDENTICAL timing under any wall-clock sleeve: this is the
concrete mechanism by which the sleeve decorrelates from wall-clock trend
sleeves precisely in the volatile/quiet regimes where those sleeves tend to
whipsaw (quiet-but-heavy-volume chop over-samples wall-clock momentum into
false signals; a volume clock instead recognizes that chop as "a lot of
elapsed business time with no net progress" and stays flat; conversely a
quiet-tape breakout that a fixed wall-clock window would react to slowly is
picked up quickly once volume actually arrives).

OHLCV-only, per-symbol single-asset, pure Python (no numpy/scipy),
``None``-safe graceful guards throughout (never raise),
``decision_cadence_seconds >= 1800`` (inherited >=30m floor).  Data-local: no
fetching, no artifact writes, no hidden environment state.  Registration is
intentionally deferred (see the commented ``@register`` line below the class)
so this sleeve is wired into the candidate library atomically together with
its ``research_only`` tier hint in a follow-up integration step, never
before.  Must still be validated on the data-bearing machine.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.strategies.external_alpha_sleeves import _EPS, _Snapshot
from lumina_quant.strategies.return_rider_alpha_sleeves import (
    _restore_deque,
    _ReturnRiderBase,
    _RiderCommon,
    _RiderState,
    _safe_non_negative_int,
)
from lumina_quant.tuning import HyperParam


def _median(values: Any) -> float | None:
    """Return the median of the finite values in ``values``, or ``None`` if empty."""
    try:
        seq = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    except Exception:
        return None
    n = len(seq)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return seq[mid]
    return 0.5 * (seq[mid - 1] + seq[mid])


# @register("strategy", "VolumeClockMomentumRiderStrategy", interface="event_driven")  # applied atomically with the research_only tier hint in integration — do NOT apply earlier (live-safety)
class VolumeClockMomentumRiderStrategy(_ReturnRiderBase):
    """Ride momentum sampled in VOLUME time (a subordinated clock) rather than wall-clock time.

    Per accepted wall-clock bar, accumulate dollar volume (``close * volume``)
    into a bucket sized ``reference * vbar_mult`` (reference = rolling median
    per-bar dollar volume over ``vbar_ref_window`` wall-clock bars); each time
    the bucket fills, a volume bar CLOSES at that wall-clock bar's close price
    (a single heavy bar can close several volume bars in one step).  The
    signal is the log-return momentum over the last ``vmom_bars`` volume-bar
    closes: LONG when ``>= vmom_entry``, SHORT when ``<= -vmom_entry``.  A
    liveness guard suppresses the signal when the volume clock has not ticked
    (no volume bar closed) within ``max_stale_bars`` wall-clock bars.  See the
    module docstring for the subordination theory and the wall-clock
    decorrelation argument.  Ride/exit inherited from :class:`_ReturnRiderBase`.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "VolumeClockMomentumRiderStrategy"
    strategy_id = "volume_clock_momentum_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "vbar_ref_window": HyperParam.integer("vbar_ref_window", default=50, low=2, high=4096),
            "vbar_mult": HyperParam.floating("vbar_mult", default=1.0, low=1e-6, high=1000.0),
            "vmom_bars": HyperParam.integer("vmom_bars", default=10, low=1, high=2048),
            "vmom_entry": HyperParam.floating("vmom_entry", default=0.02, low=0.0, high=5.0),
            "max_stale_bars": HyperParam.integer("max_stale_bars", default=50, low=1, high=200000),
            "trail_atr_mult": HyperParam.floating(
                "trail_atr_mult", default=3.0, low=0.1, high=20.0
            ),
            "atr_period": HyperParam.integer("atr_period", default=14, low=2, high=1024),
            "max_adds": HyperParam.integer("max_adds", default=3, low=0, high=32),
            "add_step_atr": HyperParam.floating("add_step_atr", default=1.0, low=0.0, high=20.0),
            "vol_window": HyperParam.integer("vol_window", default=48, low=2, high=4096),
            "target_vol": HyperParam.floating("target_vol", default=0.02, low=0.0, high=1.0),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=240, low=1, high=200000),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "add_alloc_fraction": HyperParam.floating(
                "add_alloc_fraction", default=0.5, low=0.0, high=1.0
            ),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.30, low=0.0, high=5.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=5000.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "min_price": HyperParam.floating("min_price", default=0.10, low=0.0, high=1_000_000.0),
        }

    def __init__(self, bars: Any, events: Any, **params: Any) -> None:
        super().__init__(bars, events, **params)
        self._vbar_closes: dict[str, deque[float]] = {
            symbol: deque(maxlen=self._vbar_history_size) for symbol in self.symbol_list
        }
        self._dv_history: dict[str, deque[float]] = {
            symbol: deque(maxlen=self.vbar_ref_window) for symbol in self.symbol_list
        }
        self._dv_accum: dict[str, float] = dict.fromkeys(self.symbol_list, 0.0)
        self._wall_bar_count: dict[str, int] = dict.fromkeys(self.symbol_list, 0)
        self._last_vbar_wall_index: dict[str, int | None] = dict.fromkeys(self.symbol_list, None)

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.vbar_ref_window = max(2, int(resolved["vbar_ref_window"]))
        self.vbar_mult = max(1e-6, float(resolved["vbar_mult"]))
        self.vmom_bars = max(1, int(resolved["vmom_bars"]))
        self.vmom_entry = max(0.0, float(resolved["vmom_entry"]))
        self.max_stale_bars = max(1, int(resolved["max_stale_bars"]))
        self._vbar_history_size = self.vmom_bars + 8

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(resolved, extra_window=2)

    # -- causal advance: volume-clock bucketing, BEFORE the base's ride/entry -
    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        # Advance the volume clock on this bar's close BEFORE the base runs
        # ride+entry, mirroring the base's own time-key dedup so a repeated
        # boundary firing never double-buckets a bar.  Bucketing uses only the
        # current bar's close/volume, so this is causal (no look-ahead).
        item = self._state.get(symbol)
        if item is not None:
            key = time_key(snapshot.time)
            if not (key and key == item.last_time_key):
                self._advance_volume_clock(symbol, snapshot)
        super()._process_symbol(symbol, snapshot)

    def _advance_volume_clock(self, symbol: str, snapshot: _Snapshot) -> None:
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        raw_volume = safe_float(snapshot.volume)
        volume = max(0.0, raw_volume) if raw_volume is not None else 0.0
        dollar_volume = close * volume

        self._wall_bar_count[symbol] = self._wall_bar_count.get(symbol, 0) + 1

        dv_hist = self._dv_history.setdefault(symbol, deque(maxlen=self.vbar_ref_window))
        dv_hist.append(dollar_volume)
        reference = _median(dv_hist)
        if reference is None or reference <= _EPS:
            # No reliable (nonzero) reference volume yet -- a near-zero/dead
            # market cannot size a volume bucket, so this bar contributes no
            # progress toward a volume-bar close.  The liveness guard below
            # is what turns "the clock never ticks" into an honest no-signal.
            return
        v_bar_size = reference * self.vbar_mult
        if v_bar_size <= _EPS:
            return

        accum = self._dv_accum.get(symbol, 0.0) + dollar_volume
        vbar_closes = self._vbar_closes.setdefault(symbol, deque(maxlen=self._vbar_history_size))
        while accum >= v_bar_size:
            # A single heavy wall-clock bar can close several volume bars; with
            # only OHLCV (no tick data) every such close is stamped with THIS
            # bar's close price -- an explicit, honestly-documented
            # OHLCV-granularity approximation (see module docstring).
            vbar_closes.append(close)
            accum -= v_bar_size
            self._last_vbar_wall_index[symbol] = self._wall_bar_count[symbol]
        self._dv_accum[symbol] = accum

    # -- symbol lookup (mirrors the sibling sleeves' pattern) ----------------
    def _symbol_for(self, item: _RiderState) -> str | None:
        for symbol, candidate in self._state.items():
            if candidate is item:
                return symbol
        return None

    def _clock_is_live(self, symbol: str) -> bool:
        """Return whether a volume bar has closed within ``max_stale_bars`` bars."""
        last_index = self._last_vbar_wall_index.get(symbol)
        if last_index is None:
            return False
        current = self._wall_bar_count.get(symbol, 0)
        return (current - last_index) <= self.max_stale_bars

    def _vclock_momentum(self, symbol: str) -> float | None:
        closes = self._vbar_closes.get(symbol)
        if closes is None or len(closes) <= self.vmom_bars:
            return None
        vals = list(closes)
        prev = vals[-1 - self.vmom_bars]
        now = vals[-1]
        if prev <= _EPS or now <= _EPS:
            return None
        momentum = math.log(now / prev)
        return momentum if math.isfinite(momentum) else None

    def _vclock_direction(self, symbol: str) -> str:
        if not self._clock_is_live(symbol):
            return ""
        momentum = self._vclock_momentum(symbol)
        if momentum is None:
            return ""
        if momentum >= self.vmom_entry:
            return "LONG"
        if momentum <= -self.vmom_entry:
            return "SHORT"
        return ""

    # -- entry/pyramid gates ---------------------------------------------------
    def _entry_decision(self, item: _RiderState) -> str:
        symbol = self._symbol_for(item)
        if symbol is None:
            return ""
        return self._vclock_direction(symbol)

    def _should_pyramid(self, item: _RiderState) -> bool:
        symbol = self._symbol_for(item)
        if symbol is None:
            return False
        return self._vclock_direction(symbol) == item.mode

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        symbol = self._symbol_for(item)
        if symbol is None:
            return {"vclock_momentum": None, "vbar_count": 0, "vbar_stale": None}
        momentum = self._vclock_momentum(symbol)
        closes = self._vbar_closes.get(symbol)
        return {
            "vclock_momentum": float(momentum) if momentum is not None else None,
            "vbar_count": len(closes) if closes is not None else 0,
            "vbar_stale": not self._clock_is_live(symbol),
        }

    # -- state -------------------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["volume_clock"] = {
            symbol: {
                "vbar_closes": list(self._vbar_closes.get(symbol, ())),
                "dv_history": list(self._dv_history.get(symbol, ())),
                "dv_accum": float(self._dv_accum.get(symbol, 0.0)),
                "wall_bar_count": int(self._wall_bar_count.get(symbol, 0)),
                "last_vbar_wall_index": self._last_vbar_wall_index.get(symbol),
            }
            for symbol in self.symbol_list
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        raw = state.get("volume_clock") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            if symbol not in self._vbar_closes or not isinstance(payload, dict):
                continue
            self._safe_restore_deque(self._vbar_closes[symbol], payload.get("vbar_closes"))
            dv_target = self._dv_history.setdefault(symbol, deque(maxlen=self.vbar_ref_window))
            self._safe_restore_deque(dv_target, payload.get("dv_history"))
            accum = safe_float(payload.get("dv_accum"))
            self._dv_accum[symbol] = accum if accum is not None else 0.0
            self._wall_bar_count[symbol] = _safe_non_negative_int(payload.get("wall_bar_count"))
            last_index = payload.get("last_vbar_wall_index")
            self._last_vbar_wall_index[symbol] = (
                _safe_non_negative_int(last_index) if last_index is not None else None
            )

    @staticmethod
    def _safe_restore_deque(target: deque[float], payload: Any) -> None:
        """Call :func:`_restore_deque`, swallowing non-iterable adversarial payloads.

        ``_restore_deque`` does ``list(payload or [])``, which raises
        ``TypeError`` for a truthy non-iterable payload (e.g. a bare ``int`` or
        ``float``); this wrapper keeps the never-raise contract for arbitrary
        deserialized input without altering the shared helper.
        """
        try:
            _restore_deque(target, payload)
        except Exception:
            target.clear()


# Exactly one variant per timeframe: "Rider" sleeves stay single-variant so
# the W3 integrator can build one candidate per symbol per timeframe without
# touching this module's logic; shape mirrors the existing rider slices (e.g.
# ``_ADAPTIVE_TREND_RIDER_SLICE`` in candidate_library.py).
_VOLUME_CLOCK_MOMENTUM_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "fast_ls",
            "vbar_ref_window": 40,
            "vbar_mult": 1.0,
            "vmom_bars": 8,
            "vmom_entry": 0.015,
            "max_stale_bars": 40,
            "trail_atr_mult": 3.0,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 200,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1h": (
        {
            "variant": "core_ls",
            "vbar_ref_window": 50,
            "vbar_mult": 1.0,
            "vmom_bars": 10,
            "vmom_entry": 0.02,
            "max_stale_bars": 50,
            "trail_atr_mult": 3.2,
            "atr_period": 14,
            "max_adds": 3,
            "add_step_atr": 1.0,
            "vol_window": 48,
            "target_vol": 0.020,
            "max_hold_bars": 180,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "4h": (
        {
            "variant": "swing_ls",
            "vbar_ref_window": 60,
            "vbar_mult": 1.1,
            "vmom_bars": 12,
            "vmom_entry": 0.03,
            "max_stale_bars": 30,
            "trail_atr_mult": 3.5,
            "atr_period": 14,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 36,
            "target_vol": 0.030,
            "max_hold_bars": 120,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
    "1d": (
        {
            "variant": "macro_ls",
            "vbar_ref_window": 40,
            "vbar_mult": 1.2,
            "vmom_bars": 14,
            "vmom_entry": 0.05,
            "max_stale_bars": 20,
            "trail_atr_mult": 4.0,
            "atr_period": 10,
            "max_adds": 2,
            "add_step_atr": 1.0,
            "vol_window": 24,
            "target_vol": 0.040,
            "max_hold_bars": 60,
            "allow_short": True,
            "add_alloc_fraction": 0.5,
        },
    ),
}

__all__ = [
    "VolumeClockMomentumRiderStrategy",
]
