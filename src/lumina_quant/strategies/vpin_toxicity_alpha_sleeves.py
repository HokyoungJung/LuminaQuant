"""VPIN-lite microstructure flow-toxicity rider: bucketed order-flow TOXICITY,
not a per-bar flow-direction proxy.

A per-symbol single-asset, return-first DIRECTIONAL sleeve that subclasses the
proven :class:`_ReturnRiderBase` (ATR trailing stop + pyramiding + vol-target
sizing; winners run, no fixed take-profit) and adds a fresh entry trigger the
existing registry does not carry.  It is wired ONLY at >=30m timeframes and
stays ``candidate_mix_type=="single"`` so it bypasses the multi-asset
admission gate.

THEORY / PROVENANCE.  Easley, Lopez de Prado & O'Hara (2012, "Flow Toxicity
and Liquidity in a High-Frequency World," Review of Financial Studies) show
that the Volume-synchronized Probability of INformed trading (VPIN) -- the
trailing average of the absolute buy/sell imbalance across a rolling window
of EQUAL-SIZED volume buckets -- rises ahead of volatility events (their
headline result documents it spiking ahead of the 2010 Flash Crash).  Absent
tick data, Bulk Volume Classification (BVC) approximates the buy/sell split
of a bar's volume from OHLCV alone: ``Phi(dP / sigma)`` (the standard normal
CDF of the bar's price change, scaled by a trailing estimate of price-change
volatility) is the BUY fraction.  Both primitives -- BVC
(:func:`~lumina_quant.indicators.vpin.bulk_volume_buy_fraction`) and the VPIN
aggregation (:func:`~lumina_quant.indicators.vpin.vpin_from_buckets`), plus
the volume-bucket accumulator
(:func:`~lumina_quant.indicators.vpin.accumulate_volume_bucket`) -- live in
:mod:`lumina_quant.indicators.vpin`.

MECHANICS.  Per accepted decision bar the sleeve:

1. Classifies the bar's volume into a BUY/SELL split via BVC, using
   ``sigma`` = the trailing sample standard deviation of close-to-close price
   DIFFERENCES over ``vol_window`` bars (the same window the base uses for
   vol-target sizing).  When ``sigma`` is not yet available (early history)
   the bar is classified NEUTRAL (buy fraction 0.5, zero imbalance
   contribution) rather than stalling the bucket clock.
2. Accumulates the classified buy/sell volume into fixed-size volume buckets
   via :func:`~lumina_quant.indicators.vpin.accumulate_volume_bucket`, sized
   as ``bucket_size = median(trailing bar volumes) * bucket_mult`` (the
   median is taken over the same bounded OHLC lookback the rider base already
   retains for ATR/vol-target sizing).  A single unusually heavy bar can
   close more than one bucket; the accumulator handles this by looping.
3. Maintains the trailing ``n_buckets`` CLOSED bucket imbalances and computes
   VPIN as their mean absolute value
   (:func:`~lumina_quant.indicators.vpin.vpin_from_buckets`).
4. Maintains a trailing ``vpin_history`` of VPIN readings and computes the
   TOXICITY PERCENTILE of the current reading within that history (reusing
   :func:`~lumina_quant.indicators.oscillators.percentile_rank`, not a
   reimplementation).

ENTRY (``toxicity_mode``):

- ``"confirm"`` (default): enter in the TREND direction only when the
  toxicity percentile is HIGH (``>= toxicity_entry``).  Rationale: sustained,
  heavily one-sided informed flow (high VPIN) confirming a live trend is the
  flash-crash-adjacent "informed flow is running the book over" regime --
  toxic pressure AND direction agreeing signals continuation, not reversal.
- ``"avoid"``: enter in the TREND direction only when the toxicity percentile
  is LOW (``<= toxicity_calm``), i.e. a benign, well-behaved order flow
  regime.  Rationale: trade the trend only when the tape is NOT currently
  showing adverse-selection stress, a defensive flow-quality filter.

Trend is the sign of the return over ``trend_lookback`` bars, gated by a
minimum absolute magnitude ``min_trend_roc``.  Ride/exit is entirely
inherited from :class:`_ReturnRiderBase` (ATR trailing stop, pyramiding,
vol-targeted sizing, ``max_hold_bars`` cap); this sleeve only decides WHETHER
and WHICH WAY to enter/add.

DISTINCT FROM every flow/imbalance signal already in the registry.
:func:`~lumina_quant.indicators.alpha_features.order_flow_imbalance` is a
per-bar, TIME-windowed ``(buy - sell) / (buy + sell)`` sum -- it never
partitions traded VOLUME itself into buckets.
:class:`~lumina_quant.strategies.robust_alpha_sleeves.
TakerFlowImbalanceContinuationStrategy` and the crowding component of
:class:`~lumina_quant.strategies.external_alpha_sleeves.
FundingDislocationTrendCarryStrategy` both consume that same TIME-windowed
imbalance directly.
:class:`~lumina_quant.strategies.micro_signal_alpha_sleeves.
IntradayFlowPressureRiderStrategy` rides a z-scored CUMULATIVE per-decision-bar
taker imbalance -- still bar-indexed, not volume-bucketed, and a directional
continuation signal rather than a toxicity regime read.  None of these
sleeves partition volume into equal-sized buckets or compute a trailing
average of bucket-level imbalance (VPIN) as a distributional flow-TOXICITY
statistic; this sleeve's entry trigger has zero mechanical overlap with any
of them.  OHLCV-only, per-symbol single-asset, ``decision_cadence_seconds >=
1800``.

An ATR trailing stop is the primary return lever; a TopCap-scale
``target_allocation`` (``tunable=False``); vol-targeted sizing; a
``max_hold_bars`` cap; per-symbol bounded state; ``None``-safe graceful
guards (never raise).  It is data-local (no fetching, no artifact writes) and
must still be validated on the data-bearing machine.

Registration is intentionally deferred (see the commented ``@register`` line
after the class): this sleeve ships alongside a batch of same-day siblings
and is wired into the plugin registry and ``candidate_library`` atomically,
with a ``research_only`` tier hint, in a follow-up integration step -- never
before.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.common import safe_float, time_key
from lumina_quant.indicators.oscillators import percentile_rank, rate_of_change
from lumina_quant.indicators.rolling_stats import sample_std
from lumina_quant.indicators.vpin import (
    accumulate_volume_bucket,
    bulk_volume_buy_fraction,
    vpin_from_buckets,
)
from lumina_quant.strategies.external_alpha_sleeves import _Snapshot
from lumina_quant.strategies.return_rider_alpha_sleeves import (
    _restore_deque,
    _ReturnRiderBase,
    _RiderCommon,
    _RiderState,
)
from lumina_quant.tuning import HyperParam


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


def _price_change_sigma(prior_closes: list[float], *, window: int) -> float | None:
    """Return the sample std (ddof=1) of trailing close-to-close price DIFFERENCES.

    Uses absolute price DIFFERENCES (not returns) so the result is on the
    same scale as the raw ``price_change`` the BVC z-score needs.  ``None``
    when there is insufficient trailing history.
    """
    win = max(2, int(window))
    if len(prior_closes) <= win:
        return None
    tail = prior_closes[-(win + 1) :]
    diffs = [current - previous for previous, current in pairwise(tail)]
    return sample_std(diffs)


def _rolling_bucket_size(volumes: list[float], *, bucket_mult: float) -> float:
    """Return ``median(volumes) * bucket_mult``, or ``0.0`` when undefined.

    A ``0.0`` bucket size is a deliberate stall signal: with fewer than 2
    finite, non-negative volume samples the sleeve cannot yet size a bucket,
    so :func:`~lumina_quant.indicators.vpin.accumulate_volume_bucket`'s own
    ``bucket_size <= 0`` guard leaves the accumulator unchanged instead of
    dividing by zero.
    """
    finite = [
        float(value)
        for value in volumes
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0.0
    ]
    if len(finite) < 2:
        return 0.0
    return statistics.median(finite) * max(0.0, float(bucket_mult))


def _trend_sign(closes: list[float], *, trend_lookback: int, min_trend_roc: float) -> int:
    """Return ``+1``/``-1``/``0`` for the prevailing trend.

    The sign of the return over ``trend_lookback`` bars, gated by a minimum
    absolute magnitude ``min_trend_roc`` so a flat/noisy path reads as
    neutral.
    """
    roc = rate_of_change(closes, period=trend_lookback)
    if roc is None or abs(roc) < min_trend_roc:
        return 0
    return 1 if roc > 0.0 else -1


@dataclass(slots=True)
class _VpinExtra:
    """Per-symbol volume-bucket accumulator plus VPIN/toxicity trailing history.

    ``bucket_buy``/``bucket_sell`` are the live volume-bucket carry consumed
    by :func:`~lumina_quant.indicators.vpin.accumulate_volume_bucket`;
    ``bucket_imbalances`` holds the trailing CLOSED bucket imbalances that
    :func:`~lumina_quant.indicators.vpin.vpin_from_buckets` averages into
    VPIN; ``vpin_history`` holds the trailing VPIN READINGS themselves,
    against which the latest reading's percentile rank is computed;
    ``volumes`` is the trailing bar-volume history used to size the next
    bucket (rolling median * ``bucket_mult``).
    """

    bucket_imbalances: deque[float]
    vpin_history: deque[float]
    volumes: deque[float]
    bucket_buy: float = 0.0
    bucket_sell: float = 0.0
    last_vpin: float | None = None
    last_percentile: float | None = None


def _new_vpin_extra(*, n_buckets: int, vpin_history: int, volume_window: int) -> _VpinExtra:
    """Build a fresh :class:`_VpinExtra` with correctly bounded deques."""
    return _VpinExtra(
        bucket_imbalances=deque(maxlen=max(2, int(n_buckets))),
        vpin_history=deque(maxlen=max(2, int(vpin_history))),
        volumes=deque(maxlen=max(2, int(volume_window))),
    )


@register("strategy", "VpinToxicityRiderStrategy", interface="event_driven")
class VpinToxicityRiderStrategy(_ReturnRiderBase):
    """Ride the trend when volume-bucketed flow toxicity (VPIN) confirms or is calm.

    Per bar, this bar's volume is BVC-classified into buy/sell, folded into
    fixed-size volume buckets, and the trailing ``n_buckets`` closed-bucket
    imbalances are averaged into VPIN; the current VPIN's percentile rank
    within its own trailing ``vpin_history`` is the toxicity read.
    ``toxicity_mode="confirm"`` (default) enters in the TREND direction when
    toxicity is HIGH (informed-flow continuation); ``"avoid"`` enters in the
    TREND direction only when toxicity is LOW (a benign-flow filter).  The
    winner is ridden with the inherited ATR trailing stop + pyramiding.

    DISTINCT from every flow/imbalance sleeve in the registry: see the module
    docstring.  OHLCV-only, per-symbol single-asset,
    ``decision_cadence_seconds >= 1800``.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "VpinToxicityRiderStrategy"
    strategy_id = "vpin_toxicity_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "bucket_mult": HyperParam.floating("bucket_mult", default=1.0, low=0.05, high=50.0),
            "n_buckets": HyperParam.integer("n_buckets", default=40, low=3, high=2000),
            "vpin_history": HyperParam.integer("vpin_history", default=100, low=3, high=20000),
            "toxicity_entry": HyperParam.floating(
                "toxicity_entry", default=0.80, low=0.0, high=1.0
            ),
            "toxicity_calm": HyperParam.floating("toxicity_calm", default=0.20, low=0.0, high=1.0),
            "toxicity_mode": HyperParam.string("toxicity_mode", default="confirm", tunable=False),
            "trend_lookback": HyperParam.integer("trend_lookback", default=20, low=2, high=20000),
            "min_trend_roc": HyperParam.floating("min_trend_roc", default=0.0, low=0.0, high=1.0),
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
        self._extra = {
            symbol: _new_vpin_extra(
                n_buckets=self.n_buckets,
                vpin_history=self.vpin_history,
                volume_window=self._history_size,
            )
            for symbol in self.symbol_list
        }

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.bucket_mult = max(1e-6, float(resolved["bucket_mult"]))
        self.n_buckets = max(2, int(resolved["n_buckets"]))
        self.vpin_history = max(2, int(resolved["vpin_history"]))
        self.toxicity_entry = max(0.0, min(1.0, float(resolved["toxicity_entry"])))
        self.toxicity_calm = max(0.0, min(1.0, float(resolved["toxicity_calm"])))
        mode = str(resolved["toxicity_mode"] or "confirm").strip().lower()
        self.toxicity_mode = mode if mode in {"confirm", "avoid"} else "confirm"
        self.trend_lookback = max(1, int(resolved["trend_lookback"]))
        self.min_trend_roc = max(0.0, float(resolved["min_trend_roc"]))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(
            resolved,
            extra_window=max(self.trend_lookback, self.n_buckets, int(resolved["vol_window"])) + 2,
        )

    # -- causal advance: BVC classify + bucket + VPIN + percentile -----------
    def _process_symbol(self, symbol: str, snapshot: _Snapshot) -> None:
        # Advance the flow-toxicity bookkeeping on this bar's close BEFORE the
        # base runs ride+entry, mirroring the base's own time-key dedup so a
        # repeated boundary firing never double-buckets a bar.  Everything in
        # ``_advance_toxicity`` uses only this bar's OHLCV and PAST closes
        # (``item.closes`` still holds history up to the PREVIOUS bar here,
        # since the base has not yet appended the current one), so deciding
        # on it at this bar's close is causal, not look-ahead.
        item = self._state.get(symbol)
        if item is not None:
            key = time_key(snapshot.time)
            if not (key and key == item.last_time_key):
                self._advance_toxicity(symbol, item, snapshot)
        super()._process_symbol(symbol, snapshot)

    def _advance_toxicity(self, symbol: str, item: _RiderState, snapshot: _Snapshot) -> None:
        extra = self._extra.get(symbol)
        if extra is None:
            return
        close = safe_float(snapshot.close)
        if close is None or close <= self.min_price:
            return
        volume_raw = safe_float(snapshot.volume)
        volume = max(0.0, volume_raw) if volume_raw is not None else 0.0
        extra.volumes.append(volume)
        prior_closes = list(item.closes)
        if not prior_closes:
            # First bar for this symbol: no prior close to classify against.
            return
        price_change = close - prior_closes[-1]
        sigma = _price_change_sigma(prior_closes, window=self.vol_window)
        buy_fraction = bulk_volume_buy_fraction(price_change, sigma)
        if buy_fraction is None:
            # Sigma not yet available (early history): classify this bar's
            # volume as neutral (zero imbalance contribution) rather than
            # stalling the bucket clock until a full vol_window has built up.
            buy_fraction = 0.5
        buy_volume = volume * buy_fraction
        sell_volume = volume * (1.0 - buy_fraction)
        bucket_size = _rolling_bucket_size(list(extra.volumes), bucket_mult=self.bucket_mult)
        new_state, completed = accumulate_volume_bucket(
            (extra.bucket_buy, extra.bucket_sell),
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            bucket_size=bucket_size,
        )
        extra.bucket_buy, extra.bucket_sell = new_state
        for imbalance in completed:
            extra.bucket_imbalances.append(imbalance)
        if len(extra.bucket_imbalances) < self.n_buckets:
            extra.last_vpin = None
            extra.last_percentile = None
            return
        vpin = vpin_from_buckets(list(extra.bucket_imbalances)[-self.n_buckets :])
        extra.last_vpin = vpin
        if vpin is None:
            extra.last_percentile = None
            return
        extra.vpin_history.append(vpin)
        extra.last_percentile = percentile_rank(list(extra.vpin_history), window=self.vpin_history)

    # -- gating ----------------------------------------------------------------
    def _symbol_for(self, item: _RiderState) -> str | None:
        for symbol, candidate in self._state.items():
            if candidate is item:
                return symbol
        return None

    def _gated_direction(self, symbol: str, item: _RiderState) -> str:
        """Return ``"LONG"``/``"SHORT"``/``""`` after the toxicity + trend gate."""
        extra = self._extra.get(symbol)
        if extra is None or extra.last_percentile is None:
            return ""
        percentile = extra.last_percentile
        if self.toxicity_mode == "avoid":
            gate = percentile <= self.toxicity_calm
        else:
            gate = percentile >= self.toxicity_entry
        if not gate:
            return ""
        trend = _trend_sign(
            list(item.closes),
            trend_lookback=self.trend_lookback,
            min_trend_roc=self.min_trend_roc,
        )
        if trend > 0:
            return "LONG"
        if trend < 0:
            return "SHORT"
        return ""

    def _entry_decision(self, item: _RiderState) -> str:
        symbol = self._symbol_for(item)
        if symbol is None:
            return ""
        return self._gated_direction(symbol, item)

    def _should_pyramid(self, item: _RiderState) -> bool:
        symbol = self._symbol_for(item)
        if symbol is None:
            return False
        return self._gated_direction(symbol, item) == item.mode

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        symbol = self._symbol_for(item)
        extra = self._extra.get(symbol) if symbol is not None else None
        return {
            "vpin": float(extra.last_vpin)
            if extra is not None and extra.last_vpin is not None
            else None,
            "toxicity_percentile": float(extra.last_percentile)
            if extra is not None and extra.last_percentile is not None
            else None,
            "toxicity_mode": self.toxicity_mode,
        }

    # -- state -------------------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["vpin_state"] = {
            symbol: {
                "bucket_imbalances": list(extra.bucket_imbalances),
                "vpin_history": list(extra.vpin_history),
                "volumes": list(extra.volumes),
                "bucket_buy": extra.bucket_buy,
                "bucket_sell": extra.bucket_sell,
                "last_vpin": extra.last_vpin,
                "last_percentile": extra.last_percentile,
            }
            for symbol, extra in self._extra.items()
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        raw = state.get("vpin_state") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        for symbol, payload in raw.items():
            extra = self._extra.get(symbol)
            if extra is None or not isinstance(payload, dict):
                continue
            _safe_restore_deque(extra.bucket_imbalances, payload.get("bucket_imbalances"))
            _safe_restore_deque(extra.vpin_history, payload.get("vpin_history"))
            _safe_restore_deque(extra.volumes, payload.get("volumes"))
            bucket_buy = safe_float(payload.get("bucket_buy"))
            extra.bucket_buy = bucket_buy if bucket_buy is not None else 0.0
            bucket_sell = safe_float(payload.get("bucket_sell"))
            extra.bucket_sell = bucket_sell if bucket_sell is not None else 0.0
            extra.last_vpin = safe_float(payload.get("last_vpin"))
            extra.last_percentile = safe_float(payload.get("last_percentile"))


# Per-timeframe candidate variants for VpinToxicityRiderStrategy, prepared for
# a later atomic wiring pass into strategy_factory.candidate_library (see the
# deferred @register comment above).  Mirrors the format of the existing
# _ADAPTIVE_TREND_RIDER_SLICE / _TAIL_INDEX_REGIME_RIDER_SLICE tables: EXACTLY
# ONE variant per timeframe, windows scaled down and trail/hold tuned wider as
# the bar length grows so each timeframe still covers a comparable wall-clock
# lookback.
_VPIN_TOXICITY_RIDER_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "toxicity_fast",
            "bucket_mult": 1.0,
            "n_buckets": 30,
            "vpin_history": 100,
            "toxicity_entry": 0.80,
            "toxicity_calm": 0.20,
            "toxicity_mode": "confirm",
            "trend_lookback": 12,
            "min_trend_roc": 0.0,
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
            "variant": "toxicity_core",
            "bucket_mult": 1.0,
            "n_buckets": 40,
            "vpin_history": 120,
            "toxicity_entry": 0.80,
            "toxicity_calm": 0.20,
            "toxicity_mode": "confirm",
            "trend_lookback": 20,
            "min_trend_roc": 0.0,
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
            "variant": "toxicity_swing",
            "bucket_mult": 1.1,
            "n_buckets": 50,
            "vpin_history": 100,
            "toxicity_entry": 0.82,
            "toxicity_calm": 0.18,
            "toxicity_mode": "confirm",
            "trend_lookback": 24,
            "min_trend_roc": 0.0,
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
            "variant": "toxicity_macro",
            "bucket_mult": 1.2,
            "n_buckets": 50,
            "vpin_history": 80,
            "toxicity_entry": 0.85,
            "toxicity_calm": 0.15,
            "toxicity_mode": "confirm",
            "trend_lookback": 20,
            "min_trend_roc": 0.0,
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
    "VpinToxicityRiderStrategy",
]
