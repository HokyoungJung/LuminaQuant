"""Spectral dominant-cycle timing rider (trailing periodogram phase).

One per-symbol single-asset DIRECTIONAL sleeve that subclasses the proven
:class:`_ReturnRiderBase` (ATR trailing stop + pyramiding + vol-target sizing;
winners run, no fixed take-profit) and adds a signal family the registry does
not carry: a TRAILING-WINDOW PERIODOGRAM cycle clock.

THEORY / PROVENANCE.  Absorbed from the precious-metal research repo's
``rfft_periodogram.py`` (demean -> real FFT -> amplitude spectrum -> dominant
period = argmax amplitude), which that repo uses to time supply-cycle troughs.
Re-implemented causally in :mod:`lumina_quant.indicators.spectral_cycle`: the
window is TRAILING (no full-sample fit), linearly detrended, and scanned by
direct Fourier projection per candidate period (pure Python, no FFT library).
The projection yields the dominant period ``P``, its amplitude, the cycle
PHASE at the latest bar (``0`` = crest, ``pi`` = trough), and a PURITY score
(the dominant period's share of scanned squared amplitude) that separates a
real periodicity from noise (a pure tone scores ~5x a white-noise window).

ENTRY.  When the trailing log-price window carries a sufficiently pure
dominant cycle, a LONG ride is armed just after the cycle TROUGH (normalized
phase in ``(0.5, 0.5 + entry_phase_band]`` — the upswing has just begun) and a
SHORT just after the CREST (phase in ``(0, entry_phase_band]``), optionally
requiring a minimum cycle amplitude in log units.  The winner is then ridden
with the inherited ATR trailing stop + pyramiding, so a cycle upswing that
morphs into a trend keeps paying.

DISTINCT from :class:`IntradaySeasonalMomentumRiderStrategy` /
:class:`SeasonalMicroBreakoutRiderStrategy` (fixed CALENDAR/time-of-day
seasonality buckets) and from the trend riders (:class:`KalmanTrendRiderStrategy`,
:class:`AdaptiveTrendRiderStrategy`: level/slope estimators): the trigger here
is the PHASE of a data-discovered spectral cycle whose period is re-estimated
every decision bar — no other registered sleeve measures periodicity.
OHLCV-only, per-symbol single-asset, ``decision_cadence_seconds >= 1800``,
``None``-safe graceful guards (never raise).  Data-local (no fetching, no
artifact writes) and must still be validated on the data-bearing machine.
"""

from __future__ import annotations

import math
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.spectral_cycle import cycle_phase_fraction, dominant_cycle
from lumina_quant.strategies.external_alpha_sleeves import _EPS
from lumina_quant.strategies.return_rider_alpha_sleeves import (
    _ReturnRiderBase,
    _RiderCommon,
    _RiderState,
)
from lumina_quant.tuning import HyperParam


@register("strategy", "SpectralCycleRiderStrategy", interface="event_driven")
class SpectralCycleRiderStrategy(_ReturnRiderBase):
    """Ride cycle upswings/downswings timed by a trailing periodogram phase.

    A trailing window of log closes is linearly detrended and scanned by direct
    Fourier projection for its dominant cycle (period, amplitude, phase,
    purity).  A LONG ride is armed just after the cycle trough (normalized
    phase in ``(0.5, 0.5 + entry_phase_band]``) and a SHORT just after the
    crest (phase in ``(0, entry_phase_band]``), gated by ``min_purity`` (the
    cycle must dominate the scanned spectrum) and ``min_amplitude_frac`` (the
    swing must be economically meaningful in log units).  The winner is ridden
    with the inherited ATR trailing stop + pyramiding.

    DISTINCT from the calendar-seasonality riders (fixed time-of-day buckets)
    and the trend riders (level/slope estimators): the trigger is the phase of
    a data-discovered spectral cycle — no other sleeve measures periodicity.
    OHLCV-only, per-symbol single-asset, ``decision_cadence_seconds >= 1800``.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "SpectralCycleRiderStrategy"
    strategy_id = "spectral_cycle_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            # Bounded highs keep the worst grid/Optuna combination cheap: the
            # periodogram scan is O(cycle_window x scanned_periods), so
            # 512 x 128 stays well under 10 ms per decision bar per symbol.
            "cycle_window": HyperParam.integer("cycle_window", default=128, low=16, high=512),
            "min_period": HyperParam.integer("min_period", default=8, low=2, high=128),
            "max_period": HyperParam.integer("max_period", default=48, low=4, high=128),
            "min_purity": HyperParam.floating("min_purity", default=0.18, low=0.0, high=1.0),
            "entry_phase_band": HyperParam.floating(
                "entry_phase_band", default=0.15, low=0.01, high=0.5
            ),
            "min_amplitude_frac": HyperParam.floating(
                "min_amplitude_frac", default=0.0, low=0.0, high=1.0
            ),
            "trail_atr_mult": HyperParam.floating(
                "trail_atr_mult", default=3.0, low=0.1, high=20.0
            ),
            "atr_period": HyperParam.integer("atr_period", default=14, low=2, high=1024),
            "max_adds": HyperParam.integer("max_adds", default=2, low=0, high=32),
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

    def _bind_params(self, resolved: dict[str, Any]) -> None:
        self.cycle_window = max(16, int(resolved["cycle_window"]))
        self.min_period = max(2, int(resolved["min_period"]))
        self.max_period = max(self.min_period + 1, int(resolved["max_period"]))
        self.min_purity = max(0.0, min(1.0, float(resolved["min_purity"])))
        self.entry_phase_band = max(0.01, min(0.5, float(resolved["entry_phase_band"])))
        self.min_amplitude_frac = max(0.0, float(resolved["min_amplitude_frac"]))
        # Per-symbol memo of the latest scan keyed by the bar's time_key, so
        # _entry_decision / _should_pyramid / _entry_metadata on the SAME bar
        # run the O(window x periods) periodogram only once.
        self._cycle_cache: dict[str, tuple[str, tuple[float, float, float, float] | None]] = {}

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(resolved, extra_window=self.cycle_window + 2)

    def _symbol_for(self, item: _RiderState) -> str | None:
        for symbol, candidate in self._state.items():
            if candidate is item:
                return symbol
        return None

    def _cycle_snapshot(self, item: _RiderState) -> tuple[float, float, float, float] | None:
        """Return ``(period, amplitude, phase, purity)`` for the trailing window."""
        symbol = self._symbol_for(item)
        if symbol is not None and item.last_time_key:
            cached = self._cycle_cache.get(symbol)
            if cached is not None and cached[0] == item.last_time_key:
                return cached[1]
        closes = list(item.closes)
        result: tuple[float, float, float, float] | None = None
        if len(closes) >= self.cycle_window:
            window = closes[-self.cycle_window :]
            if all(value > _EPS for value in window):
                logs = [math.log(value) for value in window]
                result = dominant_cycle(
                    logs, min_period=self.min_period, max_period=self.max_period
                )
        if symbol is not None and item.last_time_key:
            self._cycle_cache[symbol] = (item.last_time_key, result)
        return result

    def _cycle_direction(self, item: _RiderState) -> str:
        cycle = self._cycle_snapshot(item)
        if cycle is None:
            return ""
        _period, amplitude, phase, purity = cycle
        if purity < self.min_purity or amplitude < self.min_amplitude_frac:
            return ""
        fraction = cycle_phase_fraction(phase)
        if fraction is None:
            return ""
        if 0.5 < fraction <= 0.5 + self.entry_phase_band:
            return "LONG"
        if 0.0 < fraction <= self.entry_phase_band:
            return "SHORT"
        return ""

    def _entry_decision(self, item: _RiderState) -> str:
        return self._cycle_direction(item)

    def _should_pyramid(self, item: _RiderState) -> bool:
        return self._cycle_direction(item) == item.mode

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        cycle = self._cycle_snapshot(item)
        if cycle is None:
            return {"cycle_period": None, "cycle_purity": None, "cycle_phase_fraction": None}
        period, _amplitude, phase, purity = cycle
        return {
            "cycle_period": float(period),
            "cycle_purity": float(purity),
            "cycle_phase_fraction": cycle_phase_fraction(phase),
        }


__all__ = [
    "SpectralCycleRiderStrategy",
]
