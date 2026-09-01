"""ADF-gated own-price mean-reversion rider (stationarity-screened fade).

One per-symbol single-asset DIRECTIONAL sleeve that subclasses the proven
:class:`_ReturnRiderBase` (ATR trailing stop + pyramiding + vol-target sizing)
and adds a signal family the registry does not carry: mean-reversion entries
that are ONLY armed while the trailing price window is STATISTICALLY
mean-reverting.

THEORY / PROVENANCE.  Absorbed from the precious-metal research repo's
from-scratch cointegration toolkit (``python_adf_test`` in ``VECM.ipynb``) and
its docs' own top follow-up item: reversion trades there fire on z-score
extremes with no quantitative screen for WHETHER the series currently reverts.
This sleeve implements that missing screen via
:mod:`lumina_quant.indicators.stationarity`: an augmented Dickey-Fuller
t-statistic (pure-Python least squares, trailing window only) must reject a
unit root at ``adf_significance``, AND the AR(1) mean-reversion half-life
``-ln(2)/ln(phi)`` must be economically short (``<= max_half_life_bars``), so
the expected reversion horizon actually fits the sleeve's holding window.

ENTRY.  Only while both gates hold, the trailing log-close z-score is faded:
``z <= -zscore_entry`` arms a LONG ride back toward the mean and ``z >=
zscore_entry`` arms a SHORT (when ``allow_short``).  The ride/exit side is the
inherited ATR trailing stop + pyramiding, so a reversion that keeps running is
not cut by a fixed target.

DISTINCT from :class:`MeanReversionStdStrategy` /
:class:`VWAPCompressionReversionStrategy` / the dispersion-conditioned
reversion sleeves (unconditional z-score or compression triggers) and from
:class:`PairSpreadZScoreStrategy` / ``TimeframePairZScoreReversionStrategy``
(cross-asset hedged spreads): the fade here is gated by an explicit
unit-root REJECTION plus a half-life budget on the symbol's OWN price — no
other registered sleeve runs a stationarity test.  OHLCV-only, per-symbol
single-asset, ``decision_cadence_seconds >= 1800``, ``None``-safe graceful
guards (never raise).  Data-local (no fetching, no artifact writes) and must
still be validated on the data-bearing machine.
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.rolling_stats import sample_std
from lumina_quant.indicators.stationarity import (
    adf_critical_value,
    adf_t_statistic,
    ar1_half_life,
)
from lumina_quant.strategies.external_alpha_sleeves import _EPS
from lumina_quant.strategies.return_rider_alpha_sleeves import (
    _ReturnRiderBase,
    _RiderCommon,
    _RiderState,
)
from lumina_quant.tuning import HyperParam


@register("strategy", "AdfGatedReversionRiderStrategy", interface="event_driven")
class AdfGatedReversionRiderStrategy(_ReturnRiderBase):
    """Fade log-price z-score extremes ONLY while an ADF gate says they revert.

    Over a trailing ``stat_window`` of log closes the sleeve requires (a) the
    augmented Dickey-Fuller t-statistic to reject a unit root at
    ``adf_significance`` and (b) the AR(1) half-life to be at most
    ``max_half_life_bars``.  Only then is the z-score faded: ``z <=
    -zscore_entry`` arms a LONG, ``z >= zscore_entry`` a SHORT (when
    ``allow_short``), ridden with the inherited ATR trailing stop +
    pyramiding.

    DISTINCT from the unconditional reversion sleeves (plain z-score /
    compression triggers) and the pair-spread sleeves (cross-asset hedged
    spreads): the novelty is the explicit stationarity screen + half-life
    budget on the symbol's own price.  OHLCV-only, per-symbol single-asset,
    ``decision_cadence_seconds >= 1800``.
    """

    decision_cadence_seconds = 1800
    preferred_contract = "market_window"
    uses_timeframe_aggregator = False

    strategy_name = "AdfGatedReversionRiderStrategy"
    strategy_id = "adf_gated_reversion_rider"

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "stat_window": HyperParam.integer("stat_window", default=96, low=24, high=4096),
            "adf_lags": HyperParam.integer("adf_lags", default=1, low=0, high=8),
            "adf_significance": HyperParam.string("adf_significance", default="5%", tunable=False),
            "zscore_entry": HyperParam.floating("zscore_entry", default=1.5, low=0.1, high=10.0),
            "max_half_life_bars": HyperParam.integer(
                "max_half_life_bars", default=48, low=1, high=100000
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
        self.stat_window = max(24, int(resolved["stat_window"]))
        self.adf_lags = max(0, min(8, int(resolved["adf_lags"])))
        significance = str(resolved["adf_significance"])
        self.adf_significance = significance if significance in {"1%", "5%", "10%"} else "5%"
        self.zscore_entry = max(0.1, float(resolved["zscore_entry"]))
        self.max_half_life_bars = max(1, int(resolved["max_half_life_bars"]))

    def _common_config(self, resolved: dict[str, Any]) -> _RiderCommon:
        return self._resolve_common(resolved, extra_window=self.stat_window + 2)

    def _reversion_snapshot(self, item: _RiderState) -> tuple[float, float, float] | None:
        """Return ``(z, adf_t, half_life)`` when BOTH stationarity gates pass."""
        closes = list(item.closes)
        if len(closes) < self.stat_window:
            return None
        window = closes[-self.stat_window :]
        if any(value <= _EPS for value in window):
            return None
        logs = [math.log(value) for value in window]
        adf_stat = adf_t_statistic(logs, lags=self.adf_lags)
        critical = adf_critical_value(self.adf_significance)
        if adf_stat is None or critical is None or adf_stat > critical:
            return None
        half_life = ar1_half_life(logs)
        if half_life is None or half_life > float(self.max_half_life_bars):
            return None
        std = sample_std(logs)
        if std is None or std <= _EPS:
            return None
        z = (logs[-1] - mean(logs)) / std
        if not math.isfinite(z):
            return None
        return (z, adf_stat, half_life)

    def _reversion_direction(self, item: _RiderState) -> str:
        snapshot = self._reversion_snapshot(item)
        if snapshot is None:
            return ""
        z = snapshot[0]
        if z <= -self.zscore_entry:
            return "LONG"
        if z >= self.zscore_entry:
            return "SHORT"
        return ""

    def _entry_decision(self, item: _RiderState) -> str:
        return self._reversion_direction(item)

    def _should_pyramid(self, item: _RiderState) -> bool:
        return self._reversion_direction(item) == item.mode

    def _entry_metadata(self, item: _RiderState) -> dict[str, Any]:
        snapshot = self._reversion_snapshot(item)
        if snapshot is None:
            return {"reversion_z": None, "adf_t_stat": None, "half_life_bars": None}
        z, adf_stat, half_life = snapshot
        return {
            "reversion_z": float(z),
            "adf_t_stat": float(adf_stat),
            "half_life_bars": float(half_life),
        }


__all__ = [
    "AdfGatedReversionRiderStrategy",
]
