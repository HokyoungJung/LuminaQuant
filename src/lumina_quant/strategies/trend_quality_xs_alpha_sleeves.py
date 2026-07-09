"""Cross-sectional regression trend-QUALITY sleeve (signed-R^2 rank).

``CrossSectionalRegressionTrendQualityStrategy`` ranks the liquid cross-section
by the QUALITY of each symbol's trailing log-price trend -- the signed
coefficient of determination ``sign(slope) * R^2`` of an ordinary-least-squares
fit of ``log(close)`` on a bar-index grid -- and goes long the smoothest
uptrends / short the smoothest downtrends, excluding names whose fit quality is
below a floor (a jumpy or discrete move is treated as already-priced
information, not a tradable trend).

THEORY / PROVENANCE
-------------------
- Da, Gurun & Warachka (2014, RFS) "Frog in the Pan": momentum is stronger and
  more persistent when the underlying information (and therefore the price path)
  arrives continuously/smoothly rather than in salient discrete jumps.  Ranking
  by regression fit quality isolates exactly that continuous-information path.
- Grinblatt & Moskowitz (2004, JFE): return consistency strengthens momentum.
- Levine & Pedersen (2016, FAJ) "Which Trend Is Your Friend?": a regression-fit
  trend-strength lineage.
- DECAY caveat (Liu & Tsyvinski 2021, RFS): crypto TSMOM is a decaying factor;
  the trend-SIGN leg leans on it, so this sleeve is a falsification probe of a
  trend-quality tilt on a decaying axis, not a promise of edge.

SIGNAL SPEC
-----------
Per completed WEEKLY ISO decision bar (internal ``_week_key`` clock), per symbol
with >= ``formation_bars`` closes:

1. ``signed_r2 = sign(slope) * R^2`` of ``log(close)`` on ``[0, formation_bars)``
   (pure :func:`signed_trend_quality`); the ``t_stat`` score mode substitutes a
   ``sqrt`` -normalized slope t-statistic.
2. A symbol with ``R^2 < min_quality`` is EXCLUDED from BOTH books (still counts
   toward ``min_symbols``) -- the quality floor.
3. LONG the top quantile (smoothest uptrends), SHORT the bottom quantile
   (smoothest downtrends), inverse-realized-vol sized, with an entry/exit
   quantile hysteresis band and a hard minimum-hold in weekly decisions (shared
   :class:`OLSBasisCrossSectionalBook` engine).

DISTINCT-FROM
-------------
The occupied Kaufman-efficiency-ratio path-efficiency axis
(``TrendEfficiencyMomentumStrategy``, ``LowTurnoverTrendPersistenceStrategy``)
scores by KER * trend-sign; KER measures net-move / path-length efficiency, not
regression fit.  On a zigzag-but-net-directional path KER collapses (choppy)
while the signed-R^2 of a strong linear fit stays high; on a flat-then-jump path
KER saturates (monotone) while the signed-R^2 collapses (a single jump explains
little variance).  Those are the two divergences the build gate pins.

This module is data-local (no I/O), pure Python, completed-bar, and never raises
from ``calculate_signals``.  It ships WITHOUT ``@register`` (inert): registration,
tier hint, and candidate wiring land atomically in the integration wave.
"""

from __future__ import annotations

import math
from typing import Any

from lumina_quant.indicators.log_price_regression import (
    signed_trend_quality,
    trend_slope_t_stat,
)
from lumina_quant.strategies.ols_basis_xs_common import OLSBasisCrossSectionalBook
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "trend_quality_xs"
_STRATEGY_NAME = "CrossSectionalRegressionTrendQualityStrategy"

_SCORE_MODES = ("signed_r2", "t_stat")


class CrossSectionalRegressionTrendQualityStrategy(OLSBasisCrossSectionalBook):
    """Weekly XS long-short on signed-R^2 trend quality with a fit-quality floor.

    See the module docstring for the full theory, signal spec, and
    distinct-from rationale.  Reads only local event/bar closes; performs no I/O
    and never raises from ``calculate_signals``.
    """

    decision_cadence_seconds = 86400  # daily bars; weekly effective via _week_key
    _strategy_id = _STRATEGY_ID
    _strategy_name = _STRATEGY_NAME
    strategy_name = _STRATEGY_NAME
    strategy_id = _STRATEGY_ID
    require_sign = True

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "formation_bars": HyperParam.integer("formation_bars", default=56, low=8, high=4096),
            "min_quality": HyperParam.floating("min_quality", default=0.30, low=0.0, high=1.0),
            "score_mode": HyperParam.categorical(
                "score_mode", default="signed_r2", choices=list(_SCORE_MODES)
            ),
            "quantile_entry_pct": HyperParam.floating(
                "quantile_entry_pct", default=0.20, low=0.05, high=0.50
            ),
            "quantile_exit_pct": HyperParam.floating(
                "quantile_exit_pct", default=0.40, low=0.05, high=0.90
            ),
            "min_hold_weeks": HyperParam.integer("min_hold_weeks", default=4, low=1, high=520),
            "max_hold_weeks": HyperParam.integer("max_hold_weeks", default=52, low=1, high=20000),
            "vol_window": HyperParam.integer("vol_window", default=30, low=2, high=4096),
            "min_symbols": HyperParam.integer("min_symbols", default=5, low=2, high=512),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
            "target_gross_exposure": HyperParam.floating(
                "target_gross_exposure", default=1.0, low=0.0, high=3.0
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
        self._window = max(8, int(resolved["formation_bars"]))
        self.min_quality = max(0.0, min(1.0, float(resolved["min_quality"])))
        mode = str(resolved["score_mode"]).lower()
        self.score_mode = mode if mode in _SCORE_MODES else "signed_r2"
        self.quantile_entry = max(0.01, min(0.50, float(resolved["quantile_entry_pct"])))
        self.quantile_exit = max(
            self.quantile_entry, min(0.90, float(resolved["quantile_exit_pct"]))
        )
        self.min_hold_decisions = max(1, int(resolved["min_hold_weeks"]))
        self.max_hold_decisions = max(self.min_hold_decisions, int(resolved["max_hold_weeks"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.allow_short = bool(resolved["allow_short"])
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        self._finalize_state()

    def _score_symbol(self, closes: list[float]) -> tuple[float, bool] | None:
        quality = signed_trend_quality(closes, window=self._window)
        if quality is None:
            return None
        signed_r2, r2, slope = quality
        tradeable = r2 >= self.min_quality
        if self.score_mode == "t_stat":
            t_stat = trend_slope_t_stat(r2, self._window)
            if t_stat is None:
                return None
            sign = 1.0 if slope > 0.0 else (-1.0 if slope < 0.0 else 0.0)
            score = sign * t_stat / math.sqrt(self._window)
        else:
            score = signed_r2
        return float(score), bool(tradeable)


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the integration wave (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits).  Cross-sectional
# multi-asset book: admission route is ``allow_multi_asset=True`` -- long-short
# by trend quality, NOT carry, so it is honestly EXCLUDED from any fake carry
# tag.  Data-PC gates it on RPT >= 10bps per split and incremental orthogonal
# factor_ic vs {TrendEfficiencyMomentum, LowTurnoverTrendPersistence}.
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "trend_quality",
    "regression",
    "frog_in_the_pan",
    "signed_r2",
    "low_turnover",
    "crypto",
)

# DAILY bars only: the 56-bar formation window is ~8 weeks of 1d bars, and the
# weekly decision clock is only honest at the 1d cadence.
_TREND_QUALITY_XS_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": (
        {
            "variant": "signed_r2_8wk",
            "formation_bars": 56,
            "min_quality": 0.30,
            "score_mode": "signed_r2",
            "quantile_entry_pct": 0.20,
            "quantile_exit_pct": 0.40,
            "min_hold_weeks": 4,
            "vol_window": 30,
            "min_symbols": 5,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
        {
            "variant": "signed_r2_26wk",
            "formation_bars": 182,
            "min_quality": 0.30,
            "score_mode": "signed_r2",
            "quantile_entry_pct": 0.20,
            "quantile_exit_pct": 0.40,
            "min_hold_weeks": 4,
            "vol_window": 45,
            "min_symbols": 5,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
    ),
}

__all__ = ["CrossSectionalRegressionTrendQualityStrategy"]
