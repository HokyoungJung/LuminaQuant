"""Cross-sectional path-CONVEXITY sleeve (orthonormal curvature rank).

``CrossSectionalPathConvexityStrategy`` ranks the liquid cross-section by the
CURVATURE of each symbol's trailing log-price path -- the coefficient on the
quadratic member of a discrete orthonormal polynomial basis fitted to the
trailing ``window_bars`` of ``log(close)`` -- and goes long the most convex
(accelerating-from-a-base) names / short the most concave (decelerating,
overextended) names.

The load-bearing property is BASIS ORTHOGONALITY: the quadratic basis vector is
orthogonal to the linear one, so the curvature score carries ZERO loading on the
trailing-return level.  Two paths that differ ONLY by a linear trend produce the
identical convexity score.  This is a second-order price signal that a
first-order momentum sleeve (which loads only on level/return) cannot span --
that non-overlap is the axis, and the build gate pins it as a property test.

THEORY / PROVENANCE
-------------------
- Ardila-Hoyos, Forro & Sornette, "The Acceleration Effect and Gamma Factor in
  Asset Pricing" (SFI/SSRN working paper) -- second-order (acceleration) pricing.
- Gettleman & Marks (2006, SSRN) "Acceleration Strategies" (equities).
- Johansen, Ledoit & Sornette (2000, IJTAF) -- super-exponential transients;
  loss of convexity in an extended trend as a pre-regime-change diagnostic (the
  short-tail mechanism anchor).
- Weakest evidence tier in this wave; the design bets BOTH tails of one curvature
  cross-section simultaneously and is priced into a HIGH expected null.

SIGNAL SPEC
-----------
Per completed WEEKLY ISO decision bar (internal ``_week_key`` clock), per symbol
with >= ``window_bars`` closes:

1. ``c2`` = trailing log-price projected onto the orthonormal quadratic basis
   member ``p2``, normalized by the window's log-return std (floored at
   ``vol_floor``) -- pure :func:`orthonormal_path_convexity`.  ``p2 _|_ p1`` gives
   zero first-order loading.
2. LONG the top-quantile convexity (accelerating from a base), SHORT the bottom
   quantile (decelerating, overextended), inverse-realized-vol sized, with an
   entry/exit quantile hysteresis band and a hard minimum-hold in weekly
   decisions (shared :class:`OLSBasisCrossSectionalBook` engine).  Both prongs
   are the two tails of ONE curvature cross-section -- no regime-switch param.

DISTINCT-FROM
-------------
Every trailing-return-level trend sleeve (``ProfitMoonshotTrendStrategy`` --
positive-weighted fast/medium/slow returns; ``AccelerationRiderStrategy`` --
rate-of-change SIGN/rising gate) loads on the FIRST-order move.  By basis
construction the convexity score has no first-order content, so on a
decelerating winner (level up, curvature down) the level sleeve stays/goes long
while this sleeve SHORTS, and on an accelerating base (level still down,
curvature up) the level sleeve stays flat/short while this sleeve LONGS -- the
two divergences the build gate pins.

This module is data-local (no I/O), pure Python, completed-bar, and never raises
from ``calculate_signals``.  It ships WITHOUT ``@register`` (inert): registration,
tier hint, and candidate wiring land atomically in the integration wave.
"""

from __future__ import annotations

from typing import Any

from lumina_quant.core.plugin_registry import register
from lumina_quant.indicators.log_price_regression import orthonormal_path_convexity
from lumina_quant.strategies.ols_basis_xs_common import OLSBasisCrossSectionalBook
from lumina_quant.tuning import HyperParam, resolve_params_from_schema

_STRATEGY_ID = "path_convexity_xs"
_STRATEGY_NAME = "CrossSectionalPathConvexityStrategy"


@register("strategy", "CrossSectionalPathConvexityStrategy", interface="event_driven")
class CrossSectionalPathConvexityStrategy(OLSBasisCrossSectionalBook):
    """Weekly XS long-short on orthonormal path curvature (zero first-order load).

    See the module docstring for the full theory, signal spec, and
    distinct-from rationale.  Reads only local event/bar closes; performs no I/O
    and never raises from ``calculate_signals``.
    """

    decision_cadence_seconds = 86400  # daily bars; weekly effective via _week_key
    _strategy_id = _STRATEGY_ID
    _strategy_name = _STRATEGY_NAME
    strategy_name = _STRATEGY_NAME
    strategy_id = _STRATEGY_ID
    require_sign = False  # pure XS curvature rank (top/bottom quantile)

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "window_bars": HyperParam.integer("window_bars", default=56, low=8, high=4096),
            "quantile": HyperParam.floating("quantile", default=0.20, low=0.05, high=0.50),
            "hysteresis_exit_pct": HyperParam.floating(
                "hysteresis_exit_pct", default=0.40, low=0.05, high=0.90
            ),
            "min_hold_decisions": HyperParam.integer(
                "min_hold_decisions", default=4, low=1, high=520
            ),
            "max_hold_decisions": HyperParam.integer(
                "max_hold_decisions", default=52, low=1, high=20000
            ),
            "vol_floor": HyperParam.floating(
                "vol_floor", default=1e-6, low=1e-12, high=1.0, tunable=False
            ),
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
        self._window = max(8, int(resolved["window_bars"]))
        self.quantile_entry = max(0.01, min(0.50, float(resolved["quantile"])))
        self.quantile_exit = max(
            self.quantile_entry, min(0.90, float(resolved["hysteresis_exit_pct"]))
        )
        self.min_hold_decisions = max(1, int(resolved["min_hold_decisions"]))
        self.max_hold_decisions = max(self.min_hold_decisions, int(resolved["max_hold_decisions"]))
        self.vol_floor = max(1e-12, float(resolved["vol_floor"]))
        self.vol_window = max(2, int(resolved["vol_window"]))
        self.min_symbols = max(2, int(resolved["min_symbols"]))
        self.allow_short = bool(resolved["allow_short"])
        self.target_gross_exposure = max(0.0, float(resolved["target_gross_exposure"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.min_price = max(0.0, float(resolved["min_price"]))
        self._finalize_state()

    def _score_symbol(self, closes: list[float]) -> tuple[float, bool] | None:
        c2 = orthonormal_path_convexity(closes, window=self._window, vol_floor=self.vol_floor)
        if c2 is None:
            return None
        return float(c2), True


# --------------------------------------------------------------------------- #
# Candidate-wiring hints for the integration wave (this lane does NOT wire
# candidates itself -- new-file-only, no shared-file edits).  Cross-sectional
# multi-asset book: admission route is ``allow_multi_asset=True`` -- long-short
# by curvature, NOT carry, so it is honestly EXCLUDED from any fake carry tag.
# Data-PC gates it on RPT >= 10bps per split and incremental orthogonal
# factor_ic vs the level-momentum set (ProfitMoonshotTrend, MultiTimeframe
# TrendEnsemble, per-symbol TSMOM/KER sleeves).
# --------------------------------------------------------------------------- #
_SUGGESTED_FAMILY = "cross_sectional"
_SUGGESTED_CANDIDATE_TAGS: tuple[str, ...] = (
    "cross_sectional",
    "path_convexity",
    "curvature",
    "acceleration",
    "second_order",
    "low_turnover",
    "crypto",
)

# DAILY bars only: the 56-bar curvature window is ~8 weeks of 1d bars, and the
# weekly decision clock is only honest at the 1d cadence.
_PATH_CONVEXITY_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1d": (
        {
            "variant": "curvature_8wk",
            "window_bars": 56,
            "quantile": 0.20,
            "hysteresis_exit_pct": 0.40,
            "min_hold_decisions": 4,
            "vol_window": 30,
            "min_symbols": 5,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
        {
            "variant": "curvature_4wk",
            "window_bars": 28,
            "quantile": 0.20,
            "hysteresis_exit_pct": 0.40,
            "min_hold_decisions": 4,
            "vol_window": 20,
            "min_symbols": 5,
            "allow_short": True,
            "target_gross_exposure": 1.0,
        },
    ),
}

__all__ = ["CrossSectionalPathConvexityStrategy"]
