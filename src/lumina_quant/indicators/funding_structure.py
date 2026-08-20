"""Funding-rate structure primitives: momentum, term-structure spread, implied basis.

Fills the census-verified indicator gap "funding-rate term-structure/momentum
indicators" with FOUR latest-scalar primitives over the ``feature_points``
funding-rate print series (perp funding settles every 8h on Binance USD-M):

- :func:`funding_momentum` -- EWMA of funding first differences + rolling OLS
  level slope, EXTRACTED VERBATIM (parity-locked) from the previously private
  ``_funding_momentum`` in
  ``strategies/cross_sectional_funding_momentum_carry.py``; the consuming
  sleeve now imports this function, so signals are byte-identical.
- :func:`funding_term_structure_spread` -- mean of the last ``short_window=3``
  funding prints minus the mean of the last ``long_window=21`` prints (1d vs 7d
  at the 8h cadence): the single-tenor perp analog of a carry term-structure
  slope (is CURRENT carry rich or poor vs its own trailing regime).
- :func:`funding_implied_basis_bps` -- the latest funding print expressed in
  basis points per funding interval (the standing, lagged-TWAP funding leg).
- :func:`basis_funding_gap_bps` -- instantaneous ``basis_bps(mark, index)``
  minus :func:`funding_implied_basis_bps`: the funding-expectation error the
  ``BasisFundingGapConvergenceStrategy`` sleeve fades.

External prior (pre-registered):
- Koijen, Moskowitz, Pedersen, Vrugt (2018, JFE) "Carry" -- carry
  slope/term structure predicts returns across asset classes.
- Schmeling, Schrimpf, Todorov (BIS WP 2023) "Crypto carry" -- perp funding as
  the crypto carry leg, persistent and partly predictable.
- Ackerer, Hugonnier & Jermann, "Perpetual Futures Pricing" (2023 wp/SSRN):
  perp premium equals the expectation of future funding flows, making
  premium-vs-funding disagreement a convergence object; plus the Binance
  funding specification itself (funding = clamp(premium-index TWAP) +
  interest), which makes the basis-funding gap a near-mechanical predictor of
  the next funding print.

EXPECTED NULL (verbatim, pre-registered): "Extraction leg: byte-identical, no
null. TS-agreement leg: the spread adds no selectivity -- funding momentum
already embeds the level shift, so the agreement filter removes entries at
random and net Sharpe of the carry sleeve is unchanged or slightly worse from
foregone trades."

FALSIFYING MEASUREMENT (verbatim, pre-registered): "Extraction: parity
fixtures -- indicator output equals strategy-private ``_funding_momentum``
exactly; sleeve tests byte-identical. TS-agreement (data-PC, flag ON vs OFF):
paired per-split net-Sharpe delta of CrossSectionalFundingMomentumCarry at the
10/15/20/30bps grid; kill the flag if median delta <= 0 or trade count drops
>40% (filter too destructive)."

Conventions: pure stdlib (no numpy/scipy/sklearn/statsmodels), latest-scalar
``float | None``, never-raise (garbage/short/degenerate input propagates
``None``).  The 3/21-print windows are PRE-REGISTERED constants, not a tuning
axis.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from lumina_quant.indicators.alpha_features import basis_bps, finite_floats
from lumina_quant.indicators.common import safe_float

_EPS = 1e-12

__all__ = [
    "basis_funding_gap_bps",
    "funding_implied_basis_bps",
    "funding_momentum",
    "funding_term_structure_spread",
]


def _ols_slope(values: list[float]) -> float:
    """Least-squares slope of ``values`` vs an index 0..n-1 (0.0 if degenerate)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / float(n)
    num = 0.0
    den = 0.0
    for i, value in enumerate(values):
        dx = i - mean_x
        num += dx * (value - mean_y)
        den += dx * dx
    if den <= _EPS:
        return 0.0
    return num / den


def funding_momentum(
    funding: Iterable[Any] | None, *, diff_window: int, slope_window: int, ewma_span: int
) -> float | None:
    """Delta-funding momentum: EWMA of first differences + rolling OLS slope.

    Uses strictly the funding prints provided (the just-closed interval is the
    last element -- observable, no lookahead).  Returns ``None`` when there is
    too little history to form a first difference.

    PARITY LOCK: the arithmetic below is extracted VERBATIM from the previously
    strategy-private ``_funding_momentum`` in
    ``strategies/cross_sectional_funding_momentum_carry.py``; on the finite
    float lists that sleeve produces the output is bit-for-bit identical (the
    only addition is the never-raise ``finite_floats`` coercion, a no-op on
    such input).
    """
    values = finite_floats(funding)
    if len(values) < 2:
        return None
    diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
    diff_tail = diffs[-max(1, int(diff_window)) :]
    # EWMA of the first differences (most-recent weighted highest).
    span = max(1, int(ewma_span))
    alpha = 2.0 / (span + 1.0)
    ewma = diff_tail[0]
    for value in diff_tail[1:]:
        ewma = alpha * value + (1.0 - alpha) * ewma
    # Rolling OLS slope of the funding LEVEL over the trailing slope window.
    level_tail = values[-max(2, int(slope_window)) :]
    slope = _ols_slope(level_tail)
    return float(ewma) + float(slope)


def funding_term_structure_spread(
    funding: Iterable[Any] | None, *, short_window: int = 3, long_window: int = 21
) -> float | None:
    """Short-minus-long mean of the trailing funding prints (carry-regime slope).

    ``mean(last short_window prints) - mean(last long_window prints)`` -- with
    the pre-registered defaults (3 vs 21 prints = 1d vs 7d at the 8h funding
    cadence) a POSITIVE spread means current carry is rich vs its own trailing
    regime, a NEGATIVE spread means it is poor.  Returns ``None`` when fewer
    than ``long_window`` finite prints are available or the windows are
    degenerate (``short_window >= long_window``), never raises.
    """
    try:
        short = int(short_window)
        long = int(long_window)
    except Exception:
        return None
    if short < 1 or long <= short:
        return None
    values = finite_floats(funding)
    if len(values) < long:
        return None
    short_mean = sum(values[-short:]) / float(short)
    long_mean = sum(values[-long:]) / float(long)
    return float(short_mean - long_mean)


def funding_implied_basis_bps(funding_rate: Any) -> float | None:
    """Latest funding print expressed in basis points per funding interval.

    Binance funding is a decimal rate charged once per interval (e.g. 0.0001 =
    1bp per 8h); this is the standing (lagged premium-TWAP) funding leg that
    :func:`basis_funding_gap_bps` compares the instantaneous basis against.
    ``None`` on missing/non-finite input, never raises.
    """
    rate = safe_float(funding_rate)
    if rate is None:
        return None
    return float(rate * 10_000.0)


def basis_funding_gap_bps(mark_price: Any, index_price: Any, funding_rate: Any) -> float | None:
    """Instantaneous basis minus the funding-implied basis, in basis points.

    ``basis_bps(mark, index) - funding_implied_basis_bps(funding_rate)``: a
    large POSITIVE gap means the instantaneous mark-index premium is richer
    than the standing funding print, so the NEXT funding print mechanically
    catches up and the basis itself tends to mean-revert (the convergence
    object of Ackerer-Hugonnier-Jermann).  ``None`` when either leg is
    unavailable, never raises.
    """
    basis = basis_bps(safe_float(mark_price), safe_float(index_price))
    implied = funding_implied_basis_bps(funding_rate)
    if basis is None or implied is None:
        return None
    return float(basis - implied)
