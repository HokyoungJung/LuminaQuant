"""Behavioral-value indicators: salience-theory value and prospect-theory value.

Two latest-scalar behavioral valuations of a trailing daily-return history,
consumed by the weekly cross-sectional fade sleeves in
``strategies/behavioral_value_xs_alpha_sleeves.py``:

- :func:`salience_theory_value` -- Bordalo-Gennaioli-Shleifer salience
  distortion.  Investors overweight days whose return was salient RELATIVE to
  the market that day.  Per day ``d`` with own return ``r_d`` and market return
  ``m_d``, salience ``sigma_d = |r_d - m_d| / (|r_d| + |m_d| + theta)``
  (``theta = 0.1``); days are ranked by ``sigma`` descending and weighted
  ``w_d proportional to delta**rank`` (``delta = 0.7``, BGS canonical); the
  salience-theory value is ``ST = sum(w_d * r_d) - mean(r_d)`` -- the
  salience-induced distortion of the perceived vs objective mean return.
- :func:`prospect_theory_value` -- Tversky-Kahneman (1992) valuation of the
  sorted return histogram with equal outcome probabilities ``1/n``:
  ``TK = sum(pi_i * v(r_i))`` with the value function ``v(x) = x**alpha`` for
  gains and ``v(x) = -lam * (-x)**alpha`` for losses (``alpha = 0.88``,
  ``lam = 2.25``) and rank-dependent decision weights ``pi_i`` from the
  cumulative probability weighting ``w(p) = p**g / (p**g + (1-p)**g)**(1/g)``
  applied from the tails inward (``g = 0.61`` gains / ``g = 0.69`` losses).
  All constants are the TK-1992 canonical estimates and are frozen defaults.

EXTERNAL PRIOR (verbatim from the pre-registered proposals)
-----------------------------------------------------------
- Bordalo, Gennaioli & Shleifer (QJE 2012, salience theory of choice under
  risk; canonical theta=0.1, delta=0.7); Cosemans & Frehen (JFE 2021) negative
  ST premium in equities surviving MAX/skew/IVOL controls; post-2022 crypto
  replication: "Salience theory and cryptocurrency returns" (Finance Research
  Letters, ~2022) reports the same negative ST premium in coin cross-sections
  (the equity anchor is certain).
- Tversky & Kahneman (1992) parameter set; Barberis, Jin & Wang (JF 2021)
  "Prospect theory and stock market anomalies" -- TK value negatively predicts
  XS returns beyond MAX/skew/IVOL; post-2022 crypto replications of TK-value
  pricing in coin cross-sections (IRFA/FRL 2023-2024 strand; the equity anchor
  and the mechanism are the load-bearing prior).

EXPECTED NULL / FALSIFYING MEASUREMENT (verbatim, pre-registered)
-----------------------------------------------------------------
ST null: "In this 110-name liquid universe the ST distortion is subsumed by
the lottery blend (skew+MAX) already traded by LotterySkewnessStrategy:
residualized ST carries zero incremental IC, or the quintile spread dies
inside the 20-30bps cost grid."  ST falsifier: "On data-PC: weekly rank IC of
ST residualized (Gram-Schmidt) against lottery score, 21d momentum, and IVOL
over the full sample; admission requires incremental orthogonal factor_ic > 0
vs LotterySkewnessStrategy plus the standard gates (DSR>=0.90, SPA<=0.05,
net-20bps edge>0).  Residual IC <= 0 on validation-forward folds kills the
lane in one measurement."

PT null: "TK value on a 110-name liquid universe collapses onto the lottery
blend and/or onto ST value: incremental orthogonal IC vs
LotterySkewnessStrategy is zero, or it loses the pre-registered tournament
against salience-theory-value-xs."  PT falsifier: "Same single measurement as
ST lane: validation-forward weekly rank IC of TK residualized against lottery
score, momentum, IVOL, AND ST value; residual IC <= 0 or tournament loss vs ST
(when |rho|>0.6) kills it.  Standard admission gates apply on top."

CONTRACT
--------
Pure numpy + stdlib (no scipy/sklearn/statsmodels).  Latest-scalar
``float | None``; never raises on adversarial input (non-numeric, empty,
short, non-finite, mismatched lengths -> ``None``).  Deterministic: fixed
tie-breaks (stable argsort -- earlier day wins salience ties) and fixed-order
reductions.  No annualization is performed here; any annualization elsewhere
must go through ``lumina_quant.indicators.annualization``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_EPS = 1e-12


def _coerce_param(raw: Any) -> float | None:
    """Coerce an explicitly supplied scalar parameter to a finite float."""
    try:
        value = float(raw)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _coerce_1d(values: Any) -> np.ndarray | None:
    """Best-effort coercion to a 1-D float array; ``None`` on failure."""
    if values is None:
        return None
    try:
        arr = np.asarray(list(values), dtype=float)
    except Exception:
        return None
    if arr.ndim != 1:
        return None
    return arr


def salience_theory_value(
    returns: Any,
    market_returns: Any,
    theta: float = 0.1,
    delta: float = 0.7,
) -> float | None:
    """Return the BGS salience-theory value (distorted-minus-objective mean).

    ``returns`` and ``market_returns`` are positionally aligned daily return
    series (own return vs the equal-weight universe return of the SAME day).
    Both are trimmed to their common trailing length; any position where either
    value is non-finite is dropped pairwise.  Salience ties are broken by the
    stable original day order (earlier day ranks more salient).

    Args:
        returns: Own daily returns, oldest to newest.
        market_returns: Equal-weight market daily returns, same alignment.
        theta: Salience level-dampening constant (BGS canonical ``0.1``).
        delta: Salience rank-discount in ``(0, 1]`` (BGS canonical ``0.7``).

    Returns:
        ``sum(w_d * r_d) - mean(r_d)`` as a float, or ``None`` when fewer than
        two finite aligned pairs remain or the result is non-finite.  Never
        raises.
    """
    own = _coerce_1d(returns)
    market = _coerce_1d(market_returns)
    if own is None or market is None:
        return None
    count = min(own.size, market.size)
    if count < 2:
        return None
    own = own[-count:]
    market = market[-count:]
    mask = np.isfinite(own) & np.isfinite(market)
    own = own[mask]
    market = market[mask]
    n = own.size
    if n < 2:
        return None
    theta_f = _coerce_param(theta)
    delta_f = _coerce_param(delta)
    if theta_f is None or delta_f is None or theta_f <= 0.0 or not 0.0 < delta_f <= 1.0:
        return None
    sigma = np.abs(own - market) / (np.abs(own) + np.abs(market) + theta_f)
    # Rank days by salience descending; stable sort makes ties deterministic.
    order = np.argsort(-sigma, kind="stable")
    weights = delta_f ** np.arange(n, dtype=float)
    total = float(weights.sum())
    if total <= _EPS:
        return None
    weights /= total
    distorted = float(np.dot(weights, own[order]))
    objective = float(own.mean())
    value = distorted - objective
    return float(value) if math.isfinite(value) else None


def _tk_probability_weight(p: np.ndarray, gamma: float) -> np.ndarray:
    """TK-1992 probability weighting ``w(p) = p^g / (p^g + (1-p)^g)^(1/g)``."""
    p = np.clip(p, 0.0, 1.0)
    num = p**gamma
    den = (num + (1.0 - p) ** gamma) ** (1.0 / gamma)
    return num / den


def prospect_theory_value(
    returns: Any,
    alpha: float = 0.88,
    lam: float = 2.25,
    gamma_gain: float = 0.61,
    gamma_loss: float = 0.69,
) -> float | None:
    """Return the TK-1992 prospect-theory value of the sorted return histogram.

    Each observed return is one outcome with probability ``1/n`` (the
    Barberis-Jin-Wang construction).  Outcomes are sorted ascending; decision
    weights are cumulative-weighting differences applied from the loss tail
    upward on losses (``r < 0``) and from the gain tail downward on gains
    (``r >= 0``); the value function is ``x**alpha`` for gains and
    ``-lam * (-x)**alpha`` for losses.

    Args:
        returns: Daily returns (order-irrelevant; sorted internally).
        alpha: Value-function curvature (TK-1992 canonical ``0.88``).
        lam: Loss aversion (TK-1992 canonical ``2.25``).
        gamma_gain: Gain-side weighting curvature (TK-1992 canonical ``0.61``).
        gamma_loss: Loss-side weighting curvature (TK-1992 canonical ``0.69``).

    Returns:
        ``sum(pi_i * v(r_i))`` as a float, or ``None`` when fewer than two
        finite samples remain or the result is non-finite.  Never raises.
    """
    arr = _coerce_1d(returns)
    if arr is None:
        return None
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 2:
        return None
    alpha_f = _coerce_param(alpha)
    lam_f = _coerce_param(lam)
    g_gain = _coerce_param(gamma_gain)
    g_loss = _coerce_param(gamma_loss)
    if (
        alpha_f is None
        or lam_f is None
        or g_gain is None
        or g_loss is None
        or not 0.0 < alpha_f <= 1.0
        or lam_f < 0.0
        or not 0.0 < g_gain <= 1.0
        or not 0.0 < g_loss <= 1.0
    ):
        return None

    ordered = np.sort(arr, kind="stable")
    n_loss = int(np.count_nonzero(ordered < 0.0))
    probs = np.arange(n + 1, dtype=float) / float(n)

    pi = np.empty(n, dtype=float)
    if n_loss > 0:
        # Losses (indices 0..n_loss-1, most negative first): cumulate from the
        # loss tail: pi_i = w-((i+1)/n) - w-(i/n).
        w_loss = _tk_probability_weight(probs[: n_loss + 1], g_loss)
        pi[:n_loss] = np.diff(w_loss)
    if n_loss < n:
        # Gains (indices n_loss..n-1): cumulate from the gain tail:
        # pi_i = w+((n-i)/n) - w+((n-i-1)/n).
        top_counts = np.arange(n - n_loss, 0, -1, dtype=float) / float(n)
        w_hi = _tk_probability_weight(top_counts, g_gain)
        w_lo = _tk_probability_weight(top_counts - 1.0 / float(n), g_gain)
        pi[n_loss:] = w_hi - w_lo

    value = np.empty(n, dtype=float)
    value[:n_loss] = -lam_f * np.power(-ordered[:n_loss], alpha_f)
    value[n_loss:] = np.power(ordered[n_loss:], alpha_f)

    tk = float(np.dot(pi, value))
    return tk if math.isfinite(tk) else None


__all__ = ["prospect_theory_value", "salience_theory_value"]
