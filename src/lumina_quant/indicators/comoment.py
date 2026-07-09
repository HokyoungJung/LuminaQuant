"""Rolling co-moment primitives (conditional semibeta + standardized coskewness).

Shared, pure-Python co-moment machinery consumed by the two cross-sectional
co-moment sleeves:

- ``CrossSectionalDownsideBetaAsymmetryStrategy`` ranks symbols on the
  benchmark-sign-conditioned beta asymmetry ``beta_minus - beta_plus``
  (Ang-Chen-Xing relative downside beta).
- ``SystematicCoskewnessPremiumStrategy`` ranks symbols on the
  beta-residualized third co-moment with the market (Harvey-Siddique
  standardized coskewness).

Both characteristics are built on the SAME aligned-return / demean /
covariance kernel (:func:`_aligned_returns`), so the shared numerics live here
once rather than being re-derived per sleeve.  House style mirrors
``indicators/rolling_stats.py``: ``math``/``statistics`` only (no numpy needed),
``(n - 1)`` sample normalization, ``1e-12`` degenerate-variance guards, and
``None`` (never an exception) on empty / short / non-finite / degenerate input.
"""

from __future__ import annotations

import math
from statistics import mean

_VAR_EPS = 1e-12
_BETA_CLAMP = 10.0


def _aligned_returns(asset_returns, market_returns) -> tuple[list[float], list[float]] | None:
    """Trim both return series to their common trailing length.

    Mirrors the alignment guards of ``rolling_stats.rolling_beta``: both series
    are trimmed to their shared trailing length, at least two samples are
    required, and any non-finite value rejects the whole window.
    """
    count = min(len(asset_returns), len(market_returns))
    if count < 2:
        return None
    asset_tail = [float(value) for value in list(asset_returns)[-count:]]
    market_tail = [float(value) for value in list(market_returns)[-count:]]
    if not all(math.isfinite(value) for value in asset_tail):
        return None
    if not all(math.isfinite(value) for value in market_tail):
        return None
    return asset_tail, market_tail


def _side_beta(
    asset_side: list[float], market_side: list[float], min_side_obs: int
) -> float | None:
    """OLS beta ``Cov(asset, market) / Var(market)`` over one conditioned subsample."""
    count = len(market_side)
    if count < max(2, int(min_side_obs)):
        return None
    mean_market = mean(market_side)
    var_market = sum((value - mean_market) ** 2 for value in market_side) / float(count - 1)
    if var_market <= _VAR_EPS:
        return None
    mean_asset = mean(asset_side)
    cov = sum(
        (asset - mean_asset) * (market - mean_market)
        for asset, market in zip(asset_side, market_side, strict=False)
    ) / float(count - 1)
    beta = cov / var_market
    if not math.isfinite(beta):
        return None
    return max(-_BETA_CLAMP, min(_BETA_CLAMP, beta))


def conditional_semibeta(
    asset_returns,
    market_returns,
    *,
    threshold: float = 0.0,
    min_side_obs: int = 20,
) -> tuple[float | None, float | None]:
    """Return ``(beta_minus, beta_plus)`` conditioned on the market-return sign.

    Aligned ``(asset, market)`` pairs are partitioned by ``market < threshold``
    (the down side) versus ``market >= threshold`` (the up side); each side
    yields the OLS beta of the asset on the market over that subsample with
    ``(n - 1)`` normalization.  A side with fewer than ``min_side_obs``
    observations, or with degenerate market variance, returns ``None`` for that
    side (the caller abstains rather than trading a noise estimate).  Pure,
    deterministic, and never raises.

    Args:
        asset_returns: The asset return series (log or simple; caller's choice).
        market_returns: The benchmark return series, aligned bar-for-bar.
        threshold: The market-return partition point (default ``0.0``).
        min_side_obs: Minimum observations required on a side to return its beta.

    Returns:
        A ``(beta_minus, beta_plus)`` tuple; either entry may be ``None``.
    """
    aligned = _aligned_returns(asset_returns, market_returns)
    if aligned is None:
        return (None, None)
    asset_tail, market_tail = aligned
    thr = float(threshold)
    if not math.isfinite(thr):
        return (None, None)
    down_asset: list[float] = []
    down_market: list[float] = []
    up_asset: list[float] = []
    up_market: list[float] = []
    for asset, market in zip(asset_tail, market_tail, strict=False):
        if market < thr:
            down_asset.append(asset)
            down_market.append(market)
        else:
            up_asset.append(asset)
            up_market.append(market)
    beta_minus = _side_beta(down_asset, down_market, min_side_obs)
    beta_plus = _side_beta(up_asset, up_market, min_side_obs)
    return (beta_minus, beta_plus)


def standardized_coskewness(
    asset_returns,
    market_returns,
    *,
    beta_residualize: bool = True,
) -> float | None:
    """Return the Harvey-Siddique standardized coskewness of asset with market.

    Both series are demeaned over the aligned window.  When
    ``beta_residualize`` is set (the default and the published construction),
    the linear co-moment already owned by the beta / BAB axis is stripped first
    via ``eps_i = (r_i - mean) - beta * eps_m`` where ``beta = Cov / Var`` and
    ``eps_m`` is the demeaned market return.  The standardized third co-moment
    is then::

        coskew = mean(eps_i * eps_m ** 2) / (sample_std(eps_i) * var(eps_m))

    Negative coskewness marks assets that pay off badly exactly when market
    variance spikes (the insurance a short-coskew premium is paid to write).
    ``None`` is returned on short history, non-finite input, or degenerate
    market / residual variance.  Pure, deterministic, and never raises.

    Args:
        asset_returns: The asset return series.
        market_returns: The benchmark return series, aligned bar-for-bar.
        beta_residualize: Strip the linear beta component before the co-moment.

    Returns:
        The standardized coskewness, or ``None`` when undefined.
    """
    aligned = _aligned_returns(asset_returns, market_returns)
    if aligned is None:
        return None
    asset_tail, market_tail = aligned
    count = len(asset_tail)
    if count < 3:
        return None
    mean_asset = mean(asset_tail)
    mean_market = mean(market_tail)
    eps_market = [value - mean_market for value in market_tail]
    var_market = sum(value * value for value in eps_market) / float(count - 1)
    if var_market <= _VAR_EPS:
        return None
    asset_demeaned = [value - mean_asset for value in asset_tail]
    if beta_residualize:
        cov = sum(
            asset * market for asset, market in zip(asset_demeaned, eps_market, strict=False)
        ) / float(count - 1)
        beta = cov / var_market
        if not math.isfinite(beta):
            return None
        eps_asset = [
            asset - beta * market for asset, market in zip(asset_demeaned, eps_market, strict=False)
        ]
    else:
        eps_asset = asset_demeaned
    mean_eps_asset = sum(eps_asset) / float(count)
    var_eps_asset = sum((value - mean_eps_asset) ** 2 for value in eps_asset) / float(count - 1)
    if var_eps_asset <= _VAR_EPS:
        return None
    std_eps_asset = math.sqrt(var_eps_asset)
    numerator = sum(
        asset * market * market for asset, market in zip(eps_asset, eps_market, strict=False)
    ) / float(count)
    coskew = numerator / (std_eps_asset * var_market)
    if not math.isfinite(coskew):
        return None
    return float(coskew)


__all__ = ["conditional_semibeta", "standardized_coskewness"]
