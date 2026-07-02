"""Pure ``erf``-based Black-Scholes analytics (research / validation only).

This module is a net-new, self-contained pricing/greeks/implied-vol kernel used
for offline analytics and cross-checks.  It has NO execution path: nothing in the
crypto/perp live or backtest pipeline imports it, and it never mutates any metric
payload.  The standard-normal CDF / inverse-CDF are taken from
``strategy_factory.research_metrics`` (``_norm_cdf`` / ``_norm_ppf``) so there is
a single, scipy-free source of truth for the Gaussian primitives.

All functions are deterministic and closed-form (or a deterministic bisection for
implied volatility), so a frozen golden fixture pins the numerics exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lumina_quant.strategy_factory.research_metrics import _norm_cdf

# ``OptionType`` is a plain string discriminator to keep the module dependency-free.
CALL = "call"
PUT = "put"
_VALID_OPTION_TYPES = frozenset({CALL, PUT})

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    """Standard-normal probability density (no scipy dependency)."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _normalize_option_type(option_type: str) -> str:
    token = str(option_type or "").strip().lower()
    if token in {"c", "call"}:
        return CALL
    if token in {"p", "put"}:
        return PUT
    raise ValueError(f"option_type must be 'call' or 'put'; received {option_type!r}")


@dataclass(frozen=True, slots=True)
class BlackScholesGreeks:
    """Closed-form Black-Scholes price and first-order greeks (per unit)."""

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def d1_d2(
    *,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    vol: float,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Return the Black-Scholes ``d1`` / ``d2`` terms.

    ``tau`` is time-to-expiry in years and ``vol`` the annualized volatility.
    """
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive")
    if tau <= 0.0:
        raise ValueError("tau (time-to-expiry) must be positive")
    if vol <= 0.0:
        raise ValueError("vol must be positive")
    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(spot / strike) + (rate - dividend_yield + 0.5 * vol * vol) * tau) / (
        vol * sqrt_tau
    )
    d2 = d1 - vol * sqrt_tau
    return d1, d2


def black_scholes_price(
    *,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    vol: float,
    option_type: str = CALL,
    dividend_yield: float = 0.0,
) -> float:
    """Black-Scholes(-Merton) price of a European option (continuous dividend)."""
    kind = _normalize_option_type(option_type)
    d1, d2 = d1_d2(
        spot=spot,
        strike=strike,
        tau=tau,
        rate=rate,
        vol=vol,
        dividend_yield=dividend_yield,
    )
    disc_r = math.exp(-rate * tau)
    disc_q = math.exp(-dividend_yield * tau)
    if kind == CALL:
        return spot * disc_q * _norm_cdf(d1) - strike * disc_r * _norm_cdf(d2)
    return strike * disc_r * _norm_cdf(-d2) - spot * disc_q * _norm_cdf(-d1)


def black_scholes_greeks(
    *,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    vol: float,
    option_type: str = CALL,
    dividend_yield: float = 0.0,
) -> BlackScholesGreeks:
    """Closed-form price + first-order greeks.

    ``vega`` and ``rho`` are per 1.00 (100%) change in vol / rate; ``theta`` is
    per year (divide by 365 for per-calendar-day).
    """
    kind = _normalize_option_type(option_type)
    d1, d2 = d1_d2(
        spot=spot,
        strike=strike,
        tau=tau,
        rate=rate,
        vol=vol,
        dividend_yield=dividend_yield,
    )
    sqrt_tau = math.sqrt(tau)
    disc_r = math.exp(-rate * tau)
    disc_q = math.exp(-dividend_yield * tau)
    pdf_d1 = _norm_pdf(d1)

    price = black_scholes_price(
        spot=spot,
        strike=strike,
        tau=tau,
        rate=rate,
        vol=vol,
        option_type=kind,
        dividend_yield=dividend_yield,
    )
    gamma = disc_q * pdf_d1 / (spot * vol * sqrt_tau)
    vega = spot * disc_q * pdf_d1 * sqrt_tau
    if kind == CALL:
        delta = disc_q * _norm_cdf(d1)
        theta = (
            -(spot * disc_q * pdf_d1 * vol) / (2.0 * sqrt_tau)
            - rate * strike * disc_r * _norm_cdf(d2)
            + dividend_yield * spot * disc_q * _norm_cdf(d1)
        )
        rho = strike * tau * disc_r * _norm_cdf(d2)
    else:
        delta = -disc_q * _norm_cdf(-d1)
        theta = (
            -(spot * disc_q * pdf_d1 * vol) / (2.0 * sqrt_tau)
            + rate * strike * disc_r * _norm_cdf(-d2)
            - dividend_yield * spot * disc_q * _norm_cdf(-d1)
        )
        rho = -strike * tau * disc_r * _norm_cdf(-d2)
    return BlackScholesGreeks(
        price=float(price),
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
    )


def implied_volatility(
    *,
    price: float,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    option_type: str = CALL,
    dividend_yield: float = 0.0,
    vol_low: float = 1e-6,
    vol_high: float = 10.0,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """Recover Black-Scholes implied volatility via deterministic bisection.

    Returns the annualized volatility that reprices the option to ``price``.
    Raises ``ValueError`` if the target price is outside the arbitrage bounds
    bracketed by ``[vol_low, vol_high]`` (fail-loud — no silent clamp).
    """
    kind = _normalize_option_type(option_type)
    if price <= 0.0:
        raise ValueError("target option price must be positive")

    def _diff(vol: float) -> float:
        return (
            black_scholes_price(
                spot=spot,
                strike=strike,
                tau=tau,
                rate=rate,
                vol=vol,
                option_type=kind,
                dividend_yield=dividend_yield,
            )
            - price
        )

    lo = float(vol_low)
    hi = float(vol_high)
    f_lo = _diff(lo)
    f_hi = _diff(hi)
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if f_lo * f_hi > 0.0:
        raise ValueError("target price not bracketed by [vol_low, vol_high]; widen the bracket")

    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        f_mid = _diff(mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return float(mid)
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return float(0.5 * (lo + hi))


__all__ = [
    "CALL",
    "PUT",
    "BlackScholesGreeks",
    "black_scholes_greeks",
    "black_scholes_price",
    "d1_d2",
    "implied_volatility",
]
