"""Golden fixture + invariants for the erf-based Black-Scholes analytics."""

from __future__ import annotations

import math

import pytest

from lumina_quant.research import options_math as om

_RTOL = 1e-10

# Frozen golden reference (textbook European option, S=K=100, T=1, r=5%, sigma=20%).
_CALL_GOLDEN = {
    "price": 10.450583572185565,
    "delta": 0.6368306511756191,
    "gamma": 0.018762017345846895,
    "vega": 37.52403469169379,
    "theta": -6.414027546438197,
    "rho": 53.232481545376345,
}
_PUT_GOLDEN = {
    "price": 5.573526022256971,
    "delta": -0.3631693488243809,
    "gamma": 0.018762017345846895,
    "vega": 37.52403469169379,
    "theta": -1.657880423934626,
    "rho": -41.89046090469506,
}


def _call_greeks() -> om.BlackScholesGreeks:
    return om.black_scholes_greeks(
        spot=100.0, strike=100.0, tau=1.0, rate=0.05, vol=0.20, option_type="call"
    )


def _put_greeks() -> om.BlackScholesGreeks:
    return om.black_scholes_greeks(
        spot=100.0, strike=100.0, tau=1.0, rate=0.05, vol=0.20, option_type="put"
    )


def test_call_greeks_match_golden() -> None:
    greeks = _call_greeks()
    for key, expected in _CALL_GOLDEN.items():
        assert getattr(greeks, key) == pytest.approx(expected, rel=_RTOL)


def test_put_greeks_match_golden() -> None:
    greeks = _put_greeks()
    for key, expected in _PUT_GOLDEN.items():
        assert getattr(greeks, key) == pytest.approx(expected, rel=_RTOL)


def test_put_call_parity_holds() -> None:
    call = _call_greeks()
    put = _put_greeks()
    lhs = call.price - put.price
    rhs = 100.0 - 100.0 * math.exp(-0.05 * 1.0)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_price_matches_greeks_price() -> None:
    price = om.black_scholes_price(
        spot=100.0, strike=100.0, tau=1.0, rate=0.05, vol=0.20, option_type="call"
    )
    assert price == pytest.approx(_CALL_GOLDEN["price"], rel=_RTOL)


def test_implied_volatility_round_trips() -> None:
    price = om.black_scholes_price(
        spot=100.0, strike=110.0, tau=0.5, rate=0.03, vol=0.35, option_type="put"
    )
    iv = om.implied_volatility(
        price=price, spot=100.0, strike=110.0, tau=0.5, rate=0.03, option_type="put"
    )
    assert iv == pytest.approx(0.35, abs=1e-6)


def test_invalid_inputs_fail_loud() -> None:
    with pytest.raises(ValueError):
        om.black_scholes_price(
            spot=100.0, strike=100.0, tau=0.0, rate=0.05, vol=0.20, option_type="call"
        )
    with pytest.raises(ValueError):
        om.black_scholes_greeks(
            spot=-1.0, strike=100.0, tau=1.0, rate=0.05, vol=0.20, option_type="call"
        )
    with pytest.raises(ValueError):
        om.black_scholes_price(
            spot=100.0, strike=100.0, tau=1.0, rate=0.05, vol=0.20, option_type="banana"
        )


def test_implied_vol_unbracketed_price_raises() -> None:
    with pytest.raises(ValueError):
        om.implied_volatility(
            price=1e6,
            spot=100.0,
            strike=100.0,
            tau=1.0,
            rate=0.05,
            option_type="call",
            vol_high=1.0,
        )
