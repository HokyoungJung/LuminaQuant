"""Deterministic tests for the HAR-RV indicator (Corsi 2009 primitives).

Covers the pre-registered falsifiers from the ind-har-rv proposal: on a
synthetic log-RV series generated from a known HAR data-generating process the
fitted coefficients must recover truth within tolerance and the forecast must
beat a naive lag-1 forecast on MSE; on white-noise RV it must NOT beat the
honest naive (the expanding-mean forecast -- the anti-lookahead /
implementation-bias guard; a lag-1 naive is legitimately beaten by ANY
mean-reverting forecast on iid data, so the mean forecast is the binding
no-skill reference there).  Plus UTC-day RV aggregation goldens, never-raise
guards, determinism, and the annualization-convention check.

All randomness is a seeded inline LCG (never the ``random`` module).
"""

from __future__ import annotations

import math

from lumina_quant.indicators.annualization import annualize_per_bar_vol
from lumina_quant.indicators.har_rv import (
    DEFAULT_HAR_LAGS,
    daily_realized_variance,
    har_annualized_vol_forecast,
    har_design,
    har_fit,
    har_rv_forecast,
    log_rv_transform,
)

# --------------------------------------------------------------------------- #
# LCG (deterministic, no `random` module)
# --------------------------------------------------------------------------- #


def _lcg_stream(seed: int):
    state = seed & 0x7FFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / float(0x7FFFFFFF)


def _gauss(stream) -> float:
    # Irwin-Hall(12) - 6: mean 0, variance 1, deterministic.
    return sum(next(stream) for _ in range(12)) - 6.0


def _har_dgp(n: int, seed: int, *, noise: float = 0.05) -> tuple[list[float], tuple[float, ...]]:
    """Log-RV series from a known HAR process; returns (rv_series, true_coefs)."""
    mu = -9.0
    beta_d, beta_w, beta_m = 0.4, 0.3, 0.2
    const = mu * (1.0 - beta_d - beta_w - beta_m)
    stream = _lcg_stream(seed)
    log_rv = [mu + noise * _gauss(stream) for _ in range(22)]
    for _ in range(n - 22):
        mean_d = log_rv[-1]
        mean_w = sum(log_rv[-5:]) / 5.0
        mean_m = sum(log_rv[-22:]) / 22.0
        nxt = const + beta_d * mean_d + beta_w * mean_w + beta_m * mean_m
        log_rv.append(nxt + noise * _gauss(stream))
    return [math.exp(x) for x in log_rv], (const, beta_d, beta_w, beta_m)


# --------------------------------------------------------------------------- #
# daily_realized_variance
# --------------------------------------------------------------------------- #


def test_daily_rv_hand_golden_and_day_boundary() -> None:
    day = 86_400
    closes = [100.0, 101.0, 100.5, 100.8, 102.0, 101.5]
    epochs = [0.0, 300.0, 600.0, 900.0, float(day + 100), float(day + 400)]
    days, rv = daily_realized_variance(closes, epochs, min_intraday_returns=1)
    assert days == (0, 1)
    expected_day0 = (
        math.log(101.0 / 100.0) ** 2 + math.log(100.5 / 101.0) ** 2 + math.log(100.8 / 100.5) ** 2
    )
    # The 100.8 -> 102.0 return crosses the UTC boundary and must be EXCLUDED.
    expected_day1 = math.log(101.5 / 102.0) ** 2
    assert abs(rv[0] - expected_day0) < 1e-15
    assert abs(rv[1] - expected_day1) < 1e-15


def test_daily_rv_min_intraday_returns_floor() -> None:
    closes = [100.0, 101.0, 100.5, 100.8]
    epochs = [0.0, 300.0, 600.0, 900.0]  # only 3 intraday returns
    assert daily_realized_variance(closes, epochs) == ((), ())  # default floor 12
    days, rv = daily_realized_variance(closes, epochs, min_intraday_returns=3)
    assert days == (0,) and len(rv) == 1


def test_daily_rv_never_raises_on_garbage() -> None:
    assert daily_realized_variance(None, None) == ((), ())
    assert daily_realized_variance(42, 42) == ((), ())
    closes = [100.0, "x", None, -5.0, float("nan"), 101.0, 100.0, 102.0]
    epochs = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0, "y", 420.0]
    _days, rv = daily_realized_variance(closes, epochs, min_intraday_returns=1)
    assert all(value >= 0.0 and math.isfinite(value) for value in rv)


# --------------------------------------------------------------------------- #
# design / fit / forecast
# --------------------------------------------------------------------------- #


def test_har_design_block_means_golden() -> None:
    series = [float(v) for v in range(30)]
    built = har_design(series, lags=DEFAULT_HAR_LAGS)
    assert built is not None
    design, target = built
    assert design.shape == (8, 4)
    assert target[0] == 22.0
    row = design[0]
    assert row[0] == 1.0
    assert row[1] == 21.0  # lag-1 block
    assert row[2] == sum(range(17, 22)) / 5.0  # weekly block
    assert row[3] == sum(range(22)) / 22.0  # monthly block


def test_har_dgp_coefficient_recovery() -> None:
    rv, true_coefs = _har_dgp(900, seed=20260820)
    built = har_design(log_rv_transform(rv), lags=DEFAULT_HAR_LAGS)
    assert built is not None
    coefs = har_fit(*built)
    assert coefs is not None
    # Intercept tolerance is looser (scale ~ -0.9); slopes must be tight.
    assert abs(coefs[0] - true_coefs[0]) < 0.45
    for got, want in zip(coefs[1:], true_coefs[1:], strict=True):
        assert abs(got - want) < 0.12


def test_har_forecast_beats_naive_lag1_on_har_dgp() -> None:
    rv, _ = _har_dgp(700, seed=31337)
    log_rv = [math.log(value) for value in rv]
    har_sq_err = []
    naive_sq_err = []
    for t in range(550, 700):
        forecast = har_rv_forecast(rv[:t])
        assert forecast is not None and forecast > 0.0
        har_sq_err.append((math.log(forecast) - log_rv[t]) ** 2)
        naive_sq_err.append((log_rv[t - 1] - log_rv[t]) ** 2)
    assert sum(har_sq_err) < sum(naive_sq_err)


def test_har_forecast_does_not_beat_mean_on_white_noise() -> None:
    # Anti-implementation-bias guard: on iid log-RV there is NO forecastable
    # structure, so HAR must not meaningfully beat the expanding-mean naive
    # (any material "win" here would indicate lookahead or fitting bias).
    stream = _lcg_stream(777)
    log_rv = [-9.0 + 0.3 * _gauss(stream) for _ in range(700)]
    rv = [math.exp(x) for x in log_rv]
    har_sq_err = []
    mean_sq_err = []
    for t in range(550, 700):
        forecast = har_rv_forecast(rv[:t])
        assert forecast is not None
        expanding_mean = sum(log_rv[:t]) / float(t)
        har_sq_err.append((math.log(forecast) - log_rv[t]) ** 2)
        mean_sq_err.append((expanding_mean - log_rv[t]) ** 2)
    assert sum(har_sq_err) >= 0.98 * sum(mean_sq_err)


def test_har_forecast_never_raises_and_fails_closed() -> None:
    assert har_rv_forecast(None) is None
    assert har_rv_forecast("abc") is None
    assert har_rv_forecast(42) is None
    assert har_rv_forecast([]) is None
    assert har_rv_forecast([1e-4] * 10) is None  # short history
    assert har_rv_forecast([-1.0] * 100) is None  # negative variance
    assert har_rv_forecast([1e-4] * 30 + [float("nan")] + [1e-4] * 30) is None
    constant = har_rv_forecast([2.5e-4] * 120)  # rank-deficient but solvable
    assert constant is None or (math.isfinite(constant) and constant >= 0.0)
    assert har_rv_forecast([1e-4] * 120, lags=(0,)) is None  # degenerate lag


def test_har_forecast_two_run_determinism() -> None:
    rv, _ = _har_dgp(400, seed=555)
    first = har_rv_forecast(rv)
    second = har_rv_forecast(list(rv))
    assert first is not None
    assert first == second


def test_har_annualization_convention() -> None:
    rv, _ = _har_dgp(400, seed=99)
    forecast = har_rv_forecast(rv)
    assert forecast is not None
    times = [86_400.0 * step for step in range(64)]  # daily cadence
    annualized = har_annualized_vol_forecast(rv, times=times)
    assert annualized is not None
    # Must equal the canonical annualization-module path exactly (365.25d year).
    expected = annualize_per_bar_vol(math.sqrt(forecast), 365.25)
    assert annualized == expected
    assert har_annualized_vol_forecast(rv) is None  # no cadence -> None
    assert har_annualized_vol_forecast([1.0] * 5, times=times) is None
