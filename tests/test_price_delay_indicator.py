"""Closed-form unit tests for the Hou-Moskowitz ``price_delay_share`` numeric."""

from __future__ import annotations

import math

from lumina_quant.indicators.price_delay import price_delay_share


def _lcg(seed: int):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / float(0x7FFFFFFF)


def _benchmark(n: int) -> list[float]:
    gen = _lcg(7)
    return [0.03 * (1 if next(gen) > 0.5 else -1) + (next(gen) - 0.5) * 2e-4 for _ in range(n)]


def test_pure_lag_loader_is_high_delay() -> None:
    bench = _benchmark(200)
    lag1 = [0.0, *bench[:-1]]
    asset = [0.8 * value for value in lag1]
    delay = price_delay_share(asset, bench, lags=5, min_obs=30)
    assert delay is not None and delay > 0.9, delay


def test_pure_contemporaneous_loader_is_zero_delay() -> None:
    bench = _benchmark(200)
    asset = [0.8 * value for value in bench]
    delay = price_delay_share(asset, bench, lags=5, min_obs=30)
    assert delay is not None and delay < 0.05, delay


def test_pure_idiosyncratic_is_undefined() -> None:
    bench = _benchmark(200)
    gen = _lcg(999)
    asset = [(next(gen) - 0.5) * 0.05 for _ in range(200)]
    assert price_delay_share(asset, bench, lags=5, min_obs=30) is None


def test_negation_invariance() -> None:
    bench = _benchmark(200)
    lag1 = [0.0, *bench[:-1]]
    asset = [0.4 * bench[i] + 0.9 * lag1[i] for i in range(200)]
    base = price_delay_share(asset, bench, lags=5, min_obs=30)
    negated = price_delay_share([-v for v in asset], [-v for v in bench], lags=5, min_obs=30)
    assert base is not None and negated is not None
    assert abs(base - negated) < 1e-12, (base, negated)


def test_lag_weighted_mode_bounds() -> None:
    bench = _benchmark(200)
    lag1 = [0.0, *bench[:-1]]
    pure_lag = [0.8 * value for value in lag1]
    pure_contemp = [0.8 * value for value in bench]
    d_lag = price_delay_share(pure_lag, bench, lags=5, min_obs=30, score_mode="lag_weighted")
    d_contemp = price_delay_share(
        pure_contemp, bench, lags=5, min_obs=30, score_mode="lag_weighted"
    )
    assert d_lag is not None and d_contemp is not None
    assert 0.0 <= d_contemp <= 1.0 and 0.0 <= d_lag <= 1.0
    assert d_lag > d_contemp


def test_short_history_returns_none() -> None:
    bench = _benchmark(20)
    asset = [0.8 * value for value in bench]
    assert price_delay_share(asset, bench, lags=5, min_obs=30) is None


def test_degenerate_inputs_never_raise() -> None:
    assert price_delay_share([], [], lags=5) is None
    assert price_delay_share([1.0, 2.0], [1.0, 2.0], lags=5) is None
    assert price_delay_share([float("nan")] * 60, [1.0] * 60, lags=5) is None
    # A constant asset (zero variance) has no defined regression target.
    bench = _benchmark(80)
    assert price_delay_share([0.0] * 80, bench, lags=5, min_obs=30) is None
    # Garbage / non-numeric entries are dropped, never raised on.
    assert price_delay_share(["x", None, object()], bench, lags=5) is None


def test_range_bounded() -> None:
    bench = _benchmark(200)
    lag1 = [0.0, *bench[:-1]]
    for coef_c, coef_l in ((0.9, 0.1), (0.5, 0.5), (0.1, 0.9)):
        asset = [coef_c * bench[i] + coef_l * lag1[i] for i in range(200)]
        delay = price_delay_share(asset, bench, lags=5, min_obs=30)
        assert delay is not None
        assert 0.0 <= delay <= 1.0
        assert math.isfinite(delay)
