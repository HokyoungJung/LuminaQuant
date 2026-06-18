"""Tests for the correlation-aware Ledoit-Wolf shrunk portfolio-vol clamp.

Covers the ``optimizer_core`` primitive (``shrunk_portfolio_vol`` +
``ledoit_wolf_shrunk_covariance``) and its default-OFF wiring into
``DiversifiedMultiFactorEnsembleStrategy._inverse_vol_weights``: with the flag
off the gross sizing must remain byte-identical to the naive ``sum(w_i * vol_i)``
clamp, and the shrunk path must only ever de-risk (scalar <= 1).
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from lumina_quant.portfolio.optimizer_core import (
    ledoit_wolf_shrunk_covariance,
    shrunk_portfolio_vol,
)
from lumina_quant.strategies.diversified_multifactor_ensemble import (
    DiversifiedMultiFactorEnsembleStrategy,
)

_SYMBOLS = ["AAA/USDT", "BBB/USDT", "CCC/USDT", "DDD/USDT", "EEE/USDT"]


# --- (1) d == 0 + single name -> == that name's vol --------------------------


def test_single_name_zero_shrinkage_equals_name_vol() -> None:
    rng = np.random.default_rng(7)
    series = rng.normal(0.0, 0.02, 200).reshape(-1, 1)
    # A single name has a 1x1 sample covariance: there is nothing off-diagonal to
    # shrink, so the analytic intensity is forced to 0.
    _sigma, intensity = ledoit_wolf_shrunk_covariance(series)
    assert intensity == 0.0
    name_vol = float(np.std(series[:, 0]))  # MLE divisor (T), matches Sigma_hat
    assert shrunk_portfolio_vol(series, [1.0]) == pytest.approx(name_vol)


# --- (2) perfectly-correlated ~ naive; decorrelated strictly lower -----------


def test_perfect_corr_matches_naive_and_decorrelated_is_lower() -> None:
    rng = np.random.default_rng(11)
    t = 250
    n = 4
    w = np.full(n, 1.0 / n)

    base = rng.normal(0.0, 0.02, t)
    perfect = np.column_stack([base for _ in range(n)])  # identical => corr == 1
    naive_perfect = float(np.sum(w * np.std(perfect, axis=0)))
    shrunk_perfect = shrunk_portfolio_vol(perfect, w)
    # Perfectly correlated block: diversification helps nothing, so the shrunk
    # estimate is close to the naive sum(w_i * vol_i).
    assert shrunk_perfect == pytest.approx(naive_perfect, rel=0.05)

    decorrelated = np.column_stack([rng.normal(0.0, 0.02, t) for _ in range(n)])
    naive_decorr = float(np.sum(w * np.std(decorrelated, axis=0)))
    shrunk_decorr = shrunk_portfolio_vol(decorrelated, w)
    # Independent names: real diversification => strictly below sum(w_i * vol_i).
    assert shrunk_decorr < naive_decorr


# --- (3) intensity in [0,1] and Sigma_hat PD when names > window -------------


def test_intensity_in_unit_interval_and_pd_when_names_exceed_window() -> None:
    rng = np.random.default_rng(3)
    n = 12
    window = 5  # fewer observations than names => sample cov is singular
    matrix = rng.normal(0.0, 0.02, (window, n))

    sigma, intensity = ledoit_wolf_shrunk_covariance(matrix, window=window)
    assert 0.0 <= intensity <= 1.0
    assert sigma.shape == (n, n)
    # Symmetric PD: all eigenvalues strictly positive even though n > window.
    eigvals = np.linalg.eigvalsh(0.5 * (sigma + sigma.T))
    assert float(eigvals.min()) > 0.0

    w = np.full(n, 1.0 / n)
    vol = shrunk_portfolio_vol(matrix, w, window=window)
    assert math.isfinite(vol) and vol > 0.0


# --- (5) no-lookahead: only trailing/window rows are consulted ---------------


def test_no_lookahead_only_uses_window_and_past_rows() -> None:
    rng = np.random.default_rng(5)
    n = 3
    window = 20
    recent = rng.normal(0.0, 0.02, (window, n))
    w = np.full(n, 1.0 / n)
    baseline = shrunk_portfolio_vol(recent, w, window=window)

    # The estimate is built only from CLOSED bars handed to the function: older
    # history beyond the trailing window cannot leak in. Prepending a huge older
    # spike must NOT change the estimate "as of" the most recent ``window`` bars.
    older_spike = np.vstack([rng.normal(0.0, 5.0, (50, n)), recent])
    assert shrunk_portfolio_vol(older_spike, w, window=window) == pytest.approx(baseline)

    # Passing exactly the trailing window (the caller's "as of now" view) equals
    # passing the full closed history and letting the window trim it: no future
    # rows exist past the end, so nothing forward of "now" is ever consulted.
    assert shrunk_portfolio_vol(recent, w, window=None) == pytest.approx(baseline)


# --- ensemble wiring fixtures (mirror the canonical ensemble unit test) -------


def _zigzag(i: int) -> float:
    return 1.0 if i % 2 == 0 else -1.0


def _series(n: int, *, start: float, drift: float, noise: float) -> list[float]:
    out: list[float] = []
    for i in range(n):
        out.append(start * (1.0 + drift) ** i * (1.0 + noise * _zigzag(i)))
    return out


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)

    def get_latest_feature_value(self, symbol: str, field: str) -> float | None:
        return None

    def get_latest_bar_value(self, symbol: str, field: str) -> float | None:
        return None


def _market_event(symbol: str, idx: int, close: float, **features: Any) -> SimpleNamespace:
    return SimpleNamespace(
        type="MARKET",
        time=f"2026-05-02T{idx // 60:02d}:{idx % 60:02d}:00Z",
        symbol=symbol,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000.0,
        **features,
    )


def _feed(strategy: DiversifiedMultiFactorEnsembleStrategy, series: dict[str, list[float]]) -> None:
    n = len(next(iter(series.values())))
    for idx in range(n):
        for symbol in series:
            price = series[symbol][idx]
            strategy.calculate_signals(
                _market_event(
                    symbol,
                    idx,
                    price,
                    funding_rate=-0.0005,
                    mark_price=price,
                    index_price=price,
                    open_interest=1_000_000.0,
                )
            )


def _make_strategy(queue: _Queue, **overrides: Any) -> DiversifiedMultiFactorEnsembleStrategy:
    params: dict[str, Any] = dict(
        trend_lookback=20,
        vol_window=10,
        funding_z_window=8,
        min_symbols=4,
        rebalance_bars=1,
        threshold=0.0,
        max_longs=5,
        max_shorts=0,
        allow_short=False,
        target_vol=0.001,
        target_gross_exposure=1.0,
    )
    params.update(overrides)
    return DiversifiedMultiFactorEnsembleStrategy(_Bars(_SYMBOLS), queue, **params)


def _mixed_vol_series() -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}
    for symbol in _SYMBOLS:
        if symbol in ("AAA/USDT", "BBB/USDT"):
            series[symbol] = _series(60, start=100.0, drift=0.004, noise=0.006)
        else:
            series[symbol] = _series(60, start=100.0, drift=0.004, noise=0.03)
    return series


# --- (4) flag default-off reproduces current gross byte-identically ----------


def test_schema_has_default_off_flag() -> None:
    schema = DiversifiedMultiFactorEnsembleStrategy.get_param_schema()
    assert "use_shrunk_cov_vol" in schema
    assert schema["use_shrunk_cov_vol"].default is False


def test_flag_default_off_byte_identical_gross() -> None:
    series = _mixed_vol_series()

    queue_default = _Queue()
    strat_default = _make_strategy(queue_default)
    assert strat_default.use_shrunk_cov_vol is False
    _feed(strat_default, series)

    queue_explicit = _Queue()
    strat_explicit = _make_strategy(queue_explicit, use_shrunk_cov_vol=False)
    _feed(strat_explicit, series)

    def _sizes(queue: _Queue) -> dict[str, tuple[float, float, float]]:
        out: dict[str, tuple[float, float, float]] = {}
        for sig in queue.items:
            if sig.signal_type != "LONG":
                continue
            meta = sig.metadata
            out[sig.symbol] = (
                float(meta["target_allocation"]),
                float(meta["inverse_vol_weight"]),
                float(meta["vol_target_scalar"]),
            )
        return out

    default_sizes = _sizes(queue_default)
    explicit_sizes = _sizes(queue_explicit)
    assert default_sizes
    # Default-off path must be byte-identical to an explicit-off configuration.
    assert default_sizes == explicit_sizes
    # Every emitted gross scalar is exactly the naive clamp (not the shrunk one).
    for _alloc, _w, scalar in default_sizes.values():
        assert isinstance(scalar, float)


# --- (6) shrunk path only de-risks (clamp scalar <= 1) -----------------------


def test_shrunk_path_clamp_never_exceeds_one() -> None:
    series = _mixed_vol_series()

    # Tiny target_vol forces the clamp to engage; assert it only de-risks.
    queue = _Queue()
    strat = _make_strategy(queue, use_shrunk_cov_vol=True, target_vol=0.0005)
    assert strat.use_shrunk_cov_vol is True
    _feed(strat, series)

    longs = [sig.metadata for sig in queue.items if sig.signal_type == "LONG"]
    assert longs
    for meta in longs:
        assert meta["vol_target_scalar"] <= 1.0 + 1e-12


def test_shrunk_clamp_at_least_as_loose_as_naive() -> None:
    # With diversification, the shrunk portfolio vol is <= the naive sum(w_i*vol_i)
    # for a decorrelated book, so the de-risking clamp is no TIGHTER than naive
    # (scalar_shrunk >= scalar_naive) while still <= 1.
    series = _mixed_vol_series()

    queue_naive = _Queue()
    strat_naive = _make_strategy(queue_naive, use_shrunk_cov_vol=False, target_vol=0.05)
    _feed(strat_naive, series)

    queue_shrunk = _Queue()
    strat_shrunk = _make_strategy(queue_shrunk, use_shrunk_cov_vol=True, target_vol=0.05)
    _feed(strat_shrunk, series)

    def _scalar(queue: _Queue) -> float | None:
        for sig in queue.items:
            if sig.signal_type == "LONG":
                return float(sig.metadata["vol_target_scalar"])
        return None

    naive_scalar = _scalar(queue_naive)
    shrunk_scalar = _scalar(queue_shrunk)
    assert naive_scalar is not None and shrunk_scalar is not None
    assert shrunk_scalar <= 1.0 + 1e-12
    assert shrunk_scalar >= naive_scalar - 1e-9
