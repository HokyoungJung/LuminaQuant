"""Config-gated STRATEGY-IMPROVE fixes: flag-OFF byte-identity + flag-ON behavior.

Every improvement is a strategy-local, default-OFF flag. With the flag OFF the
code path must be byte-identical to the legacy behavior (proved here by feeding
the same inputs and asserting equality / unchanged defaults); with the flag ON
the corrected behavior engages.
"""

from __future__ import annotations

import math
import queue
import random
from types import SimpleNamespace

import pytest

from lumina_quant.core.events import MarketEvent
from lumina_quant.indicators.advanced_alpha import perp_crowding_score
from lumina_quant.strategies.alpha101_formula import Alpha101FormulaStrategy
from lumina_quant.strategies.crypto_fx_alpha_zoo_state import CryptoFxAlphaZooStateStrategy
from lumina_quant.strategies.cross_sectional_funding_momentum_carry import (
    CrossSectionalFundingMomentumCarryStrategy,
)
from lumina_quant.strategies.lag_convergence import LagConvergenceStrategy


class _Bars:
    def __init__(self, symbols):
        self.symbol_list = list(symbols)


# ===========================================================================
# (1) lag_convergence: cointegration gate + beta/vol-neutral leg sizing
# ===========================================================================


def _lag_strategy(**kwargs):
    return LagConvergenceStrategy(
        _Bars(["X/USDT", "Y/USDT"]),
        queue.Queue(),
        symbol_x="X/USDT",
        symbol_y="Y/USDT",
        **kwargs,
    )


def test_lag_defaults_are_off():
    strat = _lag_strategy()
    assert strat.require_cointegration is False
    assert strat.beta_neutral_sizing is False


def test_lag_leg_strengths_off_is_one_to_one():
    strat = _lag_strategy()
    for value in (100.0, 101.0, 99.5, 102.0, 98.0):
        strat._x_history.append(value)
        strat._y_history.append(value * 3.0)
    assert strat._leg_strengths() == (1.0, 1.0)


def test_lag_leg_strengths_on_is_vol_neutral():
    strat = _lag_strategy(beta_neutral_sizing=True, beta_window=20)
    rng = random.Random(7)
    x = 100.0
    y = 100.0
    for _ in range(40):
        x *= 1.0 + rng.gauss(0.0, 0.0005)  # low vol leg
        y *= 1.0 + rng.gauss(0.0, 0.02)  # high vol leg
        strat._x_history.append(x)
        strat._y_history.append(y)
    sx, sy = strat._leg_strengths()
    # Lower-vol X keeps full size; higher-vol Y is scaled down below 1.
    assert sx == pytest.approx(1.0)
    assert 0.0 < sy < 1.0


def test_lag_cointegration_gate_true_on_cointegrated_false_on_random_walks():
    rng = random.Random(11)
    coint = _lag_strategy(require_cointegration=True, coint_window=48, coint_max_tstat=-2.0)
    common = 0.0
    for _ in range(60):
        common += rng.gauss(0.0, 0.01)
        coint._x_history.append(math.exp(common + rng.gauss(0.0, 0.001)))
        coint._y_history.append(math.exp(common + rng.gauss(0.0, 0.001)))
    assert coint._spread_is_stationary() is True

    # Non-cointegrated: independent trends with differing curvature => the
    # beta-adjusted spread keeps a (quadratic) unit-root-like trend, so ADF
    # cannot reject non-stationarity (deterministic, seed-free).
    walk = _lag_strategy(require_cointegration=True, coint_window=48, coint_max_tstat=-2.0)
    for t in range(60):
        walk._x_history.append(math.exp(0.03 * t))
        walk._y_history.append(math.exp(0.005 * t + 0.0004 * t * t))
    assert walk._spread_is_stationary() is False


def _feed_lag(strat, xs, ys):
    out = []
    events = strat.events
    # Provide a bars stub that returns aligned closes/timestamps.
    rows = {"X/USDT": None, "Y/USDT": None}

    class _LiveBars:
        symbol_list = ["X/USDT", "Y/USDT"]

        def get_latest_bar_datetime(self, symbol):
            return rows[symbol][0]

        def get_latest_bar_value(self, symbol, field):
            return rows[symbol][1]

    strat.bars = _LiveBars()
    for ts, (px, py) in enumerate(zip(xs, ys, strict=True)):
        rows["X/USDT"] = (ts, px)
        rows["Y/USDT"] = (ts, py)
        strat.calculate_signals(MarketEvent(ts, "Y/USDT", py, py, py, py, 1000.0))
        while not events.empty():
            sig = events.get()
            out.append((int(sig.datetime), sig.symbol, sig.signal_type, round(sig.strength, 9)))
    return out


def test_lag_entries_off_all_unit_strength_gate_suppresses():
    # A divergent pair whose relative momentum crosses the entry threshold.
    n = 40
    xs = [100.0 * (1.0 + 0.01) ** i for i in range(n)]
    ys = [100.0 * (1.0 - 0.008) ** i for i in range(n)]

    off = _lag_strategy(entry_threshold=0.005, exit_threshold=0.001, lag_bars=2)
    off_sig = _feed_lag(off, xs, ys)
    entries_off = [s for s in off_sig if s[2] in ("LONG", "SHORT")]
    assert entries_off, "baseline should produce entries on a divergent pair"
    # Flag OFF => every leg is unit strength (fixed 1:1 book).
    assert all(s[3] == 1.0 for s in entries_off)

    gated = _lag_strategy(
        entry_threshold=0.005,
        exit_threshold=0.001,
        lag_bars=2,
        require_cointegration=True,
        coint_window=20,
        coint_max_tstat=-2.5,
    )
    gated_sig = _feed_lag(gated, xs, ys)
    entries_gated = [s for s in gated_sig if s[2] in ("LONG", "SHORT")]
    # The cointegration gate refuses entries on the non-mean-reverting spread.
    assert len(entries_gated) < len(entries_off)


# ===========================================================================
# (2) crypto_fx_alpha_zoo_state: rolling-beta residualization
# ===========================================================================


def _crypto_strategy(**kwargs):
    return CryptoFxAlphaZooStateStrategy(
        _Bars(["BTC/USDT", "ETH/USDT"]),
        queue.Queue(),
        fast_lookback_bars=1,
        slow_lookback_bars=4,
        history_window=16,
        use_fx_filter=False,
        require_calibrated_edge=False,
        **kwargs,
    )


def _feed_crypto_2x_beta(strat, n=30):
    rets = [0.02, -0.015, 0.03, -0.01, 0.025, -0.02]
    btc = 100.0
    eth = 100.0
    for i in range(n):
        r = rets[i % len(rets)]
        btc *= 1.0 + r
        eth *= 1.0 + 2.0 * r  # ETH per-bar return is exactly 2x BTC => beta = 2
        strat.calculate_signals(MarketEvent(i, "BTC/USDT", btc, btc, btc, btc, 1000.0))
        strat.calculate_signals(MarketEvent(i, "ETH/USDT", eth, eth, eth, eth, 1000.0))


def test_crypto_default_off():
    assert _crypto_strategy().use_rolling_beta is False


def test_crypto_residual_beta_recovers_two():
    strat = _crypto_strategy()
    _feed_crypto_2x_beta(strat)
    beta = strat._residual_beta(strat._state["ETH/USDT"], strat._state["BTC/USDT"])
    assert beta == pytest.approx(2.0, abs=1e-6)


def test_crypto_rolling_beta_changes_residual_score():
    strat = _crypto_strategy()
    _feed_crypto_2x_beta(strat)
    strat.use_rolling_beta = False
    score_off = strat._score_symbol("ETH/USDT")
    strat.use_rolling_beta = True
    score_on = strat._score_symbol("ETH/USDT")
    assert score_off is not None and score_on is not None
    # Stripping the 2x market beta yields a materially different residual signal.
    assert score_off != pytest.approx(score_on)


# ===========================================================================
# (3) alpha101_formula: true cross-sectional ranking
# ===========================================================================


def test_alpha101_default_off():
    strat = Alpha101FormulaStrategy(_Bars(["A/USDT", "B/USDT"]), queue.Queue())
    assert strat.cross_sectional_rank is False


def test_alpha101_cross_sectional_zscore_standardizes_over_peers():
    strat = Alpha101FormulaStrategy(
        _Bars(["A/USDT", "B/USDT"]),
        queue.Queue(),
        cross_sectional_rank=True,
        score_window=8,
    )
    item_a = strat._state["A/USDT"]
    item_b = strat._state["B/USDT"]
    # Give A a self-history so the single-asset fallback is well-defined.
    for value in (0.1, -0.2, 0.05, 0.3, -0.1, 0.2, -0.05, 0.15):
        item_a.scores.append(value)

    strat._xs_latest = {}
    # Only A has printed this bar => fewer than 2 peers => time-series fallback.
    z_fallback = strat._cross_sectional_zscore(item_a, "A/USDT", "t1", 1.0)
    assert z_fallback is not None

    # Now B prints the same bar => true cross-sectional z over {A=1.0, B=3.0}.
    z_b = strat._cross_sectional_zscore(item_b, "B/USDT", "t1", 3.0)
    assert z_b == pytest.approx(1.0 / math.sqrt(2.0))
    # Re-scoring A now sees both peers -> symmetric negative z.
    z_a = strat._cross_sectional_zscore(item_a, "A/USDT", "t1", 1.0)
    assert z_a == pytest.approx(-1.0 / math.sqrt(2.0))


# ===========================================================================
# (4) cross_sectional_funding_momentum_carry + perp_crowding_score
# ===========================================================================

_FUND_SYMBOLS = ["AAA/USDT", "BBB/USDT", "CCC/USDT", "DDD/USDT"]


def test_funding_carry_default_off():
    strat = CrossSectionalFundingMomentumCarryStrategy(_Bars(_FUND_SYMBOLS), queue.Queue())
    assert strat.true_carry_sign is False


def test_funding_true_carry_sign_keeps_only_collecting_side():
    strat = CrossSectionalFundingMomentumCarryStrategy(
        _Bars(_FUND_SYMBOLS), queue.Queue(), true_carry_sign=True
    )
    # Latest funding levels: AAA long-collect (neg), BBB short-collect (pos),
    # CCC long-PAYS (pos) -> dropped, DDD short-PAYS (neg) -> dropped.
    strat._state["AAA/USDT"].funding_rate.append(-0.001)
    strat._state["BBB/USDT"].funding_rate.append(0.001)
    strat._state["CCC/USDT"].funding_rate.append(0.001)
    strat._state["DDD/USDT"].funding_rate.append(-0.001)
    targets = {
        "AAA/USDT": ("LONG", 1.0, {}),
        "BBB/USDT": ("SHORT", -1.0, {}),
        "CCC/USDT": ("LONG", 0.9, {}),
        "DDD/USDT": ("SHORT", -0.9, {}),
    }
    kept = strat._filter_carry_sign(targets)
    assert set(kept) == {"AAA/USDT", "BBB/USDT"}


def _crowding_kwargs():
    return dict(
        funding_rate=[0.0002 + 0.00005 * math.sin(i / 7.0) for i in range(120)],
        open_interest=[1_000_000.0 + 900.0 * i for i in range(120)],
        window=96,
    )


def test_perp_crowding_flag_off_is_byte_identical():
    kwargs = _crowding_kwargs()
    default = perp_crowding_score(**kwargs)
    explicit_off = perp_crowding_score(**kwargs, true_carry_sign=False)
    assert default == explicit_off
    for key in ("carry_harvest_sign", "carry_harvest_side", "latest_funding_rate"):
        assert key not in default


def test_perp_crowding_true_carry_sign_positive_funding_is_short():
    kwargs = _crowding_kwargs()
    kwargs["funding_rate"][-1] = 0.01  # positive funding -> short collects
    res = perp_crowding_score(**kwargs, true_carry_sign=True)
    assert res["carry_harvest_sign"] == -1.0
    assert res["carry_harvest_side"] == "SHORT"
    assert res["latest_funding_rate"] == pytest.approx(0.01)


def test_perp_crowding_true_carry_sign_negative_funding_is_long():
    kwargs = _crowding_kwargs()
    kwargs["funding_rate"][-1] = -0.01  # negative funding -> long collects
    res = perp_crowding_score(**kwargs, true_carry_sign=True)
    assert res["carry_harvest_sign"] == 1.0
    assert res["carry_harvest_side"] == "LONG"
