"""Unit tests for the additive Qlib Alpha158-style factor library.

Covers hand-computed spot checks for representative factors, run-twice
determinism, trailing-window invariance (adding older history does not change
the latest value), net-new de-duplication against the existing Alpha101 /
crypto_fx surfaces, the documented IC/ICIR eligibility bar, and the default-OFF
config gate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lumina_quant.configuration.schema import RuntimeConfig, StrategiesConfig
from lumina_quant.indicators import qlib158


def _spot_context() -> dict[str, list[float]]:
    return {
        "open": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        "high": [12.0, 12.0, 12.0, 12.0, 12.0, 20.0],
        "low": [8.0, 8.0, 8.0, 8.0, 8.0, 5.0],
        "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        "volume": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
        "vwap": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    }


@pytest.mark.parametrize(
    "name, expected",
    [
        # net-new range-normalised KBAR + price ratios (latest bar only)
        ("KMID2", 5.0 / 15.0),
        ("KUP2", 5.0 / 15.0),
        ("KLOW2", 5.0 / 15.0),
        ("KSFT", 0.5),
        ("KSFT2", 5.0 / 15.0),
        ("OPEN0", 10.0 / 15.0),
        ("HIGH0", 20.0 / 15.0),
        ("LOW0", 5.0 / 15.0),
        ("VWAP0", 10.0 / 15.0),
        # rolling grid (window 5)
        ("ROC5", 10.0 / 15.0),
        ("MA5", 13.0 / 15.0),
        ("STD5", math.sqrt(2.5) / 15.0),
        ("BETA5", 1.0 / 15.0),
        ("RSQR5", 1.0),
        ("RESI5", 0.0),
        ("MAX5", 20.0 / 15.0),
        ("MIN5", 5.0 / 15.0),
        ("QTLU5", 14.2 / 15.0),
        ("QTLD5", 11.8 / 15.0),
        ("RANK5", 1.0),
        ("RSV5", 10.0 / 15.0),
        ("IMAX5", 0.8),
        ("IMIN5", 0.8),
        ("IMXD5", 0.0),
        ("CNTP5", 1.0),
        ("CNTN5", 0.0),
        ("CNTD5", 1.0),
        ("SUMP5", 1.0),
        ("SUMN5", 0.0),
        ("SUMD5", 1.0),
        ("VMA5", 130.0 / 150.0),
    ],
)
def test_factor_matches_hand_computed(name, expected):
    value = qlib158.evaluate_qlib158_factor(name, _spot_context())
    assert value is not None
    assert value == pytest.approx(expected)


def test_bounded_factors_stay_in_range():
    ctx = _spot_context()
    corr = qlib158.evaluate_qlib158_factor("CORR5", ctx)
    assert corr is not None
    assert -1.0 <= corr <= 1.0
    for name in ("WVMA5", "VSTD5", "VSUMP5", "VSUMN5"):
        value = qlib158.evaluate_qlib158_factor(name, ctx)
        assert value is not None
        assert math.isfinite(value)


def _random_context(n: int, *, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, size=n))
    high = close + np.abs(rng.normal(0.0, 0.5, size=n)) + 0.1
    low = close - np.abs(rng.normal(0.0, 0.5, size=n)) - 0.1
    open_ = close + rng.normal(0.0, 0.3, size=n)
    volume = 1000.0 + np.abs(rng.normal(0.0, 100.0, size=n))
    vwap = (high + low + close) / 3.0
    return {
        "open": open_.tolist(),
        "high": high.tolist(),
        "low": low.tolist(),
        "close": close.tolist(),
        "volume": volume.tolist(),
        "vwap": vwap.tolist(),
    }


def test_run_twice_is_deterministic():
    ctx = _random_context(90, seed=7)
    first = qlib158.compute_qlib158_row(ctx)
    second = qlib158.compute_qlib158_row(ctx)
    assert first.keys() == second.keys()
    for key in first:
        a, b = first[key], second[key]
        if a is None or b is None:
            assert a is b
        else:
            assert a == b


def test_trailing_window_invariance_under_older_history():
    # Latest factor values must depend only on the trailing window; prepending
    # older bars beyond the longest window (60, +1 for diffs) leaves them equal.
    base = _random_context(90, seed=11)
    older = _random_context(20, seed=99)
    extended = {field: older[field] + base[field] for field in base}

    base_row = qlib158.compute_qlib158_row(base)
    extended_row = qlib158.compute_qlib158_row(extended)
    assert base_row.keys() == extended_row.keys()
    for key in base_row:
        a, b = base_row[key], extended_row[key]
        if a is None or b is None:
            assert a is b, key
        else:
            assert a == pytest.approx(b, rel=1e-12, abs=1e-12), key


def test_library_is_net_new_only():
    names = set(qlib158.qlib158_factor_names())
    # Excluded: crypto_fx KBAR body/shape ratios already shipped elsewhere.
    assert {"kmid", "klen", "kup", "klow"}.isdisjoint(names)
    # Excluded: Alpha101 numeric ids.
    assert not any(name.lower().startswith("alpha_") for name in names)
    # Full Alpha158 rolling grid (29 families x 5 windows) + 9 static net-new.
    assert len(names) == 29 * len(qlib158.DEFAULT_QLIB158_WINDOWS) + 9


def test_short_context_and_missing_field_return_none():
    short = {"close": [1.0, 2.0]}
    assert qlib158.evaluate_qlib158_factor("MA5", short) is None
    # KMID2 needs open/high/low/close; missing fields -> None, not a crash.
    assert qlib158.evaluate_qlib158_factor("KMID2", {"close": [1.0]}) is None
    assert qlib158.evaluate_qlib158_factor("MA5", {}) is None


def test_eligibility_bar_marks_without_dropping():
    default_bar = qlib158.Qlib158EligibilityBar()
    assert default_bar.min_abs_ic == qlib158.MIN_ABS_IC
    assert default_bar.min_icir == qlib158.MIN_ICIR

    passing = qlib158.evaluate_factor_eligibility(abs_ic=0.05, icir=0.5)
    assert passing["passes"] is True
    assert passing["disposition"] == "eligible"

    low_ic = qlib158.evaluate_factor_eligibility(abs_ic=0.005, icir=0.9)
    assert low_ic["passes"] is False
    assert low_ic["disposition"] == "below_ic_bar"
    # below-bar factors are still reported, never silently discarded.
    assert low_ic["abs_ic"] == pytest.approx(0.005)

    low_icir = qlib158.evaluate_factor_eligibility(abs_ic=0.5, icir=0.1)
    assert low_icir["disposition"] == "below_icir_bar"

    missing = qlib158.evaluate_factor_eligibility(abs_ic=None, icir=0.5)
    assert missing["disposition"] == "insufficient_stats"
    assert missing["passes"] is False


def test_config_gate_defaults_off():
    cfg = RuntimeConfig()
    assert cfg.strategies.qlib158_formula.enabled is False
    assert cfg.strategies.qlib158_formula.min_abs_ic == pytest.approx(0.02)
    assert cfg.strategies.qlib158_formula.min_icir == pytest.approx(0.30)
    # Standalone dataclass default is also OFF.
    assert StrategiesConfig().qlib158_formula.enabled is False
