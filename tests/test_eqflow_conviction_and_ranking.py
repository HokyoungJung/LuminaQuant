"""Eq-flow complements B2 / A3a / A3b (deterministic, ASCII-only).

B2  = momentum-crash-scaled conviction twin (``crash_scaled_conviction_variant``)
A3a = validation-saturation conviction twin (``val_saturation_conviction_variant``)
A3b = bar_count-aware partial-fold weighting (``partial_fold_bar_count_weighting``)

Every switch defaults OFF; these tests flip them via ``monkeypatch.setitem`` on the
module-level ``_EQFLOW_VARIANT_STATE`` holder so the process global is restored
after each test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_alpha_zoo_69_asset_monthly_refit_walkforward as module

AGGRESSIVE_LABEL = "profile_optuna:growth_mdd20_gross8_69_asset_profile_optuna"
FALLBACK_BALANCED = "strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna"
FALLBACK_GROWTH = "strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna"

# The two calmar80-gate scaled cells the equity-flow diagnosis names (the +7.99%
# "clean #1" candidate and its sibling); each grows one twin per family.
GATE_MDD20_SUFFIX = "_val_ret02_calmar80_gate_val_mdd20_scaled"
GATE_MDD15_SUFFIX = "_val_ret02_calmar80_gate_val_mdd15_scaled"


def _minimal_row() -> dict[str, object]:
    return {"selection_reasons": [], "uses_locked_oos_for_selection": False}


def _year_fold() -> module.MonthlyFold:
    """A fold whose train+validation window holds >= 181 daily bars (the DM
    state machine's minimum history)."""
    return module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2024-06-01"), pd.Timestamp("2024-12-31")),
        validation=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )


def _year_index() -> pd.DatetimeIndex:
    return pd.date_range("2024-06-01", "2025-03-31", freq="1D")


def _aggressive_returns(
    fold: module.MonthlyFold,
    index: pd.DatetimeIndex,
    *,
    train_daily: float = 0.01,
    drop_days: int = 0,
    val_daily: float = 0.01,
    oos_daily: float = 0.03,
) -> pd.Series:
    """Aggressive-sleeve return stream with a controllable terminal train drop.

    ``drop_days`` negative bars at the end of TRAIN carve a deep drawdown inside
    the trailing 180-bar window; the (positive) validation slope then supplies
    the terminal rebound so the DM state resolves to bear + rebound (mu 0.0).
    ``drop_days == 0`` leaves a calm uptrend (mu 1.0).
    """
    returns = pd.Series(0.0, index=index, dtype=float)
    train_bars = index[(index >= fold.train[0]) & (index <= fold.train[1])]
    val_bars = index[(index >= fold.validation[0]) & (index <= fold.validation[1])]
    oos_bars = index[(index >= fold.locked_oos[0]) & (index <= fold.locked_oos[1])]
    train_vals = np.full(len(train_bars), float(train_daily))
    if drop_days:
        train_vals[len(train_bars) - drop_days :] = -0.02
    returns.loc[train_bars] = train_vals
    returns.loc[val_bars] = float(val_daily)
    returns.loc[oos_bars] = float(oos_daily)
    return returns


def _switch_candidates(
    aggressive_returns: pd.Series, index: pd.DatetimeIndex
) -> list[module.CandidateResult]:
    flat = pd.Series(0.001, index=index, dtype=float)
    return [
        module.CandidateResult(
            "profile_optuna", AGGRESSIVE_LABEL, "agg", _minimal_row(), aggressive_returns
        ),
        module.CandidateResult(
            "strict_efficiency", FALLBACK_BALANCED, "fb", _minimal_row(), flat.copy()
        ),
        module.CandidateResult(
            "strict_efficiency", FALLBACK_GROWTH, "fg", _minimal_row(), flat.copy()
        ),
    ]


def _oos_total(candidate: module.CandidateResult, fold: module.MonthlyFold) -> float:
    return module._period_metrics(candidate.returns, fold.locked_oos)["total_return"]


# --------------------------------------------------------------------------- #
# (a) both conviction flags OFF -> exactly the legacy 72 labels, no twins
# --------------------------------------------------------------------------- #
def test_conviction_flags_off_emit_legacy_72_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, "crash_scaled_conviction_variant", False)
    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, "val_saturation_conviction_variant", False)
    fold = _year_fold()
    index = _year_index()
    candidates = _switch_candidates(_aggressive_returns(fold, index, drop_days=31), index)

    switched = module._dynamic_conviction_switch_candidates(candidates, fold)

    assert len(switched) == 72
    assert not any(c.candidate_label.endswith("_dm_crash_scaled") for c in switched)
    assert not any(c.candidate_label.endswith("_val_sat80") for c in switched)


# --------------------------------------------------------------------------- #
# (b) B2 momentum-crash state machine, multiplier, and twin-return shaping
# --------------------------------------------------------------------------- #
def test_dm_crash_state_and_multiplier_three_regimes() -> None:
    calm = np.full(250, 0.002)
    bear = np.full(250, -0.002)
    bear_rebound = np.concatenate([np.full(240, -0.003), np.full(10, 0.02)])

    assert module._dm_crash_bear_rebound_state(calm) == (0, 0)
    assert module._dm_crash_multiplier(0, 0) == 1.0
    assert module._dm_crash_bear_rebound_state(bear) == (1, 0)
    assert module._dm_crash_multiplier(1, 0) == 0.5
    assert module._dm_crash_bear_rebound_state(bear_rebound) == (1, 1)
    assert module._dm_crash_multiplier(1, 1) == 0.0
    # thin history -> neutral state, never raises
    assert module._dm_crash_bear_rebound_state(np.full(10, 0.01)) == (0, 0)


def test_dm_crash_twin_returns_scale_only_locked_oos() -> None:
    index = _year_index()
    locked = (pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31"))
    base = pd.Series(0.02, index=index, dtype=float)
    pre_mask = index < locked[0]
    pre_len = int(pre_mask.sum())
    streams = {
        1.0: np.full(pre_len, 0.002),
        0.5: np.full(pre_len, -0.002),
        0.0: np.concatenate([np.full(pre_len - 10, -0.003), np.full(10, 0.02)]),
    }
    for expected_mu, stream in streams.items():
        twin, mu, _bear, _rebound = module._dm_crash_scaled_twin_returns(base, stream, locked)
        assert mu == expected_mu
        oos_mask = (twin.index >= locked[0]) & (twin.index <= locked[1])
        assert np.allclose(twin[oos_mask].to_numpy(), base[oos_mask].to_numpy() * expected_mu)
        # pre-OOS bars are left untouched regardless of mu
        assert np.allclose(twin[~oos_mask].to_numpy(), base[~oos_mask].to_numpy())


def test_dm_crash_twin_emitted_and_throttles_oos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, "crash_scaled_conviction_variant", True)
    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, "val_saturation_conviction_variant", False)
    fold = _year_fold()
    index = _year_index()
    # train rallies then drops for 31 bars (deep dd), validation rebounds -> mu 0.0
    candidates = _switch_candidates(_aggressive_returns(fold, index, drop_days=31), index)

    switched = module._dynamic_conviction_switch_candidates(candidates, fold)
    twins = [c for c in switched if c.candidate_label.endswith("_dm_crash_scaled")]

    assert len(twins) == 16  # two named cells x four thresholds x two fallbacks
    for twin in twins:
        assert twin.row["dm_crash_mu"] == 0.0
        assert twin.row["dm_crash_bear"] == 1
        assert twin.row["dm_crash_rebound"] == 1
        # locked-OOS window fully de-risked to cash by the crash multiplier
        assert _oos_total(twin, fold) == 0.0
    # the base scaled cell it twins is still deployed (twin genuinely throttled it)
    base_mdd20 = [c for c in switched if c.candidate_label.endswith(GATE_MDD20_SUFFIX)]
    assert base_mdd20
    assert _oos_total(base_mdd20[0], fold) > 0.0  # +0.03/bar oos stream, unthrottled


def test_dm_crash_twin_no_throttle_on_calm_uptrend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, "crash_scaled_conviction_variant", True)
    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, "val_saturation_conviction_variant", False)
    fold = _year_fold()
    index = _year_index()
    # calm uptrend everywhere -> no bear -> mu 1.0 -> twin == base
    candidates = _switch_candidates(_aggressive_returns(fold, index, drop_days=0), index)

    switched = module._dynamic_conviction_switch_candidates(candidates, fold)
    by_label = {c.candidate_label: c for c in switched}
    twins = [c for c in switched if c.candidate_label.endswith("_dm_crash_scaled")]

    assert len(twins) == 16
    for twin in twins:
        assert twin.row["dm_crash_mu"] == 1.0
        base_label = twin.candidate_label[: -len("_dm_crash_scaled")]
        assert _oos_total(twin, fold) == _oos_total(by_label[base_label], fold)


# --------------------------------------------------------------------------- #
# (c) A3a validation saturation: eff() edges + twin decision uses saturated value
# --------------------------------------------------------------------------- #
def test_val_saturation_eff_is_monotone_reflection() -> None:
    eff = module._saturate_validation_return
    assert eff(0.5) == 0.5
    assert eff(0.80) == 0.80
    assert eff(1.20) == pytest.approx(0.40)
    assert eff(1.80) == 0.0
    past_ceiling = [eff(v) for v in (0.80, 0.90, 1.00, 1.10, 1.20)]
    assert all(past_ceiling[i] >= past_ceiling[i + 1] for i in range(len(past_ceiling) - 1))
    assert past_ceiling[0] > past_ceiling[-1]


def test_val_saturation_twin_levers_less_on_extreme_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, "val_saturation_conviction_variant", True)
    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, "crash_scaled_conviction_variant", False)
    fold = _year_fold()
    index = _year_index()
    # +1.2%/day validation -> validation return > 0.80 ceiling (extreme / overfit-flag)
    aggressive = _aggressive_returns(fold, index, val_daily=0.012)
    raw_val = module._period_metrics(aggressive, fold.validation)["total_return"]
    assert raw_val > module.VAL_SATURATION_CEILING

    switched = module._dynamic_conviction_switch_candidates(
        _switch_candidates(aggressive, index), fold
    )
    sat_twins = [c for c in switched if c.candidate_label.endswith("_val_sat80")]

    assert len(sat_twins) == 16
    assert all(c.row["val_saturation_ceiling"] == module.VAL_SATURATION_CEILING for c in sat_twins)
    assert all(c.row["val_saturation_applied"] is True for c in sat_twins)

    # each saturated twin sizes strictly smaller than its raw scaled base cell
    base_mdd20 = {
        c.candidate_label: c for c in switched if c.candidate_label.endswith(GATE_MDD20_SUFFIX)
    }
    sat_mdd20 = [c for c in sat_twins if c.candidate_label.endswith("_val_mdd20_scaled_val_sat80")]
    assert sat_mdd20
    for twin in sat_mdd20:
        base_label = twin.candidate_label[: -len("_val_sat80")]
        assert twin.row["risk_scale"] < base_mdd20[base_label].row["risk_scale"]


def test_val_saturation_flag_off_emits_no_twin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, "val_saturation_conviction_variant", False)
    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, "crash_scaled_conviction_variant", False)
    fold = _year_fold()
    index = _year_index()
    aggressive = _aggressive_returns(fold, index, val_daily=0.012)
    switched = module._dynamic_conviction_switch_candidates(
        _switch_candidates(aggressive, index), fold
    )
    assert not any(c.candidate_label.endswith("_val_sat80") for c in switched)


# --------------------------------------------------------------------------- #
# (d)/(e) A3b bar_count-aware partial-fold weighting in the aggregate rankings
# --------------------------------------------------------------------------- #
def _fold_row(label: str, ret: float, mdd: float, bar_count: int) -> dict[str, object]:
    return {
        "candidate_label": label,
        "family": "f",
        "clean_promotion_eligible": True,
        "train": {"total_return": 0.2, "mdd": 0.05},
        "validation": {"total_return": 0.1, "mdd": 0.04},
        "locked_oos": {"total_return": ret, "mdd": mdd, "bar_count": bar_count},
        "ready_for_paper": True,
    }


def test_partial_fold_weighting_downweights_partial_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, "partial_fold_bar_count_weighting", True)
    rows = [
        _fold_row("p", 0.30, 0.10, 4 * 48),  # 4-day partial fold, big positive outlier
        _fold_row("p", 0.01, 0.05, 30 * 48),
        _fold_row("p", 0.02, 0.06, 30 * 48),
    ]

    agg = module._aggregate_rows(rows)[0]

    weights = agg["fold_bar_weights"]
    assert weights[0] < 1.0  # partial fold shrinks
    assert weights[1] == 1.0 and weights[2] == 1.0  # full folds keep weight 1.0
    # the big positive outlier was a partial fold -> bar-weighted comp is smaller
    assert agg["bar_weighted_compounded_oos_return"] < agg["compounded_oos_return"]
    assert agg["bar_weighted_positive_oos_folds"] < agg["positive_oos_folds"]
    # headline full-fold MDD excludes the partial fold's 0.10 drawdown
    assert agg["max_full_fold_oos_mdd"] == 0.06
    assert agg["max_oos_mdd"] == 0.10


def test_partial_fold_weighting_flips_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        # partial_spike: +30% on a 4-day partial fold, flat full folds
        _fold_row("partial_spike", 0.30, 0.02, 4 * 48),
        _fold_row("partial_spike", 0.0, 0.0, 30 * 48),
        _fold_row("partial_spike", 0.0, 0.0, 30 * 48),
        # steady_full: distributed +6% across full folds
        _fold_row("steady_full", 0.06, 0.02, 30 * 48),
        _fold_row("steady_full", 0.06, 0.02, 30 * 48),
        _fold_row("steady_full", 0.06, 0.02, 30 * 48),
    ]

    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, "partial_fold_bar_count_weighting", False)
    off = module._aggregate_rows(rows)
    assert off[0]["candidate_label"] == "partial_spike"  # raw compounded ranks the spike first

    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, "partial_fold_bar_count_weighting", True)
    on = module._aggregate_rows(rows)
    assert on[0]["candidate_label"] == "steady_full"  # bar-weighted ranks the distributed one first


def test_partial_fold_weighting_off_is_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, "partial_fold_bar_count_weighting", False)
    rows = [
        _fold_row("a", 0.10, 0.05, 30 * 48),
        _fold_row("a", -0.05, 0.06, 30 * 48),
        _fold_row("b", 0.02, 0.03, 4 * 48),
    ]

    out = module._aggregate_rows(rows)

    for row in out:
        assert "fold_bar_weights" not in row
        assert "bar_weighted_compounded_oos_return" not in row
        assert "bar_weighted_positive_oos_folds" not in row
        assert "max_full_fold_oos_mdd" not in row
    # legacy primary sort key: compounded OOS return, descending
    comps = [row["compounded_oos_return"] for row in out]
    assert comps == sorted(comps, reverse=True)
