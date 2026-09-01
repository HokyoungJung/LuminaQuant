"""D4 regime-gated strict_efficiency:static_guarded twin (eq-flow v5 follow-up).

Deterministic, ASCII-only. The pre-registered switch
``regime_gated_strict_efficiency_variant`` is flipped per test via
``monkeypatch.setitem`` so the module default (OFF) is never mutated globally.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_alpha_zoo_69_asset_monthly_refit_walkforward as module

FLAG = "regime_gated_strict_efficiency_variant"
TWIN_LABEL = "strict_efficiency:static_guarded_regime_gated"


def _index() -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", "2025-03-31", freq="1D")


def _fold() -> module.MonthlyFold:
    return module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )


def _profile_stream(profile_id: str, returns: pd.Series) -> module.grid_hybrid.ProfileStream:
    splits = ("train", "validation", "locked_oos")
    return module.grid_hybrid.ProfileStream(
        profile_id=profile_id,
        candidate_tier="cross_candidate_hybrid_input",
        leverage_map={},
        gross_notional_fraction=1.0,
        asset_gross_notional_fraction={},
        selected_model_ids=(profile_id,),
        returns=returns.sort_index(),
        turnover_by_split=dict.fromkeys(splits, 1.0),
        trade_events_by_split=dict.fromkeys(splits, 1),
        liquidation_count_by_split=dict.fromkeys(splits, 0),
    )


def _static_row() -> dict:
    return {
        "profile_id": "static_guarded",
        "final_weights": {"p0": 1.0},
        "guardrail_notes": [],
    }


def _build(returns: pd.Series):
    streams = [_profile_stream("p0", returns)]
    static_row = _static_row()
    fold = _fold()
    base_static = module._combine_profile_returns(streams, {"p0": 1.0})
    return streams, static_row, fold, base_static


def _emit(monkeypatch, *, flag: bool, family: str, relaxed: bool, returns: pd.Series):
    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, FLAG, flag)
    streams, static_row, fold, base_static = _build(returns)
    extra = module._maybe_regime_gated_static_guarded_twin(
        family=family,
        relaxed=relaxed,
        fold=fold,
        profile_streams=streams,
        static_row=static_row,
    )
    return extra, static_row, fold, base_static


def test_flag_off_emits_no_twin_and_is_identity(monkeypatch) -> None:
    returns = pd.Series(0.002, index=_index(), dtype=float)
    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, FLAG, False)
    streams, static_row, fold, _ = _build(returns)
    legacy = [
        module._candidate_eval(
            family="strict_efficiency",
            label="strict_efficiency:static_guarded",
            row=static_row,
            returns=module._combine_profile_returns(streams, {"p0": 1.0}),
        )
    ]
    extra = module._maybe_regime_gated_static_guarded_twin(
        family="strict_efficiency",
        relaxed=False,
        fold=fold,
        profile_streams=streams,
        static_row=static_row,
    )
    assert extra == []
    combined = [*legacy, *extra]
    assert len(combined) == len(legacy)
    assert all(left is right for left, right in zip(combined, legacy))
    assert TWIN_LABEL not in [candidate.candidate_label for candidate in combined]


def test_flag_on_gate_passes_keeps_oos(monkeypatch) -> None:
    returns = pd.Series(0.002, index=_index(), dtype=float)
    extra, static_row, fold, base_static = _emit(
        monkeypatch, flag=True, family="strict_efficiency", relaxed=False, returns=returns
    )
    assert len(extra) == 1
    twin = extra[0]
    assert twin.candidate_label == TWIN_LABEL
    assert twin.family == "strict_efficiency"
    assert twin.row["regime_gate_passed"] is True
    assert twin.row["regime_gate_validation_return"] > 0.0
    assert twin.row["regime_gate_validation_mdd"] <= module.REGIME_GATE_MAX_VALIDATION_MDD
    # Gate deployed -> locked-OOS window is the untouched base static stream.
    base_oos = module._period_metrics(base_static, fold.locked_oos)
    assert base_oos["total_return"] > 0.0
    assert module._period_metrics(twin.returns, fold.locked_oos) == base_oos
    # Copy semantics: the base static_guarded row is never mutated.
    assert static_row["guardrail_notes"] == []
    assert any("regime_gated_static_guarded" in note for note in twin.row["guardrail_notes"])


def test_flag_on_bleeding_validation_zeros_oos(monkeypatch) -> None:
    returns = pd.Series(0.002, index=_index(), dtype=float)
    returns.loc["2025-02-01":"2025-02-28"] = -0.01  # net-negative validation window
    extra, _static_row, fold, base_static = _emit(
        monkeypatch, flag=True, family="strict_efficiency", relaxed=False, returns=returns
    )
    assert len(extra) == 1
    twin = extra[0]
    assert twin.row["regime_gate_passed"] is False
    assert twin.row["regime_gate_validation_return"] < 0.0
    # Locked-OOS window zeroed to cash; base OOS was non-zero (gate changed it).
    oos_slice = twin.returns.loc["2025-03-01":"2025-03-31"]
    assert bool((oos_slice == 0.0).all())
    assert module._period_metrics(twin.returns, fold.locked_oos)["total_return"] == 0.0
    assert module._period_metrics(base_static, fold.locked_oos)["total_return"] > 0.0
    # Train + validation windows identical to base (pre-OOS untouched).
    assert module._period_metrics(twin.returns, fold.train) == module._period_metrics(
        base_static, fold.train
    )
    assert module._period_metrics(twin.returns, fold.validation) == module._period_metrics(
        base_static, fold.validation
    )


def test_flag_on_high_validation_mdd_fails_gate(monkeypatch) -> None:
    idx = _index()
    returns = pd.Series(0.002, index=idx, dtype=float)
    val_days = pd.date_range("2025-02-01", "2025-02-28", freq="1D")
    # Build up gains, then a sharp mid-window drawdown that recovers: total return
    # stays positive but the peak-to-trough drawdown exceeds the 0.12 budget.
    returns.loc[val_days] = 0.01
    returns.loc[val_days[14]] = -0.15
    extra, _static_row, _fold_obj, _base_static = _emit(
        monkeypatch, flag=True, family="strict_efficiency", relaxed=False, returns=returns
    )
    assert len(extra) == 1
    twin = extra[0]
    assert twin.row["regime_gate_validation_return"] > 0.0
    assert twin.row["regime_gate_validation_mdd"] > module.REGIME_GATE_MAX_VALIDATION_MDD
    assert twin.row["regime_gate_passed"] is False
    oos_slice = twin.returns.loc["2025-03-01":"2025-03-31"]
    assert bool((oos_slice == 0.0).all())


def test_relaxed_family_flag_on_emits_no_twin(monkeypatch) -> None:
    returns = pd.Series(0.002, index=_index(), dtype=float)
    extra, _static_row, _fold, _base_static = _emit(
        monkeypatch, flag=True, family="relaxed_efficiency", relaxed=True, returns=returns
    )
    assert extra == []
