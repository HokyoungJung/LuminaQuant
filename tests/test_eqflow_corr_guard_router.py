"""C1 eq-flow complement: corr-guarded twin of the pre-registered risk-trimmed
lagged-router candidate.

Deterministic, ASCII-only, no ``random`` module: synthetic per-symbol panels are
built from fixed trig sequences so the average-pairwise-correlation crash guard has
a reproducible engage / neutral decision. The ``corr_guard_router_variant`` switch is
flipped per test via ``monkeypatch.setitem`` on ``_EQFLOW_VARIANT_STATE`` so the
default OFF path stays byte-identical to the legacy candidate set.
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

from scripts.research import run_monthly_refit_walkforward as module

RELAXED_LABEL = "relaxed_efficiency:growth_mdd20_gross8_69_asset_relaxed_efficiency_repair_optuna"
STRICT_LABEL = "strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna"
GUARD_LABEL = module.PREREGISTERED_LAGGED_LEAF_ROUTER_CORR_GUARD_LABEL
RISK_TRIMMED_LABEL = module.PREREGISTERED_LAGGED_LEAF_ROUTER_RISK_TRIMMED_LABEL
OOS_START = pd.Timestamp("2025-03-01")
OOS_END = pd.Timestamp("2025-03-31")

PRIOR_COMPLETED_RETURNS = {
    RELAXED_LABEL: [0.00, -0.01, 0.12, 0.10],
    STRICT_LABEL: [0.03, 0.02, 0.01, 0.00],
}
PRIOR_COMPLETED_FOLDS = ("2024-11", "2024-12", "2025-01", "2025-02")


def _fold() -> module.MonthlyFold:
    return module.MonthlyFold(
        fold_id="2025-03",
        refit_at=OOS_START,
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(OOS_START, OOS_END),
    )


def _leaf_candidate(
    label: str, family: str, *, oos_daily: float, base_daily: float = 0.003
) -> module.CandidateResult:
    index = pd.date_range("2025-01-01", "2025-03-31", freq="1D")
    returns = pd.Series(base_daily, index=index, dtype=float)
    returns.loc[OOS_START:OOS_END] = oos_daily
    return module.CandidateResult(
        family=family,
        candidate_label=label,
        source_profile_id=label,
        row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
        returns=returns,
    )


def _router_candidates() -> list[module.CandidateResult]:
    # Relaxed is the prior-shadow winner (its OOS is deliberately the WORST here, so
    # the router cannot have chosen by same-month OOS); it is what the pre-registered
    # risk-trimmed row and its corr-guard twin deploy.
    return [
        _leaf_candidate(RELAXED_LABEL, "relaxed_efficiency", oos_daily=-0.020),
        _leaf_candidate(STRICT_LABEL, "strict_efficiency", oos_daily=0.020, base_daily=0.002),
    ]


def _symbol_streams(
    n_symbols: int, *, common_tail: bool, oos_override: float | None = None
) -> list[object]:
    """Per-symbol ``CandidateStream`` panel over a pre-OOS + OOS daily index.

    ``common_tail`` injects a shared return component into the LAST
    ``CORR_GUARD_WINDOW`` pre-OOS bars (idiosyncratic distinct-frequency trig before)
    so the trailing correlation z-spikes at the final pre-OOS position; otherwise each
    symbol stays on its own frequency and the basket is decorrelated. ``oos_override``
    overwrites ONLY the locked-OOS bars (for the no-look-ahead check).
    """
    stream_index = pd.date_range("2024-01-01", "2025-03-31", freq="1D")
    pre_positions = np.flatnonzero(np.asarray(stream_index < OOS_START))
    window = module.CORR_GUARD_WINDOW
    bars = np.arange(len(stream_index), dtype=float)
    common = 0.05 * np.sin(2.0 * np.pi * bars / 23.0)
    streams: list[object] = []
    for i in range(n_symbols):
        idiosyncratic = 0.01 * np.sin(2.0 * np.pi * (bars + i) / (7.0 + 5.0 * i))
        values = idiosyncratic.copy()
        if common_tail:
            tail = pre_positions[-window:]
            values[tail] = common[tail] + 1e-6 * idiosyncratic[tail]
        returns = pd.Series(values, index=stream_index, dtype=float)
        if oos_override is not None:
            returns.loc[OOS_START:] = oos_override
        streams.append(
            module.broad69.CandidateStream(
                row={"symbol": f"SYM{i}"},
                returns=returns,
                position=pd.Series(0.0, index=stream_index, dtype=float),
            )
        )
    return streams


def _route(individual_streams: list[object] | None) -> dict[str, module.CandidateResult]:
    routed = module._lagged_shadow_leaf_router_candidates(
        _router_candidates(),
        _fold(),
        prior_completed_returns=PRIOR_COMPLETED_RETURNS,
        prior_completed_fold_ids=PRIOR_COMPLETED_FOLDS,
        individual_streams=individual_streams,
    )
    return {candidate.candidate_label: candidate for candidate in routed}


def _oos_slice(candidate: module.CandidateResult) -> np.ndarray:
    returns = candidate.returns
    mask = (returns.index >= OOS_START) & (returns.index <= OOS_END)
    return returns[mask].to_numpy()


def _pre_oos_slice(candidate: module.CandidateResult) -> np.ndarray:
    returns = candidate.returns
    return returns[returns.index < OOS_START].to_numpy()


def test_flag_off_output_is_byte_identical_and_has_no_guard_label(monkeypatch) -> None:
    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, "corr_guard_router_variant", False)
    with_streams = _route(_symbol_streams(5, common_tail=True))
    without_streams = _route(None)
    # No new label and identical candidate roster whether or not streams are supplied.
    assert GUARD_LABEL not in with_streams
    assert GUARD_LABEL not in without_streams
    assert list(with_streams) == list(without_streams)
    for label, candidate in with_streams.items():
        pd.testing.assert_series_equal(candidate.returns, without_streams[label].returns)


def test_flag_on_high_correlation_engages_and_derisks_oos(monkeypatch) -> None:
    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, "corr_guard_router_variant", True)
    routed = _route(_symbol_streams(5, common_tail=True))
    assert GUARD_LABEL in routed
    guard = routed[GUARD_LABEL]
    base = routed[RISK_TRIMMED_LABEL]
    assert guard.row["corr_guard_engaged"] is True
    assert guard.row["corr_guard_rho"] >= module.CORR_GUARD_ABS_FLOOR
    assert guard.row["corr_guard_z"] >= module.CORR_GUARD_Z_ENTER
    # Twin is an exact copy of the risk-trimmed row except the guard fields / label.
    assert guard.row["selected_candidate_label"] == base.row["selected_candidate_label"]
    assert guard.row["router_branch"] == base.row["router_branch"]
    assert guard.source_profile_id == GUARD_LABEL
    # Locked-OOS slice is scaled by the de-risk factor; pre-OOS bars are untouched.
    assert _oos_slice(guard) == pytest.approx(_oos_slice(base) * module.CORR_GUARD_DERISK_SCALE)
    assert _pre_oos_slice(guard) == pytest.approx(_pre_oos_slice(base))


def test_flag_on_decorrelated_panel_stays_neutral(monkeypatch) -> None:
    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, "corr_guard_router_variant", True)
    routed = _route(_symbol_streams(5, common_tail=False))
    guard = routed[GUARD_LABEL]
    base = routed[RISK_TRIMMED_LABEL]
    assert guard.row["corr_guard_engaged"] is False
    assert guard.row["corr_guard_rho"] < module.CORR_GUARD_ABS_FLOOR
    assert _oos_slice(guard) == pytest.approx(_oos_slice(base))


def test_guard_decision_ignores_locked_oos_bars(monkeypatch) -> None:
    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, "corr_guard_router_variant", True)
    baseline = module._corr_guard_router_decision(_symbol_streams(5, common_tail=True), _fold())
    perturbed = module._corr_guard_router_decision(
        _symbol_streams(5, common_tail=True, oos_override=0.5), _fold()
    )
    assert baseline == perturbed
    assert baseline[0] is True


def test_flag_on_without_streams_emits_neutral_twin(monkeypatch) -> None:
    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, "corr_guard_router_variant", True)
    routed = _route(None)
    assert GUARD_LABEL in routed
    guard = routed[GUARD_LABEL]
    base = routed[RISK_TRIMMED_LABEL]
    assert guard.row["corr_guard_engaged"] is False
    assert guard.row["corr_guard_rho"] is None
    assert guard.row["corr_guard_z"] is None
    assert _oos_slice(guard) == pytest.approx(_oos_slice(base))


def test_too_few_symbol_streams_stays_neutral(monkeypatch) -> None:
    monkeypatch.setitem(module._EQFLOW_VARIANT_STATE, "corr_guard_router_variant", True)
    engaged, rho, z = module._corr_guard_router_decision(
        _symbol_streams(module.CORR_GUARD_MIN_SYMBOLS - 1, common_tail=True), _fold()
    )
    assert engaged is False
    assert rho is None
    assert z is None
