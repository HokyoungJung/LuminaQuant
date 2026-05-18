from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_live_notional_risk_aligned_alpha_zoo.py"
SPEC = importlib.util.spec_from_file_location("run_live_notional_risk_aligned_alpha_zoo", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_paper_equivalent_sizing_contract_matches_replay_notional() -> None:
    risk_caps = MODULE._risk_caps_for_contract(leverage=7.0, allocation_fraction=0.15)

    payload = MODULE._paper_equivalent_sizing(
        leverage=7.0,
        allocation_fraction=0.15,
        sizing_mode="isolated_margin_fraction",
        risk_caps=risk_caps,
    )

    assert payload["expected_replay_notional"] == 10_500.0
    assert payload["live_notional"] == 10_500.0
    assert payload["notional_parity_passed"] is True
    assert payload["risk_check_passed"] is True


def test_cost_sensitivity_required_grids_are_explicit() -> None:
    assert set(MODULE.SLIPPAGE_FEE_BPS_GRID) >= {1.0, 3.0, 5.0, 10.0, 20.0}
    assert set(MODULE.FUNDING_BPS_PER_DAY_GRID) >= {1.0, 2.0, 5.0, 10.0, 20.0}


def test_incumbent_tie_breaker_preserves_requested_7x_15pct_contract(tmp_path: Path) -> None:
    csv_path = tmp_path / "candidates.csv"
    csv_path.write_text(
        "\n".join(
            [
                (
                    "candidate_name,leverage,allocation_fraction,"
                    "frozen_train_validation_rank,tv_selection_score,"
                    "locked_oos_liquidation_count,total_account_wipeout_count,"
                    "locked_oos_gate_pass,live_promotion_possible"
                ),
                "alpha_zoo_fast_residual,7,0.15,34,5.703989805016097,0,0,True,True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    base_payload = {
        "selection": {
            "live_promoted_candidate": {
                "candidate_name": "alpha_zoo_fast_residual",
                "leverage": 6.0,
                "allocation_fraction": 0.175,
                "tv_selection_score": 5.703989805016102,
                "locked_oos": {"total_return": 0.3053573988518672},
            }
        }
    }

    payload = MODULE._prefer_incumbent_contract_on_tv_tie(
        base_payload,
        candidate_csv_path=csv_path,
        incumbent_candidate="alpha_zoo_fast_residual",
        incumbent_leverage=7.0,
        incumbent_allocation=0.15,
    )

    promoted = payload["selection"]["live_promoted_candidate"]
    assert promoted["leverage"] == 7.0
    assert promoted["allocation_fraction"] == 0.15
    assert promoted["tv_selection_score"] == 5.703989805016097
    assert payload["selection"]["incumbent_contract_tie_breaker"]["applied"] is True
