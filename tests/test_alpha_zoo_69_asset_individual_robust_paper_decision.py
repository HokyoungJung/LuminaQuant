from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "ops" / "write_alpha_zoo_69_asset_individual_robust_paper_decision.py"
    spec = importlib.util.spec_from_file_location(
        "write_alpha_zoo_69_asset_individual_robust_paper_decision", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture_payload(*, min_oos: float = -0.0728) -> dict:
    candidate = "individual_robust:hybrid_v3_5"
    return {
        "universe": {"symbol_count": 69, "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"]},
        "folds": [
            {
                "fold_id": "2026-06",
                "refit_at": "2026-06-01T00:00:00",
                "train": {"start": "2025-01-01T00:00:00", "end": "2026-03-31T23:30:00"},
                "validation": {
                    "start": "2026-04-01T00:00:00",
                    "end": "2026-05-31T23:30:00",
                },
                "locked_oos": {
                    "start": "2026-06-01T00:00:00",
                    "end": "2026-06-01T07:00:00",
                },
            }
        ],
        "aggregate_rankings": [
            {
                "candidate_label": candidate,
                "family": "individual_robust",
                "fold_count": 10,
                "compounded_oos_return": 0.1778,
                "positive_oos_folds": 5,
                "positive_validation_folds": 10,
                "ready_for_paper_folds": 7,
                "min_oos_return": min_oos,
                "latest_oos_return": 0.0005,
                "mean_validation_return": 0.334,
                "min_validation_return": 0.245,
                "max_oos_mdd": 0.15,
            }
        ],
        "fold_candidate_rows": [
            {
                "fold_id": "2026-06",
                "family": "individual_robust",
                "candidate_label": candidate,
                "source_profile_id": "hybrid_v3_5_optuna_three_profile_blend",
                "final_weights": {
                    "individual_robust_balanced_mdd10_gross3_core10": 0.6,
                    "individual_robust_growth_mdd14_gross5_core14": 0.4,
                },
                "ready_for_paper": True,
                "selection_reasons": [],
                "validation": {"total_return": 0.3144, "mdd": 0.08},
                "locked_oos": {"total_return": 0.0005, "mdd": 0.0055},
            }
        ],
        "fold_summaries": [
            {
                "fold_id": "2026-06",
                "individual_robust_aux": {
                    "profile_rows": [
                        {
                            "profile_id": "individual_robust_balanced_mdd10_gross3_core10",
                            "gross_notional_fraction": 0.7,
                            "selected_sleeve_count": 10,
                            "asset_gross_notional_fraction": {
                                "BTC/USDT": 0.5,
                                "ETH/USDT": 0.2,
                            },
                            "selection_reasons": [],
                            "ready_for_paper": True,
                        },
                        {
                            "profile_id": "individual_robust_growth_mdd14_gross5_core14",
                            "gross_notional_fraction": 0.4,
                            "selected_sleeve_count": 14,
                            "asset_gross_notional_fraction": {
                                "BTC/USDT": 0.1,
                                "SOL/USDT": 0.3,
                            },
                            "selection_reasons": [],
                            "ready_for_paper": True,
                        },
                    ],
                    "selected_sleeve_rows": [
                        {
                            "parent_profile_id": "individual_robust_balanced_mdd10_gross3_core10",
                            "symbol": "BTC/USDT",
                            "timeframe": "30m",
                            "weighted_notional_fraction": 0.5,
                        }
                    ],
                },
            }
        ],
    }


def test_individual_robust_paper_decision_passes_gate_and_computes_exposure(
    tmp_path: Path,
) -> None:
    module = _load_module()
    artifact = tmp_path / "walkforward.json"
    artifact.write_text(module.json.dumps(_fixture_payload()), encoding="utf-8")

    payload = module.build_individual_robust_paper_decision_payload(
        walkforward_artifact_path=artifact,
        latest_fold_artifact_path=artifact,
    )

    assert payload["decision"] == "paper_shadow_selected"
    assert payload["ready_for_paper_shadow"] is True
    assert payload["ready_for_real"] is False
    assert payload["monitoring_universe"]["symbol_count"] == 3
    exposure = payload["selected_symbol_exposure"]["asset_gross_notional_fraction"]
    assert exposure["BTC/USDT"] == pytest.approx(0.34)
    assert exposure["ETH/USDT"] == pytest.approx(0.12)
    assert exposure["SOL/USDT"] == pytest.approx(0.12)
    assert payload["latest_fold_allocation"]["selected_sleeve_count"] == 1


def test_individual_robust_paper_decision_quarantines_failed_min_oos(tmp_path: Path) -> None:
    module = _load_module()
    artifact = tmp_path / "walkforward.json"
    artifact.write_text(module.json.dumps(_fixture_payload(min_oos=-0.12)), encoding="utf-8")

    payload = module.build_individual_robust_paper_decision_payload(
        walkforward_artifact_path=artifact,
        latest_fold_artifact_path=artifact,
    )

    assert payload["decision"] == "paper_shadow_quarantine"
    assert payload["ready_for_paper_shadow"] is False
    failed = [check["name"] for check in payload["walkforward_gate"]["checks"] if not check["pass"]]
    assert failed == ["min_monthly_oos_return"]
