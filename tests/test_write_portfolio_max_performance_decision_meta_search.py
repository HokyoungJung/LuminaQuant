from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "write_portfolio_max_performance_decision.py"
SPEC = importlib.util.spec_from_file_location(
    "write_portfolio_max_performance_decision", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load write_portfolio_max_performance_decision module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _incumbent_bundle_payload() -> dict:
    return {
        "selection_basis": "incumbent_saved_one_shot_weights",
        "candidates": [
            {
                "candidate_id": "incumbent_component",
                "name": "incumbent_component",
                "strategy_class": "StubStrategy",
                "timeframe": "1h",
                "portfolio_weight": 1.0,
            }
        ],
    }


def _incumbent_portfolio_payload() -> dict:
    return {
        "portfolio_metrics": {
            "train": {"total_return": 0.03, "sharpe": 1.0},
            "val": {"total_return": 0.025, "sharpe": 1.1},
            "oos": {
                "total_return": 0.04,
                "sharpe": 1.5,
                "sortino": 2.0,
                "calmar": 4.0,
                "max_drawdown": 0.06,
                "volatility": 0.12,
            },
        },
        "weights": [
            {"candidate_id": "incumbent_component", "name": "incumbent_component", "weight": 1.0}
        ],
    }


def _meta_summary_payload(path: Path) -> dict:
    return {
        "artifact_kind": "portfolio_superiority_meta_search",
        "selection_basis": "validation_objective_then_locked_oos",
        "universe_name": "u1_raw_basis",
        "winner_status": "promoted_challenger",
        "winner": {
            "train": {"total_return": 0.05, "sharpe": 0.8},
            "val": {"total_return": 0.04, "sharpe": 1.2},
            "oos": {
                "total_return": 0.08,
                "sharpe": 2.0,
                "sortino": 2.7,
                "calmar": 5.0,
                "max_drawdown": 0.05,
                "volatility": 0.10,
            },
            "weights": [
                {"candidate_key": "incumbent", "weight": 0.2},
                {"candidate_key": "autoresearch_pair_55_45", "weight": 0.8},
            ],
            "rejection_reasons": [],
            "path": str(path.resolve()),
        },
    }


def test_build_decision_includes_promoted_meta_search_entry(tmp_path: Path) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    meta_summary = tmp_path / "u1_raw_basis_summary_latest.json"

    incumbent_bundle.write_text(json.dumps(_incumbent_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(json.dumps(_incumbent_portfolio_payload()), encoding="utf-8")
    meta_summary.write_text(json.dumps(_meta_summary_payload(meta_summary)), encoding="utf-8")

    payload = MODULE.build_portfolio_max_performance_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tmp_path / "missing_tuned.json",
        dynamic_comparison_path=tmp_path / "missing_dynamic.json",
        overlay_comparison_path=tmp_path / "missing_overlay.json",
        regime_switch_comparison_path=tmp_path / "missing_regime.json",
        grouped_static_blend_path=tmp_path / "missing_static_blend.json",
        backbone_triplet_path=tmp_path / "missing_triplet.json",
        anchored_comparison_path=tmp_path / "missing_anchor.json",
        meta_search_summary_paths=(meta_summary,),
    )

    assert "portfolio_superiority_meta_search" in payload["supporting_artifacts"]
    assert any(
        entry["candidate_key"] == "portfolio_superiority_meta_portfolio"
        for entry in payload["candidates"]
    )
    assert payload["winner"]["candidate_key"] == "portfolio_superiority_meta_portfolio"
