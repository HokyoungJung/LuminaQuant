from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "write_portfolio_max_performance_decision.py"
SPEC = importlib.util.spec_from_file_location(
    "write_portfolio_max_performance_decision_meta",
    MODULE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load write_portfolio_max_performance_decision module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _portfolio_payload(
    *, oos_return: float, oos_sharpe: float, oos_max_dd: float
) -> dict[str, object]:
    return {
        "artifact_kind": "portfolio_optimization",
        "portfolio_metrics": {
            "train": {
                "total_return": 0.02,
                "sharpe": 0.20,
                "sortino": 0.30,
                "calmar": 1.0,
                "max_drawdown": 0.05,
                "volatility": 0.10,
            },
            "val": {
                "total_return": 0.03,
                "sharpe": 0.60,
                "sortino": 0.80,
                "calmar": 2.0,
                "max_drawdown": 0.02,
                "volatility": 0.10,
            },
            "oos": {
                "total_return": oos_return,
                "sharpe": oos_sharpe,
                "sortino": oos_sharpe + 1.0,
                "calmar": 12.0,
                "max_drawdown": oos_max_dd,
                "volatility": 0.10,
            },
        },
        "weights": [{"candidate_id": "stub", "name": "stub", "weight": 1.0}],
    }


def _meta_payload(*, oos_return: float, oos_sharpe: float, oos_max_dd: float) -> dict[str, object]:
    payload = _portfolio_payload(
        oos_return=oos_return, oos_sharpe=oos_sharpe, oos_max_dd=oos_max_dd
    )
    payload["artifact_kind"] = "portfolio_superiority_meta_portfolio"
    payload["selection_basis"] = "validation_objective_then_locked_oos_robust_promotion"
    payload["universe"] = "raw_basis"
    payload["fallback_retune_required"] = False
    payload["weights"] = [
        {"candidate_id": "incumbent", "name": "incumbent", "weight": 0.4},
        {"candidate_id": "raw55_45", "name": "raw55_45", "weight": 0.6},
    ]
    payload["oos_monthly_returns"] = [
        {"month": "2026-02", "total_return": 0.025, "days": 20},
        {"month": "2026-03", "total_return": 0.028, "days": 20},
        {"month": "2026-04", "total_return": 0.027, "days": 20},
    ]
    return payload


def test_build_portfolio_max_performance_decision_includes_meta_candidate(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    dynamic = tmp_path / "dynamic.json"
    overlay = tmp_path / "overlay.json"
    regime = tmp_path / "regime.json"
    meta = tmp_path / "meta.json"

    incumbent_bundle.write_text(
        json.dumps({"selection_basis": "bundle", "candidates": []}), encoding="utf-8"
    )
    incumbent_portfolio.write_text(
        json.dumps(_portfolio_payload(oos_return=0.05, oos_sharpe=3.0, oos_max_dd=0.02)),
        encoding="utf-8",
    )
    dynamic.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    overlay.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    regime.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    meta.write_text(
        json.dumps(_meta_payload(oos_return=0.06, oos_sharpe=3.8, oos_max_dd=0.015)),
        encoding="utf-8",
    )

    payload = MODULE.build_portfolio_max_performance_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tmp_path / "missing_tuned.json",
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        regime_switch_comparison_path=regime,
        grouped_allocator_path=None,
        grouped_strict_validation_path=None,
        grouped_static_blend_path=None,
        portfolio_superiority_meta_path=meta,
        backbone_triplet_path=tmp_path / "missing_triplet.json",
        anchored_comparison_path=tmp_path / "missing_anchored.json",
    )

    meta_entry = next(
        entry
        for entry in payload["candidates"]
        if entry["candidate_key"] == "portfolio_superiority_meta_portfolio"
    )
    assert meta_entry["label"] == "Robust meta-portfolio challenger"


def test_build_portfolio_max_performance_decision_can_promote_meta_candidate(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    dynamic = tmp_path / "dynamic.json"
    overlay = tmp_path / "overlay.json"
    regime = tmp_path / "regime.json"
    meta = tmp_path / "meta.json"

    incumbent_bundle.write_text(
        json.dumps({"selection_basis": "bundle", "candidates": []}), encoding="utf-8"
    )
    incumbent_portfolio.write_text(
        json.dumps(_portfolio_payload(oos_return=0.05, oos_sharpe=3.0, oos_max_dd=0.02)),
        encoding="utf-8",
    )
    dynamic.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    overlay.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    regime.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    meta.write_text(
        json.dumps(_meta_payload(oos_return=0.07, oos_sharpe=4.2, oos_max_dd=0.015)),
        encoding="utf-8",
    )

    payload = MODULE.build_portfolio_max_performance_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tmp_path / "missing_tuned.json",
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        regime_switch_comparison_path=regime,
        grouped_allocator_path=None,
        grouped_strict_validation_path=None,
        grouped_static_blend_path=None,
        portfolio_superiority_meta_path=meta,
        backbone_triplet_path=tmp_path / "missing_triplet.json",
        anchored_comparison_path=tmp_path / "missing_anchored.json",
    )

    assert payload["winner"]["candidate_key"] == "portfolio_superiority_meta_portfolio"
