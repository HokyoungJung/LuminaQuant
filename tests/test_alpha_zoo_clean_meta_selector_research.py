from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research import run_alpha_zoo_clean_meta_selector_research as module


def _row(label: str, fold: str, *, validation: float, oos: float) -> dict[str, object]:
    return {
        "candidate_label": label,
        "family": "strict_efficiency",
        "fold_id": fold,
        "clean_promotion_eligible": True,
        "nested_hybrid_dependency": False,
        "post_oos_research_variant": False,
        "selection_reasons": [],
        "train": {"total_return": 0.20, "mdd": 0.05},
        "validation": {"total_return": validation, "mdd": 0.04},
        "locked_oos": {"total_return": oos, "mdd": 0.03},
    }


def test_selector_score_ignores_locked_oos_fields() -> None:
    params = {
        "family_group": "strict_relaxed",
        "validation_mdd_cap": 0.25,
        "train_mdd_cap": 0.40,
        "validation_spike_cap": 0.50,
        "return_weight": 8.0,
        "calmar_weight": 0.05,
        "mdd_weight": 2.0,
        "spike_penalty_weight": 2.0,
    }
    base = _row("a", "2026-01", validation=0.12, oos=-0.90)
    changed = dict(base)
    changed["locked_oos"] = {"total_return": 3.0, "mdd": 0.01}

    assert module._selector_score(base, params) == pytest.approx(
        module._selector_score(changed, params)
    )


def test_evaluate_selector_selects_by_validation_not_oos() -> None:
    params = {
        "family_group": "strict_relaxed",
        "validation_mdd_cap": 0.25,
        "train_mdd_cap": 0.40,
        "validation_spike_cap": 0.50,
        "return_weight": 8.0,
        "calmar_weight": 0.05,
        "mdd_weight": 2.0,
        "spike_penalty_weight": 2.0,
    }
    rows_by_fold = {
        "2026-01": [
            _row("low_validation_oos_winner", "2026-01", validation=0.02, oos=0.50),
            _row("high_validation_oos_loser", "2026-01", validation=0.15, oos=-0.10),
        ]
    }

    result = module.evaluate_selector(rows_by_fold=rows_by_fold, params=params)

    assert result["choices"][0]["candidate_label"] == "high_validation_oos_loser"
    assert result["uses_locked_oos_for_fold_selection"] is False
    assert result["requires_fresh_forward_shadow"] is True
    assert result["clean_promotion_eligible"] is False
    assert result["deployment_label"] == "shadow-freeze-only"
    assert (
        "post_oos_selector_grid_ranking_uses_historical_locked_oos"
        in result["gate_summary"]["label_blockers"]
    )


def test_run_writes_shadow_artifact(tmp_path: Path) -> None:
    source = {
        "artifact_kind": "test_monthly_wf",
        "protocol": {"selection_inputs": ["train", "validation"]},
        "fold_candidate_rows": [
            _row("a", "2026-01", validation=0.10, oos=0.05),
            _row("b", "2026-02", validation=0.12, oos=0.04),
        ],
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    payload = module.run(source_path, tmp_path / "out")

    assert payload["best_selector"]["ready_for_real"] is False
    assert payload["best_selector"]["evidence_class"] == "shadow-freeze-only"
    assert payload["selector_policy"]["evidence_class_cap"] == "shadow-freeze-only"
    assert payload["best_selector"]["positive_oos_folds"] == 2
    assert Path(payload["output_paths"]["json"]).exists()
    assert Path(payload["output_paths"]["markdown"]).exists()
    manifest_path = Path(payload["freeze_manifest_path"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["evidence_class_cap"] == "shadow-freeze-only"
    assert "locked_oos" in manifest["banned_selection_fields"]
    assert manifest["trial_budget"]["grid_candidate_count"] == payload["grid_candidate_count"]
