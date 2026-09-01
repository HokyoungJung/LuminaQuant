from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "research" / "profit_moonshot_research.py"
)
SPEC = importlib.util.spec_from_file_location("profit_moonshot_research", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
profit_moonshot_research = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = profit_moonshot_research
SPEC.loader.exec_module(profit_moonshot_research)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_summary_promotes_live_equivalent_candidate_over_vector_report_only(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "vector_candidates.json",
        {
            "artifact_kind": "vector_research_candidates",
            "candidates": [
                {
                    "candidate_id": "vector-high-return",
                    "strategy_class": "ProfitMoonshotBreakoutStrategy",
                    "family": "moonshot_breakout",
                    "metrics": {
                        "val": {
                            "total_return": 0.42,
                            "max_drawdown": 0.18,
                            "sharpe": 3.0,
                            "sortino": 4.0,
                            "trade_count": 40,
                            "liquidation_count": 0,
                            "final_equity": 142000.0,
                        }
                    },
                    "promoted": True,
                }
            ],
        },
    )
    _write_json(
        tmp_path / "live_equivalent_revalidation_latest.json",
        {
            "artifact_kind": "live_equivalent_revalidation",
            "mode_candidate_rows": [
                {
                    "mode": "profit_moonshot_reversion_mode",
                    "selection_role": "alpha",
                    "selection_eligible": True,
                    "status": "live_equivalent_validated",
                    "val_total_return": 0.08,
                    "val_max_drawdown": 0.04,
                    "val_sharpe": 1.5,
                    "val_sortino": 2.2,
                    "val_trade_count": 12,
                    "val_liquidation_count": 0,
                    "val_final_equity": 108000.0,
                },
                {
                    "mode": "cash_fallback",
                    "selection_role": "fallback",
                    "selection_eligible": False,
                    "fallback_eligible": True,
                    "status": "eligible_conservative_cash_fallback",
                    "alpha_blocking_reasons": "fallback_only;val_total_return_not_positive",
                    "val_total_return": 0.0,
                    "val_trade_count": 0,
                },
            ],
        },
    )

    summary = profit_moonshot_research.build_summary(
        input_dir=tmp_path,
        output_dir=tmp_path,
        generated_at="2026-05-01T00:00:00Z",
        top_n=10,
    )

    assert summary["candidate_count"] == 3
    assert summary["promotion_eligible_count"] == 1
    assert summary["promoted_candidate"]["candidate_id"] == "profit_moonshot_reversion_mode"
    assert summary["promoted_candidate"]["source_kind"] == "live_equivalent"
    assert summary["best_return_candidate"]["candidate_id"] == "profit_moonshot_reversion_mode"
    assert summary["best_report_only_candidate"]["candidate_id"] == "profit_moonshot_reversion_mode"
    blocker_names = {row["blocker"] for row in summary["blocker_summary"]}
    assert "fallback_only" in blocker_names
    assert summary["source_counts"] == {"live_equivalent": 2, "vector": 1}


def test_main_writes_json_and_markdown_with_required_metric_columns(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "single_live.json",
        {
            "artifact_kind": "live_equivalent_revalidation",
            "rows": [
                {
                    "candidate_id": "live-alpha",
                    "selection_eligible": True,
                    "status": "live_equivalent_validated",
                    "metrics": {
                        "val": {
                            "total_return": 0.03,
                            "max_drawdown": 0.02,
                            "sharpe": 1.1,
                            "sortino": 1.4,
                            "trades": 5,
                            "liquidations": 0,
                            "final_equity": 103000.0,
                        }
                    },
                }
            ],
        },
    )

    rc = profit_moonshot_research.main(
        [
            "--input-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--generated-at",
            "2026-05-01T00:00:00Z",
        ]
    )

    assert rc == 0
    summary_path = tmp_path / "out" / profit_moonshot_research.SUMMARY_JSON_NAME
    markdown_path = tmp_path / "out" / profit_moonshot_research.SUMMARY_MD_NAME
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["promoted_candidate"]["candidate_id"] == "live-alpha"
    assert payload["best_return_candidate"]["candidate_id"] == "live-alpha"
    assert (
        "| rank | candidate | source | split | return | MDD | Sharpe | Sortino | trades | liq | final equity | blockers |"
        in markdown
    )
    assert "Promoted Candidate" in markdown
    assert "live-alpha" in markdown


def test_oversized_artifact_is_skipped_with_scan_issue(tmp_path: Path) -> None:
    (tmp_path / "too_large.json").write_text("{" + " " * 64 + "}", encoding="utf-8")

    summary = profit_moonshot_research.build_summary(
        input_dir=tmp_path,
        output_dir=tmp_path,
        generated_at="2026-05-01T00:00:00Z",
        max_bytes=16,
    )

    assert summary["candidate_count"] == 0
    assert summary["skipped_artifacts"][0]["reason"] == "skipped_max_bytes"
    assert summary["decision"] == "no_live_equivalent_promotion_candidate"


def test_profit_moonshot_continuation_validator_requires_improvement(tmp_path: Path):
    module_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "research"
        / "validate_profit_moonshot_continuation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_profit_moonshot_continuation", module_path
    )
    validator = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = validator
    spec.loader.exec_module(validator)

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "decision": "promoted_candidate_found",
                "promoted_candidate": {
                    "mode": "profit_moonshot_adaptive_momentum_boost_mode",
                    "total_return": validator.BASELINE_VAL_RETURN + 0.001,
                    "sharpe": 0.01,
                    "sortino": 0.01,
                    "trades": 10,
                    "liquidations": 0,
                    "blockers": [],
                },
            }
        ),
        encoding="utf-8",
    )

    result = validator.validate(summary_path)

    assert result["passed"] is True
    assert result["improved_over_baseline"] is True


def test_profit_moonshot_continuation_validator_prefers_oos_operator_override(tmp_path: Path):
    module_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "research"
        / "validate_profit_moonshot_continuation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_profit_moonshot_continuation", module_path
    )
    validator = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = validator
    spec.loader.exec_module(validator)

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "decision": "operator_oos_override_candidate_found",
                "promoted_candidate": {
                    "mode": "val_only_winner",
                    "total_return": 0.012,
                    "sharpe": 0.02,
                    "sortino": 0.02,
                    "trades": 20,
                    "liquidations": 0,
                    "blockers": [],
                },
                "operator_oos_override": {
                    "user_gate_pass": True,
                    "candidate": {
                        "mode": "oos_winner",
                        "primary_split": "oos",
                        "total_return": validator.BASELINE_VAL_RETURN + 0.002,
                        "sharpe": 0.10,
                        "sortino": 0.12,
                        "trades": 30,
                        "liquidations": 0,
                        "blockers": [],
                    },
                },
                "ranked_candidates": [
                    {
                        "mode": "val_only_winner",
                        "promotion_eligible": True,
                        "total_return": 0.012,
                        "sharpe": 0.02,
                        "sortino": 0.02,
                        "trades": 20,
                        "liquidations": 0,
                        "blockers": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = validator.validate(summary_path)

    assert result["passed"] is True
    assert result["candidate_mode"] == "oos_winner"
    assert result["candidate_primary_split"] == "oos"
    assert result["operator_oos_override_active"] is True
