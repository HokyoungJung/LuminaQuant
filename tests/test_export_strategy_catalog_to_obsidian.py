from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import threading
from pathlib import Path, PurePosixPath

import pytest


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "research" / "export_strategy_catalog_to_obsidian.py"
    spec = importlib.util.spec_from_file_location(
        "export_strategy_catalog_to_obsidian_script", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load Obsidian exporter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_relationship_graph_adds_primary_and_cross_family_links(tmp_path: Path) -> None:
    catalog = {
        "strategies": [
            {"strategy": "HybridStrategy", "family": "trend"},
            {"strategy": "CarryStrategy", "family": "carry"},
        ],
        "families": [{"family": "trend"}, {"family": "carry"}],
    }
    graph_path = tmp_path / "relationships.json"
    graph_path.write_text(
        json.dumps(
            {
                "schema": "luminaquant.strategy_relationships.v1",
                "nodes": [
                    {"id": "HybridStrategy", "type": "strategy_or_candidate"},
                    {"id": "CarryStrategy", "type": "strategy_or_candidate"},
                    {"id": "trend", "type": "family_or_axis"},
                    {"id": "carry", "type": "family_or_axis"},
                ],
                "edges": [
                    {
                        "source_id": "HybridStrategy",
                        "target_id": "trend",
                        "relation": "member_of",
                    },
                    {
                        "source_id": "HybridStrategy",
                        "target_id": "carry",
                        "relation": "cross_family_axis",
                    },
                    {
                        "source_id": "CarryStrategy",
                        "target_id": "carry",
                        "relation": "member_of",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    enriched, identity = MODULE._apply_relationship_graph(catalog, graph_path)

    hybrid = enriched["strategies"][0]
    assert identity["sha256"]
    assert hybrid["relationships"] == [
        {
            "category": "Families",
            "direction": "outbound",
            "relation": "cross_family_axis",
            "target": "carry",
        }
    ]


def _catalog() -> dict:
    metrics = {
        "provenance": {
            "artifact_kind": "common_period_1m_cold_start_strategy_screen",
            "selection_provenance": "unsealed",
        },
        "full": {
            "status": "pass",
            "return": 0.10,
            "sharpe": 1.2,
            "max_drawdown": 0.03,
            "trades": 12,
        },
        "recent": {
            "status": "pass",
            "return": 0.02,
            "sharpe": 0.8,
            "max_drawdown": 0.02,
            "trades": 3,
        },
        "comparable": True,
    }
    return {
        "schema_version": "lumina_quant.strategy_catalog.v1",
        "generated_at_utc": "2026-08-12T00:00:00Z",
        "counts": {
            "registry": 2,
            "comparison_rows": 2,
            "in_scope": 1,
            "verified_in_scope": 1,
            "scope_unverified": 0,
            "excluded": 1,
        },
        "metric_contract": {
            "artifact_kind": "common_period_1m_cold_start_strategy_screen",
            "selection_provenance": "unsealed",
            "timeframe": "1m",
            "promotion_use": "forbidden",
        },
        "scorecard_summary": {
            "raw_registry_diagnostic": {
                "strategy_count": 2,
                "evaluated_strategy_count": 2,
                "not_evaluated_strategy_count": 0,
                "comparable_count": 1,
                "positive_both_count": 1,
                "scope": "registry_accounting_with_supplied_common_screen_rows",
                "matched_control_policy": "raw_total_return_not_identification",
            },
            "catalog_controlled_diagnostic": {
                "strategy_count": 1,
                "comparable_count": 1,
                "positive_both_count": 1,
                "scope": "explicit_sub_minute_excluded_scope_unverified_retained",
                "matched_control_policy": "unmatched_rebalancing_metrics_suppressed",
            },
            "verified_ge_1m_controlled": {
                "strategy_count": 1,
                "comparable_count": 1,
                "positive_both_count": 1,
                "scope": "verified_in_scope_only",
                "matched_control_policy": "unmatched_rebalancing_metrics_suppressed",
            },
        },
        "raw_registry_scorecard_rows": [
            {
                "strategy": "ExampleStrategy",
                "scope_status": "verified_in_scope",
                "metrics": {
                    "provenance": {"artifact_kind": "common_period_1m_cold_start_strategy_screen"},
                    "full": {
                        "return": 0.10,
                        "sharpe": 1.2,
                        "max_drawdown": 0.03,
                    },
                    "recent": {
                        "return": 0.02,
                        "sharpe": 0.8,
                        "max_drawdown": 0.02,
                    },
                    "comparable": True,
                },
            },
            {
                "strategy": "OneSecondStrategy",
                "scope_status": "excluded_sub_minute",
                "metrics": {
                    "provenance": {"artifact_kind": "common_period_1m_cold_start_strategy_screen"},
                    "full": {
                        "return": None,
                        "sharpe": None,
                        "max_drawdown": None,
                    },
                    "recent": {
                        "return": None,
                        "sharpe": None,
                        "max_drawdown": None,
                    },
                    "comparable": False,
                },
            },
        ],
        "exclusions": [
            {
                "strategy": "OneSecondStrategy",
                "exclusion_reason": "requires_sub_minute_timeframes_only",
                "scope_status": "excluded_sub_minute",
            }
        ],
        "evidence_sources": [
            {
                "id": "paper-1",
                "title": "Primary Paper",
                "authors": "Researcher",
                "grade": "A",
                "type": "peer_reviewed",
                "publication": "Journal",
                "date": "2025-01-01",
                "url": "https://example.test/paper",
                "supports": "A design prior.",
                "limitations": "Not repository performance.",
            }
        ],
        "strategies": [
            {
                "strategy": "ExampleStrategy",
                "family": "trend_momentum",
                "tier": "research_only",
                "module": "lumina_quant.strategies.example",
                "execution_interface": "event_driven",
                "runner_kind": "event_backtest_engine",
                "live_execution_supported": False,
                "evidence_ids": ["paper-1"],
                "family_context_evidence_ids": ["paper-1"],
                "cadence": {
                    "scope_status": "verified_in_scope",
                    "cadence_status": "explicit_in_scope",
                    "decision_cadence_seconds": 3600,
                    "required_timeframes": ["1h"],
                },
                "required_features": ["ohlcv"],
                "candidate_library": {
                    "mapped": True,
                    "candidate_count": 1,
                    "families": ["trend"],
                    "timeframes": ["1h"],
                },
                "metrics": metrics,
                "research_note": "",
            }
        ],
        "families": [
            {
                "family": "trend_momentum",
                "title": "Trend and Momentum",
                "thesis": "Test continuation with conservative costs.",
                "research_decision": "prioritize_slow_ablation",
                "evidence_ids": ["paper-1"],
                "strategy_count": 1,
                "verified_strategy_count": 1,
                "scope_unverified_strategy_count": 0,
                "scorecard_scope": "verified_ge_1m_controlled",
                "comparable_count": 1,
                "coverage_ratio": 1.0,
                "positive_both_count": 1,
                "catalog_diagnostic_comparable_count": 1,
                "catalog_diagnostic_positive_both_count": 1,
                "full_median_return": 0.10,
                "recent_median_return": 0.02,
                "full_median_sharpe": 1.2,
                "recent_median_sharpe": 0.8,
                "full_median_max_drawdown": 0.03,
                "recent_median_max_drawdown": 0.02,
                "strategies": ["ExampleStrategy"],
            }
        ],
    }


def _write_catalog(path: Path, payload: dict | None = None) -> Path:
    path.write_text(json.dumps(payload or _catalog()), encoding="utf-8")
    return path


def test_namespace_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="unsafe Obsidian namespace"):
        MODULE._safe_namespace("../outside")
    with pytest.raises(ValueError, match="unsafe Obsidian namespace"):
        MODULE._safe_namespace("/absolute")
    with pytest.raises(ValueError, match="unsafe Obsidian namespace"):
        MODULE.apply_staged_namespace(
            Path("/tmp/not-used"),
            vault_root=Path("/tmp/not-used"),
            namespace=PurePosixPath("../outside"),
            generated_at="2026-08-12T00:00:00Z",
        )


def test_catalog_rejects_implausibly_future_generation_time() -> None:
    payload = _catalog()
    payload["generated_at_utc"] = "2999-01-01T00:00:00Z"

    with pytest.raises(ValueError, match="implausibly in the future"):
        MODULE._validate_catalog(payload)


def test_catalog_accepts_explicitly_unevaluated_research_strategy() -> None:
    payload = _catalog()
    metrics = payload["strategies"][0]["metrics"]
    metrics.update(
        {
            "comparable": False,
            "provenance": {"artifact_kind": "not_evaluated_in_supplied_comparison"},
            "full": {
                "status": "not_available",
                "return": None,
                "sharpe": None,
                "max_drawdown": None,
                "trades": None,
            },
            "recent": {
                "status": "not_available",
                "return": None,
                "sharpe": None,
                "max_drawdown": None,
                "trades": None,
            },
        }
    )
    for label in ("catalog_controlled_diagnostic", "verified_ge_1m_controlled"):
        payload["scorecard_summary"][label]["comparable_count"] = 0
        payload["scorecard_summary"][label]["positive_both_count"] = 0
    family = payload["families"][0]
    family.update(
        {
            "comparable_count": 0,
            "coverage_ratio": 0.0,
            "positive_both_count": 0,
            "catalog_diagnostic_comparable_count": 0,
            "catalog_diagnostic_positive_both_count": 0,
            "full_median_return": None,
            "recent_median_return": None,
            "full_median_sharpe": None,
            "recent_median_sharpe": None,
            "full_median_max_drawdown": None,
            "recent_median_max_drawdown": None,
        }
    )

    MODULE._validate_catalog(payload)


def test_catalog_rejects_scope_counts_not_derived_from_rows() -> None:
    payload = _catalog()
    payload["strategies"][0]["cadence"]["scope_status"] = "scope_unverified"

    with pytest.raises(ValueError, match="scope counts do not match strategy rows"):
        MODULE._validate_catalog(payload)


def test_catalog_rejects_raw_scorecard_not_derived_from_rows() -> None:
    payload = _catalog()
    payload["raw_registry_scorecard_rows"][0]["metrics"]["recent"]["return"] = -0.02

    with pytest.raises(ValueError, match="raw scorecard summary does not match raw rows"):
        MODULE._validate_catalog(payload)


def test_catalog_rejects_family_aggregates_not_derived_from_rows() -> None:
    payload = _catalog()
    payload["families"][0]["full_median_return"] = 0.11

    with pytest.raises(ValueError, match="full_median_return does not match strategy rows"):
        MODULE._validate_catalog(payload)


def test_stage_builds_link_complete_graph(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.json")
    namespace = MODULE._safe_namespace("LuminaQuant/Generated")
    stage, manifest = MODULE.stage_catalog(
        MODULE._load_catalog(catalog_path),
        catalog_path=catalog_path,
        staging_root=tmp_path / "stage",
        namespace=namespace,
        generated_at="2026-08-12T00:00:00Z",
    )

    assert manifest["counts"] == {
        "note_count": 4,
        "link_count": 13,
        "broken_link_count": 0,
        "strategies": 1,
        "families": 1,
        "evidence": 1,
    }
    assert (stage / "_generated_manifest.json").is_file()
    strategy_note = (
        stage / "Strategies" / f"{MODULE._note_filename('ExampleStrategy')}.md"
    ).read_text(encoding="utf-8")
    assert "독립 OOS 또는 배포 증거가 아닙니다" in strategy_note
    assert "Full | pass | 10.000% | 1.200 | 3.000% | 12" in strategy_note
    assert "## 전략 직접 근거" in strategy_note
    assert "## 전략군 설계 맥락" in strategy_note
    assert f'id: "{MODULE._note_id("strategy", "ExampleStrategy")}"' in strategy_note
    assert 'execution_interface: "event_driven"' in strategy_note
    assert 'runner_kind: "event_backtest_engine"' in strategy_note
    assert "live_execution_supported: false" in strategy_note
    index_note = (stage / "Strategy Research Index.md").read_text(encoding="utf-8")
    assert 'id: "lq-generated-strategy-research-index"' in index_note
    assert "do_not_promote / research_only_no_execution" in index_note
    assert "인터페이스 능력 표시일 뿐" in index_note
    assert MODULE.validate_graph(stage, namespace)["broken_link_count"] == 0


def test_polars_batch_research_note_warns_and_rejects_live_claim(tmp_path: Path) -> None:
    payload = _catalog()
    strategy = payload["strategies"][0]
    strategy.update(
        {
            "execution_interface": "polars_batch",
            "runner_kind": "dedicated_dacapogo_daily_research",
        }
    )
    catalog_path = _write_catalog(tmp_path / "catalog.json", payload)
    stage, _ = MODULE.stage_catalog(
        MODULE._load_catalog(catalog_path),
        catalog_path=catalog_path,
        staging_root=tmp_path / "stage",
        namespace=MODULE._safe_namespace("LuminaQuant/Generated"),
        generated_at="2026-08-12T00:00:00Z",
    )
    note = (stage / "Strategies" / f"{MODULE._note_filename('ExampleStrategy')}.md").read_text(
        encoding="utf-8"
    )
    assert "polars_batch research-only" in note
    assert "이벤트/라이브 실행을 지원하지 않습니다" in note

    strategy["live_execution_supported"] = True
    with pytest.raises(ValueError, match="unsafe live support claim"):
        MODULE._validate_catalog(payload)


def test_repository_dacapogo_diagnostic_is_hash_bound_and_rendered() -> None:
    root = Path(__file__).resolve().parents[1]
    catalog = MODULE._load_catalog(
        root / "docs" / "research_note" / "evidence" / "strategy_catalog_20260812.json"
    )
    strategy = next(
        row for row in catalog["strategies"] if row["strategy"] == "DacapogoDailySourceStrategy"
    )

    note = MODULE._strategy_note(
        strategy,
        namespace=MODULE._safe_namespace("LuminaQuant/Generated"),
        generated_at=catalog["generated_at_utc"],
    )

    assert "## 전용 연구 진단" in note
    assert "412a3dc92ed98cb1ba06704d76efd411c76a0e8e8f1eb7967e098a94851c70bf" in note
    assert "legacy entry last return | 9.065% | -2.514%" in note
    assert "daily v2 ml return | 0.367% | 0.559%" in note
    assert "선택·승격·실행에는 사용할 수 없습니다" in note


def test_stage_fails_closed_on_unknown_evidence_link(tmp_path: Path) -> None:
    payload = _catalog()
    payload["strategies"][0]["evidence_ids"] = ["missing-paper"]
    catalog_path = _write_catalog(tmp_path / "catalog.json", payload)

    with pytest.raises(ValueError, match="unknown evidence"):
        MODULE.stage_catalog(
            MODULE._load_catalog(catalog_path),
            catalog_path=catalog_path,
            staging_root=tmp_path / "stage",
            namespace=MODULE._safe_namespace("LuminaQuant/Generated"),
            generated_at="2026-08-12T00:00:00Z",
        )


def test_stage_disambiguates_cross_platform_filename_collision(tmp_path: Path) -> None:
    payload = _catalog()
    duplicate = dict(payload["evidence_sources"][0])
    duplicate["id"] = "paper:1"
    payload["evidence_sources"][0]["id"] = "paper/1"
    payload["evidence_sources"].append(duplicate)
    payload["strategies"][0]["evidence_ids"] = []
    payload["strategies"][0]["family_context_evidence_ids"] = []
    payload["families"][0]["evidence_ids"] = []
    catalog_path = _write_catalog(tmp_path / "catalog.json", payload)

    stage, _ = MODULE.stage_catalog(
        MODULE._load_catalog(catalog_path),
        catalog_path=catalog_path,
        staging_root=tmp_path / "stage",
        namespace=MODULE._safe_namespace("LuminaQuant/Generated"),
        generated_at="2026-08-12T00:00:00Z",
    )
    filenames = [path.name.casefold() for path in (stage / "Evidence").glob("*.md")]
    assert len(filenames) == len(set(filenames)) == 2


def test_apply_replaces_only_marked_namespace_and_keeps_backup(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.json")
    namespace = MODULE._safe_namespace("LuminaQuant/Generated")
    stage, _ = MODULE.stage_catalog(
        MODULE._load_catalog(catalog_path),
        catalog_path=catalog_path,
        staging_root=tmp_path / "stage",
        namespace=namespace,
        generated_at="2026-08-12T00:00:00Z",
    )
    vault = tmp_path / "vault"
    vault.mkdir()
    user_note = vault / "My Note.md"
    user_note.write_text("user work", encoding="utf-8")

    first = MODULE.apply_staged_namespace(
        stage,
        vault_root=vault,
        namespace=namespace,
        generated_at="2026-08-12T00:00:00Z",
    )
    assert first["backup"] is None
    assert user_note.read_text(encoding="utf-8") == "user work"

    second = MODULE.apply_staged_namespace(
        stage,
        vault_root=vault,
        namespace=namespace,
        generated_at="2026-08-12T00:01:00Z",
    )
    assert second["backup"] is not None
    assert Path(second["backup"]).is_dir()
    assert user_note.read_text(encoding="utf-8") == "user work"


def test_apply_rejects_tampered_stage_and_preserves_vault(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.json")
    namespace = MODULE._safe_namespace("LuminaQuant/Generated")
    stage, _ = MODULE.stage_catalog(
        MODULE._load_catalog(catalog_path),
        catalog_path=catalog_path,
        staging_root=tmp_path / "stage",
        namespace=namespace,
        generated_at="2026-08-12T00:00:00Z",
    )
    (stage / "Strategies" / f"{MODULE._note_filename('ExampleStrategy')}.md").write_text(
        "tampered", encoding="utf-8"
    )
    vault = tmp_path / "vault"
    vault.mkdir()
    user_note = vault / "My Note.md"
    user_note.write_text("user work", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        MODULE.apply_staged_namespace(
            stage,
            vault_root=vault,
            namespace=namespace,
            generated_at="2026-08-12T00:00:00Z",
        )

    assert user_note.read_text(encoding="utf-8") == "user work"
    assert not (vault / "LuminaQuant" / "Generated").exists()


def test_generated_tree_rejects_undeclared_nested_manifest(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.json")
    namespace = MODULE._safe_namespace("LuminaQuant/Generated")
    stage, _ = MODULE.stage_catalog(
        MODULE._load_catalog(catalog_path),
        catalog_path=catalog_path,
        staging_root=tmp_path / "stage",
        namespace=namespace,
        generated_at="2026-08-12T00:00:00Z",
    )
    nested_marker = stage / "Strategies" / "_generated_manifest.json"
    nested_marker.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="modified, missing, or extra files"):
        MODULE._verify_generated_tree(stage, namespace)


def test_apply_rejects_stage_replaced_after_expected_identity_was_captured(
    tmp_path: Path,
) -> None:
    namespace = MODULE._safe_namespace("LuminaQuant/Generated")
    first_path = _write_catalog(tmp_path / "first.json")
    stage, _ = MODULE.stage_catalog(
        MODULE._load_catalog(first_path),
        catalog_path=first_path,
        staging_root=tmp_path / "stage",
        namespace=namespace,
        generated_at="2026-08-12T00:00:00Z",
    )
    expected = MODULE._content_identity(stage / "_generated_manifest.json")

    replacement = _catalog()
    replacement["generated_at_utc"] = "2026-08-12T00:01:00Z"
    second_path = _write_catalog(tmp_path / "second.json", replacement)
    replaced_stage, _ = MODULE.stage_catalog(
        MODULE._load_catalog(second_path),
        catalog_path=second_path,
        staging_root=tmp_path / "stage",
        namespace=namespace,
        generated_at="2026-08-12T00:01:00Z",
    )
    assert replaced_stage == stage

    vault = tmp_path / "vault"
    vault.mkdir()
    with pytest.raises(ValueError, match="changed before apply"):
        MODULE.apply_staged_namespace(
            stage,
            vault_root=vault,
            namespace=namespace,
            generated_at="2026-08-12T00:02:00Z",
            expected_stage_manifest=expected,
        )
    assert not (vault / "LuminaQuant" / "Generated").exists()


def test_apply_rejects_user_edits_inside_generated_namespace(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.json")
    namespace = MODULE._safe_namespace("LuminaQuant/Generated")
    stage, _ = MODULE.stage_catalog(
        MODULE._load_catalog(catalog_path),
        catalog_path=catalog_path,
        staging_root=tmp_path / "stage",
        namespace=namespace,
        generated_at="2026-08-12T00:00:00Z",
    )
    vault = tmp_path / "vault"
    vault.mkdir()
    MODULE.apply_staged_namespace(
        stage,
        vault_root=vault,
        namespace=namespace,
        generated_at="2026-08-12T00:00:00Z",
    )
    manual = vault / "LuminaQuant" / "Generated" / "Manual annotation.md"
    manual.write_text("preserve me", encoding="utf-8")

    with pytest.raises(ValueError, match="extra files"):
        MODULE.apply_staged_namespace(
            stage,
            vault_root=vault,
            namespace=namespace,
            generated_at="2026-08-12T00:01:00Z",
        )

    assert manual.read_text(encoding="utf-8") == "preserve me"


def test_apply_rejects_staging_vault_overlap(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.json")
    namespace = MODULE._safe_namespace("LuminaQuant/Generated")
    vault = tmp_path / "vault"
    vault.mkdir()
    stage, _ = MODULE.stage_catalog(
        MODULE._load_catalog(catalog_path),
        catalog_path=catalog_path,
        staging_root=vault / "staging",
        namespace=namespace,
        generated_at="2026-08-12T00:00:00Z",
    )

    with pytest.raises(ValueError, match="must not overlap"):
        MODULE.apply_staged_namespace(
            stage,
            vault_root=vault,
            namespace=namespace,
            generated_at="2026-08-12T00:00:00Z",
        )


def test_apply_rejects_intermediate_symlink(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.json")
    namespace = MODULE._safe_namespace("LuminaQuant/Generated")
    stage, _ = MODULE.stage_catalog(
        MODULE._load_catalog(catalog_path),
        catalog_path=catalog_path,
        staging_root=tmp_path / "stage",
        namespace=namespace,
        generated_at="2026-08-12T00:00:00Z",
    )
    vault = tmp_path / "vault"
    outside = tmp_path / "outside"
    vault.mkdir()
    outside.mkdir()
    (vault / "LuminaQuant").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked path component"):
        MODULE.apply_staged_namespace(
            stage,
            vault_root=vault,
            namespace=namespace,
            generated_at="2026-08-12T00:00:00Z",
        )
    assert list(outside.iterdir()) == []


def test_apply_rejects_older_catalog(tmp_path: Path) -> None:
    namespace = MODULE._safe_namespace("LuminaQuant/Generated")
    newer_payload = _catalog()
    newer_payload["generated_at_utc"] = "2026-08-12T00:01:00Z"
    newer_catalog = _write_catalog(tmp_path / "newer.json", newer_payload)
    newer_stage, _ = MODULE.stage_catalog(
        MODULE._load_catalog(newer_catalog),
        catalog_path=newer_catalog,
        staging_root=tmp_path / "newer-stage",
        namespace=namespace,
        generated_at="2026-08-12T00:01:00Z",
    )
    vault = tmp_path / "vault"
    vault.mkdir()
    MODULE.apply_staged_namespace(
        newer_stage,
        vault_root=vault,
        namespace=namespace,
        generated_at="2026-08-12T00:01:00Z",
    )

    older_catalog = _write_catalog(tmp_path / "older.json")
    older_stage, _ = MODULE.stage_catalog(
        MODULE._load_catalog(older_catalog),
        catalog_path=older_catalog,
        staging_root=tmp_path / "older-stage",
        namespace=namespace,
        generated_at="2026-08-12T00:00:00Z",
    )
    with pytest.raises(ValueError, match="older catalog"):
        MODULE.apply_staged_namespace(
            older_stage,
            vault_root=vault,
            namespace=namespace,
            generated_at="2026-08-12T00:02:00Z",
        )


def test_stale_guard_allows_recovery_from_legacy_future_pin() -> None:
    MODULE._assert_not_stale(
        {
            "catalog_generated_at_utc": "2999-01-01T00:00:00Z",
            "source_catalog": {"sha256": "future-pin"},
        },
        {
            "catalog_generated_at_utc": "2026-08-12T00:00:00Z",
            "source_catalog": {"sha256": "sane"},
        },
    )


def test_receipt_is_published_before_next_export_can_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    namespace = MODULE._safe_namespace("LuminaQuant/Generated")
    first_path = _write_catalog(tmp_path / "first.json")
    first_stage, _ = MODULE.stage_catalog(
        MODULE._load_catalog(first_path),
        catalog_path=first_path,
        staging_root=tmp_path / "first-stage",
        namespace=namespace,
        generated_at="2026-08-12T00:00:00Z",
    )
    second_payload = _catalog()
    second_payload["generated_at_utc"] = "2026-08-12T00:01:00Z"
    second_path = _write_catalog(tmp_path / "second.json", second_payload)
    second_stage, _ = MODULE.stage_catalog(
        MODULE._load_catalog(second_path),
        catalog_path=second_path,
        staging_root=tmp_path / "second-stage",
        namespace=namespace,
        generated_at="2026-08-12T00:01:00Z",
    )
    first_identity = MODULE._content_identity(first_stage / "_generated_manifest.json")
    second_identity = MODULE._content_identity(second_stage / "_generated_manifest.json")
    vault = tmp_path / "vault"
    vault.mkdir()
    receipt = tmp_path / "receipt.json"

    first_receipt_started = threading.Event()
    release_first_receipt = threading.Event()
    second_entered_lock = threading.Event()
    original_write = MODULE._atomic_write_json
    original_verify = MODULE._verify_generated_tree

    def coordinated_write(path: Path, payload: dict) -> None:
        if path == receipt and threading.current_thread().name == "first-export":
            first_receipt_started.set()
            if not release_first_receipt.wait(timeout=5):
                raise TimeoutError("test did not release first receipt")
        original_write(path, payload)

    def coordinated_verify(path: Path, target_namespace: PurePosixPath) -> dict:
        if path == second_stage and threading.current_thread().name == "second-export":
            second_entered_lock.set()
        return original_verify(path, target_namespace)

    monkeypatch.setattr(MODULE, "_atomic_write_json", coordinated_write)
    monkeypatch.setattr(MODULE, "_verify_generated_tree", coordinated_verify)
    errors: list[BaseException] = []

    def publish(
        stage: Path,
        *,
        generated_at: str,
        expected: dict,
        writer: str,
    ) -> None:
        try:
            MODULE.apply_staged_namespace(
                stage,
                vault_root=vault,
                namespace=namespace,
                generated_at=generated_at,
                expected_stage_manifest=expected,
                receipt_path=receipt,
                receipt_payload={"writer": writer},
            )
        except BaseException as exc:  # pragma: no cover - surfaced by assertion below
            errors.append(exc)

    first = threading.Thread(
        target=publish,
        name="first-export",
        args=(first_stage,),
        kwargs={
            "generated_at": "2026-08-12T00:00:00Z",
            "expected": first_identity,
            "writer": "first",
        },
    )
    second = threading.Thread(
        target=publish,
        name="second-export",
        args=(second_stage,),
        kwargs={
            "generated_at": "2026-08-12T00:01:00Z",
            "expected": second_identity,
            "writer": "second",
        },
    )
    first.start()
    assert first_receipt_started.wait(timeout=5)
    second.start()
    try:
        assert not second_entered_lock.wait(timeout=0.2)
    finally:
        release_first_receipt.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    installed = vault / "LuminaQuant" / "Generated" / "_generated_manifest.json"
    final_receipt = json.loads(receipt.read_text(encoding="utf-8"))
    assert final_receipt["writer"] == "second"
    assert final_receipt["installed_manifest"] == MODULE._content_identity(installed)
    assert final_receipt["installed_manifest"] == second_identity


def test_full_export_serializes_same_staging_namespace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    namespace = MODULE._safe_namespace("LuminaQuant/Generated")
    first_path = _write_catalog(tmp_path / "first.json")
    second_payload = _catalog()
    second_payload["generated_at_utc"] = "2026-08-12T00:01:00Z"
    second_path = _write_catalog(tmp_path / "second.json", second_payload)
    staging_root = tmp_path / "stage"
    vault = tmp_path / "vault"
    vault.mkdir()

    first_entered_stage = threading.Event()
    release_first_stage = threading.Event()
    second_entered_stage = threading.Event()
    original_stage = MODULE.stage_catalog

    def coordinated_stage(*args, **kwargs):
        if threading.current_thread().name == "first-export":
            first_entered_stage.set()
            if not release_first_stage.wait(timeout=5):
                raise TimeoutError("test did not release first stage")
        elif threading.current_thread().name == "second-export":
            second_entered_stage.set()
        return original_stage(*args, **kwargs)

    monkeypatch.setattr(MODULE, "stage_catalog", coordinated_stage)
    errors: list[BaseException] = []

    def publish(path: Path) -> None:
        try:
            MODULE.export_catalog_snapshot(
                catalog_path=path,
                staging_root=staging_root,
                namespace=namespace,
                vault_root=vault,
                apply=True,
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    first = threading.Thread(target=publish, name="first-export", args=(first_path,))
    second = threading.Thread(target=publish, name="second-export", args=(second_path,))
    first.start()
    assert first_entered_stage.wait(timeout=5)
    second.start()
    try:
        assert not second_entered_stage.wait(timeout=0.2)
    finally:
        release_first_stage.set()
    first.join(timeout=10)
    second.join(timeout=10)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    receipt = json.loads(
        (staging_root / "obsidian_export_receipt.json").read_text(encoding="utf-8")
    )
    attempt = json.loads(
        (staging_root / "obsidian_export_attempt_latest.json").read_text(encoding="utf-8")
    )
    assert receipt["source_catalog"] == MODULE._content_identity(second_path)
    assert attempt["status"] == "applied"
    installed = vault / "LuminaQuant" / "Generated" / "_generated_manifest.json"
    assert receipt["installed_manifest"] == MODULE._content_identity(installed)


def test_failed_apply_records_distinct_current_attempt(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.json")
    staging_root = tmp_path / "stage"

    with pytest.raises(FileNotFoundError, match="vault root not found"):
        MODULE.export_catalog_snapshot(
            catalog_path=catalog_path,
            staging_root=staging_root,
            namespace=MODULE._safe_namespace("LuminaQuant/Generated"),
            vault_root=tmp_path / "missing-vault",
            apply=True,
        )

    attempt = json.loads(
        (staging_root / "obsidian_export_attempt_latest.json").read_text(encoding="utf-8")
    )
    assert attempt["status"] == "failed"
    assert attempt["error_type"] == "FileNotFoundError"
    assert attempt["error_errno"] is None
    assert "vault root not found" in attempt["error_message"]
    assert attempt["requested_vault_root"] == str((tmp_path / "missing-vault").resolve())
    assert attempt["source_catalog"] == MODULE._content_identity(catalog_path)
    assert not (staging_root / "obsidian_export_receipt.json").exists()


def test_failed_apply_records_errno_and_requested_vault(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.json")
    staging_root = tmp_path / "stage"
    vault = tmp_path / "vault"
    vault.mkdir()

    def fail_apply(*args, **kwargs):
        raise OSError(5, "injected mount failure")

    monkeypatch.setattr(MODULE, "apply_staged_namespace", fail_apply)
    with pytest.raises(OSError, match="injected mount failure"):
        MODULE.export_catalog_snapshot(
            catalog_path=catalog_path,
            staging_root=staging_root,
            namespace=MODULE._safe_namespace("LuminaQuant/Generated"),
            vault_root=vault,
            apply=True,
        )

    attempt = json.loads(
        (staging_root / "obsidian_export_attempt_latest.json").read_text(encoding="utf-8")
    )
    assert attempt["status"] == "failed"
    assert attempt["error_type"] == "OSError"
    assert attempt["error_errno"] == 5
    assert "injected mount failure" in attempt["error_message"]
    assert attempt["requested_vault_root"] == str(vault.resolve())


def test_durable_replace_fsyncs_source_and_destination_parents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_parent = tmp_path / "source"
    destination_parent = tmp_path / "destination"
    source_parent.mkdir()
    destination_parent.mkdir()
    source = source_parent / "payload"
    destination = destination_parent / "payload"
    source.write_text("content", encoding="utf-8")
    synced: list[Path] = []
    monkeypatch.setattr(MODULE, "_fsync_directory", synced.append)

    MODULE._durable_replace(source, destination)

    assert destination.read_text(encoding="utf-8") == "content"
    assert synced == [source_parent, destination_parent]


@pytest.mark.parametrize("failure_point", ("receipt", "journal_clear"))
def test_apply_recovers_installed_namespace_and_receipt_after_finalize_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_point: str,
) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.json")
    namespace = MODULE._safe_namespace("LuminaQuant/Generated")
    stage, _ = MODULE.stage_catalog(
        MODULE._load_catalog(catalog_path),
        catalog_path=catalog_path,
        staging_root=tmp_path / "stage",
        namespace=namespace,
        generated_at="2026-08-12T00:00:00Z",
    )
    expected = MODULE._content_identity(stage / "_generated_manifest.json")
    vault = tmp_path / "vault"
    vault.mkdir()
    receipt = tmp_path / "receipt.json"
    original_write = MODULE._atomic_write_json
    original_clear = MODULE._clear_journal
    failed = False

    def fail_receipt_once(path: Path, payload: dict) -> None:
        nonlocal failed
        if not failed and path == receipt:
            failed = True
            raise OSError("injected receipt failure")
        original_write(path, payload)

    def fail_clear_once(path: Path) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected journal clear failure")
        original_clear(path)

    if failure_point == "receipt":
        monkeypatch.setattr(MODULE, "_atomic_write_json", fail_receipt_once)
    else:
        monkeypatch.setattr(MODULE, "_clear_journal", fail_clear_once)
    with pytest.raises(OSError, match="injected"):
        MODULE.apply_staged_namespace(
            stage,
            vault_root=vault,
            namespace=namespace,
            generated_at="2026-08-12T00:00:00Z",
            expected_stage_manifest=expected,
            receipt_path=receipt,
            receipt_payload={"writer": "first"},
        )

    destination = vault / "LuminaQuant" / "Generated"
    journal = vault / "LuminaQuant" / ".Generated.swap-journal.json"
    assert destination.is_dir()
    assert journal.is_file()

    monkeypatch.setattr(MODULE, "_atomic_write_json", original_write)
    monkeypatch.setattr(MODULE, "_clear_journal", original_clear)
    MODULE.apply_staged_namespace(
        stage,
        vault_root=vault,
        namespace=namespace,
        generated_at="2026-08-12T00:01:00Z",
        expected_stage_manifest=expected,
        receipt_path=receipt,
        receipt_payload={"writer": "second"},
    )

    assert not journal.exists()
    final_receipt = json.loads(receipt.read_text(encoding="utf-8"))
    assert final_receipt["writer"] == "second"
    assert final_receipt["installed_manifest"] == MODULE._content_identity(
        destination / "_generated_manifest.json"
    )


def test_apply_refuses_unmarked_existing_namespace(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.json")
    namespace = MODULE._safe_namespace("LuminaQuant/Generated")
    stage, _ = MODULE.stage_catalog(
        MODULE._load_catalog(catalog_path),
        catalog_path=catalog_path,
        staging_root=tmp_path / "stage",
        namespace=namespace,
        generated_at="2026-08-12T00:00:00Z",
    )
    vault = tmp_path / "vault"
    destination = vault / "LuminaQuant" / "Generated"
    destination.mkdir(parents=True)
    (destination / "Manual.md").write_text("manual", encoding="utf-8")

    with pytest.raises(ValueError, match="marker missing or invalid"):
        MODULE.apply_staged_namespace(
            stage,
            vault_root=vault,
            namespace=namespace,
            generated_at="2026-08-12T00:00:00Z",
        )
    assert (destination / "Manual.md").read_text(encoding="utf-8") == "manual"


def test_recovery_rejects_incoming_manifest_not_bound_to_journal(
    tmp_path: Path,
) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.json")
    namespace = MODULE._safe_namespace("LuminaQuant/Generated")
    stage, _ = MODULE.stage_catalog(
        MODULE._load_catalog(catalog_path),
        catalog_path=catalog_path,
        staging_root=tmp_path / "stage",
        namespace=namespace,
        generated_at="2026-08-12T00:00:00Z",
    )
    vault = tmp_path / "vault"
    destination = vault / "LuminaQuant" / "Generated"
    destination.parent.mkdir(parents=True)
    incoming = destination.parent / ".Generated.incoming-recovery"
    shutil.copytree(stage, incoming)
    journal = destination.parent / ".Generated.swap-journal.json"
    journal.write_text(
        json.dumps(
            {
                "destination": str(destination),
                "incoming": str(incoming),
                "backup": None,
                "target_manifest_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incoming digest mismatch"):
        MODULE._recover_apply_journal(
            journal_path=journal,
            destination=destination,
            namespace=namespace,
            vault_root=vault,
        )
    assert not destination.exists()
    assert incoming.exists()


def test_recovery_rejects_symlinked_backup_ancestor(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.json")
    namespace = MODULE._safe_namespace("LuminaQuant/Generated")
    stage, _ = MODULE.stage_catalog(
        MODULE._load_catalog(catalog_path),
        catalog_path=catalog_path,
        staging_root=tmp_path / "stage",
        namespace=namespace,
        generated_at="2026-08-12T00:00:00Z",
    )
    vault = tmp_path / "vault"
    destination = vault / "LuminaQuant" / "Generated"
    destination.parent.mkdir(parents=True)
    incoming = destination.parent / ".Generated.incoming-recovery"
    shutil.copytree(stage, incoming)
    outside = tmp_path / "outside"
    outside.mkdir()
    (vault / ".luminaquant-generated-backups").symlink_to(
        outside,
        target_is_directory=True,
    )
    backup = vault / ".luminaquant-generated-backups" / "backup"
    journal = destination.parent / ".Generated.swap-journal.json"
    journal.write_text(
        json.dumps(
            {
                "destination": str(destination),
                "incoming": str(incoming),
                "backup": str(backup),
                "target_manifest_sha256": MODULE._content_identity(
                    incoming / "_generated_manifest.json"
                )["sha256"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="symlinked path component"):
        MODULE._recover_apply_journal(
            journal_path=journal,
            destination=destination,
            namespace=namespace,
            vault_root=vault,
        )
    assert list(outside.iterdir()) == []
