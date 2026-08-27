from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "research" / "build_strategy_catalog.py"
    spec = importlib.util.spec_from_file_location("build_strategy_catalog_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load build_strategy_catalog.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _source(source_id: str = "paper-1") -> dict[str, str]:
    return {
        "id": source_id,
        "title": "Paper",
        "authors": "Researcher",
        "publication": "Journal",
        "date": "2026",
        "type": "peer_reviewed",
        "grade": "A",
        "url": "https://doi.org/10.0000/example",
        "supports": "A bounded design prior.",
        "limitations": "Not repository performance.",
    }


def _evidence_payload() -> dict:
    return {
        "schema_version": "lumina_quant.strategy_evidence.v1",
        "as_of_date": "2026-08-12",
        "scope": ">=1-minute strategies",
        "principles": ["External results are design priors."],
        "sources": [_source()],
        "families": {
            "trend_momentum": {
                "title": "Trend",
                "thesis": "Design prior only.",
                "research_decision": "test",
                "evidence_ids": ["paper-1"],
            },
            "rebalancing_diversification": {
                "title": "Rebalancing",
                "thesis": "Require an identical-basket control.",
                "research_decision": "matched_control_required",
                "evidence_ids": ["paper-1"],
            },
            "unresolved": {
                "title": "Unresolved",
                "thesis": "Do not infer an economic family.",
                "research_decision": "classify_first",
                "evidence_ids": ["paper-1"],
            },
        },
        "strategy_overrides": {},
        "recommended_experiments": [],
        "hard_rejections": ["Do not transfer external performance."],
    }


def _comparison_contract(strategy_count: int = 1) -> dict:
    return {
        "artifact_kind": "common_period_1m_cold_start_strategy_screen",
        "generated_at_utc": "2026-08-12T00:00:00Z",
        "strategy_count": strategy_count,
        "timeframe": "1m",
        "windows": {
            "full": {
                "start": "2026-07-01T00:00:00Z",
                "end_exclusive": "2026-08-12T00:00:00Z",
                "days": 42,
            },
            "recent": {
                "start": "2026-08-01T00:00:00Z",
                "end_exclusive": "2026-08-12T00:00:00Z",
                "days": 11,
            },
        },
        "selection_provenance": "unsealed",
        "full_window_role": "diagnostic_screen",
        "recent_window_role": "nested_cold_start_sensitivity_not_independent_oos",
        "promotion_use": "forbidden",
        "summary": {"sha256": "summary"},
    }


class OneSecondOnly:
    required_timeframes = ("1s",)


class OneMinuteDecision:
    decision_cadence_seconds = 60


class MixedDependency:
    required_timeframes = ("1s", "1h")
    decision_cadence_seconds = 3600


class FastDecisionOnMinuteData:
    required_timeframes = ("1m",)
    decision_cadence_seconds = 30


class UnknownCadence:
    pass


def test_scope_excludes_sub_minute_only_and_labels_unknown() -> None:
    sub_minute = MODULE.strategy_scope_metadata(OneSecondOnly)
    one_minute = MODULE.strategy_scope_metadata(OneMinuteDecision)
    unknown = MODULE.strategy_scope_metadata(UnknownCadence)

    assert sub_minute["in_scope"] is False
    assert sub_minute["exclusion_reason"] == "requires_sub_minute_timeframe_dependency"
    assert MODULE.strategy_scope_metadata(MixedDependency)["in_scope"] is False
    assert MODULE.strategy_scope_metadata(FastDecisionOnMinuteData)["in_scope"] is False
    assert one_minute["in_scope"] is True
    assert one_minute["cadence_status"] == "explicit_in_scope"
    assert unknown["in_scope"] is True
    assert unknown["cadence_status"] == "unknown"
    assert unknown["scope_status"] == "scope_unverified"
    assert one_minute["scope_status"] == "verified_in_scope"


def test_family_resolution_priority() -> None:
    overrides = {"NamedStrategy": {"family": "evidence_family"}}
    assert MODULE.resolve_family("NamedStrategy", ["mean_reversion"], overrides) == (
        "evidence_family",
        "evidence_override",
    )
    assert MODULE.resolve_family("CandidateStrategy", ["mean_reversion"], {}) == (
        "mean_reversion_relative_value",
        "candidate_library",
    )
    assert MODULE.resolve_family("FallbackBreakoutStrategy", ["trend", "carry"], {}) == (
        "unresolved",
        "unresolved_ambiguous_or_unmapped",
    )


def test_comparison_mapping_preserves_na_and_provenance(tmp_path: Path) -> None:
    path = tmp_path / "comparison.csv"
    fields = [
        "strategy",
        "tier",
        "symbols",
        "full_status",
        "recent_status",
        "full_return",
        "recent_return",
        "full_max_drawdown",
        "recent_max_drawdown",
        "full_sharpe",
        "recent_sharpe",
        "full_trades",
        "recent_trades",
        "full_log_daily",
        "recent_log_daily",
        "comparable",
        "delta_log_daily",
        "daily_gap",
        "robust_log_daily",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "strategy": "ExampleStrategy",
                "tier": "research_only",
                "symbols": "3",
                "full_status": "pass",
                "recent_status": "resource_excluded",
                "full_return": "0.12",
                "recent_return": "",
                "full_max_drawdown": "0.04",
                "recent_max_drawdown": "",
                "full_sharpe": "1.5",
                "recent_sharpe": "",
                "full_trades": "12",
                "recent_trades": "0",
                "full_log_daily": "0.001",
                "recent_log_daily": "",
                "comparable": "false",
                "delta_log_daily": "",
                "daily_gap": "",
                "robust_log_daily": "",
            }
        )

    row = MODULE.load_comparison_rows(path, _comparison_contract())["ExampleStrategy"]

    assert row["full"]["return"] == 0.12
    assert row["recent"]["return"] is None
    assert row["comparable"] is False
    assert row["provenance"]["selection_provenance"] == "unsealed"
    assert row["provenance"]["artifact_kind"] == "common_period_1m_cold_start_strategy_screen"
    assert row["provenance"]["recent_window_role"].endswith("not_independent_oos")


def test_explicit_missing_inputs_fail_closed(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="comparison CSV file not found"):
        MODULE.load_comparison_rows(missing)
    with pytest.raises(FileNotFoundError, match="evidence file not found"):
        MODULE._load_json_object(missing, label="evidence")


def test_comparison_contract_binds_exact_csv_and_methodology(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "strategy_comparison.csv"
    csv_path.write_text("strategy\nExampleStrategy\n", encoding="utf-8")
    data = csv_path.read_bytes()
    summary_path = tmp_path / "common_period_summary.json"
    summary = {
        "artifact_kind": "common_period_1m_cold_start_strategy_screen",
        "generated_at_utc": "2026-08-12T01:02:03Z",
        "strategy_count": 1,
        "windows": {
            "full": {
                "start": "2026-07-01T00:00:00Z",
                "end_exclusive": "2026-08-12T00:00:00Z",
                "days": 42,
            },
            "recent": {
                "start": "2026-08-01T00:00:00Z",
                "end_exclusive": "2026-08-12T00:00:00Z",
                "days": 11,
            },
        },
        "status_counts": {
            "full": {"pass": 1},
            "recent": {"pass": 1},
        },
        "window_provenance": {
            label: {
                "git_commit": "abc123",
                "integrity_policy": {
                    "no_gap_fill": True,
                    "no_interpolation": True,
                    "no_synthetic_rows": True,
                },
            }
            for label in ("full", "recent")
        },
        "methodology": {
            "timeframe": "exact 1m [start,end), no interpolation/gap fill/synthetic rows"
        },
        "limitations": ["Not independent OOS: selection provenance is unsealed."],
        "artifacts": {
            csv_path.name: {
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        },
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    contract = MODULE.load_comparison_contract(summary_path, csv_path)

    assert contract["timeframe"] == "1m"
    assert contract["strategy_count"] == 1
    csv_path.write_text("strategy\nTampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        MODULE.load_comparison_contract(summary_path, csv_path)


def test_evidence_references_fail_closed() -> None:
    payload = _evidence_payload()
    payload["families"]["trend_momentum"]["evidence_ids"] = ["missing"]
    with pytest.raises(ValueError, match="unknown sources"):
        MODULE._normalize_evidence(payload)


def test_evidence_schema_and_required_fields_fail_closed() -> None:
    payload = _evidence_payload()
    payload["schema_version"] = "wrong"
    with pytest.raises(ValueError, match="schema_version"):
        MODULE._normalize_evidence(payload)

    payload = _evidence_payload()
    del payload["sources"][0]["limitations"]
    with pytest.raises(ValueError, match="missing fields"):
        MODULE._normalize_evidence(payload)


def test_repository_evidence_semantically_classifies_every_in_scope_strategy() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = MODULE._normalize_evidence(
        json.loads(
            (root / "docs" / "research_note" / "strategy_evidence_20260812.json").read_text(
                encoding="utf-8"
            )
        )
    )
    catalog = MODULE.build_catalog(
        strategy_map=MODULE.get_catalog_strategy_map(),
        candidate_index=MODULE._candidate_index(),
        comparison_rows={},
        evidence=evidence,
        generated_at="2026-08-12T00:00:00Z",
        comparison_source={"status": "not_supplied"},
        evidence_source={"status": "loaded"},
    )

    assert catalog["counts"]["registry"] == 147
    assert catalog["counts"]["in_scope"] == 146
    assert catalog["counts"]["verified_in_scope"] == 122
    assert catalog["counts"]["scope_unverified"] == 24
    assert catalog["counts"]["excluded"] == 1
    assert catalog["counts"]["unresolved_family"] == 0
    raw_scorecard = catalog["scorecard_summary"]["raw_registry_diagnostic"]
    assert raw_scorecard["strategy_count"] == 147
    assert raw_scorecard["evaluated_strategy_count"] == 0
    assert raw_scorecard["not_evaluated_strategy_count"] == 147
    assert [row["strategy"] for row in catalog["exclusions"]] == ["MicroRangeExpansion1sStrategy"]
    dacapogo = next(
        row for row in catalog["strategies"] if row["strategy"] == "DacapogoDailySourceStrategy"
    )
    assert dacapogo["tier"] == "research_only"
    assert dacapogo["family"] == "breakout"
    assert dacapogo["execution_interface"] == "polars_batch"
    assert dacapogo["runner_kind"] == "dedicated_dacapogo_daily_research"
    assert dacapogo["live_execution_supported"] is False
    assert dacapogo["cadence"]["scope_status"] == "verified_in_scope"
    assert dacapogo["cadence"]["required_timeframes"] == ["1d"]
    assert (
        "none is registered for event-driven, live, or real-money execution"
        in (dacapogo["research_note"])
    )
    diagnostic = dacapogo["dedicated_diagnostic"]
    assert diagnostic["artifact"] == {
        "path": "var/reports/common_period_reval_20260812/dacapogo_common_metrics.json",
        "bytes": 20410,
        "sha256": "412a3dc92ed98cb1ba06704d76efd411c76a0e8e8f1eb7967e098a94851c70bf",
    }
    assert diagnostic["decision"] == {"gate_pass": False, "locked_action": "cash"}


def test_artifacts_are_consistent_and_do_not_claim_independent_oos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class ExampleStrategy:
        __module__ = "lumina_quant.strategies.example_trend"
        decision_cadence_seconds = 3600
        required_features = ("ohlcv",)

    monkeypatch.setattr(MODULE, "get_strategy_tier", lambda _name: "research_only")
    monkeypatch.setattr(MODULE, "get_strategy_param_schema", lambda _name: {"lookback": {}})
    comparison = {
        "ExampleStrategy": {
            "provenance": {
                "artifact_kind": "common_period_1m_cold_start_strategy_screen",
                "selection_provenance": "unsealed",
                "full_window_role": "diagnostic_screen",
                "recent_window_role": "nested_cold_start_sensitivity_not_independent_oos",
            },
            "tier": "research_only",
            "symbols": 2,
            "full": {
                "status": "pass",
                "return": 0.10,
                "max_drawdown": 0.03,
                "sharpe": 1.2,
                "trades": 10,
                "log_daily": 0.001,
            },
            "recent": {
                "status": "pass",
                "return": 0.02,
                "max_drawdown": 0.02,
                "sharpe": 0.8,
                "trades": 3,
                "log_daily": 0.0005,
            },
            "comparable": True,
            "delta_log_daily": -0.0005,
            "daily_gap": -0.0005,
            "robust_log_daily": 0.0005,
        }
    }
    evidence = MODULE._normalize_evidence(_evidence_payload())
    contract = _comparison_contract()
    catalog = MODULE.build_catalog(
        strategy_map={"ExampleStrategy": ExampleStrategy},
        candidate_index={
            "ExampleStrategy": {
                "families": ["trend"],
                "timeframes": ["1h"],
                "candidate_count": 1,
            }
        },
        comparison_rows=comparison,
        evidence=evidence,
        generated_at="2026-08-12T00:00:00Z",
        comparison_source={"status": "loaded", "contract": contract},
        evidence_source={"status": "loaded"},
    )

    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest_first = MODULE.write_artifacts(catalog, first)
    manifest_second = MODULE.write_artifacts(catalog, second)

    assert manifest_first == manifest_second
    assert len(manifest_first["artifacts"]) == 5
    payload = json.loads((first / "strategy_catalog.json").read_text(encoding="utf-8"))
    assert payload["counts"]["in_scope"] == 1
    assert payload["counts"]["verified_in_scope"] == 1
    assert payload["counts"]["scope_unverified"] == 0
    assert payload["families"][0]["comparable_count"] == 1
    csv_row = next(csv.DictReader((first / "strategy_catalog.csv").open(encoding="utf-8")))
    assert csv_row["execution_interface"] == "event_driven"
    assert csv_row["runner_kind"] == "event_backtest_engine"
    assert csv_row["live_execution_supported"] == "False"
    markdown = (first / "strategy_scorecards.md").read_text(encoding="utf-8")
    assert "독립 OOS·배포 증거가 아닙니다" in markdown
    assert "locked OOS performance" not in markdown


def test_family_aggregation_requires_finite_metrics_on_both_windows() -> None:
    row = {
        "strategy": "IncompleteStrategy",
        "family": "trend_momentum",
        "metrics": {
            "comparable": True,
            "full": {"return": 0.1, "sharpe": 1.0, "max_drawdown": 0.03},
            "recent": {"return": 0.02, "sharpe": None, "max_drawdown": 0.02},
        },
    }

    family = MODULE._build_family_rows([row], {})[0]

    assert family["comparable_count"] == 0
    assert family["full_median_return"] is None


def test_verified_scorecard_excludes_cadence_unverified_diagnostics() -> None:
    def row(name: str, scope_status: str) -> dict:
        return {
            "strategy": name,
            "family": "trend_momentum",
            "cadence": {"scope_status": scope_status},
            "metrics": {
                "comparable": True,
                "full": {"return": 0.1, "sharpe": 1.0, "max_drawdown": 0.03},
                "recent": {"return": 0.02, "sharpe": 0.8, "max_drawdown": 0.02},
            },
        }

    verified = row("VerifiedStrategy", "verified_in_scope")
    unverified = row("UnknownCadenceStrategy", "scope_unverified")
    raw_sub_minute = dict(unverified["metrics"])
    summary = MODULE._build_scorecard_summary(
        [verified["metrics"], unverified["metrics"], raw_sub_minute],
        [verified, unverified],
    )
    family = MODULE._build_family_rows([verified, unverified], {})[0]

    assert summary["raw_registry_diagnostic"]["positive_both_count"] == 3
    assert summary["raw_registry_diagnostic"]["evaluated_strategy_count"] == 3
    assert summary["raw_registry_diagnostic"]["not_evaluated_strategy_count"] == 0
    assert summary["catalog_controlled_diagnostic"]["positive_both_count"] == 2
    assert summary["verified_ge_1m_controlled"] == {
        "strategy_count": 1,
        "comparable_count": 1,
        "positive_both_count": 1,
        "scope": "verified_in_scope_only",
        "matched_control_policy": "unmatched_rebalancing_metrics_suppressed",
    }
    assert family["verified_strategy_count"] == 1
    assert family["comparable_count"] == 1
    assert family["catalog_diagnostic_comparable_count"] == 2


def test_rebalancing_metrics_are_suppressed_without_matched_control() -> None:
    metrics = {
        "provenance": {
            "artifact_kind": "common_period_1m_cold_start_strategy_screen",
            "selection_provenance": "unsealed",
        },
        "full": {
            "status": "pass",
            "return": 0.08,
            "max_drawdown": 0.03,
            "sharpe": 2.0,
            "trades": 20,
        },
        "recent": {
            "status": "pass",
            "return": 0.02,
            "max_drawdown": 0.02,
            "sharpe": 1.0,
            "trades": 5,
        },
        "comparable": True,
        "daily_gap": 0.001,
        "delta_log_daily": 0.001,
        "robust_log_daily": 0.001,
    }

    result = MODULE._apply_metric_semantics(
        "rebalancing_diversification",
        metrics,
    )

    assert result["full"]["status"] == "matched_control_missing"
    assert result["full"]["return"] is None
    assert result["recent"]["sharpe"] is None
    assert result["comparable"] is False
    assert result["raw_total_return_diagnostic"]["full"]["return"] == 0.08


def test_candidate_only_definitions_are_reconciled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExampleStrategy:
        decision_cadence_seconds = 60

    monkeypatch.setattr(MODULE, "get_strategy_tier", lambda _name: "research_only")
    monkeypatch.setattr(MODULE, "get_strategy_param_schema", lambda _name: {})
    evidence = MODULE._normalize_evidence(_evidence_payload())
    catalog = MODULE.build_catalog(
        strategy_map={"ExampleStrategy": ExampleStrategy},
        candidate_index={
            "ExampleStrategy": {
                "families": ["trend"],
                "timeframes": ["1m"],
                "candidate_count": 1,
            },
            "CandidateOnlyStrategy": {
                "families": ["event_alpha"],
                "timeframes": ["5m"],
                "candidate_count": 2,
            },
        },
        comparison_rows={},
        evidence=evidence,
        generated_at="2026-08-12T00:00:00Z",
        comparison_source={"status": "not_supplied"},
        evidence_source={"status": "loaded"},
    )

    assert catalog["counts"]["candidate_only_strategy_classes"] == 1
    assert catalog["candidate_only_strategy_classes"][0]["strategy_class"] == (
        "CandidateOnlyStrategy"
    )


def test_param_schema_errors_are_not_silently_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExampleStrategy:
        decision_cadence_seconds = 60

    monkeypatch.setattr(MODULE, "get_strategy_tier", lambda _name: "research_only")

    def fail(_name: str) -> dict:
        raise RuntimeError("broken schema")

    monkeypatch.setattr(MODULE, "get_strategy_param_schema", fail)
    with pytest.raises(RuntimeError, match="broken schema"):
        MODULE.build_catalog(
            strategy_map={"ExampleStrategy": ExampleStrategy},
            candidate_index={},
            comparison_rows={},
            evidence=MODULE._normalize_evidence(_evidence_payload()),
            generated_at="2026-08-12T00:00:00Z",
            comparison_source={"status": "not_supplied"},
            evidence_source={"status": "loaded"},
        )


def test_source_metadata_is_reproducible_across_directories(
    tmp_path: Path,
) -> None:
    first = tmp_path / "a" / "evidence.json"
    second = tmp_path / "b" / "evidence.json"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("same", encoding="utf-8")
    second.write_text("same", encoding="utf-8")

    assert MODULE._source_metadata(
        first,
        supplied=True,
        kind="evidence",
    ) == MODULE._source_metadata(second, supplied=True, kind="evidence")
