from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "search_portfolio_superiority_meta.py"
SPEC = importlib.util.spec_from_file_location("search_portfolio_superiority_meta", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load search_portfolio_superiority_meta module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _stream(values: list[float], *, start_day: int = 1) -> list[dict[str, Any]]:
    return [
        {
            "datetime": f"2026-02-{start_day + idx:02d}T00:00:00Z",
            "t": 1_700_000_000_000 + (idx * 86_400_000),
            "v": value,
        }
        for idx, value in enumerate(values)
    ]


def _candidate(
    candidate_key: str,
    *,
    lineage: str | None = None,
    train: list[float] | None = None,
    val: list[float] | None = None,
    oos: list[float] | None = None,
    stream_key: str = "portfolio_return_streams",
) -> dict[str, Any]:
    payload = {
        "artifact_kind": "portfolio_candidate_artifact",
        stream_key: {
            "train": _stream(train or [0.01, 0.01, 0.01], start_day=1),
            "val": _stream(val or [0.01, 0.01, 0.01], start_day=10),
            "oos": _stream(oos or [0.03, 0.02], start_day=20),
        },
    }
    out = {"candidate_key": candidate_key, "label": candidate_key, "payload": payload}
    if lineage is not None:
        out["lineage"] = lineage
    return out


class DummyGuard:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rss_log_path = self.output_dir / "dummy_rss_latest.jsonl"
        self.rss_log_path.write_text("", encoding="utf-8")
        self.summary_path = self.output_dir / "dummy_memory_latest.json"

    def checkpoint(self, event: str, context: dict[str, Any] | None = None) -> None:
        self.rss_log_path.write_text(
            self.rss_log_path.read_text(encoding="utf-8")
            + json.dumps({"event": event, "context": context or {}})
            + "\n",
            encoding="utf-8",
        )

    def sample(self, *, event: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        self.checkpoint(event, context)
        return {"event": event, "context": context or {}}

    def finalize(
        self,
        *,
        status: str,
        error: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "status": status,
            "error": error,
            "context": context or {},
            "summary_path": str(self.summary_path.resolve()),
            "memory_policy": {"heavy_lock_path": "/tmp/test.lock", "explicit_budget_bytes": 123},
        }
        self.summary_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def release(self) -> None:
        return None


def test_build_basis_search_universes_separates_raw_and_derived() -> None:
    universes = MODULE.build_basis_search_universes(
        incumbent=_candidate("incumbent"),
        raw_55_45=_candidate("autoresearch_pair_55_45"),
        derived_80_20=_candidate("incumbent_autoresearch_static_blend"),
        soft_allocator=_candidate("soft_allocator"),
        regime_switch=_candidate("regime_switching_portfolio"),
        grouped_base=_candidate("grouped_base"),
    )

    raw_names = {row["candidate_key"] for row in universes[MODULE.RAW_UNIVERSE_NAME]}
    derived_names = {row["candidate_key"] for row in universes[MODULE.DERIVED_UNIVERSE_NAME]}
    assert "autoresearch_pair_55_45" in raw_names
    assert "incumbent_autoresearch_static_blend" not in raw_names
    assert "incumbent_autoresearch_static_blend" in derived_names
    assert "autoresearch_pair_55_45" not in derived_names


def test_ensure_basis_dedupe_rejects_raw_and_derived_mix() -> None:
    with pytest.raises(MODULE.BasisUniverseError, match="cannot coexist"):
        MODULE.ensure_basis_dedupe(
            [
                _candidate("incumbent"),
                _candidate("autoresearch_pair_55_45"),
                _candidate("incumbent_autoresearch_static_blend"),
            ]
        )


def test_robustness_gate_failures_cover_thresholds() -> None:
    failures = MODULE.robustness_gate_failures(
        candidate_metrics={
            "train": {"total_return": -0.01, "sharpe": -0.2},
            "val": {"total_return": 0.0},
            "oos": {"total_return": 0.01, "sharpe": 1.4, "max_drawdown": 0.11},
        },
        incumbent_oos={"total_return": 0.02, "sharpe": 1.0, "max_drawdown": 0.05},
        monthly_returns=[{"month": "2026-02", "total_return": 0.01, "days": 20}],
    )

    assert failures == [
        "train_total_return<=0",
        "val_total_return<=0",
        "train_sharpe<-0.10",
        "oos_total_return_delta<=0",
        "oos_monthly_mean<0.02",
        "oos_drawdown_worse_without_sharpe_relief",
    ]


def test_robustness_gate_allows_sharpe_relief_with_worse_drawdown() -> None:
    failures = MODULE.robustness_gate_failures(
        candidate_metrics={
            "train": {"total_return": 0.05, "sharpe": 0.2},
            "val": {"total_return": 0.04},
            "oos": {"total_return": 0.08, "sharpe": 1.8, "max_drawdown": 0.08},
        },
        incumbent_oos={"total_return": 0.02, "sharpe": 1.2, "max_drawdown": 0.05},
        monthly_returns=[{"month": "2026-02", "total_return": 0.03, "days": 20}],
    )

    assert failures == []


def test_normalize_candidate_accepts_direct_return_streams_payload() -> None:
    normalized = MODULE.normalize_candidate(
        _candidate(
            "pair_tactical_mode",
            lineage=MODULE.NEUTRAL_LINEAGE,
            stream_key="return_streams",
            train=[0.002, 0.001],
            val=[0.003, 0.001],
            oos=[0.004, 0.002],
        )
    )

    assert normalized["candidate_key"] == "pair_tactical_mode"
    assert normalized["portfolio_return_streams"]["oos"]
    assert normalized["portfolio_metrics"]["oos"]["total_return"] > 0.0
    assert normalized["participation"]["oos"]["active_days"] == 2
    assert normalized["participation"]["oos"]["total_days"] == 2
    assert normalized["participation"]["oos"]["active_day_ratio"] == 1.0


def test_sparse_component_gate_fails_when_sparse_candidate_exceeds_cap() -> None:
    dense = MODULE.normalize_candidate(
        _candidate(
            "dense_core",
            lineage=MODULE.NEUTRAL_LINEAGE,
            oos=[0.01, 0.01, 0.01, 0.01],
        )
    )
    sparse = MODULE.normalize_candidate(
        _candidate(
            "sparse_pair",
            lineage=MODULE.NEUTRAL_LINEAGE,
            oos=[0.02] + ([0.0] * 19),
        )
    )

    reasons, diagnostics = MODULE.sparse_component_gate_failures(
        candidates=[dense, sparse],
        weights=[0.75, 0.25],
    )

    assert reasons == [
        f"sparse_component_weight_above_cap:sparse_pair:0.250>{MODULE.SPARSE_COMPONENT_WEIGHT_CAP:.3f}"
    ]
    sparse_diag = next(row for row in diagnostics if row["candidate_key"] == "sparse_pair")
    assert sparse_diag["is_sparse"] is True
    assert sparse_diag["oos_active_day_ratio"] == 0.05


def test_sparse_component_gate_also_flags_too_few_active_days() -> None:
    dense = MODULE.normalize_candidate(
        _candidate(
            "dense_core",
            lineage=MODULE.NEUTRAL_LINEAGE,
            oos=[0.01, 0.01, 0.01, 0.01],
        )
    )
    tiny_tail = MODULE.normalize_candidate(
        _candidate(
            "tiny_tail",
            lineage=MODULE.NEUTRAL_LINEAGE,
            oos=[0.03],
        )
    )

    reasons, diagnostics = MODULE.sparse_component_gate_failures(
        candidates=[dense, tiny_tail],
        weights=[0.75, 0.25],
    )

    assert reasons == [
        f"sparse_component_weight_above_cap:tiny_tail:0.250>{MODULE.SPARSE_COMPONENT_WEIGHT_CAP:.3f}"
    ]
    tiny_diag = next(row for row in diagnostics if row["candidate_key"] == "tiny_tail")
    assert tiny_diag["is_sparse"] is True
    assert tiny_diag["sparse_reason"] == "active_days"


def test_evaluate_weight_combo_carries_sparse_component_rejection_reason() -> None:
    incumbent = MODULE.normalize_candidate(
        _candidate(
            "incumbent",
            lineage=MODULE.NEUTRAL_LINEAGE,
            oos=[0.01, 0.01, 0.01, 0.01],
        )
    )
    sparse = MODULE.normalize_candidate(
        _candidate(
            "sparse_pair",
            lineage=MODULE.NEUTRAL_LINEAGE,
            train=[0.01, 0.01, 0.01],
            val=[0.01, 0.01, 0.01],
            oos=[0.03] + ([0.0] * 19),
        )
    )

    result = MODULE.evaluate_weight_combo(
        candidates=[incumbent, sparse],
        weights=[0.75, 0.25],
        incumbent_oos=dict((incumbent.get("portfolio_metrics") or {}).get("oos") or {}),
    )

    assert result["promotable"] is False
    assert any(
        reason.startswith("sparse_component_weight_above_cap:sparse_pair")
        for reason in result["rejection_reasons"]
    )
    sparse_diag = next(
        row for row in result["component_participation"] if row["candidate_key"] == "sparse_pair"
    )
    assert sparse_diag["is_sparse"] is True


def test_normalize_candidate_marks_realistic_single_day_pair_as_sparse() -> None:
    pair = MODULE.normalize_candidate(
        {
            "candidate_key": "pair_tactical_mode",
            "label": "Pair tactical fast exit",
            "payload": {
                "artifact_kind": "pair_tactical_candidate_payload",
                "return_streams": {
                    "train": [{"datetime": "2025-01-01T00:00:00Z", "v": 0.01}],
                    "val": [{"datetime": "2026-01-01T00:00:00Z", "v": 0.01}],
                    "oos": [{"datetime": "2026-03-01T00:00:00Z", "v": 0.03}],
                },
                "portfolio_metrics": {
                    "train": {"total_return": 0.01, "sharpe": 1.0, "max_drawdown": 0.0},
                    "val": {"total_return": 0.01, "sharpe": 1.0, "max_drawdown": 0.0},
                    "oos": {"total_return": 0.03, "sharpe": 3.0, "max_drawdown": 0.0},
                },
            },
        }
    )

    reasons, diagnostics = MODULE.sparse_component_gate_failures(
        candidates=[pair],
        weights=[0.25],
    )

    assert reasons == [
        f"sparse_component_weight_above_cap:pair_tactical_mode:0.250>{MODULE.SPARSE_COMPONENT_WEIGHT_CAP:.3f}"
    ]
    assert diagnostics[0]["oos_active_days"] == 1
    assert diagnostics[0]["is_sparse"] is True


def test_run_meta_search_writes_artifacts_and_memory_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        MODULE,
        "acquire_portfolio_memory_guard",
        lambda **kwargs: DummyGuard(Path(kwargs["output_dir"])),
    )

    candidates = MODULE.build_basis_search_universes(
        incumbent=_candidate("incumbent", oos=[0.01, 0.01, 0.01]),
        raw_55_45=_candidate("autoresearch_pair_55_45", oos=[0.04, 0.03, 0.02]),
        derived_80_20=_candidate("incumbent_autoresearch_static_blend", oos=[0.03, 0.03, 0.02]),
        soft_allocator=_candidate(
            "soft_allocator", train=[-0.01, -0.01], val=[-0.01, -0.01], oos=[0.0, 0.0, 0.0]
        ),
        regime_switch=_candidate("regime_switching_portfolio", oos=[0.015, 0.01, 0.01]),
        grouped_base=_candidate("grouped_base", oos=[0.012, 0.011, 0.01]),
    )[MODULE.RAW_UNIVERSE_NAME]

    result = MODULE.run_meta_search(
        universe_name=MODULE.RAW_UNIVERSE_NAME,
        candidates=candidates,
        incumbent_key="incumbent",
        output_dir=tmp_path,
        weight_step=0.5,
        top_k=10,
    )

    assert Path(result["leaderboard_json_path"]).exists()
    assert Path(result["rejections_json_path"]).exists()
    assert Path(result["memory_ledger_path"]).exists()
    summary = json.loads(Path(result["summary_json_path"]).read_text(encoding="utf-8"))
    ledger = json.loads(Path(result["memory_ledger_path"]).read_text(encoding="utf-8"))
    assert summary["universe_name"] == MODULE.RAW_UNIVERSE_NAME
    assert summary["winner_status"] == "promoted_challenger"
    assert summary["winner"]["promotable"] is True
    assert ledger["combination_count"] == summary["combination_count"]
    assert ledger["candidate_count"] == len(candidates)
    assert any(row["lineage"] == MODULE.RAW_55_45_LINEAGE for row in summary["candidate_universe"])
