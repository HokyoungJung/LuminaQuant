from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from lumina_quant import portfolio_split_contract as contract

ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {path.name} module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SEARCH_MODULE_PATH = ROOT / "scripts" / "research" / "search_portfolio_four_sleeve_anchored.py"
SEARCH_MODULE = None
if SEARCH_MODULE_PATH.exists():
    SEARCH_MODULE = _load_script_module(
        "search_portfolio_four_sleeve_anchored_contract",
        SEARCH_MODULE_PATH,
    )

DYNAMIC_MODULE_PATH = ROOT / "scripts" / "research" / "run_causal_dynamic_portfolio.py"
DYNAMIC_MODULE = None
if DYNAMIC_MODULE_PATH.exists():
    DYNAMIC_MODULE = _load_script_module(
        "run_causal_dynamic_portfolio_contract",
        DYNAMIC_MODULE_PATH,
    )

OVERLAY_MODULE_PATH = ROOT / "scripts" / "research" / "run_causal_overlay_portfolio.py"
OVERLAY_MODULE = None
if OVERLAY_MODULE_PATH.exists():
    OVERLAY_MODULE = _load_script_module(
        "run_causal_overlay_portfolio_contract",
        OVERLAY_MODULE_PATH,
    )


class _DummyLock:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


class _DummyRSSGuard:
    def __init__(self, **kwargs: Any) -> None:
        self.log_path = Path(kwargs["log_path"])
        self.label = str(kwargs["label"])
        self.budget_bytes = int(kwargs["budget_bytes"])

    def finalize(self, *, status: str, error: str | None = None) -> dict[str, Any]:
        return {
            "status": status,
            "error": error,
            "peak_rss_bytes": 123,
            "soft_limit_bytes": 456,
            "hard_limit_bytes": 789,
            "rss_log_path": str(self.log_path),
        }


class _CrashGuard:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.finalize_calls: list[dict[str, Any]] = []
        self.released = False

    def sample(self, *, event: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"event": event, "context": context or {}}

    def checkpoint(self, event: str, context: dict[str, Any] | None = None) -> None:
        return None

    def finalize(
        self,
        *,
        status: str,
        error: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {"status": status, "error": error, "context": context or {}}
        self.finalize_calls.append(payload)
        summary_path = (
            self.output_dir / "_memory_guard" / "portfolio_four_sleeve_search_memory_latest.json"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def release(self) -> None:
        self.released = True


class _ArtifactGuard:
    def __init__(self, *, run_name: str, output_dir: Path, budget_bytes: int | None) -> None:
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.budget_bytes = budget_bytes
        self.released = False

    def sample(self, *, event: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"event": event, "context": context or {}}

    def checkpoint(self, event: str, context: dict[str, Any] | None = None) -> None:
        return None

    def finalize(
        self,
        *,
        status: str,
        error: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        memory_dir = self.output_dir / "_memory_guard"
        memory_dir.mkdir(parents=True, exist_ok=True)
        rss_log_path = memory_dir / f"{self.run_name}_rss_latest.jsonl"
        rss_log_path.write_text("", encoding="utf-8")
        payload = {
            "artifact_kind": "portfolio_followup_memory_summary",
            "run_name": self.run_name,
            "label": f"portfolio_followup::{self.run_name}",
            "memory_policy": contract.memory_policy_payload(budget_bytes=self.budget_bytes),
            "context": dict(context or {}),
            "status": status,
            "error": error,
            "rss_log_path": str(rss_log_path),
        }
        summary_path = memory_dir / f"{self.run_name}_memory_latest.json"
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    def release(self) -> None:
        self.released = True


def _allocator_result() -> dict[str, Any]:
    return {
        "split_metrics": {
            "train": {"total_return": 0.02, "sharpe": 1.0},
            "val": {"total_return": 0.03, "sharpe": 1.2},
            "oos": {"total_return": 0.01, "sharpe": 0.8},
        },
        "all_metrics": {},
        "allocations": [
            {
                "date": "2026-02-02",
                "weights": {"c1": 0.7, "c2": 0.3},
                "cash_weight": 0.0,
            }
        ],
        "meta": {
            "c1": {"name": "candidate_1", "strategy_class": "StubStrategy", "timeframe": "1d"},
            "c2": {"name": "candidate_2", "strategy_class": "StubStrategy", "timeframe": "1d"},
        },
    }


def _observed_budget_evidence(
    *,
    captured: dict[str, Any],
    result: dict[str, Any],
    summary_path: Path,
) -> dict[str, Any]:
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload = dict(result.get("payload") or {})
    memory_summary = dict(payload.get("memory_summary") or {})
    return {
        "captured_budget_bytes": captured.get("budget_bytes"),
        "payload_memory_policy_budget_bytes": dict(payload.get("memory_policy") or {}).get(
            "explicit_budget_bytes"
        ),
        "memory_summary_budget_bytes": dict(memory_summary.get("memory_policy") or {}).get(
            "explicit_budget_bytes"
        ),
        "summary_file_budget_bytes": dict(summary_payload.get("memory_policy") or {}).get(
            "explicit_budget_bytes"
        ),
    }


def test_portfolio_memory_guard_finalize_writes_memory_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy_lock = _DummyLock()
    monkeypatch.setattr(
        contract,
        "HeavyRunLock",
        type(
            "DummyHeavyRunLock",
            (),
            {"acquire": staticmethod(lambda **kwargs: dummy_lock)},
        ),
    )
    monkeypatch.setattr(contract, "acquire_session_memory_lease", lambda **kwargs: _DummyLock())
    monkeypatch.setattr(contract, "RSSGuard", _DummyRSSGuard)

    budget_bytes = 8 * 1024 * 1024 * 1024
    guard = contract.acquire_portfolio_memory_guard(
        run_name="portfolio_four_sleeve_search",
        output_dir=tmp_path / "search",
        input_path=tmp_path / "bundle.json",
        metadata={"grid_size": 384},
        budget_bytes=budget_bytes,
    )

    summary = guard.finalize(
        status="completed",
        context={"best_params": {"target_vol": 0.10}},
    )
    payload = json.loads(guard.summary_path.read_text(encoding="utf-8"))

    assert payload == summary
    assert payload["artifact_kind"] == "portfolio_followup_memory_summary"
    assert payload["run_name"] == "portfolio_four_sleeve_search"
    assert payload["label"] == "portfolio_followup::portfolio_four_sleeve_search"
    assert payload["memory_policy"]["explicit_budget_bytes"] == budget_bytes
    assert payload["memory_policy"]["heavy_lock_path"] == str(
        contract.PORTFOLIO_FOLLOWUP_HEAVY_LOCK_PATH.resolve()
    )
    assert payload["memory_policy"]["session_memory_lease_path"] == str(
        contract.PORTFOLIO_FOLLOWUP_SESSION_MEMORY_LEASE_PATH.resolve()
    )
    assert "/src/var/" not in payload["memory_policy"]["session_memory_lease_path"]
    assert payload["context"] == {"best_params": {"target_vol": 0.10}}
    assert payload["rss_log_path"] == str(guard.rss_log_path)
    assert dummy_lock.released is False


def test_portfolio_followup_default_budget_bytes_prefers_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LQ_PORTFOLIO_FOLLOWUP_BUDGET_GIB", "5.5")

    assert contract.portfolio_followup_default_budget_bytes() == int(5.5 * 1024 * 1024 * 1024)


def test_portfolio_memory_guard_acquires_session_memory_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    heavy_lock = _DummyLock()
    session_lock = _DummyLock()
    monkeypatch.setattr(
        contract,
        "HeavyRunLock",
        type(
            "DummyHeavyRunLock",
            (),
            {"acquire": staticmethod(lambda **kwargs: heavy_lock)},
        ),
    )
    monkeypatch.setattr(
        contract,
        "acquire_session_memory_lease",
        lambda **kwargs: captured.update(kwargs) or session_lock,
    )
    monkeypatch.setattr(contract, "RSSGuard", _DummyRSSGuard)

    budget_bytes = 6 * 1024 * 1024 * 1024
    guard = contract.acquire_portfolio_memory_guard(
        run_name="portfolio_optimizer",
        output_dir=tmp_path / "optimizer",
        input_path=tmp_path / "bundle.json",
        budget_bytes=budget_bytes,
    )
    guard.release()

    assert captured["requested_budget_bytes"] == budget_bytes
    assert captured["effective_budget_bytes"] == budget_bytes
    assert (
        Path(captured["lock_path"]).resolve()
        == contract.PORTFOLIO_FOLLOWUP_SESSION_MEMORY_LEASE_PATH.resolve()
    )
    assert session_lock.released is True
    assert heavy_lock.released is True


@pytest.mark.parametrize(
    ("module_name", "script_relpath", "parser_factory"),
    [
        (
            "refresh_final_portfolio_validation_data_contract",
            "scripts/research/refresh_final_portfolio_validation_data.py",
            "build_parser",
        ),
        (
            "validate_saved_incumbent_portfolio_contract",
            "scripts/research/validate_saved_incumbent_portfolio.py",
            "build_parser",
        ),
        (
            "validate_saved_incumbent_portfolio_continuity_contract",
            "scripts/research/validate_saved_incumbent_portfolio_continuity.py",
            "build_parser",
        ),
        (
            "run_portfolio_optimization_contract",
            "scripts/run_portfolio_optimization.py",
            "_build_parser",
        ),
    ],
)
def test_followup_parser_defaults_re_read_env_budget(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    script_relpath: str,
    parser_factory: str,
) -> None:
    module = _load_script_module(module_name, ROOT / script_relpath)
    monkeypatch.setenv("LQ_PORTFOLIO_FOLLOWUP_BUDGET_GIB", "5.5")

    args = getattr(module, parser_factory)().parse_args([])

    assert args.memory_budget_bytes == int(5.5 * 1024 * 1024 * 1024)


def test_anchored_search_finalizes_failed_guard_when_optimizer_crashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if SEARCH_MODULE is None:
        pytest.skip("search wrapper module missing")

    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "selection_basis": "incumbent_anchor_rolling_gate",
                "rolling_admission_blocked": False,
                "selected_team": [
                    {"candidate_id": "c1"},
                    {"candidate_id": "c2"},
                    {"candidate_id": "c3"},
                    {"candidate_id": "c4"},
                ],
            }
        ),
        encoding="utf-8",
    )

    crash_guard = _CrashGuard(tmp_path / "search")
    monkeypatch.setattr(
        SEARCH_MODULE,
        "acquire_portfolio_memory_guard",
        lambda **kwargs: crash_guard,
    )
    monkeypatch.setattr(
        SEARCH_MODULE,
        "iter_search_grid",
        lambda: [
            {
                "correlation_threshold": 0.35,
                "cost_penalty": 0.0,
                "max_strategy_cap": 0.15,
                "max_family_cap": 0.45,
                "target_vol": 0.06,
            }
        ],
    )

    def _boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("optimizer crashed")

    monkeypatch.setattr(SEARCH_MODULE, "_run_optimizer", _boom)

    with pytest.raises(RuntimeError, match="optimizer crashed"):
        SEARCH_MODULE.run_search(
            bundle_path=bundle_path,
            search_dir=tmp_path / "search",
            tuned_dir=tmp_path / "tuned",
            comparison_json_path=tmp_path / "comparison.json",
            comparison_md_path=tmp_path / "comparison.md",
        )

    assert crash_guard.finalize_calls == [
        {"status": "failed", "error": "optimizer crashed", "context": {}}
    ]
    assert crash_guard.released is True
    summary_path = (
        tmp_path / "search" / "_memory_guard" / "portfolio_four_sleeve_search_memory_latest.json"
    )
    assert summary_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_dynamic_report_passes_explicit_8gib_budget_and_emits_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if DYNAMIC_MODULE is None:
        pytest.skip("dynamic portfolio module missing")

    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps({"selected_team": []}), encoding="utf-8")
    captured: dict[str, Any] = {}

    def _fake_acquire(**kwargs: Any) -> _ArtifactGuard:
        captured.update(kwargs)
        return _ArtifactGuard(
            run_name=str(kwargs["run_name"]),
            output_dir=Path(kwargs["output_dir"]),
            budget_bytes=kwargs.get("budget_bytes"),
        )

    monkeypatch.setattr(DYNAMIC_MODULE, "acquire_portfolio_memory_guard", _fake_acquire)
    monkeypatch.setattr(DYNAMIC_MODULE, "resolve_incumbent_bundle_path", lambda path: Path(path))
    monkeypatch.setattr(DYNAMIC_MODULE, "_load_candidates", lambda path: [{"candidate_id": "c1"}])
    monkeypatch.setattr(
        DYNAMIC_MODULE,
        "search_dynamic_allocator",
        lambda rows, progress_callback=None: {
            "params": {"lookback_days": 14, "rebalance_days": 7},
            "objective": 1.23,
            "result": _allocator_result(),
        },
    )

    result = DYNAMIC_MODULE.write_dynamic_allocator_report(
        input_path=bundle_path,
        output_dir=tmp_path / "dynamic",
    )

    expected_budget = contract.PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
    summary_path = (
        tmp_path / "dynamic" / "_memory_guard" / "causal_dynamic_portfolio_memory_latest.json"
    )
    assert _observed_budget_evidence(
        captured=captured,
        result=result,
        summary_path=summary_path,
    ) == {
        "captured_budget_bytes": expected_budget,
        "payload_memory_policy_budget_bytes": expected_budget,
        "memory_summary_budget_bytes": expected_budget,
        "summary_file_budget_bytes": expected_budget,
    }


def test_overlay_report_passes_explicit_8gib_budget_and_emits_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if OVERLAY_MODULE is None:
        pytest.skip("overlay portfolio module missing")

    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps({"selected_team": []}), encoding="utf-8")
    backbone_path = tmp_path / "backbone.json"
    backbone_path.write_text(json.dumps({"weights": []}), encoding="utf-8")
    captured: dict[str, Any] = {}

    def _fake_acquire(**kwargs: Any) -> _ArtifactGuard:
        captured.update(kwargs)
        return _ArtifactGuard(
            run_name=str(kwargs["run_name"]),
            output_dir=Path(kwargs["output_dir"]),
            budget_bytes=kwargs.get("budget_bytes"),
        )

    monkeypatch.setattr(OVERLAY_MODULE, "acquire_portfolio_memory_guard", _fake_acquire)
    monkeypatch.setattr(OVERLAY_MODULE, "resolve_incumbent_bundle_path", lambda path: Path(path))
    monkeypatch.setattr(
        OVERLAY_MODULE._helper, "_load_candidates", lambda path: [{"candidate_id": "c1"}]
    )
    monkeypatch.setattr(OVERLAY_MODULE, "_load_backbone_weights", lambda path: {"c1": 1.0})
    monkeypatch.setattr(
        OVERLAY_MODULE,
        "search_overlay_allocator",
        lambda rows, backbone_weights, progress_callback=None: {
            "params": {"lookback_days": 14, "rebalance_days": 7},
            "objective": 0.98,
            "result": _allocator_result(),
        },
    )

    result = OVERLAY_MODULE.write_overlay_report(
        input_path=bundle_path,
        backbone_path=backbone_path,
        output_dir=tmp_path / "overlay",
    )

    expected_budget = contract.PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
    summary_path = (
        tmp_path / "overlay" / "_memory_guard" / "causal_overlay_portfolio_memory_latest.json"
    )
    assert _observed_budget_evidence(
        captured=captured,
        result=result,
        summary_path=summary_path,
    ) == {
        "captured_budget_bytes": expected_budget,
        "payload_memory_policy_budget_bytes": expected_budget,
        "memory_summary_budget_bytes": expected_budget,
        "summary_file_budget_bytes": expected_budget,
    }
