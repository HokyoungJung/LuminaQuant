from __future__ import annotations

import importlib.util
import math
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "search_portfolio_four_sleeve_anchored.py"
MODULE = None
if MODULE_PATH.exists():
    SPEC = importlib.util.spec_from_file_location(
        "search_portfolio_four_sleeve_anchored", MODULE_PATH
    )
    if SPEC is None or SPEC.loader is None:
        raise RuntimeError("Failed to load search_portfolio_four_sleeve_anchored module")
    MODULE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MODULE)


def _grid_size() -> int:
    if MODULE is None:
        raise AssertionError(
            "Expected scripts/research/search_portfolio_four_sleeve_anchored.py to exist."
        )
    for attr_name in ("SEARCH_GRID", "PARAM_GRID", "SEARCH_SPACE", "GRID_OPTIONS"):
        grid = getattr(MODULE, attr_name, None)
        if isinstance(grid, dict) and grid:
            size = 1
            for values in grid.values():
                size *= len(list(values or []))
            return int(size)

    for attr_name in (
        "iter_search_grid",
        "_iter_search_grid",
        "build_search_grid",
        "_build_search_grid",
    ):
        attr = getattr(MODULE, attr_name, None)
        if callable(attr):
            built = attr()
            if isinstance(built, dict) and built:
                size = 1
                for values in built.values():
                    size *= len(list(values or []))
                return int(size)
            return len(list(built or []))

    raise AssertionError(
        "Expected search_portfolio_four_sleeve_anchored to expose a search-grid helper "
        "(SEARCH_GRID/PARAM_GRID or iter_search_grid/build_search_grid)."
    )


def test_search_wrapper_script_exists() -> None:
    assert MODULE_PATH.exists(), "missing scripts/research/search_portfolio_four_sleeve_anchored.py"


def test_search_grid_declares_full_384_run_sweep() -> None:
    assert _grid_size() == 384


def test_search_grid_has_expected_axis_cardinalities() -> None:
    if MODULE is None:
        pytest.skip("search wrapper module missing")
    expected = {
        "correlation_threshold": 4,
        "cost_penalty": 4,
        "max_strategy_cap": 4,
        "max_family_cap": 2,
        "target_vol": 3,
    }

    for attr_name in ("GRID_VALUES", "SEARCH_GRID", "PARAM_GRID", "SEARCH_SPACE", "GRID_OPTIONS"):
        grid = getattr(MODULE, attr_name, None)
        if isinstance(grid, dict) and grid:
            normalized = {str(key): len(list(values or [])) for key, values in grid.items()}
            break
    else:
        pytest.skip("module does not expose grid axes as a dictionary")

    assert expected.items() <= normalized.items()
    assert math.prod(normalized[name] for name in expected) == 384


def test_run_search_writes_summary_and_best_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if MODULE is None:
        pytest.skip("search wrapper module missing")

    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "selection_basis": "incumbent_anchor_rolling_gate",
                "rolling_admission_blocked": False,
                "selected_team": [
                    {"candidate_id": "c1", "name": "c1"},
                    {"candidate_id": "c2", "name": "c2"},
                    {"candidate_id": "c3", "name": "c3"},
                    {"candidate_id": "c4", "name": "c4"},
                ],
            }
        ),
        encoding="utf-8",
    )
    rolling_gate_path = tmp_path / "rolling_gate.json"
    rolling_gate_path.write_text(
        json.dumps({"selection_basis": "train_val_only", "survives_train_val": True}),
        encoding="utf-8",
    )
    incumbent_path = tmp_path / "incumbent.json"
    incumbent_path.write_text(
        json.dumps({"portfolio_metrics": {"oos": {"total_return": 0.05}}, "weights": []}),
        encoding="utf-8",
    )
    equal_weight_path = tmp_path / "equal_weight.json"
    equal_weight_path.write_text(
        json.dumps({"metrics": {"oos": {"total_return": 0.04}}, "selection": []}),
        encoding="utf-8",
    )
    prior_tuned_path = tmp_path / "prior_tuned.json"
    prior_tuned_path.write_text(
        json.dumps({"portfolio_metrics": {"oos": {"total_return": 0.03}}, "weights": []}),
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    class DummyGuard:
        def checkpoint(self, event: str, context: dict[str, Any] | None = None) -> None:
            captured.setdefault("checkpoints", []).append((event, context))

        def sample(self, *, event: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
            captured.setdefault("samples", []).append((event, context))
            return {"event": event, "context": context or {}}

        def finalize(
            self,
            *,
            status: str,
            error: str | None = None,
            context: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            captured["finalize"] = {"status": status, "error": error, "context": context or {}}
            return {"status": status, "error": error}

        def release(self) -> None:
            captured["released"] = True

    def _fake_acquire(**kwargs: Any) -> DummyGuard:
        captured["budget_bytes"] = kwargs.get("budget_bytes")
        return DummyGuard()

    def _fake_run_optimizer(
        bundle_path: Path,
        output_dir: Path,
        params: dict[str, float],
    ) -> dict[str, Any]:
        val_sharpe = (
            params["correlation_threshold"] + params["target_vol"] + params["max_strategy_cap"]
        )
        val_return = params["max_family_cap"] / 10.0
        max_drawdown = params["cost_penalty"] / 10.0
        report = {
            "portfolio_metrics": {
                "val": {
                    "sharpe": val_sharpe,
                    "total_return": val_return,
                    "max_drawdown": max_drawdown,
                    "turnover": params["cost_penalty"],
                },
                "oos": {
                    "sharpe": val_sharpe / 2.0,
                    "total_return": val_return / 2.0,
                    "max_drawdown": max_drawdown,
                },
            },
            "weights": [
                {"candidate_id": "c1", "weight": 0.25},
                {"candidate_id": "c2", "weight": 0.25},
                {"candidate_id": "c3", "weight": 0.25},
                {"candidate_id": "c4", "weight": 0.25},
            ],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "portfolio_optimization_latest.json"
        md_path = output_dir / "portfolio_optimization_latest.md"
        json_path.write_text(json.dumps(report), encoding="utf-8")
        md_path.write_text("# fake optimization\n", encoding="utf-8")
        return {
            "report": report,
            "json_path": json_path,
            "md_path": md_path,
            "stdout": "",
        }

    monkeypatch.setattr(MODULE, "acquire_portfolio_memory_guard", _fake_acquire)
    monkeypatch.setattr(MODULE, "_run_optimizer", _fake_run_optimizer)

    result = MODULE.run_search(
        bundle_path=bundle_path,
        search_dir=tmp_path / "search",
        tuned_dir=tmp_path / "tuned",
        comparison_json_path=tmp_path / "comparison.json",
        comparison_md_path=tmp_path / "comparison.md",
        rolling_gate_path=rolling_gate_path,
        incumbent_portfolio_path=incumbent_path,
        equal_weight_path=equal_weight_path,
        prior_tuned_path=prior_tuned_path,
    )

    summary = json.loads(Path(result["summary_json_path"]).read_text(encoding="utf-8"))
    assert summary["runs"] == 384
    assert summary["objective"] == MODULE.OBJECTIVE_FORMULA
    assert summary["best_params"] == {
        "correlation_threshold": 0.65,
        "cost_penalty": 0.0,
        "max_strategy_cap": 0.3,
        "max_family_cap": 0.55,
        "target_vol": 0.1,
    }
    assert Path(result["tuned_json_path"]).exists()
    assert Path(result["comparison_json_path"]).exists()
    assert captured.get("budget_bytes") == MODULE.PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
    assert captured.get("released") is True


def test_run_search_refuses_blocked_bundle(tmp_path: Path) -> None:
    if MODULE is None:
        pytest.skip("search wrapper module missing")

    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(
        json.dumps({"rolling_admission_blocked": True, "selected_team": []}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError):
        MODULE.run_search(bundle_path=bundle_path, search_dir=tmp_path / "search")


def _bundle_payload(*, blocked: bool = False) -> dict[str, Any]:
    rows = []
    for idx in range(4):
        rows.append(
            {
                "candidate_id": f"c{idx}",
                "name": f"candidate_{idx}",
                "strategy_class": "StubStrategy",
                "family": "trend" if idx % 2 == 0 else "cross_sectional",
                "strategy_timeframe": "1h",
                "timeframe": "1h",
                "symbols": ["BTC/USDT"],
                "return_streams": {
                    "train": [{"t": float(i), "v": 0.0001} for i in range(5)],
                    "val": [{"t": float(i), "v": 0.0002} for i in range(5)],
                    "oos": [{"t": float(i), "v": 0.00015} for i in range(5)],
                },
                "pass": True,
            }
        )
    return {
        "artifact_kind": "portfolio_four_sleeve_anchored_bundle",
        "selection_basis": "incumbent_anchor_rolling_gate",
        "selected_team": rows,
        "candidates": rows,
        "rolling_gate": {"survives_train_val": not blocked, "selection_basis": "train_val_only"},
        "rolling_admission_blocked": blocked,
    }


def _optimizer_payload(
    *,
    val_sharpe: float,
    val_return: float,
    val_max_drawdown: float,
    turnover: float,
    weights: list[float],
) -> dict[str, Any]:
    return {
        "portfolio_metrics": {
            "val": {
                "sharpe": val_sharpe,
                "total_return": val_return,
                "max_drawdown": val_max_drawdown,
                "turnover": turnover,
            },
            "oos": {
                "sharpe": 1.0,
                "total_return": 0.02,
                "max_drawdown": 0.05,
                "turnover": turnover,
            },
        },
        "weights": [
            {"candidate_id": f"c{idx}", "weight": weight} for idx, weight in enumerate(weights)
        ],
    }


def test_search_wrapper_selects_best_objective_and_writes_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if MODULE is None:
        pytest.skip("search wrapper module missing")

    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    search_output_dir = tmp_path / "search"
    optimizer_output_dir = tmp_path / "optimizer"

    grid = [
        {
            "correlation_threshold": 0.35,
            "cost_penalty": 0.0,
            "max_strategy_cap": 0.15,
            "max_family_cap": 0.45,
            "target_vol": 0.06,
        },
        {
            "correlation_threshold": 0.45,
            "cost_penalty": 0.1,
            "max_strategy_cap": 0.20,
            "max_family_cap": 0.45,
            "target_vol": 0.08,
        },
        {
            "correlation_threshold": 0.55,
            "cost_penalty": 0.2,
            "max_strategy_cap": 0.25,
            "max_family_cap": 0.55,
            "target_vol": 0.10,
        },
    ]
    monkeypatch.setattr(MODULE, "iter_search_grid", lambda: list(grid))

    optimizer_payloads = iter(
        [
            _optimizer_payload(
                val_sharpe=1.10,
                val_return=0.030,
                val_max_drawdown=0.060,
                turnover=0.20,
                weights=[0.25, 0.25, 0.25, 0.25],
            ),
            _optimizer_payload(
                val_sharpe=1.35,
                val_return=0.028,
                val_max_drawdown=0.040,
                turnover=0.18,
                weights=[0.40, 0.30, 0.20, 0.10],
            ),
            _optimizer_payload(
                val_sharpe=1.00,
                val_return=0.020,
                val_max_drawdown=0.030,
                turnover=0.25,
                weights=[0.70, 0.10, 0.10, 0.10],
            ),
        ]
    )

    def fake_run_optimizer(*args, **kwargs):
        payload = next(optimizer_payloads)
        output_dir = Path(kwargs["output_dir"] if "output_dir" in kwargs else args[1])
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "portfolio_optimization_latest.json"
        md_path = output_dir / "portfolio_optimization_latest.md"
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        md_path.write_text("# optimizer\n", encoding="utf-8")
        return {
            "report": payload,
            "json_path": json_path,
            "md_path": md_path,
            "stdout": "",
        }

    monkeypatch.setattr(MODULE, "_run_optimizer", fake_run_optimizer)

    captured: dict[str, Any] = {}

    class DummyMemoryGuard:
        def __init__(self, output_dir: Path, budget_bytes: int) -> None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.guard = type("Guard", (), {"budget_bytes": budget_bytes})()
            self.rss_log_path = (
                self.output_dir / "_memory_guard" / "portfolio_four_sleeve_search_rss_latest.jsonl"
            )
            self.rss_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.rss_log_path.write_text("", encoding="utf-8")

        def sample(self, *, event: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
            self.rss_log_path.write_text(
                self.rss_log_path.read_text(encoding="utf-8")
                + json.dumps({"event": event, "context": context or {}})
                + "\n",
                encoding="utf-8",
            )
            return {"event": event}

        def checkpoint(self, event: str, context: dict[str, Any] | None = None) -> None:
            self.sample(event=event, context=context)

        def finalize(
            self,
            *,
            status: str,
            error: str | None = None,
            context: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            summary_path = (
                self.output_dir
                / "_memory_guard"
                / "portfolio_four_sleeve_search_memory_latest.json"
            )
            payload = {
                "status": status,
                "error": error,
                "context": context or {},
                "rss_log_path": str(self.rss_log_path),
                "summary_path": str(summary_path),
            }
            summary_path.write_text(json.dumps(payload), encoding="utf-8")
            return payload

        def release(self) -> None:
            return None

    def fake_acquire_portfolio_memory_guard(**kwargs):
        captured.update(kwargs)
        return DummyMemoryGuard(Path(kwargs["output_dir"]), int(kwargs["budget_bytes"]))

    monkeypatch.setattr(
        MODULE, "acquire_portfolio_memory_guard", fake_acquire_portfolio_memory_guard
    )

    fixed_budget_bytes = MODULE.PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
    comparison_json_path = tmp_path / "comparison.json"
    comparison_md_path = tmp_path / "comparison.md"
    result = MODULE.run_search(
        bundle_path=bundle_path,
        search_dir=search_output_dir,
        tuned_dir=optimizer_output_dir,
        comparison_json_path=comparison_json_path,
        comparison_md_path=comparison_md_path,
    )

    summary = json.loads(Path(result["summary_json_path"]).read_text(encoding="utf-8"))
    assert summary["runs"] == 3
    assert summary["best_params"] == grid[1]
    assert summary["best_metric"] == pytest.approx(
        MODULE._objective_from_report(
            _optimizer_payload(
                val_sharpe=1.35,
                val_return=0.028,
                val_max_drawdown=0.040,
                turnover=0.18,
                weights=[0.40, 0.30, 0.20, 0.10],
            )
        )[1]
    )
    assert captured["budget_bytes"] == fixed_budget_bytes
    assert Path(
        search_output_dir / "_memory_guard" / "portfolio_four_sleeve_search_rss_latest.jsonl"
    ).exists()
    assert Path(result["tuned_json_path"]).exists()
    assert Path(result["summary_json_path"]).exists()
    assert Path(result["summary_md_path"]).exists()
    assert comparison_json_path.exists()
    assert comparison_md_path.exists()


def test_search_wrapper_blocks_when_rolling_gate_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if MODULE is None:
        pytest.skip("search wrapper module missing")

    bundle_path = tmp_path / "blocked_bundle.json"
    bundle_path.write_text(json.dumps(_bundle_payload(blocked=True)), encoding="utf-8")

    class DummyMemoryGuard:
        def __init__(self) -> None:
            self.guard = type("Guard", (), {"budget_bytes": 123})()

        def sample(self, *, event: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
            return {"event": event}

        def checkpoint(self, event: str, context: dict[str, Any] | None = None) -> None:
            return None

        def finalize(
            self,
            *,
            status: str,
            error: str | None = None,
            context: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            return {"status": status, "error": error, "context": context or {}}

        def release(self) -> None:
            return None

    monkeypatch.setattr(
        MODULE, "acquire_portfolio_memory_guard", lambda **kwargs: DummyMemoryGuard()
    )

    with pytest.raises(RuntimeError, match="rolling admission is blocked"):
        MODULE.run_search(
            bundle_path=bundle_path,
            search_dir=tmp_path / "search",
            tuned_dir=tmp_path / "optimizer",
            comparison_json_path=tmp_path / "comparison.json",
            comparison_md_path=tmp_path / "comparison.md",
        )
