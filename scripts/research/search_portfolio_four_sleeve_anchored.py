"""Sequential anchored search for the incumbent-aware four-sleeve portfolio."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

from lumina_quant.portfolio_split_contract import (
    FOLLOWUP_ROOT,
    PORTFOLIO_CURRENT_OPTIMIZATION,
    PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    acquire_portfolio_memory_guard,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_PATH = FOLLOWUP_ROOT / "portfolio_four_sleeve_anchored_bundle_latest.json"
DEFAULT_SEARCH_DIR = FOLLOWUP_ROOT / "portfolio_four_sleeve_search"
DEFAULT_TUNED_DIR = FOLLOWUP_ROOT / "portfolio_four_sleeve_tuned"
DEFAULT_COMPARISON_JSON = FOLLOWUP_ROOT / "portfolio_four_sleeve_comparison_latest.json"
DEFAULT_COMPARISON_MD = FOLLOWUP_ROOT / "portfolio_four_sleeve_comparison_latest.md"
DEFAULT_EQUAL_WEIGHT_PATH = FOLLOWUP_ROOT / "committee_portfolio_followup_latest.json"
DEFAULT_PRIOR_TUNED_PATH = (
    FOLLOWUP_ROOT / "portfolio_opt_tuned" / "portfolio_optimization_latest.json"
)
DEFAULT_ROLLING_GATE_PATH = FOLLOWUP_ROOT / "rolling_breakout_30m_gate_latest.json"
SEARCH_GRID: dict[str, tuple[float, ...]] = {
    "correlation_threshold": (0.35, 0.45, 0.55, 0.65),
    "cost_penalty": (0.0, 0.1, 0.2, 0.35),
    "max_strategy_cap": (0.15, 0.20, 0.25, 0.30),
    "max_family_cap": (0.45, 0.55),
    "target_vol": (0.06, 0.08, 0.10),
}
OBJECTIVE_FORMULA = (
    "val_sharpe + (12 * val_return) - (4 * val_max_drawdown) - (0.75 * hhi) - (0.15 * turnover)"
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def iter_search_grid() -> list[dict[str, float]]:
    keys = tuple(SEARCH_GRID.keys())
    values = tuple(SEARCH_GRID[key] for key in keys)
    return [
        {key: float(value) for key, value in zip(keys, combo, strict=True)}
        for combo in product(*values)
    ]


def build_search_grid() -> list[dict[str, float]]:
    return iter_search_grid()


def _weight_hhi(weights: list[dict[str, Any]]) -> float:
    return float(
        sum(_safe_float(row.get("weight"), 0.0) ** 2 for row in weights if isinstance(row, dict))
    )


def _objective_from_report(report: dict[str, Any]) -> tuple[float, list[float]]:
    val = dict((report.get("portfolio_metrics") or {}).get("val") or {})
    weights = [dict(row) for row in list(report.get("weights") or []) if isinstance(row, dict)]
    hhi = _weight_hhi(weights)
    objective = (
        _safe_float(val.get("sharpe"), 0.0)
        + (12.0 * _safe_float(val.get("total_return", val.get("return")), 0.0))
        - (4.0 * _safe_float(val.get("max_drawdown", val.get("mdd")), 0.0))
        - (0.75 * hhi)
        - (0.15 * _safe_float(val.get("turnover"), 0.0))
    )
    metric_tuple = [
        float(objective),
        _safe_float(val.get("sharpe"), 0.0),
        _safe_float(val.get("total_return", val.get("return")), 0.0),
        _safe_float(val.get("max_drawdown", val.get("mdd")), 0.0),
        float(hhi),
    ]
    return float(objective), metric_tuple


def _run_optimizer(bundle_path: Path, output_dir: Path, params: dict[str, float]) -> dict[str, Any]:
    script = ROOT / "scripts" / "run_portfolio_optimization.py"
    cmd = [
        sys.executable,
        str(script),
        "--research-report",
        str(bundle_path),
        "--team-report",
        str(bundle_path),
        "--output-dir",
        str(output_dir),
        "--fit-split",
        "val",
        "--report-split",
        "oos",
        "--correlation-threshold",
        str(params["correlation_threshold"]),
        "--cost-penalty",
        str(params["cost_penalty"]),
        "--max-strategy-cap",
        str(params["max_strategy_cap"]),
        "--max-family-cap",
        str(params["max_family_cap"]),
        "--target-vol",
        str(params["target_vol"]),
        "--max-strategies",
        "4",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or result.stdout.strip() or "portfolio optimizer failed"
        )
    latest_json = output_dir / "portfolio_optimization_latest.json"
    latest_md = output_dir / "portfolio_optimization_latest.md"
    report = json.loads(latest_json.read_text(encoding="utf-8"))
    return {
        "report": report,
        "json_path": latest_json,
        "md_path": latest_md,
        "stdout": result.stdout,
    }


def _realized_search_params(
    report: dict[str, Any], requested_params: dict[str, float]
) -> dict[str, float]:
    constraints = dict(report.get("constraints") or {})
    return {
        "correlation_threshold": float(requested_params["correlation_threshold"]),
        "cost_penalty": float(requested_params["cost_penalty"]),
        "target_vol": float(requested_params["target_vol"]),
        "max_strategy_cap": _safe_float(
            constraints.get("max_strategy"),
            requested_params["max_strategy_cap"],
        ),
        "max_family_cap": _safe_float(
            constraints.get("max_family"),
            requested_params["max_family_cap"],
        ),
    }


def _copy_best_outputs(source_json: Path, source_md: Path, tuned_dir: Path) -> tuple[Path, Path]:
    tuned_dir.mkdir(parents=True, exist_ok=True)
    target_json = tuned_dir / "portfolio_optimization_latest.json"
    target_md = tuned_dir / "portfolio_optimization_latest.md"
    shutil.copy2(source_json, target_json)
    if source_md.exists():
        shutil.copy2(source_md, target_md)
    return target_json, target_md


def _portfolio_section(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = dict(payload.get("portfolio_metrics") or payload.get("metrics") or {})
    rows = [
        dict(row)
        for row in list(payload.get("weights") or payload.get("selection") or [])
        if isinstance(row, dict)
    ]
    return {
        "path": str(path.resolve()),
        "val": dict(metrics.get("val") or {}),
        "oos": dict(metrics.get("oos") or {}),
        "weights": rows,
    }


def _build_comparison_payload(
    *,
    best_json_path: Path,
    best_report: dict[str, Any],
    rolling_gate_path: Path,
    incumbent_path: Path,
    equal_weight_path: Path,
    prior_tuned_path: Path,
    selection_basis: str,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_kind": "portfolio_four_sleeve_comparison",
        "selection_basis": selection_basis,
        "rolling_gate": json.loads(rolling_gate_path.read_text(encoding="utf-8"))
        if rolling_gate_path.exists()
        else {},
        "anchored_four_sleeve_tuned": {
            "path": str(best_json_path.resolve()),
            "val": dict((best_report.get("portfolio_metrics") or {}).get("val") or {}),
            "oos": dict((best_report.get("portfolio_metrics") or {}).get("oos") or {}),
            "weights": [
                dict(row) for row in list(best_report.get("weights") or []) if isinstance(row, dict)
            ],
        },
        "portfolio_four_sleeve_tuned": {
            "path": str(best_json_path.resolve()),
            "val": dict((best_report.get("portfolio_metrics") or {}).get("val") or {}),
            "oos": dict((best_report.get("portfolio_metrics") or {}).get("oos") or {}),
            "weights": [
                dict(row) for row in list(best_report.get("weights") or []) if isinstance(row, dict)
            ],
        },
        "current_one_shot_incumbent": _portfolio_section(incumbent_path),
        "equal_weight_diagnostic": _portfolio_section(equal_weight_path),
        "prior_exact_window_frozen_tuned": _portfolio_section(prior_tuned_path),
    }


def _write_comparison_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# portfolio four-sleeve comparison",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- selection_basis: `{payload.get('selection_basis')}`",
        "",
    ]
    for key in (
        "current_one_shot_incumbent",
        "equal_weight_diagnostic",
        "prior_exact_window_frozen_tuned",
        "anchored_four_sleeve_tuned",
    ):
        section = dict(payload.get(key) or {})
        if not section:
            continue
        oos = dict(section.get("oos") or {})
        lines.append(
            f"- {key}: return={_safe_float(oos.get('total_return', oos.get('return')), 0.0):.4%} | "
            f"sharpe={_safe_float(oos.get('sharpe'), 0.0):.3f} | "
            f"sortino={_safe_float(oos.get('sortino'), 0.0):.3f} | "
            f"calmar={_safe_float(oos.get('calmar'), 0.0):.3f} | "
            f"max_dd={_safe_float(oos.get('max_drawdown', oos.get('mdd')), 0.0):.4%}"
        )
    rolling_gate = dict(payload.get("rolling_gate") or {})
    if rolling_gate:
        lines.extend(
            [
                "",
                "## rolling gate",
                f"- selection_basis: `{rolling_gate.get('selection_basis')}`",
                f"- survives_train_val: `{rolling_gate.get('survives_train_val')}`",
                f"- recommended_action: `{rolling_gate.get('recommended_action')}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_search(
    *,
    bundle_path: Path | str = DEFAULT_BUNDLE_PATH,
    search_dir: Path | str = DEFAULT_SEARCH_DIR,
    tuned_dir: Path | str = DEFAULT_TUNED_DIR,
    comparison_json_path: Path | str = DEFAULT_COMPARISON_JSON,
    comparison_md_path: Path | str = DEFAULT_COMPARISON_MD,
    rolling_gate_path: Path | str = DEFAULT_ROLLING_GATE_PATH,
    incumbent_portfolio_path: Path | str = PORTFOLIO_CURRENT_OPTIMIZATION,
    equal_weight_path: Path | str = DEFAULT_EQUAL_WEIGHT_PATH,
    prior_tuned_path: Path | str = DEFAULT_PRIOR_TUNED_PATH,
) -> dict[str, str]:
    resolved_bundle = Path(bundle_path).resolve()
    resolved_search_dir = Path(search_dir).resolve()
    resolved_tuned_dir = Path(tuned_dir).resolve()
    resolved_comparison_json = Path(comparison_json_path).resolve()
    resolved_comparison_md = Path(comparison_md_path).resolve()
    resolved_gate = Path(rolling_gate_path).resolve()
    resolved_incumbent = Path(incumbent_portfolio_path).resolve()
    resolved_equal_weight = Path(equal_weight_path).resolve()
    resolved_prior_tuned = Path(prior_tuned_path).resolve()
    resolved_search_dir.mkdir(parents=True, exist_ok=True)
    resolved_tuned_dir.mkdir(parents=True, exist_ok=True)

    bundle_payload = json.loads(resolved_bundle.read_text(encoding="utf-8"))
    if bool(bundle_payload.get("rolling_admission_blocked")):
        raise RuntimeError("rolling admission is blocked; anchored four-sleeve search must not run")

    guard = acquire_portfolio_memory_guard(
        run_name="portfolio_four_sleeve_search",
        output_dir=resolved_search_dir,
        input_path=resolved_bundle,
        metadata={"grid_size": len(iter_search_grid())},
        budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    )
    best: dict[str, Any] | None = None
    tmp_run_dir = resolved_search_dir / "_current_run"
    try:
        for idx, params in enumerate(iter_search_grid(), start=1):
            if tmp_run_dir.exists():
                shutil.rmtree(tmp_run_dir)
            tmp_run_dir.mkdir(parents=True, exist_ok=True)
            guard.checkpoint("search_run_start", {"run_index": idx, **params})
            result = _run_optimizer(resolved_bundle, tmp_run_dir, params)
            objective, metric_tuple = _objective_from_report(result["report"])
            realized_params = _realized_search_params(result["report"], params)
            guard.sample(
                event="search_run_complete", context={"run_index": idx, "objective": objective}
            )
            if best is None or objective > float(best["objective"]):
                best_json, best_md = _copy_best_outputs(
                    result["json_path"],
                    result["md_path"],
                    resolved_tuned_dir,
                )
                best = {
                    "objective": objective,
                    "metric_tuple": metric_tuple,
                    "requested_params": dict(params),
                    "realized_params": dict(realized_params),
                    "json_path": best_json,
                    "md_path": best_md,
                    "report": dict(result["report"]),
                }
        if best is None:
            raise RuntimeError("search produced no candidate reports")

        summary = {
            "generated_at": datetime.now(UTC).isoformat(),
            "objective": OBJECTIVE_FORMULA,
            "runs": len(iter_search_grid()),
            "best_metric": list(best["metric_tuple"]),
            "best_params": dict(best["realized_params"]),
            "best_requested_params": dict(best["requested_params"]),
            "bundle_path": str(resolved_bundle),
            "output_json_path": str(Path(best["json_path"]).resolve()),
        }
        summary_json = resolved_search_dir / "best_search_summary.json"
        summary_md = resolved_search_dir / "best_search_summary.md"
        summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        summary_md.write_text(
            "\n".join(
                [
                    "# anchored four-sleeve portfolio search",
                    "",
                    f"- objective: `{summary['objective']}`",
                    f"- runs: `{summary['runs']}`",
                    f"- best_params: `{json.dumps(summary['best_params'], sort_keys=True)}`",
                    f"- best_metric: `{json.dumps(summary['best_metric'])}`",
                    f"- bundle_path: `{summary['bundle_path']}`",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        comparison_payload = _build_comparison_payload(
            best_json_path=Path(best["json_path"]).resolve(),
            best_report=dict(best["report"]),
            rolling_gate_path=resolved_gate,
            incumbent_path=resolved_incumbent,
            equal_weight_path=resolved_equal_weight,
            prior_tuned_path=resolved_prior_tuned,
            selection_basis=str(
                bundle_payload.get("selection_basis") or "incumbent_anchor_rolling_gate"
            ),
        )
        resolved_comparison_json.parent.mkdir(parents=True, exist_ok=True)
        resolved_comparison_json.write_text(
            json.dumps(comparison_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _write_comparison_markdown(comparison_payload, resolved_comparison_md)
        guard.finalize(
            status="completed",
            context={"best_params": best["realized_params"], "best_metric": best["metric_tuple"]},
        )
        return {
            "summary_json_path": str(summary_json),
            "summary_md_path": str(summary_md),
            "tuned_json_path": str(Path(best["json_path"]).resolve()),
            "tuned_md_path": str(Path(best["md_path"]).resolve()),
            "comparison_json_path": str(resolved_comparison_json),
            "comparison_md_path": str(resolved_comparison_md),
        }
    except Exception as exc:
        guard.finalize(status="failed", error=str(exc))
        raise
    finally:
        guard.release()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search anchored four-sleeve portfolio weights.")
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE_PATH))
    parser.add_argument("--search-dir", default=str(DEFAULT_SEARCH_DIR))
    parser.add_argument("--tuned-dir", default=str(DEFAULT_TUNED_DIR))
    parser.add_argument("--comparison-json", default=str(DEFAULT_COMPARISON_JSON))
    parser.add_argument("--comparison-md", default=str(DEFAULT_COMPARISON_MD))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_search(
        bundle_path=Path(args.bundle).resolve(),
        search_dir=Path(args.search_dir).resolve(),
        tuned_dir=Path(args.tuned_dir).resolve(),
        comparison_json_path=Path(args.comparison_json).resolve(),
        comparison_md_path=Path(args.comparison_md).resolve(),
    )
    print(result["summary_json_path"])
    print(result["tuned_json_path"])
    print(result["comparison_json_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
