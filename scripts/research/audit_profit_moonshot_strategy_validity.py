#!/usr/bin/env python3
"""Audit profit-moonshot final-selection rows for live strategy validity.

The audit is intentionally artifact-only.  It does not search for new alpha or
rerun backtests; it verifies that the final-selection artifact has fail-closed
strategy-validity metadata, records source closure, and summarizes which rows
remain deployable after theory/live-validity gates.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_BASE_DIR = Path("var/reports/profit_moonshot_20260501/live_final_selection_20260510")
DEFAULT_OUTPUT_DIR = DEFAULT_BASE_DIR / "strategy_validity_audit"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except TypeError, ValueError:
        return float(default)


def _read_json(path: Path | str | None) -> dict[str, Any]:
    if path is None or not str(path).strip():
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _final_selection_module() -> Any:
    module_path = Path(__file__).with_name("write_profit_moonshot_live_final_selection.py")
    spec = importlib.util.spec_from_file_location(
        "write_profit_moonshot_live_final_selection_for_audit", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import final selection module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json_row_count(payload: Mapping[str, Any]) -> int:
    for key in ("rows", "tuning_results", "retune_results", "diagnostic_quarantine"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    count = 0
    for value in payload.values():
        if isinstance(value, Mapping) and value:
            count += 1
    return count


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _artifact_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _json_row_count(_read_json(path))
    if suffix == ".csv":
        return _csv_row_count(path)
    if suffix in {".md", ".txt", ".log"}:
        return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
    return 0


def build_closure_manifest(
    *,
    source_artifacts: Mapping[str, str],
    rows: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    required_roles = {
        "final_selection_json",
        "final_selection_md",
        "liquidation_validation",
        "candidate_portfolio",
        "candidate_hybrid",
    }
    optional_roles = {
        "merged_candidate_csv",
        "current_base",
        "passing_artifacts",
        "legacy_hybrid",
        "refresh_json",
    }
    role_aliases = {
        "liquidation_json": "liquidation_validation",
        "candidate_portfolio_json": "candidate_portfolio",
        "candidate_hybrid_json": "candidate_hybrid",
        "legacy_hybrid_json": "legacy_hybrid",
    }

    manifest: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for source_key, raw_path in sorted(source_artifacts.items()):
        role = role_aliases.get(source_key, source_key)
        if not str(raw_path).strip():
            if role in optional_roles:
                missing.append({"source_role": role, "reason": "not_provided"})
            elif role in required_roles:
                missing.append({"source_role": role, "reason": "required_not_provided"})
            continue
        path = Path(raw_path)
        if not path.exists():
            missing.append({"source_role": role, "path": str(path), "reason": "missing_file"})
            continue
        manifest.append(
            {
                "source_role": role,
                "path": str(path),
                "artifact_kind": path.suffix.lstrip(".") or "file",
                "row_count": _artifact_row_count(path),
            }
        )

    manifest.append(
        {
            "source_role": "per_row_per_sleeve_sources",
            "path": "inline:final_selection.rows.strategy_validity",
            "artifact_kind": "inline_audit_index",
            "row_count": sum(len(_as_list(_as_dict(row).get("sleeves"))) for row in rows),
        }
    )
    return manifest, missing


def build_strategy_validity_audit_payload(
    *,
    final_selection_payload: Mapping[str, Any],
    source_artifacts: Mapping[str, str],
) -> dict[str, Any]:
    final_module = _final_selection_module()
    rows = [
        _as_dict(row)
        for row in _as_list(final_selection_payload.get("rows"))
        if isinstance(row, Mapping)
    ]
    audit_rows = []
    missing_metadata_rows: list[str] = []
    invalid_rows: list[str] = []
    deployable_invalid_rows: list[str] = []
    for row in rows:
        validity = _as_dict(row.get("strategy_validity"))
        metadata_present = final_module._strategy_validity_metadata_present(validity)
        passed = final_module._strategy_validity_passes(validity)
        name = str(row.get("name") or "")
        if not metadata_present:
            missing_metadata_rows.append(name)
        if metadata_present and not passed:
            invalid_rows.append(name)
        if bool(_as_dict(row.get("decision_gates")).get("deployable_candidate")) and not passed:
            deployable_invalid_rows.append(name)
        audit_rows.append(
            {
                "name": name,
                "kind": row.get("kind"),
                "candidate_derived": bool(row.get("candidate_derived")),
                "benchmark_only": bool(row.get("benchmark_only")),
                "deployable_candidate": bool(
                    _as_dict(row.get("decision_gates")).get("deployable_candidate")
                ),
                "source_artifact": row.get("source_artifact"),
                "sleeves": list(row.get("sleeves") or []),
                "strategy_validity": validity,
                "rejection_reasons": list(row.get("rejection_reasons") or []),
            }
        )

    closure_manifest, missing_optional_sources = build_closure_manifest(
        source_artifacts=source_artifacts,
        rows=rows,
    )
    source_pool_summary = _source_pool_summary(
        str(source_artifacts.get("merged_candidate_csv") or ""),
        final_module=final_module,
    )
    status = "pass"
    if missing_metadata_rows or deployable_invalid_rows:
        status = "fail"
    elif invalid_rows:
        status = "pass_with_rejections"

    return {
        "artifact_kind": "profit_moonshot_strategy_validity_audit",
        "generated_at_utc": _utc_now_iso(),
        "status": status,
        "no_new_alpha_search": True,
        "selection_policy": {
            "selection_inputs": ["train", "validation"],
            "locked_oos": "report_only_gate_only",
            "uses_locked_oos_for_selection": False,
            "strategy_validity_required": True,
            "calendar_primary_alpha_rejected": True,
        },
        "summary": {
            "row_count": len(audit_rows),
            "missing_metadata_count": len(missing_metadata_rows),
            "strategy_invalid_count": len(invalid_rows),
            "deployable_invalid_count": len(deployable_invalid_rows),
            "deployable_valid_count": sum(
                1
                for row in audit_rows
                if row["deployable_candidate"]
                and final_module._strategy_validity_passes(row["strategy_validity"])
            ),
        },
        "missing_metadata_rows": missing_metadata_rows,
        "strategy_invalid_rows": invalid_rows,
        "deployable_invalid_rows": deployable_invalid_rows,
        "closure_manifest": closure_manifest,
        "missing_optional_sources": missing_optional_sources,
        "source_pool_summary": source_pool_summary,
        "rows": audit_rows,
    }


def _source_pool_summary(path_value: Path | str, *, final_module: Any) -> dict[str, Any]:
    path_token = str(path_value or "").strip()
    if not path_token:
        return {"available": False, "path": "", "reason": "not_provided"}
    path = Path(path_token)
    if not path.exists() or not path.is_file():
        return {"available": False, "path": str(path), "reason": "not_available"}
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = str(row.get("name") or "")
            family = str(row.get("family") or "")
            raw = {"sleeves": [name], "strategy_family": family}
            validity = final_module._strategy_validity(
                kind="source_pool_candidate",
                name=name,
                raw=raw,
                source_artifact=str(path),
                candidate_derived=True,
                benchmark_only=False,
            )
            train_return = _safe_float(row.get("train_total_return"))
            val_return = _safe_float(row.get("val_total_return"))
            train_mdd = _safe_float(row.get("train_max_drawdown"))
            val_mdd = _safe_float(row.get("val_max_drawdown"))
            rows.append(
                {
                    "name": name,
                    "family": family,
                    "success_candidate": _truthy(row.get("success_candidate")),
                    "replay_survivor": _truthy(row.get("replay_survivor")),
                    "strategy_validity_pass": bool(validity.get("pass")),
                    "strategy_validity_reasons": list(validity.get("rejection_reasons") or []),
                    "train_total_return": train_return,
                    "validation_total_return": val_return,
                    "oos_total_return": _safe_float(row.get("oos_total_return")),
                    "train_max_drawdown": train_mdd,
                    "validation_max_drawdown": val_mdd,
                    "oos_max_drawdown": _safe_float(row.get("oos_max_drawdown")),
                    "train_val_screen_score": train_return + val_return - train_mdd - val_mdd,
                }
            )
    dynamic_rows = [row for row in rows if bool(row["strategy_validity_pass"])]
    calendar_rows = [row for row in rows if not bool(row["strategy_validity_pass"])]
    dynamic_success = [row for row in dynamic_rows if bool(row["success_candidate"])]
    dynamic_ranked = list(dynamic_rows)
    dynamic_success.sort(
        key=lambda row: (_safe_float(row["train_val_screen_score"]), row["name"]), reverse=True
    )
    dynamic_ranked.sort(
        key=lambda row: (_safe_float(row["train_val_screen_score"]), row["name"]), reverse=True
    )
    return {
        "available": True,
        "path": str(path),
        "row_count": len(rows),
        "strategy_valid_rows": len(dynamic_rows),
        "calendar_primary_invalid_rows": len(calendar_rows),
        "success_candidate_count": sum(1 for row in rows if bool(row["success_candidate"])),
        "strategy_valid_success_candidate_count": len(dynamic_success),
        "top_strategy_valid_success_candidates": dynamic_success[:20],
        "top_strategy_valid_candidates": dynamic_ranked[:20],
    }


def build_markdown(payload: Mapping[str, Any]) -> str:
    summary = _as_dict(payload.get("summary"))
    lines = [
        "# Profit moonshot strategy-validity audit",
        "",
        f"- generated_at_utc: `{payload.get('generated_at_utc')}`",
        f"- status: `{payload.get('status')}`",
        f"- row_count: `{summary.get('row_count')}`",
        f"- strategy_invalid_count: `{summary.get('strategy_invalid_count')}`",
        f"- deployable_valid_count: `{summary.get('deployable_valid_count')}`",
        f"- deployable_invalid_count: `{summary.get('deployable_invalid_count')}`",
        f"- no_new_alpha_search: `{payload.get('no_new_alpha_search')}`",
        "",
        "## Row audit",
        "",
        "| Kind | Name | Deployable | Strategy pass | Primary signal | Reasons |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in _as_list(payload.get("rows")):
        if not isinstance(row, Mapping):
            continue
        validity = _as_dict(row.get("strategy_validity"))
        lines.append(
            "| "
            f"`{row.get('kind')}` | `{row.get('name')}` | `{bool(row.get('deployable_candidate'))}` | "
            f"`{bool(validity.get('pass'))}` | `{validity.get('primary_signal_type')}` | "
            f"`{', '.join(str(item) for item in _as_list(validity.get('rejection_reasons')))}` |"
        )
    lines.extend(["", "## Closure manifest", ""])
    for entry in _as_list(payload.get("closure_manifest")):
        if isinstance(entry, Mapping):
            lines.append(
                f"- `{entry.get('source_role')}` `{entry.get('path')}` rows=`{entry.get('row_count')}`"
            )
    pool = _as_dict(payload.get("source_pool_summary"))
    lines.extend(["", "## Source pool summary", ""])
    lines.append(f"- available: `{pool.get('available')}`")
    lines.append(f"- row_count: `{pool.get('row_count')}`")
    lines.append(f"- strategy_valid_rows: `{pool.get('strategy_valid_rows')}`")
    lines.append(f"- calendar_primary_invalid_rows: `{pool.get('calendar_primary_invalid_rows')}`")
    lines.append(
        f"- strategy_valid_success_candidate_count: `{pool.get('strategy_valid_success_candidate_count')}`"
    )
    lines.extend(["", "### Top strategy-valid success candidates from source pool", ""])
    for row in _as_list(pool.get("top_strategy_valid_success_candidates"))[:20]:
        if isinstance(row, Mapping):
            lines.append(
                f"- `{row.get('name')}` family=`{row.get('family')}` "
                f"train=`{row.get('train_total_return')}` val=`{row.get('validation_total_return')}` "
                f"oos=`{row.get('oos_total_return')}` score=`{row.get('train_val_screen_score')}`"
            )
    lines.extend(
        ["", "### Top strategy-valid candidates before success/liquidation promotion gates", ""]
    )
    for row in _as_list(pool.get("top_strategy_valid_candidates"))[:20]:
        if isinstance(row, Mapping):
            lines.append(
                f"- `{row.get('name')}` family=`{row.get('family')}` success=`{row.get('success_candidate')}` "
                f"train=`{row.get('train_total_return')}` val=`{row.get('validation_total_return')}` "
                f"oos=`{row.get('oos_total_return')}` score=`{row.get('train_val_screen_score')}`"
            )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-selection-json",
        default=str(
            DEFAULT_BASE_DIR / "final_decision" / "profit_moonshot_live_final_selection_latest.json"
        ),
    )
    parser.add_argument(
        "--final-selection-md",
        default=str(
            DEFAULT_BASE_DIR / "final_decision" / "profit_moonshot_live_final_selection_latest.md"
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--merged-candidate-csv", default="")
    parser.add_argument("--current-base", default="")
    parser.add_argument("--passing-artifacts", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    final_selection_path = Path(args.final_selection_json)
    final_selection_payload = _read_json(final_selection_path)
    final_sources = _as_dict(final_selection_payload.get("source_artifacts"))
    source_artifacts = {
        "final_selection_json": str(final_selection_path),
        "final_selection_md": str(Path(args.final_selection_md)),
        "merged_candidate_csv": str(args.merged_candidate_csv or ""),
        "current_base": str(args.current_base or ""),
        "passing_artifacts": str(args.passing_artifacts or ""),
        **{key: str(value) for key, value in final_sources.items()},
    }
    payload = build_strategy_validity_audit_payload(
        final_selection_payload=final_selection_payload,
        source_artifacts=source_artifacts,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "strategy_validity_audit_latest.json"
    md_path = output_dir / "strategy_validity_audit_latest.md"
    _write_json(json_path, payload)
    md_path.write_text(build_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_path": str(json_path),
                "markdown_path": str(md_path),
                "status": payload["status"],
            }
        )
    )
    return 1 if payload.get("status") == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
