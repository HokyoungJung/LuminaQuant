"""Build the incumbent-only bundle for one-shot portfolio follow-up work."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP_ROOT = ROOT / "var" / "reports" / "exact_window_backtests" / "followup_status"
DEFAULT_CURRENT_BUNDLE = FOLLOWUP_ROOT / "portfolio_one_shot_current_bundle_latest.json"
DEFAULT_CURRENT_PORTFOLIO = (
    FOLLOWUP_ROOT / "portfolio_one_shot_current_opt" / "portfolio_optimization_latest.json"
)
DEFAULT_OUTPUT_JSON = FOLLOWUP_ROOT / "portfolio_one_shot_incumbent_bundle_latest.json"
DEFAULT_OUTPUT_MD = FOLLOWUP_ROOT / "portfolio_one_shot_incumbent_bundle_latest.md"
SPLIT_CONTRACT_PATH = ROOT / "src" / "lumina_quant" / "portfolio_split_contract.py"

_DEFAULT_SPLIT_CONTRACT = {
    "train_start": "2025-01-01T00:00:00Z",
    "train_end_exclusive": "2026-01-01T00:00:00Z",
    "val_start": "2026-01-01T00:00:00Z",
    "val_end_exclusive": "2026-02-01T00:00:00Z",
    "oos_start": "2026-02-01T00:00:00Z",
}
_REQUIRED_SPLIT_KEYS = tuple(_DEFAULT_SPLIT_CONTRACT.keys())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_split_payload(module: Any) -> dict[str, Any] | None:
    for attr_name in (
        "split_contract_payload",
        "build_split_contract_payload",
        "get_portfolio_split_contract",
    ):
        attr = getattr(module, attr_name, None)
        if callable(attr):
            candidate = attr()
            if isinstance(candidate, dict):
                return dict(candidate)
    for attr_name in ("SPLIT_CONTRACT", "PORTFOLIO_SPLIT_CONTRACT"):
        candidate = getattr(module, attr_name, None)
        if isinstance(candidate, dict):
            return dict(candidate)
    if all(hasattr(module, key.upper()) for key in _REQUIRED_SPLIT_KEYS):
        return {key: getattr(module, key.upper()) for key in _REQUIRED_SPLIT_KEYS}
    return None


def _resolve_split_contract() -> dict[str, Any]:
    payload = {**_DEFAULT_SPLIT_CONTRACT, "source": "local_default_constants"}
    if not SPLIT_CONTRACT_PATH.exists():
        return payload

    spec = importlib.util.spec_from_file_location("portfolio_split_contract", SPLIT_CONTRACT_PATH)
    if spec is None or spec.loader is None:
        payload["source"] = str(SPLIT_CONTRACT_PATH.resolve())
        return payload

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        payload["source"] = str(SPLIT_CONTRACT_PATH.resolve())
        return payload
    candidate = _maybe_split_payload(module)
    if not isinstance(candidate, dict):
        payload["source"] = str(SPLIT_CONTRACT_PATH.resolve())
        return payload

    resolved = dict(payload)
    for key in _REQUIRED_SPLIT_KEYS:
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            resolved[key] = value.strip()
    resolved["source"] = str(SPLIT_CONTRACT_PATH.resolve())
    return resolved


def _row_lookup_maps(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "").strip()
        name = str(row.get("name") or "").strip()
        if candidate_id and candidate_id not in by_id:
            by_id[candidate_id] = dict(row)
        if name and name not in by_name:
            by_name[name] = dict(row)
    return by_id, by_name


def _weight_lookup_key(weight_row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(weight_row.get("candidate_id") or "").strip(),
        str(weight_row.get("name") or "").strip(),
    )


def _weight_share(weight_row: dict[str, Any]) -> float:
    if "weight_share" in weight_row:
        return _safe_float(weight_row.get("weight_share"), 0.0)
    return _safe_float(weight_row.get("weight"), 0.0)


def build_portfolio_one_shot_incumbent_bundle(
    *,
    current_bundle_path: Path | str = DEFAULT_CURRENT_BUNDLE,
    current_portfolio_path: Path | str = DEFAULT_CURRENT_PORTFOLIO,
) -> dict[str, Any]:
    bundle_path = Path(current_bundle_path)
    portfolio_path = Path(current_portfolio_path)
    bundle_payload = dict(_load_json(bundle_path))
    portfolio_payload = dict(_load_json(portfolio_path))

    rows = [
        dict(row) for row in list(bundle_payload.get("candidates") or []) if isinstance(row, dict)
    ]
    if not rows:
        raise RuntimeError(f"no candidate rows found in {bundle_path}")

    weight_rows = [
        dict(row) for row in list(portfolio_payload.get("weights") or []) if isinstance(row, dict)
    ]
    if not weight_rows:
        raise RuntimeError(f"no incumbent weights found in {portfolio_path}")

    rows_by_id, rows_by_name = _row_lookup_maps(rows)
    selected_rows: list[dict[str, Any]] = []
    missing_keys: list[str] = []
    total_weight = 0.0
    gross_exposure = _safe_float(portfolio_payload.get("gross_exposure"), 0.0)
    cash_weight = _safe_float(portfolio_payload.get("cash_weight"), 0.0)

    for rank, weight_row in enumerate(weight_rows, start=1):
        candidate_id, name = _weight_lookup_key(weight_row)
        lookup_key = candidate_id or name
        source_row = rows_by_id.get(candidate_id) or rows_by_name.get(name)
        if source_row is None:
            missing_keys.append(lookup_key or f"rank-{rank}")
            continue

        weight_share = _weight_share(weight_row)
        gross_weight = _safe_float(weight_row.get("weight"), 0.0)
        total_weight += weight_share
        raw_notes = source_row.get("notes")
        if isinstance(raw_notes, str):
            notes = [raw_notes] if raw_notes.strip() else []
        else:
            notes = [str(item) for item in list(raw_notes or []) if str(item).strip()]
        notes.append(
            "Included in the incumbent one-shot backbone at saved "
            f"share {weight_share:.2%} and gross weight {gross_weight:.2%}."
        )

        selected = dict(source_row)
        selected.setdefault("pass", True)
        selected["notes"] = notes
        selected["incumbent_rank"] = rank
        selected["portfolio_weight"] = weight_share
        selected["_portfolio_weight"] = weight_share
        selected["gross_weight"] = gross_weight
        selected["selection_basis"] = "incumbent_saved_one_shot_weights"
        selected["incumbent_weight_source"] = str(portfolio_path.resolve())
        if not selected.get("timeframe") and selected.get("strategy_timeframe"):
            selected["timeframe"] = selected.get("strategy_timeframe")
        selected_rows.append(selected)

    if missing_keys:
        raise RuntimeError(
            "incumbent portfolio references candidates that are missing from the current bundle: "
            + ", ".join(sorted(missing_keys))
        )

    split_contract = _resolve_split_contract()
    incumbent_names = [
        str(row.get("name") or row.get("candidate_id") or "") for row in selected_rows
    ]
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_kind": "portfolio_one_shot_incumbent_bundle",
        "schema_version": "1.0",
        "selection_basis": "incumbent_saved_one_shot_weights",
        "split_contract": split_contract,
        "source_paths": {
            "current_bundle": str(bundle_path.resolve()),
            "current_portfolio": str(portfolio_path.resolve()),
        },
        "comparison_scope": ["current_one_shot_incumbent_only"],
        "weights": weight_rows,
        "candidates": selected_rows,
        "selected_team": selected_rows,
        "portfolio_metrics": dict(portfolio_payload.get("portfolio_metrics") or {}),
        "portfolio_return_streams": dict(portfolio_payload.get("portfolio_return_streams") or {}),
        "gross_exposure": gross_exposure,
        "cash_weight": cash_weight,
        "incumbent_summary": {
            "component_count": len(selected_rows),
            "component_names": incumbent_names,
            "weight_total": total_weight,
            "gross_exposure": gross_exposure,
            "cash_weight": cash_weight,
        },
        "notes": [
            "Bundle is restricted to the currently saved one-shot incumbent sleeves only.",
            "The incumbent bundle preserves saved weight ordering so challenger lanes can remain incumbent-first.",
            f"Locked OOS starts at {split_contract['oos_start']} and is excluded from tuning decisions.",
        ],
    }
    return payload


def write_portfolio_one_shot_incumbent_bundle(
    *,
    current_bundle_path: Path | str = DEFAULT_CURRENT_BUNDLE,
    current_portfolio_path: Path | str = DEFAULT_CURRENT_PORTFOLIO,
    output_json_path: Path | str = DEFAULT_OUTPUT_JSON,
    output_md_path: Path | str = DEFAULT_OUTPUT_MD,
) -> dict[str, Any]:
    payload = build_portfolio_one_shot_incumbent_bundle(
        current_bundle_path=current_bundle_path,
        current_portfolio_path=current_portfolio_path,
    )
    json_path = Path(output_json_path)
    md_path = Path(output_md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# portfolio one-shot incumbent bundle",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- selection_basis: `{payload['selection_basis']}`",
        f"- current_bundle: `{payload['source_paths']['current_bundle']}`",
        f"- current_portfolio: `{payload['source_paths']['current_portfolio']}`",
        f"- oos_start: `{payload['split_contract']['oos_start']}`",
        "",
        "## incumbent sleeves",
    ]
    for row in list(payload.get("candidates") or []):
        val = dict(row.get("val") or {})
        weight = _safe_float(row.get("portfolio_weight"), 0.0)
        lines.append(
            f"- `{row.get('name')}` | strategy={row.get('strategy_class')} | "
            f"tf={row.get('timeframe') or row.get('strategy_timeframe')} | "
            f"weight={weight:.2%} | "
            f"val_return={_safe_float(val.get('return'), 0.0):.4%} | "
            f"val_sharpe={_safe_float(val.get('sharpe'), 0.0):.3f}"
        )
    portfolio_oos = dict((payload.get("portfolio_metrics") or {}).get("oos") or {})
    lines.extend(
        [
            "",
            "## incumbent portfolio oos",
            f"- total_return: `{_safe_float(portfolio_oos.get('total_return'), 0.0):.4%}`",
            f"- sharpe: `{_safe_float(portfolio_oos.get('sharpe'), 0.0):.3f}`",
            f"- sortino: `{_safe_float(portfolio_oos.get('sortino'), 0.0):.3f}`",
            f"- calmar: `{_safe_float(portfolio_oos.get('calmar'), 0.0):.3f}`",
            f"- max_drawdown: `{_safe_float(portfolio_oos.get('max_drawdown'), 0.0):.4%}`",
            f"- volatility: `{_safe_float(portfolio_oos.get('volatility'), 0.0):.4%}`",
            "",
            "## notes",
        ]
    )
    for note in list(payload.get("notes") or []):
        lines.append(f"- {note}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "payload": payload,
        "json_path": str(json_path.resolve()),
        "md_path": str(md_path.resolve()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write the incumbent-only one-shot bundle artifact."
    )
    parser.add_argument("--current-bundle", default=str(DEFAULT_CURRENT_BUNDLE))
    parser.add_argument("--current-portfolio", default=str(DEFAULT_CURRENT_PORTFOLIO))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = write_portfolio_one_shot_incumbent_bundle(
        current_bundle_path=Path(args.current_bundle).resolve(),
        current_portfolio_path=Path(args.current_portfolio).resolve(),
        output_json_path=Path(args.output_json).resolve(),
        output_md_path=Path(args.output_md).resolve(),
    )
    print(result["json_path"])
    print(result["md_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
