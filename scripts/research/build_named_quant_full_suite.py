#!/usr/bin/env python3
"""Build the deterministic research-only union of both named-quant suites."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAIN = ROOT / "configs/research/named_quant_crypto_tradfi_suite_v1.json"
DEFAULT_CLAUDE = ROOT / "configs/research/named_quant_claude_suite_v1.json"
DEFAULT_OUTPUT = ROOT / "configs/research/named_quant_full_suite_v1.json"


def _merge_rows(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    *,
    id_key: str,
    label: str,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in [*first, *second]:
        if not isinstance(row, dict) or not str(row.get(id_key) or ""):
            raise ValueError(f"{label} row missing {id_key}")
        row_id = str(row[id_key])
        if row_id in merged and merged[row_id] != row:
            raise ValueError(f"conflicting duplicate {label} id: {row_id}")
        merged.setdefault(row_id, deepcopy(row))
    return list(merged.values())


def _merge_sleeves(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(first)
    for sleeve_id, sleeve in second.items():
        if sleeve_id in merged and merged[sleeve_id] != sleeve:
            raise ValueError(f"conflicting duplicate sleeve id: {sleeve_id}")
        merged.setdefault(sleeve_id, deepcopy(sleeve))
    return merged


def build_full_suite(main: dict[str, Any], claude: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(main, dict) or not isinstance(claude, dict):
        raise TypeError("both suites must be JSON objects")
    result = deepcopy(main)
    allocator = deepcopy(claude.get("allocator") or {})
    recipes = deepcopy(claude.get("portfolio_recipes") or [])
    recipes.extend(
        deepcopy(row)
        for row in main.get("portfolio_recipes") or []
        if row.get("method") in {"two_stage_hrp", "deep_rl_portfolio"}
    )
    result.update(
        {
            "suite_id": "named_quant_full_suite_v1",
            "cell_id": "named_quant_full_hierarchical_cell_v1",
            "status": "research_only_pending_data_pc_backtest",
            "research_only": True,
            "promotion_eligible": False,
            "allow_real_money": False,
            "performance_claim": "none; hypotheses are unmeasured until data-PC validation",
            "source_suite_ids": [main.get("suite_id"), claude.get("suite_id")],
            "candidates": _merge_rows(
                list(main.get("candidates") or []),
                list(claude.get("candidates") or []),
                id_key="candidate_id",
                label="candidate",
            ),
            "evidence_sources": _merge_rows(
                list(main.get("evidence_sources") or []),
                list(claude.get("evidence_sources") or []),
                id_key="source_id",
                label="source",
            ),
            "source_artifacts": _merge_rows(
                list(main.get("source_artifacts") or []),
                list(claude.get("source_artifacts") or []),
                id_key="id",
                label="source artifact",
            ),
            "sleeves": _merge_sleeves(
                dict(main.get("sleeves") or {}), dict(claude.get("sleeves") or {})
            ),
            "allocator": allocator,
            "allocator_params": deepcopy(claude.get("allocator_params") or {}),
            "allocator_variants": deepcopy(claude.get("allocator_variants") or []),
            "allocator_variants_execution": claude.get("allocator_variants_execution"),
            "allocator_evidence_refs": deepcopy(claude.get("allocator_evidence_refs") or []),
            "allocation_input_policy": deepcopy(claude.get("allocation_input_policy") or {}),
            "families": deepcopy(claude.get("families") or []),
            "portfolio_recipes": recipes,
            "method": str(allocator.get("method") or "hrp"),
            "upper": deepcopy(allocator.get("upper")),
            "min_sleeves": int(allocator.get("min_sleeves", 1)),
            "gross_cap": float(allocator.get("gross_cap", 1.0)),
            "allocation_cell_source": {
                "suite_id": claude.get("suite_id"),
                "cell_id": claude.get("cell_id"),
            },
            "asset_level_allocation_study": deepcopy(claude.get("asset_level_allocation_study")),
        }
    )
    return result


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-suite", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--claude-suite", type=Path, default=DEFAULT_CLAUDE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = build_full_suite(_load(args.main_suite), _load(args.claude_suite))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
