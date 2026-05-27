#!/usr/bin/env python3
"""Run Crypto/FX Alpha Zoo factor screen on real current-tail data.

The runner still accepts explicit CSV/parquet inputs for tests and controlled
probes, but when no input is supplied it discovers the latest real
profit-moonshot current-tail joined panel cache.  Source coverage is recorded
fail-closed so smoke/default-filled data cannot masquerade as live evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from lumina_quant.alpha_zoo.crypto_fx_factors import (
    add_forward_return_label,
    assign_time_splits,
    build_crypto_fx_factor_specs,
    compute_factor_frame,
    screen_factor_frame,
)
from lumina_quant.alpha_zoo.factor_card import build_factor_card
from lumina_quant.research.crypto_fx_alpha_zoo_real_data import (
    RealDataBundle,
    build_candidate_outcome_records,
    load_real_data_bundle,
    summarize_factor_source_coverage,
    write_candidate_outcome_ledger,
)

DEFAULT_OUTPUT_DIR = (
    "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "crypto_fx_alpha_zoo_real_data_20260514"
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_screen_payload(
    frame: pd.DataFrame,
    *,
    horizon: int = 4,
    top_n: int = 20,
    source_ref: str = "local:crypto_fx_alpha_zoo_screen",
    source_coverage: dict[str, Any] | None = None,
    ledger_output: str | Path | None = None,
    entry_quantile: float = 0.9,
    max_ledger_records_per_factor_side_split: int = 120,
) -> dict[str, Any]:
    specs = build_crypto_fx_factor_specs()
    factors = compute_factor_frame(frame, specs=specs)
    factors = assign_time_splits(factors)
    labeled = add_forward_return_label(factors, horizon=horizon)
    screen = screen_factor_frame(labeled, top_n=top_n)
    spec_by_name = {spec.name: spec for spec in specs}
    cards = []
    for selected in screen["selected_factors"]:
        spec = spec_by_name[str(selected["factor"])]
        card = build_factor_card(
            spec,
            metrics=selected,
            selected_using_splits=("train", "validation"),
            uses_locked_oos_for_selection=False,
            source_refs=(source_ref,),
        )
        cards.append(card.to_dict())
    factor_coverage = summarize_factor_source_coverage(labeled)
    ledger_summary: dict[str, Any] = {"enabled": bool(ledger_output), "record_count": 0}
    if ledger_output:
        records = build_candidate_outcome_records(
            labeled,
            list(screen["selected_factors"]),
            entry_quantile=entry_quantile,
            max_records_per_factor_side_split=max_ledger_records_per_factor_side_split,
        )
        ledger_summary = write_candidate_outcome_ledger(ledger_output, records)
        ledger_summary["enabled"] = True
    source_validity = dict(
        (source_coverage or {}).get("strategy_validity") or {"pass": True, "rejection_reasons": []}
    )
    card_valid = all(bool(card["strategy_validity"].get("pass")) for card in cards)
    strategy_rejections = list(source_validity.get("rejection_reasons") or [])
    if not card_valid:
        strategy_rejections.append("selected_factor_card_invalid")
    return {
        "artifact_kind": "crypto_fx_alpha_zoo_real_data_screen_bundle",
        "schema_version": 2,
        "selection_policy": "train_validation_only_locked_oos_report_only",
        "uses_locked_oos_for_selection": False,
        "locked_oos_role": "gate_report_only_after_candidate_freeze",
        "calendar_primary": False,
        "factor_count": len(specs),
        "row_count": len(labeled),
        "screen": screen,
        "factor_cards": cards,
        "source_coverage": source_coverage or {},
        "factor_source_coverage": factor_coverage,
        "candidate_outcome_ledger": ledger_summary,
        "strategy_validity": {
            "pass": not strategy_rejections,
            "calendar_primary": False,
            "uses_locked_oos_for_selection": False,
            "locked_oos_role": "gate_report_only",
            "primary_signal_type": "formulaic_state_factor",
            "rejection_reasons": sorted(set(strategy_rejections)),
        },
    }


def _bundle_from_args(args: argparse.Namespace) -> RealDataBundle:
    return load_real_data_bundle(
        input_path=args.input,
        current_tail_cache=args.current_tail_cache,
        external_state_csv=args.external_state_csv,
        strict_real_data=bool(args.strict_real_data),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="",
        help="CSV/parquet with long rows or current-tail wide panel; omitted discovers latest cache",
    )
    parser.add_argument(
        "--current-tail-cache",
        default="",
        help="Explicit joined_panel_*.parquet current-tail cache path",
    )
    parser.add_argument(
        "--external-state-csv", default="", help="Optional lagged external/FRED state CSV"
    )
    parser.add_argument(
        "--strict-real-data",
        action="store_true",
        help="Fail if required real OHLCV coverage is missing",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--ledger-output",
        default="",
        help="JSONL ledger output; default is output-dir candidate_outcome_ledger_latest.jsonl",
    )
    parser.add_argument("--entry-quantile", type=float, default=0.9)
    parser.add_argument("--max-ledger-records-per-factor-side-split", type=int, default=120)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    ledger_output = (
        Path(args.ledger_output).expanduser().resolve()
        if str(args.ledger_output).strip()
        else output_dir / "candidate_outcome_ledger_latest.jsonl"
    )
    bundle = _bundle_from_args(args)
    source_ref = f"input:{bundle.metadata.get('source_path')}"
    payload = build_screen_payload(
        bundle.frame,
        horizon=max(1, int(args.horizon)),
        top_n=max(0, int(args.top_n)),
        source_ref=source_ref,
        source_coverage=bundle.metadata,
        ledger_output=ledger_output,
        entry_quantile=float(args.entry_quantile),
        max_ledger_records_per_factor_side_split=max(
            1, int(args.max_ledger_records_per_factor_side_split)
        ),
    )
    _write_json(output_dir / "crypto_fx_alpha_zoo_screen_latest.json", payload)
    _write_json(
        output_dir / "candidate_outcome_ledger_summary_latest.json",
        dict(payload["candidate_outcome_ledger"]),
    )
    summary = [
        "# Crypto/FX Alpha Zoo real-data screen",
        "",
        f"- factor_count: `{payload['factor_count']}`",
        f"- row_count: `{payload['row_count']}`",
        f"- source_path: `{bundle.metadata.get('source_path')}`",
        f"- selection_policy: `{payload['selection_policy']}`",
        "- uses_locked_oos_for_selection: `False`",
        "- calendar_primary: `False`",
        f"- strategy_validity_pass: `{payload['strategy_validity']['pass']}`",
        f"- ledger_records: `{payload['candidate_outcome_ledger'].get('record_count', 0)}`",
        "",
        "## Source coverage",
    ]
    for symbol, item in dict(bundle.metadata.get("input", {}).get("symbol_coverage", {})).items():
        summary.append(
            f"- `{symbol}` rows `{item.get('rows')}` observed `{','.join(item.get('observed_fields') or [])}` "
            f"imputed `{','.join(item.get('imputed_fields') or [])}` required_ohlcv_observed `{item.get('required_ohlcv_observed')}`"
        )
    summary.extend(["", "## Selected factors"])
    for row in payload["screen"]["selected_factors"]:
        summary.append(f"- `{row['factor']}` score `{row['selection_score']}`")
    (output_dir / "crypto_fx_alpha_zoo_screen_latest.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
