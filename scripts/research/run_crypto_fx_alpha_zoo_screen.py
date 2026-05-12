#!/usr/bin/env python3
"""Run a deterministic Crypto/FX Alpha Zoo v0 factor screen."""

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


def _load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"unsupported input format: {path}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_screen_payload(
    frame: pd.DataFrame,
    *,
    horizon: int = 4,
    top_n: int = 20,
    source_ref: str = "local:crypto_fx_alpha_zoo_screen",
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
    return {
        "artifact_kind": "crypto_fx_alpha_zoo_v0_screen_bundle",
        "schema_version": 1,
        "selection_policy": "train_validation_only_locked_oos_report_only",
        "uses_locked_oos_for_selection": False,
        "factor_count": len(specs),
        "row_count": len(labeled),
        "screen": screen,
        "factor_cards": cards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV/parquet with timestamp,symbol,OHLCV rows")
    parser.add_argument("--output-dir", default="var/reports/crypto_fx_alpha_zoo_v0")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    payload = build_screen_payload(
        _load_frame(input_path),
        horizon=max(1, int(args.horizon)),
        top_n=max(0, int(args.top_n)),
        source_ref=f"input:{input_path}",
    )
    _write_json(output_dir / "crypto_fx_alpha_zoo_screen_latest.json", payload)
    summary = [
        "# Crypto/FX Alpha Zoo v0 screen",
        "",
        f"- factor_count: `{payload['factor_count']}`",
        f"- row_count: `{payload['row_count']}`",
        "- selection_policy: `train_validation_only_locked_oos_report_only`",
        "- uses_locked_oos_for_selection: `False`",
        "",
        "## Selected factors",
    ]
    for row in payload["screen"]["selected_factors"]:
        summary.append(f"- `{row['factor']}` score `{row['selection_score']}`")
    (output_dir / "crypto_fx_alpha_zoo_screen_latest.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
