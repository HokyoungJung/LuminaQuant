"""CLI: build a quality-gated sleeve allocation manifest from a JSON input file.

Thin wrapper around
:func:`lumina_quant.portfolio.quality_gated_allocation.build_allocation_manifest`.
No network access, no data downloads, no filesystem hashing of its own: the
input file's ``source_artifacts`` rows must already carry the ``sha256`` /
``max_age_hours`` / ``ready`` fields the live
``ArtifactPortfolioModeStrategy`` consumer checks, so the CLI does zero I/O
beyond reading the input JSON and writing the output JSON. That keeps the run
deterministic -- an identical input file always produces a byte-identical
manifest, because nothing here reads the wall clock or the filesystem.

Input JSON shape::

    {
      "method": "erc",                 # optional, default "erc" ("erc"|"hrp")
      "upper": 0.6,                    # optional per-sleeve cap (float or {id: cap})
      "min_sleeves": 1,                # optional
      "gross_cap": 1.0,                # optional
      "source_artifacts": [
        {"id": "...", "path": "...", "sha256": "...", "max_age_hours": 8760,
         "ready": true, "portfolio_ready": true}
      ],
      "sleeves": {
        "sleeve_a": {
          "returns": [0.001, -0.0005, ...],
          "turnover": 0.05,
          "strategy_class": "MovingAverageCrossStrategy",
          "symbols": ["BTC/USDT"],
          "params": {"short_window": 4, "long_window": 12},
          "source_artifact_id": "..."   # optional if exactly one source artifact
        }
      }
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lumina_quant.portfolio.quality_gated_allocation import (
    REFERENCE_COST_REGIME_20BPS,
    build_allocation_manifest,
)
from lumina_quant.research.cost_realism import DEFAULT_PARTICIPATION


def _load_input(input_path: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {input_path}")
    return payload


def build_manifest_from_input(payload: dict[str, Any]) -> dict[str, Any]:
    return build_allocation_manifest(
        dict(payload.get("sleeves") or {}),
        source_artifacts=list(payload.get("source_artifacts") or []),
        regime=REFERENCE_COST_REGIME_20BPS,
        participation=float(payload.get("participation", DEFAULT_PARTICIPATION)),
        method=str(payload.get("method", "erc")),
        upper=payload.get("upper"),
        min_sleeves=int(payload.get("min_sleeves", 1)),
        gross_cap=float(payload.get("gross_cap", 1.0)),
    )


def write_manifest(*, input_path: Path, output_path: Path) -> Path:
    payload = _load_input(input_path)
    manifest = build_manifest_from_input(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="input sleeves JSON path")
    parser.add_argument("--output", type=Path, required=True, help="output manifest JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = write_manifest(
        input_path=Path(args.input).resolve(),
        output_path=Path(args.output).resolve(),
    )
    print(str(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
