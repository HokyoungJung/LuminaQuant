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
      "method": "erc",                 # optional, default "erc" ("erc"|"hrp"|
                                       #   "hrp_dendrogram"|"herc"|"nco"|"wasserstein_dro"|
                                       #   "graph_inverse_centrality")
      "allocator_params": {...},        # optional kwargs for the opt-in allocators
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
    allocator = payload.get("allocator")
    allocator = allocator if isinstance(allocator, dict) else {}

    def setting(name: str, default: Any) -> Any:
        return payload[name] if name in payload else allocator.get(name, default)

    min_families = (
        setting("min_families", None)
        if "min_families" in payload or "min_families" in allocator
        else None
    )
    locked_oos_evaluation = payload.get("locked_oos_evaluation")
    if "locked_oos_evaluation" in payload and not isinstance(locked_oos_evaluation, dict):
        raise ValueError("locked_oos_evaluation must be an object")
    return build_allocation_manifest(
        dict(payload.get("sleeves") or {}),
        source_artifacts=list(payload.get("source_artifacts") or []),
        regime=REFERENCE_COST_REGIME_20BPS,
        participation=float(payload.get("participation", DEFAULT_PARTICIPATION)),
        method=str(setting("method", "erc")),
        upper=setting("upper", None),
        min_sleeves=int(setting("min_sleeves", 1)),
        gross_cap=float(setting("gross_cap", 1.0)),
        turnover_penalty_lambda=float(setting("turnover_penalty_lambda", 0.0)),
        correlation_shrinkage=setting("correlation_shrinkage", None),
        family_momentum_window=int(setting("family_momentum_window", 0)),
        family_momentum_tilt_strength=float(setting("family_momentum_tilt_strength", 0.5)),
        family_momentum_tilt_cap=float(setting("family_momentum_tilt_cap", 0.30)),
        min_families=None if min_families is None else int(min_families),
        # Optional kwargs for the opt-in hierarchical/robust allocators
        # (hrp_dendrogram / herc / nco / wasserstein_dro / graph_inverse_centrality).
        allocator_params=(
            dict(setting("allocator_params", {}))
            if isinstance(setting("allocator_params", None), dict)
            else None
        ),
        locked_oos_evaluation=(
            dict(locked_oos_evaluation) if isinstance(locked_oos_evaluation, dict) else None
        ),
    )


def write_manifest(*, input_path: Path, output_path: Path) -> Path:
    payload = _load_input(input_path)
    manifest = build_manifest_from_input(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return output_path


def validate_cell_spec(payload: dict[str, Any]) -> tuple[bool, str]:
    """Validate a PRE-REGISTERED allocation cell spec (data, not a runnable input).

    A cell spec declares MEMBERSHIP only: every sleeve carries ``returns`` /
    ``turnover`` == null (the data-PC fills them after materializing the streams),
    so the spec deliberately cannot pass through the returns-required manifest
    builder. This validator checks the pre-registration is well-formed:

    * a non-empty ``sleeves`` object with at least ``allocator.min_sleeves`` entries;
    * each sleeve declares a string ``family``, an explicit ``returns``: null and
      ``turnover``: null, and a ``returns_source`` provenance block;
    * at least ``allocator.min_families`` (default 3) DISTINCT families; and
    * at least one ``source_artifacts`` provenance row.

    Returns ``(ok, message)``. Never raises on a well-formed-JSON dict -- shape
    violations are reported as ``(False, reason)``.
    """
    if not isinstance(payload, dict):
        return False, "cell spec must be a JSON object"
    sleeves = payload.get("sleeves")
    if not isinstance(sleeves, dict) or not sleeves:
        return False, "cell spec must declare a non-empty 'sleeves' object"
    allocator = payload.get("allocator")
    allocator = allocator if isinstance(allocator, dict) else {}
    min_families = int(allocator.get("min_families", 3))
    min_sleeves = int(allocator.get("min_sleeves", 2))
    if len(sleeves) < max(1, min_sleeves):
        return (
            False,
            f"cell spec declares {len(sleeves)} sleeves; "
            f"allocator.min_sleeves={min_sleeves} requires at least that many",
        )
    families: list[str] = []
    for sleeve_id, sleeve in sleeves.items():
        if not isinstance(sleeve, dict):
            return False, f"sleeve {sleeve_id!r} must be an object"
        family = sleeve.get("family")
        if not isinstance(family, str) or not family:
            return False, f"sleeve {sleeve_id!r} is missing a string 'family'"
        families.append(family)
        if "returns" not in sleeve or sleeve.get("returns") is not None:
            return (
                False,
                f"sleeve {sleeve_id!r} must declare 'returns': null "
                "(membership only; the data-PC fills the stream)",
            )
        if "turnover" not in sleeve or sleeve.get("turnover") is not None:
            return (
                False,
                f"sleeve {sleeve_id!r} must declare 'turnover': null "
                "(the data-PC fills the turnover)",
            )
        provenance = sleeve.get("returns_source")
        if not isinstance(provenance, dict) or not provenance:
            return (
                False,
                f"sleeve {sleeve_id!r} is missing a 'returns_source' provenance block",
            )
    distinct = sorted(set(families))
    if len(distinct) < max(1, min_families):
        return (
            False,
            f"cell spec has {len(distinct)} distinct families {distinct}; "
            f"allocator.min_families={min_families} requires at least that many",
        )
    source_artifacts = payload.get("source_artifacts")
    if not isinstance(source_artifacts, list) or not source_artifacts:
        return False, "cell spec must carry at least one 'source_artifacts' provenance row"
    return (
        True,
        f"OK: cell {payload.get('cell_id')!r} declares {len(sleeves)} sleeves across "
        f"{len(distinct)} distinct families {distinct} with null returns "
        "(data-PC materialization pending)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="input sleeves JSON path")
    parser.add_argument("--output", type=Path, default=None, help="output manifest JSON path")
    parser.add_argument(
        "--validate-cell-spec",
        type=Path,
        default=None,
        dest="validate_cell_spec",
        help=(
            "validate a pre-registered allocation cell spec (null-returns membership "
            "declaration) and exit 0 (valid) / 1 (invalid); does not read --input/--output"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.validate_cell_spec is not None:
        payload = _load_input(Path(args.validate_cell_spec).resolve())
        ok, message = validate_cell_spec(payload)
        print(message)
        return 0 if ok else 1
    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --validate-cell-spec is given")
    result = write_manifest(
        input_path=Path(args.input).resolve(),
        output_path=Path(args.output).resolve(),
    )
    print(str(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
