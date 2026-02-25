"""List tunable Alpha101 constant parameters and active registry overrides."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lumina_quant.indicators.formulaic_alpha import (  # noqa: E402
    ALPHA101_PARAM_REGISTRY,
    list_alpha101_tunable_params,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan Alpha101 formulas and emit tunable constant parameter keys."
    )
    parser.add_argument("--alpha", type=int, default=None, help="Optional Alpha id filter (1..101).")
    parser.add_argument(
        "--format",
        choices=("json", "table"),
        default="json",
        help="Output formatting mode.",
    )
    parser.add_argument(
        "--include-overrides",
        action="store_true",
        help="Include currently-registered override values.",
    )
    return parser


def _table_output(params: dict[str, float], overrides: dict[str, float]) -> str:
    lines = ["| key | default | override |", "|---|---:|---:|"]
    for key in sorted(params):
        default = params[key]
        override = overrides.get(key)
        override_text = "" if override is None else f"{override:.12g}"
        lines.append(f"| {key} | {default:.12g} | {override_text} |")
    return "\n".join(lines)


def main() -> None:
    args = _build_parser().parse_args()
    if args.alpha is not None and (args.alpha < 1 or args.alpha > 101):
        raise SystemExit("--alpha must be between 1 and 101")

    params = list_alpha101_tunable_params(alpha_id=args.alpha)
    overrides = ALPHA101_PARAM_REGISTRY.snapshot(prefix="alpha101.")

    if args.format == "table":
        print(_table_output(params, overrides if args.include_overrides else {}))
        return

    payload: dict[str, object] = {
        "alpha": args.alpha,
        "count": len(params),
        "params": {key: params[key] for key in sorted(params)},
    }
    if args.include_overrides:
        payload["overrides"] = {key: overrides[key] for key in sorted(overrides)}
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
