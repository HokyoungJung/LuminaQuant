#!/usr/bin/env python3
"""Build validation-only allocator candidates for a frozen strategy shortlist."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from lumina_quant.portfolio import hierarchical
from lumina_quant.portfolio.optimizer_core import ledoit_wolf_shrunk_covariance
from lumina_quant.portfolio.optimizers_extra import erc_weights, hrp_weights_from_returns
from lumina_quant.research.run_card import atomic_write_text, stable_json_dumps


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_portfolios(
    walkforward_path: Path,
    selection_path: Path,
    *,
    minimum_observations: int = 6,
) -> dict[str, object]:
    walkforward = json.loads(walkforward_path.read_bytes())
    selection = json.loads(selection_path.read_bytes())
    selected = [
        row["strategy"]
        for row in selection.get("selected", ())
        if type(row) is dict and type(row.get("strategy")) is str
    ]
    by_strategy: dict[str, dict[str, float]] = defaultdict(dict)
    for cell in walkforward.get("cells", ()):
        if (
            type(cell) is dict
            and cell.get("phase") == "validation"
            and cell.get("strategy") in selected
            and type(cell.get("fold_id")) is str
        ):
            by_strategy[str(cell["strategy"])][str(cell["fold_id"])] = float(
                cell.get("total_return") or 0.0
            )
    common_folds = sorted(
        set.intersection(*(set(by_strategy[name]) for name in selected))
        if selected and all(name in by_strategy for name in selected)
        else set()
    )
    result: dict[str, object] = {
        "artifact_kind": "lumina_quant.walkforward_portfolio_candidates.v1",
        "source": {
            "walkforward_path": str(walkforward_path.resolve()),
            "walkforward_sha256": _sha256(walkforward_path),
            "selection_path": str(selection_path.resolve()),
            "selection_sha256": _sha256(selection_path),
        },
        "selection_inputs": ["validation"],
        "locked_oos_role": "report_only",
        "strategies": selected,
        "common_validation_folds": common_folds,
        "minimum_observations": int(minimum_observations),
        "order_routing_enabled": False,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    if len(selected) < 2:
        result.update(
            status="skip_insufficient_survivors",
            reason="at least two validation-selected strategies are required",
            portfolios={},
        )
        return result
    if len(common_folds) < minimum_observations:
        result.update(
            status="skip_insufficient_observations",
            reason=(
                f"{len(common_folds)} common validation observations are below "
                f"the preregistered minimum {minimum_observations}"
            ),
            portfolios={},
        )
        return result
    matrix = np.column_stack(
        [[by_strategy[name][fold] for fold in common_folds] for name in selected]
    )
    covariance, _ = ledoit_wolf_shrunk_covariance(matrix)
    methods = {
        "equal_weight": np.full(len(selected), 1.0 / len(selected)),
        "erc": erc_weights(covariance),
        "hrp_threshold": hrp_weights_from_returns(selected, matrix),
        "hrp_dendrogram": hierarchical.hrp_dendrogram_weights(covariance),
        "herc": hierarchical.herc_weights(covariance),
        "nco": hierarchical.nco_weights(covariance),
    }
    portfolios = {
        method: {
            "weights": {
                strategy: float(weight) for strategy, weight in zip(selected, weights, strict=True)
            },
            "effective_bets": float(1.0 / np.square(weights).sum()),
            "maximum_weight": float(weights.max()),
        }
        for method, weights in methods.items()
    }
    result.update(status="complete", portfolios=portfolios)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--walkforward", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-observations", type=int, default=6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_portfolios(
        args.walkforward.resolve(),
        args.selection.resolve(),
        minimum_observations=max(2, args.minimum_observations),
    )
    atomic_write_text(args.output.resolve(), stable_json_dumps(result) + "\n")
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
