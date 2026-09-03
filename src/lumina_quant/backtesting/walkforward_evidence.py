"""Selection and report-only evidence for event-driven walk-forward results."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_walkforward(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_bytes())
    cells = payload.get("cells")
    if (
        payload.get("artifact_kind") != "lumina_quant.event_driven_walkforward.v1"
        or payload.get("selection_uses_locked_oos") is not False
        or type(cells) is not list
    ):
        raise ValueError("walkforward artifact contract is invalid")
    return payload


def _sharpe(cell: Mapping[str, Any]) -> float:
    return float((cell.get("fast_stats") or {}).get("sharpe") or 0.0)


def select_finalists(
    path: Path,
    *,
    top_n: int,
    minimum_pass_ratio: float,
    minimum_mean_sharpe: float,
) -> dict[str, Any]:
    """Rank with validation cells only; locked-OOS values are never loaded into rows."""
    if top_n <= 0 or not 0.0 <= minimum_pass_ratio <= 1.0:
        raise ValueError("selection thresholds are invalid")
    payload = _load_walkforward(path)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for cell in payload["cells"]:
        if type(cell) is dict and cell.get("phase") == "validation":
            grouped[str(cell["strategy"])].append(cell)
    ranked: list[dict[str, Any]] = []
    for strategy, cells in grouped.items():
        pass_ratio = sum(cell.get("status") == "pass" for cell in cells) / len(cells)
        returns = [float(cell.get("total_return") or 0.0) for cell in cells]
        sharpes = [_sharpe(cell) for cell in cells]
        row = {
            "strategy": strategy,
            "validation_fold_count": len(cells),
            "validation_pass_ratio": pass_ratio,
            "validation_mean_return": fmean(returns),
            "validation_mean_sharpe": fmean(sharpes),
            "validation_positive_fold_ratio": sum(value > 0 for value in returns) / len(returns),
        }
        if (
            pass_ratio >= minimum_pass_ratio
            and row["validation_mean_return"] > 0.0
            and row["validation_mean_sharpe"] >= minimum_mean_sharpe
        ):
            ranked.append(row)
    ranked.sort(
        key=lambda row: (
            row["validation_mean_sharpe"],
            row["validation_mean_return"],
            row["validation_pass_ratio"],
            row["strategy"],
        ),
        reverse=True,
    )
    return {
        "artifact_kind": "lumina_quant.walkforward_validation_selection.v1",
        "status": "complete",
        "source": {"path": str(path.resolve()), "sha256": _sha256(path)},
        "selection_uses_locked_oos": False,
        "thresholds": {
            "top_n": top_n,
            "minimum_pass_ratio": minimum_pass_ratio,
            "minimum_mean_sharpe": minimum_mean_sharpe,
            "minimum_mean_return_exclusive": 0.0,
        },
        "selected": ranked[:top_n],
        "order_routing_enabled": False,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }


def build_report_only_evaluation(
    walkforward_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    """Attach locked-OOS metrics to a previously frozen validation selection."""
    walkforward = _load_walkforward(walkforward_path)
    selection = json.loads(selection_path.read_bytes())
    if (
        selection.get("artifact_kind") != "lumina_quant.walkforward_validation_selection.v1"
        or selection.get("selection_uses_locked_oos") is not False
        or selection.get("source", {}).get("sha256") != _sha256(walkforward_path)
    ):
        raise ValueError("selection is not bound to this walkforward artifact")
    selected = [row["strategy"] for row in selection.get("selected", ())]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for cell in walkforward["cells"]:
        if type(cell) is dict and cell.get("phase") == "locked_oos":
            grouped[str(cell["strategy"])].append(cell)
    rows: list[dict[str, Any]] = []
    for strategy in selected:
        cells = grouped.get(strategy, [])
        returns = [float(cell.get("total_return") or 0.0) for cell in cells]
        rows.append(
            {
                "strategy": strategy,
                "locked_oos_fold_count": len(cells),
                "locked_oos_pass_ratio": (
                    sum(cell.get("status") == "pass" for cell in cells) / len(cells)
                    if cells
                    else 0.0
                ),
                "locked_oos_mean_return": fmean(returns) if returns else 0.0,
                "locked_oos_mean_sharpe": (
                    fmean(_sharpe(cell) for cell in cells) if cells else 0.0
                ),
                "locked_oos_positive_fold_ratio": (
                    sum(value > 0 for value in returns) / len(returns) if returns else 0.0
                ),
            }
        )
    return {
        "artifact_kind": "lumina_quant.walkforward_report_only_evaluation.v1",
        "status": "complete",
        "walkforward": {
            "path": str(walkforward_path.resolve()),
            "sha256": _sha256(walkforward_path),
        },
        "frozen_selection": {
            "path": str(selection_path.resolve()),
            "sha256": _sha256(selection_path),
        },
        "selection_uses_locked_oos": False,
        "rows": rows,
        "order_routing_enabled": False,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
