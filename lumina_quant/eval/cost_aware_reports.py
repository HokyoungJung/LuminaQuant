"""Reporting helpers for cost-aware framework outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import polars as pl
import yaml


def _max_drawdown(equity: list[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        dd = (value / peak) - 1.0 if peak > 0 else 0.0
        worst = min(worst, dd)
    return abs(worst)


def compute_perf_metrics(returns: list[float], equity: list[float], periods_per_year: int) -> dict[str, float]:
    if not returns:
        return {"cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "total_return": 0.0, "volatility": 0.0}
    mean_r = sum(returns) / len(returns)
    variance = sum((item - mean_r) ** 2 for item in returns) / max(1, len(returns) - 1)
    std_r = variance**0.5
    sharpe = 0.0 if std_r == 0.0 else (mean_r / std_r) * (periods_per_year**0.5)
    volatility = std_r * (periods_per_year**0.5)
    total_return = (equity[-1] / equity[0]) - 1.0 if len(equity) >= 2 and equity[0] > 0 else 0.0
    years = max(1.0 / periods_per_year, len(returns) / periods_per_year)
    if total_return <= -1.0:
        cagr = -1.0
    else:
        annual_log = math.log1p(total_return) / years
        annual_log = min(annual_log, 700.0)
        cagr = math.exp(annual_log) - 1.0
    return {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": float(_max_drawdown(equity)),
        "total_return": float(total_return),
        "volatility": float(volatility),
    }


def write_report_bundle(
    *,
    run_id: str,
    output_root: str | Path,
    resolved_config: dict[str, Any],
    summary_payload: dict[str, Any],
    table_rows: list[dict[str, Any]],
) -> Path:
    """Write summary.json, tables.csv and config_resolved.yaml."""
    report_dir = Path(output_root) / run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, sort_keys=True)

    table = pl.DataFrame(table_rows) if table_rows else pl.DataFrame()
    table.write_csv(report_dir / "tables.csv")

    with (report_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=False)

    return report_dir


__all__ = ["compute_perf_metrics", "write_report_bundle"]
