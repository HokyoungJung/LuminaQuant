"""Backend payload helpers for the Next overview parity slice."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from lumina_quant.configuration import get_default_runtime_config
from lumina_quant.postgres_state import _connect_postgres
from lumina_quant.dashboard.workflow_jobs_service import load_recent_workflow_jobs
from lumina_quant.utils.performance import (
    create_annualized_volatility,
    create_cagr,
    create_calmar_ratio,
    create_drawdowns,
    create_sharpe_ratio,
    create_sortino_ratio,
)


def resolve_dashboard_postgres_dsn(dsn: str | None = None) -> str:
    return str(
        dsn
        or os.getenv("LQ_POSTGRES_DSN")
        or get_default_runtime_config().storage.postgres_dsn
        or ""
    ).strip()


def coerce_datetime_series(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame
    frame = frame.copy()
    frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    return frame


def empty_overview_payload(*, contract: Any, reason: str) -> dict[str, Any]:
    return {
        "as_of": datetime.now(UTC).isoformat(),
        "summary_metrics": [],
        "performance_metrics": {},
        "recent_runs": [],
        "workflow_jobs": [],
        "equity_curve": [],
        "drawdown_curve": [],
        "source": {
            "mode": contract.launch_mode,
            "backend": getattr(contract, "python_backend", "python"),
            "status": reason,
        },
    }


def overview_metric(label: str, value: Any, *, key: str | None = None) -> dict[str, Any]:
    return {
        "key": key or label.lower().replace(" ", "_"),
        "label": label,
        "value": value,
    }


def build_overview_payload_from_frames(
    *,
    contract: Any,
    runs_frame: pd.DataFrame,
    equity_frame: pd.DataFrame,
) -> dict[str, Any]:
    if runs_frame.empty:
        return empty_overview_payload(contract=contract, reason="no_runs")

    run = runs_frame.iloc[0]
    run_id = str(run.get("run_id") or "")
    strategy = ""
    metadata = run.get("metadata")
    if isinstance(metadata, dict):
        strategy = str(metadata.get("strategy") or "")
    strategy = strategy or str(run.get("strategy") or "")
    recent_runs = []
    for _, item in runs_frame.iterrows():
        started_at = item.get("started_at")
        started_at_value = None
        if pd.notna(started_at):
            started_at_value = pd.to_datetime(started_at, errors="coerce", utc=True)
            started_at_value = None if pd.isna(started_at_value) else started_at_value.isoformat()
        recent_runs.append(
            {
                "run_id": str(item.get("run_id") or ""),
                "mode": str(item.get("mode") or ""),
                "status": str(item.get("status") or ""),
                "strategy": str(item.get("strategy") or ""),
                "started_at": started_at_value,
            }
        )

    equity = coerce_datetime_series(equity_frame, "datetime")
    if equity.empty:
        summary_metrics = [
            overview_metric("Run ID", run_id, key="run_id"),
            overview_metric("Mode", str(run.get("mode") or "")),
            overview_metric("Status", str(run.get("status") or "")),
            overview_metric("Strategy", strategy or "unknown"),
            overview_metric("Equity Points", 0, key="equity_points"),
        ]
        payload = empty_overview_payload(contract=contract, reason="no_equity")
        payload["summary_metrics"] = summary_metrics
        payload["recent_runs"] = recent_runs
        payload["source"]["run_id"] = run_id
        return payload

    totals = pd.to_numeric(equity["total"], errors="coerce").fillna(0.0)
    timestamps = equity["datetime"]
    initial_equity = float(totals.iloc[0]) if not totals.empty else 0.0
    latest_equity = float(totals.iloc[-1]) if not totals.empty else 0.0
    total_return = 0.0
    if initial_equity:
        total_return = (latest_equity - initial_equity) / initial_equity

    running_peak = totals.cummax().replace(0.0, pd.NA)
    drawdown = ((totals - running_peak) / running_peak).fillna(0.0)
    prev = totals.to_numpy(dtype=float)[:-1]
    nxt = totals.to_numpy(dtype=float)[1:]
    returns = (
        pd.Series((nxt - prev) / pd.Series(prev).replace(0.0, 1.0).to_numpy(dtype=float))
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    periods = 252
    cagr = (
        create_cagr(latest_equity, initial_equity, len(totals), periods) if len(totals) > 1 else 0.0
    )
    annualized_vol = create_annualized_volatility(returns, periods) if returns.size else 0.0
    sharpe = create_sharpe_ratio(returns, periods=periods) if returns.size else 0.0
    sortino = create_sortino_ratio(returns, periods=periods) if returns.size else 0.0
    drawdown_levels, _ = create_drawdowns(totals.to_numpy(dtype=float))
    max_drawdown = max(drawdown_levels) if drawdown_levels else 0.0
    calmar = create_calmar_ratio(cagr, max_drawdown)

    summary_metrics = [
        overview_metric("Run ID", run_id, key="run_id"),
        overview_metric("Mode", str(run.get("mode") or "")),
        overview_metric("Status", str(run.get("status") or "")),
        overview_metric("Strategy", strategy or "unknown"),
        overview_metric("Initial Equity", round(initial_equity, 4), key="initial_equity"),
        overview_metric("Latest Equity", round(latest_equity, 4), key="latest_equity"),
        overview_metric("Total Return", round(total_return, 6), key="total_return"),
        overview_metric("Equity Points", len(equity), key="equity_points"),
    ]
    return {
        "as_of": datetime.now(UTC).isoformat(),
        "summary_metrics": summary_metrics,
        "performance_metrics": {
            "cagr": round(float(cagr), 6),
            "annualized_volatility": round(float(annualized_vol), 6),
            "sharpe_ratio": round(float(sharpe), 6),
            "sortino_ratio": round(float(sortino), 6),
            "calmar_ratio": round(float(calmar), 6),
            "max_drawdown": round(float(max_drawdown), 6),
        },
        "recent_runs": recent_runs,
        "equity_curve": [
            {
                "timestamp": value.isoformat(),
                "equity": float(total),
            }
            for value, total in zip(timestamps.tolist(), totals.tolist(), strict=False)
            if value is not pd.NaT
        ],
        "drawdown_curve": [
            {
                "timestamp": value.isoformat(),
                "drawdown": float(level),
            }
            for value, level in zip(timestamps.tolist(), drawdown.tolist(), strict=False)
            if value is not pd.NaT
        ],
        "source": {
            "mode": contract.launch_mode,
            "backend": getattr(contract, "python_backend", "python"),
            "status": "ok",
            "run_id": run_id,
        },
    }


def load_overview_payload(
    *,
    contract: Any,
    dsn: str | None = None,
    limit: int = 120,
    run_limit: int = 10,
) -> dict[str, Any]:
    if dsn is not None and not str(dsn).strip():
        return empty_overview_payload(contract=contract, reason="missing_dsn")
    resolved_dsn = resolve_dashboard_postgres_dsn(dsn)
    if not resolved_dsn:
        return empty_overview_payload(contract=contract, reason="missing_dsn")

    conn = _connect_postgres(resolved_dsn)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    run_id,
                    mode,
                    started_at,
                    ended_at,
                    status,
                    metadata_json AS metadata,
                    COALESCE(
                        (metadata_json ->> 'strategy'),
                        ''
                    ) AS strategy
                FROM runs
                ORDER BY started_at DESC
                LIMIT %s
                """,
                (int(max(1, run_limit)),),
            )
            run_rows = cursor.fetchall()
            run_columns = [description[0] for description in cursor.description or ()]
        runs = pd.DataFrame(run_rows, columns=run_columns)
        if runs.empty:
            return empty_overview_payload(contract=contract, reason="no_runs")
        workflow_jobs = load_recent_workflow_jobs(conn, limit=10)
        run_id = str(runs.iloc[0]["run_id"])
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT timeindex AS datetime, total
                FROM (
                    SELECT id, timeindex, total
                    FROM equity
                    WHERE run_id = %s
                    ORDER BY id DESC
                    LIMIT %s
                ) recent
                ORDER BY id ASC
                """,
                (run_id, int(max(10, limit))),
            )
            equity_rows = cursor.fetchall()
            equity_columns = [description[0] for description in cursor.description or ()]
        equity = pd.DataFrame(equity_rows, columns=equity_columns)
        payload = build_overview_payload_from_frames(
            contract=contract,
            runs_frame=runs,
            equity_frame=equity,
        )
        if workflow_jobs:
            payload["workflow_jobs"] = workflow_jobs
        return payload
    finally:
        conn.close()


__all__ = [
    "build_overview_payload_from_frames",
    "coerce_datetime_series",
    "empty_overview_payload",
    "load_overview_payload",
    "overview_metric",
    "resolve_dashboard_postgres_dsn",
]
