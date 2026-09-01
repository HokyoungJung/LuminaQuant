"""Backend payload helpers for the Next overview parity slice."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from lumina_quant.configuration import get_default_runtime_config
from lumina_quant.indicators.annualization import bars_per_year_from_spacing
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

# Headline metrics must describe the FULL run: load up to this many equity rows
# (newest rows win when the run is longer) instead of a 120-point tail.
FULL_EQUITY_ROW_CAP = 200_000
# Payload curves stay bounded: the full series is uniformly downsampled to at
# most this many points (first and last points always kept).
DEFAULT_CURVE_POINT_LIMIT = 600
# Cadence inference clamp — equity rows are 24/7 crypto-perp data, so the
# annualization factor is bounded by ~monthly rows and 1-minute rows on a
# 365.25-day year (365.25 * 24 * 60 = 525_960).
MIN_PERIODS_PER_YEAR = 12.0
MAX_PERIODS_PER_YEAR = 525_960.0
# Daily rows on the 24/7 calendar — used when timestamps are unusable.
FALLBACK_PERIODS_PER_YEAR = 365.25


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
        "equity_window": empty_equity_window(),
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


def empty_equity_window() -> dict[str, Any]:
    return {
        "start": None,
        "end": None,
        "points_total": 0,
        "points_returned": 0,
        "truncated": False,
    }


def infer_periods_per_year(timestamps: pd.Series) -> float:
    """Annualization factor inferred from equity-row cadence (24/7 calendar).

    Median positive spacing between timestamps drives the factor via the
    canonical ``lumina_quant.indicators.annualization`` helper; the result is
    clamped to [MIN_PERIODS_PER_YEAR, MAX_PERIODS_PER_YEAR] and falls back to
    FALLBACK_PERIODS_PER_YEAR when timestamps are unusable.
    """
    valid = pd.to_datetime(timestamps, errors="coerce", utc=True).dropna()
    if valid.empty:
        return FALLBACK_PERIODS_PER_YEAR
    epochs = valid.astype("int64").to_numpy(dtype="float64") / 1e9
    periods = bars_per_year_from_spacing(epochs)
    if periods is None:
        return FALLBACK_PERIODS_PER_YEAR
    return float(min(max(float(periods), MIN_PERIODS_PER_YEAR), MAX_PERIODS_PER_YEAR))


def downsample_curve_indices(values: Any, point_limit: int) -> list[int]:
    """Min-max bucket downsampling over ``values`` (index selection).

    Per bucket the indices of the bucket minimum AND maximum are kept, so
    extremes (drawdown troughs, equity spikes) survive the reduction and the
    rendered curve cannot contradict full-series headline tiles. First/last
    indices are always kept; the sorted unique result never exceeds
    ``point_limit``.
    """
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    points_total = len(series)
    if points_total <= 0:
        return []
    if point_limit < 2 or points_total <= point_limit:
        return list(range(points_total))
    # Endpoints consume 2 slots; each bucket contributes at most 2 indices.
    bucket_count = (int(point_limit) - 2) // 2
    if bucket_count < 1:
        return [0, points_total - 1]
    selected: set[int] = {0, points_total - 1}
    array = series.to_numpy(dtype=float)
    edges = np.linspace(1, points_total - 1, num=bucket_count + 1)
    for bucket in range(bucket_count):
        start = int(edges[bucket])
        stop = int(edges[bucket + 1])
        if stop <= start:
            continue
        window = array[start:stop]
        if np.isnan(window).all():
            selected.add(start)
            continue
        selected.add(start + int(np.nanargmin(window)))
        selected.add(start + int(np.nanargmax(window)))
    return sorted(selected)


def recent_runs_from_frame(runs_frame: pd.DataFrame) -> list[dict[str, Any]]:
    recent_runs: list[dict[str, Any]] = []
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
    return recent_runs


def build_overview_payload_from_frames(
    *,
    contract: Any,
    runs_frame: pd.DataFrame,
    equity_frame: pd.DataFrame,
    selected_run_row: Any = None,
    curve_point_limit: int = DEFAULT_CURVE_POINT_LIMIT,
    equity_truncated: bool = False,
) -> dict[str, Any]:
    if runs_frame.empty:
        return empty_overview_payload(contract=contract, reason="no_runs")

    run = selected_run_row if selected_run_row is not None else runs_frame.iloc[0]
    run_id = str(run.get("run_id") or "")
    strategy = ""
    metadata = run.get("metadata")
    if isinstance(metadata, dict):
        strategy = str(metadata.get("strategy") or "")
    strategy = strategy or str(run.get("strategy") or "")
    recent_runs = recent_runs_from_frame(runs_frame)

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
    periods = infer_periods_per_year(timestamps)
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
    curve_indices = downsample_curve_indices(totals, int(max(2, curve_point_limit)))
    curve_timestamps = timestamps.iloc[curve_indices].tolist()
    curve_totals = totals.iloc[curve_indices].tolist()
    curve_drawdowns = drawdown.iloc[curve_indices].tolist()
    valid_timestamps = timestamps.dropna()
    equity_window = {
        "start": None if valid_timestamps.empty else valid_timestamps.iloc[0].isoformat(),
        "end": None if valid_timestamps.empty else valid_timestamps.iloc[-1].isoformat(),
        "points_total": len(totals),
        "points_returned": len(curve_indices),
        "truncated": bool(equity_truncated),
    }
    return {
        "as_of": datetime.now(UTC).isoformat(),
        "summary_metrics": summary_metrics,
        # Contract-required on every status branch; load_overview_payload
        # overwrites this with the live jobs list.
        "workflow_jobs": [],
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
            for value, total in zip(curve_timestamps, curve_totals, strict=False)
            if value is not pd.NaT
        ],
        "drawdown_curve": [
            {
                "timestamp": value.isoformat(),
                "drawdown": float(level),
            }
            for value, level in zip(curve_timestamps, curve_drawdowns, strict=False)
            if value is not pd.NaT
        ],
        "equity_window": equity_window,
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
    limit: int = DEFAULT_CURVE_POINT_LIMIT,
    run_limit: int = 10,
    run_id: str | None = None,
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
        selected_run_id = str(run_id or "").strip()
        selected_row: Any = None
        if selected_run_id:
            matches = runs[runs["run_id"].astype(str) == selected_run_id]
            if not matches.empty:
                selected_row = matches.iloc[0]
            else:
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
                        WHERE run_id = %s
                        LIMIT 1
                        """,
                        (selected_run_id,),
                    )
                    selected_rows = cursor.fetchall()
                    selected_columns = [description[0] for description in cursor.description or ()]
                if not selected_rows:
                    payload = empty_overview_payload(contract=contract, reason="run_not_found")
                    payload["recent_runs"] = recent_runs_from_frame(runs)
                    payload["source"]["run_id"] = selected_run_id
                    # workflow_jobs is contract-required even when empty.
                    payload["workflow_jobs"] = workflow_jobs
                    return payload
                selected_row = pd.DataFrame(selected_rows, columns=selected_columns).iloc[0]
        target_run_id = selected_run_id or str(runs.iloc[0]["run_id"])
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
                (target_run_id, int(FULL_EQUITY_ROW_CAP)),
            )
            equity_rows = cursor.fetchall()
            equity_columns = [description[0] for description in cursor.description or ()]
        equity = pd.DataFrame(equity_rows, columns=equity_columns)
        payload = build_overview_payload_from_frames(
            contract=contract,
            runs_frame=runs,
            equity_frame=equity,
            selected_run_row=selected_row,
            curve_point_limit=int(max(2, limit)),
            equity_truncated=len(equity_rows) >= FULL_EQUITY_ROW_CAP,
        )
        # workflow_jobs is contract-required even when empty ([]): the Next
        # runtime reads payload.workflow_jobs unguarded on every success shape.
        payload["workflow_jobs"] = workflow_jobs
        return payload
    finally:
        conn.close()


__all__ = [
    "DEFAULT_CURVE_POINT_LIMIT",
    "FULL_EQUITY_ROW_CAP",
    "build_overview_payload_from_frames",
    "coerce_datetime_series",
    "downsample_curve_indices",
    "empty_equity_window",
    "empty_overview_payload",
    "infer_periods_per_year",
    "load_overview_payload",
    "overview_metric",
    "recent_runs_from_frame",
    "resolve_dashboard_postgres_dsn",
]
