"""Python-backed payload builders for the Next dashboard cutover-prep surfaces."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from lumina_quant.configuration import get_default_runtime_config
from lumina_quant.dashboard.state_store_service import (
    load_fills_state_frame,
    load_heartbeats_state_frame,
    load_market_ohlcv_frame,
    load_metrics_state_frame,
    load_optimization_results_state_frame,
    load_order_states_state_frame,
    load_orders_state_frame,
    load_risk_events_state_frame,
    load_runs_frame,
)
from lumina_quant.dashboard.workflow_jobs_service import load_recent_workflow_jobs
from lumina_quant.dashboard.overview_service import (
    build_overview_payload_from_frames,
    coerce_datetime_series,
    overview_metric,
    resolve_dashboard_postgres_dsn,
)
from lumina_quant.market_data import (
    normalize_symbol,
    normalize_timeframe_token,
    timeframe_to_milliseconds,
)
from lumina_quant.postgres_state import _connect_postgres


def _dashboard_contract() -> Any:
    from lumina_quant.dashboard.bridge import build_dashboard_bridge_contract_v2

    return build_dashboard_bridge_contract_v2()


def _parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except TypeError, ValueError:
        return float(default)
    return parsed if pd.notna(parsed) else float(default)


def _isoformat(value: Any) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.isoformat()


def _normalize_market_timeframe(value: Any) -> tuple[str, bool]:
    try:
        token = normalize_timeframe_token(str(value or "1m"))
    except Exception:
        return "1m", True
    try:
        if int(timeframe_to_milliseconds(token)) < 60_000:
            return "1m", True
    except Exception:
        return "1m", True
    return token, False


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item") and callable(value.item):
        try:
            return _normalize_json_value(value.item())
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _frame_preview(
    frame: pd.DataFrame, *, row_limit: int = 5, column_limit: int = 8
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    preview = frame.copy()
    preview = preview.loc[:, list(preview.columns[:column_limit])]
    return [
        {str(key): _normalize_json_value(value) for key, value in row.items()}
        for row in preview.head(row_limit).to_dict(orient="records")
    ]


def _frame_summary(label: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "label": label,
        "rows": len(frame.index),
        "columns": len(frame.columns),
    }


def _frame_preview_payload(label: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "label": label,
        "columns": [str(column) for column in frame.columns.tolist()[:8]],
        "rows": _frame_preview(frame),
    }


def _resolve_market_context(
    *,
    run_row: dict[str, Any],
    fills_frame: pd.DataFrame,
) -> dict[str, Any]:
    metadata = _parse_json_dict(run_row.get("metadata"))
    configured_symbols = get_default_runtime_config().trading.symbols
    symbol = ""
    if "symbol" in fills_frame.columns:
        symbol_values = fills_frame["symbol"].dropna().astype(str)
        if not symbol_values.empty:
            symbol = str(symbol_values.iloc[-1])
    if not symbol:
        metadata_symbols = metadata.get("symbols")
        if isinstance(metadata_symbols, list) and metadata_symbols:
            symbol = str(metadata_symbols[0])
    if not symbol and isinstance(configured_symbols, list) and configured_symbols:
        symbol = str(configured_symbols[0])

    timeframe, clamped = _normalize_market_timeframe(
        metadata.get("timeframe") or get_default_runtime_config().trading.timeframe
    )
    market_db_path = str(
        get_default_runtime_config().storage.market_data_parquet_path or ""
    ).strip()
    return {
        "symbol": symbol or "n/a",
        "timeframe": timeframe,
        "timeframe_clamped": clamped,
        "exchange": str(get_default_runtime_config().storage.market_data_exchange or "binance"),
        "strategy": str(run_row.get("strategy") or metadata.get("strategy") or "unknown"),
        "market_db_path": market_db_path,
        "source": "parquet" if market_db_path else "unconfigured",
    }


def _empty_surface_payload(*, run_id: str = "", reason: str) -> dict[str, Any]:
    return {
        "as_of": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "status": reason,
    }


def compute_trade_analytics(df_trades: pd.DataFrame) -> pd.DataFrame:
    if df_trades.empty:
        return df_trades.copy()

    df = df_trades.copy().sort_values("datetime").reset_index(drop=True)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    for column in ("quantity", "price", "fill_cost", "commission"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
        else:
            df[column] = 0.0
    if "direction" not in df.columns:
        df["direction"] = ""
    if "symbol" not in df.columns:
        df["symbol"] = "UNKNOWN"

    positions: dict[str, float] = {}
    avg_cost: dict[str, float] = {}
    entry_times: dict[str, Any] = {}
    realized_pnl: list[float] = []
    realized_return: list[float] = []
    position_after: list[float] = []
    avg_cost_after: list[float] = []
    closed_qty: list[float] = []
    close_side: list[str | None] = []
    holding_sec: list[float] = []

    for _, row in df.iterrows():
        symbol = str(row["symbol"])
        qty = float(row["quantity"])
        price = float(row["price"])
        commission = float(row["commission"])
        direction = str(row["direction"]).upper()
        signed = qty if direction == "BUY" else -qty
        event_time = row.get("datetime", pd.NaT)
        event_time = event_time if pd.notna(event_time) else pd.NaT

        pos = float(positions.get(symbol, 0.0))
        avg = float(avg_cost.get(symbol, 0.0))
        entry_time = entry_times.get(symbol)

        pnl = 0.0
        ret = float("nan")
        closes = 0.0
        close_label = None
        hold_seconds = float("nan")

        if pos == 0 or (pos > 0 and signed > 0) or (pos < 0 and signed < 0):
            new_pos = pos + signed
            if new_pos != 0:
                if pos == 0:
                    new_avg = price
                else:
                    new_avg = ((abs(pos) * avg) + (abs(signed) * price)) / abs(new_pos)
            else:
                new_avg = 0.0
            if pos == 0 and new_pos != 0:
                new_entry_time = event_time
            elif new_pos == 0:
                new_entry_time = None
            else:
                new_entry_time = entry_time
        else:
            closes = min(abs(pos), abs(signed))
            if pos > 0 and signed < 0:
                pnl = (price - avg) * closes - commission
                close_label = "LONG"
            elif pos < 0 and signed > 0:
                pnl = (avg - price) * closes - commission
                close_label = "SHORT"

            basis = max(abs(avg * closes), abs(price * closes))
            if basis > 1e-12:
                ret = pnl / basis

            if pd.notna(event_time) and entry_time is not None and pd.notna(entry_time):
                hold_seconds = max(0.0, float((event_time - entry_time).total_seconds()))

            new_pos = pos + signed
            if new_pos == 0:
                new_avg = 0.0
                new_entry_time = None
            elif (pos > 0 and new_pos > 0) or (pos < 0 and new_pos < 0):
                new_avg = avg
                new_entry_time = entry_time
            else:
                new_avg = price
                new_entry_time = event_time

        positions[symbol] = new_pos
        avg_cost[symbol] = new_avg
        entry_times[symbol] = new_entry_time
        realized_pnl.append(pnl)
        realized_return.append(ret)
        position_after.append(new_pos)
        avg_cost_after.append(new_avg)
        closed_qty.append(closes)
        close_side.append(close_label)
        holding_sec.append(hold_seconds)

    df["realized_pnl"] = realized_pnl
    df["realized_return_pct"] = pd.Series(realized_return, dtype="float64") * 100.0
    df["position_after"] = position_after
    df["avg_cost_after"] = avg_cost_after
    df["closed_qty"] = closed_qty
    df["close_side"] = close_side
    df["holding_sec"] = holding_sec
    df["cum_realized_pnl"] = df["realized_pnl"].cumsum()
    df["notional"] = df["quantity"] * df["price"]
    return df


def _closed_trade_analytics(trade_analytics: pd.DataFrame) -> pd.DataFrame:
    if trade_analytics.empty:
        return trade_analytics
    if "closed_qty" in trade_analytics.columns:
        return trade_analytics[trade_analytics["closed_qty"] > 0].copy()
    return trade_analytics[trade_analytics["realized_pnl"] != 0].copy()


def _streak_groups(outcomes: list[bool]) -> list[tuple[bool, int]]:
    if not outcomes:
        return []
    groups: list[tuple[bool, int]] = []
    current = outcomes[0]
    length = 1
    for outcome in outcomes[1:]:
        if outcome == current:
            length += 1
            continue
        groups.append((current, length))
        current = outcome
        length = 1
    groups.append((current, length))
    return groups


def _load_runs(dsn: str, *, run_limit: int) -> pd.DataFrame:
    return load_runs_frame(
        dsn,
        coerce_datetime=coerce_datetime_series,
        limit=run_limit,
    )


def _load_metrics(dsn: str, run_id: str, *, point_limit: int) -> pd.DataFrame:
    return load_metrics_state_frame(
        dsn,
        run_id,
        coerce_datetime=coerce_datetime_series,
        parse_json_dict=_parse_json_dict,
        max_points=point_limit,
    )


def _load_fills(dsn: str, run_id: str, *, fill_limit: int) -> pd.DataFrame:
    return load_fills_state_frame(
        dsn,
        run_id,
        coerce_datetime=coerce_datetime_series,
        max_points=fill_limit,
    )


def _load_orders(dsn: str, run_id: str, *, order_limit: int) -> pd.DataFrame:
    return load_orders_state_frame(
        dsn,
        run_id,
        coerce_datetime=coerce_datetime_series,
        max_points=order_limit,
    )


def _load_risk_events(dsn: str, run_id: str, *, limit: int) -> pd.DataFrame:
    return load_risk_events_state_frame(
        dsn,
        run_id,
        coerce_datetime=coerce_datetime_series,
        max_points=limit,
    )


def _load_heartbeats(dsn: str, run_id: str, *, limit: int) -> pd.DataFrame:
    return load_heartbeats_state_frame(
        dsn,
        run_id,
        coerce_datetime=coerce_datetime_series,
        max_points=limit,
    )


def _load_order_states(dsn: str, run_id: str, *, limit: int) -> pd.DataFrame:
    return load_order_states_state_frame(
        dsn,
        run_id,
        coerce_datetime=coerce_datetime_series,
        max_points=limit,
    )


def _load_optimization_results(dsn: str, *, point_limit: int) -> pd.DataFrame:
    return load_optimization_results_state_frame(
        dsn,
        resolve_postgres_dsn=resolve_dashboard_postgres_dsn,
        coerce_datetime=coerce_datetime_series,
        parse_json_dict=_parse_json_dict,
        max_points=point_limit,
    )


def _load_market(
    *,
    market_db_path: str,
    symbol: str,
    timeframe: str,
    exchange: str,
    point_limit: int,
) -> pd.DataFrame:
    if not market_db_path.strip() or not symbol.strip():
        return pd.DataFrame()
    return load_market_ohlcv_frame(
        market_db_path,
        symbol,
        timeframe,
        exchange,
        normalize_symbol=normalize_symbol,
        resolve_dashboard_market_timeframe=_normalize_market_timeframe,
        timeframe_to_milliseconds=timeframe_to_milliseconds,
        coerce_datetime=coerce_datetime_series,
        max_points=point_limit,
    )


def _load_recent_workflow_jobs_frame(dsn: str, *, limit: int) -> pd.DataFrame:
    conn = _connect_postgres(dsn)
    try:
        return pd.DataFrame(load_recent_workflow_jobs(conn, limit=limit))
    finally:
        conn.close()


def build_performance_price_payload(
    *,
    overview_payload: dict[str, Any],
    metrics_frame: pd.DataFrame,
    fills_frame: pd.DataFrame,
) -> dict[str, Any]:
    payload = {
        **_empty_surface_payload(
            run_id=str(overview_payload.get("source", {}).get("run_id") or ""),
            reason=str(overview_payload.get("source", {}).get("status") or "unknown"),
        ),
        "source": overview_payload.get("source", {}),
        "summary_metrics": overview_payload.get("summary_metrics", []),
        "performance_metrics": overview_payload.get("performance_metrics", {}),
        "equity_curve": overview_payload.get("equity_curve", []),
        "drawdown_curve": overview_payload.get("drawdown_curve", []),
        "benchmark_curve": [],
        "funding_curve": [],
        "trade_markers": [],
    }
    if payload["status"] != "ok":
        return payload

    if not metrics_frame.empty:
        benchmark_series = pd.to_numeric(metrics_frame.get("benchmark_price"), errors="coerce")
        funding_series = pd.to_numeric(metrics_frame.get("funding"), errors="coerce").fillna(0.0)
        payload["benchmark_curve"] = [
            {
                "timestamp": _isoformat(row.datetime),
                "price": float(row.benchmark_price),
            }
            for row in metrics_frame.assign(benchmark_price=benchmark_series).itertuples(
                index=False
            )
            if pd.notna(row.benchmark_price) and _isoformat(row.datetime) is not None
        ]
        payload["funding_curve"] = [
            {
                "timestamp": _isoformat(row.datetime),
                "funding": float(row.funding),
            }
            for row in metrics_frame.assign(funding=funding_series).itertuples(index=False)
            if _isoformat(row.datetime) is not None
        ]

    trade_analytics = compute_trade_analytics(fills_frame)
    payload["trade_markers"] = [
        {
            "timestamp": _isoformat(row.datetime),
            "symbol": str(row.symbol),
            "direction": str(row.direction),
            "price": _safe_float(row.price),
            "quantity": _safe_float(row.quantity),
            "realized_pnl": _safe_float(row.realized_pnl),
            "realized_return_pct": None
            if pd.isna(row.realized_return_pct)
            else _safe_float(row.realized_return_pct),
            "position_after": _safe_float(row.position_after),
        }
        for row in trade_analytics.tail(12).itertuples(index=False)
        if _isoformat(row.datetime) is not None
    ]
    return payload


def build_execution_analytics_payload(
    *,
    run_id: str,
    fills_frame: pd.DataFrame,
    orders_frame: pd.DataFrame,
) -> dict[str, Any]:
    payload = {
        **_empty_surface_payload(run_id=run_id, reason="ok"),
        "summary": {
            "buy_fills": 0,
            "sell_fills": 0,
            "avg_qty": 0.0,
            "avg_notional": 0.0,
            "total_commission": 0.0,
            "avg_trade_return_pct": 0.0,
            "best_trade_pnl": 0.0,
            "worst_trade_pnl": 0.0,
            "win_streak_max": 0,
            "loss_streak_max": 0,
            "win_streak_avg": 0.0,
            "loss_streak_avg": 0.0,
            "holding_time_avg_sec": 0.0,
            "long_trades": 0,
            "long_win_rate": 0.0,
            "short_trades": 0,
            "short_win_rate": 0.0,
            "order_count": len(orders_frame.index),
            "closed_trade_count": 0,
        },
        "direction_breakdown": [],
        "order_status": [],
        "recent_closed_trades": [],
    }

    if fills_frame.empty and orders_frame.empty:
        payload["status"] = "no_execution_data"
        return payload

    trade_analytics = compute_trade_analytics(fills_frame)
    closed = _closed_trade_analytics(trade_analytics)
    summary = payload["summary"]
    if not fills_frame.empty:
        direction = (
            fills_frame.get("direction", pd.Series(dtype="object"))
            .fillna("")
            .astype(str)
            .str.upper()
        )
        summary["buy_fills"] = int((direction == "BUY").sum())
        summary["sell_fills"] = int((direction == "SELL").sum())
        summary["avg_qty"] = round(
            _safe_float(fills_frame.get("quantity", pd.Series(dtype="float64")).mean()), 6
        )
        summary["avg_notional"] = round(
            _safe_float(trade_analytics.get("notional", pd.Series(dtype="float64")).mean()), 6
        )
        summary["total_commission"] = round(
            _safe_float(fills_frame.get("commission", pd.Series(dtype="float64")).sum()),
            6,
        )

    if not closed.empty:
        returns = pd.to_numeric(closed["realized_return_pct"], errors="coerce").dropna()
        pnls = pd.to_numeric(closed["realized_pnl"], errors="coerce").fillna(0.0)
        decisive = pnls[pnls != 0.0]
        summary["closed_trade_count"] = len(closed.index)
        summary["avg_trade_return_pct"] = round(_safe_float(returns.mean()), 6)
        summary["best_trade_pnl"] = round(_safe_float(pnls.max()), 6)
        summary["worst_trade_pnl"] = round(_safe_float(pnls.min()), 6)
        summary["holding_time_avg_sec"] = round(
            _safe_float(pd.to_numeric(closed["holding_sec"], errors="coerce").dropna().mean()),
            6,
        )

        streaks = _streak_groups((decisive > 0.0).tolist())
        wins = [length for is_win, length in streaks if is_win]
        losses = [length for is_win, length in streaks if not is_win]
        summary["win_streak_max"] = max(wins, default=0)
        summary["loss_streak_max"] = max(losses, default=0)
        summary["win_streak_avg"] = round(sum(wins) / len(wins), 6) if wins else 0.0
        summary["loss_streak_avg"] = round(sum(losses) / len(losses), 6) if losses else 0.0

        for label, close_side in (("Long", "LONG"), ("Short", "SHORT")):
            part = closed[closed["close_side"] == close_side]
            trade_count = len(part.index)
            win_rate = 0.0
            if trade_count:
                win_rate = float(
                    (pd.to_numeric(part["realized_pnl"], errors="coerce").fillna(0.0) > 0.0).mean()
                )
            payload["direction_breakdown"].append(
                {
                    "direction": label,
                    "closed_trades": trade_count,
                    "win_rate": round(win_rate, 6),
                }
            )
            if label == "Long":
                summary["long_trades"] = trade_count
                summary["long_win_rate"] = round(win_rate, 6)
            else:
                summary["short_trades"] = trade_count
                summary["short_win_rate"] = round(win_rate, 6)

        payload["recent_closed_trades"] = [
            {
                "timestamp": _isoformat(row.datetime),
                "symbol": str(row.symbol),
                "close_side": str(row.close_side or "n/a"),
                "realized_pnl": _safe_float(row.realized_pnl),
                "realized_return_pct": None
                if pd.isna(row.realized_return_pct)
                else _safe_float(row.realized_return_pct),
                "holding_sec": None if pd.isna(row.holding_sec) else _safe_float(row.holding_sec),
            }
            for row in closed.tail(10).itertuples(index=False)
            if _isoformat(row.datetime) is not None
        ]

    if not orders_frame.empty:
        order_status = (
            orders_frame.get("status", pd.Series(dtype="object"))
            .fillna("UNKNOWN")
            .astype(str)
            .value_counts()
        )
        payload["order_status"] = [
            {"status": str(status), "count": int(count)} for status, count in order_status.items()
        ]

    return payload


def build_market_data_payload(
    *,
    run_row: dict[str, Any],
    fills_frame: pd.DataFrame,
    market_frame: pd.DataFrame,
) -> dict[str, Any]:
    run_id = str(run_row.get("run_id") or "")
    context = _resolve_market_context(run_row=run_row, fills_frame=fills_frame)
    payload = {
        **_empty_surface_payload(run_id=run_id, reason="ok"),
        "market_context": context,
        "summary_metrics": [],
        "recent_bars": [],
        "indicator_summary": [
            overview_metric("Strategy", context["strategy"], key="strategy"),
            overview_metric(
                "Indicator Mode",
                "price-only parity preview",
                key="indicator_mode",
            ),
        ],
        "warnings": [],
    }
    if market_frame.empty:
        payload["status"] = "no_market_data"
        payload["warnings"].append(
            "No market OHLCV rows were available for the configured symbol/timeframe/exchange."
        )
        return payload

    close_series = pd.to_numeric(market_frame.get("close"), errors="coerce")
    volume_series = pd.to_numeric(market_frame.get("volume"), errors="coerce")
    high_series = pd.to_numeric(market_frame.get("high"), errors="coerce")
    low_series = pd.to_numeric(market_frame.get("low"), errors="coerce")
    latest_close = close_series.dropna().iloc[-1] if close_series.notna().any() else None
    first_close = close_series.dropna().iloc[0] if close_series.notna().any() else None
    price_change_pct = None
    if first_close not in (None, 0):
        price_change_pct = (_safe_float(latest_close) - float(first_close)) / float(first_close)

    payload["summary_metrics"] = [
        overview_metric("Market Bars", len(market_frame.index), key="market_bars"),
        overview_metric(
            "Latest Close",
            None if latest_close is None else round(float(latest_close), 6),
            key="latest_close",
        ),
        overview_metric(
            "Latest Volume",
            None
            if volume_series.dropna().empty
            else round(float(volume_series.dropna().iloc[-1]), 6),
            key="latest_volume",
        ),
        overview_metric(
            "Price Change %",
            None if price_change_pct is None else round(float(price_change_pct), 6),
            key="price_change_pct",
        ),
    ]
    payload["indicator_summary"].extend(
        [
            overview_metric(
                "Price Range",
                (
                    "n/a"
                    if high_series.dropna().empty or low_series.dropna().empty
                    else f"{float(low_series.min()):.4f} - {float(high_series.max()):.4f}"
                ),
                key="price_range",
            ),
            overview_metric(
                "Timeframe Clamped",
                "yes" if context["timeframe_clamped"] else "no",
                key="timeframe_clamped",
            ),
        ]
    )
    payload["recent_bars"] = [
        {
            "timestamp": _isoformat(row.datetime),
            "open": _normalize_json_value(getattr(row, "open", None)),
            "high": _normalize_json_value(getattr(row, "high", None)),
            "low": _normalize_json_value(getattr(row, "low", None)),
            "close": _normalize_json_value(getattr(row, "close", None)),
            "volume": _normalize_json_value(getattr(row, "volume", None)),
        }
        for row in market_frame.tail(24).itertuples(index=False)
        if _isoformat(row.datetime) is not None
    ]
    return payload


def build_optimization_insights_payload(
    *,
    run_id: str,
    optimization_frame: pd.DataFrame,
) -> dict[str, Any]:
    payload = {
        **_empty_surface_payload(run_id=run_id, reason="ok"),
        "summary_metrics": [],
        "stage_breakdown": [],
        "top_candidates": [],
        "best_candidate": None,
    }
    if optimization_frame.empty:
        payload["status"] = "no_optimization_results"
        return payload

    sharpe = pd.to_numeric(optimization_frame.get("sharpe"), errors="coerce")
    robustness = pd.to_numeric(optimization_frame.get("robustness_score"), errors="coerce")
    stages = (
        optimization_frame.get("stage", pd.Series(dtype="object")).fillna("unknown").astype(str)
    )
    payload["summary_metrics"] = [
        overview_metric("Rows", len(optimization_frame.index), key="rows"),
        overview_metric(
            "Best Sharpe",
            None if sharpe.dropna().empty else round(float(sharpe.max()), 6),
            key="best_sharpe",
        ),
        overview_metric(
            "Median Sharpe",
            None if sharpe.dropna().empty else round(float(sharpe.median()), 6),
            key="median_sharpe",
        ),
        overview_metric(
            "Median Robustness",
            None if robustness.dropna().empty else round(float(robustness.median()), 6),
            key="median_robustness",
        ),
        overview_metric("Stages", stages.nunique(), key="stage_count"),
    ]

    for stage_name, group in optimization_frame.assign(stage=stages).groupby("stage", dropna=False):
        stage_sharpe = pd.to_numeric(group.get("sharpe"), errors="coerce")
        stage_robustness = pd.to_numeric(group.get("robustness_score"), errors="coerce")
        payload["stage_breakdown"].append(
            {
                "stage": str(stage_name),
                "count": len(group.index),
                "median_sharpe": (
                    None if stage_sharpe.dropna().empty else round(float(stage_sharpe.median()), 6)
                ),
                "median_robustness": (
                    None
                    if stage_robustness.dropna().empty
                    else round(float(stage_robustness.median()), 6)
                ),
            }
        )

    ordered = optimization_frame.assign(_sharpe=sharpe).sort_values(
        by=["_sharpe", "created_at"],
        ascending=[False, False],
        na_position="last",
    )
    candidate_records = []
    for row in ordered.head(12).itertuples(index=False):
        candidate_records.append(
            {
                "created_at": _isoformat(getattr(row, "created_at", None)),
                "run_id": str(getattr(row, "run_id", "") or ""),
                "stage": str(getattr(row, "stage", "") or ""),
                "sharpe": _normalize_json_value(getattr(row, "sharpe", None)),
                "train_sharpe": _normalize_json_value(getattr(row, "train_sharpe", None)),
                "robustness_score": _normalize_json_value(getattr(row, "robustness_score", None)),
                "cagr": _normalize_json_value(getattr(row, "cagr", None)),
                "mdd": _normalize_json_value(getattr(row, "mdd", None)),
                "params": _normalize_json_value(getattr(row, "params", {})),
            }
        )
    payload["top_candidates"] = candidate_records
    payload["best_candidate"] = candidate_records[0] if candidate_records else None
    return payload


def build_raw_data_payload(
    *,
    run_id: str,
    context: dict[str, Any],
    frames: list[tuple[str, pd.DataFrame]],
) -> dict[str, Any]:
    return {
        **_empty_surface_payload(run_id=run_id, reason="ok"),
        "context": context,
        "frame_summaries": [_frame_summary(label, frame) for label, frame in frames],
        "previews": [_frame_preview_payload(label, frame) for label, frame in frames],
    }


def _build_markdown_report(report: dict[str, Any], cutover_gate: dict[str, Any]) -> str:
    evidence = "\n".join(f"- {item}" for item in cutover_gate["evidence"])
    return (
        f"# {report['title']}\n\n"
        f"- Generated: {report['generated_at']}\n"
        f"- Run ID: {report['run_id'] or 'n/a'}\n"
        f"- Strategy: {report['strategy'] or 'unknown'}\n"
        f"- Total Return: {report['total_return']}\n"
        f"- Latest Equity: {report['latest_equity']}\n"
        f"- Realized PnL: {report['realized_pnl']}\n"
        f"- Closed Trades: {report['closed_trade_count']}\n"
        f"- Risk Events: {report['risk_event_count']}\n"
        f"- Heartbeats: {report['heartbeat_count']}\n"
        f"- Default Launcher: {cutover_gate['default_launcher']}\n\n"
        f"## Cutover Gate Evidence\n{evidence}\n"
    )


def build_report_export_payload(
    *,
    run_row: dict[str, Any],
    overview_payload: dict[str, Any],
    fills_frame: pd.DataFrame,
    risk_frame: pd.DataFrame,
    heartbeats_frame: pd.DataFrame,
) -> dict[str, Any]:
    run_id = str(run_row.get("run_id") or "")
    status = str(overview_payload.get("source", {}).get("status") or "unknown")
    payload = {
        **_empty_surface_payload(run_id=run_id, reason=status),
        "filenames": {
            "json": "luminaquant-dashboard-report.json",
            "markdown": "luminaquant-dashboard-report.md",
        },
        "json_report": {},
        "markdown_report": "",
        "cutover_gate": {
            "default_launcher": "next",
            "status": "available",
            "evidence": [
                "Performance & Price route is available in Next.js.",
                "Execution Analytics route is available in Next.js.",
                "Market Data route is available in Next.js.",
                "Optimization Insights route is available in Next.js.",
                "Raw Data route is available in Next.js.",
                "Report Export route is available in Next.js.",
                "Next is now the default launcher and the retired legacy entrypoint only remains as an explicit compatibility stub.",
            ],
        },
    }
    if status != "ok":
        return payload

    trade_analytics = compute_trade_analytics(fills_frame)
    closed = _closed_trade_analytics(trade_analytics)
    latest_equity = None
    total_return = None
    for metric in overview_payload.get("summary_metrics", []):
        key = str(metric.get("key") or "")
        if key == "latest_equity":
            latest_equity = metric.get("value")
        elif key == "total_return":
            total_return = metric.get("value")

    generated_at = datetime.now(UTC).isoformat()
    report = {
        "title": "LuminaQuant Dashboard Snapshot",
        "generated_at": generated_at,
        "run_id": run_id,
        "strategy": str(run_row.get("strategy") or "unknown"),
        "mode": str(run_row.get("mode") or ""),
        "status": str(run_row.get("status") or ""),
        "period_start": _isoformat(run_row.get("started_at")),
        "period_end": overview_payload.get("equity_curve", [{}])[-1].get("timestamp")
        if overview_payload.get("equity_curve")
        else None,
        "total_return": total_return,
        "latest_equity": latest_equity,
        "realized_pnl": round(
            _safe_float(closed.get("realized_pnl", pd.Series(dtype="float64")).sum()), 6
        ),
        "closed_trade_count": len(closed.index),
        "risk_event_count": len(risk_frame.index),
        "heartbeat_count": len(heartbeats_frame.index),
        "performance_metrics": overview_payload.get("performance_metrics", {}),
    }
    date_prefix = generated_at[:10]
    run_token = run_id or "no-run"
    payload["filenames"] = {
        "json": f"{date_prefix}-{run_token}-dashboard-report.json",
        "markdown": f"{date_prefix}-{run_token}-dashboard-report.md",
    }
    payload["json_report"] = report
    payload["markdown_report"] = _build_markdown_report(report, payload["cutover_gate"])
    return payload


def load_performance_price_payload(
    *,
    dsn: str | None = None,
    point_limit: int = 240,
    fill_limit: int = 80,
) -> dict[str, Any]:
    if dsn is not None and not str(dsn).strip():
        return {
            **_empty_surface_payload(reason="missing_dsn"),
            "source": {"status": "missing_dsn"},
            "summary_metrics": [],
            "performance_metrics": {},
            "equity_curve": [],
            "drawdown_curve": [],
            "benchmark_curve": [],
            "funding_curve": [],
            "trade_markers": [],
        }
    resolved_dsn = resolve_dashboard_postgres_dsn(dsn)
    if not resolved_dsn:
        return {
            **_empty_surface_payload(reason="missing_dsn"),
            "source": {"status": "missing_dsn"},
            "summary_metrics": [],
            "performance_metrics": {},
            "equity_curve": [],
            "drawdown_curve": [],
            "benchmark_curve": [],
            "funding_curve": [],
            "trade_markers": [],
        }

    runs = _load_runs(resolved_dsn, run_limit=10)
    if runs.empty:
        return {
            **_empty_surface_payload(reason="no_runs"),
            "source": {"status": "no_runs"},
            "summary_metrics": [],
            "performance_metrics": {},
            "equity_curve": [],
            "drawdown_curve": [],
            "benchmark_curve": [],
            "funding_curve": [],
            "trade_markers": [],
        }

    run_id = str(runs.iloc[0]["run_id"] or "")
    metrics = _load_metrics(resolved_dsn, run_id, point_limit=point_limit)
    fills = _load_fills(resolved_dsn, run_id, fill_limit=fill_limit)
    overview_payload = build_overview_payload_from_frames(
        contract=_dashboard_contract(),
        runs_frame=runs,
        equity_frame=metrics[["datetime", "total"]].copy()
        if {"datetime", "total"}.issubset(metrics.columns)
        else pd.DataFrame(),
    )
    return build_performance_price_payload(
        overview_payload=overview_payload,
        metrics_frame=metrics,
        fills_frame=fills,
    )


def load_execution_analytics_payload(
    *,
    dsn: str | None = None,
    fill_limit: int = 200,
    order_limit: int = 200,
) -> dict[str, Any]:
    if dsn is not None and not str(dsn).strip():
        return {
            **_empty_surface_payload(reason="missing_dsn"),
            "summary": {},
            "direction_breakdown": [],
            "order_status": [],
            "recent_closed_trades": [],
        }
    resolved_dsn = resolve_dashboard_postgres_dsn(dsn)
    if not resolved_dsn:
        return {
            **_empty_surface_payload(reason="missing_dsn"),
            "summary": {},
            "direction_breakdown": [],
            "order_status": [],
            "recent_closed_trades": [],
        }
    runs = _load_runs(resolved_dsn, run_limit=1)
    if runs.empty:
        return {
            **_empty_surface_payload(reason="no_runs"),
            "summary": {},
            "direction_breakdown": [],
            "order_status": [],
            "recent_closed_trades": [],
        }
    run_id = str(runs.iloc[0]["run_id"] or "")
    fills = _load_fills(resolved_dsn, run_id, fill_limit=fill_limit)
    orders = _load_orders(resolved_dsn, run_id, order_limit=order_limit)
    return build_execution_analytics_payload(
        run_id=run_id,
        fills_frame=fills,
        orders_frame=orders,
    )


def load_report_export_payload(
    *,
    dsn: str | None = None,
    point_limit: int = 240,
    fill_limit: int = 200,
    event_limit: int = 50,
) -> dict[str, Any]:
    if dsn is not None and not str(dsn).strip():
        return {
            **_empty_surface_payload(reason="missing_dsn"),
            "filenames": {},
            "json_report": {},
            "markdown_report": "",
            "cutover_gate": {},
        }
    resolved_dsn = resolve_dashboard_postgres_dsn(dsn)
    if not resolved_dsn:
        return {
            **_empty_surface_payload(reason="missing_dsn"),
            "filenames": {},
            "json_report": {},
            "markdown_report": "",
            "cutover_gate": {},
        }
    runs = _load_runs(resolved_dsn, run_limit=10)
    if runs.empty:
        return {
            **_empty_surface_payload(reason="no_runs"),
            "filenames": {},
            "json_report": {},
            "markdown_report": "",
            "cutover_gate": {},
        }
    run_row = runs.iloc[0].to_dict()
    run_id = str(run_row.get("run_id") or "")
    metrics = _load_metrics(resolved_dsn, run_id, point_limit=point_limit)
    fills = _load_fills(resolved_dsn, run_id, fill_limit=fill_limit)
    risk = _load_risk_events(resolved_dsn, run_id, limit=event_limit)
    heartbeats = _load_heartbeats(resolved_dsn, run_id, limit=event_limit)
    overview_payload = build_overview_payload_from_frames(
        contract=_dashboard_contract(),
        runs_frame=runs,
        equity_frame=metrics[["datetime", "total"]].copy()
        if {"datetime", "total"}.issubset(metrics.columns)
        else pd.DataFrame(),
    )
    return build_report_export_payload(
        run_row=run_row,
        overview_payload=overview_payload,
        fills_frame=fills,
        risk_frame=risk,
        heartbeats_frame=heartbeats,
    )


def load_market_data_payload(
    *,
    dsn: str | None = None,
    point_limit: int = 240,
    fill_limit: int = 80,
) -> dict[str, Any]:
    if dsn is not None and not str(dsn).strip():
        return {
            **_empty_surface_payload(reason="missing_dsn"),
            "market_context": {},
            "summary_metrics": [],
            "recent_bars": [],
            "indicator_summary": [],
            "warnings": [],
        }
    resolved_dsn = resolve_dashboard_postgres_dsn(dsn)
    if not resolved_dsn:
        return {
            **_empty_surface_payload(reason="missing_dsn"),
            "market_context": {},
            "summary_metrics": [],
            "recent_bars": [],
            "indicator_summary": [],
            "warnings": [],
        }
    runs = _load_runs(resolved_dsn, run_limit=10)
    if runs.empty:
        return {
            **_empty_surface_payload(reason="no_runs"),
            "market_context": {},
            "summary_metrics": [],
            "recent_bars": [],
            "indicator_summary": [],
            "warnings": [],
        }

    run_row = runs.iloc[0].to_dict()
    run_id = str(run_row.get("run_id") or "")
    fills = _load_fills(resolved_dsn, run_id, fill_limit=fill_limit)
    market_context = _resolve_market_context(run_row=run_row, fills_frame=fills)
    market = _load_market(
        market_db_path=str(market_context.get("market_db_path") or ""),
        symbol=str(market_context.get("symbol") or ""),
        timeframe=str(market_context.get("timeframe") or "1m"),
        exchange=str(market_context.get("exchange") or "binance"),
        point_limit=point_limit,
    )
    return build_market_data_payload(
        run_row=run_row,
        fills_frame=fills,
        market_frame=market,
    )


def load_optimization_insights_payload(
    *,
    dsn: str | None = None,
    point_limit: int = 200,
) -> dict[str, Any]:
    if dsn is not None and not str(dsn).strip():
        return {
            **_empty_surface_payload(reason="missing_dsn"),
            "summary_metrics": [],
            "stage_breakdown": [],
            "top_candidates": [],
            "best_candidate": None,
        }
    resolved_dsn = resolve_dashboard_postgres_dsn(dsn)
    if not resolved_dsn:
        return {
            **_empty_surface_payload(reason="missing_dsn"),
            "summary_metrics": [],
            "stage_breakdown": [],
            "top_candidates": [],
            "best_candidate": None,
        }
    runs = _load_runs(resolved_dsn, run_limit=1)
    run_id = str(runs.iloc[0]["run_id"] or "") if not runs.empty else ""
    optimization = _load_optimization_results(resolved_dsn, point_limit=point_limit)
    return build_optimization_insights_payload(
        run_id=run_id,
        optimization_frame=optimization,
    )


def load_raw_data_payload(
    *,
    dsn: str | None = None,
    point_limit: int = 60,
) -> dict[str, Any]:
    if dsn is not None and not str(dsn).strip():
        return {
            **_empty_surface_payload(reason="missing_dsn"),
            "context": {},
            "frame_summaries": [],
            "previews": [],
        }
    resolved_dsn = resolve_dashboard_postgres_dsn(dsn)
    if not resolved_dsn:
        return {
            **_empty_surface_payload(reason="missing_dsn"),
            "context": {},
            "frame_summaries": [],
            "previews": [],
        }
    runs = _load_runs(resolved_dsn, run_limit=10)
    if runs.empty:
        return {
            **_empty_surface_payload(reason="no_runs"),
            "context": {},
            "frame_summaries": [],
            "previews": [],
        }

    run_row = runs.iloc[0].to_dict()
    run_id = str(run_row.get("run_id") or "")
    fills = _load_fills(resolved_dsn, run_id, fill_limit=point_limit)
    market_context = _resolve_market_context(run_row=run_row, fills_frame=fills)
    frames: list[tuple[str, pd.DataFrame]] = [
        ("Runs", runs.head(point_limit)),
        ("Equity", _load_metrics(resolved_dsn, run_id, point_limit=point_limit)),
        ("Fills", fills.head(point_limit)),
        ("Orders", _load_orders(resolved_dsn, run_id, order_limit=point_limit)),
        ("Risk Events", _load_risk_events(resolved_dsn, run_id, limit=point_limit)),
        ("Heartbeats", _load_heartbeats(resolved_dsn, run_id, limit=point_limit)),
        ("Order State Events", _load_order_states(resolved_dsn, run_id, limit=point_limit)),
        (
            "Market OHLCV",
            _load_market(
                market_db_path=str(market_context.get("market_db_path") or ""),
                symbol=str(market_context.get("symbol") or ""),
                timeframe=str(market_context.get("timeframe") or "1m"),
                exchange=str(market_context.get("exchange") or "binance"),
                point_limit=point_limit,
            ),
        ),
        ("Optimization Results", _load_optimization_results(resolved_dsn, point_limit=point_limit)),
        ("Workflow Jobs", _load_recent_workflow_jobs_frame(resolved_dsn, limit=point_limit)),
    ]
    context = {
        "run_id": run_id,
        "source": "postgres",
        "market": (
            f"{market_context.get('symbol', 'n/a')} "
            f"{market_context.get('timeframe', 'n/a')} "
            f"({market_context.get('exchange', 'n/a')})"
        ),
    }
    return build_raw_data_payload(
        run_id=run_id,
        context=context,
        frames=frames,
    )


__all__ = [
    "build_execution_analytics_payload",
    "build_market_data_payload",
    "build_optimization_insights_payload",
    "build_performance_price_payload",
    "build_raw_data_payload",
    "build_report_export_payload",
    "compute_trade_analytics",
    "load_execution_analytics_payload",
    "load_market_data_payload",
    "load_optimization_insights_payload",
    "load_performance_price_payload",
    "load_raw_data_payload",
    "load_report_export_payload",
]

_FN_MAP: dict[str, Any] = {}  # populated lazily below


def _get_fn_map():
    return {
        "load_performance_price_payload": load_performance_price_payload,
        "load_execution_analytics_payload": load_execution_analytics_payload,
        "load_market_data_payload": load_market_data_payload,
        "load_optimization_insights_payload": load_optimization_insights_payload,
        "load_raw_data_payload": load_raw_data_payload,
        "load_report_export_payload": load_report_export_payload,
    }


def main(argv: list[str] | None = None) -> int:
    r"""Module-mode entry for all cutover-surface routes.

    Each of the 6 routes that share this module passes ``--fn <function_name>``
    so the correct payload builder is invoked.

    Example::

        uv run python -m lumina_quant.dashboard.cutover_surfaces_service \\
            --fn load_performance_price_payload --json
    """
    import argparse
    import json

    fn_map = _get_fn_map()
    parser = argparse.ArgumentParser(
        prog="lumina_quant.dashboard.cutover_surfaces_service",
        description="Emit a cutover-surface dashboard payload as JSON.",
    )
    parser.add_argument(
        "--fn",
        choices=list(fn_map.keys()),
        default="load_performance_price_payload",
        help="Payload builder to invoke (default: load_performance_price_payload).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Output as JSON (default and only output mode).",
    )
    parser.add_argument(
        "--point-limit",
        type=int,
        default=240,
        dest="point_limit",
        help="Max metric/equity points (default: 240).",
    )
    parser.add_argument(
        "--fill-limit", type=int, default=80, dest="fill_limit", help="Max fill rows (default: 80)."
    )
    parser.add_argument(
        "--order-limit",
        type=int,
        default=200,
        dest="order_limit",
        help="Max order rows (default: 200).",
    )
    parser.add_argument(
        "--event-limit",
        type=int,
        default=50,
        dest="event_limit",
        help="Max risk/heartbeat event rows (default: 50).",
    )
    args = parser.parse_args(argv)

    fn = fn_map[args.fn]
    # Pass only the kwargs each function accepts; unused kwargs are silently dropped.
    import inspect

    sig = inspect.signature(fn)
    kwargs: dict[str, Any] = {}
    if "point_limit" in sig.parameters:
        kwargs["point_limit"] = args.point_limit
    if "fill_limit" in sig.parameters:
        kwargs["fill_limit"] = args.fill_limit
    if "order_limit" in sig.parameters:
        kwargs["order_limit"] = args.order_limit
    if "event_limit" in sig.parameters:
        kwargs["event_limit"] = args.event_limit

    payload = fn(**kwargs)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
