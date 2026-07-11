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
    FULL_EQUITY_ROW_CAP,
    build_overview_payload_from_frames,
    coerce_datetime_series,
    downsample_curve_indices,
    infer_periods_per_year,
    overview_metric,
    recent_runs_from_frame,
    resolve_dashboard_postgres_dsn,
)
from lumina_quant.market_data import (
    normalize_symbol,
    normalize_timeframe_token,
    timeframe_to_milliseconds,
)
from lumina_quant.postgres_state import _connect_postgres

# Indicator windows for the market-data surface (pure pandas/numpy, no scipy).
RSI_WINDOW = 14
ATR_WINDOW = 14
REALIZED_VOL_WINDOW = 30
RANGE_WINDOW = 20
# recent_bars covers the same window the summary metrics use.
RECENT_BAR_LIMIT = 240
# Latest runs exposed in every cutover payload for the UI run selector.
RUN_SELECTOR_LIMIT = 10


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


# Not delegated to utils.numeric.safe_float: pd.notna rejects NaN but PASSES inf,
# whereas canonical finite_only rejects both -> not behavior-equivalent.
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


def _observed_symbols(fills_frame: pd.DataFrame) -> list[str]:
    """Sorted distinct symbols observed in the run's fills (fallback: config)."""
    if "symbol" in fills_frame.columns:
        values = fills_frame["symbol"].dropna().astype(str)
        symbols = sorted({value for value in values.tolist() if value})
        if symbols:
            return symbols
    configured = get_default_runtime_config().trading.symbols
    if isinstance(configured, list):
        return sorted({str(symbol) for symbol in configured if str(symbol)})
    return []


def _resolve_market_context(
    *,
    run_row: dict[str, Any],
    fills_frame: pd.DataFrame,
    symbol: str | None = None,
) -> dict[str, Any]:
    metadata = _parse_json_dict(run_row.get("metadata"))
    configured_symbols = get_default_runtime_config().trading.symbols
    symbol = str(symbol or "").strip()
    if not symbol and "symbol" in fills_frame.columns:
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


def _resolve_benchmark_symbol(run_row: dict[str, Any] | None) -> str | None:
    """The run's primary/benchmark symbol, or None when unknown.

    Backtests record ``benchmark_price`` as the close of ``symbol_list[0]``
    (portfolio_backtest), and the run metadata persists that list under
    ``symbols`` (cli/backtest.py). Config is only a best-effort fallback for
    runs whose metadata predates the symbols key.
    """
    metadata = _parse_json_dict((run_row or {}).get("metadata"))
    metadata_symbols = metadata.get("symbols")
    if isinstance(metadata_symbols, list) and metadata_symbols:
        first = str(metadata_symbols[0] or "").strip()
        if first:
            return first
    configured = get_default_runtime_config().trading.symbols
    if isinstance(configured, list) and configured:
        first = str(configured[0] or "").strip()
        if first:
            return first
    return None


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
    # Cross-side convention: percent-like *_pct metrics travel as raw fractions
    # (0.05 = 5%); the frontend multiplies by 100 at render.
    df["realized_return_pct"] = pd.Series(realized_return, dtype="float64")
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


def _load_run_row_exact(dsn: str, run_id: str) -> dict[str, Any] | None:
    """Exact parameterized run lookup (mirrors overview_service).

    Runs older than any recency-limited selector window still resolve; only a
    genuinely missing run_id returns ``None``.
    """
    conn = _connect_postgres(dsn)
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
                WHERE run_id = %s
                LIMIT 1
                """,
                (run_id,),
            )
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description or ()]
    finally:
        conn.close()
    if not rows:
        return None
    return dict(zip(columns, rows[0], strict=False))


def _select_run_row(
    dsn: str,
    runs: pd.DataFrame,
    run_id: str | None,
) -> dict[str, Any] | None:
    """Pick the requested run row (falling back to the latest run).

    Returns ``None`` only when an explicitly requested run_id does not exist
    in the runs table at all.
    """
    requested = str(run_id or "").strip()
    if not requested:
        return runs.iloc[0].to_dict() if not runs.empty else None
    if not runs.empty:
        matches = runs[runs["run_id"].astype(str) == requested]
        if not matches.empty:
            return matches.iloc[0].to_dict()
    return _load_run_row_exact(dsn, requested)


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
    run_row: dict[str, Any] | None = None,
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
        # Contract: string|null — the symbol whose close was recorded as
        # benchmark_price (first entry of the run's symbol list).
        "benchmark_symbol": _resolve_benchmark_symbol(run_row),
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


def _last_finite(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    valid = series.dropna()
    if valid.empty:
        return None
    value = float(valid.iloc[-1])
    return value if math.isfinite(value) else None


def _compute_market_indicators(market_frame: pd.DataFrame) -> dict[str, float | None]:
    """Deterministic indicator set from the loaded OHLCV window (no scipy).

    Percent-like keys (atr_pct, realized_vol_30, dist_from_20bar_*) are raw
    fractions per the cross-side convention.
    """
    close = pd.to_numeric(market_frame.get("close"), errors="coerce")
    high = pd.to_numeric(market_frame.get("high"), errors="coerce")
    low = pd.to_numeric(market_frame.get("low"), errors="coerce")
    volume = pd.to_numeric(market_frame.get("volume"), errors="coerce")
    last_close = _last_finite(close)

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing == EMA with alpha = 1/window; all-gain windows resolve
    # to RSI 100 naturally via the inf ratio.
    avg_gain = gain.ewm(alpha=1.0 / RSI_WINDOW, adjust=False, min_periods=RSI_WINDOW).mean()
    avg_loss = loss.ewm(alpha=1.0 / RSI_WINDOW, adjust=False, min_periods=RSI_WINDOW).mean()
    rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    rsi_last = _last_finite(rsi)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / ATR_WINDOW, adjust=False, min_periods=ATR_WINDOW).mean()
    atr_last = _last_finite(atr)
    atr_pct = None
    if atr_last is not None and last_close:
        atr_pct = atr_last / last_close

    realized_vol = None
    window_returns = close.pct_change().dropna().tail(REALIZED_VOL_WINDOW)
    if len(window_returns) >= 2:
        per_bar_vol = float(window_returns.std(ddof=1))
        if math.isfinite(per_bar_vol):
            realized_vol = per_bar_vol * math.sqrt(
                infer_periods_per_year(market_frame.get("datetime"))
            )

    dist_from_high = None
    dist_from_low = None
    window_high = high.dropna().tail(RANGE_WINDOW)
    window_low = low.dropna().tail(RANGE_WINDOW)
    if last_close is not None and not window_high.empty and float(window_high.max()):
        dist_from_high = last_close / float(window_high.max()) - 1.0
    if last_close is not None and not window_low.empty and float(window_low.min()):
        dist_from_low = last_close / float(window_low.min()) - 1.0

    window_volume = None
    if volume.notna().any():
        window_volume = float(volume.dropna().sum())

    return {
        "rsi_14": rsi_last,
        "atr_14": atr_last,
        "atr_pct": atr_pct,
        "realized_vol_30": realized_vol,
        "dist_from_20bar_high": dist_from_high,
        "dist_from_20bar_low": dist_from_low,
        "last_close": last_close,
        "window_volume": window_volume,
    }


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def build_market_data_payload(
    *,
    run_row: dict[str, Any],
    fills_frame: pd.DataFrame,
    market_frame: pd.DataFrame,
    symbol: str | None = None,
) -> dict[str, Any]:
    run_id = str(run_row.get("run_id") or "")
    context = _resolve_market_context(run_row=run_row, fills_frame=fills_frame, symbol=symbol)
    payload = {
        **_empty_surface_payload(run_id=run_id, reason="ok"),
        "market_context": context,
        "symbols": _observed_symbols(fills_frame),
        "summary_metrics": [],
        "recent_bars": [],
        "bar_window": {"start": None, "end": None, "bar_count": 0},
        "indicator_summary": [
            overview_metric("Strategy", context["strategy"], key="strategy"),
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
        # Raw fraction per the cross-side convention; frontend renders x100.
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
    indicators = _compute_market_indicators(market_frame)
    payload["indicator_summary"].extend(
        [
            overview_metric(
                f"RSI ({RSI_WINDOW})", _round_or_none(indicators["rsi_14"]), key="rsi_14"
            ),
            overview_metric(
                f"ATR ({ATR_WINDOW})", _round_or_none(indicators["atr_14"]), key="atr_14"
            ),
            overview_metric("ATR %", _round_or_none(indicators["atr_pct"]), key="atr_pct"),
            overview_metric(
                f"Realized Vol ({REALIZED_VOL_WINDOW} bars, annualized)",
                _round_or_none(indicators["realized_vol_30"]),
                key="realized_vol_30",
            ),
            overview_metric(
                f"Dist From {RANGE_WINDOW}-Bar High",
                _round_or_none(indicators["dist_from_20bar_high"]),
                key="dist_from_20bar_high",
            ),
            overview_metric(
                f"Dist From {RANGE_WINDOW}-Bar Low",
                _round_or_none(indicators["dist_from_20bar_low"]),
                key="dist_from_20bar_low",
            ),
            overview_metric(
                "Last Close", _round_or_none(indicators["last_close"]), key="last_close"
            ),
            overview_metric(
                "Window Volume",
                _round_or_none(indicators["window_volume"]),
                key="window_volume",
            ),
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
    payload["bar_window"] = {
        "start": _isoformat(market_frame["datetime"].iloc[0])
        if "datetime" in market_frame.columns
        else None,
        "end": _isoformat(market_frame["datetime"].iloc[-1])
        if "datetime" in market_frame.columns
        else None,
        "bar_count": len(market_frame.index),
    }
    payload["recent_bars"] = [
        {
            "timestamp": _isoformat(row.datetime),
            "open": _normalize_json_value(getattr(row, "open", None)),
            "high": _normalize_json_value(getattr(row, "high", None)),
            "low": _normalize_json_value(getattr(row, "low", None)),
            "close": _normalize_json_value(getattr(row, "close", None)),
            "volume": _normalize_json_value(getattr(row, "volume", None)),
        }
        for row in market_frame.tail(RECENT_BAR_LIMIT).itertuples(index=False)
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

    # cagr/mdd are persisted as percent-magnitude TEXT ('12.3456', 'N/A') by
    # cli/optimize.py; the contract declares number|null RAW FRACTIONS, so
    # coerce here ('N/A' -> None via NaN, percent magnitude -> fraction).
    cagr_fraction = pd.to_numeric(optimization_frame.get("cagr"), errors="coerce") / 100.0
    mdd_fraction = pd.to_numeric(optimization_frame.get("mdd"), errors="coerce") / 100.0
    ordered = optimization_frame.assign(
        _sharpe=sharpe, cagr=cagr_fraction, mdd=mdd_fraction
    ).sort_values(
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


def _empty_performance_price_payload(reason: str, *, runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **_empty_surface_payload(reason=reason),
        "source": {"status": reason},
        "runs": runs,
        "summary_metrics": [],
        "performance_metrics": {},
        "equity_curve": [],
        "drawdown_curve": [],
        "benchmark_curve": [],
        "funding_curve": [],
        # Contract-required even on empty shapes (null = unknown).
        "benchmark_symbol": None,
        "trade_markers": [],
    }


def load_performance_price_payload(
    *,
    dsn: str | None = None,
    point_limit: int = 240,
    fill_limit: int = 80,
    run_id: str | None = None,
) -> dict[str, Any]:
    if dsn is not None and not str(dsn).strip():
        return _empty_performance_price_payload("missing_dsn", runs=[])
    resolved_dsn = resolve_dashboard_postgres_dsn(dsn)
    if not resolved_dsn:
        return _empty_performance_price_payload("missing_dsn", runs=[])

    runs = _load_runs(resolved_dsn, run_limit=RUN_SELECTOR_LIMIT)
    if runs.empty:
        return _empty_performance_price_payload("no_runs", runs=[])
    runs_list = recent_runs_from_frame(runs)
    run_row = _select_run_row(resolved_dsn, runs, run_id)
    if run_row is None:
        return _empty_performance_price_payload("run_not_found", runs=runs_list)

    selected_run_id = str(run_row.get("run_id") or "")
    # Headline metrics must describe the FULL run (mirrors the overview P0
    # fix): compute performance/summary metrics from the full equity series
    # and only downsample the payload curves to point_limit.
    metrics = _load_metrics(resolved_dsn, selected_run_id, point_limit=FULL_EQUITY_ROW_CAP)
    fills = _load_fills(resolved_dsn, selected_run_id, fill_limit=fill_limit)
    curve_point_limit = int(max(2, point_limit))
    overview_payload = build_overview_payload_from_frames(
        contract=_dashboard_contract(),
        runs_frame=runs,
        equity_frame=metrics[["datetime", "total"]].copy()
        if {"datetime", "total"}.issubset(metrics.columns)
        else pd.DataFrame(),
        selected_run_row=run_row,
        curve_point_limit=curve_point_limit,
        equity_truncated=len(metrics.index) >= FULL_EQUITY_ROW_CAP,
    )
    curve_driver = (
        metrics["total"] if "total" in metrics.columns else pd.Series(range(len(metrics.index)))
    )
    curve_indices = downsample_curve_indices(curve_driver, curve_point_limit)
    payload = build_performance_price_payload(
        overview_payload=overview_payload,
        metrics_frame=metrics.iloc[curve_indices],
        fills_frame=fills,
        run_row=run_row,
    )
    payload["runs"] = runs_list
    payload["equity_window"] = overview_payload.get("equity_window")
    return payload


def _empty_execution_analytics_payload(
    reason: str, *, runs: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        **_empty_surface_payload(reason=reason),
        "runs": runs,
        "summary": {},
        "direction_breakdown": [],
        "order_status": [],
        "recent_closed_trades": [],
    }


def load_execution_analytics_payload(
    *,
    dsn: str | None = None,
    fill_limit: int = 200,
    order_limit: int = 200,
    run_id: str | None = None,
) -> dict[str, Any]:
    if dsn is not None and not str(dsn).strip():
        return _empty_execution_analytics_payload("missing_dsn", runs=[])
    resolved_dsn = resolve_dashboard_postgres_dsn(dsn)
    if not resolved_dsn:
        return _empty_execution_analytics_payload("missing_dsn", runs=[])
    runs = _load_runs(resolved_dsn, run_limit=RUN_SELECTOR_LIMIT)
    if runs.empty:
        return _empty_execution_analytics_payload("no_runs", runs=[])
    runs_list = recent_runs_from_frame(runs)
    run_row = _select_run_row(resolved_dsn, runs, run_id)
    if run_row is None:
        return _empty_execution_analytics_payload("run_not_found", runs=runs_list)
    selected_run_id = str(run_row.get("run_id") or "")
    fills = _load_fills(resolved_dsn, selected_run_id, fill_limit=fill_limit)
    orders = _load_orders(resolved_dsn, selected_run_id, order_limit=order_limit)
    payload = build_execution_analytics_payload(
        run_id=selected_run_id,
        fills_frame=fills,
        orders_frame=orders,
    )
    payload["runs"] = runs_list
    return payload


def _empty_report_export_payload(reason: str, *, runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **_empty_surface_payload(reason=reason),
        "runs": runs,
        "filenames": {},
        "json_report": {},
        "markdown_report": "",
        "cutover_gate": {},
    }


def load_report_export_payload(
    *,
    dsn: str | None = None,
    point_limit: int = 240,
    fill_limit: int = 200,
    event_limit: int = 50,
    run_id: str | None = None,
) -> dict[str, Any]:
    if dsn is not None and not str(dsn).strip():
        return _empty_report_export_payload("missing_dsn", runs=[])
    resolved_dsn = resolve_dashboard_postgres_dsn(dsn)
    if not resolved_dsn:
        return _empty_report_export_payload("missing_dsn", runs=[])
    runs = _load_runs(resolved_dsn, run_limit=RUN_SELECTOR_LIMIT)
    if runs.empty:
        return _empty_report_export_payload("no_runs", runs=[])
    runs_list = recent_runs_from_frame(runs)
    run_row = _select_run_row(resolved_dsn, runs, run_id)
    if run_row is None:
        return _empty_report_export_payload("run_not_found", runs=runs_list)
    selected_run_id = str(run_row.get("run_id") or "")
    # Headline metrics must describe the FULL run (mirrors the overview P0
    # fix and load_performance_price_payload): compute report metrics from the
    # full equity series and only downsample the payload curves.
    metrics = _load_metrics(resolved_dsn, selected_run_id, point_limit=FULL_EQUITY_ROW_CAP)
    fills = _load_fills(resolved_dsn, selected_run_id, fill_limit=fill_limit)
    risk = _load_risk_events(resolved_dsn, selected_run_id, limit=event_limit)
    heartbeats = _load_heartbeats(resolved_dsn, selected_run_id, limit=event_limit)
    overview_payload = build_overview_payload_from_frames(
        contract=_dashboard_contract(),
        runs_frame=runs,
        equity_frame=metrics[["datetime", "total"]].copy()
        if {"datetime", "total"}.issubset(metrics.columns)
        else pd.DataFrame(),
        selected_run_row=run_row,
        curve_point_limit=int(max(2, point_limit)),
        equity_truncated=len(metrics.index) >= FULL_EQUITY_ROW_CAP,
    )
    payload = build_report_export_payload(
        run_row=run_row,
        overview_payload=overview_payload,
        fills_frame=fills,
        risk_frame=risk,
        heartbeats_frame=heartbeats,
    )
    payload["runs"] = runs_list
    return payload


def _empty_market_data_payload(reason: str, *, runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **_empty_surface_payload(reason=reason),
        "runs": runs,
        "market_context": {},
        "symbols": [],
        "summary_metrics": [],
        "recent_bars": [],
        "bar_window": {"start": None, "end": None, "bar_count": 0},
        "indicator_summary": [],
        "warnings": [],
    }


def load_market_data_payload(
    *,
    dsn: str | None = None,
    point_limit: int = 240,
    fill_limit: int = 80,
    run_id: str | None = None,
    symbol: str | None = None,
) -> dict[str, Any]:
    if dsn is not None and not str(dsn).strip():
        return _empty_market_data_payload("missing_dsn", runs=[])
    resolved_dsn = resolve_dashboard_postgres_dsn(dsn)
    if not resolved_dsn:
        return _empty_market_data_payload("missing_dsn", runs=[])
    runs = _load_runs(resolved_dsn, run_limit=RUN_SELECTOR_LIMIT)
    if runs.empty:
        return _empty_market_data_payload("no_runs", runs=[])
    runs_list = recent_runs_from_frame(runs)
    run_row = _select_run_row(resolved_dsn, runs, run_id)
    if run_row is None:
        return _empty_market_data_payload("run_not_found", runs=runs_list)

    selected_run_id = str(run_row.get("run_id") or "")
    fills = _load_fills(resolved_dsn, selected_run_id, fill_limit=fill_limit)
    market_context = _resolve_market_context(run_row=run_row, fills_frame=fills, symbol=symbol)
    market = _load_market(
        market_db_path=str(market_context.get("market_db_path") or ""),
        symbol=str(market_context.get("symbol") or ""),
        timeframe=str(market_context.get("timeframe") or "1m"),
        exchange=str(market_context.get("exchange") or "binance"),
        point_limit=point_limit,
    )
    payload = build_market_data_payload(
        run_row=run_row,
        fills_frame=fills,
        market_frame=market,
        symbol=symbol,
    )
    payload["runs"] = runs_list
    return payload


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


def _empty_raw_data_payload(reason: str, *, runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **_empty_surface_payload(reason=reason),
        "runs": runs,
        "context": {},
        "frame_summaries": [],
        "previews": [],
    }


def load_raw_data_payload(
    *,
    dsn: str | None = None,
    point_limit: int = 60,
    run_id: str | None = None,
) -> dict[str, Any]:
    if dsn is not None and not str(dsn).strip():
        return _empty_raw_data_payload("missing_dsn", runs=[])
    resolved_dsn = resolve_dashboard_postgres_dsn(dsn)
    if not resolved_dsn:
        return _empty_raw_data_payload("missing_dsn", runs=[])
    runs = _load_runs(resolved_dsn, run_limit=RUN_SELECTOR_LIMIT)
    if runs.empty:
        return _empty_raw_data_payload("no_runs", runs=[])
    runs_list = recent_runs_from_frame(runs)
    run_row = _select_run_row(resolved_dsn, runs, run_id)
    if run_row is None:
        return _empty_raw_data_payload("run_not_found", runs=runs_list)

    selected_run_id = str(run_row.get("run_id") or "")
    fills = _load_fills(resolved_dsn, selected_run_id, fill_limit=point_limit)
    market_context = _resolve_market_context(run_row=run_row, fills_frame=fills)
    frames: list[tuple[str, pd.DataFrame]] = [
        ("Runs", runs.head(point_limit)),
        ("Equity", _load_metrics(resolved_dsn, selected_run_id, point_limit=point_limit)),
        ("Fills", fills.head(point_limit)),
        ("Orders", _load_orders(resolved_dsn, selected_run_id, order_limit=point_limit)),
        ("Risk Events", _load_risk_events(resolved_dsn, selected_run_id, limit=point_limit)),
        ("Heartbeats", _load_heartbeats(resolved_dsn, selected_run_id, limit=point_limit)),
        (
            "Order State Events",
            _load_order_states(resolved_dsn, selected_run_id, limit=point_limit),
        ),
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
        "run_id": selected_run_id,
        "source": "postgres",
        "market": (
            f"{market_context.get('symbol', 'n/a')} "
            f"{market_context.get('timeframe', 'n/a')} "
            f"({market_context.get('exchange', 'n/a')})"
        ),
    }
    payload = build_raw_data_payload(
        run_id=selected_run_id,
        context=context,
        frames=frames,
    )
    payload["runs"] = runs_list
    return payload


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
    parser.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="Build the payload for this run instead of the latest one.",
    )
    parser.add_argument(
        "--symbol",
        dest="symbol",
        default=None,
        help="Market-data only: build market context/bars for this symbol.",
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
    if "run_id" in sig.parameters:
        kwargs["run_id"] = args.run_id
    if "symbol" in sig.parameters:
        kwargs["symbol"] = args.symbol

    payload = fn(**kwargs)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
