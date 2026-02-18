import html
import importlib
import json
import math
import os
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lumina_quant.config import BacktestConfig, BaseConfig, OptimizationConfig
from lumina_quant.utils.performance import (
    create_alpha_beta,
    create_annualized_volatility,
    create_cagr,
    create_calmar_ratio,
    create_drawdowns,
    create_information_ratio,
    create_sharpe_ratio,
    create_sortino_ratio,
)
from plotly.subplots import make_subplots
from strategies import (
    get_default_grid_config,
    get_default_optuna_config,
    get_default_strategy_params,
    get_strategy_names,
)

st.set_page_config(layout="wide", page_title="LuminaQuant Dashboard")
st.title("LuminaQuant: Full Trading Intelligence")

DEFAULT_DB_PATH = "data/lumina_quant.db"
DEFAULT_REFRESH_INTERVAL_SEC = 5
DEFAULT_WINDOW_POINTS = 2500
DEFAULT_DOWNSAMPLE_TARGET_POINTS = 5000
DEFAULT_RUN_STALE_SEC = 180
WORKFLOW_LOG_DIR = os.path.join("logs", "workflow_jobs")
WORKFLOW_CONTROL_DIR = os.path.join("logs", "control")
METRIC_DEFINITIONS = {
    "Total Return": "(Final Equity / Initial Equity) - 1 over selected period.",
    "Cumulative Return": "Compounded series from periodic returns across selected window.",
    "CAGR": "Annualized growth rate from period start to end.",
    "Ann. Volatility": "Annualized standard deviation of periodic returns.",
    "Sharpe Ratio": "Risk-adjusted return using total volatility.",
    "Sortino Ratio": "Risk-adjusted return using downside volatility only.",
    "Calmar Ratio": "CAGR divided by absolute max drawdown.",
    "Max Drawdown": "Largest peak-to-trough equity decline in selected period.",
    "Total Net Profit": "Closed-trade net result after commission over selected period.",
    "Gross Profit": "Sum of all positive closed-trade realized PnL.",
    "Gross Loss": "Sum of all negative closed-trade realized PnL.",
    "Profit Factor": "Gross Profit divided by absolute Gross Loss.",
    "Expected Payoff": "Average realized PnL per closed trade.",
    "Recovery Factor": "Total net profit divided by maximal equity drawdown amount.",
    "Open P/L": "Unrealized PnL estimated as total (closed + open) minus closed-trade net PnL.",
    "Total (C+O)": "Combined closed and open PnL relative to initial equity.",
    "R/MDD": "Total (closed + open) PnL divided by equity maximal drawdown.",
    "Profit Trades (% of total)": "Count of profitable closed trades and share versus total closed trades.",
    "Loss Trades (% of total)": "Count of losing closed trades and share versus total closed trades.",
    "Avg Profit Trade": "Average realized PnL across profitable closed trades.",
    "Avg Loss Trade": "Average realized PnL across losing closed trades.",
    "Payoff Ratio": "Average profit trade divided by absolute average loss trade.",
    "AHPR": "Arithmetic mean of per-trade holding period growth factors.",
    "GHPR": "Geometric mean of per-trade holding period growth factors.",
    "LR Correlation": "Correlation between equity curve and linear time trend.",
    "LR Std Error": "Standard error of equity against linear trend fit.",
    "Z-Score": "Runs-test Z-score of win/loss sequence randomness.",
    "Long Trades (Win %)": "Closed long-side trades and long-side win rate.",
    "Short Trades (Win %)": "Closed short-side trades and short-side win rate.",
    "Avg Holding Time": "Average holding time of closed quantities.",
    "Max Holding Time": "Maximum holding time of closed quantities.",
    "Min Holding Time": "Minimum holding time of closed quantities.",
    "Equity Drawdown Absolute": "Initial equity minus minimum equity (never below zero).",
    "Equity Drawdown Maximal": "Largest absolute peak-to-trough equity drawdown amount.",
    "Equity Drawdown Relative %": "Largest relative peak-to-trough equity drawdown percent.",
    "Balance Drawdown Absolute": "Initial balance minus minimum realized-balance level.",
    "Balance Drawdown Maximal": "Largest absolute peak-to-trough realized-balance drawdown amount.",
    "Balance Drawdown Relative %": "Largest relative peak-to-trough realized-balance drawdown percent.",
    "Alpha": "Strategy excess annualized return versus benchmark after beta adjustment.",
    "Beta": "Sensitivity of strategy returns to benchmark returns.",
    "Information Ratio": "Mean active return divided by active return volatility.",
    "Funding (Net)": "Accumulated funding payments (positive means paid by longs, received by shorts when negative).",
    "Win Rate": "Proportion of closed trades with positive realized PnL.",
}

MIRROR_DASHBOARD_CSS = """
<style>
.lq-mirror-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 0.65rem;
    margin: 0.2rem 0 0.7rem 0;
}
.lq-mirror-card {
    background: radial-gradient(120% 120% at 0% 0%, #102348 0%, #0b1834 68%, #081124 100%);
    border: 1px solid #17356e;
    border-radius: 12px;
    padding: 0.78rem 0.9rem;
    box-shadow: inset 0 0 0 1px rgba(13, 183, 158, 0.05), 0 8px 20px rgba(0, 0, 0, 0.18);
}
.lq-mirror-card .label {
    color: #7ea4db;
    font-size: 0.74rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.34rem;
}
.lq-mirror-card .value {
    font-size: 1.72rem;
    font-weight: 760;
    letter-spacing: 0.01em;
    line-height: 1.08;
    margin-bottom: 0.23rem;
    color: #dce9ff;
}
.lq-mirror-card .sub {
    color: #92b3df;
    font-size: 0.81rem;
    line-height: 1.2;
}
.lq-mirror-card.up .value {
    color: #24d37b;
}
.lq-mirror-card.down .value {
    color: #ff5f5f;
}
</style>
"""


def _safe_interval_sec(value):
    try:
        parsed = int(value)
    except Exception:
        return DEFAULT_REFRESH_INTERVAL_SEC
    return max(1, min(300, parsed))


def _setup_auto_refresh(enabled, interval_sec):
    if not enabled:
        return 0, "manual"

    interval_sec = _safe_interval_sec(interval_sec)
    interval_ms = interval_sec * 1000
    try:
        module = importlib.import_module("streamlit_autorefresh")
        counter = int(
            module.st_autorefresh(interval=interval_ms, key=f"lq-autorefresh-{interval_ms}")
        )
        return counter, "streamlit_autorefresh"
    except Exception:
        st.markdown(f"<meta http-equiv='refresh' content='{interval_sec}'>", unsafe_allow_html=True)
        return int(time.time() // interval_sec), "meta_refresh"


def _coerce_datetime(df, column):
    if df.empty or column not in df.columns:
        return df
    df[column] = pd.to_datetime(df[column], errors="coerce", utc=True).dt.tz_localize(None)
    return df


def _utc_now_iso():
    return datetime.now(UTC).isoformat()


def _ensure_workflow_jobs_schema(db_path):
    if not db_path:
        return
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS workflow_jobs (
                job_id TEXT PRIMARY KEY,
                workflow TEXT NOT NULL,
                status TEXT NOT NULL,
                requested_mode TEXT,
                strategy TEXT,
                command_json TEXT,
                env_json TEXT,
                pid INTEGER,
                run_id TEXT,
                started_at TEXT,
                ended_at TEXT,
                exit_code INTEGER,
                log_path TEXT,
                stop_file TEXT,
                metadata_json TEXT,
                last_updated TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_workflow_jobs_started_at
                ON workflow_jobs(started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_workflow_jobs_status
                ON workflow_jobs(status);
            """
        )
        conn.commit()
    finally:
        conn.close()


def _insert_workflow_job_row(db_path, row):
    _ensure_workflow_jobs_schema(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO workflow_jobs(
                job_id, workflow, status, requested_mode, strategy, command_json,
                env_json, pid, run_id, started_at, ended_at, exit_code,
                log_path, stop_file, metadata_json, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("job_id"),
                row.get("workflow"),
                row.get("status"),
                row.get("requested_mode"),
                row.get("strategy"),
                row.get("command_json"),
                row.get("env_json"),
                row.get("pid"),
                row.get("run_id"),
                row.get("started_at"),
                row.get("ended_at"),
                row.get("exit_code"),
                row.get("log_path"),
                row.get("stop_file"),
                row.get("metadata_json"),
                row.get("last_updated") or _utc_now_iso(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _update_workflow_job_row(db_path, job_id, **updates):
    if not updates:
        return
    _ensure_workflow_jobs_schema(db_path)
    fields = dict(updates)
    fields["last_updated"] = _utc_now_iso()
    assignments = ", ".join(f"{key} = ?" for key in fields)
    values = [*list(fields.values()), job_id]
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            f"UPDATE workflow_jobs SET {assignments} WHERE job_id = ?",
            values,
        )
        conn.commit()
    finally:
        conn.close()


def _is_process_running(pid):
    try:
        pid = int(pid)
    except Exception:
        return False
    if pid <= 0:
        return False
    if sys.platform.startswith("win"):
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = (completed.stdout or "").upper()
        return str(pid) in output and "NO TASKS" not in output
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _terminate_process(pid):
    try:
        pid = int(pid)
    except Exception:
        return False, "invalid pid"
    if pid <= 0:
        return False, "invalid pid"
    if sys.platform.startswith("win"):
        proc = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        ok = proc.returncode == 0
        detail = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return ok, detail.strip()
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        return False, str(exc)
    return True, "terminated"


def _tail_text_file(path, max_chars=20000):
    if not path or not os.path.exists(path):
        return ""
    try:
        max_bytes = max(4096, int(max_chars) * 3)
    except Exception:
        max_bytes = 60000
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        total = f.tell()
        f.seek(max(0, total - max_bytes), os.SEEK_SET)
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


@st.cache_data
def load_workflow_jobs(db_path, refresh_counter=0, limit=200):
    _ = refresh_counter
    if not os.path.exists(db_path):
        return pd.DataFrame()
    _ensure_workflow_jobs_schema(db_path)
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT job_id, workflow, status, requested_mode, strategy, command_json,
                   env_json, pid, run_id, started_at, ended_at, exit_code,
                   log_path, stop_file, metadata_json, last_updated
            FROM workflow_jobs
            ORDER BY COALESCE(started_at, last_updated) DESC
            LIMIT ?
            """,
            conn,
            params=[int(max(1, limit))],
        )
        for col in ["started_at", "ended_at", "last_updated"]:
            df = _coerce_datetime(df, col)
        return df
    finally:
        conn.close()


def _refresh_workflow_jobs(db_path):
    _ensure_workflow_jobs_schema(db_path)
    if "workflow_processes" not in st.session_state:
        st.session_state["workflow_processes"] = {}
    managed = st.session_state["workflow_processes"]

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT job_id, status, pid
            FROM workflow_jobs
            WHERE status IN ('RUNNING', 'STOP_REQUESTED')
            """
        ).fetchall()
    finally:
        conn.close()

    for job_id, status, pid in rows:
        entry = managed.get(job_id)
        if entry is not None:
            proc = entry.get("process")
            if proc is None:
                continue
            exit_code = proc.poll()
            if exit_code is None:
                continue
            final_status = "COMPLETED" if exit_code == 0 else "FAILED"
            if status == "STOP_REQUESTED":
                final_status = "STOPPED" if exit_code == 0 else "FAILED"
            _update_workflow_job_row(
                db_path,
                job_id,
                status=final_status,
                ended_at=_utc_now_iso(),
                exit_code=int(exit_code),
            )
            managed.pop(job_id, None)
            continue

        if not _is_process_running(pid):
            final_status = "STOPPED" if status == "STOP_REQUESTED" else "EXITED"
            _update_workflow_job_row(
                db_path,
                job_id,
                status=final_status,
                ended_at=_utc_now_iso(),
            )


def _annotate_run_health(runs_df, stale_after_sec=DEFAULT_RUN_STALE_SEC):
    if runs_df.empty:
        return runs_df
    out = runs_df.copy()
    out = _coerce_datetime(out, "last_heartbeat_at")
    out = _coerce_datetime(out, "last_equity_at")
    now = pd.Timestamp.utcnow().tz_localize(None)
    effective = []
    telemetry_age = []

    for _, row in out.iterrows():
        status = str(row.get("status", "")).upper()
        latest = row.get("last_heartbeat_at")
        eq = row.get("last_equity_at")
        if pd.notna(eq) and (pd.isna(latest) or eq > latest):
            latest = eq

        age_sec = None
        if pd.notna(latest):
            age_sec = max(0.0, float((now - latest).total_seconds()))
        telemetry_age.append(age_sec)

        if status == "RUNNING":
            if age_sec is None:
                effective.append("RUNNING_NO_TELEMETRY")
            elif age_sec > float(stale_after_sec):
                effective.append("RUNNING_STALE")
            else:
                effective.append("RUNNING_HEALTHY")
        else:
            effective.append(status or "UNKNOWN")

    out["telemetry_age_sec"] = telemetry_age
    out["effective_status"] = effective
    return out


def _build_stop_file_path(job_id):
    os.makedirs(WORKFLOW_CONTROL_DIR, exist_ok=True)
    return os.path.join(WORKFLOW_CONTROL_DIR, f"{job_id}.stop")


def _launch_managed_job(
    *,
    db_path,
    workflow,
    script_name,
    script_args,
    env_overrides,
    requested_mode=None,
    strategy=None,
    run_id=None,
    stop_file=None,
    metadata=None,
):
    if "workflow_processes" not in st.session_state:
        st.session_state["workflow_processes"] = {}

    os.makedirs(WORKFLOW_LOG_DIR, exist_ok=True)
    job_id = str(uuid.uuid4())
    log_path = os.path.join(WORKFLOW_LOG_DIR, f"{workflow}_{job_id}.log")
    command = [sys.executable, script_name, *list(script_args)]
    env = os.environ.copy()
    env.update({k: str(v) for k, v in (env_overrides or {}).items()})

    started_at = _utc_now_iso()
    with open(log_path, "a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=os.getcwd(),
        )

    _insert_workflow_job_row(
        db_path,
        {
            "job_id": job_id,
            "workflow": str(workflow),
            "status": "RUNNING",
            "requested_mode": requested_mode,
            "strategy": strategy,
            "command_json": json.dumps(command, ensure_ascii=True),
            "env_json": json.dumps(env_overrides or {}, ensure_ascii=True),
            "pid": int(process.pid),
            "run_id": run_id,
            "started_at": started_at,
            "ended_at": None,
            "exit_code": None,
            "log_path": log_path,
            "stop_file": stop_file,
            "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
            "last_updated": started_at,
        },
    )

    st.session_state["workflow_processes"][job_id] = {
        "process": process,
        "log_path": log_path,
        "stop_file": stop_file,
    }
    return job_id


def _request_job_stop(db_path, stop_file):
    if not stop_file:
        return False
    parent = os.path.dirname(stop_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(stop_file, "w", encoding="utf-8") as f:
        f.write(_utc_now_iso())
    return True


@st.cache_data
def load_runs(db_path, refresh_counter=0):
    _ = refresh_counter
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT
                r.run_id,
                r.mode,
                r.started_at,
                r.ended_at,
                r.status,
                r.metadata,
                COALESCE(eq.equity_rows, 0) AS equity_rows,
                COALESCE(fl.fill_rows, 0) AS fill_rows,
                COALESCE(od.order_rows, 0) AS order_rows,
                COALESCE(rk.risk_rows, 0) AS risk_rows,
                COALESCE(hb.hb_rows, 0) AS hb_rows,
                eq.last_equity_at,
                hb.last_heartbeat_at
            FROM runs r
            LEFT JOIN (
                SELECT run_id, COUNT(*) AS equity_rows, MAX(timeindex) AS last_equity_at
                FROM equity
                GROUP BY run_id
            ) eq ON eq.run_id = r.run_id
            LEFT JOIN (SELECT run_id, COUNT(*) AS fill_rows FROM fills GROUP BY run_id) fl ON fl.run_id = r.run_id
            LEFT JOIN (SELECT run_id, COUNT(*) AS order_rows FROM orders GROUP BY run_id) od ON od.run_id = r.run_id
            LEFT JOIN (SELECT run_id, COUNT(*) AS risk_rows FROM risk_events GROUP BY run_id) rk ON rk.run_id = r.run_id
            LEFT JOIN (
                SELECT run_id, COUNT(*) AS hb_rows, MAX(heartbeat_time) AS last_heartbeat_at
                FROM heartbeats
                GROUP BY run_id
            ) hb ON hb.run_id = r.run_id
            ORDER BY r.started_at DESC
            LIMIT 300
            """,
            conn,
        )
        df = _coerce_datetime(df, "started_at")
        df = _coerce_datetime(df, "ended_at")
        df = _coerce_datetime(df, "last_equity_at")
        df = _coerce_datetime(df, "last_heartbeat_at")
        return df
    finally:
        conn.close()


@st.cache_data
def load_equity_sqlite(db_path, run_id, refresh_counter=0, max_points=DEFAULT_WINDOW_POINTS):
    _ = refresh_counter
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT timeindex AS datetime, total, cash, metadata
            FROM (
                SELECT id, timeindex, total, cash, metadata
                FROM equity
                WHERE run_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[run_id, int(max(1, max_points))],
        )
        return _coerce_datetime(df, "datetime")
    finally:
        conn.close()


@st.cache_data
def load_metrics_sqlite(db_path, run_id, refresh_counter=0):
    _ = refresh_counter
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT
                timeindex AS datetime,
                total,
                cash,
                metadata
            FROM equity
            WHERE run_id = ?
            ORDER BY id ASC
            """,
            conn,
            params=[run_id],
        )
        df = _coerce_datetime(df, "datetime")
        if df.empty:
            return df

        benchmark = []
        funding = []
        symbol = []
        for meta in df["metadata"].tolist() if "metadata" in df.columns else []:
            info = _parse_json_dict(meta)
            benchmark.append(info.get("benchmark_price"))
            funding.append(info.get("funding_total"))
            symbol.append(info.get("symbol"))

        if benchmark:
            df["benchmark_price"] = pd.to_numeric(pd.Series(benchmark), errors="coerce")
        if funding:
            df["funding"] = pd.to_numeric(pd.Series(funding), errors="coerce").fillna(0.0)
        if symbol:
            df["event_symbol"] = pd.Series(symbol)
        return df
    finally:
        conn.close()


@st.cache_data
def load_fills_sqlite(db_path, run_id, refresh_counter=0, max_points=DEFAULT_WINDOW_POINTS):
    _ = refresh_counter
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT
                fill_time AS datetime,
                symbol,
                side AS direction,
                quantity,
                fill_cost,
                commission,
                fill_price AS price,
                status,
                metadata,
                exchange_order_id,
                client_order_id
            FROM (
                SELECT id, fill_time, symbol, side, quantity, fill_cost, commission,
                       fill_price, status, metadata, exchange_order_id, client_order_id
                FROM fills
                WHERE run_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[run_id, int(max(1, max_points))],
        )
        if not df.empty and "direction" in df.columns:
            df["direction"] = (
                df["direction"]
                .fillna("")
                .astype(str)
                .str.upper()
                .map({"BUY": "BUY", "SELL": "SELL", "BUY_LONG": "BUY", "SELL_SHORT": "SELL"})
                .fillna(df["direction"])
            )
        return _coerce_datetime(df, "datetime")
    finally:
        conn.close()


@st.cache_data
def load_orders_sqlite(db_path, run_id, refresh_counter=0, max_points=DEFAULT_WINDOW_POINTS):
    _ = refresh_counter
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT created_at, symbol, side, order_type, quantity, price, status,
                   client_order_id, exchange_order_id, metadata
            FROM (
                SELECT id, created_at, symbol, side, order_type, quantity, price, status,
                       client_order_id, exchange_order_id, metadata
                FROM orders
                WHERE run_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[run_id, int(max(1, max_points))],
        )
        return _coerce_datetime(df, "created_at")
    finally:
        conn.close()


@st.cache_data
def load_risk_events_sqlite(db_path, run_id, refresh_counter=0, max_points=5000):
    _ = refresh_counter
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT event_time, reason, details
            FROM (
                SELECT id, event_time, reason, details
                FROM risk_events
                WHERE run_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[run_id, int(max(1, max_points))],
        )
        return _coerce_datetime(df, "event_time")
    finally:
        conn.close()


@st.cache_data
def load_heartbeats_sqlite(db_path, run_id, refresh_counter=0, max_points=5000):
    _ = refresh_counter
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT heartbeat_time, status, details
            FROM (
                SELECT id, heartbeat_time, status, details
                FROM heartbeats
                WHERE run_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[run_id, int(max(1, max_points))],
        )
        return _coerce_datetime(df, "heartbeat_time")
    finally:
        conn.close()


@st.cache_data
def load_order_states_sqlite(db_path, run_id, refresh_counter=0, max_points=10000):
    _ = refresh_counter
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT event_time, symbol, client_order_id, exchange_order_id, state, message, details
            FROM (
                SELECT id, event_time, symbol, client_order_id, exchange_order_id, state, message, details
                FROM order_state_events
                WHERE run_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[run_id, int(max(1, max_points))],
        )
        return _coerce_datetime(df, "event_time")
    finally:
        conn.close()


@st.cache_data
def load_optimization_results_sqlite(db_path, refresh_counter=0, max_points=10000):
    _ = refresh_counter
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT id, run_id, stage, created_at, params_json, sharpe, cagr, mdd,
                   train_sharpe, robustness_score, extra_json
            FROM (
                SELECT id, run_id, stage, created_at, params_json, sharpe, cagr, mdd,
                       train_sharpe, robustness_score, extra_json
                FROM optimization_results
                ORDER BY id DESC
                LIMIT ?
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[int(max(1, max_points))],
        )
        if df.empty:
            return df
        df = _coerce_datetime(df, "created_at")
        df["params"] = df["params_json"].apply(_parse_json_dict)
        df["extra"] = df["extra_json"].apply(_parse_json_dict)
        return df
    except sqlite3.OperationalError:
        return pd.DataFrame()
    finally:
        conn.close()


@st.cache_data
def load_market_ohlcv_sqlite(
    db_path,
    symbol,
    timeframe,
    exchange_id,
    refresh_counter=0,
    max_points=DEFAULT_WINDOW_POINTS,
):
    _ = refresh_counter
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT datetime, open, high, low, close, volume
            FROM (
                SELECT timestamp_ms, datetime, open, high, low, close, volume
                FROM market_ohlcv
                WHERE exchange = ? AND symbol = ? AND timeframe = ?
                ORDER BY timestamp_ms DESC
                LIMIT ?
            ) recent
            ORDER BY datetime ASC
            """,
            conn,
            params=[str(exchange_id).lower(), symbol, timeframe, int(max(1, max_points))],
        )
        return _coerce_datetime(df, "datetime")
    finally:
        conn.close()


@st.cache_data
def load_equity_csv(refresh_counter=0, max_points=DEFAULT_WINDOW_POINTS):
    _ = refresh_counter
    if not os.path.exists("equity.csv"):
        return pd.DataFrame()
    df = pd.read_csv("equity.csv")
    df = _coerce_datetime(df, "datetime")
    if max_points > 0 and len(df) > max_points:
        df = df.tail(max_points).reset_index(drop=True)
    return df


@st.cache_data
def load_trades_csv(refresh_counter=0, max_points=DEFAULT_WINDOW_POINTS):
    _ = refresh_counter
    if not os.path.exists("trades.csv"):
        return pd.DataFrame()
    df = pd.read_csv("trades.csv")
    df = _coerce_datetime(df, "datetime")
    if max_points > 0 and len(df) > max_points:
        df = df.tail(max_points).reset_index(drop=True)
    return df


@st.cache_data
def load_params(strategy_name):
    path = os.path.join("best_optimized_parameters", strategy_name, "best_params.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _strategy_default_params(strategy_name):
    return get_default_strategy_params(strategy_name)


def _strategy_default_optuna(strategy_name):
    return get_default_optuna_config(strategy_name)


def _strategy_default_grid(strategy_name):
    return get_default_grid_config(strategy_name)


def _merged_strategy_params(strategy_name, loaded_params):
    params = _strategy_default_params(strategy_name)
    if isinstance(loaded_params, dict):
        params.update(loaded_params)
    return params


def _save_strategy_params(strategy_name, params):
    out_dir = os.path.join("best_optimized_parameters", strategy_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "best_params.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    return out_path


def _parse_symbols_csv(raw_symbols):
    return [token.strip() for token in str(raw_symbols).split(",") if token.strip()]


def _build_runtime_env_overrides(
    *,
    initial_capital,
    leverage,
    timeframe,
    symbols,
    strategy_name,
    optuna_config,
    grid_config,
):
    return {
        "LQ__TRADING__INITIAL_CAPITAL": str(float(initial_capital)),
        "LQ__BACKTEST__LEVERAGE": str(int(leverage)),
        "LQ__TRADING__TIMEFRAME": str(timeframe),
        "LQ__TRADING__SYMBOLS": json.dumps(list(symbols), ensure_ascii=True),
        "LQ__OPTIMIZATION__STRATEGY": str(strategy_name),
        "LQ__OPTIMIZATION__OPTUNA": json.dumps(optuna_config, ensure_ascii=True),
        "LQ__OPTIMIZATION__GRID": json.dumps(grid_config, ensure_ascii=True),
    }


def _run_python_script(script_name, script_args, env_overrides, timeout_sec):
    cmd = [sys.executable, script_name, *script_args]
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})
    started = time.time()
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=max(30, int(timeout_sec)),
            check=False,
        )
        elapsed = time.time() - started
        output = completed.stdout or ""
        if completed.stderr:
            output += "\n" + completed.stderr
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "elapsed_sec": elapsed,
            "command": cmd,
            "output": output.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - started
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", errors="replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", errors="replace")
        if err:
            out += "\n" + err
        return {
            "ok": False,
            "returncode": None,
            "elapsed_sec": elapsed,
            "command": cmd,
            "output": out.strip(),
            "timed_out": True,
        }


def _run_ghost_cleanup_script(
    *,
    db_path,
    stale_sec,
    startup_grace_sec,
    close_status,
    force_kill_stop_requested_after_sec,
    apply_changes,
):
    cmd = [
        sys.executable,
        "scripts/cleanup_ghost_runs.py",
        "--db",
        str(db_path),
        "--stale-sec",
        str(int(stale_sec)),
        "--startup-grace-sec",
        str(int(startup_grace_sec)),
        "--close-status",
        str(close_status),
        "--force-kill-stop-requested-after-sec",
        str(int(force_kill_stop_requested_after_sec)),
    ]
    if apply_changes:
        cmd.append("--apply")

    started = time.time()
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.time() - started

    payload = None
    stdout_text = (completed.stdout or "").strip()
    stderr_text = (completed.stderr or "").strip()
    if stdout_text:
        try:
            payload = json.loads(stdout_text)
        except Exception:
            payload = None

    output = stdout_text
    if stderr_text:
        output = (output + "\n" + stderr_text).strip()

    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "elapsed_sec": elapsed,
        "command": cmd,
        "output": output,
        "payload": payload,
    }


def _parse_json_dict(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _compute_rsi_series(close_series, period):
    try:
        period = max(2, int(period))
    except Exception:
        period = 14
    close = pd.to_numeric(close_series, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi = rsi.where(~((avg_loss <= 0.0) & (avg_gain > 0.0)), 100.0)
    rsi = rsi.where(~((avg_loss <= 0.0) & (avg_gain <= 0.0)), 0.0)
    return rsi


def _build_strategy_indicator_frame(market_df, strategy_name, strategy_params):
    if market_df.empty or "datetime" not in market_df.columns or "close" not in market_df.columns:
        return pd.DataFrame()

    frame = market_df[["datetime", "close"]].copy()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["close"]).sort_values("datetime").reset_index(drop=True)
    if frame.empty:
        return frame

    if strategy_name == "RsiStrategy":
        period = strategy_params.get("rsi_period", 14)
        frame["rsi"] = _compute_rsi_series(frame["close"], period)
    elif strategy_name == "MovingAverageCrossStrategy":
        try:
            short_window = max(2, int(strategy_params.get("short_window", 10)))
        except Exception:
            short_window = 10
        try:
            long_window = int(strategy_params.get("long_window", 30))
        except Exception:
            long_window = 30
        long_window = max(short_window + 1, long_window)
        frame["short_ma"] = (
            frame["close"]
            .rolling(
                window=short_window,
                min_periods=short_window,
            )
            .mean()
        )
        frame["long_ma"] = (
            frame["close"]
            .rolling(
                window=long_window,
                min_periods=long_window,
            )
            .mean()
        )

    return frame


def _build_pair_indicator_frame(market_x_df, market_y_df, strategy_params):
    if market_x_df.empty or market_y_df.empty:
        return pd.DataFrame()
    if "datetime" not in market_x_df.columns or "datetime" not in market_y_df.columns:
        return pd.DataFrame()
    if "close" not in market_x_df.columns or "close" not in market_y_df.columns:
        return pd.DataFrame()

    frame_x = market_x_df[["datetime", "close"]].copy()
    frame_x = frame_x.rename(columns={"close": "close_x"})
    frame_x["close_x"] = pd.to_numeric(frame_x["close_x"], errors="coerce")

    frame_y = market_y_df[["datetime", "close"]].copy()
    frame_y = frame_y.rename(columns={"close": "close_y"})
    frame_y["close_y"] = pd.to_numeric(frame_y["close_y"], errors="coerce")

    merged = pd.merge(frame_x, frame_y, on="datetime", how="inner")
    merged = merged.dropna(subset=["close_x", "close_y"]).sort_values("datetime")
    merged = merged.reset_index(drop=True)
    if merged.empty:
        return merged

    lookback = max(10, int(strategy_params.get("lookback_window", 96)))
    hedge_window = max(lookback, int(strategy_params.get("hedge_window", 192)))
    use_log_price = bool(strategy_params.get("use_log_price", True))

    if use_log_price:
        merged["x_price"] = np.log(merged["close_x"])
        merged["y_price"] = np.log(merged["close_y"])
    else:
        merged["x_price"] = merged["close_x"]
        merged["y_price"] = merged["close_y"]

    rolling_cov = (
        merged["x_price"]
        .rolling(window=hedge_window, min_periods=hedge_window)
        .cov(merged["y_price"])
    )
    rolling_var = merged["y_price"].rolling(window=hedge_window, min_periods=hedge_window).var()
    merged["hedge_ratio"] = rolling_cov / rolling_var.replace(0.0, np.nan)
    merged["spread"] = merged["x_price"] - (merged["hedge_ratio"] * merged["y_price"])

    merged["spread_mean"] = merged["spread"].rolling(window=lookback, min_periods=lookback).mean()
    merged["spread_std"] = merged["spread"].rolling(window=lookback, min_periods=lookback).std()
    merged["zscore"] = (merged["spread"] - merged["spread_mean"]) / merged["spread_std"].replace(
        0.0, np.nan
    )
    merged["correlation"] = (
        merged["x_price"].rolling(window=lookback, min_periods=lookback).corr(merged["y_price"])
    )
    return merged


def _latest_running_run_id(runs_df):
    if runs_df.empty or "status" not in runs_df.columns:
        return None
    running = runs_df[runs_df["status"].fillna("").astype(str).str.upper() == "RUNNING"]
    if running.empty:
        return None
    return str(running.iloc[0]["run_id"])


def _latest_run_with_equity(runs_df):
    if runs_df.empty or "equity_rows" not in runs_df.columns:
        return None
    with_data = runs_df[pd.to_numeric(runs_df["equity_rows"], errors="coerce").fillna(0) > 0]
    if with_data.empty:
        return None
    return str(with_data.iloc[0]["run_id"])


def _equity_rows_for_run(runs_df, run_id):
    if runs_df.empty or not run_id:
        return 0
    matched = runs_df.loc[runs_df["run_id"] == run_id, "equity_rows"]
    if matched.empty:
        return 0
    try:
        return int(float(matched.iloc[0]))
    except Exception:
        return 0


def _extract_run_symbols(runs_df, run_id):
    if runs_df.empty or not run_id or "metadata" not in runs_df.columns:
        return list(BaseConfig.SYMBOLS)
    row = runs_df.loc[runs_df["run_id"] == run_id]
    if row.empty:
        return list(BaseConfig.SYMBOLS)
    metadata = _parse_json_dict(row.iloc[0].get("metadata"))
    symbols = metadata.get("symbols")
    if isinstance(symbols, list) and symbols:
        return [str(s) for s in symbols]
    return list(BaseConfig.SYMBOLS)


def _compute_data_latency_seconds(df):
    if df.empty or "datetime" not in df.columns:
        return None, None
    latest_ts = pd.to_datetime(df["datetime"], errors="coerce").dropna()
    if latest_ts.empty:
        return None, None
    latest = latest_ts.max()
    now = pd.Timestamp.utcnow().tz_localize(None)
    return latest, max(0.0, float((now - latest).total_seconds()))


def _resolve_period_window(
    period_preset, custom_start, custom_end, reference_df, dt_col="datetime"
):
    if period_preset == "All":
        return None, None

    ref_end = None
    if not reference_df.empty and dt_col in reference_df.columns:
        ts = pd.to_datetime(reference_df[dt_col], errors="coerce").dropna()
        if not ts.empty:
            ref_end = ts.max()
    if ref_end is None:
        ref_end = pd.Timestamp.utcnow().tz_localize(None)

    if period_preset == "Custom":
        start_ts = pd.Timestamp(custom_start) if custom_start else None
        end_ts = pd.Timestamp(custom_end) if custom_end else None
        if end_ts is not None:
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        return start_ts, end_ts

    days_map = {
        "1D": 1,
        "7D": 7,
        "30D": 30,
        "90D": 90,
        "180D": 180,
        "365D": 365,
    }
    days = days_map.get(period_preset)
    if days is None:
        return None, None
    return ref_end - pd.Timedelta(days=days), ref_end


def _apply_period_filter(df, dt_col, start_ts, end_ts):
    if df.empty or dt_col not in df.columns:
        return df
    out = df.copy()
    ts = pd.to_datetime(out[dt_col], errors="coerce")
    mask = ts.notna()
    if start_ts is not None:
        mask = mask & (ts >= start_ts)
    if end_ts is not None:
        mask = mask & (ts <= end_ts)
    return out.loc[mask].reset_index(drop=True)


def _safe_float(value, default=0.0):
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _safe_div(numerator, denominator, default=0.0):
    den = _safe_float(denominator, 0.0)
    if den == 0.0:
        return float(default)
    return float(_safe_float(numerator, 0.0) / den)


def _format_duration_seconds(value):
    sec = _safe_float(value, 0.0)
    if sec <= 0.0:
        return "0s"
    total = round(sec)
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _metric_value(metric_name, performance, summary):
    if isinstance(performance, dict) and metric_name in performance:
        return performance.get(metric_name)

    if not isinstance(summary, dict):
        return None

    mapping = {
        "Total Net Profit": summary.get("total_net_profit"),
        "Open P/L": summary.get("open_pnl"),
        "Total (C+O)": summary.get("total_c_plus_o"),
        "Gross Profit": summary.get("gross_profit"),
        "Gross Loss": summary.get("gross_loss"),
        "Profit Factor": summary.get("profit_factor"),
        "Expected Payoff": summary.get("expected_payoff"),
        "Recovery Factor": summary.get("recovery_factor"),
        "R/MDD": summary.get("r_mdd"),
        "Profit Trades (% of total)": summary.get("profit_trades_text"),
        "Loss Trades (% of total)": summary.get("loss_trades_text"),
        "Avg Profit Trade": summary.get("avg_profit_trade"),
        "Avg Loss Trade": summary.get("avg_loss_trade"),
        "Payoff Ratio": summary.get("payoff_ratio"),
        "AHPR": summary.get("ahpr"),
        "GHPR": summary.get("ghpr"),
        "LR Correlation": summary.get("lr_correlation"),
        "LR Std Error": summary.get("lr_std_error"),
        "Z-Score": summary.get("z_score"),
        "Long Trades (Win %)": summary.get("long_trades_win_pct"),
        "Short Trades (Win %)": summary.get("short_trades_win_pct"),
        "Avg Holding Time": summary.get("holding_time_avg_sec"),
        "Max Holding Time": summary.get("holding_time_max_sec"),
        "Min Holding Time": summary.get("holding_time_min_sec"),
        "Equity Drawdown Absolute": summary.get("equity_drawdown_absolute"),
        "Equity Drawdown Maximal": summary.get("equity_drawdown_maximal"),
        "Equity Drawdown Relative %": summary.get("equity_drawdown_relative_pct"),
        "Balance Drawdown Absolute": summary.get("balance_drawdown_absolute"),
        "Balance Drawdown Maximal": summary.get("balance_drawdown_maximal"),
        "Balance Drawdown Relative %": summary.get("balance_drawdown_relative_pct"),
    }
    return mapping.get(metric_name)


def _drawdown_stats(values, initial_value):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "absolute": 0.0,
            "maximal": 0.0,
            "relative_pct": 0.0,
        }

    peak = np.maximum.accumulate(arr)
    drawdown_abs = peak - arr
    drawdown_rel = np.divide(
        drawdown_abs,
        np.where(peak == 0.0, np.nan, peak),
        dtype=np.float64,
    )
    drawdown_rel = np.nan_to_num(drawdown_rel, nan=0.0, posinf=0.0, neginf=0.0)

    absolute = max(0.0, _safe_float(initial_value, 0.0) - float(np.min(arr)))
    maximal = float(np.max(drawdown_abs))
    relative = float(np.max(drawdown_rel))
    return {
        "absolute": float(absolute),
        "maximal": float(maximal),
        "relative_pct": float(relative),
    }


def _streak_groups(sequence):
    groups = []
    if not sequence:
        return groups
    current = sequence[0]
    count = 1
    for value in sequence[1:]:
        if value == current:
            count += 1
            continue
        groups.append((current, count))
        current = value
        count = 1
    groups.append((current, count))
    return groups


def _runs_test_zscore(binary_outcomes):
    if not binary_outcomes:
        return 0.0
    outcomes = [bool(value) for value in binary_outcomes]
    wins = int(sum(outcomes))
    losses = int(len(outcomes) - wins)
    if wins == 0 or losses == 0:
        return 0.0

    runs = 1
    for idx in range(1, len(outcomes)):
        if outcomes[idx] != outcomes[idx - 1]:
            runs += 1

    total = wins + losses
    expected_runs = 1.0 + (2.0 * wins * losses / float(total))
    variance = (
        2.0
        * wins
        * losses
        * ((2.0 * wins * losses) - wins - losses)
        / float((total**2) * (total - 1))
    )
    if variance <= 0.0:
        return 0.0

    diff = float(runs) - expected_runs
    correction = 0.0
    if diff > 0.0:
        correction = 0.5
    elif diff < 0.0:
        correction = -0.5
    return float((diff - correction) / math.sqrt(variance))


def _linear_regression_diagnostics(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return 0.0, 0.0

    x = np.arange(arr.size, dtype=np.float64)
    x_center = x - float(np.mean(x))
    y_center = arr - float(np.mean(arr))
    denom = float(np.sum(x_center**2))
    if denom <= 0.0:
        return 0.0, 0.0

    slope = float(np.sum(x_center * y_center) / denom)
    intercept = float(np.mean(arr) - slope * np.mean(x))
    y_hat = intercept + (slope * x)
    residuals = arr - y_hat
    sse = float(np.sum(residuals**2))
    stderr = math.sqrt(sse / float(max(1, arr.size - 2)))

    y_std = float(np.std(arr))
    if y_std <= 0.0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(x, arr)[0, 1])
        if not math.isfinite(corr):
            corr = 0.0
    return corr, float(stderr)


def _build_monthly_returns_table(df_equity, performance):
    if (
        not isinstance(performance, dict)
        or "return_series" not in performance
        or df_equity.empty
        or "datetime" not in df_equity.columns
    ):
        return pd.DataFrame()

    returns = pd.Series(performance.get("return_series", []), dtype="float64")
    if returns.empty:
        return pd.DataFrame()

    timestamps = pd.to_datetime(df_equity["datetime"], errors="coerce").iloc[1:]
    min_len = min(len(returns), len(timestamps))
    if min_len <= 0:
        return pd.DataFrame()

    frame = pd.DataFrame(
        {
            "datetime": timestamps.iloc[-min_len:].to_numpy(),
            "ret": returns.iloc[-min_len:].to_numpy(),
        }
    )
    frame = frame.dropna(subset=["datetime", "ret"])
    if frame.empty:
        return pd.DataFrame()

    frame["year"] = frame["datetime"].dt.year
    frame["month"] = frame["datetime"].dt.month
    monthly = (
        frame.groupby(["year", "month"], observed=False)["ret"]
        .apply(lambda s: float(np.prod(1.0 + s.to_numpy(dtype=np.float64)) - 1.0))
        .unstack("month")
    )
    if monthly.empty:
        return pd.DataFrame()
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    monthly = monthly.reindex(columns=list(range(1, 13)))
    monthly.columns = month_names
    return monthly


def _build_mt5_summary_rows(summary):
    if not isinstance(summary, dict):
        return pd.DataFrame()

    rows = [
        {
            "Section": "Profitability",
            "Metric": "Total Net Profit",
            "Value": _format_metric_value("Total Net Profit", summary.get("total_net_profit")),
        },
        {
            "Section": "Profitability",
            "Metric": "Open P/L",
            "Value": _format_metric_value("Open P/L", summary.get("open_pnl")),
        },
        {
            "Section": "Profitability",
            "Metric": "Total (C+O)",
            "Value": _format_metric_value("Total (C+O)", summary.get("total_c_plus_o")),
        },
        {
            "Section": "Profitability",
            "Metric": "Gross Profit",
            "Value": _format_metric_value("Gross Profit", summary.get("gross_profit")),
        },
        {
            "Section": "Profitability",
            "Metric": "Gross Loss",
            "Value": _format_metric_value("Gross Loss", summary.get("gross_loss")),
        },
        {
            "Section": "Profitability",
            "Metric": "Profit Factor",
            "Value": _format_metric_value("Profit Factor", summary.get("profit_factor")),
        },
        {
            "Section": "Profitability",
            "Metric": "Expected Payoff",
            "Value": _format_metric_value("Expected Payoff", summary.get("expected_payoff")),
        },
        {
            "Section": "Profitability",
            "Metric": "Recovery Factor",
            "Value": _format_metric_value("Recovery Factor", summary.get("recovery_factor")),
        },
        {
            "Section": "Profitability",
            "Metric": "R/MDD",
            "Value": _format_metric_value("R/MDD", summary.get("r_mdd")),
        },
        {
            "Section": "Trade Quality",
            "Metric": "AHPR",
            "Value": _format_metric_value("AHPR", summary.get("ahpr")),
        },
        {
            "Section": "Trade Quality",
            "Metric": "GHPR",
            "Value": _format_metric_value("GHPR", summary.get("ghpr")),
        },
        {
            "Section": "Trade Quality",
            "Metric": "LR Correlation",
            "Value": _format_metric_value("LR Correlation", summary.get("lr_correlation")),
        },
        {
            "Section": "Trade Quality",
            "Metric": "LR Std Error",
            "Value": _format_metric_value("LR Std Error", summary.get("lr_std_error")),
        },
        {
            "Section": "Trade Quality",
            "Metric": "Z-Score",
            "Value": _format_metric_value("Z-Score", summary.get("z_score")),
        },
        {
            "Section": "Direction",
            "Metric": "Long Trades (Win %)",
            "Value": _format_metric_value(
                "Long Trades (Win %)", summary.get("long_trades_win_pct")
            ),
        },
        {
            "Section": "Direction",
            "Metric": "Short Trades (Win %)",
            "Value": _format_metric_value(
                "Short Trades (Win %)", summary.get("short_trades_win_pct")
            ),
        },
        {
            "Section": "Direction",
            "Metric": "Profit Trades (% of total)",
            "Value": _format_metric_value(
                "Profit Trades (% of total)", summary.get("profit_trades_text")
            ),
        },
        {
            "Section": "Direction",
            "Metric": "Loss Trades (% of total)",
            "Value": _format_metric_value(
                "Loss Trades (% of total)", summary.get("loss_trades_text")
            ),
        },
        {
            "Section": "Direction",
            "Metric": "Avg Profit Trade",
            "Value": _format_metric_value("Avg Profit Trade", summary.get("avg_profit_trade")),
        },
        {
            "Section": "Direction",
            "Metric": "Avg Loss Trade",
            "Value": _format_metric_value("Avg Loss Trade", summary.get("avg_loss_trade")),
        },
        {
            "Section": "Direction",
            "Metric": "Payoff Ratio",
            "Value": _format_metric_value("Payoff Ratio", summary.get("payoff_ratio")),
        },
        {
            "Section": "Holding",
            "Metric": "Min Holding Time",
            "Value": _format_metric_value("Min Holding Time", summary.get("holding_time_min_sec")),
        },
        {
            "Section": "Holding",
            "Metric": "Avg Holding Time",
            "Value": _format_metric_value("Avg Holding Time", summary.get("holding_time_avg_sec")),
        },
        {
            "Section": "Holding",
            "Metric": "Max Holding Time",
            "Value": _format_metric_value("Max Holding Time", summary.get("holding_time_max_sec")),
        },
        {
            "Section": "Drawdown",
            "Metric": "Equity Drawdown Absolute",
            "Value": _format_metric_value(
                "Equity Drawdown Absolute", summary.get("equity_drawdown_absolute")
            ),
        },
        {
            "Section": "Drawdown",
            "Metric": "Equity Drawdown Maximal",
            "Value": _format_metric_value(
                "Equity Drawdown Maximal", summary.get("equity_drawdown_maximal")
            ),
        },
        {
            "Section": "Drawdown",
            "Metric": "Equity Drawdown Relative %",
            "Value": _format_metric_value(
                "Equity Drawdown Relative %", summary.get("equity_drawdown_relative_pct")
            ),
        },
        {
            "Section": "Drawdown",
            "Metric": "Balance Drawdown Absolute",
            "Value": _format_metric_value(
                "Balance Drawdown Absolute", summary.get("balance_drawdown_absolute")
            ),
        },
        {
            "Section": "Drawdown",
            "Metric": "Balance Drawdown Maximal",
            "Value": _format_metric_value(
                "Balance Drawdown Maximal", summary.get("balance_drawdown_maximal")
            ),
        },
        {
            "Section": "Drawdown",
            "Metric": "Balance Drawdown Relative %",
            "Value": _format_metric_value(
                "Balance Drawdown Relative %", summary.get("balance_drawdown_relative_pct")
            ),
        },
    ]
    return pd.DataFrame(rows)


def _format_metric_value(name, value):
    if value is None:
        return "N/A"
    if name in {
        "Long Trades (Win %)",
        "Short Trades (Win %)",
        "Profit Trades (% of total)",
        "Loss Trades (% of total)",
    }:
        return str(value)
    if name in {"Avg Holding Time", "Max Holding Time", "Min Holding Time"}:
        return _format_duration_seconds(value)
    if name in {
        "Total Return",
        "Cumulative Return",
        "CAGR",
        "Ann. Volatility",
        "Max Drawdown",
        "Win Rate",
        "Equity Drawdown Relative %",
        "Balance Drawdown Relative %",
    }:
        return f"{_safe_float(value):.2%}"
    if name == "Profit Factor":
        try:
            parsed = float(value)
        except Exception:
            parsed = 0.0
        return "inf" if math.isinf(parsed) else f"{parsed:.3f}"
    if name in {"Funding (Net)", "Alpha", "Beta", "Information Ratio"}:
        return f"{_safe_float(value):.6f}"
    if name in {
        "Total Net Profit",
        "Open P/L",
        "Total (C+O)",
        "Gross Profit",
        "Gross Loss",
        "Expected Payoff",
        "Avg Profit Trade",
        "Avg Loss Trade",
        "Equity Drawdown Absolute",
        "Equity Drawdown Maximal",
        "Balance Drawdown Absolute",
        "Balance Drawdown Maximal",
    }:
        return f"{_safe_float(value):.4f}"
    if name in {"R/MDD", "Payoff Ratio"}:
        return f"{_safe_float(value):.4f}"
    if name == "DD Duration":
        return str(int(_safe_float(value)))
    return f"{_safe_float(value):.4f}"


def _downsample_frame(df, target_points):
    if df.empty:
        return df
    target = max(1, int(target_points))
    total = len(df)
    if total <= target:
        return df
    step = max(1, total // target)
    indices = list(range(0, total, step))
    if indices[-1] != total - 1:
        indices.append(total - 1)
    return df.iloc[indices].reset_index(drop=True)


def compute_trade_analytics(df_trades):
    if df_trades.empty:
        return df_trades
    df = df_trades.copy().sort_values("datetime").reset_index(drop=True)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for col in ["quantity", "price", "fill_cost", "commission"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
    if "direction" not in df.columns:
        df["direction"] = ""
    if "symbol" not in df.columns:
        df["symbol"] = "UNKNOWN"

    positions = {}
    avg_cost = {}
    entry_times = {}
    realized_pnl = []
    realized_return = []
    position_after = []
    avg_cost_after = []
    closed_qty = []
    close_side = []
    holding_sec = []

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

            entry_basis = abs(avg * closes)
            exit_basis = abs(price * closes)
            basis = max(entry_basis, exit_basis)
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
    df["realized_return_pct"] = pd.Series(realized_return) * 100.0
    df["position_after"] = position_after
    df["avg_cost_after"] = avg_cost_after
    df["closed_qty"] = closed_qty
    df["close_side"] = close_side
    df["holding_sec"] = holding_sec
    df["cum_realized_pnl"] = df["realized_pnl"].cumsum()
    df["notional"] = df["quantity"] * df["price"]
    return df


def build_summary(df_equity, df_trades):
    out = {
        "period_start": None,
        "period_end": None,
        "period_days": 0.0,
        "bars": len(df_equity),
        "fills": len(df_trades),
        "buy_fills": 0,
        "sell_fills": 0,
        "fills_per_day": 0.0,
        "avg_qty": 0.0,
        "avg_notional": 0.0,
        "total_commission": 0.0,
        "realized_pnl": 0.0,
        "win_rate": 0.0,
        "avg_trade_return_pct": 0.0,
        "best_trade_pnl": 0.0,
        "worst_trade_pnl": 0.0,
        "initial_equity": 0.0,
        "final_equity": 0.0,
        "total_net_profit": 0.0,
        "open_pnl": 0.0,
        "total_c_plus_o": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "profit_factor": 0.0,
        "expected_payoff": 0.0,
        "recovery_factor": 0.0,
        "r_mdd": 0.0,
        "profit_trades_count": 0,
        "loss_trades_count": 0,
        "profit_trades_pct": 0.0,
        "loss_trades_pct": 0.0,
        "profit_trades_text": "0 (0.00%)",
        "loss_trades_text": "0 (0.00%)",
        "avg_profit_trade": 0.0,
        "avg_loss_trade": 0.0,
        "payoff_ratio": 0.0,
        "ahpr": 0.0,
        "ghpr": 0.0,
        "lr_correlation": 0.0,
        "lr_std_error": 0.0,
        "z_score": 0.0,
        "closed_trades": 0,
        "wins": 0,
        "losses": 0,
        "long_trades": 0,
        "short_trades": 0,
        "long_win_rate": 0.0,
        "short_win_rate": 0.0,
        "long_trades_win_pct": "0 (0.00%)",
        "short_trades_win_pct": "0 (0.00%)",
        "win_streak_max": 0,
        "loss_streak_max": 0,
        "win_streak_avg": 0.0,
        "loss_streak_avg": 0.0,
        "max_consecutive_profit_count": 0,
        "max_consecutive_loss_count": 0,
        "max_consecutive_profit_amount": 0.0,
        "max_consecutive_loss_amount": 0.0,
        "holding_time_min_sec": 0.0,
        "holding_time_avg_sec": 0.0,
        "holding_time_max_sec": 0.0,
        "equity_drawdown_absolute": 0.0,
        "equity_drawdown_maximal": 0.0,
        "equity_drawdown_relative_pct": 0.0,
        "balance_drawdown_absolute": 0.0,
        "balance_drawdown_maximal": 0.0,
        "balance_drawdown_relative_pct": 0.0,
    }
    totals_arr = np.array([], dtype=np.float64)
    if not df_equity.empty and "datetime" in df_equity.columns:
        dt = pd.to_datetime(df_equity["datetime"], errors="coerce").dropna()
        if not dt.empty:
            start = dt.min()
            end = dt.max()
            out["period_start"] = str(start)
            out["period_end"] = str(end)
            seconds = max(0.0, float((end - start).total_seconds()))
            out["period_days"] = seconds / 86400.0
    if not df_equity.empty and "total" in df_equity.columns:
        totals = pd.to_numeric(df_equity["total"], errors="coerce").dropna()
        if not totals.empty:
            out["initial_equity"] = float(totals.iloc[0])
            out["final_equity"] = float(totals.iloc[-1])
            out["total_c_plus_o"] = float(out["final_equity"] - out["initial_equity"])
            totals_arr = totals.to_numpy(dtype=np.float64)
            dd_stats = _drawdown_stats(totals_arr, out["initial_equity"])
            out["equity_drawdown_absolute"] = float(dd_stats["absolute"])
            out["equity_drawdown_maximal"] = float(dd_stats["maximal"])
            out["equity_drawdown_relative_pct"] = float(dd_stats["relative_pct"])
            corr, stderr = _linear_regression_diagnostics(totals_arr)
            out["lr_correlation"] = float(corr)
            out["lr_std_error"] = float(stderr)

    if not df_trades.empty:
        out["buy_fills"] = (
            int((df_trades["direction"] == "BUY").sum()) if "direction" in df_trades.columns else 0
        )
        out["sell_fills"] = (
            int((df_trades["direction"] == "SELL").sum()) if "direction" in df_trades.columns else 0
        )
        out["avg_qty"] = (
            float(df_trades["quantity"].mean()) if "quantity" in df_trades.columns else 0.0
        )
        out["avg_notional"] = (
            float(df_trades["notional"].mean()) if "notional" in df_trades.columns else 0.0
        )
        out["total_commission"] = (
            float(df_trades["commission"].sum()) if "commission" in df_trades.columns else 0.0
        )
        out["realized_pnl"] = (
            float(df_trades["realized_pnl"].sum()) if "realized_pnl" in df_trades.columns else 0.0
        )
        out["total_net_profit"] = float(out["realized_pnl"])
        closed = (
            df_trades[df_trades["closed_qty"] > 0]
            if "closed_qty" in df_trades.columns
            else pd.DataFrame()
        )
        if not closed.empty:
            pnl_series = pd.to_numeric(closed.get("realized_pnl"), errors="coerce").fillna(0.0)
            out["closed_trades"] = len(closed)
            out["wins"] = int((pnl_series > 0.0).sum())
            out["losses"] = int((pnl_series < 0.0).sum())
            out["profit_trades_count"] = int((pnl_series > 0.0).sum())
            out["loss_trades_count"] = int((pnl_series < 0.0).sum())
            denom = max(1, out["wins"] + out["losses"])
            out["win_rate"] = float(out["wins"] / float(denom))
            out["profit_trades_pct"] = float(
                out["profit_trades_count"] / float(out["closed_trades"])
            )
            out["loss_trades_pct"] = float(out["loss_trades_count"] / float(out["closed_trades"]))
            out["profit_trades_text"] = (
                f"{out['profit_trades_count']} ({out['profit_trades_pct']:.2%})"
            )
            out["loss_trades_text"] = f"{out['loss_trades_count']} ({out['loss_trades_pct']:.2%})"
            if "realized_return_pct" in closed.columns:
                out["avg_trade_return_pct"] = float(
                    pd.to_numeric(closed["realized_return_pct"], errors="coerce").mean()
                )
            out["best_trade_pnl"] = float(pnl_series.max())
            out["worst_trade_pnl"] = float(pnl_series.min())

            out["gross_profit"] = float(pnl_series[pnl_series > 0.0].sum())
            out["gross_loss"] = float(pnl_series[pnl_series < 0.0].sum())
            profit_only = pnl_series[pnl_series > 0.0]
            loss_only = pnl_series[pnl_series < 0.0]
            out["avg_profit_trade"] = float(profit_only.mean()) if not profit_only.empty else 0.0
            out["avg_loss_trade"] = float(loss_only.mean()) if not loss_only.empty else 0.0
            if out["avg_loss_trade"] < 0.0:
                out["payoff_ratio"] = float(out["avg_profit_trade"] / abs(out["avg_loss_trade"]))
            out["expected_payoff"] = _safe_div(out["total_net_profit"], len(closed), 0.0)
            if out["gross_loss"] < 0.0:
                out["profit_factor"] = float(out["gross_profit"] / abs(out["gross_loss"]))
            elif out["gross_profit"] > 0.0:
                out["profit_factor"] = float("inf")

            trade_returns = (
                pd.to_numeric(closed.get("realized_return_pct"), errors="coerce") / 100.0
            )
            trade_returns = trade_returns[np.isfinite(trade_returns)]
            trade_returns = trade_returns[trade_returns > -1.0]
            if not trade_returns.empty:
                growth = 1.0 + trade_returns.to_numpy(dtype=np.float64)
                out["ahpr"] = float(np.mean(growth))
                if np.all(growth > 0.0):
                    out["ghpr"] = float(np.prod(growth) ** (1.0 / float(growth.size)))

            decisive = closed[
                pd.to_numeric(closed["realized_pnl"], errors="coerce").fillna(0.0) != 0.0
            ]
            if not decisive.empty:
                outcomes = list((decisive["realized_pnl"] > 0.0).to_numpy(dtype=bool))
                out["z_score"] = float(_runs_test_zscore(outcomes))
                streaks = _streak_groups(outcomes)
                win_lengths = [length for flag, length in streaks if flag]
                loss_lengths = [length for flag, length in streaks if not flag]
                out["win_streak_max"] = int(max(win_lengths) if win_lengths else 0)
                out["loss_streak_max"] = int(max(loss_lengths) if loss_lengths else 0)
                out["win_streak_avg"] = float(np.mean(win_lengths)) if win_lengths else 0.0
                out["loss_streak_avg"] = float(np.mean(loss_lengths)) if loss_lengths else 0.0

                current_sign = 0
                current_sum = 0.0
                current_count = 0
                best_profit_sum = float("-inf")
                best_profit_count = 0
                best_loss_sum = float("inf")
                best_loss_count = 0

                for value in (
                    pd.to_numeric(decisive["realized_pnl"], errors="coerce").fillna(0.0).to_list()
                ):
                    sign = 1 if value > 0.0 else -1
                    if current_sign == sign:
                        current_sum += float(value)
                        current_count += 1
                    else:
                        if current_sign == 1:
                            best_profit_sum = max(best_profit_sum, current_sum)
                            best_profit_count = max(best_profit_count, current_count)
                        elif current_sign == -1:
                            best_loss_sum = min(best_loss_sum, current_sum)
                            best_loss_count = max(best_loss_count, current_count)
                        current_sign = sign
                        current_sum = float(value)
                        current_count = 1

                if current_sign == 1:
                    best_profit_sum = max(best_profit_sum, current_sum)
                    best_profit_count = max(best_profit_count, current_count)
                elif current_sign == -1:
                    best_loss_sum = min(best_loss_sum, current_sum)
                    best_loss_count = max(best_loss_count, current_count)

                out["max_consecutive_profit_amount"] = (
                    float(best_profit_sum) if math.isfinite(best_profit_sum) else 0.0
                )
                out["max_consecutive_loss_amount"] = (
                    float(best_loss_sum) if math.isfinite(best_loss_sum) else 0.0
                )
                out["max_consecutive_profit_count"] = int(best_profit_count)
                out["max_consecutive_loss_count"] = int(best_loss_count)

            if "close_side" in closed.columns:
                long_closed = closed[closed["close_side"] == "LONG"]
                short_closed = closed[closed["close_side"] == "SHORT"]
                out["long_trades"] = len(long_closed)
                out["short_trades"] = len(short_closed)
                if len(long_closed) > 0:
                    long_wins = int((long_closed["realized_pnl"] > 0.0).sum())
                    out["long_win_rate"] = float(long_wins / float(len(long_closed)))
                if len(short_closed) > 0:
                    short_wins = int((short_closed["realized_pnl"] > 0.0).sum())
                    out["short_win_rate"] = float(short_wins / float(len(short_closed)))
                out["long_trades_win_pct"] = f"{out['long_trades']} ({out['long_win_rate']:.2%})"
                out["short_trades_win_pct"] = f"{out['short_trades']} ({out['short_win_rate']:.2%})"

            if "holding_sec" in closed.columns:
                hold = pd.to_numeric(closed["holding_sec"], errors="coerce")
                hold = hold[np.isfinite(hold)]
                hold = hold[hold >= 0.0]
                if not hold.empty:
                    out["holding_time_min_sec"] = float(hold.min())
                    out["holding_time_avg_sec"] = float(hold.mean())
                    out["holding_time_max_sec"] = float(hold.max())

            if out["initial_equity"] > 0.0:
                balance_curve = np.array(
                    [
                        out["initial_equity"],
                        *(out["initial_equity"] + pnl_series.cumsum()).tolist(),
                    ],
                    dtype=np.float64,
                )
                bal_dd = _drawdown_stats(balance_curve, out["initial_equity"])
                out["balance_drawdown_absolute"] = float(bal_dd["absolute"])
                out["balance_drawdown_maximal"] = float(bal_dd["maximal"])
                out["balance_drawdown_relative_pct"] = float(bal_dd["relative_pct"])

            dd_denom = out["balance_drawdown_maximal"]
            if dd_denom <= 0.0:
                dd_denom = out["equity_drawdown_maximal"]
            out["recovery_factor"] = _safe_div(out["total_net_profit"], dd_denom, 0.0)

    if (
        out["balance_drawdown_maximal"] <= 0.0
        and out["initial_equity"] > 0.0
        and totals_arr.size > 0
    ):
        out["balance_drawdown_absolute"] = out["equity_drawdown_absolute"]
        out["balance_drawdown_maximal"] = out["equity_drawdown_maximal"]
        out["balance_drawdown_relative_pct"] = out["equity_drawdown_relative_pct"]

    if out["period_days"] > 0:
        out["fills_per_day"] = out["fills"] / out["period_days"]

    out["open_pnl"] = float(out["total_c_plus_o"] - out["total_net_profit"])
    out["r_mdd"] = _safe_div(out["total_c_plus_o"], out["equity_drawdown_maximal"], 0.0)
    return out


def build_performance_metrics(df_equity):
    if df_equity.empty or "total" not in df_equity.columns:
        return {}

    totals = pd.to_numeric(df_equity["total"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if totals.size < 2:
        return {}

    periods = 252
    prev = totals[:-1]
    nxt = totals[1:]
    returns = np.divide(nxt - prev, np.where(prev == 0.0, 1.0, prev), dtype=float)
    if returns.size > 0 and not math.isfinite(returns[0]):
        returns[0] = 0.0

    benchmark_returns = np.array([], dtype=float)
    benchmark_series = None
    if "benchmark_price" in df_equity.columns:
        benchmark_values = pd.to_numeric(df_equity["benchmark_price"], errors="coerce").fillna(0.0)
        benchmark_series = benchmark_values.to_numpy(dtype=float)
        if benchmark_series.size >= 2:
            b_prev = benchmark_series[:-1]
            b_next = benchmark_series[1:]
            benchmark_returns = np.divide(
                b_next - b_prev,
                np.where(b_prev == 0.0, 1.0, b_prev),
                dtype=float,
            )

    total_return = _safe_float((totals[-1] - totals[0]) / totals[0] if totals[0] else 0.0)
    cumulative_returns = pd.Series(returns).fillna(0.0).add(1.0).cumprod().sub(1.0)
    cumulative_return = _safe_float(
        cumulative_returns.iloc[-1] if not cumulative_returns.empty else 0.0
    )
    cagr = _safe_float(create_cagr(totals[-1], totals[0], len(totals), periods))
    vol = _safe_float(create_annualized_volatility(returns, periods))
    sharpe = _safe_float(create_sharpe_ratio(returns, periods=periods))
    sortino = _safe_float(create_sortino_ratio(returns, periods=periods))
    drawdown, dd_duration = create_drawdowns(totals)
    max_dd = _safe_float(max(drawdown) if drawdown else 0.0)
    calmar = _safe_float(create_calmar_ratio(cagr, max_dd))

    alpha = 0.0
    beta = 0.0
    info_ratio = 0.0
    if benchmark_returns.size > 1 and returns.size > 1:
        min_len = min(len(returns), len(benchmark_returns))
        alpha, beta = create_alpha_beta(
            returns[:min_len], benchmark_returns[:min_len], periods=periods
        )
        alpha = _safe_float(alpha)
        beta = _safe_float(beta)
        info_ratio = _safe_float(
            create_information_ratio(returns[:min_len], benchmark_returns[:min_len])
        )

    funding_net = 0.0
    if "funding" in df_equity.columns:
        funding_net = _safe_float(
            pd.to_numeric(df_equity["funding"], errors="coerce").fillna(0.0).iloc[-1]
        )

    return {
        "Total Return": total_return,
        "Cumulative Return": cumulative_return,
        "CAGR": cagr,
        "Ann. Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Max Drawdown": max_dd,
        "DD Duration": int(dd_duration),
        "Alpha": alpha,
        "Beta": beta,
        "Information Ratio": info_ratio,
        "Funding (Net)": funding_net,
        "benchmark_series": benchmark_series,
        "return_series": returns,
        "cum_return_series": cumulative_returns,
    }


def _format_signed_dollar(value, digits=2):
    parsed = _safe_float(value, 0.0)
    prefix = "+" if parsed > 0 else ""
    return f"{prefix}${parsed:,.{digits}f}"


def _tone_class(value, invert=False):
    parsed = _safe_float(value, 0.0)
    if invert:
        parsed = -parsed
    if abs(parsed) < 1e-12:
        return ""
    return "up" if parsed > 0 else "down"


def _build_balance_equity_frame(df_equity, trade_analytics, initial_equity):
    empty = pd.DataFrame(
        columns=[
            "datetime",
            "equity",
            "balance",
            "open_pnl",
            "drawdown_abs",
            "drawdown_signed",
            "cum_realized_pnl",
            "cum_total_pnl",
        ]
    )
    if df_equity.empty or "datetime" not in df_equity.columns or "total" not in df_equity.columns:
        return empty

    frame = df_equity[["datetime", "total"]].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["equity"] = pd.to_numeric(frame["total"], errors="coerce")
    frame = (
        frame.dropna(subset=["datetime", "equity"]).sort_values("datetime").reset_index(drop=True)
    )
    if frame.empty:
        return empty

    baseline = _safe_float(initial_equity, 0.0)
    if baseline <= 0.0:
        baseline = _safe_float(frame["equity"].iloc[0], 0.0)

    frame["cum_realized_pnl"] = 0.0
    if (
        not trade_analytics.empty
        and "datetime" in trade_analytics.columns
        and "realized_pnl" in trade_analytics.columns
    ):
        closed = trade_analytics.copy()
        closed["datetime"] = pd.to_datetime(closed["datetime"], errors="coerce")
        closed = closed.dropna(subset=["datetime"])
        if "closed_qty" in closed.columns:
            closed_qty = pd.to_numeric(closed["closed_qty"], errors="coerce").fillna(0.0)
            closed = closed.loc[closed_qty > 0.0]
        if not closed.empty:
            closed["realized_pnl"] = pd.to_numeric(closed["realized_pnl"], errors="coerce").fillna(
                0.0
            )
            closed = (
                closed.groupby("datetime", as_index=False)["realized_pnl"]
                .sum()
                .sort_values("datetime")
            )
            closed["cum_realized_pnl"] = closed["realized_pnl"].cumsum()
            aligned = pd.merge_asof(
                frame[["datetime"]],
                closed[["datetime", "cum_realized_pnl"]],
                on="datetime",
                direction="backward",
            )
            frame["cum_realized_pnl"] = (
                pd.to_numeric(aligned["cum_realized_pnl"], errors="coerce").fillna(0.0).to_numpy()
            )

    frame["balance"] = baseline + frame["cum_realized_pnl"]
    frame["open_pnl"] = frame["equity"] - frame["balance"]
    frame["cum_total_pnl"] = frame["equity"] - baseline
    equity_peak = frame["equity"].cummax()
    frame["drawdown_abs"] = equity_peak - frame["equity"]
    frame["drawdown_signed"] = -frame["drawdown_abs"]
    return frame


def _build_mirror_snapshot(summary, balance_equity_df):
    total_trades = int(summary.get("closed_trades", summary.get("fills", 0)) or 0)
    wins = int(summary.get("wins", 0) or 0)
    losses = int(summary.get("losses", 0) or 0)
    closed_pnl = _safe_float(summary.get("total_net_profit", summary.get("realized_pnl", 0.0)), 0.0)
    total_c_plus_o = _safe_float(summary.get("total_c_plus_o"), 0.0)
    open_pnl = _safe_float(summary.get("open_pnl"), 0.0)
    if not balance_equity_df.empty:
        total_c_plus_o = _safe_float(balance_equity_df["cum_total_pnl"].iloc[-1], total_c_plus_o)
        open_pnl = _safe_float(balance_equity_df["open_pnl"].iloc[-1], open_pnl)

    equity_mdd = _safe_float(summary.get("equity_drawdown_maximal"), 0.0)
    equity_mdd_rel = _safe_float(summary.get("equity_drawdown_relative_pct"), 0.0)
    r_mdd = _safe_div(total_c_plus_o, equity_mdd, 0.0)
    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": _safe_float(summary.get("win_rate"), 0.0),
        "closed_pnl": closed_pnl,
        "open_pnl": open_pnl,
        "total_c_plus_o": total_c_plus_o,
        "equity_mdd": equity_mdd,
        "equity_mdd_rel": equity_mdd_rel,
        "r_mdd": r_mdd,
    }


def _render_mirror_cards(snapshot):
    cards = [
        {
            "label": "TOTAL TRADES",
            "value": f"{int(snapshot.get('total_trades', 0)):,}",
            "sub": f"{int(snapshot.get('wins', 0))}W / {int(snapshot.get('losses', 0))}L",
            "tone": "",
        },
        {
            "label": "WIN RATE",
            "value": f"{_safe_float(snapshot.get('win_rate'), 0.0):.1%}",
            "sub": "closed trades",
            "tone": _tone_class(snapshot.get("win_rate")),
        },
        {
            "label": "CLOSED PNL",
            "value": _format_signed_dollar(snapshot.get("closed_pnl"), digits=2),
            "sub": "realized",
            "tone": _tone_class(snapshot.get("closed_pnl")),
        },
        {
            "label": "OPEN P/L",
            "value": _format_signed_dollar(snapshot.get("open_pnl"), digits=2),
            "sub": "unrealized",
            "tone": _tone_class(snapshot.get("open_pnl")),
        },
        {
            "label": "TOTAL (C+O)",
            "value": _format_signed_dollar(snapshot.get("total_c_plus_o"), digits=2),
            "sub": "closed + open",
            "tone": _tone_class(snapshot.get("total_c_plus_o")),
        },
        {
            "label": "EQUITY MDD",
            "value": _format_signed_dollar(-_safe_float(snapshot.get("equity_mdd"), 0.0), digits=2),
            "sub": f"{_safe_float(snapshot.get('equity_mdd_rel'), 0.0):.2%}",
            "tone": _tone_class(snapshot.get("equity_mdd"), invert=True),
        },
        {
            "label": "R/MDD",
            "value": f"{_safe_float(snapshot.get('r_mdd'), 0.0):.2f}x",
            "sub": "total pnl / eq mdd",
            "tone": _tone_class(snapshot.get("r_mdd")),
        },
    ]
    html_blocks = []
    for card in cards:
        tone = str(card.get("tone") or "")
        html_blocks.append(
            "".join(
                [
                    f"<div class='lq-mirror-card {tone}'>",
                    f"<div class='label'>{html.escape(str(card['label']))}</div>",
                    f"<div class='value'>{html.escape(str(card['value']))}</div>",
                    f"<div class='sub'>{html.escape(str(card['sub']))}</div>",
                    "</div>",
                ]
            )
        )
    st.markdown(MIRROR_DASHBOARD_CSS, unsafe_allow_html=True)
    st.markdown(f"<div class='lq-mirror-grid'>{''.join(html_blocks)}</div>", unsafe_allow_html=True)


def _apply_mirror_figure_style(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#060b17",
        plot_bgcolor="#09162e",
        font=dict(color="#adc5eb"),
        margin=dict(l=56, r=24, t=54, b=52),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.13, xanchor="left", x=0.0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(45,80,136,0.32)", zeroline=False)


def _build_mirror_equity_curve_figure(balance_equity_df):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=balance_equity_df["datetime"],
            y=balance_equity_df["cum_total_pnl"],
            mode="lines",
            name="Cumulative PnL",
            line=dict(color="#1fd27c", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(31,210,124,0.20)",
        )
    )
    _apply_mirror_figure_style(fig)
    fig.update_layout(title="EQUITY CURVE (CUMULATIVE PNL)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(45,80,136,0.32)", tickprefix="$")
    fig.add_hline(y=0.0, line_dash="dot", line_color="rgba(157,180,221,0.5)")
    return fig


def _build_mirror_balance_equity_figure(balance_equity_df, snapshot):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=balance_equity_df["datetime"],
            y=balance_equity_df["drawdown_signed"],
            mode="lines",
            name="Drawdown",
            line=dict(color="#ff5f5f", width=1.6, dash="dot"),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=balance_equity_df["datetime"],
            y=balance_equity_df["equity"],
            mode="lines",
            name="Equity",
            line=dict(color="#22d37f", width=2.2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=balance_equity_df["datetime"],
            y=balance_equity_df["balance"],
            mode="lines",
            name="Balance",
            line=dict(color="#3d84ff", width=2.0, shape="hv"),
        ),
        secondary_y=False,
    )
    _apply_mirror_figure_style(fig)
    fig.update_layout(title="BALANCE & EQUITY (TIME-BASED)")
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(45,80,136,0.32)",
        tickprefix="$",
        secondary_y=False,
    )
    fig.update_yaxes(
        showgrid=False,
        tickprefix="$",
        secondary_y=True,
    )
    fig.add_hline(y=0.0, line_dash="dot", line_color="rgba(157,180,221,0.5)", secondary_y=True)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.99,
        y=1.13,
        text=(f"Eq MDD: ${_safe_float(snapshot.get('equity_mdd'), 0.0):,.2f}"),
        showarrow=False,
        font=dict(size=14, color="#ff5f5f"),
        xanchor="right",
    )
    return fig


def _serialize_balance_equity_frame(df_balance_equity, limit=1500):
    if df_balance_equity.empty:
        return []
    view = df_balance_equity[
        ["datetime", "equity", "balance", "open_pnl", "drawdown_signed"]
    ].copy()
    if len(view) > int(limit):
        view = _downsample_frame(view, int(limit))
    payload = []
    for row in view.itertuples(index=False):
        dt = pd.to_datetime(row.datetime, errors="coerce")
        payload.append(
            {
                "datetime": dt.isoformat() if pd.notna(dt) else None,
                "equity": _safe_float(row.equity, 0.0),
                "balance": _safe_float(row.balance, 0.0),
                "open_pnl": _safe_float(row.open_pnl, 0.0),
                "drawdown": _safe_float(row.drawdown_signed, 0.0),
            }
        )
    return payload


def build_report_payload(
    summary,
    performance,
    run_id,
    source,
    strategy_name,
    period_preset,
    df_equity,
    df_trades,
    df_risk,
    df_hb,
    runtime_overrides,
    strategy_params,
    mirror_snapshot=None,
    balance_equity_series=None,
):
    perf_export = {
        k: v
        for k, v in performance.items()
        if k not in {"benchmark_series", "return_series", "cum_return_series"}
    }
    mt5_rows = _build_mt5_summary_rows(summary)
    monthly_table = _build_monthly_returns_table(df_equity, performance)
    monthly_payload = {}
    if not monthly_table.empty:
        monthly_payload = {
            str(idx): {
                str(col): (
                    None
                    if pd.isna(monthly_table.loc[idx, col])
                    else float(monthly_table.loc[idx, col])
                )
                for col in monthly_table.columns
            }
            for idx in monthly_table.index
        }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "source": source,
        "strategy": strategy_name,
        "strategy_params": strategy_params,
        "period_preset": period_preset,
        "runtime_overrides": runtime_overrides,
        "summary": summary,
        "performance": perf_export,
        "equity_rows": len(df_equity),
        "trade_rows": len(df_trades),
        "risk_rows": len(df_risk),
        "heartbeat_rows": len(df_hb),
        "mt5_summary": mt5_rows.to_dict(orient="records"),
        "monthly_returns": monthly_payload,
        "mirror_snapshot": mirror_snapshot or {},
        "balance_equity_series": balance_equity_series or [],
    }


def save_report_snapshot(payload):
    out_dir = os.path.join("reports", "dashboard")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"dashboard_report_{ts}.json")
    md_path = os.path.join(out_dir, f"dashboard_report_{ts}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    s = payload["summary"]
    mirror = payload.get("mirror_snapshot") or {}
    balance_series = payload.get("balance_equity_series") or []
    mirror_block = ""
    if mirror:
        mirror_block = (
            f"\n## Mirror KPI Strip\n"
            f"- Total Trades: {int(mirror.get('total_trades', 0)):,} "
            f"({int(mirror.get('wins', 0))}W / {int(mirror.get('losses', 0))}L)\n"
            f"- Win Rate: {_safe_float(mirror.get('win_rate'), 0.0):.2%}\n"
            f"- Closed PnL: {_format_signed_dollar(mirror.get('closed_pnl'), digits=2)}\n"
            f"- Open P/L: {_format_signed_dollar(mirror.get('open_pnl'), digits=2)}\n"
            f"- Total (C+O): {_format_signed_dollar(mirror.get('total_c_plus_o'), digits=2)}\n"
            f"- Equity MDD: ${_safe_float(mirror.get('equity_mdd'), 0.0):,.2f} "
            f"({_safe_float(mirror.get('equity_mdd_rel'), 0.0):.2%})\n"
            f"- R/MDD: {_safe_float(mirror.get('r_mdd'), 0.0):.2f}x\n"
        )

    markdown = (
        f"# Dashboard Snapshot Report\n\n"
        f"- Generated: {payload['generated_at']}\n"
        f"- Run ID: {payload['run_id']}\n"
        f"- Source: {payload['source']}\n"
        f"- Strategy: {payload['strategy']}\n\n"
        f"## Summary\n"
        f"- Period: {s['period_start']} -> {s['period_end']}\n"
        f"- Bars: {s['bars']}\n"
        f"- Fills: {s['fills']} (BUY {s['buy_fills']} / SELL {s['sell_fills']})\n"
        f"- Avg fills/day: {s['fills_per_day']:.2f}\n"
        f"- Avg qty: {s['avg_qty']:.4f}\n"
        f"- Avg notional: {s['avg_notional']:.2f}\n"
        f"- Commission: {s['total_commission']:.4f}\n"
        f"- Realized PnL: {s['realized_pnl']:.4f}\n"
        f"- Win rate: {s['win_rate']:.2%}\n"
        f"- Avg trade return: {s['avg_trade_return_pct']:.4f}%\n"
        f"- Best trade PnL: {s['best_trade_pnl']:.4f}\n"
        f"- Worst trade PnL: {s['worst_trade_pnl']:.4f}\n"
        f"- Gross Profit / Gross Loss: {s['gross_profit']:.4f} / {s['gross_loss']:.4f}\n"
        f"- Profit Factor: {_format_metric_value('Profit Factor', s['profit_factor'])}\n"
        f"- Recovery Factor: {s['recovery_factor']:.4f}\n"
        f"- Long Trades (Win %): {s['long_trades_win_pct']}\n"
        f"- Short Trades (Win %): {s['short_trades_win_pct']}\n"
        f"- Holding (min/avg/max): {_format_duration_seconds(s['holding_time_min_sec'])} / "
        f"{_format_duration_seconds(s['holding_time_avg_sec'])} / "
        f"{_format_duration_seconds(s['holding_time_max_sec'])}\n"
        f"\n## Drawdown\n"
        f"- Equity DD (Abs/Max/Rel): {s['equity_drawdown_absolute']:.4f} / "
        f"{s['equity_drawdown_maximal']:.4f} / {s['equity_drawdown_relative_pct']:.2%}\n"
        f"- Balance DD (Abs/Max/Rel): {s['balance_drawdown_absolute']:.4f} / "
        f"{s['balance_drawdown_maximal']:.4f} / {s['balance_drawdown_relative_pct']:.2%}\n"
        f"\n## Streaks\n"
        f"- Max win/loss streak: {int(s['win_streak_max'])} / {int(s['loss_streak_max'])}\n"
        f"- Avg win/loss streak: {s['win_streak_avg']:.2f} / {s['loss_streak_avg']:.2f}\n"
        f"- Max consecutive profit/loss: {s['max_consecutive_profit_amount']:.4f} / "
        f"{s['max_consecutive_loss_amount']:.4f}\n"
        f"\n## Export Payload\n"
        f"- Balance/Equity points: {len(balance_series)}\n"
        f"{mirror_block}"
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    return json_path, md_path, markdown


st.sidebar.header("Configuration")
data_source = st.sidebar.selectbox("Data Source", ["Auto", "SQLite", "CSV"])
db_path = st.sidebar.text_input("SQLite Path", value=DEFAULT_DB_PATH)
market_exchange = st.sidebar.text_input(
    "Market Exchange", value=getattr(BaseConfig, "MARKET_DATA_EXCHANGE", "binance")
)
market_timeframe = st.sidebar.text_input(
    "Market Timeframe", value=getattr(BaseConfig, "TIMEFRAME", "1m")
)
strategy_options = get_strategy_names()
default_strategy_name = "RsiStrategy"
default_strategy_index = (
    strategy_options.index(default_strategy_name)
    if default_strategy_name in strategy_options
    else 0
)
strategy_name = st.sidebar.selectbox(
    "Select Strategy",
    strategy_options,
    index=default_strategy_index,
)

auto_refresh_enabled = st.sidebar.toggle("Auto Refresh", value=True)
refresh_interval_sec = st.sidebar.slider(
    "Refresh Interval (sec)", min_value=1, max_value=120, value=5
)
max_points = st.sidebar.slider("Max Data Points", min_value=500, max_value=100000, value=2500)
auto_downsample = st.sidebar.toggle("Auto Downsample Plots", value=True)
downsample_target_points = st.sidebar.slider(
    "Downsample Target Points",
    min_value=1000,
    max_value=50000,
    value=DEFAULT_DOWNSAMPLE_TARGET_POINTS,
    step=500,
    disabled=not auto_downsample,
)
pin_to_running = st.sidebar.toggle("Pin to RUNNING live run", value=True)
run_stale_sec = st.sidebar.slider(
    "RUNNING Stale Threshold (sec)",
    min_value=30,
    max_value=3600,
    value=DEFAULT_RUN_STALE_SEC,
    step=30,
)
period_preset = st.sidebar.selectbox(
    "Chart Period",
    ["All", "1D", "7D", "30D", "90D", "180D", "365D", "Custom"],
    index=2,
)
custom_start = None
custom_end = None
if period_preset == "Custom":
    default_custom_end = pd.Timestamp.utcnow().date()
    default_custom_start = (pd.Timestamp.utcnow() - pd.Timedelta(days=30)).date()
    custom_start = st.sidebar.date_input("Custom Start", value=default_custom_start)
    custom_end = st.sidebar.date_input("Custom End", value=default_custom_end)

refresh_counter, refresh_mode = _setup_auto_refresh(auto_refresh_enabled, refresh_interval_sec)
params = load_params(strategy_name)
st.sidebar.caption(
    f"Refresh mode: {refresh_mode} | tick: {refresh_counter} | interval: {_safe_interval_sec(refresh_interval_sec)}s"
)

st.sidebar.divider()
st.sidebar.subheader("Execution Lab")
runner_initial_capital = st.sidebar.number_input(
    "Initial Equity Override",
    min_value=100.0,
    max_value=100000000.0,
    value=float(BaseConfig.INITIAL_CAPITAL),
    step=100.0,
)
runner_leverage = st.sidebar.number_input(
    "Backtest Leverage Override",
    min_value=1,
    max_value=20,
    value=int(BacktestConfig.LEVERAGE),
    step=1,
)
runner_symbols_raw = st.sidebar.text_input(
    "Symbols (comma-separated)", value=", ".join(list(BaseConfig.SYMBOLS))
)
runner_symbols = _parse_symbols_csv(runner_symbols_raw)
if not runner_symbols:
    runner_symbols = list(BaseConfig.SYMBOLS)

runner_timeframe = st.sidebar.text_input("Timeframe Override", value=str(BaseConfig.TIMEFRAME))
runner_timeout_sec = st.sidebar.slider(
    "Runner Timeout (sec)", min_value=30, max_value=3600, value=900, step=30
)
runner_data_source = st.sidebar.selectbox("Runner Data Source", ["auto", "csv", "db"], index=0)

strategy_params = _merged_strategy_params(strategy_name, params)
with st.sidebar.expander("Strategy Parameter Overrides", expanded=False):
    if strategy_name == "RsiStrategy":
        strategy_params["rsi_period"] = int(
            st.number_input(
                "rsi_period",
                min_value=2,
                max_value=200,
                value=int(strategy_params.get("rsi_period", 14)),
                step=1,
            )
        )
        strategy_params["oversold"] = float(
            st.number_input(
                "oversold",
                min_value=1.0,
                max_value=99.0,
                value=float(strategy_params.get("oversold", 30.0)),
                step=0.5,
            )
        )
        strategy_params["overbought"] = float(
            st.number_input(
                "overbought",
                min_value=1.0,
                max_value=99.0,
                value=float(strategy_params.get("overbought", 70.0)),
                step=0.5,
            )
        )
    elif strategy_name == "MovingAverageCrossStrategy":
        strategy_params["short_window"] = int(
            st.number_input(
                "short_window",
                min_value=2,
                max_value=500,
                value=int(strategy_params.get("short_window", 10)),
                step=1,
            )
        )
        strategy_params["long_window"] = int(
            st.number_input(
                "long_window",
                min_value=3,
                max_value=1000,
                value=int(strategy_params.get("long_window", 30)),
                step=1,
            )
        )
    elif strategy_name == "PairTradingZScoreStrategy":
        strategy_params["lookback_window"] = int(
            st.number_input(
                "lookback_window",
                min_value=10,
                max_value=2000,
                value=int(strategy_params.get("lookback_window", 96)),
                step=1,
            )
        )
        strategy_params["hedge_window"] = int(
            st.number_input(
                "hedge_window",
                min_value=10,
                max_value=4000,
                value=int(strategy_params.get("hedge_window", 192)),
                step=1,
            )
        )
        strategy_params["entry_z"] = float(
            st.number_input(
                "entry_z",
                min_value=0.1,
                max_value=10.0,
                value=float(strategy_params.get("entry_z", 2.0)),
                step=0.05,
            )
        )
        strategy_params["exit_z"] = float(
            st.number_input(
                "exit_z",
                min_value=0.0,
                max_value=5.0,
                value=float(strategy_params.get("exit_z", 0.35)),
                step=0.05,
            )
        )
        strategy_params["stop_z"] = float(
            st.number_input(
                "stop_z",
                min_value=0.2,
                max_value=12.0,
                value=float(strategy_params.get("stop_z", 3.5)),
                step=0.1,
            )
        )
        strategy_params["min_correlation"] = float(
            st.number_input(
                "min_correlation",
                min_value=-1.0,
                max_value=1.0,
                value=float(strategy_params.get("min_correlation", 0.15)),
                step=0.05,
            )
        )
        strategy_params["max_hold_bars"] = int(
            st.number_input(
                "max_hold_bars",
                min_value=1,
                max_value=10000,
                value=int(strategy_params.get("max_hold_bars", 240)),
                step=1,
            )
        )
        strategy_params["cooldown_bars"] = int(
            st.number_input(
                "cooldown_bars",
                min_value=0,
                max_value=5000,
                value=int(strategy_params.get("cooldown_bars", 6)),
                step=1,
            )
        )
        strategy_params["reentry_z_buffer"] = float(
            st.number_input(
                "reentry_z_buffer",
                min_value=0.0,
                max_value=5.0,
                value=float(strategy_params.get("reentry_z_buffer", 0.20)),
                step=0.01,
            )
        )
        strategy_params["min_z_turn"] = float(
            st.number_input(
                "min_z_turn",
                min_value=0.0,
                max_value=5.0,
                value=float(strategy_params.get("min_z_turn", 0.05)),
                step=0.01,
            )
        )
        strategy_params["stop_loss_pct"] = float(
            st.number_input(
                "stop_loss_pct",
                min_value=0.001,
                max_value=0.50,
                value=float(strategy_params.get("stop_loss_pct", 0.04)),
                step=0.001,
            )
        )
        strategy_params["min_abs_beta"] = float(
            st.number_input(
                "min_abs_beta",
                min_value=0.0,
                max_value=20.0,
                value=float(strategy_params.get("min_abs_beta", 0.02)),
                step=0.01,
            )
        )
        strategy_params["max_abs_beta"] = float(
            st.number_input(
                "max_abs_beta",
                min_value=0.1,
                max_value=30.0,
                value=float(strategy_params.get("max_abs_beta", 6.0)),
                step=0.1,
            )
        )
        strategy_params["min_volume_window"] = int(
            st.number_input(
                "min_volume_window",
                min_value=1,
                max_value=5000,
                value=int(strategy_params.get("min_volume_window", 24)),
                step=1,
            )
        )
        strategy_params["min_volume_ratio"] = float(
            st.number_input(
                "min_volume_ratio",
                min_value=0.0,
                max_value=5.0,
                value=float(strategy_params.get("min_volume_ratio", 0.0)),
                step=0.01,
            )
        )
        strategy_params["use_log_price"] = bool(
            st.checkbox(
                "use_log_price",
                value=bool(strategy_params.get("use_log_price", True)),
            )
        )

        default_symbol_x = str(
            strategy_params.get("symbol_x") or (runner_symbols[0] if runner_symbols else "")
        )
        default_symbol_y = str(
            strategy_params.get("symbol_y")
            or (runner_symbols[1] if len(runner_symbols) > 1 else default_symbol_x)
        )
        strategy_params["symbol_x"] = st.text_input("symbol_x", value=default_symbol_x)
        strategy_params["symbol_y"] = st.text_input("symbol_y", value=default_symbol_y)

with st.sidebar.expander("Optimization Search Space", expanded=False):
    default_optuna_cfg = _strategy_default_optuna(strategy_name)
    default_grid_cfg = _strategy_default_grid(strategy_name)
    if strategy_name == str(OptimizationConfig.STRATEGY_NAME):
        if isinstance(OptimizationConfig.OPTUNA_CONFIG, dict) and OptimizationConfig.OPTUNA_CONFIG:
            default_optuna_cfg = dict(OptimizationConfig.OPTUNA_CONFIG)
        if isinstance(OptimizationConfig.GRID_CONFIG, dict) and OptimizationConfig.GRID_CONFIG:
            default_grid_cfg = dict(OptimizationConfig.GRID_CONFIG)

    default_optuna_json = json.dumps(default_optuna_cfg, indent=2, ensure_ascii=False)
    default_grid_json = json.dumps(default_grid_cfg, indent=2, ensure_ascii=False)
    optuna_json_raw = st.text_area("OPTUNA config JSON", value=default_optuna_json, height=140)
    grid_json_raw = st.text_area("GRID config JSON", value=default_grid_json, height=140)
    optimize_folds = st.number_input(
        "Walk-forward folds",
        min_value=1,
        max_value=20,
        value=int(OptimizationConfig.WALK_FORWARD_FOLDS),
        step=1,
    )
    optimize_trials = st.number_input(
        "Optuna trials",
        min_value=1,
        max_value=5000,
        value=int(
            default_optuna_cfg.get("n_trials", OptimizationConfig.OPTUNA_CONFIG.get("n_trials", 20))
        ),
        step=1,
    )
    optimize_workers = st.number_input(
        "Optimization workers",
        min_value=1,
        max_value=max(1, (os.cpu_count() or 4) * 2),
        value=int(OptimizationConfig.MAX_WORKERS),
        step=1,
    )
    persist_best_params = st.checkbox(
        "Persist best params after optimize", value=bool(OptimizationConfig.PERSIST_BEST_PARAMS)
    )

optuna_config_for_runner = dict(OptimizationConfig.OPTUNA_CONFIG)
grid_config_for_runner = dict(OptimizationConfig.GRID_CONFIG)
opt_space_error = None
try:
    parsed_optuna = json.loads(optuna_json_raw)
    parsed_grid = json.loads(grid_json_raw)
    if isinstance(parsed_optuna, dict):
        optuna_config_for_runner = parsed_optuna
    else:
        opt_space_error = "OPTUNA config JSON must be an object."
    if isinstance(parsed_grid, dict):
        grid_config_for_runner = parsed_grid
    else:
        opt_space_error = "GRID config JSON must be an object."
except Exception as exc:
    opt_space_error = f"Optimization JSON parse error: {exc}"

runner_env_overrides = _build_runtime_env_overrides(
    initial_capital=runner_initial_capital,
    leverage=runner_leverage,
    timeframe=runner_timeframe,
    symbols=runner_symbols,
    strategy_name=strategy_name,
    optuna_config=optuna_config_for_runner,
    grid_config=grid_config_for_runner,
)

if "runner_last_result" not in st.session_state:
    st.session_state["runner_last_result"] = None

df_equity = pd.DataFrame()
df_trades = pd.DataFrame()
df_orders = pd.DataFrame()
df_risk = pd.DataFrame()
df_hb = pd.DataFrame()
df_order_states = pd.DataFrame()
df_market = pd.DataFrame()
df_optimize = pd.DataFrame()
active_run_id = None
resolved_source = None
runs_df = pd.DataFrame()

_ensure_workflow_jobs_schema(db_path)
_refresh_workflow_jobs(db_path)

use_sqlite = data_source == "SQLite" or (data_source == "Auto" and os.path.exists(db_path))
if use_sqlite and os.path.exists(db_path):
    runs_df = load_runs(db_path, refresh_counter=refresh_counter)
    runs_df = _annotate_run_health(runs_df, stale_after_sec=run_stale_sec)
    df_optimize = load_optimization_results_sqlite(db_path, refresh_counter=refresh_counter)
    if not runs_df.empty:
        run_options = runs_df["run_id"].astype(str).tolist()
        default_run = _latest_running_run_id(runs_df) if pin_to_running else None
        if default_run and _equity_rows_for_run(runs_df, default_run) <= 0:
            default_run = None
        if not default_run:
            default_run = _latest_run_with_equity(runs_df)
        if not default_run:
            default_run = run_options[0]

        if "dashboard_run_id" not in st.session_state:
            st.session_state["dashboard_run_id"] = default_run
        if pin_to_running:
            st.session_state["dashboard_run_id"] = default_run
        if st.session_state["dashboard_run_id"] not in run_options:
            st.session_state["dashboard_run_id"] = default_run

        selected_idx = run_options.index(st.session_state["dashboard_run_id"])
        active_run_id = st.sidebar.selectbox("Run ID", run_options, index=selected_idx)
        st.session_state["dashboard_run_id"] = active_run_id

        df_equity = load_equity_sqlite(
            db_path, active_run_id, refresh_counter=refresh_counter, max_points=max_points
        )
        df_trades = load_fills_sqlite(
            db_path, active_run_id, refresh_counter=refresh_counter, max_points=max_points
        )
        df_orders = load_orders_sqlite(
            db_path, active_run_id, refresh_counter=refresh_counter, max_points=max_points
        )
        df_risk = load_risk_events_sqlite(db_path, active_run_id, refresh_counter=refresh_counter)
        df_hb = load_heartbeats_sqlite(db_path, active_run_id, refresh_counter=refresh_counter)
        df_order_states = load_order_states_sqlite(
            db_path, active_run_id, refresh_counter=refresh_counter
        )
        resolved_source = "SQLite"

if resolved_source is None or (df_equity.empty and data_source == "Auto"):
    fallback_equity = load_equity_csv(refresh_counter=refresh_counter, max_points=max_points)
    fallback_trades = load_trades_csv(refresh_counter=refresh_counter, max_points=max_points)
    if not fallback_equity.empty:
        df_equity = fallback_equity
        df_trades = fallback_trades
        active_run_id = None
        resolved_source = "CSV"

if resolved_source is None:
    resolved_source = "None"

period_start, period_end = _resolve_period_window(
    period_preset,
    custom_start,
    custom_end,
    df_equity,
    dt_col="datetime",
)

df_equity = _apply_period_filter(df_equity, "datetime", period_start, period_end)
df_trades = _apply_period_filter(df_trades, "datetime", period_start, period_end)
df_orders = _apply_period_filter(df_orders, "created_at", period_start, period_end)
df_risk = _apply_period_filter(df_risk, "event_time", period_start, period_end)
df_hb = _apply_period_filter(df_hb, "heartbeat_time", period_start, period_end)
df_order_states = _apply_period_filter(df_order_states, "event_time", period_start, period_end)

if auto_downsample and not df_equity.empty:
    plot_equity = _downsample_frame(df_equity, downsample_target_points)
else:
    plot_equity = df_equity

if auto_downsample and not df_trades.empty:
    plot_trades = _downsample_frame(df_trades, downsample_target_points)
else:
    plot_trades = df_trades

if not runs_df.empty and active_run_id:
    symbols = _extract_run_symbols(runs_df, active_run_id)
else:
    symbols = list(BaseConfig.SYMBOLS)
market_symbol = st.sidebar.selectbox(
    "Market Symbol", symbols if symbols else list(BaseConfig.SYMBOLS)
)

if os.path.exists(db_path):
    df_market = load_market_ohlcv_sqlite(
        db_path,
        market_symbol,
        market_timeframe,
        market_exchange,
        refresh_counter=refresh_counter,
        max_points=max_points,
    )
df_market = _apply_period_filter(df_market, "datetime", period_start, period_end)
if auto_downsample and not df_market.empty:
    plot_market = _downsample_frame(df_market, downsample_target_points)
else:
    plot_market = df_market

trade_analytics = compute_trade_analytics(df_trades)
summary = build_summary(df_equity, trade_analytics)
performance = build_performance_metrics(df_equity)
latest_data_ts, latency_sec = _compute_data_latency_seconds(df_equity)
mirror_balance_equity = _build_balance_equity_frame(
    df_equity,
    trade_analytics,
    initial_equity=summary.get("initial_equity", 0.0),
)
mirror_snapshot = _build_mirror_snapshot(summary, mirror_balance_equity)

st.subheader("PC Mirror Snapshot")
st.caption("Reference-style KPI strip + equity/balance/drawdown narrative")
_render_mirror_cards(mirror_snapshot)
mirror_left, mirror_right = st.columns(2)
if not mirror_balance_equity.empty:
    with mirror_left:
        st.plotly_chart(
            _build_mirror_equity_curve_figure(mirror_balance_equity),
            use_container_width=True,
            key="mirror_equity_curve",
        )
    with mirror_right:
        st.plotly_chart(
            _build_mirror_balance_equity_figure(mirror_balance_equity, mirror_snapshot),
            use_container_width=True,
            key="mirror_balance_equity",
        )
else:
    with mirror_left:
        st.info("No equity data for mirror-style curve yet.")
    with mirror_right:
        st.info("No balance/equity timeline for mirror-style chart yet.")

st.header("Overview")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Source", resolved_source)
col2.metric("Bars", f"{summary['bars']}")
col3.metric("Fills", f"{summary['fills']}")
col4.metric("Avg Fills/Day", f"{summary['fills_per_day']:.2f}")
col5.metric("Closed PnL", _format_signed_dollar(summary.get("total_net_profit"), digits=2))
col6.metric("Win Rate", f"{summary['win_rate']:.2%}")

eq1, eq2, eq3, eq4 = st.columns(4)
eq1.metric("Initial Equity", f"{summary['initial_equity']:.2f}")
eq2.metric("Final Equity", f"{summary['final_equity']:.2f}")
eq3.metric("Configured Initial Equity", f"{runner_initial_capital:.2f}")
eq4.metric("Configured Leverage", f"{int(runner_leverage)}x")

mx1, mx2, mx3, mx4 = st.columns(4)
mx1.metric("Open P/L", _format_signed_dollar(summary.get("open_pnl"), digits=2))
mx2.metric("Total (C+O)", _format_signed_dollar(summary.get("total_c_plus_o"), digits=2))
mx3.metric("R/MDD", f"{_safe_float(summary.get('r_mdd'), 0.0):.2f}x")
mx4.metric(
    "Equity MDD",
    (
        f"${_safe_float(summary.get('equity_drawdown_maximal'), 0.0):,.2f} "
        f"({_safe_float(summary.get('equity_drawdown_relative_pct'), 0.0):.2%})"
    ),
)

if performance:
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("Total Return", f"{performance.get('Total Return', 0.0):.2%}")
    p2.metric("Cumulative Return", f"{performance.get('Cumulative Return', 0.0):.2%}")
    p3.metric("Sharpe", f"{performance.get('Sharpe Ratio', 0.0):.3f}")
    p4.metric("Sortino", f"{performance.get('Sortino Ratio', 0.0):.3f}")
    p5.metric("Max Drawdown", f"{performance.get('Max Drawdown', 0.0):.2%}")
    p6.metric("Funding (Net)", f"{performance.get('Funding (Net)', 0.0):.4f}")

pf1, pf2, pf3, pf4, pf5, pf6 = st.columns(6)
pf1.metric("Profit Factor", _format_metric_value("Profit Factor", summary.get("profit_factor")))
pf2.metric("Recovery Factor", f"{_safe_float(summary.get('recovery_factor')):.3f}")
pf3.metric("Expected Payoff", f"{_safe_float(summary.get('expected_payoff')):.4f}")
pf4.metric("Profit Trades", str(summary.get("profit_trades_text", "0 (0.00%)")))
pf5.metric("Loss Trades", str(summary.get("loss_trades_text", "0 (0.00%)")))
pf6.metric("Payoff Ratio", f"{_safe_float(summary.get('payoff_ratio')):.3f}")

dir1, dir2, dir3, dir4 = st.columns(4)
dir1.metric("Long Trades (Win %)", str(summary.get("long_trades_win_pct", "0 (0.00%)")))
dir2.metric("Short Trades (Win %)", str(summary.get("short_trades_win_pct", "0 (0.00%)")))
dir3.metric("Avg Holding", _format_duration_seconds(summary.get("holding_time_avg_sec", 0.0)))
dir4.metric(
    "Avg Profit / Loss",
    (
        f"{_safe_float(summary.get('avg_profit_trade'), 0.0):.2f} / "
        f"{_safe_float(summary.get('avg_loss_trade'), 0.0):.2f}"
    ),
)

if latest_data_ts is not None and latency_sec is not None:
    st.caption(
        f"Data timestamp: {latest_data_ts} | latency: {latency_sec:.1f}s | "
        f"period preset: {period_preset} | range: {summary['period_start']} -> {summary['period_end']}"
    )

if not runs_df.empty and "effective_status" in runs_df.columns:
    healthy_runs = int((runs_df["effective_status"] == "RUNNING_HEALTHY").sum())
    stale_runs = int((runs_df["effective_status"] == "RUNNING_STALE").sum())
    no_telemetry_runs = int((runs_df["effective_status"] == "RUNNING_NO_TELEMETRY").sum())
    st.caption(
        f"Live run health -> healthy: {healthy_runs}, stale: {stale_runs}, "
        f"no-telemetry: {no_telemetry_runs} (threshold={int(run_stale_sec)}s)"
    )

st.subheader("Strategy Parameters")
st.json(strategy_params)

tab_overview, tab_exec, tab_risk, tab_market, tab_opt, tab_report, tab_raw = st.tabs(
    [
        "Performance & Price",
        "Execution Analytics",
        "Risk & Health",
        "Market Data",
        "Optimization Insights",
        "Report Export",
        "Raw Data",
    ]
)

with tab_overview:
    with st.expander("Metric Definitions", expanded=False):
        metric_rows = []
        for metric_name, definition in METRIC_DEFINITIONS.items():
            metric_rows.append(
                {
                    "Metric": metric_name,
                    "Value": _format_metric_value(
                        metric_name,
                        _metric_value(metric_name, performance, summary),
                    ),
                    "Definition": definition,
                }
            )
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True)

    mt5_rows = _build_mt5_summary_rows(summary)
    if not mt5_rows.empty:
        st.subheader("MT5-Style Summary Grid")
        st.dataframe(mt5_rows, use_container_width=True, hide_index=True)

    st.caption(
        f"Data context: run={active_run_id if active_run_id else 'N/A'} | "
        f"source={resolved_source} | symbol={market_symbol} | timeframe={market_timeframe} | "
        f"exchange={market_exchange}"
    )

    if not plot_equity.empty:
        fig_equity = go.Figure()
        fig_equity.add_trace(
            go.Scatter(
                x=plot_equity["datetime"],
                y=plot_equity["total"],
                mode="lines",
                name="Strategy Equity",
                line=dict(color="#0db39e", width=2),
            )
        )
        fig_equity.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Equity",
            template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(fig_equity, use_container_width=True)

        if "benchmark_price" in plot_equity.columns:
            benchmark_series = pd.to_numeric(plot_equity["benchmark_price"], errors="coerce")
            if benchmark_series.notna().any():
                fig_benchmark = go.Figure()
                fig_benchmark.add_trace(
                    go.Scatter(
                        x=plot_equity["datetime"],
                        y=benchmark_series,
                        mode="lines",
                        line=dict(color="#805ad5", width=1.5),
                        name="Benchmark Price",
                    )
                )
                fig_benchmark.update_layout(
                    title="Benchmark Price (from Equity Metadata)",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    template="plotly_white",
                    hovermode="x unified",
                )
                st.plotly_chart(fig_benchmark, use_container_width=True)

        if "funding" in plot_equity.columns:
            funding_series = pd.to_numeric(plot_equity["funding"], errors="coerce")
            if funding_series.notna().any():
                fig_funding = go.Figure()
                fig_funding.add_trace(
                    go.Scatter(
                        x=plot_equity["datetime"],
                        y=funding_series,
                        mode="lines",
                        line=dict(color="#dd6b20", width=2),
                        name="Funding (Net)",
                    )
                )
                fig_funding.update_layout(
                    title="Funding (Net) Over Time",
                    xaxis_title="Date",
                    yaxis_title="Funding",
                    template="plotly_white",
                    hovermode="x unified",
                )
                st.plotly_chart(fig_funding, use_container_width=True)

        if (
            performance
            and len(performance.get("cum_return_series", [])) == len(df_equity.index) - 1
        ):
            cum_returns = pd.Series(performance["cum_return_series"])
            fig_cum_ret = go.Figure()
            fig_cum_ret.add_trace(
                go.Scatter(
                    x=df_equity["datetime"].iloc[1:],
                    y=cum_returns,
                    mode="lines",
                    line=dict(color="#2b6cb0", width=2),
                    name="Cumulative Return",
                )
            )
            fig_cum_ret.update_layout(
                title="Cumulative Return",
                xaxis_title="Date",
                yaxis_title="Return",
                template="plotly_white",
                hovermode="x unified",
                yaxis_tickformat=".2%",
            )
            st.plotly_chart(fig_cum_ret, use_container_width=True)

        roll_max = plot_equity["total"].cummax()
        drawdown = (plot_equity["total"] - roll_max) / roll_max
        fig_dd = go.Figure()
        fig_dd.add_trace(
            go.Scatter(
                x=plot_equity["datetime"],
                y=drawdown,
                fill="tozeroy",
                name="Drawdown",
                line=dict(color="#f05a66"),
            )
        )
        fig_dd.update_layout(title="Drawdown", yaxis_title="Drawdown", template="plotly_white")
        st.plotly_chart(fig_dd, use_container_width=True)

        monthly_table = _build_monthly_returns_table(df_equity, performance)
        if not monthly_table.empty:
            values = monthly_table.to_numpy(dtype=float)
            text_vals = [
                [f"{_safe_float(v):.2%}" if np.isfinite(v) else "" for v in row] for row in values
            ]
            fig_monthly = go.Figure(
                data=[
                    go.Heatmap(
                        z=values,
                        x=list(monthly_table.columns),
                        y=[str(idx) for idx in monthly_table.index],
                        colorscale="RdBu",
                        zmid=0.0,
                        text=text_vals,
                        texttemplate="%{text}",
                        hovertemplate="Year %{y} | %{x}: %{z:.2%}<extra></extra>",
                    )
                ]
            )
            fig_monthly.update_layout(
                title="Monthly Returns Heatmap",
                xaxis_title="Month",
                yaxis_title="Year",
                template="plotly_white",
            )
            st.plotly_chart(fig_monthly, use_container_width=True)

    if not plot_market.empty:
        fig_price = go.Figure()
        fig_price.add_trace(
            go.Candlestick(
                x=plot_market["datetime"],
                open=plot_market["open"],
                high=plot_market["high"],
                low=plot_market["low"],
                close=plot_market["close"],
                name=f"{market_symbol} price",
            )
        )

        if not plot_trades.empty:
            symbol_trades = plot_trades[plot_trades["symbol"] == market_symbol]
            if not symbol_trades.empty:
                enriched = compute_trade_analytics(symbol_trades)
                for direction, color, symbol_shape in [
                    ("BUY", "#0db39e", "triangle-up"),
                    ("SELL", "#f05a66", "triangle-down"),
                ]:
                    part = enriched[enriched["direction"] == direction]
                    if part.empty:
                        continue
                    custom_cols = [
                        part["quantity"],
                        part["position_after"],
                        part["realized_pnl"],
                        part["realized_return_pct"],
                    ]
                    custom_data = pd.concat(custom_cols, axis=1).to_numpy()
                    fig_price.add_trace(
                        go.Scatter(
                            x=part["datetime"],
                            y=part["price"],
                            mode="markers",
                            name=direction,
                            marker=dict(symbol=symbol_shape, size=10, color=color),
                            customdata=custom_data,
                            hovertemplate=(
                                "%{x}<br>"
                                f"{direction} @ %{{y:.4f}}<br>"
                                "Qty: %{customdata[0]:.4f}<br>"
                                "Position after: %{customdata[1]:.4f}<br>"
                                "Realized PnL: %{customdata[2]:.4f}<br>"
                                "Trade Return: %{customdata[3]:.4f}%<extra></extra>"
                            ),
                        )
                    )

        fig_price.update_layout(
            title=f"{market_symbol} Price + Buy/Sell Markers",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
            xaxis_rangeslider_visible=True,
        )
        st.plotly_chart(fig_price, use_container_width=True)

with tab_exec:
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("BUY fills", f"{summary['buy_fills']}")
    col_b.metric("SELL fills", f"{summary['sell_fills']}")
    col_c.metric("Avg Qty", f"{summary['avg_qty']:.4f}")
    col_d.metric("Avg Notional", f"{summary['avg_notional']:.2f}")

    col_e, col_f, col_g, col_h = st.columns(4)
    col_e.metric("Total Commission", f"{summary['total_commission']:.4f}")
    col_f.metric("Avg Trade Return", f"{summary['avg_trade_return_pct']:.4f}%")
    col_g.metric("Best Trade PnL", f"{summary['best_trade_pnl']:.4f}")
    col_h.metric("Worst Trade PnL", f"{summary['worst_trade_pnl']:.4f}")

    col_i, col_j, col_k, col_l = st.columns(4)
    col_i.metric("Max Win Streak", f"{int(summary['win_streak_max'])}")
    col_j.metric("Max Loss Streak", f"{int(summary['loss_streak_max'])}")
    col_k.metric(
        "Avg Win/Loss Streak", f"{summary['win_streak_avg']:.2f} / {summary['loss_streak_avg']:.2f}"
    )
    col_l.metric("Avg Holding Time", _format_duration_seconds(summary["holding_time_avg_sec"]))

    direction_table = pd.DataFrame(
        [
            {
                "Direction": "Long",
                "Closed Trades": int(summary.get("long_trades", 0)),
                "Win Rate": f"{_safe_float(summary.get('long_win_rate')):.2%}",
            },
            {
                "Direction": "Short",
                "Closed Trades": int(summary.get("short_trades", 0)),
                "Win Rate": f"{_safe_float(summary.get('short_win_rate')):.2%}",
            },
        ]
    )
    st.dataframe(direction_table, use_container_width=True, hide_index=True)

    if not trade_analytics.empty:
        closed = (
            trade_analytics[trade_analytics["closed_qty"] > 0]
            if "closed_qty" in trade_analytics.columns
            else trade_analytics[trade_analytics["realized_pnl"] != 0]
        )
        if not closed.empty:
            fig_trade_pnl = go.Figure()
            fig_trade_pnl.add_trace(
                go.Bar(
                    x=closed["datetime"],
                    y=closed["realized_pnl"],
                    name="Realized PnL per closing trade",
                )
            )
            fig_trade_pnl.update_layout(
                title="Trade-by-Trade Realized PnL", template="plotly_white"
            )
            st.plotly_chart(fig_trade_pnl, use_container_width=True)

            fig_cum = go.Figure()
            fig_cum.add_trace(
                go.Scatter(
                    x=closed["datetime"],
                    y=closed["cum_realized_pnl"],
                    mode="lines",
                    name="Cumulative Realized PnL",
                )
            )
            fig_cum.update_layout(title="Cumulative Realized PnL", template="plotly_white")
            st.plotly_chart(fig_cum, use_container_width=True)

            decisive = closed[
                pd.to_numeric(closed["realized_pnl"], errors="coerce").fillna(0.0) != 0.0
            ]
            if not decisive.empty:
                outcomes = list((decisive["realized_pnl"] > 0.0).to_numpy(dtype=bool))
                streaks = _streak_groups(outcomes)
                streak_rows = []
                for flag, length in streaks:
                    streak_rows.append(
                        {
                            "Result": "Win" if flag else "Loss",
                            "Length": int(length),
                        }
                    )
                if streak_rows:
                    streak_df = pd.DataFrame(streak_rows)
                    dist = (
                        streak_df.groupby(["Length", "Result"], observed=False)
                        .size()
                        .reset_index(name="Count")
                    )
                    fig_streak = go.Figure()
                    for label, color in [("Win", "#2f855a"), ("Loss", "#c53030")]:
                        part = dist[dist["Result"] == label]
                        if part.empty:
                            continue
                        fig_streak.add_trace(
                            go.Bar(
                                x=part["Length"],
                                y=part["Count"],
                                name=label,
                                marker_color=color,
                            )
                        )
                    fig_streak.update_layout(
                        title="Win/Loss Streak Distribution",
                        xaxis_title="Streak Length",
                        yaxis_title="Occurrences",
                        barmode="group",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_streak, use_container_width=True)

    if not df_orders.empty:
        status_counts = df_orders["status"].fillna("UNKNOWN").astype(str).value_counts()
        fig_status = go.Figure(
            data=[
                go.Pie(
                    labels=status_counts.index.tolist(),
                    values=status_counts.values.tolist(),
                    hole=0.4,
                )
            ]
        )
        fig_status.update_layout(title="Order Status Distribution")
        st.plotly_chart(fig_status, use_container_width=True)

with tab_risk:
    if not df_risk.empty:
        reason_counts = df_risk["reason"].fillna("UNKNOWN").astype(str).value_counts()
        fig_risk = go.Figure(
            data=[go.Bar(x=reason_counts.index.tolist(), y=reason_counts.values.tolist())]
        )
        fig_risk.update_layout(title="Risk Event Counts by Reason", template="plotly_white")
        st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.info("No risk events recorded for selected run/data source.")

    if not df_hb.empty:
        hb = df_hb.copy()
        hb = hb.sort_values("heartbeat_time")
        hb["delta_sec"] = hb["heartbeat_time"].diff().dt.total_seconds()
        avg_hb = float(hb["delta_sec"].dropna().mean()) if hb["delta_sec"].notna().any() else 0.0
        st.metric("Avg Heartbeat Interval (sec)", f"{avg_hb:.2f}")

        fig_hb = go.Figure()
        fig_hb.add_trace(
            go.Scatter(
                x=hb["heartbeat_time"],
                y=hb["delta_sec"],
                mode="lines+markers",
                name="Heartbeat interval",
            )
        )
        fig_hb.update_layout(title="Heartbeat Interval Trend", template="plotly_white")
        st.plotly_chart(fig_hb, use_container_width=True)
    else:
        st.info("No heartbeats recorded for selected run/data source.")

    if not df_order_states.empty:
        state_counts = df_order_states["state"].fillna("UNKNOWN").astype(str).value_counts()
        fig_state = go.Figure(
            data=[go.Bar(x=state_counts.index.tolist(), y=state_counts.values.tolist())]
        )
        fig_state.update_layout(title="Order State Event Counts", template="plotly_white")
        st.plotly_chart(fig_state, use_container_width=True)

    trace_parts = []
    if not df_orders.empty and "created_at" in df_orders.columns:
        t = df_orders[["created_at", "symbol", "side", "status"]].copy()
        t = t.rename(columns={"created_at": "event_time", "side": "event_detail"})
        t["event_type"] = "order"
        trace_parts.append(t)
    if not df_risk.empty and "event_time" in df_risk.columns:
        t = df_risk[["event_time", "reason"]].copy()
        t["symbol"] = ""
        t["status"] = ""
        t = t.rename(columns={"reason": "event_detail"})
        t["event_type"] = "risk"
        trace_parts.append(t)
    if not df_hb.empty and "heartbeat_time" in df_hb.columns:
        t = df_hb[["heartbeat_time", "status"]].copy()
        t = t.rename(columns={"heartbeat_time": "event_time"})
        t["symbol"] = ""
        t["event_detail"] = "heartbeat"
        t["event_type"] = "heartbeat"
        trace_parts.append(t)
    if not df_order_states.empty and "event_time" in df_order_states.columns:
        t = df_order_states[["event_time", "symbol", "state", "message"]].copy()
        t = t.rename(columns={"state": "status", "message": "event_detail"})
        t["event_type"] = "order_state"
        trace_parts.append(t)

    if trace_parts:
        trace_df = pd.concat(trace_parts, ignore_index=True)
        trace_df = trace_df.sort_values("event_time", ascending=False).head(500)
        st.subheader("Strategy Process Trace")
        st.dataframe(trace_df, use_container_width=True)

with tab_market:
    if not plot_market.empty:
        colm1, colm2, colm3, colm4 = st.columns(4)
        colm1.metric("Market Bars", f"{len(plot_market)}")
        colm2.metric("First Price", f"{float(plot_market['close'].iloc[0]):.4f}")
        colm3.metric("Last Price", f"{float(plot_market['close'].iloc[-1]):.4f}")
        high_val = float(pd.to_numeric(plot_market["high"], errors="coerce").max())
        low_val = float(pd.to_numeric(plot_market["low"], errors="coerce").min())
        colm4.metric("Range", f"{low_val:.4f} - {high_val:.4f}")

        fig_close = go.Figure()
        fig_close.add_trace(
            go.Scatter(
                x=plot_market["datetime"],
                y=plot_market["close"],
                mode="lines",
                name="Close",
            )
        )
        fig_close.update_layout(
            title=f"{market_symbol} Close Price ({market_timeframe})",
            template="plotly_white",
        )
        st.plotly_chart(fig_close, use_container_width=True)

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=plot_market["datetime"], y=plot_market["volume"], name="Volume"))
        fig_vol.update_layout(title="Market Volume", template="plotly_white")
        st.plotly_chart(fig_vol, use_container_width=True)

        st.subheader("Strategy Indicator View")
        if strategy_name == "PairTradingZScoreStrategy":
            pair_symbol_x = str(
                strategy_params.get("symbol_x") or (runner_symbols[0] if runner_symbols else "")
            )
            pair_symbol_y = str(
                strategy_params.get("symbol_y")
                or (runner_symbols[1] if len(runner_symbols) > 1 else pair_symbol_x)
            )
            if pair_symbol_x == pair_symbol_y or not pair_symbol_x:
                st.warning("Pair strategy needs two different symbols (symbol_x / symbol_y).")
            else:
                pair_x_df = pd.DataFrame()
                pair_y_df = pd.DataFrame()
                if os.path.exists(db_path):
                    pair_x_df = load_market_ohlcv_sqlite(
                        db_path,
                        pair_symbol_x,
                        market_timeframe,
                        market_exchange,
                        refresh_counter=refresh_counter,
                        max_points=max_points,
                    )
                    pair_y_df = load_market_ohlcv_sqlite(
                        db_path,
                        pair_symbol_y,
                        market_timeframe,
                        market_exchange,
                        refresh_counter=refresh_counter,
                        max_points=max_points,
                    )

                pair_x_df = _apply_period_filter(pair_x_df, "datetime", period_start, period_end)
                pair_y_df = _apply_period_filter(pair_y_df, "datetime", period_start, period_end)
                pair_indicator_df = _build_pair_indicator_frame(
                    pair_x_df, pair_y_df, strategy_params
                )
                if pair_indicator_df.empty:
                    st.info("Not enough aligned market bars to compute pair indicators.")
                else:
                    entry_z = float(strategy_params.get("entry_z", 2.0))
                    exit_z = float(strategy_params.get("exit_z", 0.35))
                    stop_z = float(strategy_params.get("stop_z", 3.5))

                    z_series = pd.to_numeric(pair_indicator_df["zscore"], errors="coerce")
                    hedge_series = pd.to_numeric(pair_indicator_df["hedge_ratio"], errors="coerce")
                    corr_series = pd.to_numeric(pair_indicator_df["correlation"], errors="coerce")

                    latest_z = (
                        float(z_series.dropna().iloc[-1])
                        if z_series.notna().any()
                        else float("nan")
                    )
                    latest_beta = (
                        float(hedge_series.dropna().iloc[-1])
                        if hedge_series.notna().any()
                        else float("nan")
                    )
                    latest_corr = (
                        float(corr_series.dropna().iloc[-1])
                        if corr_series.notna().any()
                        else float("nan")
                    )

                    pm1, pm2, pm3, pm4 = st.columns(4)
                    pm1.metric("Pair", f"{pair_symbol_x} vs {pair_symbol_y}")
                    pm2.metric("Latest Z", f"{latest_z:.3f}" if math.isfinite(latest_z) else "N/A")
                    pm3.metric(
                        "Hedge Ratio", f"{latest_beta:.4f}" if math.isfinite(latest_beta) else "N/A"
                    )
                    pm4.metric(
                        "Correlation", f"{latest_corr:.4f}" if math.isfinite(latest_corr) else "N/A"
                    )

                    pair_plot_df = (
                        _downsample_frame(pair_indicator_df, downsample_target_points)
                        if auto_downsample
                        else pair_indicator_df
                    )

                    fig_pair_norm = go.Figure()
                    normalized_x = pair_plot_df["close_x"] / float(pair_plot_df["close_x"].iloc[0])
                    normalized_y = pair_plot_df["close_y"] / float(pair_plot_df["close_y"].iloc[0])
                    fig_pair_norm.add_trace(
                        go.Scatter(
                            x=pair_plot_df["datetime"],
                            y=normalized_x,
                            mode="lines",
                            name=f"{pair_symbol_x} (normalized)",
                            line=dict(color="#2b6cb0", width=1.8),
                        )
                    )
                    fig_pair_norm.add_trace(
                        go.Scatter(
                            x=pair_plot_df["datetime"],
                            y=normalized_y,
                            mode="lines",
                            name=f"{pair_symbol_y} (normalized)",
                            line=dict(color="#ed8936", width=1.8),
                        )
                    )
                    fig_pair_norm.update_layout(
                        title="Pair Price Inputs (Normalized)",
                        xaxis_title="Date",
                        yaxis_title="Normalized Price",
                        template="plotly_white",
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_pair_norm, use_container_width=True)

                    fig_z = go.Figure()
                    fig_z.add_trace(
                        go.Scatter(
                            x=pair_plot_df["datetime"],
                            y=pair_plot_df["zscore"],
                            mode="lines",
                            name="Z-Score",
                            line=dict(color="#2b6cb0", width=2),
                        )
                    )
                    fig_z.add_hline(y=entry_z, line_dash="dash", line_color="#c53030")
                    fig_z.add_hline(y=-entry_z, line_dash="dash", line_color="#c53030")
                    fig_z.add_hline(y=exit_z, line_dash="dot", line_color="#2f855a")
                    fig_z.add_hline(y=-exit_z, line_dash="dot", line_color="#2f855a")
                    fig_z.add_hline(y=stop_z, line_dash="dash", line_color="#7b341e")
                    fig_z.add_hline(y=-stop_z, line_dash="dash", line_color="#7b341e")
                    fig_z.update_layout(
                        title="Pair Z-Score with Entry/Exit/Stop Bands",
                        xaxis_title="Date",
                        yaxis_title="Z-Score",
                        template="plotly_white",
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_z, use_container_width=True)

                    fig_spread = go.Figure()
                    fig_spread.add_trace(
                        go.Scatter(
                            x=pair_plot_df["datetime"],
                            y=pair_plot_df["spread"],
                            mode="lines",
                            name="Spread",
                            line=dict(color="#4a5568", width=1.6),
                        )
                    )
                    fig_spread.add_trace(
                        go.Scatter(
                            x=pair_plot_df["datetime"],
                            y=pair_plot_df["spread_mean"],
                            mode="lines",
                            name="Spread Mean",
                            line=dict(color="#2f855a", width=1.2, dash="dash"),
                        )
                    )
                    fig_spread.update_layout(
                        title="Hedge-Adjusted Spread",
                        xaxis_title="Date",
                        yaxis_title="Spread",
                        template="plotly_white",
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_spread, use_container_width=True)
        else:
            indicator_df = _build_strategy_indicator_frame(
                plot_market, strategy_name, strategy_params
            )
            if indicator_df.empty:
                st.info("Not enough market bars to compute strategy indicators.")
            elif strategy_name == "RsiStrategy":
                rsi_period = max(2, int(strategy_params.get("rsi_period", 14)))
                oversold = float(strategy_params.get("oversold", 30.0))
                overbought = float(strategy_params.get("overbought", 70.0))

                rsi_series = pd.to_numeric(indicator_df["rsi"], errors="coerce")
                latest_rsi = (
                    float(rsi_series.dropna().iloc[-1])
                    if rsi_series.notna().any()
                    else float("nan")
                )
                rsi_zone = "N/A"
                if math.isfinite(latest_rsi):
                    if latest_rsi <= oversold:
                        rsi_zone = "Oversold"
                    elif latest_rsi >= overbought:
                        rsi_zone = "Overbought"
                    else:
                        rsi_zone = "Neutral"

                rm1, rm2, rm3 = st.columns(3)
                rm1.metric("RSI Period", f"{rsi_period}")
                rm2.metric(
                    "Latest RSI", f"{latest_rsi:.2f}" if math.isfinite(latest_rsi) else "N/A"
                )
                rm3.metric("RSI Zone", rsi_zone)

                fig_rsi = go.Figure()
                fig_rsi.add_trace(
                    go.Scatter(
                        x=indicator_df["datetime"],
                        y=rsi_series,
                        mode="lines",
                        name="RSI",
                        line=dict(color="#2b6cb0", width=2),
                    )
                )
                fig_rsi.add_hline(y=oversold, line_dash="dash", line_color="#2f855a")
                fig_rsi.add_hline(y=overbought, line_dash="dash", line_color="#c53030")
                fig_rsi.update_layout(
                    title=f"RSI ({rsi_period}) with Oversold/Overbought Bands",
                    xaxis_title="Date",
                    yaxis_title="RSI",
                    yaxis=dict(range=[0, 100]),
                    template="plotly_white",
                    hovermode="x unified",
                )
                st.plotly_chart(fig_rsi, use_container_width=True)

                prev_rsi = rsi_series.shift(1)
                long_entries = indicator_df[(prev_rsi >= oversold) & (rsi_series < oversold)]
                exits = indicator_df[(prev_rsi <= overbought) & (rsi_series > overbought)]
                fig_rsi_signals = go.Figure()
                fig_rsi_signals.add_trace(
                    go.Scatter(
                        x=indicator_df["datetime"],
                        y=indicator_df["close"],
                        mode="lines",
                        name="Close",
                        line=dict(color="#4a5568", width=1.5),
                    )
                )
                if not long_entries.empty:
                    fig_rsi_signals.add_trace(
                        go.Scatter(
                            x=long_entries["datetime"],
                            y=long_entries["close"],
                            mode="markers",
                            name="RSI Long Trigger",
                            marker=dict(color="#2f855a", size=8, symbol="triangle-up"),
                        )
                    )
                if not exits.empty:
                    fig_rsi_signals.add_trace(
                        go.Scatter(
                            x=exits["datetime"],
                            y=exits["close"],
                            mode="markers",
                            name="RSI Exit Trigger",
                            marker=dict(color="#c53030", size=8, symbol="triangle-down"),
                        )
                    )
                fig_rsi_signals.update_layout(
                    title="Price with RSI Trigger Points",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    template="plotly_white",
                    hovermode="x unified",
                )
                st.plotly_chart(fig_rsi_signals, use_container_width=True)

            elif strategy_name == "MovingAverageCrossStrategy":
                short_window = max(2, int(strategy_params.get("short_window", 10)))
                long_window = max(short_window + 1, int(strategy_params.get("long_window", 30)))

                ma_frame = indicator_df.copy()
                short_ma = pd.to_numeric(ma_frame.get("short_ma"), errors="coerce")
                long_ma = pd.to_numeric(ma_frame.get("long_ma"), errors="coerce")

                mm1, mm2 = st.columns(2)
                mm1.metric("Short Window", f"{short_window}")
                mm2.metric("Long Window", f"{long_window}")

                fig_ma = go.Figure()
                fig_ma.add_trace(
                    go.Scatter(
                        x=ma_frame["datetime"],
                        y=ma_frame["close"],
                        mode="lines",
                        name="Close",
                        line=dict(color="#4a5568", width=1.5),
                    )
                )
                fig_ma.add_trace(
                    go.Scatter(
                        x=ma_frame["datetime"],
                        y=short_ma,
                        mode="lines",
                        name=f"Short MA ({short_window})",
                        line=dict(color="#2b6cb0", width=1.8),
                    )
                )
                fig_ma.add_trace(
                    go.Scatter(
                        x=ma_frame["datetime"],
                        y=long_ma,
                        mode="lines",
                        name=f"Long MA ({long_window})",
                        line=dict(color="#ed8936", width=1.8),
                    )
                )

                prev_short = short_ma.shift(1)
                prev_long = long_ma.shift(1)
                cross_up = ma_frame[(prev_short <= prev_long) & (short_ma > long_ma)]
                cross_down = ma_frame[(prev_short >= prev_long) & (short_ma < long_ma)]
                if not cross_up.empty:
                    fig_ma.add_trace(
                        go.Scatter(
                            x=cross_up["datetime"],
                            y=cross_up["close"],
                            mode="markers",
                            name="MA Long Trigger",
                            marker=dict(color="#2f855a", size=8, symbol="triangle-up"),
                        )
                    )
                if not cross_down.empty:
                    fig_ma.add_trace(
                        go.Scatter(
                            x=cross_down["datetime"],
                            y=cross_down["close"],
                            mode="markers",
                            name="MA Exit Trigger",
                            marker=dict(color="#c53030", size=8, symbol="triangle-down"),
                        )
                    )

                fig_ma.update_layout(
                    title="Moving Average Strategy Inputs and Cross Triggers",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    template="plotly_white",
                    hovermode="x unified",
                )
                st.plotly_chart(fig_ma, use_container_width=True)
    else:
        st.info("No market_ohlcv rows available for selected symbol/timeframe/exchange.")

with tab_opt:
    st.subheader("Optimization Results")
    if df_optimize.empty:
        st.info("No optimization_results rows found in SQLite yet.")
    else:
        opt_run_ids = sorted(df_optimize["run_id"].dropna().astype(str).unique().tolist())
        opt_stages = sorted(df_optimize["stage"].dropna().astype(str).unique().tolist())

        opt_col_1, opt_col_2 = st.columns(2)
        selected_opt_run = opt_col_1.selectbox(
            "Optimization Run ID",
            ["All", *opt_run_ids],
            key="opt_run_id_filter",
        )
        selected_opt_stage = opt_col_2.selectbox(
            "Optimization Stage",
            ["All", *opt_stages],
            key="opt_stage_filter",
        )

        opt_filtered = df_optimize.copy()
        if selected_opt_run != "All":
            opt_filtered = opt_filtered[opt_filtered["run_id"].astype(str) == selected_opt_run]
        if selected_opt_stage != "All":
            opt_filtered = opt_filtered[opt_filtered["stage"].astype(str) == selected_opt_stage]

        if opt_filtered.empty:
            st.warning("No rows matched the optimization filters.")
        else:
            sharpe_series = pd.to_numeric(opt_filtered["sharpe"], errors="coerce")
            robust_series = pd.to_numeric(opt_filtered["robustness_score"], errors="coerce")
            train_series = pd.to_numeric(opt_filtered["train_sharpe"], errors="coerce")
            top_idx = sharpe_series.idxmax() if sharpe_series.notna().any() else None
            top_row = opt_filtered.loc[top_idx] if top_idx is not None else None

            optm1, optm2, optm3, optm4 = st.columns(4)
            optm1.metric("Rows", f"{len(opt_filtered)}")
            optm2.metric(
                "Best Sharpe",
                f"{float(sharpe_series.max()):.4f}" if sharpe_series.notna().any() else "N/A",
            )
            optm3.metric(
                "Median Sharpe",
                f"{float(sharpe_series.median()):.4f}" if sharpe_series.notna().any() else "N/A",
            )
            optm4.metric(
                "Median Robustness",
                f"{float(robust_series.median()):.4f}" if robust_series.notna().any() else "N/A",
            )

            if sharpe_series.notna().any() and train_series.notna().any():
                fig_opt_scatter = go.Figure()
                fig_opt_scatter.add_trace(
                    go.Scatter(
                        x=train_series,
                        y=sharpe_series,
                        mode="markers",
                        marker=dict(size=8, color="#2b6cb0", opacity=0.8),
                        text=opt_filtered["stage"].astype(str),
                        hovertemplate=(
                            "Stage: %{text}<br>"
                            "Train Sharpe: %{x:.4f}<br>"
                            "Current Sharpe: %{y:.4f}<extra></extra>"
                        ),
                        name="Candidates",
                    )
                )
                fig_opt_scatter.update_layout(
                    title="Optimization Candidate Quality",
                    xaxis_title="Train Sharpe",
                    yaxis_title="Sharpe",
                    template="plotly_white",
                )
                st.plotly_chart(fig_opt_scatter, use_container_width=True)

            table_cols = [
                "created_at",
                "run_id",
                "stage",
                "sharpe",
                "train_sharpe",
                "robustness_score",
                "cagr",
                "mdd",
            ]
            if "params" in opt_filtered.columns:
                opt_filtered = opt_filtered.copy()
                opt_filtered["params_view"] = opt_filtered["params"].apply(
                    lambda v: json.dumps(v, ensure_ascii=False)
                )
                table_cols.append("params_view")
            st.dataframe(
                opt_filtered[table_cols].sort_values("created_at", ascending=False).head(500),
                use_container_width=True,
            )

            if top_row is not None:
                st.caption("Best row by Sharpe")
                st.json(
                    {
                        "run_id": top_row.get("run_id"),
                        "stage": top_row.get("stage"),
                        "sharpe": float(top_row.get("sharpe", 0.0)),
                        "params": top_row.get("params", {}),
                        "extra": top_row.get("extra", {}),
                    }
                )

with tab_report:
    st.subheader("No-Code Workflow Control")
    st.caption(
        f"Runner overrides -> initial_equity={runner_initial_capital:.2f}, "
        f"leverage={runner_leverage}, timeframe={runner_timeframe}, symbols={runner_symbols}"
    )
    if opt_space_error:
        st.error(opt_space_error)

    workflow_jobs = load_workflow_jobs(db_path, refresh_counter=refresh_counter)
    active_live_jobs = pd.DataFrame()
    if not workflow_jobs.empty:
        active_live_jobs = workflow_jobs[
            (workflow_jobs["workflow"].isin(["live", "live_ws"]))
            & (workflow_jobs["status"].isin(["RUNNING", "STOP_REQUESTED"]))
        ]
    if not active_live_jobs.empty:
        st.warning(
            f"{len(active_live_jobs)} live job(s) already active. "
            "Stop existing live jobs before launching a new one."
        )

    live_col_1, live_col_2, live_col_3 = st.columns(3)
    live_runner_kind = live_col_1.selectbox(
        "Live Runner",
        ["Polling (run_live.py)", "WebSocket (run_live_ws.py)"],
        key="live_runner_kind",
    )
    live_mode = live_col_2.selectbox("Live Mode", ["paper", "real"], key="live_mode")
    live_strategy_index = (
        strategy_options.index(strategy_name) if strategy_name in strategy_options else 0
    )
    live_strategy_name = live_col_3.selectbox(
        "Live Strategy",
        strategy_options,
        index=live_strategy_index,
        key="live_strategy_name",
    )

    live_real_armed = True
    if live_mode == "real":
        st.warning(
            "Real mode sends live exchange orders. Use only after paper/soak validation is complete."
        )
        arm_col_1, arm_col_2 = st.columns(2)
        arm_ack_1 = arm_col_1.checkbox(
            "I understand this can place real orders.",
            key="arm_ack_1",
        )
        arm_ack_2 = arm_col_2.checkbox(
            "I confirmed API keys/account/margin settings.",
            key="arm_ack_2",
        )
        arm_phrase = st.text_input("Type ENABLE REAL to arm", key="arm_phrase")
        live_real_armed = arm_ack_1 and arm_ack_2 and arm_phrase.strip().upper() == "ENABLE REAL"
        if not live_real_armed:
            st.info("Real mode is locked until all arm checks are completed.")

    run_col_1, run_col_2, run_col_3 = st.columns(3)
    with run_col_1:
        if st.button("Start Backtest Job", type="primary", use_container_width=True):
            params_path = _save_strategy_params(strategy_name, strategy_params)
            backtest_run_id = str(uuid.uuid4())
            backtest_args = [
                "--data-source",
                runner_data_source,
                "--market-db-path",
                db_path,
                "--market-exchange",
                market_exchange,
                "--run-id",
                backtest_run_id,
            ]
            job_id = _launch_managed_job(
                db_path=db_path,
                workflow="backtest",
                script_name="run_backtest.py",
                script_args=backtest_args,
                env_overrides=runner_env_overrides,
                requested_mode="backtest",
                strategy=strategy_name,
                run_id=backtest_run_id,
                metadata={"strategy_params_path": params_path},
            )
            st.success(f"Backtest job launched: {job_id}")
            st.cache_data.clear()

    with run_col_2:
        if st.button("Start Optimization Job", use_container_width=True):
            optimize_run_id = str(uuid.uuid4())
            optimize_args = [
                "--folds",
                str(int(optimize_folds)),
                "--n-trials",
                str(int(optimize_trials)),
                "--max-workers",
                str(int(optimize_workers)),
                "--data-source",
                runner_data_source,
                "--market-db-path",
                db_path,
                "--market-exchange",
                market_exchange,
                "--run-id",
                optimize_run_id,
            ]
            if persist_best_params:
                optimize_args.append("--save-best-params")
            job_id = _launch_managed_job(
                db_path=db_path,
                workflow="optimize",
                script_name="optimize.py",
                script_args=optimize_args,
                env_overrides=runner_env_overrides,
                requested_mode="optimize",
                strategy=strategy_name,
                run_id=optimize_run_id,
                metadata={
                    "folds": int(optimize_folds),
                    "n_trials": int(optimize_trials),
                    "max_workers": int(optimize_workers),
                },
            )
            st.success(f"Optimization job launched: {job_id}")
            st.cache_data.clear()

    with run_col_3:
        start_live_disabled = (
            live_mode == "real" and not live_real_armed
        ) or not active_live_jobs.empty
        if st.button("Start Live Job", use_container_width=True, disabled=start_live_disabled):
            live_run_id = str(uuid.uuid4())
            stop_file = _build_stop_file_path(live_run_id)
            live_script = "run_live.py"
            live_workflow = "live"
            if "WebSocket" in live_runner_kind:
                live_script = "run_live_ws.py"
                live_workflow = "live_ws"

            live_args = [
                "--strategy",
                live_strategy_name,
                "--run-id",
                live_run_id,
                "--stop-file",
                stop_file,
            ]
            live_env = dict(runner_env_overrides)
            live_env["LQ__LIVE__MODE"] = str(live_mode)
            live_env["LQ__LIVE__EXCHANGE__NAME"] = str(market_exchange).lower()
            live_env["LQ__LIVE__EXCHANGE__LEVERAGE"] = str(int(runner_leverage))
            if live_mode == "real":
                live_args.append("--enable-live-real")
                live_env["LUMINA_ENABLE_LIVE_REAL"] = "true"

            job_id = _launch_managed_job(
                db_path=db_path,
                workflow=live_workflow,
                script_name=live_script,
                script_args=live_args,
                env_overrides=live_env,
                requested_mode=live_mode,
                strategy=live_strategy_name,
                run_id=live_run_id,
                stop_file=stop_file,
                metadata={"runner_kind": live_runner_kind},
            )
            st.success(f"Live job launched: {job_id}")
            st.cache_data.clear()

    st.subheader("Workflow Jobs")
    workflow_jobs = load_workflow_jobs(db_path, refresh_counter=refresh_counter)
    if workflow_jobs.empty:
        st.info("No workflow jobs recorded yet.")
    else:
        jobs_view = workflow_jobs.copy()
        jobs_view["command"] = jobs_view["command_json"].fillna("").astype(str).str.slice(0, 120)
        st.dataframe(
            jobs_view[
                [
                    "started_at",
                    "workflow",
                    "status",
                    "requested_mode",
                    "strategy",
                    "pid",
                    "run_id",
                    "exit_code",
                    "command",
                ]
            ],
            use_container_width=True,
        )

        active_jobs = workflow_jobs[
            workflow_jobs["status"].isin(["RUNNING", "STOP_REQUESTED"])
        ].copy()
        if not active_jobs.empty:
            ctrl_job_id = st.selectbox(
                "Control Active Job",
                active_jobs["job_id"].astype(str).tolist(),
                key="control_active_job_id",
            )
            ctrl_row = active_jobs[active_jobs["job_id"].astype(str) == str(ctrl_job_id)].iloc[0]
            ctrl_col_1, ctrl_col_2 = st.columns(2)
            can_stop = (
                bool(ctrl_row.get("stop_file")) and ctrl_row.get("status") != "STOP_REQUESTED"
            )
            if ctrl_col_1.button(
                "Request Graceful Stop",
                use_container_width=True,
                disabled=not can_stop,
                key=f"request_stop_{ctrl_job_id}",
            ):
                if _request_job_stop(db_path, str(ctrl_row.get("stop_file") or "")):
                    _update_workflow_job_row(db_path, ctrl_job_id, status="STOP_REQUESTED")
                    st.success(f"Stop requested for {ctrl_job_id}")
                    st.cache_data.clear()
                else:
                    st.error("This job does not expose a graceful stop file.")

            if ctrl_col_2.button(
                "Force Kill Process",
                use_container_width=True,
                key=f"force_kill_{ctrl_job_id}",
            ):
                ok, detail = _terminate_process(ctrl_row.get("pid"))
                if ok:
                    _update_workflow_job_row(
                        db_path,
                        ctrl_job_id,
                        status="KILLED",
                        ended_at=_utc_now_iso(),
                        exit_code=-9,
                    )
                    if "workflow_processes" in st.session_state:
                        st.session_state["workflow_processes"].pop(ctrl_job_id, None)
                    st.success(f"Killed {ctrl_job_id}")
                    st.cache_data.clear()
                else:
                    st.error(f"Kill failed: {detail}")

        log_job_id = st.selectbox(
            "Job Log Viewer",
            workflow_jobs["job_id"].astype(str).tolist(),
            key="workflow_log_viewer_job",
        )
        log_row = workflow_jobs[workflow_jobs["job_id"].astype(str) == str(log_job_id)].iloc[0]
        st.caption(f"Log path: {log_row.get('log_path')}")
        st.text_area(
            "Job Log Tail",
            value=_tail_text_file(str(log_row.get("log_path") or ""), max_chars=25000),
            height=260,
            key="workflow_log_tail_view",
        )

    st.subheader("Ghost Cleanup")
    st.caption(
        "Close stale RUNNING runs and reconcile orphan workflow_jobs. "
        "Use dry-run first, then apply."
    )
    cleanup_col_1, cleanup_col_2, cleanup_col_3, cleanup_col_4 = st.columns(4)
    cleanup_stale_sec = cleanup_col_1.number_input(
        "Stale Sec",
        min_value=60,
        max_value=86400,
        value=max(300, int(run_stale_sec)),
        step=30,
        key="ghost_cleanup_stale_sec",
    )
    cleanup_startup_grace_sec = cleanup_col_2.number_input(
        "Startup Grace Sec",
        min_value=30,
        max_value=7200,
        value=90,
        step=30,
        key="ghost_cleanup_startup_grace_sec",
    )
    cleanup_close_status = cleanup_col_3.selectbox(
        "Close Status",
        ["STOPPED", "FAILED"],
        index=0,
        key="ghost_cleanup_close_status",
    )
    cleanup_force_kill_age = cleanup_col_4.number_input(
        "Force Kill STOP_REQUESTED Age Sec",
        min_value=0,
        max_value=86400,
        value=0,
        step=30,
        key="ghost_cleanup_force_kill_age",
    )

    cleanup_btn_col_1, cleanup_btn_col_2 = st.columns(2)
    if cleanup_btn_col_1.button("Ghost Cleanup Dry-Run", use_container_width=True):
        cleanup_result = _run_ghost_cleanup_script(
            db_path=db_path,
            stale_sec=cleanup_stale_sec,
            startup_grace_sec=cleanup_startup_grace_sec,
            close_status=cleanup_close_status,
            force_kill_stop_requested_after_sec=cleanup_force_kill_age,
            apply_changes=False,
        )
        cleanup_result["mode"] = "dry_run"
        st.session_state["ghost_cleanup_last_result"] = cleanup_result

    if cleanup_btn_col_2.button("Ghost Cleanup Apply", use_container_width=True):
        cleanup_result = _run_ghost_cleanup_script(
            db_path=db_path,
            stale_sec=cleanup_stale_sec,
            startup_grace_sec=cleanup_startup_grace_sec,
            close_status=cleanup_close_status,
            force_kill_stop_requested_after_sec=cleanup_force_kill_age,
            apply_changes=True,
        )
        cleanup_result["mode"] = "apply"
        st.session_state["ghost_cleanup_last_result"] = cleanup_result
        st.cache_data.clear()

    cleanup_result = st.session_state.get("ghost_cleanup_last_result")
    if cleanup_result:
        if cleanup_result.get("ok"):
            st.success(
                f"Ghost cleanup {cleanup_result.get('mode', 'run')} completed in "
                f"{cleanup_result.get('elapsed_sec', 0.0):.2f}s"
            )
        else:
            st.error(f"Ghost cleanup failed (returncode={cleanup_result.get('returncode')})")
        st.caption(f"Command: {' '.join(cleanup_result.get('command', []))}")
        if cleanup_result.get("payload") is not None:
            st.json(cleanup_result["payload"])
        st.text_area(
            "Ghost Cleanup Output",
            value=cleanup_result.get("output", ""),
            height=240,
            key="ghost_cleanup_output_view",
        )

    payload = build_report_payload(
        summary,
        performance,
        active_run_id,
        resolved_source,
        strategy_name,
        period_preset,
        df_equity,
        trade_analytics,
        df_risk,
        df_hb,
        runtime_overrides={
            "initial_capital": float(runner_initial_capital),
            "backtest_leverage": int(runner_leverage),
            "symbols": runner_symbols,
            "timeframe": runner_timeframe,
            "runner_data_source": runner_data_source,
            "runner_timeout_sec": int(runner_timeout_sec),
        },
        strategy_params=strategy_params,
        mirror_snapshot=mirror_snapshot,
        balance_equity_series=_serialize_balance_equity_frame(mirror_balance_equity, limit=1500),
    )

    if st.button("Generate Snapshot Report", type="primary"):
        json_path, md_path, markdown = save_report_snapshot(payload)
        st.success(f"Report saved: {json_path} | {md_path}")
        st.download_button(
            label="Download Markdown Report",
            data=markdown,
            file_name=os.path.basename(md_path),
            mime="text/markdown",
        )
        st.download_button(
            label="Download JSON Report",
            data=json.dumps(payload, indent=2, ensure_ascii=False),
            file_name=os.path.basename(json_path),
            mime="application/json",
        )

    if payload.get("mt5_summary"):
        st.subheader("Snapshot MT5 Summary")
        st.dataframe(pd.DataFrame(payload.get("mt5_summary", [])), use_container_width=True)

    monthly_payload = payload.get("monthly_returns") or {}
    if monthly_payload:
        st.subheader("Snapshot Monthly Returns")
        monthly_df = pd.DataFrame.from_dict(monthly_payload, orient="index")
        monthly_df.index.name = "Year"
        st.dataframe(monthly_df, use_container_width=True)

    st.subheader("Current Snapshot Preview")
    st.json(payload)

with tab_raw:
    st.caption(
        f"Run: {active_run_id if active_run_id else 'N/A'} | Source: {resolved_source} | "
        f"Market: {market_symbol} {market_timeframe} ({market_exchange})"
    )
    with st.expander("Runs"):
        st.dataframe(runs_df)
    with st.expander("Equity"):
        st.dataframe(df_equity)
    with st.expander("Fills (enriched)"):
        st.dataframe(trade_analytics)
    with st.expander("Orders"):
        st.dataframe(df_orders)
    with st.expander("Risk Events"):
        st.dataframe(df_risk)
    with st.expander("Heartbeats"):
        st.dataframe(df_hb)
    with st.expander("Order State Events"):
        st.dataframe(df_order_states)
    with st.expander("Market OHLCV"):
        st.dataframe(df_market)
    with st.expander("Optimization Results"):
        st.dataframe(df_optimize)
    with st.expander("Workflow Jobs"):
        st.dataframe(load_workflow_jobs(db_path, refresh_counter=refresh_counter))

if df_equity.empty:
    st.warning(
        "No equity data found for current selection. "
        "Start a backtest job from Report Export tab, or switch source/period. "
        f"Configured initial equity is {runner_initial_capital:.2f}."
    )
