import json
import os
import sqlite3

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide", page_title="LuminaQuant Dashboard")
st.title("LuminaQuant: Trading Performance")


@st.cache_data
def load_runs(db_path):
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT run_id, mode, started_at, ended_at, status FROM runs ORDER BY started_at DESC",
            conn,
        )
        return df
    finally:
        conn.close()


@st.cache_data
def load_equity_sqlite(db_path, run_id):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT timeindex AS datetime, total, cash FROM equity WHERE run_id = ? ORDER BY id",
            conn,
            params=[run_id],
        )
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            if "benchmark_price" not in df.columns:
                df["benchmark_price"] = pd.NA
        return df
    finally:
        conn.close()


@st.cache_data
def load_fills_sqlite(db_path, run_id):
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
                fill_price AS price
            FROM fills
            WHERE run_id = ?
            ORDER BY id
            """,
            conn,
            params=[run_id],
        )
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df["direction"] = (
                df["direction"]
                .str.upper()
                .map({"BUY": "BUY", "SELL": "SELL", "BUY_LONG": "BUY", "SELL_SHORT": "SELL"})
                .fillna(df["direction"])
            )
        return df
    finally:
        conn.close()


@st.cache_data
def load_equity_csv():
    if not os.path.exists("equity.csv"):
        return None
    df = pd.read_csv("equity.csv")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


@st.cache_data
def load_trades_csv():
    if not os.path.exists("trades.csv"):
        return None
    df = pd.read_csv("trades.csv")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


@st.cache_data
def load_params(strategy):
    path = os.path.join("best_optimized_parameters", strategy, "best_params.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


st.sidebar.header("Configuration")
data_source = st.sidebar.selectbox(
    "Data Source",
    ["Auto", "SQLite", "CSV"],
)
db_path = st.sidebar.text_input("SQLite Path", value="logs/lumina_quant.db")
strategy_name = st.sidebar.selectbox(
    "Select Strategy", ["RsiStrategy", "MovingAverageCrossStrategy"]
)
params = load_params(strategy_name)

df = None
df_trades = None
active_run_id = None

use_sqlite = data_source == "SQLite" or (data_source == "Auto" and os.path.exists(db_path))

if use_sqlite and os.path.exists(db_path):
    runs_df = load_runs(db_path)
    if runs_df.empty:
        st.sidebar.warning("No runs found in SQLite. Falling back to CSV.")
    else:
        run_options = runs_df["run_id"].tolist()
        active_run_id = st.sidebar.selectbox("Run ID", run_options)
        df = load_equity_sqlite(db_path, active_run_id)
        df_trades = load_fills_sqlite(db_path, active_run_id)

if df is None:
    df = load_equity_csv()
    df_trades = load_trades_csv()

st.header("1. Performance Metrics")

if df is not None and not df.empty:
    initial_equity = float(df["total"].iloc[0])
    final_equity = float(df["total"].iloc[-1])
    total_return = (final_equity - initial_equity) / initial_equity if initial_equity else 0.0

    roll_max = df["total"].cummax()
    drawdown = (df["total"] - roll_max) / roll_max
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{total_return:.2%}")
    col2.metric("Final Equity", f"${final_equity:,.2f}")
    col3.metric("Max Drawdown", f"{max_dd:.2%}")
    if df_trades is not None:
        col4.metric("Total Fills", f"{len(df_trades)}")

    st.subheader(f"Best Parameters for {strategy_name}")
    st.json(params)

    st.header("2. Price Action & Trade Entries")
    fig_price = go.Figure()

    if "benchmark_price" in df.columns and df["benchmark_price"].notna().any():
        fig_price.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["benchmark_price"],
                mode="lines",
                name="Asset Price",
                line=dict(color="#1f77b4", width=1),
            )
        )

    if df_trades is not None and not df_trades.empty and "price" in df_trades.columns:
        buys = df_trades[df_trades["direction"] == "BUY"]
        sells = df_trades[df_trades["direction"] == "SELL"]
        if not buys.empty:
            fig_price.add_trace(
                go.Scatter(
                    x=buys["datetime"],
                    y=buys["price"],
                    mode="markers",
                    name="Buy",
                    marker=dict(symbol="triangle-up", size=10, color="#0db39e"),
                )
            )
        if not sells.empty:
            fig_price.add_trace(
                go.Scatter(
                    x=sells["datetime"],
                    y=sells["price"],
                    mode="markers",
                    name="Sell",
                    marker=dict(symbol="triangle-down", size=10, color="#f05a66"),
                )
            )

    fig_price.update_layout(
        title="Asset Price with Buy/Sell Points",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
    )
    st.plotly_chart(fig_price, use_container_width=True)

    st.header("3. Equity Curve")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["total"],
            mode="lines",
            name="Strategy Equity",
            line=dict(color="#0db39e", width=2),
        )
    )

    if "benchmark_price" in df.columns and df["benchmark_price"].notna().any():
        norm_benchmark = (df["benchmark_price"] / df["benchmark_price"].iloc[0]) * initial_equity
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=norm_benchmark,
                mode="lines",
                name="Benchmark (Buy & Hold)",
                line=dict(color="#f05a66", width=2, dash="dash"),
            )
        )

    fig.update_layout(
        title="Portfolio Value over Time",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        template="plotly_white",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.header("4. Drawdown Analysis")
    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=drawdown,
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="#f05a66"),
        )
    )
    fig_dd.update_layout(
        title="Underwater Plot",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    with st.expander("View Raw Data"):
        st.dataframe(df)

    if active_run_id:
        st.caption(f"Active Run: {active_run_id}")
else:
    st.warning("No equity data found. Run backtest/live first.")
