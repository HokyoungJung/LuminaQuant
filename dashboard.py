import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os

# Page Config
st.set_page_config(layout="wide", page_title="Quants Agent Dashboard")

st.title("🤖 Quants Agent: Trading Performance")

# Sidebar
st.sidebar.header("Configuration")
strategy_name = st.sidebar.selectbox(
    "Select Strategy", ["RsiStrategy", "MovingAverageCrossStrategy"]
)


@st.cache_data
def load_data():
    if not os.path.exists("equity.csv"):
        return None
    df = pd.read_csv("equity.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def load_params(strategy):
    path = os.path.join("best_optimized_parameters", strategy, "best_params.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


# Load Data
df = load_data()
params = load_params(strategy_name)


@st.cache_data
def load_trades():
    if not os.path.exists("trades.csv"):
        return None
    df_t = pd.read_csv("trades.csv")
    df_t["datetime"] = pd.to_datetime(df_t["datetime"])
    return df_t


df_trades = load_trades()

# Metrics Section
st.header("1. Performance Metrics")

if df is not None:
    # ROI
    initial_equity = df["total"].iloc[0]
    final_equity = df["total"].iloc[-1]
    total_return = (final_equity - initial_equity) / initial_equity

    # Max Drawdown
    roll_max = df["total"].cummax()
    drawdown = (df["total"] - roll_max) / roll_max
    max_dd = drawdown.min()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{total_return:.2%}")
    col2.metric("Final Equity", f"${final_equity:,.2f}")
    col3.metric("Max Drawdown", f"{max_dd:.2%}")
    if df_trades is not None:
        col4.metric("Total Trades", f"{len(df_trades)}")

    # Display Optimized Params
    st.subheader(f"🏆 Best Parameters for {strategy_name}")
    st.json(params)

    # --- PRICE CHART & TRADES ---
    st.header("2. Price Action & Trade Entries")
    fig_price = go.Figure()

    # Price Line
    if "benchmark_price" in df.columns:
        fig_price.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["benchmark_price"],
                mode="lines",
                name="Asset Price",
                line=dict(color="#636EFA", width=1),
            )
        )

    # Trade Markers
    if df_trades is not None:
        # Buys
        buys = df_trades[df_trades["direction"] == "BUY"]
        if not buys.empty:
            fig_price.add_trace(
                go.Scatter(
                    x=buys["datetime"],
                    y=buys["price"],
                    mode="markers",
                    name="Buy",
                    marker=dict(symbol="triangle-up", size=10, color="#00CC96"),
                )
            )

        # Sells
        sells = df_trades[df_trades["direction"] == "SELL"]
        if not sells.empty:
            fig_price.add_trace(
                go.Scatter(
                    x=sells["datetime"],
                    y=sells["price"],
                    mode="markers",
                    name="Sell",
                    marker=dict(symbol="triangle-down", size=10, color="#EF553B"),
                )
            )

    fig_price.update_layout(
        title="Asset Price with Buy/Sell Points",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_dark",
        hovermode="x unified",
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # --- EQUITY CURVE ---
    st.header("3. Equity Curve")

    # Interactive Plotly Chart
    fig = go.Figure()

    # Strategy Equity
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["total"],
            mode="lines",
            name="Strategy Equity",
            line=dict(color="#00CC96", width=2),
        )
    )

    # Benchmark (Normalized to Initial Capital)
    if "benchmark_price" in df.columns:
        norm_benchmark = (
            df["benchmark_price"] / df["benchmark_price"].iloc[0]
        ) * initial_equity
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=norm_benchmark,
                mode="lines",
                name="Benchmark (Buy & Hold)",
                line=dict(color="#EF553B", width=2, dash="dash"),
            )
        )

    fig.update_layout(
        title="Portfolio Value over Time",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        template="plotly_dark",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown Chart
    st.header("4. Drawdown Analysis")
    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=drawdown,
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="#FF4136"),
        )
    )
    fig_dd.update_layout(
        title="Underwater Plot", yaxis_title="Drawdown (%)", template="plotly_dark"
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # Data Table
    with st.expander("View Raw Data"):
        st.dataframe(df)

else:
    st.warning("No 'equity.csv' found. Please run 'run_backtest.py' first.")
