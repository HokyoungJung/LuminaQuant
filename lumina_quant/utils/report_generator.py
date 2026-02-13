import json
import os
from datetime import datetime

import pandas as pd
from lumina_quant.utils.performance import (
    create_annualized_volatility,
    create_cagr,
    create_calmar_ratio,
    create_drawdowns,
    create_sharpe_ratio,
    create_sortino_ratio,
)


class ReportGenerator:
    """Generates markdown reports from trading data."""

    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_report(self, report_type="Backtest", strategy_name="Unknown"):
        """Main entry point to generate a report."""
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{self.output_dir}/{report_type}_Report_{date_str}.md"

        # Load Data
        equity_df = self._load_equity(report_type)
        trades_df = self._load_trades(report_type)
        params = self._load_params(strategy_name)

        content = []
        content.append(f"# 📄 Trading Report: {report_type}")
        content.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"**Strategy:** {strategy_name}")
        content.append("")

        # 1. Strategy Parameters
        content.append("## ⚙️ Strategy Configuration")
        if params:
            content.append("| Parameter | Value |")
            content.append("| :--- | :--- |")
            for k, v in params.items():
                content.append(f"| {k} | {v} |")
        else:
            content.append("No parameters found.")
        content.append("")

        # 2. Performance Metrics
        content.append("## 📈 Performance Metrics")
        if equity_df is not None:
            metrics = self._calculate_metrics(equity_df)
            content.append("| Metric | Value |")
            content.append("| :--- | :--- |")
            for k, v in metrics.items():
                content.append(f"| {k} | {v} |")
        else:
            content.append("⚠️ No Equity Data Found.")
        content.append("")

        # 3. Trade Analysis
        content.append("## 📊 Trade Analysis")
        if trades_df is not None and not trades_df.empty:
            trade_stats = self._analyze_trades(trades_df)
            content.append("| Metric | Value |")
            content.append("| :--- | :--- |")
            for k, v in trade_stats.items():
                content.append(f"| {k} | {v} |")

            content.append("")
            content.append("### Recent Trades")
            content.append(trades_df.tail(10).to_markdown(index=False))
        else:
            content.append("No trades recorded.")
        content.append("")

        # Save
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(content))

        return filename

    def _load_equity(self, report_type):
        filename = "live_equity.csv" if report_type == "Live" else "equity.csv"
        if os.path.exists(filename):
            return pd.read_csv(filename)
        return None

    def _load_trades(self, report_type):
        filename = "live_trades.csv" if report_type == "Live" else "trades.csv"
        if os.path.exists(filename):
            return pd.read_csv(filename)
        return None

    def _load_params(self, strategy_name):
        path = os.path.join("best_optimized_parameters", strategy_name, "best_params.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def _calculate_metrics(self, df):
        # Assumes df has 'total' and 'returns' (or we calc them)
        if "total" not in df.columns:
            return {}

        # Recalculate returns if needed
        if "returns" not in df.columns:
            df["returns"] = df["total"].pct_change().fillna(0)

        total_series = df["total"].values
        returns = df["returns"].values
        periods = 252  # Default to daily

        # Stats
        total_ret = (total_series[-1] - total_series[0]) / total_series[0]
        cagr = create_cagr(total_series[-1], total_series[0], len(df), periods)
        sharpe = create_sharpe_ratio(returns, periods)
        sortino = create_sortino_ratio(returns, periods)
        vol = create_annualized_volatility(returns, periods)
        drawdown, _ = create_drawdowns(total_series)
        max_dd = max(drawdown) if len(drawdown) > 0 else 0
        calmar = create_calmar_ratio(cagr, max_dd)

        return {
            "Total Return": f"{total_ret:.2%}",
            "CAGR": f"{cagr:.2%}",
            "Sharpe Ratio": f"{sharpe:.4f}",
            "Sortino Ratio": f"{sortino:.4f}",
            "Max Drawdown": f"{max_dd:.2%}",
            "Calmar Ratio": f"{calmar:.4f}",
            "Annualized Volatility": f"{vol:.2%}",
        }

    def _analyze_trades(self, df):
        total_trades = len(df)
        buys = len(df[df["direction"] == "BUY"])
        sells = len(df[df["direction"] == "SELL"])

        # Naive Win Rate (requires logic to match Buy/Sell pairs, complex)
        # For now, just show counts

        return {
            "Total Transactions": total_trades,
            "Buy Orders": buys,
            "Sell Orders": sells,
        }


if __name__ == "__main__":
    # Test
    gen = ReportGenerator()
    print(f"Generated: {gen.generate_report('Backtest', 'RsiStrategy')}")
