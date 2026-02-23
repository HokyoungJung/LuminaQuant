import json
import math
import os
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
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
    """Generates local markdown/json reports from trading data."""

    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_report(self, report_type="Backtest", strategy_name="Unknown", source_dir="."):
        """Generate a local report artifact pair (md + json)."""
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        md_path = os.path.join(self.output_dir, f"{report_type}_Report_{date_str}.md")
        json_path = os.path.join(self.output_dir, f"{report_type}_Report_{date_str}.json")

        equity_df = self._prepare_equity(self._load_equity(report_type, source_dir))
        trades_df = self._prepare_trades(self._load_trades(report_type, source_dir))
        trades_df = self._compute_trade_analytics(trades_df)
        params = self._load_params(strategy_name, source_dir)

        performance = self._calculate_metrics(equity_df)
        trade_stats = self._analyze_trades(trades_df, equity_df)
        monthly_returns = self._build_monthly_returns(equity_df)
        mt5_rows = self._build_mt5_rows(performance, trade_stats)
        balance_equity_series = self._build_balance_equity_series(equity_df, trades_df)
        mirror_snapshot = {
            "total_trades": int(
                trade_stats.get("closed_trades", trade_stats.get("total_transactions", 0))
            ),
            "wins": int(trade_stats.get("wins", 0)),
            "losses": int(trade_stats.get("losses", 0)),
            "win_rate": float(trade_stats.get("win_rate", 0.0)),
            "closed_pnl": float(trade_stats.get("total_net_profit", 0.0)),
            "open_pnl": float(trade_stats.get("open_pnl", 0.0)),
            "total_c_plus_o": float(trade_stats.get("total_c_plus_o", 0.0)),
            "equity_mdd": float(trade_stats.get("equity_drawdown_maximal", 0.0)),
            "equity_mdd_rel": float(trade_stats.get("equity_drawdown_relative_pct", 0.0)),
            "r_mdd": float(trade_stats.get("r_mdd", 0.0)),
        }

        payload = {
            "generated_at": datetime.now().isoformat(),
            "report_type": report_type,
            "strategy": strategy_name,
            "strategy_params": params,
            "equity_rows": len(equity_df),
            "trade_rows": len(trades_df),
            "performance": performance,
            "trade_analysis": trade_stats,
            "mt5_summary": mt5_rows,
            "monthly_returns": self._serialize_monthly_returns(monthly_returns),
            "mirror_snapshot": mirror_snapshot,
            "balance_equity_series": balance_equity_series,
        }

        markdown = self._build_markdown(payload, trades_df, monthly_returns)
        with open(md_path, "w", encoding="utf-8") as file:
            file.write(markdown)
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

        return md_path

    def _load_equity(self, report_type, source_dir="."):
        filename = "live_equity.csv" if report_type == "Live" else "equity.csv"
        path = os.path.join(source_dir, filename)
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()

    def _load_trades(self, report_type, source_dir="."):
        filename = "live_trades.csv" if report_type == "Live" else "trades.csv"
        path = os.path.join(source_dir, filename)
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()

    def _load_params(self, strategy_name, source_dir="."):
        path = os.path.join(
            source_dir,
            "best_optimized_parameters",
            strategy_name,
            "best_params.json",
        )
        if os.path.exists(path):
            with open(path, encoding="utf-8") as file:
                return json.load(file)
        return {}

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        if not np.isfinite(out):
            return float(default)
        return out

    @staticmethod
    def _safe_div(numerator: Any, denominator: Any, default: float = 0.0) -> float:
        den = ReportGenerator._safe_float(denominator, 0.0)
        if abs(den) <= 1e-12:
            return float(default)
        return float(ReportGenerator._safe_float(numerator, 0.0) / den)

    @staticmethod
    def _format_pct(value: Any) -> str:
        return f"{ReportGenerator._safe_float(value):.2%}"

    @staticmethod
    def _format_num(value: Any, digits: int = 4) -> str:
        parsed = ReportGenerator._safe_float(value, 0.0)
        if math.isinf(parsed):
            return "inf"
        return f"{parsed:.{digits}f}"

    @staticmethod
    def _format_duration(seconds: Any) -> str:
        sec = round(max(0.0, ReportGenerator._safe_float(seconds, 0.0)))
        hours = sec // 3600
        minutes = (sec % 3600) // 60
        remain = sec % 60
        if hours > 0:
            return f"{hours}h {minutes}m"
        if minutes > 0:
            return f"{minutes}m {remain}s"
        return f"{remain}s"

    def _prepare_equity(self, df):
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        if "datetime" in out.columns:
            out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
        if "total" not in out.columns:
            return pd.DataFrame()
        out["total"] = pd.to_numeric(out["total"], errors="coerce")
        out = out.dropna(subset=["total"]).reset_index(drop=True)
        if out.empty:
            return out

        if "returns" in out.columns:
            returns = pd.to_numeric(out["returns"], errors="coerce")
        else:
            returns = out["total"].pct_change()
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["returns"] = returns
        return out

    def _prepare_trades(self, df):
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        if "datetime" in out.columns:
            out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
        if "symbol" not in out.columns:
            out["symbol"] = "UNKNOWN"
        if "direction" not in out.columns:
            out["direction"] = "BUY"
        for col in ["quantity", "price", "fill_cost", "commission"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
            else:
                out[col] = 0.0
        return out

    def _calculate_metrics(self, equity_df):
        if equity_df.empty:
            return {}

        total_series = equity_df["total"].to_numpy(dtype=np.float64)
        returns = equity_df["returns"].to_numpy(dtype=np.float64)
        periods = 252
        total_ret = self._safe_div(total_series[-1] - total_series[0], total_series[0], 0.0)
        cagr = create_cagr(total_series[-1], total_series[0], len(equity_df), periods)
        sharpe = create_sharpe_ratio(returns, periods)
        sortino = create_sortino_ratio(returns, periods)
        vol = create_annualized_volatility(returns, periods)
        drawdown, dd_duration = create_drawdowns(total_series)
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
        calmar = create_calmar_ratio(cagr, max_dd)

        return {
            "start_equity": float(total_series[0]),
            "end_equity": float(total_series[-1]),
            "bars": len(equity_df),
            "total_return": float(total_ret),
            "cagr": float(cagr),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "annualized_volatility": float(vol),
            "max_drawdown": float(max_dd),
            "drawdown_duration": int(dd_duration),
        }

    @staticmethod
    def _drawdown_stats(values: Any, initial_value: Any) -> dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"absolute": 0.0, "maximal": 0.0, "relative_pct": 0.0}

        peak = np.maximum.accumulate(arr)
        drawdown_abs = peak - arr
        drawdown_rel = np.divide(
            drawdown_abs,
            np.where(peak == 0.0, np.nan, peak),
            dtype=np.float64,
        )
        drawdown_rel = np.nan_to_num(drawdown_rel, nan=0.0, posinf=0.0, neginf=0.0)
        absolute = max(0.0, ReportGenerator._safe_float(initial_value, 0.0) - float(np.min(arr)))
        maximal = float(np.max(drawdown_abs))
        relative = float(np.max(drawdown_rel))
        return {"absolute": absolute, "maximal": maximal, "relative_pct": relative}

    @staticmethod
    def _streak_groups(sequence: Sequence[bool]) -> list[tuple[bool, int]]:
        if not sequence:
            return []
        groups = []
        current = sequence[0]
        count = 1
        for value in sequence[1:]:
            if value == current:
                count += 1
            else:
                groups.append((current, count))
                current = value
                count = 1
        groups.append((current, count))
        return groups

    @staticmethod
    def _runs_test_zscore(binary_outcomes: Sequence[bool]) -> float:
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

    def _compute_trade_analytics(self, trades_df):
        if trades_df.empty:
            return trades_df

        df = trades_df.copy().sort_values("datetime").reset_index(drop=True)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

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
            symbol = str(row.get("symbol", "UNKNOWN"))
            qty = abs(self._safe_float(row.get("quantity", 0.0), 0.0))
            price = self._safe_float(row.get("price", 0.0), 0.0)
            commission = self._safe_float(row.get("commission", 0.0), 0.0)
            direction = str(row.get("direction", "BUY")).upper()
            signed = qty if direction == "BUY" else -qty
            event_time = row.get("datetime", pd.NaT)
            if pd.isna(event_time):
                event_time = pd.NaT

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
                if abs(new_pos) > 0:
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

    def _build_balance_equity_series(self, equity_df, trades_df, max_points=1500):
        if (
            equity_df.empty
            or "datetime" not in equity_df.columns
            or "total" not in equity_df.columns
        ):
            return []

        frame = equity_df[["datetime", "total"]].copy()
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame["equity"] = pd.to_numeric(frame["total"], errors="coerce")
        frame = (
            frame.dropna(subset=["datetime", "equity"])
            .sort_values("datetime")
            .reset_index(drop=True)
        )
        if frame.empty:
            return []

        initial_equity = self._safe_float(frame["equity"].iloc[0], 0.0)
        frame["cum_realized_pnl"] = 0.0

        if (
            trades_df is not None
            and not trades_df.empty
            and "datetime" in trades_df.columns
            and "realized_pnl" in trades_df.columns
        ):
            closed = trades_df.copy()
            closed["datetime"] = pd.to_datetime(closed["datetime"], errors="coerce")
            closed = closed.dropna(subset=["datetime"])
            if "closed_qty" in closed.columns:
                closed_qty = pd.to_numeric(closed["closed_qty"], errors="coerce").fillna(0.0)
                closed = closed.loc[closed_qty > 0.0]
            if not closed.empty:
                closed["realized_pnl"] = pd.to_numeric(
                    closed["realized_pnl"], errors="coerce"
                ).fillna(0.0)
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
                    pd.to_numeric(aligned["cum_realized_pnl"], errors="coerce")
                    .fillna(0.0)
                    .to_numpy()
                )

        frame["balance"] = initial_equity + frame["cum_realized_pnl"]
        frame["open_pnl"] = frame["equity"] - frame["balance"]
        equity_peak = frame["equity"].cummax()
        frame["drawdown"] = -(equity_peak - frame["equity"])

        if len(frame) > int(max_points):
            step = max(1, len(frame) // int(max_points))
            keep = list(range(0, len(frame), step))
            if keep[-1] != len(frame) - 1:
                keep.append(len(frame) - 1)
            frame = frame.iloc[keep].reset_index(drop=True)

        payload = []
        for row in frame.itertuples(index=False):
            payload.append(
                {
                    "datetime": row.datetime.isoformat() if pd.notna(row.datetime) else None,
                    "equity": self._safe_float(row.equity, 0.0),
                    "balance": self._safe_float(row.balance, 0.0),
                    "open_pnl": self._safe_float(row.open_pnl, 0.0),
                    "drawdown": self._safe_float(row.drawdown, 0.0),
                }
            )
        return payload

    def _analyze_trades(self, trades_df, equity_df):
        out = {
            "total_transactions": 0,
            "buy_orders": 0,
            "sell_orders": 0,
            "closed_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_net_profit": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "expected_payoff": 0.0,
            "recovery_factor": 0.0,
            "r_mdd": 0.0,
            "open_pnl": 0.0,
            "total_c_plus_o": 0.0,
            "avg_trade_return_pct": 0.0,
            "best_trade_pnl": 0.0,
            "worst_trade_pnl": 0.0,
            "profit_trades_count": 0,
            "loss_trades_count": 0,
            "profit_trades_pct": 0.0,
            "loss_trades_pct": 0.0,
            "profit_trades_text": "0 (0.00%)",
            "loss_trades_text": "0 (0.00%)",
            "avg_profit_trade": 0.0,
            "avg_loss_trade": 0.0,
            "payoff_ratio": 0.0,
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
            "z_score": 0.0,
            "initial_equity": 0.0,
            "final_equity": 0.0,
            "equity_drawdown_absolute": 0.0,
            "equity_drawdown_maximal": 0.0,
            "equity_drawdown_relative_pct": 0.0,
            "holding_time_min_sec": 0.0,
            "holding_time_avg_sec": 0.0,
            "holding_time_max_sec": 0.0,
            "balance_drawdown_absolute": 0.0,
            "balance_drawdown_maximal": 0.0,
            "balance_drawdown_relative_pct": 0.0,
        }
        if trades_df.empty:
            return out

        out["total_transactions"] = len(trades_df)
        out["buy_orders"] = int((trades_df["direction"].astype(str).str.upper() == "BUY").sum())
        out["sell_orders"] = int((trades_df["direction"].astype(str).str.upper() == "SELL").sum())

        closed = (
            trades_df[trades_df["closed_qty"] > 0]
            if "closed_qty" in trades_df.columns
            else pd.DataFrame()
        )
        if closed.empty:
            return out

        pnl = pd.to_numeric(closed.get("realized_pnl"), errors="coerce").fillna(0.0)
        out["closed_trades"] = len(closed)
        out["wins"] = int((pnl > 0.0).sum())
        out["losses"] = int((pnl < 0.0).sum())
        out["profit_trades_count"] = int((pnl > 0.0).sum())
        out["loss_trades_count"] = int((pnl < 0.0).sum())
        denom = max(1, out["wins"] + out["losses"])
        out["win_rate"] = float(out["wins"] / float(denom))
        out["total_net_profit"] = float(pnl.sum())
        out["gross_profit"] = float(pnl[pnl > 0.0].sum())
        out["gross_loss"] = float(pnl[pnl < 0.0].sum())
        out["profit_trades_pct"] = float(out["profit_trades_count"] / float(out["closed_trades"]))
        out["loss_trades_pct"] = float(out["loss_trades_count"] / float(out["closed_trades"]))
        out["profit_trades_text"] = f"{out['profit_trades_count']} ({out['profit_trades_pct']:.2%})"
        out["loss_trades_text"] = f"{out['loss_trades_count']} ({out['loss_trades_pct']:.2%})"
        profit_only = pnl[pnl > 0.0]
        loss_only = pnl[pnl < 0.0]
        out["avg_profit_trade"] = float(profit_only.mean()) if not profit_only.empty else 0.0
        out["avg_loss_trade"] = float(loss_only.mean()) if not loss_only.empty else 0.0
        if out["avg_loss_trade"] < 0.0:
            out["payoff_ratio"] = float(out["avg_profit_trade"] / abs(out["avg_loss_trade"]))
        out["expected_payoff"] = self._safe_div(out["total_net_profit"], len(closed), 0.0)

        if out["gross_loss"] < 0.0:
            out["profit_factor"] = float(out["gross_profit"] / abs(out["gross_loss"]))
        elif out["gross_profit"] > 0.0:
            out["profit_factor"] = float("inf")

        trade_returns = pd.to_numeric(closed.get("realized_return_pct"), errors="coerce")
        trade_returns = trade_returns.replace([np.inf, -np.inf], np.nan).dropna()
        out["avg_trade_return_pct"] = self._safe_float(trade_returns.mean(), 0.0)
        out["best_trade_pnl"] = self._safe_float(pnl.max(), 0.0)
        out["worst_trade_pnl"] = self._safe_float(pnl.min(), 0.0)

        decisive = closed[pnl != 0.0]
        if not decisive.empty:
            outcomes = list(
                (pd.to_numeric(decisive["realized_pnl"], errors="coerce") > 0.0).to_numpy()
            )
            out["z_score"] = float(self._runs_test_zscore(outcomes))
            streaks = self._streak_groups(outcomes)
            win_lengths = [length for flag, length in streaks if flag]
            loss_lengths = [length for flag, length in streaks if not flag]
            out["win_streak_max"] = int(max(win_lengths) if win_lengths else 0)
            out["loss_streak_max"] = int(max(loss_lengths) if loss_lengths else 0)
            out["win_streak_avg"] = float(np.mean(win_lengths)) if win_lengths else 0.0
            out["loss_streak_avg"] = float(np.mean(loss_lengths)) if loss_lengths else 0.0

        if "close_side" in closed.columns:
            long_closed = closed[closed["close_side"] == "LONG"]
            short_closed = closed[closed["close_side"] == "SHORT"]
            out["long_trades"] = len(long_closed)
            out["short_trades"] = len(short_closed)
            if len(long_closed) > 0:
                out["long_win_rate"] = float((long_closed["realized_pnl"] > 0.0).mean())
            if len(short_closed) > 0:
                out["short_win_rate"] = float((short_closed["realized_pnl"] > 0.0).mean())
            out["long_trades_win_pct"] = f"{out['long_trades']} ({out['long_win_rate']:.2%})"
            out["short_trades_win_pct"] = f"{out['short_trades']} ({out['short_win_rate']:.2%})"

        if "holding_sec" in closed.columns:
            hold = pd.to_numeric(closed["holding_sec"], errors="coerce")
            hold = hold.replace([np.inf, -np.inf], np.nan).dropna()
            hold = hold[hold >= 0.0]
            if not hold.empty:
                out["holding_time_min_sec"] = float(hold.min())
                out["holding_time_avg_sec"] = float(hold.mean())
                out["holding_time_max_sec"] = float(hold.max())

        initial_equity = 0.0
        if not equity_df.empty and "total" in equity_df.columns:
            initial_equity = self._safe_float(equity_df["total"].iloc[0], 0.0)
            out["initial_equity"] = float(initial_equity)
            out["final_equity"] = self._safe_float(equity_df["total"].iloc[-1], 0.0)
            totals = (
                pd.to_numeric(equity_df["total"], errors="coerce")
                .dropna()
                .to_numpy(dtype=np.float64)
            )
            if totals.size > 0:
                eq_dd = self._drawdown_stats(totals, initial_equity)
                out["equity_drawdown_absolute"] = float(eq_dd["absolute"])
                out["equity_drawdown_maximal"] = float(eq_dd["maximal"])
                out["equity_drawdown_relative_pct"] = float(eq_dd["relative_pct"])
        if initial_equity > 0.0:
            balance_curve = np.array(
                [
                    initial_equity,
                    *(initial_equity + pnl.cumsum()).tolist(),
                ],
                dtype=np.float64,
            )
            dd_stats = self._drawdown_stats(balance_curve, initial_equity)
            out["balance_drawdown_absolute"] = float(dd_stats["absolute"])
            out["balance_drawdown_maximal"] = float(dd_stats["maximal"])
            out["balance_drawdown_relative_pct"] = float(dd_stats["relative_pct"])
            out["recovery_factor"] = self._safe_div(
                out["total_net_profit"], out["balance_drawdown_maximal"], 0.0
            )

        out["total_c_plus_o"] = float(out["final_equity"] - out["initial_equity"])
        out["open_pnl"] = float(out["total_c_plus_o"] - out["total_net_profit"])
        out["r_mdd"] = self._safe_div(out["total_c_plus_o"], out["equity_drawdown_maximal"], 0.0)

        return out

    def _build_monthly_returns(self, equity_df):
        if equity_df.empty or "datetime" not in equity_df.columns:
            return pd.DataFrame()

        frame = equity_df[["datetime", "returns"]].copy()
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame["returns"] = pd.to_numeric(frame["returns"], errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["datetime", "returns"])
        if frame.empty:
            return pd.DataFrame()

        frame["year"] = frame["datetime"].dt.year
        frame["month"] = frame["datetime"].dt.month
        monthly = (
            frame.groupby(["year", "month"], observed=False)["returns"]
            .apply(lambda s: float(np.prod(1.0 + s.to_numpy(dtype=np.float64)) - 1.0))
            .unstack("month")
        )
        if monthly.empty:
            return monthly
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

    def _serialize_monthly_returns(self, monthly_df):
        if monthly_df is None or monthly_df.empty:
            return {}
        payload = {}
        for year in monthly_df.index:
            payload[str(year)] = {}
            for month in monthly_df.columns:
                value = monthly_df.loc[year, month]
                if pd.isna(value):
                    payload[str(year)][str(month)] = None
                else:
                    payload[str(year)][str(month)] = float(value)
        return payload

    def _build_mt5_rows(self, performance, trade_stats):
        pf_value = trade_stats.get("profit_factor", 0.0)
        profit_factor = "inf" if math.isinf(pf_value) else self._format_num(pf_value, 3)
        return [
            {
                "Section": "Profitability",
                "Metric": "Total Net Profit",
                "Value": self._format_num(trade_stats.get("total_net_profit"), 4),
            },
            {
                "Section": "Profitability",
                "Metric": "Open P/L",
                "Value": self._format_num(trade_stats.get("open_pnl"), 4),
            },
            {
                "Section": "Profitability",
                "Metric": "Total (C+O)",
                "Value": self._format_num(trade_stats.get("total_c_plus_o"), 4),
            },
            {
                "Section": "Profitability",
                "Metric": "Gross Profit",
                "Value": self._format_num(trade_stats.get("gross_profit"), 4),
            },
            {
                "Section": "Profitability",
                "Metric": "Gross Loss",
                "Value": self._format_num(trade_stats.get("gross_loss"), 4),
            },
            {
                "Section": "Profitability",
                "Metric": "Profit Factor",
                "Value": profit_factor,
            },
            {
                "Section": "Profitability",
                "Metric": "Expected Payoff",
                "Value": self._format_num(trade_stats.get("expected_payoff"), 4),
            },
            {
                "Section": "Profitability",
                "Metric": "Recovery Factor",
                "Value": self._format_num(trade_stats.get("recovery_factor"), 4),
            },
            {
                "Section": "Profitability",
                "Metric": "R/MDD",
                "Value": self._format_num(trade_stats.get("r_mdd"), 4),
            },
            {
                "Section": "Direction",
                "Metric": "Long Trades (Win %)",
                "Value": str(trade_stats.get("long_trades_win_pct", "0 (0.00%)")),
            },
            {
                "Section": "Direction",
                "Metric": "Short Trades (Win %)",
                "Value": str(trade_stats.get("short_trades_win_pct", "0 (0.00%)")),
            },
            {
                "Section": "Direction",
                "Metric": "Profit Trades (% of total)",
                "Value": str(trade_stats.get("profit_trades_text", "0 (0.00%)")),
            },
            {
                "Section": "Direction",
                "Metric": "Loss Trades (% of total)",
                "Value": str(trade_stats.get("loss_trades_text", "0 (0.00%)")),
            },
            {
                "Section": "Direction",
                "Metric": "Avg Profit Trade",
                "Value": self._format_num(trade_stats.get("avg_profit_trade"), 4),
            },
            {
                "Section": "Direction",
                "Metric": "Avg Loss Trade",
                "Value": self._format_num(trade_stats.get("avg_loss_trade"), 4),
            },
            {
                "Section": "Direction",
                "Metric": "Payoff Ratio",
                "Value": self._format_num(trade_stats.get("payoff_ratio"), 4),
            },
            {
                "Section": "Streak",
                "Metric": "Max Win/Loss Streak",
                "Value": (
                    f"{int(trade_stats.get('win_streak_max', 0))} / "
                    f"{int(trade_stats.get('loss_streak_max', 0))}"
                ),
            },
            {
                "Section": "Streak",
                "Metric": "Avg Win/Loss Streak",
                "Value": (
                    f"{self._format_num(trade_stats.get('win_streak_avg', 0.0), 2)} / "
                    f"{self._format_num(trade_stats.get('loss_streak_avg', 0.0), 2)}"
                ),
            },
            {
                "Section": "Holding",
                "Metric": "Min Holding Time",
                "Value": self._format_duration(trade_stats.get("holding_time_min_sec", 0.0)),
            },
            {
                "Section": "Holding",
                "Metric": "Avg Holding Time",
                "Value": self._format_duration(trade_stats.get("holding_time_avg_sec", 0.0)),
            },
            {
                "Section": "Holding",
                "Metric": "Max Holding Time",
                "Value": self._format_duration(trade_stats.get("holding_time_max_sec", 0.0)),
            },
            {
                "Section": "Risk",
                "Metric": "Max Drawdown",
                "Value": self._format_pct(performance.get("max_drawdown", 0.0)),
            },
            {
                "Section": "Risk",
                "Metric": "Balance DD Relative %",
                "Value": self._format_pct(trade_stats.get("balance_drawdown_relative_pct", 0.0)),
            },
            {
                "Section": "Risk",
                "Metric": "Equity DD Relative %",
                "Value": self._format_pct(trade_stats.get("equity_drawdown_relative_pct", 0.0)),
            },
            {
                "Section": "Risk",
                "Metric": "Z-Score",
                "Value": self._format_num(trade_stats.get("z_score", 0.0), 4),
            },
        ]

    def _build_markdown(self, payload, trades_df, monthly_returns):
        performance = payload.get("performance", {})
        trade_stats = payload.get("trade_analysis", {})
        params = payload.get("strategy_params", {})

        lines = []
        lines.append(f"# Trading Report: {payload.get('report_type', 'Unknown')}")
        lines.append("")
        lines.append(f"- Generated: {payload.get('generated_at', '')}")
        lines.append(f"- Strategy: {payload.get('strategy', 'Unknown')}")
        lines.append(f"- Equity rows: {payload.get('equity_rows', 0)}")
        lines.append(f"- Trade rows: {payload.get('trade_rows', 0)}")
        lines.append("")

        mirror = payload.get("mirror_snapshot", {})
        if mirror:
            lines.append("## Mirror Snapshot")
            lines.append("| Metric | Value |")
            lines.append("| :--- | :--- |")
            lines.append(
                f"| Total Trades | {int(mirror.get('total_trades', 0)):,} ({int(mirror.get('wins', 0))}W / {int(mirror.get('losses', 0))}L) |"
            )
            lines.append(f"| Win Rate | {self._format_pct(mirror.get('win_rate', 0.0))} |")
            lines.append(f"| Closed PnL | {self._format_num(mirror.get('closed_pnl', 0.0), 4)} |")
            lines.append(f"| Open P/L | {self._format_num(mirror.get('open_pnl', 0.0), 4)} |")
            lines.append(
                f"| Total (C+O) | {self._format_num(mirror.get('total_c_plus_o', 0.0), 4)} |"
            )
            lines.append(
                f"| Equity MDD | {self._format_num(mirror.get('equity_mdd', 0.0), 4)} ({self._format_pct(mirror.get('equity_mdd_rel', 0.0))}) |"
            )
            lines.append(f"| R/MDD | {self._format_num(mirror.get('r_mdd', 0.0), 4)}x |")
            lines.append("")

        lines.append("## Strategy Parameters")
        if params:
            lines.append("| Parameter | Value |")
            lines.append("| :--- | :--- |")
            for key, value in params.items():
                lines.append(f"| {key} | {value} |")
        else:
            lines.append("No parameters found.")
        lines.append("")

        lines.append("## Performance Metrics")
        perf_rows = [
            ("Total Return", self._format_pct(performance.get("total_return", 0.0))),
            ("CAGR", self._format_pct(performance.get("cagr", 0.0))),
            ("Sharpe Ratio", self._format_num(performance.get("sharpe_ratio", 0.0), 4)),
            ("Sortino Ratio", self._format_num(performance.get("sortino_ratio", 0.0), 4)),
            ("Calmar Ratio", self._format_num(performance.get("calmar_ratio", 0.0), 4)),
            (
                "Annualized Volatility",
                self._format_pct(performance.get("annualized_volatility", 0.0)),
            ),
            ("Max Drawdown", self._format_pct(performance.get("max_drawdown", 0.0))),
            ("Drawdown Duration", str(int(performance.get("drawdown_duration", 0)))),
        ]
        lines.append("| Metric | Value |")
        lines.append("| :--- | :--- |")
        for key, value in perf_rows:
            lines.append(f"| {key} | {value} |")
        lines.append("")

        lines.append("## MT5-Style Summary")
        lines.append("| Section | Metric | Value |")
        lines.append("| :--- | :--- | :--- |")
        for row in payload.get("mt5_summary", []):
            lines.append(f"| {row['Section']} | {row['Metric']} | {row['Value']} |")
        lines.append("")

        lines.append("## Trade Analysis")
        trade_rows = [
            ("Total Transactions", str(int(trade_stats.get("total_transactions", 0)))),
            ("Buy Orders", str(int(trade_stats.get("buy_orders", 0)))),
            ("Sell Orders", str(int(trade_stats.get("sell_orders", 0)))),
            ("Closed Trades", str(int(trade_stats.get("closed_trades", 0)))),
            (
                "Wins / Losses",
                f"{int(trade_stats.get('wins', 0))} / {int(trade_stats.get('losses', 0))}",
            ),
            ("Win Rate", self._format_pct(trade_stats.get("win_rate", 0.0))),
            ("Open P/L", self._format_num(trade_stats.get("open_pnl", 0.0), 4)),
            ("Total (C+O)", self._format_num(trade_stats.get("total_c_plus_o", 0.0), 4)),
            ("R/MDD", self._format_num(trade_stats.get("r_mdd", 0.0), 4)),
            (
                "Profit/Loss Trades",
                f"{trade_stats.get('profit_trades_text', '0 (0.00%)')} / "
                f"{trade_stats.get('loss_trades_text', '0 (0.00%)')}",
            ),
            (
                "Avg Profit/Loss Trade",
                f"{self._format_num(trade_stats.get('avg_profit_trade', 0.0), 4)} / "
                f"{self._format_num(trade_stats.get('avg_loss_trade', 0.0), 4)}",
            ),
            ("Payoff Ratio", self._format_num(trade_stats.get("payoff_ratio", 0.0), 4)),
            (
                "Avg Trade Return",
                f"{self._safe_float(trade_stats.get('avg_trade_return_pct'), 0.0):.4f}%",
            ),
            ("Best Trade PnL", self._format_num(trade_stats.get("best_trade_pnl", 0.0), 4)),
            ("Worst Trade PnL", self._format_num(trade_stats.get("worst_trade_pnl", 0.0), 4)),
        ]
        lines.append("| Metric | Value |")
        lines.append("| :--- | :--- |")
        for key, value in trade_rows:
            lines.append(f"| {key} | {value} |")
        lines.append("")

        lines.append("## Monthly Returns")
        if monthly_returns is not None and not monthly_returns.empty:
            table = monthly_returns.copy()
            for col in table.columns:
                table[col] = table[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.2%}")
            table.insert(0, "Year", table.index.astype(str))
            lines.append(table.to_markdown(index=False))
        else:
            lines.append("No monthly return data available.")
        lines.append("")

        lines.append("## Recent Trades")
        if trades_df is not None and not trades_df.empty:
            recent = trades_df.copy().tail(12)
            keep_cols = [
                "datetime",
                "symbol",
                "direction",
                "quantity",
                "price",
                "commission",
                "realized_pnl",
                "realized_return_pct",
                "close_side",
            ]
            cols = [col for col in keep_cols if col in recent.columns]
            lines.append(recent[cols].to_markdown(index=False))
        else:
            lines.append("No trades recorded.")

        return "\n".join(lines)


if __name__ == "__main__":
    generator = ReportGenerator()
    print(f"Generated: {generator.generate_report('Backtest', 'RsiStrategy')}")
