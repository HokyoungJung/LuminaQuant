"""Portfolio role services for sizing and performance reporting."""

from __future__ import annotations

import math

import numpy as np
from lumina_quant.optimization.native_backend import evaluate_metrics_backend


class PortfolioSizingService:
    """Stateless sizing/validation helpers for Portfolio."""

    @staticmethod
    def round_quantity(quantity: float, step: float) -> float:
        if step <= 0:
            return float(quantity)
        return math.floor(float(quantity) / float(step)) * float(step)

    @staticmethod
    def risk_based_quantity(
        *,
        signal,
        current_price: float,
        equity: float,
        risk_per_trade: float,
        default_stop_loss_pct: float,
        max_symbol_exposure_pct: float,
        target_allocation: float,
        max_order_value: float,
    ) -> float:
        risk_amount = max(float(equity) * float(risk_per_trade), 0.0)
        if risk_amount <= 0:
            return 0.0

        if signal.stop_loss is not None:
            stop_price = float(signal.stop_loss)
        else:
            if signal.signal_type == "LONG":
                stop_price = float(current_price) * (1.0 - float(default_stop_loss_pct))
            else:
                stop_price = float(current_price) * (1.0 + float(default_stop_loss_pct))

        stop_distance = abs(float(current_price) - stop_price)
        if stop_distance <= 0:
            stop_distance = float(current_price) * float(default_stop_loss_pct)
        if stop_distance <= 0:
            return 0.0

        quantity = risk_amount / stop_distance

        exposure_cap_pct = float(max_symbol_exposure_pct)
        if float(target_allocation) > 0:
            exposure_cap_pct = min(exposure_cap_pct, float(target_allocation))
        notional_cap = float(equity) * exposure_cap_pct
        if notional_cap > 0:
            quantity = min(quantity, notional_cap / float(current_price))

        if float(max_order_value) > 0:
            quantity = min(quantity, float(max_order_value) / float(current_price))

        return max(float(quantity), 0.0)

    @staticmethod
    def validate_and_round_quantity(
        *,
        quantity: float,
        price: float,
        min_qty: float,
        qty_step: float,
        min_notional: float,
    ) -> float:
        qty = PortfolioSizingService.round_quantity(float(quantity), float(qty_step))
        if qty < float(min_qty):
            return 0.0
        if qty * float(price) < float(min_notional):
            return 0.0
        return qty


class PortfolioPerformanceService:
    """Performance/statistics computations extracted from Portfolio."""

    @staticmethod
    def _safe_scalar(value, default: float = 0.0) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        if not np.isfinite(out):
            return float(default)
        return out

    @staticmethod
    def build_summary_stats(
        *, equity_curve, config, total_funding_paid: float, liquidation_count: int
    ) -> list[tuple[str, str]]:
        total_series = equity_curve["total"].to_numpy()
        returns = equity_curve["returns"].fill_null(0.0).to_numpy()
        benchmark_returns = equity_curve["benchmark_returns"].fill_null(0.0).to_numpy()

        if len(total_series) < 2:
            return [("Status", "Not enough data")]

        from lumina_quant.utils.performance import PerformanceMetrics

        periods = getattr(config, "ANNUAL_PERIODS", 252)

        total_return = (total_series[-1] - total_series[0]) / total_series[0]

        benchmark_prices = equity_curve["benchmark_price"].fill_null(0.0).to_numpy()
        positive_idx = np.flatnonzero(benchmark_prices > 0.0)
        if positive_idx.size > 0:
            first_price = PortfolioPerformanceService._safe_scalar(
                benchmark_prices[int(positive_idx[0])],
                1.0,
            )
        else:
            first_price = 1.0
        if first_price <= 0.0:
            first_price = 1.0
        last_price = PortfolioPerformanceService._safe_scalar(benchmark_prices[-1], first_price)
        benchmark_unrealized = (last_price - first_price) / first_price

        cagr = PortfolioPerformanceService._safe_scalar(
            PerformanceMetrics.cagr(total_series[-1], total_series[0], len(total_series), periods)
        )
        volatility = PortfolioPerformanceService._safe_scalar(
            PerformanceMetrics.annualized_volatility(returns, periods)
        )
        sharpe_ratio = PortfolioPerformanceService._safe_scalar(
            PerformanceMetrics.sharpe_ratio(returns, periods=periods)
        )
        sortino_ratio = PortfolioPerformanceService._safe_scalar(
            PerformanceMetrics.sortino_ratio(returns, periods=periods)
        )

        drawdown, max_dd_duration = PerformanceMetrics.drawdowns(total_series)
        max_dd = PortfolioPerformanceService._safe_scalar(max(drawdown), 0.0)
        calmar_ratio = PortfolioPerformanceService._safe_scalar(
            PerformanceMetrics.calmar_ratio(cagr, max_dd)
        )
        alpha, beta = PerformanceMetrics.alpha_beta(returns, benchmark_returns, periods=periods)
        alpha = PortfolioPerformanceService._safe_scalar(alpha)
        beta = PortfolioPerformanceService._safe_scalar(beta)
        info_ratio = PortfolioPerformanceService._safe_scalar(
            PerformanceMetrics.information_ratio(returns, benchmark_returns)
        )

        winning_days = len(returns[returns > 0])
        total_days = len(returns) - 1
        win_rate = winning_days / total_days if total_days > 0 else 0.0

        return [
            ("Total Return", "%0.2f%%" % (total_return * 100.0)),
            ("Benchmark Return", "%0.2f%%" % (benchmark_unrealized * 100.0)),
            ("CAGR", "%0.2f%%" % (cagr * 100.0)),
            ("Ann. Volatility", "%0.2f%%" % (volatility * 100.0)),
            ("Sharpe Ratio", "%0.4f" % sharpe_ratio),
            ("Sortino Ratio", "%0.4f" % sortino_ratio),
            ("Calmar Ratio", "%0.4f" % calmar_ratio),
            ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
            ("DD Duration", "%d bars" % max_dd_duration),
            ("Alpha", "%0.4f" % alpha),
            ("Beta", "%0.4f" % beta),
            ("Information Ratio", "%0.4f" % info_ratio),
            ("Daily Win Rate", "%0.2f%%" % (win_rate * 100.0)),
            ("Funding (Net)", "%0.4f" % float(total_funding_paid)),
            ("Liquidations", "%d" % int(liquidation_count)),
        ]

    @staticmethod
    def build_fast_stats(*, metric_totals, config) -> dict[str, float | str]:
        total_series = np.asarray(metric_totals, dtype=np.float64)
        if total_series.size < 2:
            return {
                "status": "not_enough_data",
                "sharpe": -999.0,
                "cagr": 0.0,
                "max_drawdown": 0.0,
            }

        periods = int(getattr(config, "ANNUAL_PERIODS", 252))
        sharpe_ratio, cagr, max_dd = evaluate_metrics_backend(total_series, periods)
        sharpe_ratio = float(sharpe_ratio)
        cagr = float(cagr)
        max_dd = float(max_dd)

        if not np.isfinite(sharpe_ratio):
            sharpe_ratio = -999.0
        if not np.isfinite(cagr):
            cagr = 0.0
        if not np.isfinite(max_dd):
            max_dd = 0.0

        return {
            "status": "ok",
            "sharpe": sharpe_ratio,
            "cagr": cagr,
            "max_drawdown": max_dd,
        }
