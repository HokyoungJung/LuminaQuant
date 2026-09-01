"""Portfolio role services for sizing and performance reporting."""

from __future__ import annotations

import math

import numpy as np
from lumina_quant.optimization.native_backend import evaluate_metrics_backend
from lumina_quant.utils.risk_free import (
    resolve_risk_free_config,
    sharpe_ratio as compute_sharpe_ratio,
    sortino_ratio as compute_sortino_ratio,
)


class PortfolioSizingService:
    """Stateless sizing/validation helpers for Portfolio."""

    LEGACY_NOTIONAL_CAP_MODES = frozenset({"", "legacy_notional_cap", "notional_cap"})
    NOTIONAL_FRACTION_MODES = frozenset({"notional_fraction", "notional_target_fraction"})
    ISOLATED_MARGIN_FRACTION_MODES = frozenset(
        {"isolated_margin_fraction", "margin_fraction", "isolated_margin"}
    )

    @staticmethod
    def normalize_target_allocation_mode(mode: str | None) -> str:
        token = str(mode or "legacy_notional_cap").strip().lower()
        if token in PortfolioSizingService.LEGACY_NOTIONAL_CAP_MODES:
            return "legacy_notional_cap"
        if token in PortfolioSizingService.NOTIONAL_FRACTION_MODES:
            return "notional_fraction"
        if token in PortfolioSizingService.ISOLATED_MARGIN_FRACTION_MODES:
            return "isolated_margin_fraction"
        raise ValueError(f"Unsupported target allocation mode: {mode}")

    @staticmethod
    def target_notional_fraction(
        *,
        target_allocation: float,
        leverage: float,
        target_allocation_mode: str | None,
    ) -> float:
        """Return intended notional/equity for an explicit sizing contract."""
        allocation = max(0.0, float(target_allocation))
        mode = PortfolioSizingService.normalize_target_allocation_mode(target_allocation_mode)
        if mode == "isolated_margin_fraction":
            return allocation * max(float(leverage), 0.0)
        return allocation

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
        target_allocation_mode: str = "legacy_notional_cap",
        leverage: float = 1.0,
        max_order_notional_pct: float = 0.0,
        allow_metadata_risk_override: bool = False,
        max_leverage: float = 0.0,
    ) -> float:
        # Config ceilings captured before any signal-metadata override is applied.
        # When allow_metadata_risk_override is False (the SAFE default) overrides may
        # only LOWER these ceilings; any override that would RAISE a risk cap above its
        # configured value is clamped back to the ceiling.
        config_max_symbol_exposure_pct = float(max_symbol_exposure_pct)
        config_max_order_value = float(max_order_value)
        config_max_order_notional_pct = float(max_order_notional_pct)
        leverage_ceiling = float(max_leverage) if float(max_leverage) > 0.0 else float(leverage)

        metadata = dict(getattr(signal, "metadata", None) or {})
        override_target_allocation = metadata.get("target_allocation")
        if (
            override_target_allocation is None
            and metadata.get("target_allocation_scale") is not None
        ):
            override_target_allocation = float(target_allocation) * float(
                metadata.get("target_allocation_scale")
            )
        if override_target_allocation is not None:
            # target_allocation override may remain unclamped here: it is transitively
            # bounded by the now-clamped exposure cap applied below (lines ~150-160).
            target_allocation = max(0.0, float(override_target_allocation))

        override_target_mode = metadata.get("target_allocation_mode", metadata.get("sizing_mode"))
        if override_target_mode is not None:
            target_allocation_mode = str(override_target_mode)
        target_allocation_mode = PortfolioSizingService.normalize_target_allocation_mode(
            target_allocation_mode
        )

        override_leverage = metadata.get("leverage")
        if override_leverage is not None:
            leverage = max(0.0, float(override_leverage))
            if not allow_metadata_risk_override:
                # leverage override may LOWER but not RAISE the configured ceiling.
                leverage = min(leverage, leverage_ceiling)

        override_max_symbol_exposure = metadata.get("max_symbol_exposure_pct")
        if override_max_symbol_exposure is not None:
            max_symbol_exposure_pct = max(0.0, float(override_max_symbol_exposure))
            if not allow_metadata_risk_override:
                max_symbol_exposure_pct = min(
                    max_symbol_exposure_pct, config_max_symbol_exposure_pct
                )

        override_max_order_notional_pct = metadata.get("max_order_notional_pct")
        if override_max_order_notional_pct is not None:
            max_order_notional_pct = max(0.0, float(override_max_order_notional_pct))
            # Clamp only when a cap is configured (> 0). 0.0 means "no cap configured";
            # leave the override since max_order_value + exposure caps still bind.
            if not allow_metadata_risk_override and config_max_order_notional_pct > 0.0:
                max_order_notional_pct = min(max_order_notional_pct, config_max_order_notional_pct)

        override_max_order_value = metadata.get("max_order_value")
        if override_max_order_value is None and metadata.get("max_order_value_scale") is not None:
            override_max_order_value = float(max_order_value) * float(
                metadata.get("max_order_value_scale")
            )
        if override_max_order_value is not None:
            max_order_value = max(0.0, float(override_max_order_value))
            # Clamp only when a fixed per-order cap is configured (> 0).
            if not allow_metadata_risk_override and config_max_order_value > 0.0:
                max_order_value = min(max_order_value, config_max_order_value)

        if float(current_price) <= 0 or float(equity) <= 0:
            return 0.0

        target_notional_fraction = PortfolioSizingService.target_notional_fraction(
            target_allocation=float(target_allocation),
            leverage=float(leverage),
            target_allocation_mode=target_allocation_mode,
        )

        if target_allocation_mode == "legacy_notional_cap":
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
        else:
            quantity = (float(equity) * float(target_notional_fraction)) / float(current_price)

        exposure_cap_pct = float(max_symbol_exposure_pct)
        if target_allocation_mode == "legacy_notional_cap" and float(target_notional_fraction) > 0:
            exposure_cap_pct = min(exposure_cap_pct, float(target_notional_fraction))
        notional_cap = float(equity) * exposure_cap_pct
        if notional_cap > 0:
            quantity = min(quantity, notional_cap / float(current_price))

        max_order_notional_cap = float(equity) * float(max_order_notional_pct)
        if max_order_notional_cap > 0:
            quantity = min(quantity, max_order_notional_cap / float(current_price))

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

        if len(returns) == len(total_series) and len(returns) > 1:
            returns = returns[1:]
        if len(benchmark_returns) == len(total_series) and len(benchmark_returns) > 1:
            benchmark_returns = benchmark_returns[1:]

        from lumina_quant.utils.performance import PerformanceMetrics

        periods = getattr(config, "ANNUAL_PERIODS", 252)
        timestamps = None
        for column in ("datetime", "time", "timestamp"):
            if column in getattr(equity_curve, "columns", []):
                try:
                    timestamps = equity_curve[column].to_numpy()
                except Exception:
                    timestamps = None
                break
        resolved_rf = resolve_risk_free_config(
            config,
            periods_per_year=int(periods),
            timestamps=timestamps,
            size=len(returns),
        )

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
            compute_sharpe_ratio(
                returns,
                periods_per_year=int(periods),
                risk_free_per_period=np.asarray(resolved_rf.periodic_rates, dtype=float),
            )
        )
        sortino_ratio = PortfolioPerformanceService._safe_scalar(
            compute_sortino_ratio(
                returns,
                periods_per_year=int(periods),
                target_per_period=np.asarray(resolved_rf.periodic_sortino_targets, dtype=float),
            )
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
            ("Risk-Free Reference", f"{resolved_rf.mode}:{resolved_rf.tenor}"),
            ("Risk-Free Annual", "%0.2f%%" % (float(resolved_rf.annual_rate) * 100.0)),
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
