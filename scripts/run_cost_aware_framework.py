"""Run cost-aware, timeframe-tunable strategy experiments."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from lumina_quant.backtesting.cost_models import (
    CostModelParams,
    ExecutionPolicy,
    simulate_market_fill,
)
from lumina_quant.backtesting.liquidity_metrics import compute_liquidity_metrics
from lumina_quant.backtesting.timeframe_panel import (
    build_timeframe_panel,
    build_timeframe_panel_from_frames,
)
from lumina_quant.calibration.cost_calibration import calibrate_impact_coefficients
from lumina_quant.configuration.experiment_loader import load_experiment_config
from lumina_quant.eval.cost_aware_reports import compute_perf_metrics, write_report_bundle
from lumina_quant.market_data import normalize_timeframe_token, timeframe_to_milliseconds
from lumina_quant.portfolio.cost_aware_constructor import (
    ConstructorParams,
    CostAwarePortfolioConstructor,
)

# Ensure built-ins register themselves.
from lumina_quant.strategies import plugins as _plugins  # noqa: F401
from lumina_quant.strategies.plugin_interface import get_plugin


def _coerce_datetime(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC).replace(tzinfo=None)


def _bars_per_year(timeframe: str) -> int:
    tf_ms = max(1, int(timeframe_to_milliseconds(timeframe)))
    return max(1, int((365 * 24 * 60 * 60 * 1000) / tf_ms))


def _generate_synthetic_base_frames(
    assets: list[str],
    start_date: str,
    bars: int,
    seed: int,
) -> dict[str, pl.DataFrame]:
    rng = random.Random(seed)
    start = _coerce_datetime(start_date)
    output: dict[str, pl.DataFrame] = {}
    for idx, asset in enumerate(assets):
        ts_list: list[datetime] = []
        open_list: list[float] = []
        high_list: list[float] = []
        low_list: list[float] = []
        close_list: list[float] = []
        vol_list: list[float] = []

        price = 100.0 + (idx * 8.0)
        for step in range(int(bars)):
            ts = start + timedelta(seconds=step)
            drift = 0.00012 * (idx + 1)
            shock = rng.uniform(-0.0005, 0.0005)
            open_px = max(1.0, price)
            close_px = max(1.0, open_px * (1.0 + drift + shock))
            wiggle = abs(rng.uniform(0.0001, 0.0015))
            high_px = max(open_px, close_px) * (1.0 + wiggle)
            low_px = min(open_px, close_px) * (1.0 - wiggle)
            volume = 1500.0 + (idx * 120.0) + rng.uniform(-80.0, 80.0)

            ts_list.append(ts)
            open_list.append(open_px)
            high_list.append(high_px)
            low_list.append(low_px)
            close_list.append(close_px)
            vol_list.append(max(1.0, volume))
            price = close_px

        output[asset] = pl.DataFrame(
            {
                "datetime": ts_list,
                "open": open_list,
                "high": high_list,
                "low": low_list,
                "close": close_list,
                "volume": vol_list,
            }
        )
    return output


def _frame_to_lookup(frame: pl.DataFrame) -> dict[datetime, dict[str, dict[str, float]]]:
    lookup: dict[datetime, dict[str, dict[str, float]]] = {}
    for row in frame.iter_rows(named=True):
        ts = row["datetime"]
        asset = str(row["asset"])
        bucket = lookup.setdefault(ts, {})
        bucket[asset] = {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "adv": float(row.get("adv", row["volume"])),
            "adtv": float(row.get("adtv", row["close"] * row["volume"])),
            "sigma": float(row.get("sigma", 0.0)),
            "mid": float(row.get("mid", (row["high"] + row["low"]) * 0.5)),
        }
    return lookup


def _targets_lookup(targets: pl.DataFrame) -> dict[datetime, dict[str, float]]:
    output: dict[datetime, dict[str, float]] = defaultdict(dict)
    for row in targets.select(["datetime", "asset", "target_weight"]).iter_rows(named=True):
        output[row["datetime"]][str(row["asset"])] = float(row["target_weight"])
    return output


def _run_single_backtest(
    *,
    panel: pl.DataFrame,
    strategy_block: dict[str, Any],
    timeframe: str,
    initial_capital: float,
    cost_global: dict[str, Any],
) -> dict[str, Any]:
    name = str(strategy_block["name"])
    signal_params = dict(strategy_block.get("signal_params") or {})
    plugin = get_plugin(name)

    features = plugin.compute_features(panel, signal_params)
    signal = plugin.compute_signal(features, signal_params)
    targets = plugin.signal_to_targets(signal, signal_params)

    market = _frame_to_lookup(panel)
    target_map = _targets_lookup(targets)
    timestamps = sorted(market.keys())
    assets = sorted(panel["asset"].unique().to_list())

    constructor = CostAwarePortfolioConstructor(
        ConstructorParams(
            no_trade_band_bps=float(strategy_block.get("portfolio_construction", {}).get("no_trade_band_bps", 8.0)),
            turnover_penalty=float(strategy_block.get("portfolio_construction", {}).get("turnover_penalty", 0.0)),
            cost_penalty=float(strategy_block.get("portfolio_construction", {}).get("cost_penalty", 1.0)),
            participation_penalty=float(
                strategy_block.get("portfolio_construction", {}).get("participation_penalty", 0.0)
            ),
        )
    )

    cost_params = CostModelParams(
        spread_bps=float(strategy_block.get("cost_model", {}).get("spread_bps", 4.0)),
        impact_k=float(strategy_block.get("cost_model", {}).get("k_strategy", 1.0)),
        fees_bps=float(cost_global.get("fees_bps", 0.0)),
        tax_bps=float(cost_global.get("tax_bps", 0.0)),
        participation_lambda=float(strategy_block.get("cost_model", {}).get("participation_lambda", 0.0)),
        participation_power=float(strategy_block.get("cost_model", {}).get("participation_power", 1.0)),
    )

    execution_policy = ExecutionPolicy(
        fill_basis=str(strategy_block.get("execution_model", {}).get("fill_basis", "next_open")),
        max_participation=float(strategy_block.get("execution_model", {}).get("max_participation", 0.1)),
        unfilled_policy=str(strategy_block.get("execution_model", {}).get("unfilled_policy", "carry")),
        carry_decay=float(strategy_block.get("execution_model", {}).get("carry_decay", 1.0)),
    )

    positions = dict.fromkeys(assets, 0.0)
    pending = dict.fromkeys(assets, 0.0)
    net_cash = float(initial_capital)
    gross_cash = float(initial_capital)

    net_equity = [float(initial_capital)]
    gross_equity = [float(initial_capital)]
    net_returns: list[float] = []
    gross_returns: list[float] = []

    fills: list[dict[str, Any]] = []
    turnover_sum = 0.0
    participation_values: list[float] = []
    total_notional = 0.0
    cost_totals = {"spread": 0.0, "impact": 0.0, "fees": 0.0, "total": 0.0}

    for idx in range(len(timestamps) - 1):
        ts = timestamps[idx]
        nxt = timestamps[idx + 1]
        now_slice = market.get(ts, {})
        next_slice = market.get(nxt, {})

        if not now_slice or not next_slice:
            continue

        prev_net = net_equity[-1]
        prev_gross = gross_equity[-1]

        current_weights: dict[str, float] = {}
        for asset in assets:
            px = float(now_slice.get(asset, {}).get("close", 0.0))
            value = positions[asset] * px
            current_weights[asset] = 0.0 if prev_net <= 0 else (value / prev_net)

        target_weights = {asset: float(target_map.get(ts, {}).get(asset, 0.0)) for asset in assets}
        # IMPORTANT: construction must use information known at decision time (current bar),
        # while execution simulation is applied on next bar to avoid lookahead.
        prices = {asset: float(now_slice.get(asset, {}).get("close", 0.0)) for asset in assets}
        liquidity = {asset: now_slice.get(asset, {}) for asset in assets}

        orders, _ = constructor.construct_orders(
            target_weights=target_weights,
            current_weights=current_weights,
            prices=prices,
            liquidity=liquidity,
            aum=prev_net,
            cost_params=cost_params,
            max_participation=execution_policy.max_participation,
        )

        bar_turnover_notional = 0.0
        for asset in assets:
            market_state = next_slice.get(asset, {})
            if not market_state:
                continue

            order_notional = float(orders.get(asset, 0.0))
            fill = simulate_market_fill(
                order_notional=order_notional,
                pending_notional=float(pending[asset]),
                next_open=float(market_state.get("open", 0.0)),
                next_mid=float(market_state.get("mid", None)),
                close_price=float(market_state.get("close", 0.0)),
                adtv=float(market_state.get("adtv", 1.0)),
                sigma=float(market_state.get("sigma", 0.0)),
                bar_volume=float(market_state.get("volume", 0.0)),
                params=cost_params,
                policy=execution_policy,
            )
            pending[asset] = fill.next_pending_notional

            if abs(fill.executed_notional) <= 0.0:
                continue

            quantity = fill.executed_notional / max(1e-9, fill.basis_price)
            positions[asset] += quantity
            net_cash -= quantity * fill.fill_price
            gross_cash -= quantity * fill.basis_price

            traded_notional = abs(fill.executed_notional)
            bar_turnover_notional += traded_notional
            total_notional += traded_notional
            participation_values.append(fill.participation)

            spread_cost = traded_notional * (fill.spread_bps / 10_000.0)
            impact_cost = traded_notional * (fill.impact_bps / 10_000.0)
            fees_cost = traded_notional * (fill.fees_bps / 10_000.0)
            total_cost = traded_notional * (fill.total_bps / 10_000.0)
            cost_totals["spread"] += spread_cost
            cost_totals["impact"] += impact_cost
            cost_totals["fees"] += fees_cost
            cost_totals["total"] += total_cost

            fills.append(
                {
                    "strategy": name,
                    "datetime": nxt.isoformat(),
                    "asset": asset,
                    "requested_notional": fill.requested_notional,
                    "executed_notional": fill.executed_notional,
                    "unfilled_notional": fill.unfilled_notional,
                    "basis_price": fill.basis_price,
                    "fill_price": fill.fill_price,
                    "participation": fill.participation,
                    "spread_bps": fill.spread_bps,
                    "impact_bps": fill.impact_bps,
                    "fees_bps": fill.fees_bps,
                    "total_bps": fill.total_bps,
                    "realized_slippage_bps": fill.total_bps,
                    "realized_impact_bps": fill.impact_bps,
                }
            )

        net_value = net_cash + sum(positions[asset] * float(next_slice[asset]["close"]) for asset in assets)
        gross_value = gross_cash + sum(positions[asset] * float(next_slice[asset]["close"]) for asset in assets)

        net_ret = (net_value / prev_net) - 1.0 if prev_net > 0 else 0.0
        gross_ret = (gross_value / prev_gross) - 1.0 if prev_gross > 0 else 0.0

        net_returns.append(net_ret)
        gross_returns.append(gross_ret)
        net_equity.append(net_value)
        gross_equity.append(gross_value)
        turnover_sum += bar_turnover_notional / max(1.0, prev_net)

    periods = _bars_per_year(timeframe)
    pre_metrics = compute_perf_metrics(gross_returns, gross_equity, periods)
    post_metrics = compute_perf_metrics(net_returns, net_equity, periods)

    participation_values = participation_values or [0.0]
    participation_stats = {
        "p50": float(pl.Series(participation_values).quantile(0.50, interpolation="nearest")),
        "p90": float(pl.Series(participation_values).quantile(0.90, interpolation="nearest")),
        "p99": float(pl.Series(participation_values).quantile(0.99, interpolation="nearest")),
    }

    avg_cost_bps = 0.0 if total_notional <= 0 else (cost_totals["total"] / total_notional) * 10_000.0
    sensitivity = {
        "x1.5": {
            "post_cost_sharpe": float(post_metrics["sharpe"] - (avg_cost_bps * 0.5 / 100.0)),
            "post_cost_total_return": float(post_metrics["total_return"] - (avg_cost_bps * 0.5 / 10_000.0)),
        },
        "x2.0": {
            "post_cost_sharpe": float(post_metrics["sharpe"] - (avg_cost_bps / 100.0)),
            "post_cost_total_return": float(post_metrics["total_return"] - (avg_cost_bps / 10_000.0)),
        },
    }

    mean_participation = sum(participation_values) / len(participation_values)
    capacity_decay = []
    for scale in (0.5, 1.0, 1.5, 2.0):
        capacity_decay.append(
            {
                "aum_scale": scale,
                "estimated_post_cost_sharpe": float(
                    post_metrics["sharpe"] / (1.0 + max(0.0, mean_participation) * max(0.0, scale - 1.0) * 5.0)
                ),
            }
        )

    return {
        "strategy": name,
        "timeframe": timeframe,
        "fills": fills,
        "pre_cost": pre_metrics,
        "post_cost": post_metrics,
        "turnover": float(turnover_sum),
        "trade_count": len(fills),
        "participation": participation_stats,
        "costs": {
            "spread": cost_totals["spread"],
            "impact": cost_totals["impact"],
            "fees": cost_totals["fees"],
            "total": cost_totals["total"],
            "avg_total_bps": avg_cost_bps,
        },
        "sensitivity": sensitivity,
        "capacity_decay": capacity_decay,
    }


def _build_summary(
    run_id: str,
    experiment_id: str,
    results: list[dict[str, Any]],
    calibrations: dict[str, dict[str, float]],
    *,
    ranking_objective: str,
    ranking_weights: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    objective = str(ranking_objective or "composite").strip().lower()
    if objective not in {"composite", "sharpe", "total_return", "drawdown"}:
        objective = "composite"

    default_weights = {
        "sharpe": 1.0,
        "total_return": 0.25,
        "drawdown": 0.5,
        "stability": 0.25,
    }
    weights = dict(default_weights)
    for key in tuple(default_weights.keys()):
        if key in ranking_weights:
            try:
                weights[key] = float(ranking_weights.get(key, default_weights[key]))
            except Exception:
                continue

    def _post_cost_score(metrics: dict[str, Any]) -> float:
        sharpe = float(metrics.get("sharpe", 0.0))
        total_return = float(metrics.get("total_return", 0.0))
        max_drawdown = abs(float(metrics.get("max_drawdown", 0.0)))
        volatility = max(0.0, float(metrics.get("volatility", 0.0)))
        stability = 1.0 / (1.0 + volatility)

        if objective == "sharpe":
            return sharpe
        if objective == "total_return":
            return total_return
        if objective == "drawdown":
            return -max_drawdown
        return (
            (weights["sharpe"] * sharpe)
            + (weights["total_return"] * total_return * 100.0)
            - (weights["drawdown"] * max_drawdown * 100.0)
            + (weights["stability"] * stability)
        )

    ranking: dict[str, list[dict[str, Any]]] = defaultdict(list)
    table_rows: list[dict[str, Any]] = []

    for item in results:
        post_cost_score = float(_post_cost_score(item["post_cost"]))
        ranking[item["strategy"]].append(
            {
                "timeframe": item["timeframe"],
                "post_cost_sharpe": item["post_cost"]["sharpe"],
                "post_cost_total_return": item["post_cost"]["total_return"],
                "post_cost_max_drawdown": item["post_cost"]["max_drawdown"],
                "post_cost_volatility": item["post_cost"]["volatility"],
                "post_cost_score": post_cost_score,
            }
        )
        table_rows.append(
            {
                "strategy": item["strategy"],
                "timeframe": item["timeframe"],
                "pre_sharpe": item["pre_cost"]["sharpe"],
                "post_sharpe": item["post_cost"]["sharpe"],
                "pre_total_return": item["pre_cost"]["total_return"],
                "post_total_return": item["post_cost"]["total_return"],
                "post_max_drawdown": item["post_cost"]["max_drawdown"],
                "post_volatility": item["post_cost"]["volatility"],
                "post_cost_score": post_cost_score,
                "turnover": item["turnover"],
                "trade_count": item["trade_count"],
                "participation_p90": item["participation"]["p90"],
                "spread_cost": item["costs"]["spread"],
                "impact_cost": item["costs"]["impact"],
                "fees_cost": item["costs"]["fees"],
                "total_cost": item["costs"]["total"],
                "sensitivity_x1_5_sharpe": item["sensitivity"]["x1.5"]["post_cost_sharpe"],
                "sensitivity_x2_sharpe": item["sensitivity"]["x2.0"]["post_cost_sharpe"],
            }
        )

    timeframe_ranking: dict[str, list[dict[str, Any]]] = {}
    best_timeframe: dict[str, str] = {}
    for strategy, ranks in ranking.items():
        sorted_rows = sorted(ranks, key=lambda row: float(row["post_cost_score"]), reverse=True)
        timeframe_ranking[strategy] = sorted_rows
        best_timeframe[strategy] = sorted_rows[0]["timeframe"] if sorted_rows else ""

    summary = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "results": results,
        "post_cost_timeframe_ranking": timeframe_ranking,
        "best_timeframe": best_timeframe,
        "calibration": calibrations,
        "ranking_objective": {
            "name": objective,
            "weights": weights,
        },
    }
    return summary, table_rows


def run_cost_aware_framework(
    *,
    experiment_path: str,
    output_root: str = "reports",
    run_id: str | None = None,
) -> Path:
    config, resolved = load_experiment_config(experiment_path)

    seed = int(config.run_controls.seed)
    random.seed(seed)

    if config.run_controls.use_synthetic_data:
        base_frames = _generate_synthetic_base_frames(
            assets=list(config.universe.assets),
            start_date=config.run_controls.start_date,
            bars=int(config.run_controls.synthetic_bars),
            seed=seed,
        )
        panels = build_timeframe_panel_from_frames(base_frames, list(config.timeframes))
    else:
        panels = build_timeframe_panel(
            market_data_root=config.run_controls.market_data_root,
            exchange=config.run_controls.exchange,
            assets=list(config.universe.assets),
            timeframes=list(config.timeframes),
            start_date=config.run_controls.start_date,
            end_date=config.run_controls.end_date,
        )

    results: list[dict[str, Any]] = []
    all_fills: list[dict[str, Any]] = []
    strategy_k: dict[str, float] = {}

    for strategy in config.strategies:
        if not strategy.enabled:
            continue

        strategy_block = asdict(strategy)
        strategy_k[strategy.name] = float(strategy.cost_model.get("k_strategy", 1.0))

        for timeframe in [normalize_timeframe_token(item) for item in config.timeframes]:
            panel = panels.get(timeframe, pl.DataFrame())
            if panel.is_empty():
                continue
            enriched = compute_liquidity_metrics(panel, rolling_window=int(strategy.signal_params.get("liquidity_window", 20)))
            result = _run_single_backtest(
                panel=enriched,
                strategy_block=strategy_block,
                timeframe=timeframe,
                initial_capital=float(config.run_controls.initial_capital),
                cost_global=asdict(config.cost_global),
            )
            results.append(result)
            all_fills.extend(result["fills"])

    calibrations = calibrate_impact_coefficients(all_fills, strategy_k)
    run_token = str(run_id or f"{config.experiment_id}_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}")
    summary, table_rows = _build_summary(
        run_token,
        config.experiment_id,
        results,
        calibrations,
        ranking_objective=str(config.run_controls.ranking_objective),
        ranking_weights=dict(config.run_controls.ranking_weights),
    )
    return write_report_bundle(
        run_id=run_token,
        output_root=output_root,
        resolved_config=resolved,
        summary_payload=summary,
        table_rows=table_rows,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cost-aware timeframe-tunable framework")
    parser.add_argument(
        "--experiment",
        required=True,
        help="Path to experiment YAML (e.g., configs/quant_framework/experiments/cost_aware_timeframe_sweep.yaml)",
    )
    parser.add_argument("--output-root", default="reports", help="Output root directory")
    parser.add_argument("--run-id", default=None, help="Optional run identifier")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report_dir = run_cost_aware_framework(
        experiment_path=args.experiment,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    print(f"[cost-aware-framework] report_dir={report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
