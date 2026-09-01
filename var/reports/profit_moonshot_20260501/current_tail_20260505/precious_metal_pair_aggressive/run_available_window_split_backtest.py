from __future__ import annotations

import json, math
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import numpy as np

from lumina_quant.backtesting.chunked_runner import run_backtest_chunked
from lumina_quant.backtesting.data_windowed_parquet import HistoricParquetWindowedDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.configuration.runtime_access import BaseConfig, BacktestConfig
from lumina_quant.market_data import load_data_dict_from_parquet
from lumina_quant.strategies.artifact_portfolio_mode import ArtifactPortfolioModeStrategy

MODE = "profit_moonshot_precious_metal_pair_aggressive_mode"
SYMBOLS = ["XAU/USDT", "XAG/USDT", "XPT/USDT", "XPD/USDT"]
ROOT = Path("var/reports/profit_moonshot_20260501/current_tail_20260505/precious_metal_pair_aggressive")
ROOT.mkdir(parents=True, exist_ok=True)
SPLITS = {
    "train": (datetime(2026, 1, 30, 10, 15, tzinfo=UTC), datetime(2026, 2, 15, tzinfo=UTC)),
    "val": (datetime(2026, 2, 15, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)),
    "oos": (datetime(2026, 3, 1, tzinfo=UTC), datetime(2026, 3, 7, 11, tzinfo=UTC)),
}

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default

def metrics_from_equity(equity: list[float], periods: int) -> dict[str, float]:
    arr = np.asarray([safe_float(x) for x in equity], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2 or arr[0] <= 0:
        return {"total_return": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "volatility": 0.0, "cagr": 0.0, "calmar": 0.0}
    returns = np.diff(arr) / arr[:-1]
    returns = returns[np.isfinite(returns)]
    total_return = float(arr[-1] / arr[0] - 1.0)
    if returns.size == 0:
        return {"total_return": total_return, "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "volatility": 0.0, "cagr": 0.0, "calmar": 0.0}
    std = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    mean = float(np.mean(returns))
    volatility = float(std * math.sqrt(periods)) if std > 0 else 0.0
    sharpe = float((mean / std) * math.sqrt(periods)) if std > 0 else 0.0
    downside = returns[returns < 0]
    downside_std = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    sortino = float((mean / downside_std) * math.sqrt(periods)) if downside_std > 0 else 0.0
    running_max = np.maximum.accumulate(arr)
    drawdowns = np.where(running_max > 0, (running_max - arr) / running_max, 0.0)
    max_drawdown = float(np.max(drawdowns)) if drawdowns.size else 0.0
    cagr = float((1.0 + total_return) ** (periods / returns.size) - 1.0) if total_return > -1.0 else -1.0
    calmar = float(cagr / max_drawdown) if max_drawdown > 1e-12 else 0.0
    return {"total_return": total_return, "sharpe": sharpe, "sortino": sortino, "max_drawdown": max_drawdown, "volatility": volatility, "cagr": cagr, "calmar": calmar}

def run_split(name: str, start: datetime, end: datetime) -> dict[str, Any]:
    def loader(chunk_start: datetime, chunk_end: datetime):
        return load_data_dict_from_parquet(
            "data/market_parquet",
            exchange="binance",
            symbol_list=SYMBOLS,
            timeframe="1m",
            start_date=chunk_start,
            end_date=chunk_end,
            chunk_days=3,
            warmup_bars=0,
            data_mode="legacy",
            staleness_threshold_seconds=None,
        )

    backtest = run_backtest_chunked(
        csv_dir="data",
        symbol_list=SYMBOLS,
        start_date=start,
        end_date=end,
        strategy_cls=ArtifactPortfolioModeStrategy,
        strategy_params={"portfolio_mode": MODE},
        data_loader=loader,
        chunk_days=3,
        strategy_timeframe=str(BaseConfig.TIMEFRAME),
        data_handler_cls=HistoricParquetWindowedDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        backtest_mode="windowed",
        data_handler_kwargs={"backtest_poll_seconds": 1, "backtest_window_seconds": 600},
        record_history=False,
        track_metrics=True,
        record_trades=False,
    )
    values = list(getattr(backtest.portfolio, "_metric_totals", []) or [])
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "metric_points": len(values),
        "metrics": metrics_from_equity(values, int(getattr(BacktestConfig, "ANNUAL_PERIODS", 252))),
        "trade_count": int(getattr(backtest.portfolio, "trade_count", 0)),
        "final_equity": safe_float(dict(getattr(backtest.portfolio, "current_holdings", {}) or {}).get("total")),
        "liquidation_count": len(getattr(backtest.portfolio, "liquidation_events", []) or []),
    }

split_rows = {name: run_split(name, *window) for name, window in SPLITS.items()}
payload = {
    "artifact_kind": "precious_metal_pair_available_window_split_backtest",
    "mode": MODE,
    "symbols": SYMBOLS,
    "data_contract": "legacy-windowed split evidence on current four-metal common overlap; raw-first live-equivalent train/val gate is separate and blocking",
    "splits": split_rows,
}
(ROOT / "precious_pair_available_window_split_backtest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
md = ["# Precious-metal pair aggressive mode split backtest", "", f"- mode: `{MODE}`", f"- data: {payload['data_contract']}"]
for name, row in split_rows.items():
    m = row["metrics"]
    md.append(f"- {name}: return {m['total_return']:.6%}, MDD {m['max_drawdown']:.6%}, Sharpe {m['sharpe']:.6f}, trades {row['trade_count']}, liquidations {row['liquidation_count']}")
(ROOT / "precious_pair_available_window_split_backtest.md").write_text("\n".join(md)+"\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
