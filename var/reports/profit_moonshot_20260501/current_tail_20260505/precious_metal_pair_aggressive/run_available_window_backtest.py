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
START = datetime(2026, 1, 30, 10, 15, tzinfo=UTC)
END = datetime(2026, 3, 7, 11, 0, tzinfo=UTC)
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

def parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        out = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    return out.replace(tzinfo=UTC) if out.tzinfo is None else out.astimezone(UTC)

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
    start_date=START,
    end_date=END,
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
    record_history=True,
    track_metrics=True,
    record_trades=True,
)

holdings = []
for row in list(getattr(backtest.portfolio, "all_holdings", []) or []):
    if len(row) < 5:
        continue
    dt = parse_dt(row[0])
    if dt is None:
        continue
    holdings.append((dt, safe_float(row[4])))

trades = list(getattr(backtest.portfolio, "trades", []) or [])

def trade_time(trade: Any) -> datetime | None:
    if isinstance(trade, dict):
        for key in ("datetime", "time", "timestamp", "fill_time"):
            out = parse_dt(trade.get(key))
            if out is not None:
                return out
    return parse_dt(getattr(trade, "datetime", None) or getattr(trade, "time", None))

split_rows = {}
for name, (a, b) in SPLITS.items():
    values = [total for dt, total in holdings if a <= dt < b]
    trade_count = sum(1 for trade in trades if (t := trade_time(trade)) is not None and a <= t < b)
    split_rows[name] = {
        "start": a.isoformat(),
        "end": b.isoformat(),
        "history_points": len(values),
        "metrics": metrics_from_equity(values, int(getattr(BacktestConfig, "ANNUAL_PERIODS", 252))),
        "trade_count": int(trade_count),
    }

payload = {
    "artifact_kind": "precious_metal_pair_available_window_backtest",
    "mode": MODE,
    "symbols": SYMBOLS,
    "data_contract": "legacy-windowed common-overlap evidence; raw-first live-equivalent train/val gate is handled separately",
    "window": {"start": START.isoformat(), "end": END.isoformat()},
    "splits": split_rows,
    "full_window": {
        "history_points": len([x for _, x in holdings]),
        "metrics": metrics_from_equity([x for _, x in holdings], int(getattr(BacktestConfig, "ANNUAL_PERIODS", 252))),
        "trade_count": int(getattr(backtest.portfolio, "trade_count", 0)),
        "final_equity": safe_float(dict(getattr(backtest.portfolio, "current_holdings", {}) or {}).get("total")),
        "liquidation_count": len(getattr(backtest.portfolio, "liquidation_events", []) or []),
    },
    "notes": [
        "Runs one portfolio mode only, all four metals included via XAU/XAG and XPT/XPD components.",
        "Split metrics are computed from a continuous warm-up run and sliced by timestamp to avoid OOS warm-up starvation on the short metals overlap.",
    ],
}
json_path = ROOT / "precious_pair_available_window_backtest.json"
json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
md = ["# Precious-metal pair aggressive mode available-window backtest", ""]
md.append(f"- mode: `{MODE}`")
md.append(f"- window: `{START.isoformat()}` to `{END.isoformat()}`")
md.append(f"- data: {payload['data_contract']}")
for name, row in split_rows.items():
    m = row["metrics"]
    md.append(
        f"- {name}: return {m['total_return']:.6%}, MDD {m['max_drawdown']:.6%}, "
        f"Sharpe {m['sharpe']:.6f}, trades {row['trade_count']}, points {row['history_points']}"
    )
fm = payload["full_window"]["metrics"]
md.append(
    f"- full_window: return {fm['total_return']:.6%}, MDD {fm['max_drawdown']:.6%}, "
    f"Sharpe {fm['sharpe']:.6f}, trades {payload['full_window']['trade_count']}"
)
(ROOT / "precious_pair_available_window_backtest.md").write_text("\n".join(md)+"\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
