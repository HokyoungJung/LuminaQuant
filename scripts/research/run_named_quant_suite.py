#!/usr/bin/env python3
"""Run manifest candidates through the full event-driven backtester."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from copy import deepcopy
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.configuration import get_default_runtime_config
from lumina_quant.market_data import MarketDataRepository
from lumina_quant.strategies.registry import resolve_strategy_class


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def _candidate_spec(candidate: Any, index: int) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        raise ValueError("candidate must be an object")
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    strategy_class = str(candidate.get("strategy_class") or "").strip()
    family = str(candidate.get("family") or "").strip()
    symbols = candidate.get("symbols")
    params = candidate.get("params", {})
    timeframe = str(candidate.get("strategy_timeframe") or candidate.get("timeframe") or "").strip()
    if not candidate_id:
        raise ValueError(f"candidate[{index}] missing candidate_id")
    if not strategy_class:
        raise ValueError("missing strategy_class")
    if (
        not isinstance(symbols, list)
        or not symbols
        or not all(isinstance(s, str) and s for s in symbols)
    ):
        raise ValueError("symbols must be a non-empty string list")
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    if not timeframe:
        raise ValueError("missing timeframe")
    if (
        candidate.get("timeframe")
        and candidate.get("strategy_timeframe")
        and str(candidate["timeframe"]) != str(candidate["strategy_timeframe"])
    ):
        raise ValueError("timeframe and strategy_timeframe disagree")
    return {
        "candidate_id": candidate_id,
        "strategy_class": strategy_class,
        "family": family,
        "symbols": symbols,
        "params": params,
        "timeframe": timeframe,
    }


def _returns(totals: list[float]) -> list[float]:
    if len(totals) < 2 or any(not math.isfinite(value) or value <= 0 for value in totals):
        raise RuntimeError("backtest produced insufficient or invalid equity observations")
    return [current / previous - 1.0 for previous, current in pairwise(totals)]


def _daily_returns(holdings: list[tuple[Any, ...]]) -> tuple[list[str], list[float]]:
    observations: list[tuple[datetime, float]] = []
    for row in holdings:
        timestamp = row[0]
        if not isinstance(timestamp, datetime):
            timestamp = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(UTC).replace(tzinfo=None)
        observations.append((timestamp, float(row[4])))
    daily_equity: dict[str, float] = {}
    for timestamp, total in sorted(observations):
        daily_equity[timestamp.date().isoformat()] = total
    dates = sorted(daily_equity)
    return dates[1:], _returns([daily_equity[date] for date in dates])


def _artifact_name(index: int, candidate_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate_id).strip("._") or "candidate"
    return f"{index:03d}_{safe_id}.json"


def _positive_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except TypeError, ValueError:
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _symbol_limits_from_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, float]]:
    receipt = manifest.get("universe_materialization_receipt")
    filters_by_symbol = receipt.get("binance_filters") if isinstance(receipt, dict) else None
    if not isinstance(filters_by_symbol, dict):
        return {}
    limits: dict[str, dict[str, float]] = {}
    for symbol, filters in filters_by_symbol.items():
        if not isinstance(filters, list):
            continue
        by_type = {
            str(row.get("filterType") or "").upper(): row
            for row in filters
            if isinstance(row, dict)
        }
        symbol_limits: dict[str, float] = {}
        price_filter = by_type.get("PRICE_FILTER", {})
        tick_size = _positive_float(price_filter.get("tickSize"))
        if tick_size is not None:
            symbol_limits["price_tick_size"] = tick_size
        lot_filter = by_type.get("LOT_SIZE", {})
        market_lot_filter = by_type.get("MARKET_LOT_SIZE", {})
        min_qty = _positive_float(market_lot_filter.get("minQty")) or _positive_float(
            lot_filter.get("minQty")
        )
        qty_step = _positive_float(market_lot_filter.get("stepSize")) or _positive_float(
            lot_filter.get("stepSize")
        )
        if min_qty is not None:
            symbol_limits["min_qty"] = min_qty
        if qty_step is not None:
            symbol_limits["qty_step"] = qty_step
        notional_filter = by_type.get("MIN_NOTIONAL", {})
        min_notional = _positive_float(notional_filter.get("notional")) or _positive_float(
            notional_filter.get("minNotional")
        )
        if min_notional is not None:
            symbol_limits["min_notional"] = min_notional
        if symbol_limits:
            limits[str(symbol)] = symbol_limits
    return limits


def _run_candidate(
    spec: dict[str, Any],
    *,
    repository: MarketDataRepository,
    exchange: str,
    start: datetime,
    end: datetime,
    symbol_limits: dict[str, dict[str, float]],
) -> dict[str, Any]:
    data = {
        symbol: repository.load_ohlcv(
            exchange=exchange,
            symbol=symbol,
            timeframe=spec["timeframe"],
            start_date=start,
            end_date=end,
        )
        for symbol in spec["symbols"]
    }
    missing = [symbol for symbol, frame in data.items() if frame.is_empty()]
    if missing:
        raise RuntimeError("no local OHLCV for: " + ",".join(missing))

    config = get_default_runtime_config()
    config.trading.timeframe = spec["timeframe"]
    config.backtest.persist_output = False
    config.live.symbol_limits = {
        **config.live.symbol_limits,
        **{
            symbol: {**config.live.symbol_limits.get(symbol, {}), **limits}
            for symbol, limits in symbol_limits.items()
        },
    }
    backtest = Backtest(
        "data",
        spec["symbols"],
        start,
        HistoricCSVDataHandler,
        SimulatedExecutionHandler,
        Portfolio,
        resolve_strategy_class(spec["strategy_class"], strict=True),
        strategy_params=spec["params"],
        end_date=end,
        data_dict=data,
        record_history=True,
        track_metrics=True,
        record_trades=True,
        strategy_timeframe=spec["timeframe"],
        data_handler_kwargs={
            "feature_db_path": repository.db_path,
            "feature_exchange": exchange,
        },
        config=config,
    )
    backtest.simulate_trading(output=False)
    return_timestamps, returns = _daily_returns(backtest.portfolio.all_holdings)
    initial_equity = float(backtest.portfolio.initial_capital)
    traded_notional = sum(
        abs(float(trade.get("fill_cost") or 0.0)) for trade in backtest.portfolio.trades
    )
    daily_turnover = traded_notional / initial_equity / max(1, len(returns))
    return {
        **spec,
        "status": "pass",
        "return_timestamps": return_timestamps,
        "returns": returns,
        "turnover": daily_turnover,
        "turnover_definition": "mean_daily_sum_abs_fill_notional_over_initial_equity",
        "trade_count": int(backtest.portfolio.trade_count),
        "uses_locked_oos_for_selection": False,
        "uses_locked_oos_for_sizing": False,
    }


def run_suite(
    manifest_path: Path,
    data_root: Path,
    output_dir: Path,
    *,
    exchange: str,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = manifest.get("candidates") if isinstance(manifest, dict) else None
    if not isinstance(candidates, list):
        raise ValueError("manifest candidates must be a list")
    if end <= start:
        raise ValueError("end must be after start")
    candidate_ids = [
        str(candidate.get("candidate_id") or "")
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("duplicate candidate_id")

    output_dir.mkdir(parents=True, exist_ok=True)
    repository = MarketDataRepository(str(data_root))
    symbol_limits = _symbol_limits_from_manifest(manifest)
    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        fallback_id = (
            str(candidate.get("candidate_id") or f"candidate_{index:03d}")
            if isinstance(candidate, dict)
            else f"candidate_{index:03d}"
        )
        try:
            spec = _candidate_spec(candidate, index)
            result = _run_candidate(
                spec,
                repository=repository,
                exchange=exchange,
                start=start,
                end=end,
                symbol_limits=symbol_limits,
            )
        except Exception as exc:
            result = {
                "candidate_id": fallback_id,
                "status": "fail",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "return_timestamps": [],
                "returns": [],
                "turnover": None,
            }
        artifact = output_dir / _artifact_name(index, fallback_id)
        artifact.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        results.append({**result, "artifact": artifact.name})

    summary = {
        "suite_id": manifest.get("suite_id"),
        "exchange": exchange,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "candidate_count": len(results),
        "pass_count": sum(row["status"] == "pass" for row in results),
        "fail_count": sum(row["status"] == "fail" for row in results),
        "results": results,
    }
    (output_dir / "suite_results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_path = output_dir / "suite_results.json"
    allocation_input = deepcopy(manifest)
    sleeves = allocation_input.get("sleeves")
    if isinstance(sleeves, dict):
        results_by_id = {row["candidate_id"]: row for row in results}
        for sleeve_id, sleeve in sleeves.items():
            if not isinstance(sleeve, dict):
                continue
            result = results_by_id.get(str(sleeve_id))
            if result is None or result["status"] != "pass":
                sleeve.update(
                    {
                        "returns": None,
                        "turnover": None,
                        "run_status": "fail",
                        "run_error": (result or {}).get(
                            "error", "candidate not present in results"
                        ),
                    }
                )
                continue
            sleeve.update(
                {
                    "strategy_class": result["strategy_class"],
                    "symbols": result["symbols"],
                    "params": result["params"],
                    "family": result["family"],
                    "return_timestamps": result["return_timestamps"],
                    "returns": result["returns"],
                    "returns_are_net": True,
                    "turnover": result["turnover"],
                    "run_status": "pass",
                    "returns_source": {
                        "artifact": "named_quant_data_pc_walkforward",
                        "candidate_id": result["candidate_id"],
                        "stream": "daily UTC train/validation net returns",
                        "selection_inputs": ["train", "validation"],
                        "uses_locked_oos_for_selection": False,
                        "uses_locked_oos_for_sizing": False,
                    },
                }
            )
    allocation_input["source_artifacts"] = [
        {
            "id": "named_quant_data_pc_walkforward",
            "path": str(summary_path.resolve()),
            "sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "max_age_hours": 8760,
            "ready": summary["fail_count"] == 0,
            "portfolio_ready": summary["fail_count"] == 0,
        }
    ]
    (output_dir / "allocation_input.json").write_text(
        json.dumps(allocation_input, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    summary = run_suite(
        args.manifest,
        args.data_root,
        args.output_dir,
        exchange=args.exchange,
        start=_datetime(args.start),
        end=_datetime(args.end),
    )
    raise SystemExit(1 if summary["fail_count"] else 0)


if __name__ == "__main__":
    main()
