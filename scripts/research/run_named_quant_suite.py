#!/usr/bin/env python3
"""Run manifest candidates through the full event-driven backtester."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.configuration import get_default_runtime_config
from lumina_quant.data.feature_points import FEATURE_COLUMNS, FEATURE_POINT_MAX_STALE_MS
from lumina_quant.market_data import MarketDataRepository, timeframe_to_milliseconds
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
    if not family:
        raise ValueError("missing family")
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


def _json_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "__dict__"):
        return _plain(vars(value))
    return value


def _lineage(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    receipt = manifest.get("universe_materialization_receipt")
    if not isinstance(receipt, dict):
        raise ValueError("manifest requires universe_materialization_receipt")
    config_source = os.getenv("LQ_CONFIG_PATH", "").strip()
    source = None
    if config_source:
        config_path = Path(config_source).expanduser().resolve()
        source = {
            "path": str(config_path),
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        }
    strict_execution = {
        "slippage_impact_model": "sqrt_impact",
        "slippage_impact_coefficient": 0.10,
        "require_funding_coverage": True,
        "funding_on_utc_boundary": True,
        "enforce_reduce_only": True,
        "apply_liquidity_cap_to_conditional_fills": True,
        "attach_default_protective_stop": False,
    }
    default_config = get_default_runtime_config()
    effective_defaults = _plain(default_config)
    effective_defaults.get("trading", {}).pop("timeframe", None)
    effective_defaults.get("live", {}).pop("symbol_limits", None)
    runtime = {
        "source": source,
        "default_config": effective_defaults,
        "strict_research_execution": strict_execution,
    }
    runtime["effective_sha256"] = _json_sha256(runtime)
    strategy_specs = [
        {
            key: candidate.get(key)
            for key in (
                "candidate_id",
                "strategy_class",
                "family",
                "strategy_timeframe",
                "timeframe",
                "params",
            )
        }
        for candidate in manifest.get("candidates", [])
        if isinstance(candidate, dict)
    ]
    return {
        "suite": {
            "suite_id": manifest.get("suite_id"),
            "base_strategy_spec_sha256": _json_sha256(strategy_specs),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "universe": {"receipt_sha256": _json_sha256(receipt), "receipt": receipt},
        "runtime_config": runtime,
    }


def _load_selection_artifact(
    path: Path, *, lineage: dict[str, Any], locked_start: datetime
) -> dict[str, Any]:
    selection = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(selection, dict) or selection.get("purpose") != "selection":
        raise ValueError("selection artifact must be a selection suite_results object")
    selected_lineage = selection.get("lineage")
    if not isinstance(selected_lineage, dict):
        raise ValueError("selection artifact missing lineage")
    if selected_lineage.get("suite", {}).get("suite_id") != lineage["suite"]["suite_id"]:
        raise ValueError("locked-OOS suite_id differs from selection")
    if (
        selected_lineage.get("suite", {}).get("base_strategy_spec_sha256")
        != lineage["suite"]["base_strategy_spec_sha256"]
    ):
        raise ValueError("locked-OOS base strategy spec differs from selection")
    if (
        selected_lineage.get("runtime_config", {}).get("effective_sha256")
        != lineage["runtime_config"]["effective_sha256"]
    ):
        raise ValueError("locked-OOS runtime-config identity differs from selection")
    period = selection.get("period")
    selection_end = _datetime(str(period.get("end"))) if isinstance(period, dict) else None
    if selection_end is None or selection_end >= locked_start:
        raise ValueError("selection end must be before locked-OOS start")
    return selection


def _prestart_bucket_count(frame: Any, start: datetime) -> int:
    try:
        return int(frame.filter(frame["datetime"] < start).height)
    except Exception as exc:
        raise RuntimeError("cannot verify actual pre-start warmup buckets") from exc


def _load_with_warmup(
    repository: MarketDataRepository,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    warmup_bars: int,
) -> Any:
    interval = timeframe_to_milliseconds(timeframe)
    span = max(1, warmup_bars) * interval
    for _ in range(8):
        frame = repository.load_ohlcv(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start - timedelta(milliseconds=span),
            end_date=end,
        )
        if (
            frame.is_empty()
            or warmup_bars == 0
            or _prestart_bucket_count(frame, start) >= warmup_bars
        ):
            return frame
        span *= 2
    raise RuntimeError(
        f"fewer than {warmup_bars} actual pre-start {timeframe} buckets for {symbol}"
    )


def _symbol_limits_from_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, float]]:
    receipt = manifest.get("universe_materialization_receipt")
    filters_by_symbol = receipt.get("binance_filters") if isinstance(receipt, dict) else None
    selected = receipt.get("selected_symbols")
    if selected is not None and (
        not isinstance(selected, dict)
        or any(
            not isinstance(symbols, list)
            or not all(isinstance(symbol, str) and symbol for symbol in symbols)
            for symbols in selected.values()
        )
    ):
        raise ValueError("universe receipt selected_symbols must contain symbol lists")
    selected_symbols = {symbol for symbols in (selected or {}).values() for symbol in symbols}
    if not selected_symbols:
        return {}
    if not isinstance(filters_by_symbol, dict):
        raise ValueError("universe receipt requires binance_filters")
    limits: dict[str, dict[str, float]] = {}
    for symbol in selected_symbols:
        filters = filters_by_symbol.get(symbol)
        if not isinstance(filters, list):
            raise ValueError(f"selected symbol {symbol} lacks PIT exchange filters")
        by_type = {
            str(row.get("filterType") or "").upper(): row
            for row in filters
            if isinstance(row, dict)
        }
        price_filter = by_type.get("PRICE_FILTER", {})
        tick_size = _positive_float(price_filter.get("tickSize"))
        lot_filter = by_type.get("LOT_SIZE", {})
        market_lot_filter = by_type.get("MARKET_LOT_SIZE", {})
        lot = next(
            (
                item
                for item in (market_lot_filter, lot_filter)
                if _positive_float(item.get("minQty")) is not None
                and _positive_float(item.get("stepSize")) is not None
            ),
            None,
        )
        notional_filter = by_type.get("MIN_NOTIONAL", {})
        min_notional = _positive_float(notional_filter.get("notional")) or _positive_float(
            notional_filter.get("minNotional")
        )
        if tick_size is None or lot is None or min_notional is None:
            raise ValueError(f"selected symbol {symbol} has incomplete PIT exchange limits")
        limits[symbol] = {
            "price_tick_size": tick_size,
            "min_qty": _positive_float(lot["minQty"]),
            "qty_step": _positive_float(lot["stepSize"]),
            "min_notional": min_notional,
        }
    return limits


def _required_features(strategy_cls: type, params: dict[str, Any]) -> tuple[str, ...]:
    raw = getattr(strategy_cls, "required_features", ())
    if isinstance(raw, property):
        raw = raw.__get__(strategy_cls(None, None, **params), strategy_cls)
    if isinstance(raw, str):
        raw = (raw,)
    return tuple(str(item).strip().lower() for item in tuple(raw or ()) if str(item).strip())


def _preflight_required_features(
    repository: MarketDataRepository,
    *,
    exchange: str,
    symbols: list[str],
    required_features: tuple[str, ...],
    start: datetime,
    end: datetime,
) -> None:
    if not required_features:
        return
    unsupported = sorted(set(required_features) - set(FEATURE_COLUMNS))
    if unsupported:
        raise RuntimeError("unsupported required features: " + ",".join(unsupported))
    stale = timedelta(milliseconds=FEATURE_POINT_MAX_STALE_MS)
    start_ms = int(start.replace(tzinfo=UTC).timestamp() * 1000)
    end_ms = int(end.replace(tzinfo=UTC).timestamp() * 1000)
    failures: list[str] = []
    for symbol in symbols:
        frame = repository.load_futures_feature_points(
            exchange=exchange,
            symbol=symbol,
            start_date=start - stale,
            end_date=end,
        )
        for feature in required_features:
            timestamps: list[int] = []
            if (
                not frame.is_empty()
                and feature in frame.columns
                and "timestamp_ms" in frame.columns
            ):
                for timestamp, value in frame.select("timestamp_ms", feature).iter_rows():
                    try:
                        parsed = float(value)
                    except TypeError, ValueError:
                        continue
                    if timestamp is not None and math.isfinite(parsed):
                        timestamps.append(int(timestamp))
            timestamps = sorted(set(timestamps))
            anchor = max((value for value in timestamps if value <= start_ms), default=None)
            covered = anchor is not None and start_ms - anchor <= FEATURE_POINT_MAX_STALE_MS
            points = ([anchor] if anchor is not None else []) + [
                value for value in timestamps if start_ms < value <= end_ms
            ]
            if covered:
                covered = (
                    all(
                        current - previous <= FEATURE_POINT_MAX_STALE_MS
                        for previous, current in pairwise(points)
                    )
                    and end_ms - points[-1] <= FEATURE_POINT_MAX_STALE_MS
                )
            if not covered:
                failures.append(f"{symbol}:{feature}")
    if failures:
        raise RuntimeError("missing or stale required features: " + ",".join(failures))


def _run_candidate(
    spec: dict[str, Any],
    *,
    repository: MarketDataRepository,
    exchange: str,
    start: datetime,
    end: datetime,
    symbol_limits: dict[str, dict[str, float]],
    warmup_bars: int,
) -> dict[str, Any]:
    data = {
        symbol: _load_with_warmup(
            repository,
            exchange=exchange,
            symbol=symbol,
            timeframe=spec["timeframe"],
            start=start,
            end=end,
            warmup_bars=warmup_bars,
        )
        for symbol in spec["symbols"]
    }
    missing = [symbol for symbol, frame in data.items() if frame.is_empty()]
    if missing:
        raise RuntimeError("no local OHLCV for: " + ",".join(missing))

    strategy_cls = resolve_strategy_class(spec["strategy_class"], strict=True)
    feature_start = start - timedelta(
        milliseconds=warmup_bars * timeframe_to_milliseconds(spec["timeframe"])
    )
    _preflight_required_features(
        repository,
        exchange=exchange,
        symbols=spec["symbols"],
        required_features=_required_features(strategy_cls, spec["params"]),
        start=feature_start,
        end=end,
    )

    config = get_default_runtime_config()
    config.trading.timeframe = spec["timeframe"]
    config.backtest.persist_output = False
    config.risk.attach_default_protective_stop = False
    config.execution.slippage_impact_model = "sqrt_impact"
    config.execution.slippage_impact_coefficient = 0.10
    config.execution.require_funding_coverage = True
    config.execution.funding_on_utc_boundary = True
    config.execution.enforce_reduce_only = True
    config.execution.apply_liquidity_cap_to_conditional_fills = True
    config.live.symbol_limits = {**config.live.symbol_limits, **symbol_limits}
    backtest = Backtest(
        "data",
        spec["symbols"],
        start,
        HistoricCSVDataHandler,
        SimulatedExecutionHandler,
        Portfolio,
        strategy_cls,
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
        warmup_bars=warmup_bars,
    )
    backtest.simulate_trading(output=False)
    return_timestamps, returns = _daily_returns(backtest.portfolio.all_holdings)
    initial_equity = float(backtest.portfolio.initial_capital)
    traded_notional = sum(
        abs(float(trade.get("fill_cost") or 0.0)) for trade in backtest.portfolio.trades
    )
    daily_turnover = traded_notional / initial_equity / max(1, len(returns))
    current_holdings = getattr(backtest.portfolio, "current_holdings", {})
    execution_config = getattr(getattr(backtest.portfolio, "execution_model", None), "cfg", None)
    return {
        **spec,
        "status": "pass",
        "return_timestamps": return_timestamps,
        "returns": returns,
        "returns_are_net": True,
        "turnover": daily_turnover,
        "turnover_definition": "mean_daily_sum_abs_fill_notional_over_initial_equity",
        "trade_count": int(backtest.portfolio.trade_count),
        "commission_paid": float(
            current_holdings.get("commission", 0.0) if isinstance(current_holdings, dict) else 0.0
        ),
        "net_funding_paid": float(getattr(backtest.portfolio, "total_funding_paid", 0.0)),
        "liquidation_count": len(getattr(backtest.portfolio, "liquidation_events", [])),
        "liquidation_model": "trade-price OHLC isolated liquidation approximation; mark/index retained as diagnostics",
        "warmup_bars": warmup_bars,
        "execution_model": asdict(execution_config)
        if execution_config is not None and is_dataclass(execution_config)
        else {},
        "research_execution_config": {
            "slippage_impact_model": config.execution.slippage_impact_model,
            "slippage_impact_coefficient": config.execution.slippage_impact_coefficient,
            "require_funding_coverage": config.execution.require_funding_coverage,
            "funding_on_utc_boundary": config.execution.funding_on_utc_boundary,
            "enforce_reduce_only": config.execution.enforce_reduce_only,
            "apply_liquidity_cap_to_conditional_fills": config.execution.apply_liquidity_cap_to_conditional_fills,
        },
        "default_protective_stop_attached": False,
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
    purpose: str = "selection",
    warmup_bars: int = 400,
    selection_artifact: Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = manifest.get("candidates") if isinstance(manifest, dict) else None
    if not isinstance(candidates, list):
        raise ValueError("manifest candidates must be a list")
    if end <= start:
        raise ValueError("end must be after start")
    if purpose not in {"selection", "locked_oos"}:
        raise ValueError("purpose must be 'selection' or 'locked_oos'")
    if warmup_bars < 0:
        raise ValueError("warmup_bars must be nonnegative")
    lineage = _lineage(manifest, manifest_path)
    receipt_as_of = _datetime(str(lineage["universe"]["receipt"].get("as_of")))
    if receipt_as_of > start:
        raise ValueError("universe receipt as_of must be at or before runner start")
    selection = None
    if purpose == "locked_oos":
        if selection_artifact is None:
            raise ValueError("locked_oos requires --selection-artifact")
        selection = _load_selection_artifact(
            selection_artifact, lineage=lineage, locked_start=start
        )
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
    receipt_disabled = lineage["universe"]["receipt"].get("disabled_candidates", {})
    receipt_disabled = receipt_disabled if isinstance(receipt_disabled, dict) else {}
    for index, candidate in enumerate(candidates):
        fallback_id = (
            str(candidate.get("candidate_id") or f"candidate_{index:03d}")
            if isinstance(candidate, dict)
            else f"candidate_{index:03d}"
        )
        if isinstance(candidate, dict) and candidate.get("enabled") is False:
            reason = str(candidate.get("disabled_reason") or "disabled by manifest")
            expected = receipt_disabled.get(fallback_id)
            allowed_exclusion = isinstance(
                expected, list
            ) and reason == "outside point-in-time universe: " + ", ".join(expected)
            result = {
                "candidate_id": fallback_id,
                "status": "skip",
                "reason": reason,
                "allowed_exclusion": allowed_exclusion,
                "return_timestamps": [],
                "returns": [],
                "turnover": None,
                **({"returns_are_net": True} if purpose == "locked_oos" else {}),
            }
            artifact = output_dir / _artifact_name(index, fallback_id)
            artifact.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            results.append({**result, "artifact": artifact.name})
            continue
        try:
            spec = _candidate_spec(candidate, index)
            result = _run_candidate(
                spec,
                repository=repository,
                exchange=exchange,
                start=start,
                end=end,
                symbol_limits=symbol_limits,
                warmup_bars=warmup_bars,
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
                **({"returns_are_net": True} if purpose == "locked_oos" else {}),
            }
        artifact = output_dir / _artifact_name(index, fallback_id)
        artifact.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        results.append({**result, "artifact": artifact.name})

    allocator = manifest.get("allocator")
    allocator = allocator if isinstance(allocator, dict) else {}
    min_sleeves = int(manifest.get("min_sleeves", allocator.get("min_sleeves", 1)))
    min_families = int(manifest.get("min_families", allocator.get("min_families", 1)))
    passed_families = {str(row.get("family") or "") for row in results if row["status"] == "pass"}
    disallowed_skips = [
        row["candidate_id"]
        for row in results
        if row["status"] == "skip" and not row.get("allowed_exclusion", False)
    ]
    allowed_exclusion_ids = {
        row["candidate_id"]
        for row in results
        if row["status"] == "skip" and row.get("allowed_exclusion", False)
    }
    exclusion_contract_complete = allowed_exclusion_ids == set(receipt_disabled)
    portfolio_ready = (
        not any(row["status"] == "fail" for row in results)
        and not disallowed_skips
        and exclusion_contract_complete
        and sum(row["status"] == "pass" for row in results) >= min_sleeves
        and len(passed_families) >= min_families
    )
    summary = {
        "suite_id": manifest.get("suite_id"),
        "purpose": purpose,
        "exchange": exchange,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "lineage": lineage,
        "candidate_count": len(results),
        "pass_count": sum(row["status"] == "pass" for row in results),
        "fail_count": sum(row["status"] == "fail" for row in results),
        "skip_count": sum(row["status"] == "skip" for row in results),
        "allowed_exclusions": [
            {"candidate_id": row["candidate_id"], "reason": row["reason"]}
            for row in results
            if row["status"] == "skip" and row.get("allowed_exclusion", False)
        ],
        "disallowed_skip_ids": disallowed_skips,
        "exclusion_contract": {
            "receipt_disabled_count": len(receipt_disabled),
            "allowed_exclusion_count": len(allowed_exclusion_ids),
            "complete": exclusion_contract_complete,
        },
        "readiness": {
            "portfolio_ready": portfolio_ready,
            "min_sleeves": min_sleeves,
            "passing_sleeves": sum(row["status"] == "pass" for row in results),
            "min_families": min_families,
            "passing_families": len(passed_families),
        },
        "warmup_bars": warmup_bars,
        "input_sha256": {
            "manifest": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "results": results,
    }
    if selection is not None:
        summary["selection_artifact"] = {
            "path": str(selection_artifact.resolve()),
            "sha256": hashlib.sha256(selection_artifact.read_bytes()).hexdigest(),
            "period": selection["period"],
            "manifest_sha256": selection["lineage"]["suite"]["manifest_sha256"],
            "universe_receipt_sha256": selection["lineage"]["universe"]["receipt_sha256"],
        }
    (output_dir / "suite_results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_path = output_dir / "suite_results.json"
    if purpose == "locked_oos":
        return summary

    allocation_input = deepcopy(manifest)
    sleeves = allocation_input.get("sleeves")
    if isinstance(sleeves, dict):
        results_by_id = {row["candidate_id"]: row for row in results}
        for sleeve_id, sleeve in sleeves.items():
            if not isinstance(sleeve, dict):
                continue
            sleeve["source_artifact_id"] = "named_quant_data_pc_walkforward"
            result = results_by_id.get(str(sleeve_id))
            if result is None or result["status"] != "pass":
                sleeve.update(
                    {
                        "returns": None,
                        "turnover": None,
                        "run_status": str((result or {}).get("status") or "fail"),
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
                        "stream": "daily UTC net returns over the caller-supplied selection window",
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
            "ready": portfolio_ready,
            "portfolio_ready": portfolio_ready,
            "allowed_exclusions": summary["allowed_exclusions"],
            "exclusion_contract": summary["exclusion_contract"],
            "selection_period": summary["period"],
            "lineage": lineage,
            "frozen_at": summary["period"]["end"],
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
    parser.add_argument(
        "--purpose",
        choices=("selection", "locked_oos"),
        default="selection",
        help="selection emits allocator input; locked_oos never does",
    )
    parser.add_argument("--selection-artifact", type=Path)
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=400,
        help="pre-window strategy-timeframe bars loaded for indicator state",
    )
    args = parser.parse_args()
    summary = run_suite(
        args.manifest,
        args.data_root,
        args.output_dir,
        exchange=args.exchange,
        start=_datetime(args.start),
        end=_datetime(args.end),
        purpose=args.purpose,
        warmup_bars=args.warmup_bars,
        selection_artifact=args.selection_artifact,
    )
    raise SystemExit(0 if summary["readiness"]["portfolio_ready"] else 1)


if __name__ == "__main__":
    main()
