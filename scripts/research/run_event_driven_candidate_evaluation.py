#!/usr/bin/env python3
"""Run manifest-defined candidates through the full event-driven backtester."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.configuration import get_default_runtime_config
from lumina_quant.data.feature_points import (
    BINANCE_FUNDING_SOURCE_JITTER_TOLERANCE_MS,
    FEATURE_COLUMNS,
    FEATURE_POINT_MAX_STALE_MS,
)
from lumina_quant.market_data import MarketDataRepository, timeframe_to_milliseconds
from lumina_quant.portfolio.quality_gated_allocation import (
    _materialized_return_panel_sha256,
)
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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _path_matches_sha256(value: Any, expected: Any) -> bool:
    if not isinstance(value, str) or not _is_sha256(expected):
        return False
    try:
        path = Path(value).expanduser().resolve(strict=True)
        return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
    except OSError:
        return False


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


def _data_inventory(data_root: Path) -> dict[str, Any]:
    if not data_root.is_dir():
        raise ValueError(f"data root must be an existing directory: {data_root}")
    files = [
        {
            "path": str(path.relative_to(data_root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(data_root.rglob("*"))
        if path.is_file()
    ]
    return {"root": str(data_root.resolve()), "files": files, "sha256": _json_sha256(files)}


def _cost_profile() -> tuple[dict[str, Any], Any, dict[str, Any]]:
    raw_path = os.getenv("LQ_CONFIG_PATH", "").strip()
    if not raw_path:
        raise ValueError("LQ_CONFIG_PATH must explicitly name a realistic-cost profile")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"realistic-cost profile does not exist: {path}")
    runtime_config = get_default_runtime_config()
    runtime = _plain(runtime_config)
    execution = runtime.get("execution") if isinstance(runtime, dict) else None
    if not isinstance(execution, dict):
        raise ValueError("realistic-cost profile has no execution configuration")
    required_positive = (
        "maker_fee_rate",
        "taker_fee_rate",
        "spread_rate",
        "slippage_rate",
        "slippage_impact_coefficient",
        "maintenance_margin_rate",
        "liquidation_buffer_rate",
    )
    missing = [key for key in required_positive if _positive_float(execution.get(key)) is None]
    if missing:
        raise ValueError("realistic-cost profile requires nonzero " + ", ".join(missing))
    if execution.get("slippage_impact_model") != "sqrt_impact":
        raise ValueError("realistic-cost profile requires slippage_impact_model=sqrt_impact")
    if not execution.get("require_funding_coverage") or not execution.get(
        "funding_on_utc_boundary"
    ):
        raise ValueError("realistic-cost profile requires funding coverage and UTC settlement")
    return (
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        runtime_config,
        runtime,
    )


def _source_commit() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=Path(__file__).parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("cannot determine source commit")
    return commit


def _source_identity() -> dict[str, Any]:
    root = Path(__file__).parents[2].resolve()
    paths = [Path(__file__).resolve(), *sorted((root / "src/lumina_quant").rglob("*.py"))]
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return {
        "git_commit": _source_commit(),
        "executed_python_file_count": len(paths),
        "executed_source_sha256": digest.hexdigest(),
    }


def _rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace publication is unsupported")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(target)
    raise OSError(error_number, os.strerror(error_number), str(target))


def _publish_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _lineage(
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    exchange: str,
    warmup_bars: int,
    seed: int,
    cost_profile: dict[str, Any],
    runtime_defaults: dict[str, Any],
    data_inventory: dict[str, Any],
) -> dict[str, Any]:
    receipt = manifest.get("universe_materialization_receipt")
    if not isinstance(receipt, dict):
        raise ValueError("manifest requires universe_materialization_receipt")
    strict_execution = {
        "slippage_impact_model": "sqrt_impact",
        "slippage_impact_coefficient": 0.10,
        "require_funding_coverage": True,
        "funding_on_utc_boundary": True,
        "enforce_reduce_only": True,
        "apply_liquidity_cap_to_conditional_fills": True,
        "attach_default_protective_stop": False,
    }
    effective_defaults = deepcopy(runtime_defaults)
    effective_defaults.get("trading", {}).pop("timeframe", None)
    effective_defaults.get("live", {}).pop("symbol_limits", None)
    runtime = {
        "source": cost_profile,
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
    behavioral_identity = {
        "exchange": exchange,
        "warmup_bars": warmup_bars,
        "determinism": {
            "numpy_legacy_seed": seed,
            "python_random_seed": seed,
        },
        "source": _source_identity(),
        "cost_profile": cost_profile,
        "runtime_config_sha256": runtime["effective_sha256"],
    }
    return {
        "suite": {
            "suite_id": manifest.get("suite_id"),
            "base_strategy_spec_sha256": _json_sha256(strategy_specs),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "universe": {"receipt_sha256": _json_sha256(receipt), "receipt": receipt},
        "data_inventory": data_inventory,
        "runtime_config": runtime,
        "behavioral_identity": behavioral_identity,
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
    if selected_lineage.get("behavioral_identity") != lineage["behavioral_identity"]:
        raise ValueError("locked-OOS behavior-affecting identity differs from selection")
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


def _allowed_disabled_candidate(
    *,
    candidate_id: str,
    reason: str,
    expected: Any,
    receipt: dict[str, Any],
    start: datetime,
    warmup_bars: int,
    data_inventory_sha256: str,
    candidate_symbols: Any,
    candidate_timeframe: Any,
) -> bool:
    if isinstance(expected, list):
        sources = receipt.get("sources")
        source_sha256 = receipt.get("source_sha256")
        selected = receipt.get("selected_symbols")
        selected_symbols = (
            {symbol for symbols in selected.values() for symbol in symbols}
            if isinstance(selected, dict)
            and all(
                isinstance(symbols, list)
                and all(isinstance(symbol, str) and symbol for symbol in symbols)
                for symbols in selected.values()
            )
            else set()
        )
        excluded_symbols = (
            sorted(set(candidate_symbols) - selected_symbols)
            if isinstance(candidate_symbols, list)
            and all(isinstance(symbol, str) and symbol for symbol in candidate_symbols)
            else []
        )
        return (
            bool(expected)
            and expected == excluded_symbols
            and all(
                isinstance(symbol, str) and symbol and symbol not in selected_symbols
                for symbol in expected
            )
            and reason == ("outside point-in-time universe: " + ", ".join(expected))
            and isinstance(sources, dict)
            and isinstance(source_sha256, dict)
            and set(sources) == set(source_sha256)
            and bool(sources)
            and all(
                _path_matches_sha256(identity, source_sha256.get(name))
                for name, identity in sources.items()
            )
        )
    if not isinstance(expected, dict) or set(expected) != {
        "kind",
        "reason",
        "required_buckets",
        "shortfalls",
        "timeframe",
        "candidate_symbols",
    }:
        return False
    if (
        expected.get("kind") != "insufficient_point_in_time_history"
        or expected.get("reason") != reason
        or type(expected.get("required_buckets")) is not int
        or expected["required_buckets"] != warmup_bars
        or expected.get("timeframe") != candidate_timeframe
        or not isinstance(candidate_timeframe, str)
        or not candidate_timeframe
        or expected.get("candidate_symbols") != candidate_symbols
        or not isinstance(candidate_symbols, list)
        or not candidate_symbols
        or not all(isinstance(symbol, str) and symbol for symbol in candidate_symbols)
        or not isinstance(expected.get("shortfalls"), dict)
        or not expected["shortfalls"]
        or any(
            not isinstance(symbol, str)
            or not symbol
            or type(count) is not int
            or count < 0
            or count >= warmup_bars
            for symbol, count in expected["shortfalls"].items()
        )
    ):
        return False
    eligibility = receipt.get("data_eligibility")
    if not isinstance(eligibility, dict) or "sha256" not in eligibility:
        return False
    scope = {key: value for key, value in eligibility.items() if key != "sha256"}
    return (
        set(eligibility)
        == {
            "schema",
            "start",
            "required_buckets",
            "input_data_inventory_sha256",
            "resample_receipt_path",
            "resample_receipt_sha256",
            "exclusions",
            "sha256",
        }
        and eligibility.get("schema") == "named_quant_data_eligibility.v1"
        and eligibility.get("start") == start.replace(tzinfo=UTC).isoformat()
        and eligibility.get("required_buckets") == warmup_bars
        and eligibility.get("input_data_inventory_sha256") == data_inventory_sha256
        and _path_matches_sha256(
            eligibility.get("resample_receipt_path"),
            eligibility.get("resample_receipt_sha256"),
        )
        and isinstance(eligibility.get("exclusions"), dict)
        and eligibility["exclusions"].get(candidate_id) == expected
        and eligibility.get("sha256") == _json_sha256(scope)
    )


def _required_features(strategy_cls: type, params: dict[str, Any]) -> tuple[str, ...]:
    raw = getattr(strategy_cls, "required_features", ())
    if isinstance(raw, property):
        raw = raw.__get__(strategy_cls(None, None, **params), strategy_cls)
    if isinstance(raw, str):
        raw = (raw,)
    return tuple(str(item).strip().lower() for item in tuple(raw or ()) if str(item).strip())


def _funding_feature_covered(timestamps: list[int], *, start_ms: int, end_ms: int) -> bool:
    """Require exactly one finite source row in each nominal settlement window."""
    interval_ms = FEATURE_POINT_MAX_STALE_MS
    first_boundary_ms = ((start_ms // interval_ms) + 1) * interval_ms
    last_boundary_ms = end_ms - (end_ms % interval_ms)
    return all(
        sum(
            boundary_ms <= timestamp <= boundary_ms + BINANCE_FUNDING_SOURCE_JITTER_TOLERANCE_MS
            for timestamp in timestamps
        )
        == 1
        for boundary_ms in range(first_boundary_ms, last_boundary_ms + 1, interval_ms)
    )


def _preflight_required_features(
    repository: MarketDataRepository,
    *,
    exchange: str,
    symbols: list[str],
    required_features: tuple[str, ...],
    require_utc_funding: bool = False,
    start: datetime,
    end: datetime,
) -> None:
    if require_utc_funding:
        required_features = tuple(dict.fromkeys((*required_features, "funding_fee_quote_per_unit")))
    if not required_features:
        return
    unsupported = sorted(set(required_features) - set(FEATURE_COLUMNS))
    if unsupported:
        raise RuntimeError("unsupported required features: " + ",".join(unsupported))
    stale = timedelta(milliseconds=FEATURE_POINT_MAX_STALE_MS)
    funding_source_jitter = (
        timedelta(milliseconds=BINANCE_FUNDING_SOURCE_JITTER_TOLERANCE_MS)
        if any(feature.startswith("funding_") for feature in required_features)
        else timedelta()
    )
    start_ms = int(start.replace(tzinfo=UTC).timestamp() * 1000)
    end_ms = int(end.replace(tzinfo=UTC).timestamp() * 1000)
    failures: list[str] = []
    for symbol in symbols:
        frame = repository.load_futures_feature_points(
            exchange=exchange,
            symbol=symbol,
            start_date=start - stale,
            end_date=end + funding_source_jitter,
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
            timestamps = sorted(timestamps)
            if feature.startswith("funding_"):
                if not _funding_feature_covered(timestamps, start_ms=start_ms, end_ms=end_ms):
                    failures.append(f"{symbol}:{feature}")
                continue
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
    runtime_config: Any,
    seed: int,
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
        require_utc_funding=False,
        start=feature_start,
        end=end,
    )
    # Funding fees are a portfolio accounting input, not strategy formation
    # data. Warmup signals cannot create portfolio positions, so exact fee
    # coverage is required from the first live instant rather than over the
    # strategy's pre-start formation prefix.
    _preflight_required_features(
        repository,
        exchange=exchange,
        symbols=spec["symbols"],
        required_features=(),
        require_utc_funding=True,
        start=start,
        end=end,
    )

    random.seed(seed)
    np.random.seed(seed)
    config = deepcopy(runtime_config)
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
    funding_settlement_end = end + timedelta(
        milliseconds=BINANCE_FUNDING_SOURCE_JITTER_TOLERANCE_MS
    )
    backtest = Backtest(
        "data",
        spec["symbols"],
        start,
        HistoricCSVDataHandler,
        SimulatedExecutionHandler,
        Portfolio,
        strategy_cls,
        strategy_params=spec["params"],
        end_date=funding_settlement_end,
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
    backtest.portfolio.settle_terminal_funding(funding_settlement_end)
    pending_liabilities = getattr(backtest.portfolio, "_pending_funding_liabilities", None)
    if isinstance(pending_liabilities, dict) and any(
        bool(rows) for rows in pending_liabilities.values()
    ):
        raise RuntimeError("terminal pending funding liability lacks settlement evidence")
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


def _run_suite_into(
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
    seed: int = 0,
    published_output_dir: Path | None = None,
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
    if type(seed) is not int or seed < 0 or seed > 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    if output_dir.exists():
        raise ValueError(f"output target already exists: {output_dir}")
    if not all(isinstance(candidate, dict) for candidate in candidates):
        raise ValueError("manifest candidates must contain only objects")
    candidate_ids = [str(candidate.get("candidate_id") or "") for candidate in candidates]
    if not all(candidate_ids):
        raise ValueError("candidate_id values must be non-empty")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("duplicate candidate_id")
    sleeves = manifest.get("sleeves", {})
    if not isinstance(sleeves, dict):
        raise ValueError("manifest sleeves must be an object")
    sleeve_ids = set(sleeves)
    if not all(isinstance(sleeve_id, str) and sleeve_id for sleeve_id in sleeve_ids):
        raise ValueError("manifest sleeve IDs must be non-empty strings")
    unknown_sleeves = sorted(sleeve_ids - set(candidate_ids))
    if unknown_sleeves:
        raise ValueError("manifest sleeves lack candidates: " + ", ".join(unknown_sleeves))
    cost_profile, runtime_config, runtime_defaults = _cost_profile()
    lineage = _lineage(
        manifest,
        manifest_path,
        exchange=exchange,
        warmup_bars=warmup_bars,
        seed=seed,
        cost_profile=cost_profile,
        runtime_defaults=runtime_defaults,
        data_inventory=_data_inventory(data_root),
    )
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
    output_dir.mkdir(parents=True)
    repository = MarketDataRepository(str(data_root))
    symbol_limits = _symbol_limits_from_manifest(manifest)
    results: list[dict[str, Any]] = []
    receipt_disabled = lineage["universe"]["receipt"].get("disabled_candidates", {})
    if not isinstance(receipt_disabled, dict):
        raise ValueError("universe receipt disabled_candidates must be an object")
    disabled_candidate_ids = {
        str(candidate["candidate_id"])
        for candidate in candidates
        if candidate.get("enabled") is False
    }
    if set(receipt_disabled) != disabled_candidate_ids:
        raise ValueError("universe receipt disabled_candidates must match disabled candidates")
    for index, candidate in enumerate(candidates):
        fallback_id = (
            str(candidate.get("candidate_id") or f"candidate_{index:03d}")
            if isinstance(candidate, dict)
            else f"candidate_{index:03d}"
        )
        if isinstance(candidate, dict) and candidate.get("enabled") is False:
            reason = str(candidate.get("disabled_reason") or "disabled by manifest")
            expected = receipt_disabled.get(fallback_id)
            allowed_exclusion = _allowed_disabled_candidate(
                candidate_id=fallback_id,
                reason=reason,
                expected=expected,
                receipt=lineage["universe"]["receipt"],
                start=start,
                warmup_bars=warmup_bars,
                data_inventory_sha256=lineage["data_inventory"]["sha256"],
                candidate_symbols=candidate.get("symbols"),
                candidate_timeframe=candidate.get("timeframe")
                or candidate.get("strategy_timeframe"),
            )
            if not allowed_exclusion:
                raise ValueError(f"invalid disabled candidate exclusion: {fallback_id}")
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
            _publish_json(artifact, result)
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
                runtime_config=runtime_config,
                seed=seed,
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
        _publish_json(artifact, result)
        results.append({**result, "artifact": artifact.name})

    allocator = manifest.get("allocator")
    allocator = allocator if isinstance(allocator, dict) else {}
    min_sleeves = int(manifest.get("min_sleeves", allocator.get("min_sleeves", 1)))
    min_families = int(manifest.get("min_families", allocator.get("min_families", 1)))
    results_by_id = {row["candidate_id"]: row for row in results}
    reconciliation_complete = set(results_by_id) == set(candidate_ids) and len(results) == len(
        candidate_ids
    )
    sleeve_results = [results_by_id[sleeve_id] for sleeve_id in sorted(sleeve_ids)]
    passed_families = {
        str(row.get("family") or "") for row in sleeve_results if row["status"] == "pass"
    }
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
        reconciliation_complete
        and not any(row["status"] == "fail" for row in sleeve_results)
        and not (set(disallowed_skips) & sleeve_ids)
        and exclusion_contract_complete
        and sum(row["status"] == "pass" for row in sleeve_results) >= min_sleeves
        and len(passed_families) >= min_families
    )
    final_data_inventory = _data_inventory(data_root)
    if final_data_inventory != lineage["data_inventory"]:
        raise ValueError("data inventory changed during candidate evaluation")
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
        "candidate_reconciliation": {
            "manifest_candidate_count": len(candidate_ids),
            "result_candidate_count": len(results_by_id),
            "complete": reconciliation_complete,
        },
        "readiness": {
            "portfolio_ready": portfolio_ready,
            "min_sleeves": min_sleeves,
            "passing_sleeves": sum(row["status"] == "pass" for row in sleeve_results),
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
    summary_path = output_dir / "suite_results.json"
    _publish_json(summary_path, summary)
    if purpose == "locked_oos":
        return summary

    allocation_input = deepcopy(manifest)
    sleeves = allocation_input.get("sleeves")
    if isinstance(sleeves, dict):
        results_by_id = {row["candidate_id"]: row for row in results}
        for sleeve_id, sleeve in sleeves.items():
            if not isinstance(sleeve, dict):
                continue
            sleeve["source_artifact_id"] = "event_driven_candidate_evaluation"
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
                    "returns_source": {"splits": ["train", "validation"]},
                    "uses_locked_oos_for_selection": False,
                    "uses_locked_oos_for_sizing": False,
                    "returns_lineage": {
                        "artifact": "event_driven_candidate_evaluation",
                        "candidate_id": result["candidate_id"],
                        "stream": "daily UTC net returns over the caller-supplied selection window",
                        "uses_locked_oos_for_selection": False,
                        "uses_locked_oos_for_sizing": False,
                    },
                    "fit_start": result["return_timestamps"][0],
                    "fit_end": result["return_timestamps"][-1],
                    "as_of": end.isoformat(),
                    "apply_start": end.isoformat(),
                }
            )
    allocation_sleeves = allocation_input.get("sleeves", {})
    allocation_sleeves = allocation_sleeves if isinstance(allocation_sleeves, dict) else {}
    return_panel_sha256_by_sleeve = {
        sleeve_id: _materialized_return_panel_sha256(sleeve_id, sleeve)
        for sleeve_id, sleeve in allocation_sleeves.items()
        if sleeve.get("returns")
    }
    allocation_input["source_artifacts"] = [
        {
            "id": "event_driven_candidate_evaluation",
            "path": str(((published_output_dir or output_dir) / "suite_results.json").resolve()),
            "sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "max_age_hours": 8760,
            "ready": portfolio_ready,
            "portfolio_ready": portfolio_ready,
            "allowed_exclusions": summary["allowed_exclusions"],
            "exclusion_contract": summary["exclusion_contract"],
            "selection_period": summary["period"],
            "lineage": lineage,
            "frozen_at": summary["period"]["end"],
            "return_panel_sha256_by_sleeve": return_panel_sha256_by_sleeve,
        }
    ]
    _publish_json(output_dir / "allocation_input.json", allocation_input)
    return summary


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
    seed: int = 0,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"output target already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    staging_output = staging_parent / "run"
    try:
        summary = _run_suite_into(
            manifest_path,
            data_root,
            staging_output,
            exchange=exchange,
            start=start,
            end=end,
            purpose=purpose,
            warmup_bars=warmup_bars,
            selection_artifact=selection_artifact,
            seed=seed,
            published_output_dir=output_dir,
        )
        _rename_noreplace(staging_output, output_dir)
        return summary
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


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
    parser.add_argument("--seed", type=int, default=0)
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
        seed=args.seed,
    )
    raise SystemExit(0 if summary["readiness"]["portfolio_ready"] else 1)


if __name__ == "__main__":
    main()
