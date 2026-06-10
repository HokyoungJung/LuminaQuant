"""Sweep cost-aware cadence variants for leveraged/current portfolio modes.

The harness keeps the Python-facing strategy/backtest API unchanged and runs the
same event-driven live-equivalent path used by promotion revalidation.  It first
screens all eligible modes on validation, then spends full train/OOS backtests
only on validation survivors.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import hashlib
import importlib.util
import json
import math
import os
import resource
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from time import perf_counter
from typing import Any

import polars as pl

from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.chunked_runner import (
    _capture_backtest_state,
    _restore_backtest_state,
    iter_chunk_windows,
    run_backtest_chunked,
)
from lumina_quant.backtesting.data_windowed_parquet import HistoricParquetWindowedDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.compute.ohlcv_loader import normalize_ohlcv_frame
from lumina_quant.configuration import BacktestConfigView, get_default_runtime_config
_lq_rt = get_default_runtime_config()
BacktestConfig = BacktestConfigView(_lq_rt)
BaseConfig = BacktestConfigView(_lq_rt)
from lumina_quant.data.native_raw_first_backend import raw_first_backend_diagnostics
from lumina_quant.market_data import load_data_dict_from_parquet
from lumina_quant.strategies.artifact_portfolio_mode import (
    ArtifactPortfolioModeStrategy,
    PortfolioModeDefinition,
    resolve_portfolio_mode_definition,
    supported_portfolio_modes,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "var"
    / "reports"
    / "profit_moonshot_20260501"
    / "current_tail_20260506"
    / "cadence_sweep"
)
CADENCE_KEYS = ("rebalance_bars", "evaluation_cadence_bars")
EXPOSURE_KEYS = ("gross_exposure", "target_allocation", "max_symbol_exposure_pct")
DEFAULT_CADENCE_GRID = (1, 5, 15, 30, 60, 120, 240, 360, 720, 1440)
MIN_USEFUL_ALPHA_OOS_TOTAL_RETURN = 0.008284
MAX_USEFUL_ALPHA_OOS_MDD = 0.001778
MIN_USEFUL_ALPHA_OOS_SHARPE = 1.0
METRIC_DIAGNOSTIC_KEYS = (
    "raw_total_return",
    "raw_max_drawdown",
    "min_equity",
    "raw_final_equity",
    "equity_breach_count",
    "equity_breach_observed",
)
FrozenRows = tuple[tuple[int, float, float, float, float, float], ...]

_REVALIDATE_SPEC = importlib.util.spec_from_file_location(
    "profit_moonshot_revalidate_helpers",
    Path(__file__).resolve().parent / "revalidate_live_equivalent_candidates.py",
)
if _REVALIDATE_SPEC is None or _REVALIDATE_SPEC.loader is None:
    raise RuntimeError("failed to load live-equivalent revalidation helpers")
_REVALIDATE = importlib.util.module_from_spec(_REVALIDATE_SPEC)
sys.modules[_REVALIDATE_SPEC.name] = _REVALIDATE
_REVALIDATE_SPEC.loader.exec_module(_REVALIDATE)


@dataclass(frozen=True, slots=True)
class SweepSplit:
    name: str
    start: date
    end: date

    def as_payload(self) -> dict[str, str]:
        return {
            "name": self.name,
            "start": self.start.isoformat(),
            "end_inclusive": self.end.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class CadenceCandidate:
    candidate_id: str
    base_mode: str
    variant: str
    cadence_bars: int | None
    overrides: dict[str, Any]
    definition_hash: str
    changed_components: int
    native_cadences: tuple[int, ...]
    weighted_gross_exposure: float
    weighted_target_allocation: float
    max_order_value_sum: float

    def as_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["native_cadences"] = list(self.native_cadences)
        return payload


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _rss_mib() -> float:
    # Linux ru_maxrss is KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _to_epoch_ms(value: Any) -> int:
    if isinstance(value, (int, float)):
        numeric = int(float(value))
        return numeric * 1000 if abs(numeric) < 100_000_000_000 else numeric
    ts_fn = getattr(value, "timestamp", None)
    if callable(ts_fn):
        return int(float(ts_fn()) * 1000)
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.astimezone(UTC).timestamp() * 1000)


def _freeze_ohlcv_frame(
    frame: pl.DataFrame | FrozenRows | list[tuple[Any, ...]] | tuple[tuple[Any, ...], ...],
    *,
    start_date: Any,
    end_date: Any,
) -> FrozenRows:
    if isinstance(frame, (list, tuple)):
        return tuple(
            (
                _to_epoch_ms(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
            )
            for row in frame
            if len(row) >= 6
        )
    normalized = normalize_ohlcv_frame(frame, start_date=start_date, end_date=end_date)
    if normalized is None or normalized.is_empty():
        return ()
    return tuple(
        (
            _to_epoch_ms(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
        )
        for row in normalized.iter_rows(named=False)
    )


class RawFirstChunkCache:
    """Exact raw-first chunk cache that removes repeated parquet load/row-freeze work."""

    def __init__(
        self,
        *,
        market_root: Path,
        exchange: str,
        timeframe: str,
        chunk_days: int,
        max_entries: int = 96,
    ) -> None:
        self.market_root = Path(market_root)
        self.exchange = str(exchange)
        self.timeframe = str(timeframe)
        self.chunk_days = max(1, int(chunk_days))
        self.max_entries = max(1, int(max_entries))
        self._cache: OrderedDict[tuple[str, str, str], FrozenRows] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.loads = 0
        self.load_seconds = 0.0
        self.freeze_seconds = 0.0

    def _key(self, symbol: str, chunk_start: datetime, chunk_end: datetime) -> tuple[str, str, str]:
        return (str(symbol), chunk_start.isoformat(), chunk_end.isoformat())

    def _store(self, key: tuple[str, str, str], rows: FrozenRows) -> None:
        self._cache[key] = rows
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def load(
        self, *, symbols: list[str], chunk_start: datetime, chunk_end: datetime
    ) -> dict[str, FrozenRows]:
        out: dict[str, FrozenRows] = {}
        missing: list[str] = []
        for symbol in list(symbols or []):
            key = self._key(symbol, chunk_start, chunk_end)
            cached = self._cache.get(key)
            if cached is None:
                missing.append(symbol)
                self.misses += 1
                continue
            self.hits += 1
            self._cache.move_to_end(key)
            out[str(symbol)] = cached

        if missing:
            started_load = perf_counter()
            raw = load_data_dict_from_parquet(
                str(self.market_root),
                exchange=self.exchange,
                symbol_list=missing,
                timeframe=self.timeframe,
                start_date=chunk_start,
                end_date=chunk_end,
                chunk_days=self.chunk_days,
                warmup_bars=0,
                data_mode="raw-first",
                staleness_threshold_seconds=None,
            )
            self.load_seconds += perf_counter() - started_load
            self.loads += 1
            started_freeze = perf_counter()
            for symbol, frame in raw.items():
                rows = _freeze_ohlcv_frame(frame, start_date=chunk_start, end_date=chunk_end)
                key = self._key(str(symbol), chunk_start, chunk_end)
                self._store(key, rows)
                out[str(symbol)] = rows
            self.freeze_seconds += perf_counter() - started_freeze
        return out

    def diagnostics(self) -> dict[str, Any]:
        return {
            "entries": len(self._cache),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "loads": int(self.loads),
            "load_seconds": float(self.load_seconds),
            "freeze_seconds": float(self.freeze_seconds),
            "max_entries": int(self.max_entries),
        }

    def clear(self) -> None:
        self._cache.clear()


def _trim_process_memory() -> None:
    """Return freed Python/native pages to the OS when supported.

    Exactness is unaffected: this only runs after an independent split/batch has
    already been checkpointed.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        return


def _date_from_iso(token: str, fallback: date) -> date:
    raw = str(token or "").strip()
    if not raw:
        return fallback
    return datetime.fromisoformat(raw).date()


def _split_windows(oos_end: date) -> dict[str, SweepSplit]:
    return {
        "train": SweepSplit("train", date(2025, 1, 1), date(2025, 12, 31)),
        "val": SweepSplit("val", date(2026, 1, 1), date(2026, 2, 28)),
        "oos": SweepSplit("oos", date(2026, 3, 1), oos_end),
    }


def _split_to_datetimes(split: SweepSplit) -> tuple[datetime, datetime]:
    return (
        datetime.combine(split.start, time.min).replace(tzinfo=None),
        datetime.combine(split.end, time.max).replace(tzinfo=None),
    )


def _native_cadences(definition: PortfolioModeDefinition) -> tuple[int, ...]:
    values: set[int] = set()
    for component in definition.components:
        params = dict(component.params or {})
        for key in CADENCE_KEYS:
            if key not in params:
                continue
            try:
                values.add(max(1, int(params[key])))
            except Exception:
                pass
    return tuple(sorted(values))


def _weighted_exposure(definition: PortfolioModeDefinition) -> tuple[float, float, float]:
    gross = 0.0
    target = 0.0
    max_order_value_sum = 0.0
    for component in definition.components:
        params = dict(component.params or {})
        weight = float(component.weight)
        if "gross_exposure" in params:
            gross += _safe_float(params.get("gross_exposure")) * weight
        if "target_allocation" in params:
            target += _safe_float(params.get("target_allocation")) * weight
        if "max_order_value" in params:
            max_order_value_sum += _safe_float(params.get("max_order_value")) * weight
    return gross, target, max_order_value_sum


def _has_exposure_and_cadence(definition: PortfolioModeDefinition) -> bool:
    has_exposure = False
    has_cadence = False
    for component in definition.components:
        params = dict(component.params or {})
        has_exposure = has_exposure or any(key in params for key in EXPOSURE_KEYS)
        has_cadence = has_cadence or any(key in params for key in CADENCE_KEYS)
    return bool(has_exposure and has_cadence)


def _candidate_mode_filter(mode: str, include_regex_tokens: tuple[str, ...]) -> bool:
    if not include_regex_tokens:
        return True
    return any(token in mode for token in include_regex_tokens)


def _definition_payload(
    definition: PortfolioModeDefinition,
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    components: list[dict[str, Any]] = []
    for idx, component in enumerate(definition.components):
        params = dict(component.params or {})
        if isinstance(overrides, dict):
            direct = dict(overrides.get(component.component_id) or {})
            indexed = dict(overrides.get(str(idx)) or {})
            klass = dict(overrides.get(component.strategy_class) or {})
            wildcard = dict(overrides.get("__all__") or overrides.get("*") or {})
            params.update(wildcard)
            params.update(klass)
            params.update(direct)
            params.update(indexed)
        components.append(
            {
                "component_id": component.component_id,
                "strategy_class": component.strategy_class,
                "symbols": list(component.symbols),
                "params": params,
                "weight": float(component.weight),
            }
        )
    return {
        "portfolio_mode": definition.portfolio_mode,
        "cash_weight": float(definition.cash_weight),
        "components": components,
        "watch_symbols": list(definition.watch_symbols),
    }


def _definition_hash(definition: PortfolioModeDefinition, overrides: dict[str, Any] | None) -> str:
    encoded = json.dumps(
        _definition_payload(definition, overrides),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _cadence_overrides(
    definition: PortfolioModeDefinition,
    *,
    cadence_bars: int | None,
) -> tuple[dict[str, Any], int]:
    if cadence_bars is None:
        return {}, 0
    overrides: dict[str, Any] = {}
    changed = 0
    cadence = max(1, int(cadence_bars))
    for component in definition.components:
        params = dict(component.params or {})
        component_override = {
            key: cadence
            for key in CADENCE_KEYS
            if key in params and max(1, int(params.get(key) or 1)) != cadence
        }
        if component_override:
            overrides[component.component_id] = component_override
            changed += 1
    return overrides, changed


def _build_candidates(
    *,
    modes: list[str],
    cadence_grid: tuple[int, ...],
) -> list[CadenceCandidate]:
    candidates: list[CadenceCandidate] = []
    seen: set[str] = set()
    for mode in modes:
        definition = resolve_portfolio_mode_definition(mode)
        if not _has_exposure_and_cadence(definition):
            continue
        gross, target, max_order_value_sum = _weighted_exposure(definition)
        native_cadences = _native_cadences(definition)
        grid = tuple(sorted(set(cadence_grid).union(native_cadences)))
        for cadence in (None, *grid):
            variant = "native" if cadence is None else f"cadence_{int(cadence)}b"
            overrides, changed = _cadence_overrides(definition, cadence_bars=cadence)
            if cadence is not None and changed == 0:
                continue
            definition_hash = _definition_hash(definition, overrides)
            if definition_hash in seen:
                continue
            seen.add(definition_hash)
            candidates.append(
                CadenceCandidate(
                    candidate_id=f"{mode}__{variant}",
                    base_mode=mode,
                    variant=variant,
                    cadence_bars=int(cadence) if cadence is not None else None,
                    overrides=overrides,
                    definition_hash=definition_hash,
                    changed_components=changed,
                    native_cadences=native_cadences,
                    weighted_gross_exposure=float(gross),
                    weighted_target_allocation=float(target),
                    max_order_value_sum=float(max_order_value_sum),
                )
            )
    return candidates


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"artifact_kind": "profit_moonshot_cadence_sweep_checkpoint", "runs": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"artifact_kind": "profit_moonshot_cadence_sweep_checkpoint", "runs": {}}
    if not isinstance(payload.get("runs"), dict):
        payload["runs"] = {}
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _checkpoint_get(
    checkpoint: dict[str, Any], *, candidate_id: str, split: str
) -> dict[str, Any] | None:
    runs = checkpoint.get("runs")
    if not isinstance(runs, dict):
        return None
    by_candidate = runs.get(candidate_id)
    if not isinstance(by_candidate, dict):
        return None
    row = by_candidate.get(split)
    return dict(row) if isinstance(row, dict) else None


def _checkpoint_put(
    checkpoint: dict[str, Any],
    path: Path,
    *,
    candidate_id: str,
    split: str,
    result: dict[str, Any],
) -> None:
    runs = checkpoint.setdefault("runs", {})
    if not isinstance(runs, dict):
        checkpoint["runs"] = runs = {}
    by_candidate = runs.setdefault(candidate_id, {})
    if not isinstance(by_candidate, dict):
        runs[candidate_id] = by_candidate = {}
    by_candidate[split] = dict(result)
    checkpoint["updated_at"] = _utc_now_iso()
    _write_json(path, checkpoint)


def _empty_metrics() -> dict[str, float]:
    return {
        "total_return": 0.0,
        "cagr": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "max_drawdown": 0.0,
        "volatility": 0.0,
    }


def _metrics_from_equity_totals(values: list[float]) -> dict[str, float]:
    metrics = _REVALIDATE._metrics_from_equity_totals(
        values,
        periods=int(getattr(BacktestConfig, "ANNUAL_PERIODS", 252)),
    )
    return _normalize_metrics_payload(metrics)


def _normalize_metrics_payload(
    metrics: dict[str, Any],
    *,
    final_equity: Any | None = None,
) -> dict[str, Any]:
    """Cap report-facing metrics after an account/equity breach.

    Older checkpoints may contain raw negative-equity arithmetic (for example
    total return below -100% and drawdown above 100%).  Preserve those raw values
    in diagnostic fields but never let them appear as ordinary performance.
    """
    source = dict(metrics or {})
    out: dict[str, Any] = {key: _safe_float(source.get(key), 0.0) for key in _empty_metrics()}
    raw_total_return = _safe_float(
        source.get("raw_total_return", out["total_return"]), out["total_return"]
    )
    raw_max_drawdown = _safe_float(
        source.get("raw_max_drawdown", out["max_drawdown"]), out["max_drawdown"]
    )
    min_equity = _safe_float(source.get("min_equity"), 0.0)
    raw_final_equity = _safe_float(source.get("raw_final_equity", final_equity or 0.0), 0.0)
    breach_count = _safe_float(source.get("equity_breach_count"), 0.0)
    final_value = (
        _safe_float(final_equity, float("nan")) if final_equity is not None else float("nan")
    )
    final_breach = bool(
        math.isfinite(final_value)
        and final_value <= 0.0
        and (abs(raw_total_return) > 1e-12 or abs(raw_max_drawdown) > 1e-12)
    )
    breach = bool(
        source.get("equity_breach_observed")
        or raw_total_return <= -1.0
        or raw_max_drawdown > 1.0
        or breach_count > 0.0
        or final_breach
    )
    if breach:
        out.update(
            {
                "total_return": -1.0,
                "cagr": -1.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "calmar": -1.0,
                "max_drawdown": 1.0,
                "volatility": 0.0,
            }
        )
    out.update(
        {
            "raw_total_return": raw_total_return,
            "raw_max_drawdown": raw_max_drawdown,
            "min_equity": min_equity,
            "raw_final_equity": raw_final_equity,
            "equity_breach_count": float(breach_count),
            "equity_breach_observed": bool(breach),
        }
    )
    return out


def _run_split(
    *,
    candidate: CadenceCandidate,
    split: SweepSplit,
    market_root: Path,
    exchange: str,
    timeframe: str,
    chunk_days: int,
    backtest_poll_seconds: int,
    backtest_window_seconds: int,
    data_cache: RawFirstChunkCache | None = None,
) -> dict[str, Any]:
    definition = resolve_portfolio_mode_definition(candidate.base_mode)
    symbols = list(definition.symbols)
    start_dt, end_dt = _split_to_datetimes(split)
    load_seconds = 0.0

    def _loader(chunk_start: datetime, chunk_end: datetime) -> dict[str, Any]:
        nonlocal load_seconds
        start = perf_counter()
        try:
            if data_cache is not None:
                return data_cache.load(
                    symbols=symbols,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                )
            return load_data_dict_from_parquet(
                str(market_root),
                exchange=str(exchange),
                symbol_list=symbols,
                timeframe=str(timeframe),
                start_date=chunk_start,
                end_date=chunk_end,
                chunk_days=max(1, int(chunk_days)),
                warmup_bars=0,
                data_mode="raw-first",
                staleness_threshold_seconds=None,
            )
        finally:
            load_seconds += perf_counter() - start

    started = perf_counter()
    backtest = run_backtest_chunked(
        csv_dir="data",
        symbol_list=symbols,
        start_date=start_dt,
        end_date=end_dt,
        strategy_cls=ArtifactPortfolioModeStrategy,
        strategy_params={
            "portfolio_mode": candidate.base_mode,
            "component_param_overrides": candidate.overrides,
        },
        data_loader=_loader,
        chunk_days=max(1, int(chunk_days)),
        strategy_timeframe=str(BaseConfig.TIMEFRAME),
        data_handler_cls=HistoricParquetWindowedDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        backtest_mode="windowed",
        data_handler_kwargs={
            "backtest_poll_seconds": max(1, int(backtest_poll_seconds)),
            "backtest_window_seconds": max(1, int(backtest_window_seconds)),
        },
        record_history=False,
        track_metrics=True,
        record_trades=False,
    )
    wall_seconds = perf_counter() - started
    metrics = _metrics_from_equity_totals(
        [float(item) for item in list(getattr(backtest.portfolio, "_metric_totals", []) or [])]
    )
    return {
        "candidate_id": candidate.candidate_id,
        "base_mode": candidate.base_mode,
        "variant": candidate.variant,
        "split": split.name,
        "status": "completed",
        "metrics": metrics,
        "trade_count": int(getattr(backtest.portfolio, "trade_count", 0)),
        "liquidation_count": len(getattr(backtest.portfolio, "liquidation_events", []) or []),
        "final_equity": _safe_float(
            dict(getattr(backtest.portfolio, "current_holdings", {}) or {}).get("total"),
            0.0,
        ),
        "wall_seconds": float(wall_seconds),
        "load_seconds": float(load_seconds),
        "engine_seconds": float(max(0.0, wall_seconds - load_seconds)),
        "rss_mib": _rss_mib(),
    }


def _candidate_symbols(candidate: CadenceCandidate) -> list[str]:
    return list(resolve_portfolio_mode_definition(candidate.base_mode).symbols)


def _run_split_batch(
    *,
    candidates: list[CadenceCandidate],
    split: SweepSplit,
    market_root: Path,
    exchange: str,
    timeframe: str,
    chunk_days: int,
    backtest_poll_seconds: int,
    backtest_window_seconds: int,
    data_cache: RawFirstChunkCache,
) -> dict[str, dict[str, Any]]:
    """Run same-symbol candidates interleaved by chunk to reuse exact raw data."""
    if not candidates:
        return {}
    symbols = _candidate_symbols(candidates[0])
    start_dt, end_dt = _split_to_datetimes(split)
    carries: dict[str, dict[str, Any]] = {candidate.candidate_id: {} for candidate in candidates}
    engine_seconds: dict[str, float] = {candidate.candidate_id: 0.0 for candidate in candidates}
    load_seconds = 0.0
    started = perf_counter()
    any_chunk = False

    for chunk_start, chunk_end in iter_chunk_windows(
        start_date=start_dt,
        end_date=end_dt,
        chunk_days=max(1, int(chunk_days)),
    ):
        load_started = perf_counter()
        chunk_data = data_cache.load(symbols=symbols, chunk_start=chunk_start, chunk_end=chunk_end)
        load_seconds += perf_counter() - load_started
        if not chunk_data:
            continue
        any_chunk = True

        for candidate in candidates:
            run_started = perf_counter()
            backtest = Backtest(
                csv_dir="data",
                symbol_list=symbols,
                start_date=chunk_start,
                end_date=chunk_end,
                data_handler_cls=HistoricParquetWindowedDataHandler,
                execution_handler_cls=SimulatedExecutionHandler,
                portfolio_cls=Portfolio,
                strategy_cls=ArtifactPortfolioModeStrategy,
                strategy_params={
                    "portfolio_mode": candidate.base_mode,
                    "component_param_overrides": candidate.overrides,
                },
                data_dict=chunk_data,
                data_handler_kwargs={
                    "backtest_poll_seconds": max(1, int(backtest_poll_seconds)),
                    "backtest_window_seconds": max(1, int(backtest_window_seconds)),
                },
                record_history=False,
                track_metrics=True,
                record_trades=False,
                strategy_timeframe=str(BaseConfig.TIMEFRAME),
            )
            carry = carries.get(candidate.candidate_id) or {}
            if carry:
                _restore_backtest_state(backtest, carry)
            backtest.simulate_trading(output=False)
            carries[candidate.candidate_id] = _capture_backtest_state(
                backtest,
                record_history=False,
                track_metrics=True,
                record_trades=False,
            )
            engine_seconds[candidate.candidate_id] += perf_counter() - run_started

    batch_wall = perf_counter() - started
    out: dict[str, dict[str, Any]] = {}
    per_candidate_load = float(load_seconds) / float(max(1, len(candidates)))
    for candidate in candidates:
        carry = carries.get(candidate.candidate_id) or {}
        if not any_chunk or not carry:
            out[candidate.candidate_id] = {
                "candidate_id": candidate.candidate_id,
                "base_mode": candidate.base_mode,
                "variant": candidate.variant,
                "split": split.name,
                "status": "skipped_empty_data",
                "metrics": _empty_metrics(),
                "trade_count": 0,
                "liquidation_count": 0,
                "final_equity": 0.0,
                "wall_seconds": 0.0,
                "load_seconds": 0.0,
                "engine_seconds": 0.0,
                "rss_mib": _rss_mib(),
            }
            continue
        totals = [float(item) for item in list(carry.get("metric_totals") or [])]
        portfolio_state = dict(carry.get("portfolio_state") or {})
        holdings = dict(portfolio_state.get("holdings") or {})
        out[candidate.candidate_id] = {
            "candidate_id": candidate.candidate_id,
            "base_mode": candidate.base_mode,
            "variant": candidate.variant,
            "split": split.name,
            "status": "completed",
            "metrics": _metrics_from_equity_totals(totals),
            "trade_count": int(carry.get("trade_count") or portfolio_state.get("trade_count") or 0),
            "liquidation_count": len(list(portfolio_state.get("liquidation_events") or [])),
            "final_equity": _safe_float(holdings.get("total"), 0.0),
            "wall_seconds": float(engine_seconds[candidate.candidate_id] + per_candidate_load),
            "batch_wall_seconds": float(batch_wall),
            "load_seconds": float(per_candidate_load),
            "shared_load_seconds": float(load_seconds),
            "engine_seconds": float(engine_seconds[candidate.candidate_id]),
            "rss_mib": _rss_mib(),
        }
    return out


def _screen_pass(result: dict[str, Any]) -> bool:
    metrics = _normalize_metrics_payload(
        dict(result.get("metrics") or {}),
        final_equity=result.get("final_equity")
        if str(result.get("status") or "") == "completed"
        else None,
    )
    return bool(
        str(result.get("status") or "") == "completed"
        and int(result.get("trade_count") or 0) >= 3
        and int(result.get("liquidation_count") or 0) == 0
        and not bool(metrics.get("equity_breach_observed"))
        and _safe_float(metrics.get("total_return")) > 0.0
        and _safe_float(metrics.get("sharpe")) > 0.0
        and _safe_float(metrics.get("max_drawdown")) <= 0.05
    )


def _full_gate(full_runs: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    split_metrics = {
        split_name: _normalize_metrics_payload(
            dict(dict(full_runs.get(split_name) or {}).get("metrics") or {}),
            final_equity=dict(full_runs.get(split_name) or {}).get("final_equity")
            if str(dict(full_runs.get(split_name) or {}).get("status") or "") == "completed"
            else None,
        )
        for split_name in ("train", "val", "oos")
    }
    train = split_metrics["train"]
    val = split_metrics["val"]
    oos = split_metrics["oos"]
    if _safe_float(train.get("total_return")) <= 0.0:
        reasons.append("train_total_return_not_positive")
    if _safe_float(val.get("total_return")) <= 0.0:
        reasons.append("val_total_return_not_positive")
    if _safe_float(oos.get("total_return")) <= MIN_USEFUL_ALPHA_OOS_TOTAL_RETURN:
        reasons.append("oos_return_not_above_0.8284pct_incumbent")
    if _safe_float(oos.get("max_drawdown")) >= MAX_USEFUL_ALPHA_OOS_MDD:
        reasons.append("oos_mdd_not_below_funding_guard_shadow")
    if _safe_float(oos.get("sharpe")) <= MIN_USEFUL_ALPHA_OOS_SHARPE:
        reasons.append("oos_sharpe_not_above_1.0_success_target")
    for split_name, split_result in full_runs.items():
        metrics = split_metrics.get(split_name) or _normalize_metrics_payload(
            dict(split_result.get("metrics") or {}),
            final_equity=split_result.get("final_equity")
            if str(split_result.get("status") or "") == "completed"
            else None,
        )
        if bool(metrics.get("equity_breach_observed")):
            reasons.append(f"{split_name}_equity_breach_observed")
        if int(split_result.get("trade_count") or 0) <= 0:
            reasons.append(f"{split_name}_trade_count_zero")
        if int(split_result.get("liquidation_count") or 0) > 0:
            reasons.append(f"{split_name}_liquidation_observed")
    return not reasons, reasons


def _candidate_row(
    candidate: CadenceCandidate,
    split_results: dict[str, dict[str, Any]],
    *,
    full_gate_pass: bool | None = None,
    full_gate_reasons: list[str] | None = None,
) -> dict[str, Any]:
    row = candidate.as_payload()
    row.update(
        {
            "full_gate_pass": full_gate_pass,
            "full_gate_reasons": ";".join(full_gate_reasons or []),
        }
    )
    for split_name in ("train", "val", "oos"):
        result = dict(split_results.get(split_name) or {})
        metrics = _normalize_metrics_payload(
            dict(result.get("metrics") or {}),
            final_equity=result.get("final_equity")
            if str(result.get("status") or "") == "completed"
            else None,
        )
        row[f"{split_name}_status"] = str(result.get("status") or "")
        row[f"{split_name}_trade_count"] = int(result.get("trade_count") or 0)
        row[f"{split_name}_liquidation_count"] = int(result.get("liquidation_count") or 0)
        row[f"{split_name}_wall_seconds"] = _safe_float(result.get("wall_seconds"))
        row[f"{split_name}_final_equity"] = _safe_float(result.get("final_equity"))
        for key in _empty_metrics():
            row[f"{split_name}_{key}"] = _safe_float(metrics.get(key))
        for key in METRIC_DIAGNOSTIC_KEYS:
            if key == "equity_breach_observed":
                row[f"{split_name}_{key}"] = bool(metrics.get(key))
            else:
                row[f"{split_name}_{key}"] = _safe_float(metrics.get(key))
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fmt_pct_with_breach(row: dict[str, Any], split_name: str, metric_name: str) -> str:
    value = _safe_float(row.get(f"{split_name}_{metric_name}"))
    marker = "*" if bool(row.get(f"{split_name}_equity_breach_observed")) else ""
    return f"{value:+.4%}{marker}"


def _build_markdown(payload: dict[str, Any]) -> str:
    survivors = list(payload.get("full_results") or [])
    screen_rows = list(payload.get("screen_results") or [])
    best_full = survivors[:8]
    lines = [
        "# Profit moonshot leverage/cadence sweep",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- mode_count: `{payload.get('mode_count')}`",
        f"- candidate_count: `{payload.get('candidate_count')}`",
        f"- validation_screened: `{len(screen_rows)}`",
        f"- full_survivors_tested: `{len(survivors)}`",
        f"- max_rss_mib: `{float(payload.get('max_rss_mib') or 0.0):.2f}`",
        (
            "- cost model: event-driven `SimulatedExecutionHandler` with "
            f"taker_fee=`{payload['cost_model']['taker_fee_rate']}`, "
            f"slippage=`{payload['cost_model']['slippage_rate']}`, "
            f"spread=`{payload['cost_model']['spread_rate']}`."
        ),
        "- exposure policy: cadence-only overrides; gross/target allocation and max order caps are not increased.",
        (
            "- bankruptcy/account-breach policy: report-facing return/MDD are capped at "
            "`-100%`/`100%` when equity reaches non-positive territory; raw arithmetic is "
            "preserved in JSON/CSV diagnostic fields. `*` marks such capped rows."
        ),
        "",
        "## Full survivor results",
        "",
        "| candidate | gate | train ret | train MDD | val ret | OOS ret | OOS Sharpe | OOS MDD | breach | reasons |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    if not best_full:
        lines.append("| - | - | - | - | - | - | - | - | - | no validation survivor full-tested |")
    for row in best_full:
        lines.append(
            "| `{candidate_id}` | `{gate}` | {train} | {train_mdd} | {val} | {oos} | "
            "{sharpe:.6f} | {mdd} | `{breach}` | `{reasons}` |".format(
                candidate_id=row.get("candidate_id"),
                gate=bool(row.get("full_gate_pass")),
                train=_fmt_pct_with_breach(row, "train", "total_return"),
                train_mdd=_fmt_pct_with_breach(row, "train", "max_drawdown"),
                val=_fmt_pct_with_breach(row, "val", "total_return"),
                oos=_fmt_pct_with_breach(row, "oos", "total_return"),
                sharpe=_safe_float(row.get("oos_sharpe")),
                mdd=_fmt_pct_with_breach(row, "oos", "max_drawdown"),
                breach=",".join(
                    split
                    for split in ("train", "val", "oos")
                    if bool(row.get(f"{split}_equity_breach_observed"))
                )
                or "-",
                reasons=row.get("full_gate_reasons") or "",
            )
        )
    lines.extend(
        [
            "",
            "## Top validation cadence screen",
            "",
            "| candidate | val ret | val Sharpe | val MDD | trades | native cadences |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in screen_rows[:20]:
        lines.append(
            "| `{candidate_id}` | {val:+.4%} | {sharpe:.6f} | {mdd:.4%} | {trades} | `{native}` |".format(
                candidate_id=row.get("candidate_id"),
                val=_safe_float(row.get("val_total_return")),
                sharpe=_safe_float(row.get("val_sharpe")),
                mdd=_safe_float(row.get("val_max_drawdown")),
                trades=int(row.get("val_trade_count") or 0),
                native=row.get("native_cadences") or [],
            )
        )
    lines.extend(
        [
            "",
            "## Bottleneck notes",
            "",
            f"- total_wall_seconds: `{float(payload.get('total_wall_seconds') or 0.0):.2f}`",
            f"- checkpointed_run_wall_seconds: `{float(payload.get('checkpointed_run_wall_seconds') or 0.0):.2f}`",
            f"- data_cache: `{payload.get('data_cache')}`",
            f"- native_backends: `{payload.get('native_backends')}`",
            f"- slowest_runs: `{payload.get('slowest_runs')}`",
            "- exactness note: optimized runs reuse frozen raw-first OHLCV rows but still execute the same live-equivalent strategy, portfolio, fill, fee, spread, slippage, and partial-fill engine path.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_int_grid(raw: str) -> tuple[int, ...]:
    values: set[int] = set()
    for item in str(raw or "").split(","):
        token = item.strip()
        if not token:
            continue
        values.add(max(1, int(token)))
    return tuple(sorted(values)) or DEFAULT_CADENCE_GRID


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--market-root", default=str(BaseConfig.MARKET_DATA_PARQUET_PATH))
    parser.add_argument("--exchange", default=str(BaseConfig.MARKET_DATA_EXCHANGE))
    parser.add_argument("--timeframe", default="1s")
    parser.add_argument("--chunk-days", type=int, default=1)
    parser.add_argument("--backtest-poll-seconds", type=int, default=20)
    parser.add_argument("--backtest-window-seconds", type=int, default=20)
    parser.add_argument("--oos-end-date", default="2026-05-04")
    parser.add_argument(
        "--screen-start-date",
        default="2026-02-01",
        help="Validation sub-window start for broad cadence screening.",
    )
    parser.add_argument(
        "--screen-end-date",
        default="2026-02-14",
        help="Validation sub-window end for broad cadence screening.",
    )
    parser.add_argument(
        "--cadence-grid", default=",".join(str(item) for item in DEFAULT_CADENCE_GRID)
    )
    parser.add_argument(
        "--mode-contains",
        default="profit_moonshot_,profit_reboot_,derivatives_flow_squeeze_mode",
        help="Comma-separated substrings; empty means all supported modes.",
    )
    parser.add_argument("--portfolio-modes", default="")
    parser.add_argument("--max-modes", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--survivor-count", type=int, default=12)
    parser.add_argument("--data-cache-entries", type=int, default=128)
    parser.add_argument("--no-full-survivors", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    os.environ.setdefault("LQ_BACKTEST_SUPPRESS_PARTIAL_FILL_LOGS", "1")
    os.environ.setdefault("LQ_BACKTEST_SUPPRESS_CIRCUIT_BREAKER_LOGS", "1")
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "cadence_sweep_checkpoint.json"
    checkpoint = _load_checkpoint(checkpoint_path)
    started = perf_counter()

    oos_end = _date_from_iso(args.oos_end_date, date(2026, 5, 4))
    splits = _split_windows(oos_end)
    screen_split = SweepSplit(
        "val_screen",
        _date_from_iso(args.screen_start_date, date(2026, 2, 1)),
        _date_from_iso(args.screen_end_date, date(2026, 2, 14)),
    )
    cadence_grid = _parse_int_grid(args.cadence_grid)
    requested_modes = [
        item.strip() for item in str(args.portfolio_modes or "").split(",") if item.strip()
    ]
    include_tokens = tuple(
        item.strip() for item in str(args.mode_contains or "").split(",") if item.strip()
    )
    modes = requested_modes or [
        mode
        for mode in sorted(supported_portfolio_modes())
        if _candidate_mode_filter(mode, include_tokens)
    ]
    if int(args.max_modes) > 0:
        modes = modes[: int(args.max_modes)]

    candidates = _build_candidates(modes=modes, cadence_grid=cadence_grid)
    if int(args.max_candidates) > 0:
        candidates = candidates[: int(args.max_candidates)]

    coverage_payload: dict[str, Any] = {}
    eligible_candidates: list[CadenceCandidate] = []
    for candidate in candidates:
        if candidate.base_mode not in coverage_payload:
            preflight = _REVALIDATE._mode_preflight(
                mode=candidate.base_mode,
                market_root=Path(args.market_root),
                exchange=str(args.exchange),
                timeframe=str(args.timeframe),
                splits=[
                    _REVALIDATE.SplitWindow(split.name, split.start, split.end, "cadence_sweep")
                    for split in splits.values()
                ],
            )
            coverage_payload[candidate.base_mode] = preflight.as_payload()
        preflight_payload = dict(coverage_payload[candidate.base_mode])
        if str(preflight_payload.get("status")) == "ready_for_live_equivalent_backtest":
            eligible_candidates.append(candidate)

    screen_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []
    data_cache = RawFirstChunkCache(
        market_root=Path(args.market_root),
        exchange=str(args.exchange),
        timeframe=str(args.timeframe),
        chunk_days=max(1, int(args.chunk_days)),
        max_entries=max(1, int(args.data_cache_entries)),
    )
    for idx, candidate in enumerate(eligible_candidates, start=1):
        print(
            json.dumps(
                {
                    "event": "cadence_screen_start",
                    "idx": idx,
                    "total": len(eligible_candidates),
                    "candidate_id": candidate.candidate_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        cached = _checkpoint_get(
            checkpoint,
            candidate_id=candidate.candidate_id,
            split=screen_split.name,
        )
        if cached is None:
            cached = _run_split(
                candidate=candidate,
                split=screen_split,
                market_root=Path(args.market_root),
                exchange=str(args.exchange),
                timeframe=str(args.timeframe),
                chunk_days=max(1, int(args.chunk_days)),
                backtest_poll_seconds=max(1, int(args.backtest_poll_seconds)),
                backtest_window_seconds=max(1, int(args.backtest_window_seconds)),
                data_cache=data_cache,
            )
            _checkpoint_put(
                checkpoint,
                checkpoint_path,
                candidate_id=candidate.candidate_id,
                split=screen_split.name,
                result=cached,
            )
        all_runs.append(cached)
        screen_rows.append(_candidate_row(candidate, {"val": cached}))

    screen_rows.sort(
        key=lambda row: (
            _safe_float(row.get("val_total_return")),
            _safe_float(row.get("val_sharpe")),
            -_safe_float(row.get("val_max_drawdown")),
        ),
        reverse=True,
    )
    candidate_by_id = {candidate.candidate_id: candidate for candidate in eligible_candidates}
    survivors = [
        candidate_by_id[str(row["candidate_id"])]
        for row in screen_rows
        if str(row.get("candidate_id")) in candidate_by_id
        and _screen_pass(dict(checkpoint["runs"][str(row["candidate_id"])][screen_split.name]))
    ][: max(0, int(args.survivor_count))]

    if not bool(args.no_full_survivors):
        data_cache.clear()
        _trim_process_memory()
        full_results_by_candidate: dict[str, dict[str, dict[str, Any]]] = {
            candidate.candidate_id: {} for candidate in survivors
        }
        for split_name in ("train", "val", "oos"):
            pending_by_symbols: dict[tuple[str, ...], list[CadenceCandidate]] = {}
            for candidate in survivors:
                cached = _checkpoint_get(
                    checkpoint,
                    candidate_id=candidate.candidate_id,
                    split=split_name,
                )
                if cached is not None:
                    full_results_by_candidate[candidate.candidate_id][split_name] = cached
                    all_runs.append(cached)
                    continue
                pending_by_symbols.setdefault(tuple(_candidate_symbols(candidate)), []).append(
                    candidate
                )

            for group_idx, group in enumerate(pending_by_symbols.values(), start=1):
                print(
                    json.dumps(
                        {
                            "event": "cadence_full_batch_start",
                            "group_idx": group_idx,
                            "group_count": len(pending_by_symbols),
                            "candidate_count": len(group),
                            "split": split_name,
                            "symbols": _candidate_symbols(group[0]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                batch_results = _run_split_batch(
                    candidates=group,
                    split=splits[split_name],
                    market_root=Path(args.market_root),
                    exchange=str(args.exchange),
                    timeframe=str(args.timeframe),
                    chunk_days=max(1, int(args.chunk_days)),
                    backtest_poll_seconds=max(1, int(args.backtest_poll_seconds)),
                    backtest_window_seconds=max(1, int(args.backtest_window_seconds)),
                    data_cache=data_cache,
                )
                for candidate in group:
                    cached = batch_results[candidate.candidate_id]
                    _checkpoint_put(
                        checkpoint,
                        checkpoint_path,
                        candidate_id=candidate.candidate_id,
                        split=split_name,
                        result=cached,
                    )
                    full_results_by_candidate[candidate.candidate_id][split_name] = cached
                    all_runs.append(cached)
                data_cache.clear()
                _trim_process_memory()

        for candidate in survivors:
            split_results = full_results_by_candidate.get(candidate.candidate_id, {})
            gate_pass, gate_reasons = _full_gate(split_results)
            full_rows.append(
                _candidate_row(
                    candidate,
                    split_results,
                    full_gate_pass=gate_pass,
                    full_gate_reasons=gate_reasons,
                )
            )

    full_rows.sort(
        key=lambda row: (
            bool(row.get("full_gate_pass")),
            _safe_float(row.get("oos_total_return")),
            _safe_float(row.get("oos_sharpe")),
            -_safe_float(row.get("oos_max_drawdown")),
        ),
        reverse=True,
    )
    slowest_runs = sorted(
        [
            {
                "candidate_id": run.get("candidate_id"),
                "split": run.get("split"),
                "wall_seconds": _safe_float(run.get("wall_seconds")),
                "load_seconds": _safe_float(run.get("load_seconds")),
                "engine_seconds": _safe_float(run.get("engine_seconds")),
            }
            for run in all_runs
        ],
        key=lambda item: item["wall_seconds"],
        reverse=True,
    )[:8]
    max_observed_rss_mib = max(
        [_rss_mib(), *[_safe_float(run.get("rss_mib")) for run in all_runs]],
        default=_rss_mib(),
    )
    checkpointed_run_wall_seconds = sum(_safe_float(run.get("wall_seconds")) for run in all_runs)
    payload = {
        "artifact_kind": "profit_moonshot_cadence_sweep",
        "generated_at": _utc_now_iso(),
        "mode_count": len(modes),
        "eligible_mode_count": len({candidate.base_mode for candidate in eligible_candidates}),
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible_candidates),
        "cadence_grid": list(cadence_grid),
        "split_windows": {name: split.as_payload() for name, split in splits.items()},
        "screen_window": screen_split.as_payload(),
        "coverage": coverage_payload,
        "screen_results": screen_rows,
        "full_results": full_rows,
        "cost_model": {
            "taker_fee_rate": _safe_float(getattr(BacktestConfig, "TAKER_FEE_RATE", 0.0)),
            "slippage_rate": _safe_float(getattr(BacktestConfig, "SLIPPAGE_RATE", 0.0)),
            "spread_rate": _safe_float(getattr(BacktestConfig, "SPREAD_RATE", 0.0)),
        },
        "data_cache": data_cache.diagnostics(),
        "native_backends": {
            "raw_first": raw_first_backend_diagnostics(),
        },
        "slowest_runs": slowest_runs,
        "max_rss_mib": max_observed_rss_mib,
        "checkpointed_run_wall_seconds": checkpointed_run_wall_seconds,
        "total_wall_seconds": perf_counter() - started,
        "paths": {
            "json": str((output_dir / "profit_moonshot_cadence_sweep_latest.json").resolve()),
            "markdown": str((output_dir / "profit_moonshot_cadence_sweep_latest.md").resolve()),
            "screen_csv": str(
                (output_dir / "profit_moonshot_cadence_sweep_screen_latest.csv").resolve()
            ),
            "full_csv": str(
                (output_dir / "profit_moonshot_cadence_sweep_full_latest.csv").resolve()
            ),
            "checkpoint": str(checkpoint_path.resolve()),
        },
    }
    _write_json(output_dir / "profit_moonshot_cadence_sweep_latest.json", payload)
    (output_dir / "profit_moonshot_cadence_sweep_latest.md").write_text(
        _build_markdown(payload),
        encoding="utf-8",
    )
    _write_csv(output_dir / "profit_moonshot_cadence_sweep_screen_latest.csv", screen_rows)
    _write_csv(output_dir / "profit_moonshot_cadence_sweep_full_latest.csv", full_rows)
    print(json.dumps(payload["paths"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
