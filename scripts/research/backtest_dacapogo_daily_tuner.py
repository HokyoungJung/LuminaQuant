#!/usr/bin/env python3
"""Research-only structural/ML tuner for the Dacapogo daily breakout.

The module separates orders chosen at the UTC open from later fill labels. Minute
data is used only to label orders that the daily bar says could have filled.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

try:
    from scripts.research.backtest_dacapogo_daily_ml import _validate_panel
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from backtest_dacapogo_daily_ml import _validate_panel
from lumina_quant.research.run_card import (
    atomic_output_path,
    atomic_write_text,
    runtime_provenance,
)
from lumina_quant.strategies.dacapogo_daily_source import COST

LATEST_UPSTREAM_HEAD = "633ba5d6bc0c84a20696af6b2bf807cf55d21248"
SOURCE_FILE_SHA256 = "17516d9457540e978d4828620c99794df0617c154faead5acee9f0847b5fcd8e"
RANKER_FILE_SHA256 = "e338f296b75fbf0eb9a3c2e1181fd14d813c898993ddc9f6c285a621466c37a7"
RANKER_USED_FOR_SIGNAL = False
MULTIFREQ_FILE_SHA256 = "dc675e925a0a0ac7e0c8e49e8c943aa432514ac937ef352b30027a009897115c"
NESTED_CV_FILE_SHA256 = "0c6cbd411e046724c0595f118b7fd6939bdf4f2a80841745ad1ed2f4e379543f"
BREAKOUTS = (0.025, 0.03, 0.04)
UNIVERSE_TOPKS = (10, 15)
POSITION_CAPS = (5, 10, 15)
LEVERAGES = (1, 3, 5, 10, 20)
EXIT_PROFILES = {
    "source": (0.005, 0.008),
    "wide": (0.01, 0.015),
}
SCENARIOS = (
    ("close_exit_stop_first", "close", "stop_first"),
    ("close_exit_entry_last", "close", "tp_first"),
    ("tp_sl_stop_first", "tp_sl", "stop_first"),
    ("tp_sl_tp_first", "tp_sl", "tp_first"),
)
ADVERSE_SCENARIOS = ("close_exit_stop_first", "tp_sl_stop_first")
FEATURES = (
    "prev_ret",
    "prev_range",
    "log_prev_turnover",
    "turnover_change",
    "momentum_3",
    "momentum_7",
    "open_gap",
)
EVALUATION_COLUMNS = (
    "market",
    "date",
    "breakout",
    "exit_profile",
    "scenario",
    "leverage",
    "slot_return",
    "liquidated",
    "mark_liquidation_breach",
    "ambiguous_minute",
)


@dataclass(frozen=True)
class Fold:
    number: int
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    embargo: date
    oos_start: date
    oos_end: date


@dataclass(frozen=True)
class ModelPreset:
    name: str
    family: str
    params: dict[str, Any]


def _file_identity(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def build_daily_features(panel: pl.DataFrame) -> pl.DataFrame:
    """Return information available at the current UTC-day open, never later."""
    panel = panel.sort(["market", "date"])
    prev_value = pl.col("value").shift(1).over("market")
    prev2_value = pl.col("value").shift(2).over("market")
    return (
        panel.with_columns(
            prev_value.alias("prev_turnover"),
            (pl.col("close") / pl.col("open") - 1).shift(1).over("market").alias("prev_ret"),
            (pl.col("high") / pl.col("low") - 1).shift(1).over("market").alias("prev_range"),
            prev_value.log1p().alias("log_prev_turnover"),
            pl.when(prev2_value > 0)
            .then(prev_value / prev2_value - 1)
            .otherwise(0.0)
            .alias("turnover_change"),
            (pl.col("close").shift(1).over("market") / pl.col("close").shift(3).over("market") - 1)
            .fill_null(0.0)
            .alias("momentum_3"),
            (pl.col("close").shift(1).over("market") / pl.col("close").shift(7).over("market") - 1)
            .fill_null(0.0)
            .alias("momentum_7"),
            (pl.col("open") / pl.col("close").shift(1).over("market") - 1).alias("open_gap"),
        )
        .with_columns(
            pl.col("prev_turnover")
            .rank(method="ordinal", descending=True)
            .over("date")
            .alias("turnover_rank")
        )
        .select("market", "date", "open", "high", "turnover_rank", *FEATURES)
    )


def preopen_select(
    rows: pl.DataFrame,
    predictions: np.ndarray | list[float],
    *,
    universe_topk: int,
    position_cap: int,
    breakout: float,
) -> pl.DataFrame:
    """Rank at the UTC open before consulting current high, then mark eventual fills."""
    if position_cap > universe_topk:
        raise ValueError("position_cap must be <= universe_topk")
    scored = rows.with_columns(pl.Series("prediction", predictions)).filter(
        pl.col("turnover_rank") <= universe_topk
    )
    selected = (
        scored.sort(["date", "prediction", "market"], descending=[False, True, False])
        .group_by("date", maintain_order=True)
        .head(position_cap)
    )
    return selected.with_columns(
        (pl.col("high") >= pl.col("open") * (1.0 + breakout)).alias("filled"),
        (pl.col("open") * (1.0 + breakout)).alias("entry"),
    )


def expanded_union(panel: pl.DataFrame) -> pl.DataFrame:
    """Rows requiring exact minute labels: prior-turnover top 15 and 2.5% trigger."""
    rows = build_daily_features(panel)
    return rows.filter(
        (pl.col("turnover_rank") <= max(UNIVERSE_TOPKS))
        & (pl.col("high") >= pl.col("open") * (1 + min(BREAKOUTS)))
    ).select("market", "date", "open", "high", "turnover_rank")


def make_folds(start: date, end: date) -> list[Fold]:
    folds: list[Fold] = []
    oos_start = start + timedelta(days=211)
    while oos_start <= end:
        oos_end = min(end, oos_start + timedelta(days=29))
        if (oos_end - oos_start).days + 1 < 7:
            break
        validation_end = oos_start - timedelta(days=2)
        validation_start = validation_end - timedelta(days=29)
        folds.append(
            Fold(
                len(folds) + 1,
                start,
                validation_start - timedelta(days=1),
                validation_start,
                validation_end,
                oos_start - timedelta(days=1),
                oos_start,
                oos_end,
            )
        )
        oos_start += timedelta(days=30)
    return folds


def model_presets(*, include_lightgbm: bool = True) -> tuple[ModelPreset, ...]:
    presets = [ModelPreset("turnover", "turnover", {})]
    presets += [
        ModelPreset(f"ridge_{alpha:g}", "ridge", {"alpha": alpha}) for alpha in (0.1, 1.0, 10.0)
    ]
    presets += [
        ModelPreset(
            f"tree_depth{depth}",
            "tree",
            {"max_depth": depth, "min_samples_leaf": 20, "random_state": 0},
        )
        for depth in (3, 5)
    ]
    presets += [
        ModelPreset(
            "extra_trees_128_depth6",
            "extra_trees",
            {
                "n_estimators": 128,
                "max_depth": 6,
                "min_samples_leaf": 10,
                "max_features": "sqrt",
                "random_state": 0,
                "n_jobs": 1,
            },
        )
    ]
    presets += [
        ModelPreset(
            f"hist_gbdt_leaf{leaves}",
            "hist_gbdt",
            {
                "learning_rate": 0.05,
                "max_leaf_nodes": leaves,
                "max_iter": 96,
                "max_bins": 63,
                "l2_regularization": 1.0,
                "early_stopping": False,
                "random_state": 0,
            },
        )
        for leaves in (7, 15)
    ]
    if include_lightgbm:
        presets.append(
            ModelPreset(
                "lightgbm_small",
                "lightgbm",
                {
                    "n_estimators": 128,
                    "num_leaves": 15,
                    "max_depth": 5,
                    "learning_rate": 0.05,
                    "max_bin": 63,
                    "reg_lambda": 1.0,
                    "random_state": 0,
                    "n_jobs": 1,
                    "deterministic": True,
                    "force_col_wise": True,
                    "verbosity": -1,
                },
            )
        )
    return tuple(presets)


def _predict(
    preset: ModelPreset,
    train_x: np.ndarray,
    train_y: np.ndarray,
    predict_x: np.ndarray,
    *,
    turnover: np.ndarray,
) -> np.ndarray:
    if preset.family == "turnover":
        return turnover.astype(float)
    try:
        from threadpoolctl import threadpool_limits

        if preset.family == "lightgbm":
            from lightgbm import LGBMRegressor

            model = LGBMRegressor(**preset.params)
        else:
            from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.tree import DecisionTreeRegressor

            model = (
                make_pipeline(StandardScaler(), Ridge(**preset.params))
                if preset.family == "ridge"
                else {
                    "tree": DecisionTreeRegressor,
                    "extra_trees": ExtraTreesRegressor,
                    "hist_gbdt": HistGradientBoostingRegressor,
                }[preset.family](**preset.params)
            )
    except ImportError as exc:
        raise RuntimeError(
            "daily tuner requires the research-ml extra: uv sync --extra research-ml"
        ) from exc
    with threadpool_limits(limits=1):
        model.fit(train_x, train_y)
        return np.asarray(model.predict(predict_x), dtype=float)


def _atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    with atomic_output_path(path) as temporary:
        frame.write_parquet(temporary)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, default=str) + "\n")


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _content_identity(value: Any) -> dict[str, int | str]:
    encoded = json.dumps(value, default=str, sort_keys=True, separators=(",", ":")).encode()
    return {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def _local_trade_inputs(
    data_root: Path, exchange: str, symbol: str, days: list[date]
) -> list[dict[str, Any]]:
    base = data_root / f"exchange={exchange}" / f"symbol={symbol}" / "timeframe=1m"
    return [
        {"path": str(path.resolve()), **_file_identity(path)}
        for day in days
        for path in sorted((base / f"date={day}").glob("*.parquet"))
    ]


def _expected_symbol_rows(rows: pl.DataFrame) -> int:
    return sum(
        len(SCENARIOS) * len(LEVERAGES) * len(EXIT_PROFILES)
        for row in rows.iter_rows(named=True)
        for breakout in BREAKOUTS
        if float(row["high"]) >= float(row["open"]) * (1 + breakout)
    )


def _acquisition_workers(value: str | int) -> int:
    workers = int(value)
    if not 1 <= workers <= 8:
        raise argparse.ArgumentTypeError("acquisition workers must be between 1 and 8")
    return workers


def acquire_execution_candidates(
    panel: pl.DataFrame,
    *,
    cache_dir: Path,
    data_root: Path,
    exchange: str = "binance",
    source_module: Any | None = None,
    refresh: bool = False,
    workers: int = 3,
) -> pl.DataFrame:
    """Serialize the complete cache transaction across tuner processes."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / ".execution_candidates.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _acquire_execution_candidates(
            panel,
            cache_dir=cache_dir,
            data_root=data_root,
            exchange=exchange,
            source_module=source_module,
            refresh=refresh,
            workers=workers,
        )


def _acquire_execution_candidates(
    panel: pl.DataFrame,
    *,
    cache_dir: Path,
    data_root: Path,
    exchange: str = "binance",
    source_module: Any | None = None,
    refresh: bool = False,
    workers: int = 3,
) -> pl.DataFrame:
    """Acquire exact labels with atomic, validated per-symbol resume parts."""
    if exchange != "binance":
        raise ValueError("Dacapogo execution acquisition supports only exchange='binance'")
    workers = _acquisition_workers(workers)
    if source_module is None:
        try:
            source = importlib.import_module("scripts.research.backtest_dacapogo_daily_source")
        except ModuleNotFoundError as exc:
            if exc.name != "scripts":
                raise
            source = importlib.import_module("backtest_dacapogo_daily_source")
    else:
        source = source_module
    union = expanded_union(panel)
    lookup = {
        (str(row["market"]), row["date"]): row
        for row in panel.join(
            union.select("market", "date"), on=["market", "date"], how="inner"
        ).iter_rows(named=True)
    }
    data_root = data_root.resolve()
    symbols = sorted(union["market"].unique().to_list())
    if not symbols:
        raise ValueError("expanded execution union is empty")

    def acquire_symbol(symbol: str) -> tuple[str, Path, dict[str, Any]]:
        symbol_rows = union.filter(pl.col("market") == symbol).sort("date")
        part = cache_dir / "parts" / f"{symbol}.parquet"
        expected = _expected_symbol_rows(symbol_rows)
        source_file = getattr(source, "__file__", None)
        source_path = Path(str(source_file)) if source_file else None
        execution_model_file = getattr(
            getattr(source, "execution_model_module", None), "__file__", None
        )
        days = symbol_rows["date"].to_list()
        context = {
            "exchange": exchange,
            "data_root": str(data_root),
            "symbol": symbol,
            "rows": symbol_rows.to_dicts(),
            "breakouts": BREAKOUTS,
            "leverages": LEVERAGES,
            "profiles": EXIT_PROFILES,
            "scenarios": SCENARIOS,
            "cost": COST,
            "source": _file_identity(source_path) if source_path else "injected-test-source",
            "execution_model_code": (
                _file_identity(Path(str(execution_model_file)))
                if execution_model_file
                else "injected-test-source"
            ),
            "execution_model": (
                source._execution_assumptions()
                if hasattr(source, "_execution_assumptions")
                else "injected-test-source"
            ),
            "local_trade_inputs": _local_trade_inputs(data_root, exchange, symbol, days),
        }
        context = json.loads(json.dumps(context, default=str, sort_keys=True))
        manifest_path = part.with_suffix(".manifest.json")
        cached_manifest = (
            _read_json_dict(manifest_path)
            if not refresh and part.exists() and manifest_path.exists()
            else None
        )
        cached_key = (
            str(
                _content_identity(
                    {"context": context, "acquisition": cached_manifest.get("acquisition")}
                )["sha256"]
            )
            if cached_manifest is not None
            else None
        )
        if (
            cached_manifest is not None
            and cached_manifest.get("context") == context
            and cached_manifest.get("cache_key") == cached_key
            and cached_manifest.get("rows") == expected
            and cached_manifest.get("part") == _file_identity(part)
        ):
            return symbol, part, cached_manifest
        funding = source._fetch_funding_rates(symbol, min(days), max(days))
        acquired_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        day_inputs: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        for row in symbol_rows.iter_rows(named=True):
            day = row["date"]
            bars, mark_bars, minute_source = source._load_minute_day(
                data_root, exchange, symbol, day, lookup[(symbol, day)]
            )
            if len(bars) != 1_440 or len(mark_bars) != 1_440:
                raise ValueError(f"incomplete aligned minute day: {symbol} {day}")
            day_inputs.append(
                {
                    "date": str(day),
                    "minute_source": minute_source,
                    "trade_bars": _content_identity(bars),
                    "mark_bars": _content_identity(mark_bars),
                }
            )
            for breakout in BREAKOUTS:
                if float(row["high"]) < float(row["open"]) * (1 + breakout):
                    continue
                for profile, (stop_pct, take_profit_pct) in EXIT_PROFILES.items():
                    for scenario, variant, priority in SCENARIOS:
                        for leverage in LEVERAGES:
                            result = source._simulate_trade(
                                bars,
                                mark_bars,
                                entry_trigger=float(row["open"]) * (1 + breakout),
                                leverage=leverage,
                                variant=variant,
                                same_bar_priority=priority,
                                funding_rates=funding,
                                stop_pct=stop_pct,
                                take_profit_pct=take_profit_pct,
                                round_trip_cost=COST,
                            )
                            records.append(
                                {
                                    "market": symbol,
                                    "date": day,
                                    "breakout": breakout,
                                    "exit_profile": profile,
                                    "scenario": scenario,
                                    "leverage": leverage,
                                    "minute_source": minute_source,
                                    **result,
                                }
                            )
        acquisition = {
            "acquired_at": acquired_at,
            "funding": {
                "start": str(min(days)),
                "end": str(max(days)),
                "snapshot": _content_identity(funding),
            },
            "days": day_inputs,
        }
        cache_key = str(
            _content_identity({"context": context, "acquisition": acquisition})["sha256"]
        )
        frame = pl.DataFrame(records)
        if frame.height != expected:
            raise ValueError(f"execution candidate count mismatch for {symbol}")
        frame = frame.with_columns(pl.lit(cache_key).alias("cache_key"))
        _atomic_parquet(frame, part)
        cached_manifest = {
            "context": context,
            "acquisition": acquisition,
            "cache_key": cache_key,
            "rows": expected,
            "part": _file_identity(part),
        }
        _atomic_json(cached_manifest, manifest_path)
        return symbol, part, cached_manifest

    with ThreadPoolExecutor(max_workers=min(workers, len(symbols))) as executor:
        acquired = list(executor.map(acquire_symbol, symbols))
    parts = [part for _, part, _ in acquired]
    part_manifests = {symbol: manifest for symbol, _, manifest in acquired}
    output = cache_dir / "execution_candidates.parquet"
    acquisition_manifest_path = cache_dir / "execution_acquisition_manifest.json"
    aggregate_parts = {
        symbol: {
            "cache_key": manifest["cache_key"],
            "part": manifest["part"],
            "manifest": _file_identity(
                (cache_dir / "parts" / f"{symbol}.parquet").with_suffix(".manifest.json")
            ),
        }
        for symbol, manifest in sorted(part_manifests.items())
    }
    aggregate_manifest = (
        _read_json_dict(acquisition_manifest_path)
        if not refresh and output.exists() and acquisition_manifest_path.exists()
        else None
    )
    aggregate_valid = (
        aggregate_manifest is not None
        and aggregate_manifest.get("parts") == aggregate_parts
        and aggregate_manifest.get("aggregate") == _file_identity(output)
    )
    if not aggregate_valid:
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        pl.concat([pl.scan_parquet(path) for path in parts], how="vertical").sink_parquet(temporary)
        os.replace(temporary, output)
        _atomic_json(
            {"parts": aggregate_parts, "aggregate": _file_identity(output)},
            acquisition_manifest_path,
        )
    return pl.read_parquet(output, columns=list(EVALUATION_COLUMNS))


def build_training_target(
    rows: pl.DataFrame,
    execution: pl.DataFrame,
    *,
    breakout: float,
    exit_profile: str,
) -> pl.DataFrame:
    return _attach_training_target(
        rows, _training_targets(execution, breakout=breakout, exit_profile=exit_profile)
    )


def _training_targets(
    execution: pl.DataFrame, *, breakout: float, exit_profile: str
) -> pl.DataFrame:
    return (
        execution.filter(
            (pl.col("breakout") == breakout)
            & (pl.col("exit_profile") == exit_profile)
            & (pl.col("leverage") == 1)
            & pl.col("scenario").is_in(ADVERSE_SCENARIOS)
        )
        .group_by("market", "date")
        .agg(pl.col("slot_return").min().alias("target"))
    )


def _attach_training_target(rows: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    return rows.join(targets, on=["market", "date"], how="left").with_columns(
        pl.col("target").fill_null(0.0)
    )


def _max_drawdown(values: list[float]) -> float:
    equity = peak = 1.0
    drawdown = 0.0
    for value in values:
        equity *= 1.0 + value
        peak = max(peak, equity)
        drawdown = max(drawdown, 1.0 - equity / peak)
    return drawdown


def _selected_returns(
    selected: pl.DataFrame,
    execution: pl.DataFrame,
    *,
    scenario: str,
    position_cap: int,
    doubled_cost: bool,
    breakout: float | None = None,
    exit_profile: str | None = None,
    leverage: int = 1,
) -> list[float]:
    filled = selected.filter(pl.col("filled")).select("market", "date")
    condition = (pl.col("scenario") == scenario) & (pl.col("leverage") == leverage)
    if breakout is not None:
        condition &= pl.col("breakout") == breakout
    if exit_profile is not None:
        condition &= pl.col("exit_profile") == exit_profile
    matched = execution.filter(condition).join(filled, on=["market", "date"], how="inner")
    if doubled_cost:
        matched = matched.with_columns((pl.col("slot_return") - COST).alias("slot_return"))
    by_day = dict(
        matched.group_by("date")
        .agg((pl.col("slot_return").sum() / position_cap).alias("ret"))
        .iter_rows()
    )
    days = selected["date"].unique().sort().to_list()
    return [float(by_day.get(day, 0.0)) for day in days]


def _return_lookup(
    execution: pl.DataFrame,
    *,
    breakout: float,
    exit_profile: str,
    scenario: str,
    leverage: int = 1,
) -> dict[tuple[str, date], float]:
    rows = execution.filter(
        (pl.col("breakout") == breakout)
        & (pl.col("exit_profile") == exit_profile)
        & (pl.col("scenario") == scenario)
        & (pl.col("leverage") == leverage)
    ).select("market", "date", "slot_return")
    if rows.select("market", "date").n_unique() != rows.height:
        raise ValueError("execution return lookup contains duplicate keys")
    return {
        (str(row["market"]), row["date"]): float(row["slot_return"])
        for row in rows.iter_rows(named=True)
    }


def _selected_lookup_returns(
    selected: pl.DataFrame,
    lookup: dict[tuple[str, date], float],
    *,
    position_cap: int,
    doubled_cost: bool,
) -> list[float]:
    days = selected["date"].unique().sort().to_list()
    by_day = dict.fromkeys(days, 0.0)
    for market, day in selected.filter(pl.col("filled")).select("market", "date").iter_rows():
        key = (str(market), day)
        if key not in lookup:
            raise ValueError(f"filled order has no exact execution row: {key}")
        by_day[day] += (lookup[key] - (COST if doubled_cost else 0.0)) / position_cap
    return [by_day[day] for day in days]


def _selected_adverse_returns(
    selected: pl.DataFrame,
    lookups: dict[str, dict[tuple[str, date], float]],
    *,
    position_cap: int,
) -> dict[str, dict[str, list[float]]]:
    """Build both adverse scenarios and cost stresses in one filled-order pass."""
    days = selected["date"].unique().sort().to_list()
    totals = {
        scenario: {
            "base": dict.fromkeys(days, 0.0),
            "doubled_cost": dict.fromkeys(days, 0.0),
        }
        for scenario in ADVERSE_SCENARIOS
    }
    for market, day in selected.filter(pl.col("filled")).select("market", "date").iter_rows():
        key = (str(market), day)
        for scenario in ADVERSE_SCENARIOS:
            if key not in lookups[scenario]:
                raise ValueError(f"filled order has no exact execution row: {key}")
            value = lookups[scenario][key]
            totals[scenario]["base"][day] += value / position_cap
            totals[scenario]["doubled_cost"][day] += (value - COST) / position_cap
    return {
        scenario: {
            stress: [totals[scenario][stress][day] for day in days]
            for stress in ("base", "doubled_cost")
        }
        for scenario in ADVERSE_SCENARIOS
    }


def _selected_return_grid(
    selected: pl.DataFrame,
    structural_execution: pl.DataFrame,
    *,
    position_cap: int,
) -> dict[tuple[str, int], list[float]]:
    days = selected["date"].unique().sort().to_list()
    filled = selected.filter(pl.col("filled")).select("market", "date")
    matched = structural_execution.join(filled, on=["market", "date"], how="inner")
    expected = filled.height * len(SCENARIOS) * len(LEVERAGES)
    if matched.height != expected:
        raise ValueError(f"selected execution grid is incomplete: {matched.height} != {expected}")
    grouped = {
        (str(row["scenario"]), int(row["leverage"]), row["date"]): float(row["ret"])
        for row in matched.group_by("scenario", "leverage", "date")
        .agg((pl.col("slot_return").sum() / position_cap).alias("ret"))
        .iter_rows(named=True)
    }
    return {
        (scenario, leverage): [grouped.get((scenario, leverage, day), 0.0) for day in days]
        for scenario, _, _ in SCENARIOS
        for leverage in LEVERAGES
    }


def trade_count_gate(
    selected_validation: pl.DataFrame,
    execution: pl.DataFrame,
    *,
    source_validation_fills: int,
    position_cap: int,
    breakout: float,
    exit_profile: str,
    return_lookups: dict[str, dict[tuple[str, date], float]] | None = None,
    adverse_returns: dict[str, dict[str, list[float]]] | None = None,
) -> dict[str, Any]:
    """Hard validation-only gate; no objective or OOS input is accepted."""
    filled = selected_validation.filter(pl.col("filled"))
    active_days = filled["date"].n_unique()
    validation_days = selected_validation["date"].n_unique()
    scenarios = {
        scenario: (
            adverse_returns[scenario]["doubled_cost"]
            if adverse_returns is not None
            else _selected_lookup_returns(
                selected_validation,
                return_lookups[scenario],
                position_cap=position_cap,
                doubled_cost=True,
            )
            if return_lookups is not None
            else _selected_returns(
                selected_validation,
                execution,
                scenario=scenario,
                position_cap=position_cap,
                doubled_cost=True,
                breakout=breakout,
                exit_profile=exit_profile,
            )
        )
        for scenario in ADVERSE_SCENARIOS
    }
    compounds = {
        name: math.prod(1 + value for value in values) - 1 for name, values in scenarios.items()
    }
    passed = (
        filled.height >= math.ceil(1.25 * source_validation_fills)
        and active_days / max(1, validation_days) >= 0.75
        and filled["market"].n_unique() >= 30
        and all(value > 0 for value in compounds.values())
        and all(_max_drawdown(values) <= 0.20 for values in scenarios.values())
    )
    return {
        "passed": passed,
        "fills": filled.height,
        "required_fills": math.ceil(1.25 * source_validation_fills),
        "active_day_ratio": active_days / max(1, validation_days),
        "distinct_symbols": filled["market"].n_unique(),
        "doubled_cost_compound": compounds,
        "max_drawdown": {name: _max_drawdown(values) for name, values in scenarios.items()},
    }


def choose_validation_winner(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in records if row["gate_passed"]]
    return (
        max(eligible, key=lambda row: (row["validation_score"], row["candidate_id"]))
        if eligible
        else None
    )


def _actual_strategy_rows(selections: pl.DataFrame, strategy: str) -> pl.DataFrame:
    return selections.filter(
        (pl.col("strategy") == strategy) & (pl.col("research_replay_action") == "trade")
    )


def _matrix(rows: pl.DataFrame) -> np.ndarray:
    return rows.select(FEATURES).fill_null(0.0).fill_nan(0.0).to_numpy()


def _candidate_id(preset: ModelPreset, breakout: float, topk: int, cap: int, profile: str) -> str:
    return f"{preset.name}__b{breakout:g}__u{topk}__p{cap}__{profile}"


def evaluate_walk_forward(
    panel: pl.DataFrame,
    execution: pl.DataFrame,
    source_trades: pl.DataFrame,
    folds: list[Fold],
    *,
    presets: tuple[ModelPreset, ...] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Select on validation, refit through validation, and predict OOS once per fold."""
    features = build_daily_features(panel).drop_nulls(["turnover_rank"])
    presets = presets or model_presets()
    adverse_lookups = {
        (breakout, profile): {
            scenario: _return_lookup(
                execution,
                breakout=breakout,
                exit_profile=profile,
                scenario=scenario,
            )
            for scenario in ADVERSE_SCENARIOS
        }
        for breakout in BREAKOUTS
        for profile in EXIT_PROFILES
    }
    target_frames = {
        (breakout, profile): _training_targets(execution, breakout=breakout, exit_profile=profile)
        for breakout in BREAKOUTS
        for profile in EXIT_PROFILES
    }
    validation_grid: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    daily_records: list[dict[str, Any]] = []
    for fold in folds:
        train_base = features.filter(pl.col("date").is_between(fold.train_start, fold.train_end))
        validation_base = features.filter(
            pl.col("date").is_between(fold.validation_start, fold.validation_end)
        )
        oos_base = features.filter(pl.col("date").is_between(fold.oos_start, fold.oos_end))
        if train_base.is_empty() or validation_base.is_empty() or oos_base.is_empty():
            continue
        fold_records: list[dict[str, Any]] = []
        source_fills = source_trades.filter(
            pl.col("date").is_between(fold.validation_start, fold.validation_end)
        ).height
        for breakout in BREAKOUTS:
            for topk in UNIVERSE_TOPKS:
                for profile in EXIT_PROFILES:
                    train = _attach_training_target(
                        train_base.filter(pl.col("turnover_rank") <= topk),
                        target_frames[(breakout, profile)],
                    )
                    validation = _attach_training_target(
                        validation_base.filter(pl.col("turnover_rank") <= topk),
                        target_frames[(breakout, profile)],
                    )
                    train_x = _matrix(train)
                    validation_x = _matrix(validation)
                    for preset in presets:
                        prediction = _predict(
                            preset,
                            train_x,
                            train["target"].to_numpy(),
                            validation_x,
                            turnover=validation["log_prev_turnover"].to_numpy(),
                        )
                        for cap in POSITION_CAPS:
                            if cap > topk:
                                continue
                            chosen = preopen_select(
                                validation,
                                prediction,
                                universe_topk=topk,
                                position_cap=cap,
                                breakout=breakout,
                            )
                            adverse_returns = _selected_adverse_returns(
                                chosen,
                                adverse_lookups[(breakout, profile)],
                                position_cap=cap,
                            )
                            score = min(
                                math.prod(1 + value for value in values) - 1
                                for values in (
                                    adverse_returns[scenario]["base"]
                                    for scenario in ADVERSE_SCENARIOS
                                )
                            )
                            gate = trade_count_gate(
                                chosen,
                                execution,
                                source_validation_fills=source_fills,
                                position_cap=cap,
                                breakout=breakout,
                                exit_profile=profile,
                                adverse_returns=adverse_returns,
                            )
                            record = {
                                "fold": fold.number,
                                "candidate_id": _candidate_id(preset, breakout, topk, cap, profile),
                                "model": preset.name,
                                "breakout": breakout,
                                "universe_topk": topk,
                                "position_cap": cap,
                                "exit_profile": profile,
                                "validation_score": score,
                                "gate_passed": gate["passed"],
                                "gate_fills": gate["fills"],
                                "gate_required_fills": gate["required_fills"],
                                "gate_active_day_ratio": gate["active_day_ratio"],
                                "gate_distinct_symbols": gate["distinct_symbols"],
                                "gate_close_compound": gate["doubled_cost_compound"][
                                    "close_exit_stop_first"
                                ],
                                "gate_tp_sl_compound": gate["doubled_cost_compound"][
                                    "tp_sl_stop_first"
                                ],
                                "gate_close_mdd": gate["max_drawdown"]["close_exit_stop_first"],
                                "gate_tp_sl_mdd": gate["max_drawdown"]["tp_sl_stop_first"],
                            }
                            fold_records.append(record)
                            validation_grid.append(record)
        research_best = max(
            fold_records, key=lambda row: (row["validation_score"], row["candidate_id"])
        )
        trade_count_challenger = max(
            fold_records,
            key=lambda row: (
                row["gate_fills"],
                row["validation_score"],
                row["candidate_id"],
            ),
        )
        winner = choose_validation_winner(fold_records)
        for strategy, selected_record, trades_enabled in (
            ("research_best_ungated", research_best, True),
            ("trade_count_challenger", trade_count_challenger, True),
            ("locked", winner or research_best, winner is not None),
        ):
            preset = next(item for item in presets if item.name == selected_record["model"])
            breakout = float(selected_record["breakout"])
            profile = str(selected_record["exit_profile"])
            topk = int(selected_record["universe_topk"])
            cap = int(selected_record["position_cap"])
            structural_execution = execution.filter(
                (pl.col("breakout") == breakout) & (pl.col("exit_profile") == profile)
            )
            refit = _attach_training_target(
                pl.concat([train_base, validation_base]).filter(pl.col("turnover_rank") <= topk),
                target_frames[(breakout, profile)],
            )
            oos = oos_base.filter(pl.col("turnover_rank") <= topk)
            prediction = _predict(
                preset,
                _matrix(refit),
                refit["target"].to_numpy(),
                _matrix(oos),
                turnover=oos["log_prev_turnover"].to_numpy(),
            )
            chosen_oos = preopen_select(
                oos,
                prediction,
                universe_topk=topk,
                position_cap=cap,
                breakout=breakout,
            )
            for row in chosen_oos.iter_rows(named=True):
                selections.append(
                    {
                        "fold": fold.number,
                        "strategy": strategy,
                        "candidate_id": selected_record["candidate_id"],
                        "research_replay_action": "trade" if trades_enabled else "cash",
                        "promotion_eligible": False,
                        "deploy_action": "cash",
                        "selection_status": (
                            "diagnostic_not_promoted"
                            if strategy == "trade_count_challenger"
                            else "ungated_research"
                            if strategy == "research_best_ungated"
                            else "gate_passed"
                            if trades_enabled
                            else "gate_failed_cash"
                        ),
                        "validation_gate_passed": bool(selected_record["gate_passed"]),
                        "research_best_candidate_id": research_best["candidate_id"],
                        "trade_count_challenger_candidate_id": trade_count_challenger[
                            "candidate_id"
                        ],
                        "market": row["market"],
                        "date": row["date"],
                        "prediction": row["prediction"],
                        "filled": bool(row["filled"]),
                        "breakout": breakout,
                        "exit_profile": profile,
                        "position_cap": cap,
                    }
                )
            return_grid = (
                _selected_return_grid(chosen_oos, structural_execution, position_cap=cap)
                if trades_enabled
                else {}
            )
            for leverage in LEVERAGES:
                for scenario, _, _ in SCENARIOS:
                    values = (
                        return_grid[(scenario, leverage)]
                        if trades_enabled
                        else [0.0] * chosen_oos["date"].n_unique()
                    )
                    for day, value in zip(
                        chosen_oos["date"].unique().sort().to_list(), values, strict=True
                    ):
                        daily_records.append(
                            {
                                "fold": fold.number,
                                "strategy": strategy,
                                "date": day,
                                "scenario": scenario,
                                "leverage": leverage,
                                "daily_return": value,
                                "research_replay_action": ("trade" if trades_enabled else "cash"),
                                "promotion_eligible": False,
                                "deploy_action": "cash",
                            }
                        )
    return validation_grid, selections, daily_records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="var/reports/dacapogo_binance/daily_source")
    parser.add_argument("--output-dir", default="var/reports/dacapogo_binance/daily_tuner")
    parser.add_argument("--panel-cache")
    parser.add_argument("--data-root", default="data/market_parquet")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--refresh-execution-cache", action="store_true")
    parser.add_argument("--acquisition-workers", type=_acquisition_workers, default=3)
    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    provenance = runtime_provenance(
        repo_root=Path(__file__).resolve().parents[2],
        packages=(
            "numpy",
            "polars",
            "scikit-learn",
            "lightgbm",
            "threadpoolctl",
        ),
        source_files=(Path(__file__),),
    )
    adapter_identity = provenance["source_files"][str(Path(__file__).resolve())]
    source_dir = Path(args.source_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (source_dir / ".run.lock").open("a+b") as source_lock:
        fcntl.flock(source_lock.fileno(), fcntl.LOCK_SH)
        source_summary_path = source_dir / "summary.json"
        source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
        if source_summary.get("source", {}).get("file_sha256") != SOURCE_FILE_SHA256:
            raise ValueError("source artifact is not the verified Dacapogo daily formula")
        source_trades_path = source_dir / "trades.csv"
        source_trades_identity = _file_identity(source_trades_path)
        if source_summary.get("artifacts", {}).get("trades.csv") != source_trades_identity:
            raise ValueError("source trades do not match the source summary seal")
        source_trades = pl.read_csv(
            source_trades_path, columns=["market", "date"], try_parse_dates=True
        )
        start = date.fromisoformat(source_summary["data"]["start"])
        end = date.fromisoformat(source_summary["data"]["end"])
        panel_path = (
            Path(args.panel_cache) if args.panel_cache else source_dir / "daily_panel.parquet"
        )
        if panel_path.resolve() != (source_dir / "daily_panel.parquet").resolve():
            raise ValueError("--panel-cache must be the sealed source generation panel")
        manifest_path = panel_path.with_name(f"{panel_path.name}.manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("file") != _file_identity(panel_path):
            raise ValueError("daily panel does not match its manifest seal")
        if source_summary.get("artifacts", {}).get("daily_panel.parquet") != _file_identity(
            panel_path
        ) or source_summary.get("artifacts", {}).get(
            "daily_panel.parquet.manifest.json"
        ) != _file_identity(manifest_path):
            raise ValueError("daily panel does not match the source generation")
        panel = pl.read_parquet(
            panel_path,
            columns=["market", "date", "value", "open", "high", "low", "close"],
        )
        _validate_panel(
            panel,
            tuple(source_summary["data"]["symbols"]),
            start - timedelta(days=7),
            end,
            manifest.get("audits", []),
        )
        source_summary_identity = _file_identity(source_summary_path)
        panel_identity = _file_identity(panel_path)
        panel_manifest_identity = _file_identity(manifest_path)
    execution = acquire_execution_candidates(
        panel,
        cache_dir=output,
        data_root=Path(args.data_root),
        exchange=args.exchange,
        refresh=getattr(args, "refresh_execution_cache", False),
        workers=getattr(args, "acquisition_workers", 3),
    )
    folds = make_folds(start, end)
    validation_grid, selections, daily = evaluate_walk_forward(
        panel, execution, source_trades, folds
    )
    validation_path = output / "validation_grid.csv"
    selections_path = output / "selections.csv"
    daily_path = output / "daily.csv"
    with atomic_output_path(validation_path) as temporary:
        pl.DataFrame(validation_grid).write_csv(temporary)
    with atomic_output_path(selections_path) as temporary:
        pl.DataFrame(selections).write_csv(temporary)
    with atomic_output_path(daily_path) as temporary:
        pl.DataFrame(daily).write_csv(temporary)
    summary_metrics: dict[str, dict[str, float | int]] = {}
    daily_frame = pl.DataFrame(daily)
    selection_frame = pl.DataFrame(selections)
    selection_stats: dict[str, dict[str, Any]] = {}
    for strategy in selection_frame["strategy"].unique().to_list():
        strategy_rows = selection_frame.filter(pl.col("strategy") == strategy)
        actual_rows = _actual_strategy_rows(selection_frame, str(strategy))
        filled_keys = actual_rows.filter(pl.col("filled")).select(
            "market", "date", "breakout", "exit_profile"
        )
        exact_rows = execution.join(
            filled_keys,
            on=["market", "date", "breakout", "exit_profile"],
            how="inner",
        )
        exact_counts = {
            (str(row["scenario"]), int(row["leverage"])): row
            for row in exact_rows.group_by("scenario", "leverage")
            .agg(
                pl.col("liquidated").sum().alias("liquidations"),
                pl.col("mark_liquidation_breach").sum().alias("possible_liquidations"),
                pl.col("ambiguous_minute").sum().alias("ambiguous_minutes"),
            )
            .iter_rows(named=True)
        }
        selection_stats[str(strategy)] = {
            "orders": strategy_rows.height,
            "actual_orders": actual_rows.height,
            "fills": filled_keys.height,
            "exact_counts": exact_counts,
        }
    for row in (
        daily_frame.group_by("strategy", "scenario", "leverage")
        .agg(
            pl.col("date").sort().alias("dates"),
            pl.col("daily_return").sort_by("date").alias("returns"),
        )
        .iter_rows(named=True)
    ):
        values = [float(value) for value in row["returns"]]
        latest = values[-30:]
        post_source = [
            float(value)
            for day, value in zip(row["dates"], values, strict=True)
            if day >= date(2026, 7, 22)
        ]
        key = f"{row['strategy']}__{row['scenario']}__{row['leverage']}x"
        stats = selection_stats[str(row["strategy"])]
        counts = stats["exact_counts"].get(
            (str(row["scenario"]), int(row["leverage"])),
            {"liquidations": 0, "possible_liquidations": 0, "ambiguous_minutes": 0},
        )
        summary_metrics[key] = {
            "full_total_return": math.prod(1 + value for value in values) - 1,
            "full_max_drawdown": _max_drawdown(values),
            "latest30_total_return": math.prod(1 + value for value in latest) - 1,
            "latest30_max_drawdown": _max_drawdown(latest),
            "post_2026_07_21_total_return": math.prod(1 + value for value in post_source) - 1,
            "post_2026_07_21_max_drawdown": _max_drawdown(post_source),
            "orders": stats["orders"],
            "actual_orders": stats["actual_orders"],
            "fills": stats["fills"],
            "liquidations": int(counts["liquidations"]),
            "possible_liquidations": int(counts["possible_liquidations"]),
            "ambiguous_minutes": int(counts["ambiguous_minutes"]),
        }
    first_oos = min((fold.oos_start for fold in folds), default=end + timedelta(days=1))
    source_oos_fills = source_trades.filter(pl.col("date") >= first_oos).height
    for metrics in summary_metrics.values():
        metrics["source_oos_fills"] = source_oos_fills
        metrics["fill_ratio_vs_source_oos"] = metrics["fills"] / max(1, source_oos_fills)
    execution_path = output / "execution_candidates.parquet"
    acquisition_manifest_path = output / "execution_acquisition_manifest.json"
    artifact_paths = (
        validation_path,
        selections_path,
        daily_path,
        execution_path,
        acquisition_manifest_path,
    )
    payload = {
        "artifact_kind": "dacapogo_daily_structural_ml_tuner_research",
        "strategy_tier": "research_only",
        "publication": {
            "contract": "atomic file replacement; summary seal written last",
            "authority": "summary.json",
        },
        "adapter": {"file": Path(__file__).name, **adapter_identity},
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "upstream": {
            "commit": LATEST_UPSTREAM_HEAD,
            "latest_head": LATEST_UPSTREAM_HEAD,
            "source_file_sha256": SOURCE_FILE_SHA256,
            "behavior_ranker": {
                "file_sha256": RANKER_FILE_SHA256,
                "claim": "retrospective_daily_behavior_ranking_proxy",
                "causality": "noncausal_same_day_ohlcv",
                "pnl_claim": False,
                "strategy_claim": False,
                "used_for_signal": RANKER_USED_FOR_SIGNAL,
            },
            "timing_falsification": {
                "multifrequency": {
                    "file_sha256": MULTIFREQ_FILE_SHA256,
                    "outcome": "conditional_localization_not_reconstructed",
                },
                "nested_cv": {
                    "file_sha256": NESTED_CV_FILE_SHA256,
                    "outcome": "timing_artifact_not_falsified",
                },
                "used_for_signal": False,
            },
            "input_artifact_commit": source_summary.get("source", {}).get("commit"),
            "input_provenance": {
                "artifact_commit": source_summary.get("source", {}).get("commit"),
                "artifact_source_file_sha256": source_summary.get("source", {}).get("file_sha256"),
            },
        },
        "inputs": {
            "source_summary.json": source_summary_identity,
            "source_trades.csv": source_trades_identity,
            "daily_panel.parquet": panel_identity,
            "daily_panel.manifest.json": panel_manifest_identity,
            "execution_acquisition_manifest.json": _file_identity(acquisition_manifest_path),
        },
        "data": {
            "start": str(start),
            "end": str(end),
            "panel_rows": panel.height,
            "execution_candidate_rows": execution.height,
            "survivorship_caveat": source_summary.get("data", {})
            .get("universe", {})
            .get(
                "survivorship_bias",
                "current-active universe omits contracts delisted before acquisition",
            ),
        },
        "evaluation_label": "retrospective_walk_forward_pseudo_oos_not_true_forward_oos",
        "decision_timing": "at UTC daily open; current open_gap is known, submission latency is not modeled",
        "promotion_eligible": False,
        "deploy_action": "cash",
        "walk_forward": {
            "train": "expanding minimum 180 calendar days",
            "validation_days": 30,
            "embargo_days": 1,
            "oos_days": 30,
            "oos_selection_authority": False,
            "folds": [asdict(fold) for fold in folds],
        },
        "grid": {
            "breakouts": BREAKOUTS,
            "universe_topks": UNIVERSE_TOPKS,
            "position_caps": POSITION_CAPS,
            "exit_profiles": EXIT_PROFILES,
            "models": [asdict(item) for item in model_presets()],
            "tuning_leverage": 1,
            "reported_leverages": LEVERAGES,
            "fixed_cost": COST,
        },
        "gate": {
            "trade_count": ">=1.25x exact source +4%/TOP10 validation fills",
            "active_trade_days": ">=75%",
            "distinct_symbols": ">=30",
            "adverse_scenarios": ADVERSE_SCENARIOS,
            "doubled_cost": 2 * COST,
            "max_drawdown": "<=20%",
            "fallback": "locked cash; best ungated candidate retained in selections",
            "trade_count_challenger": "validation-most fills, reported separately and never promoted without the return gate",
            "trade_count_challenger_status": "diagnostic_not_promoted",
            "trade_count_challenger_gate_authority": "must pass the same validation-only gate; diagnostic selection alone has no promotion authority",
        },
        "strategy_status": {
            strategy: {
                "status": rows[0, "selection_status"],
                "validation_gate_passed_folds": rows.filter(pl.col("validation_gate_passed"))[
                    "fold"
                ].n_unique(),
                "folds": rows["fold"].n_unique(),
            }
            for strategy in selection_frame["strategy"].unique().sort().to_list()
            for rows in [selection_frame.filter(pl.col("strategy") == strategy)]
        },
        "summaries": summary_metrics,
        "artifacts": {path.name: _file_identity(path) for path in artifact_paths},
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "acquisition_workers": getattr(args, "acquisition_workers", 3),
            "memory": "streamed full audit parquet, compact evaluation columns, per-symbol atomic parts, cached validation targets/returns",
            "provenance": provenance,
        },
    }
    summary_path = output / "summary.json"
    atomic_write_text(summary_path, json.dumps(payload, indent=2, default=str) + "\n")
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.resolve() == Path(args.source_dir).resolve():
        raise ValueError("output directory must differ from the sealed source directory")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _run(args)


def main(argv: list[str] | None = None) -> int:
    run(build_arg_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
