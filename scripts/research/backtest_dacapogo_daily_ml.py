#!/usr/bin/env python3
"""Legacy post-trigger daily-feature ML filter for the Binance Dacapogo audit."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import resource
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from lumina_quant.optimization.search_policy import LOCKED_OOS_SEARCH_FLAGS
from lumina_quant.research.run_card import (
    atomic_output_path,
    atomic_write_text,
    runtime_provenance,
)
from lumina_quant.strategies.dacapogo_daily_source import COST, TOPK

UPSTREAM_HEAD = "633ba5d6bc0c84a20696af6b2bf807cf55d21248"
SOURCE_FILE_SHA256 = "17516d9457540e978d4828620c99794df0617c154faead5acee9f0847b5fcd8e"
ALPHAS = (0.1, 1.0, 10.0)
MAX_POSITIONS = (3, 5, 10)
FEATURES = (
    "prev_ret",
    "prev_range",
    "log_prev_value",
    "value_change",
    "momentum_3",
    "momentum_7",
)
GATE_SCENARIOS = ("close_exit_stop_first", "tp_sl_stop_first")
TARGET_SCENARIO = "close_exit_stop_first"


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


def build_daily_features(panel: pl.DataFrame) -> pl.DataFrame:
    """Build daily-only features known before the current UTC day."""
    previous_value = pl.col("value").shift(1).over("market")
    prior_value = pl.col("value").shift(2).over("market")
    return (
        panel.sort(["market", "date"])
        .with_columns(
            (pl.col("close") / pl.col("open") - 1).shift(1).over("market").alias("prev_ret"),
            (pl.col("high") / pl.col("low") - 1).shift(1).over("market").alias("prev_range"),
            pl.col("value").shift(1).over("market").log1p().alias("log_prev_value"),
            pl.when(prior_value > 0)
            .then(previous_value / prior_value - 1)
            .otherwise(0.0)
            .alias("value_change"),
            (pl.col("close").shift(1).over("market") / pl.col("close").shift(3).over("market") - 1)
            .fill_null(0.0)
            .alias("momentum_3"),
            (pl.col("close").shift(1).over("market") / pl.col("close").shift(7).over("market") - 1)
            .fill_null(0.0)
            .alias("momentum_7"),
        )
        .select("market", "date", *FEATURES)
    )


def ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    predict_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Upstream-compatible deterministic ridge with train-only standardization."""
    train_x = np.asarray(train_x, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    predict_x = np.asarray(predict_x, dtype=float)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0, ddof=0)
    std = np.where(std > 0, std, 1.0)
    x_train = (train_x - mean) / std
    x_predict = (predict_x - mean) / std
    intercept = float(train_y.mean())
    lhs = x_train.T @ x_train + np.eye(x_train.shape[1]) * float(alpha)
    rhs = x_train.T @ (train_y - intercept)
    beta = np.linalg.solve(lhs, rhs)
    return intercept + x_predict @ beta


def make_folds(start: date, end: date) -> list[Fold]:
    """Create expanding 180d-train/30d-validation/1d-embargo/30d-OOS folds."""
    first_oos = start + timedelta(days=211)
    folds: list[Fold] = []
    oos_start = first_oos
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


def _compound(values: list[float]) -> float:
    return math.prod(1.0 + value for value in values) - 1.0


def _select(rows: pl.DataFrame, predictions: np.ndarray, limit: int) -> set[tuple[str, date]]:
    scored = rows.with_columns(pl.Series("prediction", predictions)).filter(
        pl.col("prediction") > 0
    )
    return {
        (str(row["market"]), row["date"])
        for row in scored.sort(["date", "prediction", "market"], descending=[False, True, False])
        .group_by("date", maintain_order=True)
        .head(limit)
        .iter_rows(named=True)
    }


def _daily_returns(
    execution: pl.DataFrame,
    keys: set[tuple[str, date]],
    start: date,
    end: date,
    *,
    extra_cost: bool = False,
) -> dict[tuple[str, int], list[float]]:
    selected = execution.join(_key_frame(keys), on=["market", "date"], how="inner")
    grouped = {
        (str(row["scenario"]), int(row["leverage"]), row["date"]): float(row["ret"])
        for row in selected.with_columns(
            (
                pl.col("slot_return") - pl.col("leverage") * COST
                if extra_cost
                else pl.col("slot_return")
            ).alias("adjusted")
        )
        .group_by("scenario", "leverage", "date")
        .agg((pl.col("adjusted").sum() / TOPK).alias("ret"))
        .iter_rows(named=True)
    }
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    slices = execution.select("scenario", "leverage").unique().iter_rows()
    return {
        (str(scenario), int(leverage)): [
            grouped.get((str(scenario), int(leverage), day), 0.0) for day in days
        ]
        for scenario, leverage in slices
    }


def _key_frame(keys: set[tuple[str, date]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "market": pl.Series([key[0] for key in keys], dtype=pl.String),
            "date": pl.Series([key[1] for key in keys], dtype=pl.Date),
        }
    )


def _gate_score(
    execution: pl.DataFrame,
    keys: set[tuple[str, date]],
    start: date,
    end: date,
) -> float | None:
    returns = _daily_returns(execution, keys, start, end, extra_cost=True)
    scores = [_compound(returns.get((scenario, 1), [])) for scenario in GATE_SCENARIOS]
    return min(scores) if all(score > 0 for score in scores) else None


def _target_score(
    execution: pl.DataFrame,
    keys: set[tuple[str, date]],
    start: date,
    end: date,
) -> float:
    returns = _daily_returns(execution, keys, start, end)
    return _compound(returns.get((TARGET_SCENARIO, 1), []))


def _feature_matrix(rows: pl.DataFrame) -> np.ndarray:
    return rows.select(FEATURES).to_numpy()


def _candidate_rows(model_rows: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    return model_rows.filter(pl.col("date").is_between(start, end)).sort(
        ["date", "entry_time", "market"]
    )


def evaluate_folds(
    model_rows: pl.DataFrame,
    execution: pl.DataFrame,
    folds: list[Fold],
) -> tuple[list[dict[str, Any]], set[tuple[str, date]], set[tuple[str, date]]]:
    """Tune on validation, then produce ungated ML and gated locked OOS keys."""
    records: list[dict[str, Any]] = []
    ml_oos: set[tuple[str, date]] = set()
    locked_oos: set[tuple[str, date]] = set()
    for fold in folds:
        train = _candidate_rows(model_rows, fold.train_start, fold.train_end)
        validation = _candidate_rows(model_rows, fold.validation_start, fold.validation_end)
        oos = _candidate_rows(model_rows, fold.oos_start, fold.oos_end)
        if train.is_empty() or validation.is_empty() or oos.is_empty():
            continue
        best: tuple[float, float, int] | None = None
        for alpha in ALPHAS:
            predictions = ridge_predict(
                _feature_matrix(train),
                train["target"].to_numpy(),
                _feature_matrix(validation),
                alpha,
            )
            for limit in MAX_POSITIONS:
                keys = _select(validation, predictions, limit)
                score = _target_score(execution, keys, fold.validation_start, fold.validation_end)
                candidate = (score, -alpha, -limit)
                if best is None or candidate > (best[0], -best[1], -best[2]):
                    best = (score, alpha, limit)
        assert best is not None
        validation_score, alpha, limit = best
        refit = pl.concat([train, validation]).sort(["date", "entry_time", "market"])
        predictions = ridge_predict(
            _feature_matrix(refit), refit["target"].to_numpy(), _feature_matrix(oos), alpha
        )
        ml_keys = _select(oos, predictions, limit)
        ml_oos.update(ml_keys)
        baseline_validation = {
            (str(row["market"]), row["date"]) for row in validation.iter_rows(named=True)
        }
        ml_validation_predictions = ridge_predict(
            _feature_matrix(train), train["target"].to_numpy(), _feature_matrix(validation), alpha
        )
        ml_validation = _select(validation, ml_validation_predictions, limit)
        ml_gate = _gate_score(execution, ml_validation, fold.validation_start, fold.validation_end)
        baseline_gate = _gate_score(
            execution, baseline_validation, fold.validation_start, fold.validation_end
        )
        action = "cash"
        if ml_gate is not None or baseline_gate is not None:
            action = (
                "research_ml"
                if (ml_gate or -math.inf) >= (baseline_gate or -math.inf)
                else "baseline"
            )
        chosen = (
            ml_keys
            if action == "research_ml"
            else {(str(row["market"]), row["date"]) for row in oos.iter_rows(named=True)}
            if action == "baseline"
            else set()
        )
        locked_oos.update(chosen)
        for row, prediction in zip(oos.iter_rows(named=True), predictions, strict=True):
            key = (str(row["market"]), row["date"])
            records.append(
                {
                    "fold": fold.number,
                    "market": key[0],
                    "date": key[1],
                    "entry_time": row["entry_time"],
                    "prediction": float(prediction),
                    "alpha": alpha,
                    "max_positions": limit,
                    "validation_target_compound_return": validation_score,
                    "research_ml": key in ml_keys,
                    "baseline": True,
                    "locked": key in chosen,
                    "research_replay_action": action,
                    "promotion_eligible": False,
                    "deploy_action": "cash",
                }
            )
    return records, ml_oos, locked_oos


def _summarize(values: list[float], counts: dict[str, int]) -> dict[str, float | int]:
    equity = np.cumprod(1.0 + np.asarray(values, dtype=float))
    peaks = np.maximum.accumulate(np.r_[1.0, equity])
    drawdown = 1.0 - np.r_[1.0, equity] / peaks
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        "total_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
        "max_drawdown": float(drawdown.max()),
        "sharpe_365": math.sqrt(365) * float(np.mean(values)) / std if std > 0 else 0.0,
        **counts,
    }


def _execution_counts(rows: pl.DataFrame) -> dict[str, int]:
    return {
        "trades": rows.height,
        "liquidations": int(rows["liquidated"].sum() or 0),
        "possible_liquidations": int(rows["mark_liquidation_breach"].sum() or 0),
        "ambiguous_minute_trades": int(rows["ambiguous_minute"].sum() or 0),
    }


def _validate_panel(
    panel: pl.DataFrame,
    symbols: tuple[str, ...],
    history_start: date,
    end: date,
    audits: list[dict[str, Any]],
) -> None:
    required = {"market", "date", "value", "open", "high", "low", "close"}
    if not required <= set(panel.columns):
        raise ValueError("daily panel cache is missing required columns")
    if (
        panel.schema["date"] != pl.Date
        or panel.select("market", "date").null_count().sum_horizontal()[0]
    ):
        raise ValueError("daily panel cache has invalid keys")
    numeric = ("value", "open", "high", "low", "close")
    invalid = panel.filter(
        pl.any_horizontal(pl.col(name).is_null() | ~pl.col(name).is_finite() for name in numeric)
        | (pl.col("value") < 0)
        | (pl.min_horizontal("open", "high", "low", "close") <= 0)
        | (pl.col("high") < pl.max_horizontal("open", "close"))
        | (pl.col("low") > pl.min_horizontal("open", "close"))
    )
    if invalid.height or panel.select("market", "date").n_unique() != panel.height:
        raise ValueError("daily panel cache has invalid or duplicate rows")
    actual_symbols = set(panel["market"].cast(pl.String).unique().to_list())
    if actual_symbols != set(symbols):
        raise ValueError("daily panel cache symbols do not match source summary")
    bounds = panel.select(
        pl.col("date").min().alias("start"), pl.col("date").max().alias("end")
    ).row(0, named=True)
    if bounds["start"] is None or bounds["start"] > history_start or bounds["end"] != end:
        raise ValueError("daily panel cache does not cover required 7-day history through end")
    ordered = panel.sort(["market", "date"])
    if ordered.filter(
        pl.col("date").diff().over("market").dt.total_days().fill_null(1) != 1
    ).height:
        raise ValueError("daily panel cache contains date gaps")
    expected: dict[str, tuple[date, date, int]] = {}
    for audit in audits:
        symbol = str(audit["symbol"])
        if symbol in expected or date.fromisoformat(audit["requested_start"]) != history_start:
            raise ValueError("daily panel audit is invalid")
        expected[symbol] = (
            date.fromisoformat(audit["start"]),
            date.fromisoformat(audit["end"]),
            int(audit["days"]),
        )
    actual = {
        str(row["market"]): (row["start"], row["end"], int(row["days"]))
        for row in ordered.group_by("market")
        .agg(
            pl.col("date").min().alias("start"),
            pl.col("date").max().alias("end"),
            pl.len().alias("days"),
        )
        .iter_rows(named=True)
    }
    if (
        set(expected) != set(symbols)
        or actual != expected
        or any(item[1] != end for item in actual.values())
    ):
        raise ValueError("daily panel cache does not match audited per-symbol coverage")


def _file_identity(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _panel_manifest_path(cache: Path) -> Path:
    return cache.with_name(f"{cache.name}.manifest.json")


def _true_forward_start(end: date, generated_on: date) -> date:
    return max(end, generated_on) + timedelta(days=1)


def _read_inputs(
    source_dir: Path,
) -> tuple[dict[str, Any], pl.DataFrame, dict[str, dict[str, int | str]]]:
    with (source_dir / ".run.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        return _read_inputs_unlocked(source_dir)


def _read_inputs_unlocked(
    source_dir: Path,
) -> tuple[dict[str, Any], pl.DataFrame, dict[str, dict[str, int | str]]]:
    summary_path = source_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("source", {}).get("file_sha256") != SOURCE_FILE_SHA256:
        raise ValueError("source artifact does not match the verified Dacapogo daily formula")
    source_files = (
        "trades.csv",
        "execution_trades.csv",
        "daily_panel.parquet",
        "daily_panel.parquet.manifest.json",
    )
    identities = {name: _file_identity(source_dir / name) for name in source_files}
    sealed = summary.get("artifacts", {})
    if any(sealed.get(name) != identity for name, identity in identities.items()):
        raise ValueError("source artifact files do not match summary integrity metadata")
    rules = summary.get("rules", {})
    if int(rules.get("topk", -1)) != TOPK or not math.isclose(
        float(rules.get("round_trip_cost", -1.0)), COST
    ):
        raise ValueError("source artifact rule constants do not match the ML adapter")
    trades = pl.read_csv(
        source_dir / "trades.csv", columns=["market", "date"], try_parse_dates=True
    )
    if trades.select("market", "date").n_unique() != trades.height:
        raise ValueError("source trades artifact contains duplicate keys")
    required = {
        "market",
        "date",
        "scenario",
        "leverage",
        "entry_time",
        "slot_return",
        "liquidated",
        "mark_liquidation_breach",
        "ambiguous_minute",
    }
    execution = pl.read_csv(
        source_dir / "execution_trades.csv",
        columns=sorted(required),
        try_parse_dates=True,
    )
    data = summary["data"]
    expected_scenarios = set(summary["execution"]["scenarios"])
    expected_leverages = set(summary["execution"]["leverages"])
    trigger_count = int(summary["execution"]["audited_trigger_symbol_days"])
    if trades.height != trigger_count:
        raise ValueError("source trade count does not match summary")
    if execution.select("market", "date", "scenario", "leverage").n_unique() != execution.height:
        raise ValueError("execution artifact contains duplicate scenario keys")
    if (
        set(execution["scenario"].unique().to_list()) != expected_scenarios
        or set(execution["leverage"].unique().to_list()) != expected_leverages
        or execution.height != trigger_count * len(expected_scenarios) * len(expected_leverages)
    ):
        raise ValueError("execution artifact does not match the summary scenario grid")
    execution_keys = execution.select("market", "date").unique()
    if (
        execution_keys.join(trades, on=["market", "date"], how="anti").height
        or trades.join(execution_keys, on=["market", "date"], how="anti").height
    ):
        raise ValueError("execution artifact is not linked to the source trades")
    invalid = execution.filter(
        pl.col("entry_time").is_null()
        | pl.col("slot_return").is_null()
        | ~pl.col("slot_return").is_finite()
        | pl.any_horizontal(
            pl.col(name).is_null()
            for name in ("liquidated", "mark_liquidation_breach", "ambiguous_minute")
        )
        | ~pl.col("market").is_in(data["symbols"])
        | pl.col("date")
        .is_between(date.fromisoformat(data["start"]), date.fromisoformat(data["end"]))
        .not_()
    )
    if invalid.height:
        raise ValueError("execution artifact contains invalid rows")
    identities["summary.json"] = _file_identity(summary_path)
    return summary, execution, identities


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    provenance = runtime_provenance(
        repo_root=Path(__file__).resolve().parents[2],
        packages=("numpy", "polars"),
        source_files=(Path(__file__),),
    )
    adapter_identity = provenance["source_files"][str(Path(__file__).resolve())]
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (source_dir / ".run.lock").open("a+b") as source_lock:
        fcntl.flock(source_lock.fileno(), fcntl.LOCK_SH)
        source, execution, input_identities = _read_inputs_unlocked(source_dir)
        symbols = tuple(source["data"]["symbols"])
        start, end = (
            date.fromisoformat(source["data"]["start"]),
            date.fromisoformat(source["data"]["end"]),
        )
        source_panel = source_dir / "daily_panel.parquet"
        cache = Path(args.panel_cache) if args.panel_cache else source_panel
        if cache.resolve() != source_panel.resolve():
            raise ValueError("--panel-cache must be the sealed source generation panel")
        manifest_path = _panel_manifest_path(cache)
        history_start = start - timedelta(days=7)
        if not cache.is_file() or not manifest_path.is_file():
            raise ValueError("daily panel must come from the sealed source generation")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("file") != _file_identity(cache):
            raise ValueError("daily panel cache does not match its integrity manifest")
        if (
            _file_identity(cache) != source["artifacts"]["daily_panel.parquet"]
            or _file_identity(manifest_path)
            != source["artifacts"]["daily_panel.parquet.manifest.json"]
        ):
            raise ValueError("daily panel cache does not match the source generation")
        panel = pl.read_parquet(
            cache, columns=["market", "date", "value", "open", "high", "low", "close"]
        )
        _validate_panel(panel, symbols, history_start, end, manifest["audits"])
        input_identities["daily_panel.parquet"] = _file_identity(cache)
        input_identities[manifest_path.name] = _file_identity(manifest_path)
    panel_source = "source_generation"
    features = build_daily_features(panel)
    target = execution.filter(
        (pl.col("scenario") == TARGET_SCENARIO) & (pl.col("leverage") == 1)
    ).select("market", "date", "entry_time", pl.col("slot_return").alias("target"))
    model_rows = (
        target.join(features, on=["market", "date"], how="inner")
        .drop_nulls(FEATURES)
        .filter(pl.all_horizontal(pl.col(name).is_finite() for name in FEATURES))
    )
    folds = make_folds(start, end)
    selections, ml_keys, locked_keys = evaluate_folds(model_rows, execution, folds)
    baseline_keys = {(str(row["market"]), row["date"]) for row in model_rows.iter_rows(named=True)}
    oos_start = folds[0].oos_start if folds else end + timedelta(days=1)
    daily_records: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for strategy, keys in (
        ("locked", locked_keys),
        ("research_ml", ml_keys),
        ("baseline", baseline_keys),
    ):
        returns = _daily_returns(execution, keys, oos_start, end)
        selected = execution.join(_key_frame(keys), on=["market", "date"], how="inner").filter(
            pl.col("date") >= oos_start
        )
        summaries[strategy] = {}
        for (scenario, leverage), values in sorted(returns.items()):
            scenario_rows = selected.filter(
                (pl.col("scenario") == scenario) & (pl.col("leverage") == leverage)
            )
            summaries[strategy][f"{scenario}_{leverage}x"] = _summarize(
                values, _execution_counts(scenario_rows)
            )
            for offset, value in enumerate(values):
                daily_records.append(
                    {
                        "date": oos_start + timedelta(days=offset),
                        "scenario": scenario,
                        "leverage": leverage,
                        "strategy": strategy,
                        "daily_return": value,
                    }
                )
    payload = {
        "artifact_kind": "dacapogo_daily_ml_walk_forward_research",
        "strategy_tier": "research_only",
        "promotion_eligible": False,
        "deploy_action": "cash",
        "publication": {
            "contract": "atomic file replacement; summary seal written last",
            "authority": "summary.json",
        },
        "adapter": {"file": Path(__file__).name, **adapter_identity},
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "upstream": {
            "latest_head": UPSTREAM_HEAD,
            "source_artifact_commit": source["source"]["commit"],
            "source_file_sha256": SOURCE_FILE_SHA256,
            "source_hash_unchanged": source["source"]["file_sha256"] == SOURCE_FILE_SHA256,
            "source_rule": "fixed exactly; ML only filters source-rule trigger rows",
        },
        "inputs": input_identities,
        "data": {
            "symbols": list(symbols),
            "start": str(start),
            "end": str(end),
            "panel_cache": str(cache),
            "panel_manifest": str(manifest_path),
            "panel_source": panel_source,
            "panel_rows": panel.height,
            "model_rows": model_rows.height,
            "survivorship_bias": source["data"]["universe"].get(
                "survivorship_bias", source["data"]["universe"].get("warning")
            ),
        },
        "model": {
            "features": list(FEATURES),
            "scope": "legacy post-trigger filter; structural at-open tuning is a separate artifact",
            "daily_only_divergence": "features use completed prior daily bars; trigger membership and execution labels are retrospective outcomes",
            "alphas": list(ALPHAS),
            "max_positions": list(MAX_POSITIONS),
            "standardization": "train-only population mean/std (ddof=0), y-mean intercept, deterministic NumPy ridge",
            "portfolio_return": f"sum selected slot_return / TOPK ({TOPK})",
        },
        "walk_forward": {
            "evaluation_label": "retrospective_walk_forward_pseudo_oos_not_true_forward_oos",
            "true_forward_oos_earliest_utc": str(
                _true_forward_start(end, datetime.now(UTC).date())
            ),
            "train": "expanding, minimum 180 calendar days",
            "validation_days": 30,
            "embargo_days": 1,
            "oos_step_days": 30,
            "final_partial_min_days": 7,
            **LOCKED_OOS_SEARCH_FLAGS,
            "folds": [fold.__dict__ for fold in folds],
            "completed_folds": len({row["fold"] for row in selections}),
        },
        "gate": {
            "cost": "2x source round-trip cost",
            "scenarios": list(GATE_SCENARIOS),
            "leverage": 1,
            "rule": "validation compound return must be >0 in both scenarios; otherwise cash",
        },
        "summaries": summaries,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "compute": "cached Polars daily features plus nine NumPy ridge candidates per fold; no minute-bar refetch",
            "provenance": provenance,
        },
    }
    selections_path = output_dir / "selections.csv"
    daily_path = output_dir / "daily.csv"
    with atomic_output_path(selections_path) as temporary:
        pl.DataFrame(selections).write_csv(temporary)
    with atomic_output_path(daily_path) as temporary:
        pl.DataFrame(daily_records).sort(["strategy", "scenario", "leverage", "date"]).write_csv(
            temporary
        )
    payload["artifacts"] = {
        path.name: _file_identity(path) for path in (selections_path, daily_path)
    }
    atomic_write_text(
        output_dir / "summary.json",
        json.dumps(payload, default=str, indent=2) + "\n",
    )
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.resolve() == Path(args.source_dir).resolve():
        raise ValueError("output directory must differ from the sealed source directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".run.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _run(args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="var/reports/dacapogo_binance/daily_source")
    parser.add_argument("--output-dir", default="var/reports/dacapogo_binance/daily_ml")
    parser.add_argument("--panel-cache")
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_arg_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
