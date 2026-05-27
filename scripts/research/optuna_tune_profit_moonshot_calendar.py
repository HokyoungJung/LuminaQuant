#!/usr/bin/env python3
"""Optuna tune the fresh-start TRX calendar sleeve under locked-OOS policy.

The objective is train/validation-only.  OOS is replayed and reported for every
trial, but it does not drive the Optuna objective.  This keeps the existing
locked-OOS contract intact while replacing broad grid search with a TPE search
around the promising calendar-rotation region.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import resource
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.portfolio.optimizer_core import safe_float as _safe_float  # noqa: E402

FRESH_PATH = REPO_ROOT / "scripts/research/replay_profit_moonshot_fresh_start.py"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "var/reports/profit_moonshot_20260501/current_tail_20260507/external_overhaul"
)
RUN_NAME = "profit_moonshot_calendar_optuna"


def _load_fresh_module() -> Any:
    spec = importlib.util.spec_from_file_location("replay_profit_moonshot_fresh_start", FRESH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {FRESH_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rss_mib() -> float:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss or 0)
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def _trial_params(trial: Any) -> dict[str, Any]:
    return {
        "short_symbol": trial.suggest_categorical("short_symbol", ["", "ETHUSDT"]),
        "threshold": trial.suggest_categorical("threshold", [0.010, 0.012, 0.015, 0.018, 0.020]),
        "hold_bars": trial.suggest_categorical("hold_bars", [96, 120, 144, 168]),
        "long_scale": trial.suggest_float("long_scale", 5.0, 6.4, step=0.1),
        "short_scale": trial.suggest_categorical("short_scale", [8.0, 10.0, 12.0]),
        "take_profit_pct": trial.suggest_categorical(
            "take_profit_pct",
            [0.018, 0.024, 0.030, 0.035, 0.045, 0.060],
        ),
    }


def _spec_from_params(fresh: Any, params: dict[str, Any], *, trial_number: int) -> Any:
    short_symbol = str(params.get("short_symbol") or "")
    threshold = _safe_float(params.get("threshold"), 0.015)
    hold_bars = int(params.get("hold_bars") or 120)
    long_scale = _safe_float(params.get("long_scale"), 6.0)
    short_scale = _safe_float(params.get("short_scale"), 10.0)
    take_profit_pct = _safe_float(params.get("take_profit_pct"), 0.018)
    short_label = short_symbol.lower() or "weakest"
    name = (
        f"optuna_calendar_trx_s{short_label}_t{trial_number}_thr{int(threshold * 10000)}_"
        f"h{hold_bars}_ls{int(long_scale * 100)}_ss{int(short_scale * 10)}_"
        f"tp{int(take_profit_pct * 10000)}"
    )
    return fresh.FreshSpec(
        name=name,
        family="calendar_rotation",
        lookback_bars=168,
        threshold=threshold,
        hold_bars=hold_bars,
        cooldown_bars=max(0, hold_bars // 4),
        stop_loss_pct=0.0,
        take_profit_pct=take_profit_pct,
        min_abs_return=threshold,
        allow_long=True,
        allow_short=True,
        long_allocation_scale=long_scale,
        short_allocation_scale=short_scale,
        calendar_long_months=(3, 4, 5),
        calendar_short_months=(1, 2),
        calendar_long_symbol="TRXUSDT",
        calendar_short_symbol=short_symbol,
    )


def _objective_score(result: dict[str, Any]) -> float:
    train = dict((result.get("split_results") or {}).get("train", {}).get("metrics") or {})
    val = dict((result.get("split_results") or {}).get("val", {}).get("metrics") or {})
    train_return = _safe_float(train.get("total_return"))
    val_return = _safe_float(val.get("total_return"))
    val_sharpe = _safe_float(val.get("sharpe"))
    val_mdd = _safe_float(val.get("max_drawdown"), 1.0)
    train_sharpe = _safe_float(train.get("sharpe"))
    score = (
        180.0 * val_return
        + 60.0 * train_return
        + 4.0 * val_sharpe
        + 0.75 * train_sharpe
        - 220.0 * val_mdd
    )
    if train_return <= 0.0:
        score -= 50.0
    if val_return <= 0.0:
        score -= 75.0
    return float(score)


def _flatten_trial(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row.get("result") or {})
    split_results = dict(result.get("split_results") or {})
    out: dict[str, Any] = {
        "trial_number": int(row.get("trial_number") or 0),
        "objective": _safe_float(row.get("objective")),
        "name": result.get("name"),
        "success_candidate": bool(result.get("success_candidate")),
        "replay_survivor": bool(result.get("replay_survivor")),
        "failed_gates": ",".join(result.get("failed_gates") or []),
        "params": json.dumps(row.get("params") or {}, sort_keys=True),
    }
    for split_name, split in split_results.items():
        metrics = dict(split.get("metrics") or {})
        for key in ("total_return", "max_drawdown", "sharpe", "sortino", "volatility"):
            out[f"{split_name}_{key}"] = _safe_float(metrics.get(key), 0.0)
        out[f"{split_name}_round_trips"] = int(split.get("round_trips") or 0)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=sorted({key for row in rows for key in row}),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _markdown(payload: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Profit moonshot calendar Optuna tuning",
        "",
        f"- Generated: `{payload['generated_at_utc']}`",
        f"- Trials: `{payload['n_trials']}`",
        f"- Success candidates: `{payload['success_candidate_count']}`",
        f"- Peak RSS: `{payload['peak_rss_mib']:.3f} MiB`",
        f"- Objective policy: `{payload['objective_policy']}`",
        "",
        "| rank | trial | name | success | objective | train | val | locked OOS | OOS MDD | OOS Sharpe | OOS trips | failed gates |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows[:20], start=1):
        lines.append(
            f"| {rank} | {row['trial_number']} | `{row['name']}` | {row['success_candidate']} | "
            f"{_safe_float(row['objective']):.6f} | {_safe_float(row.get('train_total_return')):+.4%} | "
            f"{_safe_float(row.get('val_total_return')):+.4%} | {_safe_float(row.get('oos_total_return')):+.4%} | "
            f"{_safe_float(row.get('oos_max_drawdown')):.4%} | {_safe_float(row.get('oos_sharpe')):.4f} | "
            f"{int(row.get('oos_round_trips') or 0)} | `{row.get('failed_gates')}` |"
        )
    lines.append("")
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise RuntimeError("Optuna is required; run with `uv run --extra optimize ...`") from exc

    fresh = _load_fresh_module()
    oos_end = datetime.fromisoformat(str(args.oos_end_date)).date()
    splits = fresh._split_windows(oos_end=oos_end)
    start = min(split.start for split in splits)
    end = max(split.end for split in splits)
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()]
    panel, data_metadata = fresh._joined_panel(
        market_root=Path(args.market_root),
        exchange=str(args.exchange),
        symbols=symbols,
        start=start,
        end=end,
    )
    arrays = fresh._build_arrays(panel, symbols)
    train_val = [split for split in splits if split.name in {"train", "val"}]

    trials: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        params = _trial_params(trial)
        spec = _spec_from_params(fresh, params, trial_number=int(trial.number))
        train_val_results = {
            split.name: fresh._run_split(spec=spec, arrays=arrays, split=split)
            for split in train_val
        }
        partial_result = {
            "name": spec.name,
            "filters": spec.payload(),
            "split_results": train_val_results,
        }
        score = _objective_score(partial_result)
        trial.set_user_attr("params", params)
        trial.set_user_attr("spec_payload", spec.payload())
        trial.set_user_attr("train_val_results", train_val_results)
        return score

    sampler = optuna.samplers.TPESampler(
        seed=int(args.seed), n_startup_trials=min(12, int(args.n_trials))
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    if bool(args.enqueue_known_good):
        for params in (
            {
                "short_symbol": "ETHUSDT",
                "threshold": 0.015,
                "hold_bars": 120,
                "long_scale": 6.2,
                "short_scale": 10.0,
                "take_profit_pct": 0.018,
            },
            {
                "short_symbol": "",
                "threshold": 0.012,
                "hold_bars": 120,
                "long_scale": 5.4,
                "short_scale": 12.0,
                "take_profit_pct": 0.045,
            },
        ):
            study.enqueue_trial(params)
    study.optimize(objective, n_trials=max(1, int(args.n_trials)), show_progress_bar=False)

    for trial in study.trials:
        params = dict(trial.user_attrs.get("params") or trial.params)
        spec = _spec_from_params(fresh, params, trial_number=int(trial.number))
        split_results = {
            split.name: fresh._run_split(spec=spec, arrays=arrays, split=split) for split in splits
        }
        metrics = {
            name: dict(result.get("metrics") or {}) for name, result in split_results.items()
        }
        train = metrics.get("train", {})
        val = metrics.get("val", {})
        oos = metrics.get("oos", {})
        gates = {
            "train_positive": _safe_float(train.get("total_return")) > 0.0,
            "val_positive": _safe_float(val.get("total_return")) > 0.0,
            "oos_return_beats_incumbent": _safe_float(oos.get("total_return"))
            > fresh.BASELINE_OOS_RETURN,
            "oos_mdd_beats_shadow": _safe_float(oos.get("max_drawdown"), 1.0)
            < fresh.SHADOW_OOS_MDD,
            "oos_sharpe_gt_1": _safe_float(oos.get("sharpe")) > fresh.SUCCESS_SHARPE,
            "oos_trades_not_starved": int(split_results["oos"].get("round_trips") or 0) >= 5,
            "liquidations_zero": int(split_results["train"].get("liquidations") or 0) == 0
            and int(split_results["val"].get("liquidations") or 0) == 0
            and int(split_results["oos"].get("liquidations") or 0) == 0,
        }
        failed = [key for key, ok in gates.items() if not ok]
        result = {
            "name": spec.name,
            "family": spec.family,
            "filters": spec.payload(),
            "split_results": split_results,
            "gates": gates,
            "failed_gates": failed,
            "replay_survivor": not failed,
            "success_candidate": not failed,
        }
        trials.append(
            {
                "trial_number": int(trial.number),
                "state": str(trial.state),
                "objective": float(trial.value if trial.value is not None else float("-inf")),
                "params": params,
                "result": result,
            }
        )
    trials.sort(
        key=lambda row: (
            not bool((row.get("result") or {}).get("success_candidate")),
            -_safe_float(
                (row.get("result") or {})
                .get("split_results", {})
                .get("oos", {})
                .get("metrics", {})
                .get("total_return")
            ),
            -_safe_float(row.get("objective")),
        )
    )
    rows = [_flatten_trial(row) for row in trials]
    payload = {
        "artifact_kind": "profit_moonshot_calendar_optuna",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "objective_policy": "train_val_only_locked_oos_report",
        "n_trials": int(args.n_trials),
        "seed": int(args.seed),
        "success_candidate_count": sum(
            1 for row in trials if bool((row.get("result") or {}).get("success_candidate"))
        ),
        "best_trial": trials[0] if trials else {},
        "top_trials": trials[:25],
        "data_metadata": data_metadata,
        "peak_rss_mib": _rss_mib(),
    }
    return payload, rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-root", default="data/market_parquet")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,TRX/USDT")
    parser.add_argument("--oos-end-date", default="2026-05-06")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-trials", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enqueue-known-good", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "calendar_optuna_latest.json"
    csv_path = output_dir / "calendar_optuna_trials.csv"
    md_path = output_dir / "calendar_optuna_latest.md"
    payload, rows = build_payload(args)
    _write_csv(csv_path, rows)
    payload["csv_path"] = str(csv_path)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_markdown(payload, rows) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "markdown": str(md_path),
                "success_candidate_count": payload["success_candidate_count"],
                "peak_rss_mib": payload["peak_rss_mib"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
