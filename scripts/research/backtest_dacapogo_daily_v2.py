#!/usr/bin/env python3
"""Faithful Binance port of Dacapogo's pure-daily causal OLS research model."""

from __future__ import annotations

import argparse
import csv
import fcntl
from bisect import bisect_left
import hashlib
import json
import resource
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, UTC
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from lumina_quant.research.run_card import atomic_output_path, runtime_provenance

UPSTREAM_HEAD = "633ba5d6bc0c84a20696af6b2bf807cf55d21248"
UPSTREAM_INTRODUCING_COMMIT = "38b589ed4ec19a056aa51b0a23e05e9484d05636"
UPSTREAM_RESULT_COMMIT = "77d18c26c12ccc74e2bdaa65f5a9f91e172b0215"
UPSTREAM_SOURCE_SHA256 = "18a3a3e201c594f2c665eff3eeb4874474e2cc6e0c732381471b29ba90977646"
COST = 0.0015
PARITY_START = "2026-04-01"
PARITY_END = "2026-07-21"
DIAGNOSTIC_START = "2026-07-22"
FEATURE_NAMES = (
    "prior_oc_return_robust",
    "prior_cc_return_robust",
    "prior_range_robust",
    "prior_turnover_own_percentile",
    "prior_turnover_cross_percentile",
)
DAILY_FIELDS = (
    "date",
    "decision_date",
    "eligible_slots",
    "baseline_filled_slots",
    "selected_slots",
    "filled_slots",
    "missing_trade_slots",
    "baseline_exposure",
    "ml_exposure",
    "cash_exposure",
    "baseline_1x",
    "baseline_2x",
    "ml_1x",
    "ml_2x",
    "cash",
)
TRADE_FIELDS = (
    "decision_date",
    "trade_date",
    "market",
    "open",
    "close",
    "gross_return",
    "predicted_gross_return",
    "slot_weight",
    "net_return_1x",
    "net_return_2x",
)


@dataclass(frozen=True)
class OLSModel:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray

    def predict(self, x: ArrayLike) -> float | np.ndarray:
        values = np.asarray(x, dtype=float)
        one = values.ndim == 1
        if one:
            values = values[None, :]
        predicted = np.c_[np.ones(len(values)), (values - self.mean) / self.scale] @ self.coef
        return float(predicted[0]) if one else predicted

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(FEATURE_NAMES),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef_intercept_first": self.coef.tolist(),
        }


def fit_ols(x: ArrayLike, y: ArrayLike) -> OLSModel:
    """Parameter-free OLS with preprocessing learned from training rows only."""
    values, target = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != len(FEATURE_NAMES)
        or len(values) != len(target)
        or not len(values)
    ):
        raise ValueError("non-empty aligned feature matrix required")
    if not np.isfinite(values).all() or not np.isfinite(target).all():
        raise ValueError("non-finite training data")
    mean, scale = values.mean(axis=0), values.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    coef, *_ = np.linalg.lstsq(
        np.c_[np.ones(len(values)), (values - mean) / scale], target, rcond=None
    )
    return OLSModel(mean, scale, coef)


def _percentile(value: float, prior: Sequence[float]) -> float:
    values = np.asarray(prior, dtype=float)
    if not len(values):
        return 0.5
    return float(
        (np.count_nonzero(values < value) + 0.5 * np.count_nonzero(values == value)) / len(values)
    )


def _robust(value: float, prior: Sequence[float]) -> float:
    values = np.asarray(prior, dtype=float)
    if not len(values):
        return 0.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return (value - median) / (1.4826 * mad) if mad > 0 else 0.0


def _file_identity(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _sha(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    with atomic_output_path(path) as temporary, temporary.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n"
        )


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    with (
        atomic_output_path(path) as temporary,
        temporary.open("w", encoding="utf-8", newline="") as handle,
    ):
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_panel(
    panel_path: Path, manifest_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and validate the existing sealed Binance USD-M daily panel."""
    rows, manifest, _ = _load_panel_snapshot(panel_path, manifest_path)
    return rows, manifest


def _load_panel_snapshot(
    panel_path: Path, manifest_path: Path, *, already_locked: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, int | str]]]:
    if already_locked:
        return _load_panel_unlocked(panel_path, manifest_path)
    with (panel_path.parent / ".run.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        return _load_panel_unlocked(panel_path, manifest_path)


def _load_panel_unlocked(
    panel_path: Path, manifest_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, int | str]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("file") != _file_identity(panel_path):
        raise ValueError("daily panel cache does not match its integrity manifest")
    audits = manifest.get("audits")
    if not isinstance(audits, list) or not audits:
        raise ValueError("daily panel manifest has no audits")
    starts = {str(row["requested_start"]) for row in audits}
    ends = {str(row["end"]) for row in audits}
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("daily panel audits have inconsistent bounds")
    panel = pl.read_parquet(
        panel_path, columns=["market", "date", "value", "open", "high", "low", "close"]
    )
    module_name = (
        "scripts.research.backtest_dacapogo_daily_ml"
        if __package__
        else "backtest_dacapogo_daily_ml"
    )
    validator = __import__(module_name, fromlist=["_validate_panel"])._validate_panel
    validator(
        panel,
        tuple(str(row["symbol"]) for row in audits),
        date.fromisoformat(starts.pop()),
        date.fromisoformat(ends.pop()),
        audits,
    )
    rows = [
        {
            **row,
            "date": row["date"].isoformat(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "value": float(row["value"]),
        }
        for row in panel.sort(["market", "date"]).iter_rows(named=True)
    ]
    return (
        rows,
        manifest,
        {
            panel_path.name: _file_identity(panel_path),
            manifest_path.name: _file_identity(manifest_path),
        },
    )


def build_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build candidates from source D, decided D+1, traded D+2."""
    by_market: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_market.setdefault(str(row["market"]), []).append(row)

    provisional: list[dict[str, Any]] = []
    for market, values in sorted(by_market.items()):
        values.sort(key=lambda row: str(row["date"]))
        by_day = {str(row["date"]): row for row in values}
        oc: list[float] = []
        cc: list[float] = []
        ranges: list[float] = []
        turnovers: list[float] = []
        for index, row in enumerate(values):
            current_oc = float(row["close"]) / float(row["open"]) - 1
            current_range = (float(row["high"]) - float(row["low"])) / float(row["open"])
            source_day = date.fromisoformat(str(row["date"]))
            adjacent = bool(
                index
                and date.fromisoformat(str(values[index - 1]["date"]))
                == source_day - timedelta(days=1)
            )
            current_cc = (
                float(row["close"]) / float(values[index - 1]["close"]) - 1 if adjacent else None
            )
            if current_cc is not None:
                target_day = source_day + timedelta(days=2)
                target = by_day.get(target_day.isoformat())
                provisional.append(
                    {
                        "market": market,
                        "date": target_day.isoformat(),
                        "target_date": target_day.isoformat(),
                        "decision_date": (source_day + timedelta(days=1)).isoformat(),
                        "source_date": str(row["date"]),
                        "x4": [
                            _robust(current_oc, oc),
                            _robust(current_cc, cc),
                            _robust(current_range, ranges),
                            _percentile(float(row["value"]), turnovers),
                        ],
                        "prior_value": float(row["value"]),
                        "gross_return": (
                            float(target["close"]) / float(target["open"]) - 1 if target else None
                        ),
                        "open": float(target["open"]) if target else None,
                        "close": float(target["close"]) if target else None,
                    }
                )
            oc.append(current_oc)
            ranges.append(current_range)
            turnovers.append(float(row["value"]))
            if current_cc is not None:
                cc.append(current_cc)

    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in provisional:
        by_source.setdefault(str(row["source_date"]), []).append(row)
    examples: list[dict[str, Any]] = []
    for source_date in sorted(by_source):
        group = by_source[source_date]
        values = [float(row["prior_value"]) for row in group]
        for row in sorted(group, key=lambda item: str(item["market"])):
            row["x"] = [*row["x4"], _percentile(float(row["prior_value"]), values)]
            examples.append(row)
    if not examples:
        raise ValueError("daily history is too short to build causal examples")
    return examples


def walk_forward(
    examples: list[dict[str, Any]], start: str, end: str, cost: float = COST
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exact expanding daily walk-forward with fixed eligible-universe weights."""
    examples = sorted(examples, key=lambda row: (str(row["date"]), str(row["market"])))
    days = [
        (date.fromisoformat(start) + timedelta(days=offset)).isoformat()
        for offset in range((date.fromisoformat(end) - date.fromisoformat(start)).days + 1)
    ]
    by_date: dict[str, list[dict[str, Any]]] = {}
    matured: list[dict[str, Any]] = []
    for row in examples:
        by_date.setdefault(str(row["date"]), []).append(row)
        if row["gross_return"] is not None:
            matured.append(row)
    matured_dates = [str(row["date"]) for row in matured]
    matured_x = np.asarray([row["x"] for row in matured], dtype=float)
    matured_y = np.asarray([float(row["gross_return"]) for row in matured], dtype=float)
    daily: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    for day in days:
        eligible = sorted(by_date.get(day, []), key=lambda row: str(row["market"]))
        if not eligible:
            raise ValueError(f"no prior-session-declared eligible universe for {day}")
        decision_dates = {str(row["decision_date"]) for row in eligible}
        if len(decision_dates) != 1:
            raise ValueError(f"inconsistent decision dates for {day}")
        decision_date = decision_dates.pop()
        if date.fromisoformat(decision_date) != date.fromisoformat(day) - timedelta(days=1):
            raise ValueError(f"decision date must be one day before trade date {day}")
        training_rows = bisect_left(matured_dates, decision_date)
        if not training_rows:
            raise ValueError(f"no matured training labels before {decision_date}")
        model = fit_ols(matured_x[:training_rows], matured_y[:training_rows])
        predictions = np.asarray(model.predict([row["x"] for row in eligible]), dtype=float)
        n = len(eligible)
        sums = {"baseline_1x": 0.0, "baseline_2x": 0.0, "ml_1x": 0.0, "ml_2x": 0.0}
        selected = filled = 0
        for row, predicted in zip(eligible, predictions, strict=True):
            take = float(predicted) - 2 * cost > 0
            selected += int(take)
            if row["gross_return"] is None:
                continue
            gross = float(row["gross_return"])
            sums["baseline_1x"] += (gross - cost) / n
            sums["baseline_2x"] += (gross - 2 * cost) / n
            if take:
                filled += 1
                sums["ml_1x"] += (gross - cost) / n
                sums["ml_2x"] += (gross - 2 * cost) / n
                trades.append(
                    {
                        "decision_date": decision_date,
                        "trade_date": day,
                        "market": row["market"],
                        "open": row["open"],
                        "close": row["close"],
                        "gross_return": gross,
                        "predicted_gross_return": float(predicted),
                        "slot_weight": 1 / n,
                        "net_return_1x": gross - cost,
                        "net_return_2x": gross - 2 * cost,
                    }
                )
        baseline_filled = sum(row["gross_return"] is not None for row in eligible)
        daily.append(
            {
                "date": day,
                "decision_date": decision_date,
                "eligible_slots": n,
                "baseline_filled_slots": baseline_filled,
                "selected_slots": selected,
                "filled_slots": filled,
                "missing_trade_slots": sum(row["gross_return"] is None for row in eligible),
                "baseline_exposure": baseline_filled / n,
                "ml_exposure": filled / n,
                "cash_exposure": 0.0,
                **sums,
                "cash": 0.0,
            }
        )
    return daily, trades


def _metrics(daily: list[dict[str, Any]], return_key: str, exposure_key: str) -> dict[str, Any]:
    values = np.asarray([row[return_key] for row in daily], dtype=float)
    exposure = np.asarray([row[exposure_key] for row in daily], dtype=float)
    equity = np.cumprod(1 + values)
    peak = np.maximum.accumulate(np.r_[1.0, equity])
    drawdown = np.r_[1.0, equity] / peak - 1
    return {
        "days": len(values),
        "cumulative_return": float(equity[-1] - 1) if len(values) else 0.0,
        "mean_daily_return": float(values.mean()) if len(values) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(values) else 0.0,
        "trades": int(
            sum(
                int(
                    row["filled_slots"]
                    if return_key.startswith("ml_")
                    else row["baseline_filled_slots"]
                )
                for row in daily
            )
        )
        if return_key != "cash"
        else 0,
        "active_days": int(np.count_nonzero(exposure > 0)),
        "positive_days": int(np.count_nonzero(values > 0)),
        "worst_day": float(values.min()) if len(values) else 0.0,
        "average_gross_exposure": float(exposure.mean()) if len(exposure) else 0.0,
    }


def _weekly_median(daily: list[dict[str, Any]]) -> tuple[float, dict[str, float]]:
    weeks: dict[str, list[dict[str, Any]]] = {}
    for row in daily:
        day = date.fromisoformat(str(row["date"]))
        weeks.setdefault(f"{day.isocalendar().year}-W{day.isocalendar().week:02d}", []).append(row)
    returns = {
        week: float(np.prod(np.asarray([row["ml_2x"] for row in rows], dtype=float) + 1) - 1)
        for week, rows in weeks.items()
        if len(rows) == 7
        and date.fromisoformat(str(rows[0]["date"])).weekday() == 0
        and date.fromisoformat(str(rows[-1]["date"])).weekday() == 6
    }
    return (float(np.median(list(returns.values()))) if returns else 0.0), returns


def _series(daily: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        "baseline_1x": _metrics(daily, "baseline_1x", "baseline_exposure"),
        "baseline_2x": _metrics(daily, "baseline_2x", "baseline_exposure"),
        "ml_1x": _metrics(daily, "ml_1x", "ml_exposure"),
        "ml_2x": _metrics(daily, "ml_2x", "ml_exposure"),
        "cash": _metrics(daily, "cash", "cash_exposure"),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    provenance = runtime_provenance(
        repo_root=Path(__file__).resolve().parents[2],
        packages=("numpy", "polars"),
        source_files=(Path(__file__),),
    )
    adapter_sha256 = str(provenance["source_files"][str(Path(__file__).resolve())]["sha256"])
    panel_path = Path(args.panel_cache)
    manifest_path = Path(args.panel_manifest or f"{panel_path}.manifest.json")
    output_dir = Path(args.output_dir)
    rows, _, inputs = _load_panel_snapshot(
        panel_path,
        manifest_path,
        already_locked=panel_path.parent.resolve() == output_dir.resolve(),
    )
    panel_rows = len(rows)
    panel_end = max(str(row["date"]) for row in rows)
    try:
        requested_end = date.fromisoformat(args.seen_through or panel_end).isoformat()
    except ValueError as exc:
        raise ValueError("seen_through must be an ISO date") from exc
    diagnostic_end = min(requested_end, panel_end)
    if diagnostic_end < PARITY_END:
        raise ValueError(f"panel must cover parity end {PARITY_END}")
    examples = build_examples(rows)
    del rows
    daily, trades = walk_forward(examples, PARITY_START, diagnostic_end)
    parity_daily = [row for row in daily if str(row["date"]) <= PARITY_END]
    diagnostic_daily = [row for row in daily if str(row["date"]) >= DIAGNOSTIC_START]
    parity_trades = [row for row in trades if str(row["trade_date"]) <= PARITY_END]
    weekly_median, weekly = _weekly_median(parity_daily)
    parity_metrics = _series(parity_daily)
    gate = parity_metrics["ml_2x"]["cumulative_return"] > 0 and weekly_median > 0
    training = [
        row
        for row in examples
        if row["gross_return"] is not None and str(row["date"]) <= diagnostic_end
    ]
    final_model = fit_ols(
        [row["x"] for row in training], [float(row["gross_return"]) for row in training]
    )
    limitations = [
        "retrospective expanding walk-forward pseudo-OOS; not true forward OOS",
        "current Binance USD-M universe has survivorship and listing-history limitations",
        "00:00 UTC (09:00 KST) daily opens/closes are idealized bar-boundary proxies, not fill claims",
    ]
    lock = {
        "schema": "dacapogo_binance_daily_v2",
        "strategy_tier": "research_only",
        "promotion_eligible": False,
        "deploy_action": "cash",
        "upstream": {
            "head": UPSTREAM_HEAD,
            "introducing_commit": UPSTREAM_INTRODUCING_COMMIT,
            "result_commit": UPSTREAM_RESULT_COMMIT,
            "source_file": "crypto_daily_model.py",
            "source_sha256": UPSTREAM_SOURCE_SHA256,
        },
        "local_adapter_sha256": adapter_sha256,
        "decision_time": "00:00 UTC / 09:00 KST",
        "decision_schedule": "trade D uses source D-2, is decided D-1, and holds D open-to-close",
        "intraday_rules": None,
        "cost_round_trip_1x": COST,
        "cost_stress_2x": 2 * COST,
        "cost_stress_note": "_2x means doubled COST friction stress, not 2x leverage",
        "selection": "predicted gross return - 2x cost > 0; fixed 1/N eligible slots",
        "model": final_model.to_dict(),
        "model_target": "gross_open_to_close_return",
        "training_rows": len(training),
        "parity_period": {"start": PARITY_START, "end": PARITY_END},
        "weekly_median_2x": weekly_median,
        "gate_pass": gate,
        "research_replay_action": "ml" if gate else "cash",
        "inputs": inputs,
        "limitations": limitations,
    }
    lock["lock_sha256"] = _sha(lock)
    report = {
        "artifact_kind": "dacapogo_binance_daily_v2_research",
        "strategy_tier": "research_only",
        "promotion_eligible": False,
        "deploy_action": "cash",
        "publication": {
            "contract": "atomic file replacement; manifest seal written last",
            "authority": "dacapogo_binance_daily_v2_manifest.json",
        },
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "classification": "retrospective expanding daily walk-forward pseudo-OOS; not true OOS",
        "upstream": lock["upstream"],
        "local_adapter_sha256": adapter_sha256,
        "cost_round_trip_1x": COST,
        "cost_stress_2x": 2 * COST,
        "cost_stress_note": "_2x means doubled COST friction stress, not 2x leverage",
        "parity": {
            "period": {"start": PARITY_START, "end": PARITY_END},
            "metrics": parity_metrics,
            "weekly_returns_ml_2x": weekly,
            "weekly_median_ml_2x": weekly_median,
            "gate_pass": gate,
            "research_replay_action": "ml" if gate else "cash",
            "research_replay_result": parity_metrics["ml_2x"] if gate else parity_metrics["cash"],
            "research_ml_trades": len(parity_trades),
        },
        "post_parity_diagnostic": {
            "label": "optional recent retrospective diagnostic; not true OOS",
            "period": {"start": DIAGNOSTIC_START, "end": diagnostic_end},
            "metrics": _series(diagnostic_daily),
            "research_ml_trades": len(trades) - len(parity_trades),
        },
        "inputs": inputs,
        "panel": {"rows": panel_rows, "end": panel_end},
        "limitations": limitations,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "compute": "sealed Parquet read, exact prior-only NumPy features, batched daily OLS prediction",
            "provenance": provenance,
        },
    }
    prefix = output_dir / "dacapogo_binance_daily_v2"
    lock_path = prefix.with_name(f"{prefix.name}_lock.json")
    daily_path = prefix.with_name(f"{prefix.name}_daily.csv")
    trades_path = prefix.with_name(f"{prefix.name}_trades.csv")
    summary_path = prefix.with_name(f"{prefix.name}_summary.json")
    manifest_out = prefix.with_name(f"{prefix.name}_manifest.json")
    report["artifacts"] = {
        "integrity": f"sealed by {manifest_out.name}",
        "files": [
            path.name for path in (lock_path, daily_path, trades_path, summary_path, manifest_out)
        ],
    }
    _atomic_json(lock_path, lock)
    _atomic_csv(daily_path, daily, DAILY_FIELDS)
    _atomic_csv(trades_path, trades, TRADE_FIELDS)
    _atomic_json(summary_path, report)
    _atomic_json(
        manifest_out,
        {
            "schema": "dacapogo_binance_daily_v2_manifest",
            "hash_algorithm": "sha256",
            "inputs": inputs,
            "files": {
                path.name: _file_identity(path)
                for path in (lock_path, daily_path, trades_path, summary_path)
            },
        },
    )
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".run.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _run(args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel-cache", default="var/reports/dacapogo_binance/daily_source/daily_panel.parquet"
    )
    parser.add_argument("--panel-manifest")
    parser.add_argument("--seen-through")
    parser.add_argument("--output-dir", default="var/reports/dacapogo_binance/daily_v2")
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run(build_arg_parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
