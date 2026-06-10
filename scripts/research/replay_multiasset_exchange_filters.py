#!/usr/bin/env python3
"""Stateful replay for Hyperliquid/Tickmill read-only filters on ETH 12h shock reversion."""

from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from lumina_quant.configuration import BacktestConfigView, get_default_runtime_config
from lumina_quant.market_data import load_futures_feature_points_from_db

_lq_rt = get_default_runtime_config()
BacktestConfig = BacktestConfigView(_lq_rt)
BaseConfig = BacktestConfigView(_lq_rt)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.replay_eth_shock_filters import (  # noqa: E402
    FUNDING_GUARD_OOS_MDD,
    FUNDING_GUARD_OOS_SHARPE,
    BASELINE_OOS_RETURN,
    _build_arrays,
    _fill_price,
    _joined_panel,
    _rolling_zscore,
)
from scripts.research.revalidate_live_equivalent_candidates import (  # noqa: E402
    SplitWindow,
    _metrics_from_equity_totals,
    _safe_float,
    _split_windows,
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "var/reports/profit_moonshot_20260501/current_tail_20260506/multiasset_exchange_expansion"
)


@dataclass(frozen=True, slots=True)
class ExchangeReplaySpec:
    name: str
    return_threshold: float
    excluded_hours: tuple[int, ...] = ()
    cooldown_bars: int = 0
    hl_funding_abs_cap: float = 0.0
    hl_require_funding_sign_match: bool = False
    hl_oi_z_min: float = 0.0
    hl_basis_abs_max: float = 0.0
    tickmill_macro_filter: str = ""

    def filters_payload(self) -> dict[str, Any]:
        return {
            "return_threshold": self.return_threshold,
            "excluded_hours": list(self.excluded_hours),
            "cooldown_bars": self.cooldown_bars,
            "hl_funding_abs_cap": self.hl_funding_abs_cap,
            "hl_require_funding_sign_match": self.hl_require_funding_sign_match,
            "hl_oi_z_min": self.hl_oi_z_min,
            "hl_basis_abs_max": self.hl_basis_abs_max,
            "tickmill_macro_filter": self.tickmill_macro_filter,
        }


def _parse_date(raw: str | None) -> date | None:
    token = str(raw or "").strip()
    if not token:
        return None
    return datetime.fromisoformat(token).date()


def _latest_complete_utc_day() -> date:
    configured = str(getattr(BacktestConfig, "END_DATE", "") or "").strip()
    if configured:
        try:
            return datetime.fromisoformat(configured).date()
        except Exception:
            pass
    return datetime.now(UTC).date() - timedelta(days=1)


def _iso(timestamp_ms: int | None) -> str | None:
    if timestamp_ms is None:
        return None
    return datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=UTC).isoformat()


def _load_hyperliquid_feature_hourly(
    *,
    market_root: Path,
    start: date,
    end: date,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    frame = load_futures_feature_points_from_db(
        str(market_root),
        exchange="hyperliquid",
        symbol="ETH/USDT",
        start_date=datetime.combine(start, datetime.min.time()),
        end_date=datetime.combine(end + timedelta(days=1), datetime.min.time()),
    )
    if frame.is_empty():
        empty = pl.DataFrame({"datetime": []}, schema={"datetime": pl.Datetime(time_unit="ms")})
        return empty, {
            "rows": 0,
            "has_funding": False,
            "has_open_interest_history": False,
            "has_mark_history": False,
        }
    aligned = frame
    for column in ("timestamp_ms", "funding_rate", "mark_price", "index_price", "open_interest"):
        if column not in aligned.columns:
            dtype = pl.Int64 if column == "timestamp_ms" else pl.Float64
            aligned = aligned.with_columns(pl.lit(None, dtype=dtype).alias(column))
    aligned = aligned.select(
        ["timestamp_ms", "funding_rate", "mark_price", "index_price", "open_interest"]
    ).filter(pl.col("timestamp_ms").is_not_null())
    if aligned.is_empty():
        empty = pl.DataFrame({"datetime": []}, schema={"datetime": pl.Datetime(time_unit="ms")})
        return empty, {
            "rows": 0,
            "has_funding": False,
            "has_open_interest_history": False,
            "has_mark_history": False,
        }
    metadata = {
        "rows": int(aligned.height),
        "first_timestamp_ms": int(aligned.select(pl.min("timestamp_ms")).item()),
        "last_timestamp_ms": int(aligned.select(pl.max("timestamp_ms")).item()),
        "funding_rows": int(aligned.select(pl.col("funding_rate").is_not_null().sum()).item()),
        "open_interest_rows": int(
            aligned.select(pl.col("open_interest").is_not_null().sum()).item()
        ),
        "mark_rows": int(aligned.select(pl.col("mark_price").is_not_null().sum()).item()),
    }
    metadata["first_timestamp_utc"] = _iso(metadata["first_timestamp_ms"])
    metadata["last_timestamp_utc"] = _iso(metadata["last_timestamp_ms"])
    metadata["has_funding"] = metadata["funding_rows"] > 0
    # A current snapshot row does not constitute train/val/OOS OI/mark history.
    metadata["has_open_interest_history"] = metadata["open_interest_rows"] > 24
    metadata["has_mark_history"] = metadata["mark_rows"] > 24
    hourly = (
        aligned.with_columns(
            pl.from_epoch(pl.col("timestamp_ms"), time_unit="ms").alias("datetime")
        )
        .sort("datetime")
        .group_by_dynamic("datetime", every="1h", period="1h", closed="left", label="left")
        .agg(
            [
                pl.col("funding_rate").drop_nulls().last().alias("hl_funding_rate"),
                pl.col("mark_price").drop_nulls().last().alias("hl_mark_price"),
                pl.col("index_price").drop_nulls().last().alias("hl_index_price"),
                pl.col("open_interest").drop_nulls().last().alias("hl_open_interest"),
            ]
        )
        .sort("datetime")
    )
    return hourly, metadata


def _joined_exchange_panel(
    *,
    market_root: Path,
    exchange: str,
    start: date,
    end: date,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    panel, data_metadata = _joined_panel(
        market_root=market_root, exchange=exchange, start=start, end=end
    )
    hl_features, hl_metadata = _load_hyperliquid_feature_hourly(
        market_root=market_root, start=start, end=end
    )
    if not hl_features.is_empty():
        panel = panel.join(hl_features, on="datetime", how="left")
    for column in ("hl_funding_rate", "hl_mark_price", "hl_index_price", "hl_open_interest"):
        if column not in panel.columns:
            panel = panel.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
    return panel.sort("datetime"), {**data_metadata, "hyperliquid_eth_feature_points": hl_metadata}


def _build_exchange_arrays(panel: pl.DataFrame) -> dict[str, Any]:
    arrays = _build_arrays(panel)

    def _ffill(values: np.ndarray) -> np.ndarray:
        out = np.asarray(values, dtype=float).copy()
        last = np.nan
        for idx, value in enumerate(out):
            if math.isfinite(float(value)):
                last = float(value)
            else:
                out[idx] = last
        return out

    arrays["binance_funding_rate_ffill"] = _ffill(arrays["funding_rate"])
    arrays["hl_funding_rate_ffill"] = _ffill(arrays["hl_funding_rate"])
    hl_oi = arrays.get("hl_open_interest")
    if hl_oi is None:
        hl_oi = np.full(panel.height, np.nan, dtype=float)
    arrays["hl_oi_delta"] = np.diff(hl_oi, prepend=np.nan)
    arrays["hl_oi_delta_z_72h"] = _rolling_zscore(arrays["hl_oi_delta"], 72)
    hl_mark = arrays.get("hl_mark_price")
    eth_close = arrays["ethusdt_close"]
    if hl_mark is None:
        arrays["hl_mark_basis"] = np.full(panel.height, np.nan, dtype=float)
    else:
        arrays["hl_mark_basis"] = (
            np.divide(
                hl_mark,
                eth_close,
                out=np.full(eth_close.shape, np.nan, dtype=float),
                where=np.isfinite(hl_mark) & np.isfinite(eth_close) & (eth_close > 0.0),
            )
            - 1.0
        )
    return arrays


def _hour_utc(dt: datetime) -> int:
    return int(dt.replace(tzinfo=UTC).hour)


def _filter_passes(spec: ExchangeReplaySpec, arrays: dict[str, Any], idx: int) -> tuple[bool, str]:
    dt = arrays["datetime"][idx]
    if spec.excluded_hours and _hour_utc(dt) in set(spec.excluded_hours):
        return False, "funding_hour_excluded"
    hl_funding = float(arrays["hl_funding_rate_ffill"][idx])
    if spec.hl_funding_abs_cap > 0.0:
        if not math.isfinite(hl_funding):
            return False, "hl_funding_missing"
        if abs(hl_funding) > spec.hl_funding_abs_cap:
            return False, "hl_funding_abs_cap"
    if spec.hl_require_funding_sign_match:
        binance_funding = float(arrays["binance_funding_rate_ffill"][idx])
        if not math.isfinite(hl_funding) or not math.isfinite(binance_funding):
            return False, "hl_or_binance_funding_missing"
        if hl_funding == 0.0 or binance_funding == 0.0:
            return False, "funding_sign_zero"
        if math.copysign(1.0, hl_funding) != math.copysign(1.0, binance_funding):
            return False, "hl_binance_funding_sign_divergence"
    if spec.hl_oi_z_min > 0.0:
        oi_z = float(arrays["hl_oi_delta_z_72h"][idx])
        if not math.isfinite(oi_z):
            return False, "hl_oi_history_missing"
        if abs(oi_z) < spec.hl_oi_z_min:
            return False, "hl_oi_exhaustion_missing"
    if spec.hl_basis_abs_max > 0.0:
        basis = float(arrays["hl_mark_basis"][idx])
        if not math.isfinite(basis):
            return False, "hl_mark_history_missing"
        if abs(basis) > spec.hl_basis_abs_max:
            return False, "hl_mark_basis_stress"
    if spec.tickmill_macro_filter:
        return False, "tickmill_mt5_macro_data_unavailable"
    return True, ""


def _run_split(
    *,
    spec: ExchangeReplaySpec,
    arrays: dict[str, Any],
    split: SplitWindow,
    initial_equity: float = 10_000.0,
) -> dict[str, Any]:
    timestamps = arrays["timestamp"]
    start_ts = int(datetime.combine(split.start, datetime.min.time(), tzinfo=UTC).timestamp())
    end_ts = (
        int(
            datetime.combine(
                split.end + timedelta(days=1), datetime.min.time(), tzinfo=UTC
            ).timestamp()
        )
        - 1
    )
    indices = np.flatnonzero((timestamps >= start_ts) & (timestamps <= end_ts))
    if indices.size == 0:
        return {"metrics": {}, "round_trips": 0, "fills": 0, "reject_counts": {"split_empty": 1}}

    cash = float(initial_equity)
    qty = 0.0
    position = "OUT"
    entry_price = 0.0
    bars_held = 0
    cooldown = 0
    equity_history: list[float] = []
    round_trips = 0
    fills = 0
    reject_counts: dict[str, int] = {}

    eth_close = arrays["ethusdt_close"]
    eth_open = arrays["ethusdt_open"]
    eth_high = arrays["ethusdt_high"]
    eth_low = arrays["ethusdt_low"]
    eth_volume = arrays["ethusdt_volume"]
    shock = arrays["shock_return_12h"]

    def mark_equity(price: float) -> float:
        if position == "LONG":
            return cash + qty * price
        if position == "SHORT":
            return cash - qty * price
        return cash

    def record_reject(reason: str) -> None:
        reject_counts[reason] = int(reject_counts.get(reason, 0)) + 1

    for idx in indices:
        close = float(eth_close[idx])
        if not math.isfinite(close) or close <= 0.0:
            continue
        high = float(eth_high[idx])
        low = float(eth_low[idx])
        open_ = float(eth_open[idx])
        volume = max(0.0, float(eth_volume[idx]) if math.isfinite(float(eth_volume[idx])) else 0.0)
        high_low_vol = max(0.0, (high - low) / open_) if open_ > 0.0 else 0.0

        if position != "OUT":
            bars_held += 1
            exit_reason = ""
            exit_price = close
            if position == "LONG":
                stop = entry_price * (1.0 - 0.05)
                take = entry_price * (1.0 + 0.10)
                if low <= stop:
                    exit_reason = "stop"
                    exit_price = min(open_, stop) if open_ < stop else stop
                elif high >= take:
                    exit_reason = "take_profit"
                    exit_price = max(open_, take) if open_ > take else take
                elif bars_held >= 72:
                    exit_reason = "max_hold"
                if exit_reason:
                    fill, fee_rate = _fill_price(exit_price, "SELL", high_low_vol=high_low_vol)
                    cash += qty * fill - qty * fill * fee_rate
            else:
                stop = entry_price * (1.0 + 0.05)
                take = entry_price * (1.0 - 0.10)
                if high >= stop:
                    exit_reason = "stop"
                    exit_price = max(open_, stop) if open_ > stop else stop
                elif low <= take:
                    exit_reason = "take_profit"
                    exit_price = min(open_, take) if open_ < take else take
                elif bars_held >= 72:
                    exit_reason = "max_hold"
                if exit_reason:
                    fill, fee_rate = _fill_price(exit_price, "BUY", high_low_vol=high_low_vol)
                    cash -= qty * fill + qty * fill * fee_rate
            if exit_reason:
                position = "OUT"
                qty = 0.0
                entry_price = 0.0
                bars_held = 0
                cooldown = max(0, int(spec.cooldown_bars))
                round_trips += 1
                fills += 1

        if position == "OUT":
            if cooldown > 0:
                cooldown -= 1
                record_reject("cooldown")
            else:
                shock_return = float(shock[idx])
                signal = ""
                if math.isfinite(shock_return):
                    if shock_return >= spec.return_threshold:
                        signal = "SHORT"
                    elif shock_return <= -spec.return_threshold:
                        signal = "LONG"
                if signal:
                    passed, reason = _filter_passes(spec, arrays, idx)
                    if passed:
                        notional = min(0.008 * mark_equity(close), 175.0)
                        raw_qty = math.floor((notional / close) / 0.001) * 0.001
                        max_fill_qty = volume * 0.10
                        order_qty = min(raw_qty, max_fill_qty)
                        if order_qty * close < 5.0 or order_qty <= 0.0:
                            record_reject("fill_or_min_notional")
                        elif signal == "LONG":
                            fill, fee_rate = _fill_price(close, "BUY", high_low_vol=high_low_vol)
                            cash -= order_qty * fill + order_qty * fill * fee_rate
                            qty = order_qty
                            entry_price = fill
                            position = "LONG"
                            bars_held = 0
                            fills += 1
                        else:
                            fill, fee_rate = _fill_price(close, "SELL", high_low_vol=high_low_vol)
                            cash += order_qty * fill - order_qty * fill * fee_rate
                            qty = order_qty
                            entry_price = fill
                            position = "SHORT"
                            bars_held = 0
                            fills += 1
                    else:
                        record_reject(reason)

        equity_history.append(mark_equity(close))

    if position != "OUT" and indices.size:
        idx = int(indices[-1])
        close = float(eth_close[idx])
        if position == "LONG":
            fill, fee_rate = _fill_price(close, "SELL", high_low_vol=0.0)
            cash += qty * fill - qty * fill * fee_rate
        else:
            fill, fee_rate = _fill_price(close, "BUY", high_low_vol=0.0)
            cash -= qty * fill + qty * fill * fee_rate
        equity_history[-1] = cash
        round_trips += 1
        fills += 1

    metrics = _metrics_from_equity_totals(
        equity_history, periods=int(getattr(BacktestConfig, "ANNUAL_PERIODS", 252))
    )
    return {
        "metrics": metrics,
        "round_trips": int(round_trips),
        "fills": int(fills),
        "final_equity": float(equity_history[-1]) if equity_history else float(initial_equity),
        "reject_counts": dict(sorted(reject_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def _candidate_specs() -> list[ExchangeReplaySpec]:
    base_hours = (0, 1, 8, 9, 16, 17)
    specs = [
        ExchangeReplaySpec(name="replay_base_12h_threshold_100bp", return_threshold=0.010),
        ExchangeReplaySpec(
            name="replay_funding_guard_current_threshold_80bp",
            return_threshold=0.008,
            excluded_hours=base_hours,
        ),
    ]
    for threshold, prefix, hours in ((0.010, "base", ()), (0.008, "funding_guard", base_hours)):
        for cap in (0.00005, 0.000075, 0.00010, 0.00015, 0.00025):
            specs.append(
                ExchangeReplaySpec(
                    name=f"hl_funding_abs_cap_{prefix}_{int(cap * 1_000_000)}ppm",
                    return_threshold=threshold,
                    excluded_hours=hours,
                    hl_funding_abs_cap=cap,
                )
            )
            specs.append(
                ExchangeReplaySpec(
                    name=f"hl_funding_divergence_{prefix}_{int(cap * 1_000_000)}ppm",
                    return_threshold=threshold,
                    excluded_hours=hours,
                    hl_funding_abs_cap=cap,
                    hl_require_funding_sign_match=True,
                )
            )
        for oi_z in (0.5, 1.0):
            specs.append(
                ExchangeReplaySpec(
                    name=f"hl_oi_exhaustion_{prefix}_z{str(oi_z).replace('.', '')}",
                    return_threshold=threshold,
                    excluded_hours=hours,
                    hl_oi_z_min=oi_z,
                )
            )
        for basis in (0.001, 0.0025, 0.005):
            specs.append(
                ExchangeReplaySpec(
                    name=f"hl_mark_basis_stress_{prefix}_{int(basis * 10_000)}bp",
                    return_threshold=threshold,
                    excluded_hours=hours,
                    hl_basis_abs_max=basis,
                )
            )
        for macro_filter in ("usd_risk", "xau_xag_stress", "indices_risk_off"):
            specs.append(
                ExchangeReplaySpec(
                    name=f"tickmill_macro_{macro_filter}_{prefix}",
                    return_threshold=threshold,
                    excluded_hours=hours,
                    tickmill_macro_filter=macro_filter,
                )
            )
    return specs


def _evaluate_specs(
    *,
    specs: list[ExchangeReplaySpec],
    arrays: dict[str, Any],
    splits: list[SplitWindow],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for spec in specs:
        split_results = {
            split.name: _run_split(spec=spec, arrays=arrays, split=split) for split in splits
        }
        metrics = {
            name: dict(result.get("metrics") or {}) for name, result in split_results.items()
        }
        train = metrics.get("train", {})
        val = metrics.get("val", {})
        oos = metrics.get("oos", {})
        absolute_gates = {
            "train_positive": _safe_float(train.get("total_return"), 0.0) > 0.0,
            "val_positive": _safe_float(val.get("total_return"), 0.0) > 0.0,
            "oos_return_beats_baseline": _safe_float(oos.get("total_return"), 0.0)
            > BASELINE_OOS_RETURN,
            "oos_sharpe_beats_funding_guard": _safe_float(oos.get("sharpe"), 0.0)
            > FUNDING_GUARD_OOS_SHARPE,
            "oos_mdd_beats_funding_guard": _safe_float(oos.get("max_drawdown"), 1.0)
            < FUNDING_GUARD_OOS_MDD,
            "oos_sharpe_gt_1": _safe_float(oos.get("sharpe"), 0.0) > 1.0,
            "oos_trades_not_starved": int(split_results["oos"].get("round_trips") or 0) >= 5,
        }
        absolute_shape = all(
            absolute_gates[key]
            for key in (
                "train_positive",
                "val_positive",
                "oos_return_beats_baseline",
                "oos_sharpe_beats_funding_guard",
                "oos_mdd_beats_funding_guard",
                "oos_trades_not_starved",
            )
        )
        row: dict[str, Any] = {
            "name": spec.name,
            "filters": json.dumps(spec.filters_payload(), sort_keys=True),
            "reject_top": json.dumps(
                split_results["oos"].get("reject_counts") or {}, sort_keys=True
            ),
            "absolute_gate_shape": bool(absolute_shape),
            "success_candidate": bool(absolute_shape and absolute_gates["oos_sharpe_gt_1"]),
            "replay_survivor": False,
        }
        for split_name, result in split_results.items():
            split_metrics = dict(result.get("metrics") or {})
            for key in ("total_return", "max_drawdown", "sharpe", "sortino", "volatility"):
                row[f"{split_name}_{key}"] = _safe_float(split_metrics.get(key), 0.0)
            row[f"{split_name}_round_trips"] = int(result.get("round_trips") or 0)
            row[f"{split_name}_fills"] = int(result.get("fills") or 0)
        rows.append(row)
        results.append(
            {
                "name": spec.name,
                "filters": spec.filters_payload(),
                "split_results": split_results,
                "absolute_gates": absolute_gates,
                "absolute_gate_shape": bool(absolute_shape),
                "success_candidate": bool(absolute_shape and absolute_gates["oos_sharpe_gt_1"]),
                "replay_relative_gates": {},
                "replay_survivor": False,
            }
        )

    by_name = {str(row["name"]): row for row in rows}
    replay_base = by_name.get("replay_base_12h_threshold_100bp", {})
    replay_guard = by_name.get("replay_funding_guard_current_threshold_80bp", {})
    replay_return_floor = max(
        _safe_float(replay_base.get("oos_total_return"), 0.0),
        _safe_float(replay_guard.get("oos_total_return"), 0.0),
    )
    replay_sharpe_floor = max(
        _safe_float(replay_base.get("oos_sharpe"), 0.0),
        _safe_float(replay_guard.get("oos_sharpe"), 0.0),
    )
    replay_mdd_floor = min(
        _safe_float(replay_base.get("oos_max_drawdown"), 1.0),
        _safe_float(replay_guard.get("oos_max_drawdown"), 1.0),
    )
    result_by_name = {str(item["name"]): item for item in results}
    for row in rows:
        relative_gates = {
            "train_positive": _safe_float(row.get("train_total_return"), 0.0) > 0.0,
            "val_positive": _safe_float(row.get("val_total_return"), 0.0) > 0.0,
            "oos_return_beats_replay_incumbents": _safe_float(row.get("oos_total_return"), 0.0)
            > replay_return_floor,
            "oos_sharpe_beats_replay_incumbents": _safe_float(row.get("oos_sharpe"), 0.0)
            > replay_sharpe_floor,
            "oos_mdd_beats_replay_incumbents": _safe_float(row.get("oos_max_drawdown"), 1.0)
            < replay_mdd_floor,
            "oos_trades_not_starved": int(row.get("oos_round_trips") or 0) >= 5,
        }
        survivor = all(relative_gates.values())
        row["replay_survivor"] = bool(survivor)
        row["replay_relative_gates"] = json.dumps(relative_gates, sort_keys=True)
        result_by_name[str(row["name"])]["replay_relative_gates"] = relative_gates
        result_by_name[str(row["name"])]["replay_survivor"] = bool(survivor)
    rows.sort(
        key=lambda row: (
            not bool(row["replay_survivor"]),
            -_safe_float(row.get("oos_total_return"), 0.0),
            -_safe_float(row.get("oos_sharpe"), 0.0),
            _safe_float(row.get("oos_max_drawdown"), 1.0),
        )
    )
    return rows, [result_by_name[str(row["name"])] for row in rows]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=sorted({key for row in rows for key in row}),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value, 0.0):+.4%}"


def _fmt_float(value: Any) -> str:
    return f"{_safe_float(value, 0.0):.6f}"


def _markdown(payload: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    survivors = [row for row in rows if bool(row.get("replay_survivor"))]
    lines = [
        "# Multiasset exchange stateful replay candidates",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        f"OOS window ends: `{payload['oos_end_date']}`",
        "",
        "## Gate policy",
        "",
        "- Hyperliquid and Tickmill are read-only feature/regime sources only.",
        "- Replay enforces one ETH position, fee/spread/slippage/fill, cooldown, 5% stop, 10% take-profit, and 72h max hold.",
        "- Full live-equivalent backtest slots require replay-relative survivor status; final success still requires OOS return > +0.8284%, MDD < 0.1778%, Sharpe > 1.0, liquidations 0, separated train/val/OOS raw-first evidence, and RSS < 8GB.",
        "",
        "## Data metadata",
        "",
        f"- Binance/ETH feature rows: `{payload['data_metadata']['eth_feature_points'].get('rows')}`",
        f"- Hyperliquid/ETH feature rows: `{payload['data_metadata']['hyperliquid_eth_feature_points'].get('rows')}`",
        f"- Hyperliquid OI history usable: `{payload['data_metadata']['hyperliquid_eth_feature_points'].get('has_open_interest_history')}`",
        f"- Tickmill macro replay status: `{payload['tickmill_macro_status']}`",
        "",
        "## Results",
        "",
        f"- Specs evaluated: `{len(rows)}`",
        f"- Replay survivors for one-at-a-time full backtest slots: `{len(survivors)}`",
        "",
        "| rank | replay spec | survivor | train ret | val ret | OOS ret | OOS Sharpe | OOS MDD | OOS trips | top OOS rejects |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows[:40], start=1):
        lines.append(
            f"| {rank} | `{row['name']}` | `{bool(row['replay_survivor'])}` | "
            f"{_fmt_pct(row.get('train_total_return'))} | {_fmt_pct(row.get('val_total_return'))} | "
            f"{_fmt_pct(row.get('oos_total_return'))} | {_fmt_float(row.get('oos_sharpe'))} | "
            f"{_fmt_pct(row.get('oos_max_drawdown'))} | {int(row.get('oos_round_trips') or 0)} | "
            f"`{str(row.get('reject_top') or '')[:120]}` |"
        )
    if not survivors:
        lines.extend(
            [
                "",
                "## No full-backtest slot",
                "",
                "No Hyperliquid/Tickmill read-only filter earned a replay survivor slot. Vector-only or context-only ideas are rejected before live-equivalent backtesting.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Full-backtest slots",
                "",
                "Run at most one live-equivalent raw-first mode at a time, starting from the first survivor above.",
            ]
        )
    return "\n".join(lines) + "\n"


def build_replay(
    *,
    market_root: Path,
    exchange: str,
    output_dir: Path,
    oos_end: date,
) -> dict[str, Any]:
    splits = _split_windows(oos_end=oos_end)
    load_start = min(split.start for split in splits)
    load_end = max(split.end for split in splits)
    panel, data_metadata = _joined_exchange_panel(
        market_root=market_root, exchange=exchange, start=load_start, end=load_end
    )
    arrays = _build_exchange_arrays(panel)
    specs = _candidate_specs()
    rows, results = _evaluate_specs(specs=specs, arrays=arrays, splits=splits)
    max_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    tickmill_path = output_dir / "tickmill_mt5_readonly_collection_latest.json"
    tickmill_status = "missing_report"
    if tickmill_path.exists():
        try:
            tickmill_status = str(
                json.loads(tickmill_path.read_text(encoding="utf-8")).get("status") or "unknown"
            )
        except Exception:
            tickmill_status = "unreadable_report"
    payload: dict[str, Any] = {
        "artifact_kind": "multiasset_exchange_stateful_replay",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "market_root": str(market_root),
        "exchange": str(exchange),
        "oos_end_date": oos_end.isoformat(),
        "split_windows": [split.as_payload() for split in splits],
        "data_metadata": {
            **data_metadata,
            "hourly_rows": int(panel.height),
            "first_hour": str(panel["datetime"][0]) if panel.height else "",
            "last_hour": str(panel["datetime"][-1]) if panel.height else "",
        },
        "tickmill_macro_status": tickmill_status,
        "resource_usage": {
            "max_rss_mib": round(max_rss_kb / 1024.0, 3),
            "rss_limit_mib": 8192,
        },
        "gate_thresholds": {
            "baseline_oos_return": BASELINE_OOS_RETURN,
            "funding_guard_oos_sharpe": FUNDING_GUARD_OOS_SHARPE,
            "funding_guard_oos_mdd": FUNDING_GUARD_OOS_MDD,
            "success_sharpe_min": 1.0,
        },
        "candidate_rows": rows,
        "candidate_results": results,
        "replay_survivors": [row for row in rows if bool(row.get("replay_survivor"))],
        "success_candidates": [row for row in rows if bool(row.get("success_candidate"))],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stateful_replay_candidates_latest.json"
    csv_path = output_dir / "stateful_replay_candidates_latest.csv"
    md_path = output_dir / "stateful_replay_candidates_latest.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(csv_path, rows)
    md_path.write_text(_markdown(payload, rows), encoding="utf-8")
    return {
        "payload": payload,
        "paths": {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-root", default=str(BaseConfig.MARKET_DATA_PARQUET_PATH))
    parser.add_argument("--exchange", default=str(BaseConfig.MARKET_DATA_EXCHANGE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--oos-end-date", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    oos_end = _parse_date(args.oos_end_date) or _latest_complete_utc_day()
    result = build_replay(
        market_root=Path(args.market_root),
        exchange=str(args.exchange),
        output_dir=Path(args.output_dir),
        oos_end=oos_end,
    )
    print(json.dumps(result["paths"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
