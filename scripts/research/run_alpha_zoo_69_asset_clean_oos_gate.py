#!/usr/bin/env python3
"""Evaluate a frozen 69-asset efficiency-repair artifact on a clean locked OOS split.

The runner is intentionally post-freeze: train/validation windows define the
allowed fit/selection surface, while locked-OOS is replayed only after the
artifact is frozen.  If the artifact's train/validation policy overlaps the
requested locked-OOS window, the gate fails even when replay metrics look good.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_alpha_zoo_69_asset_optuna_hybrid_refit as broad69  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_profile_optuna_hybrid_refit as profile69  # noqa: E402

DEFAULT_ARTIFACT = (
    broad69.ALPHA_V2_ROOT
    / "alpha_zoo_69_asset_efficiency_repair_optuna_20260530"
    / "alpha_zoo_69_asset_efficiency_repair_optuna_latest.json"
)
DEFAULT_OUTPUT_DIR = broad69.ALPHA_V2_ROOT / "alpha_zoo_69_asset_clean_oos_gate_20260531"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "alpha_zoo_69_asset_clean_oos_gate_latest.json"
DEFAULT_TRAIN_START = "2025-01-01T00:00:00Z"
DEFAULT_TRAIN_END = "2025-12-31T23:00:00Z"
DEFAULT_VALIDATION_START = "2026-01-01T00:00:00Z"
DEFAULT_VALIDATION_END = "2026-02-28T23:00:00Z"
DEFAULT_LOCKED_OOS_START = "2026-03-01T00:00:00Z"
DEFAULT_LOCKED_OOS_END = "2026-05-06T23:00:00Z"
DEFAULT_MAX_OOS_MDD = 0.20
MIN_LOCKED_OOS_TRADE_EVENTS = 20
SPLIT_NAMES = ("train", "validation", "locked_oos")


@dataclass(frozen=True)
class GateWindows:
    train: tuple[pd.Timestamp, pd.Timestamp]
    validation: tuple[pd.Timestamp, pd.Timestamp]
    locked_oos: tuple[pd.Timestamp, pd.Timestamp]

    def as_payload(self) -> dict[str, dict[str, Any]]:
        return {
            "train": _window_payload(self.train, "parameter_fitting_and_objective_training"),
            "validation": _window_payload(self.validation, "holdout_selection_and_report"),
            "locked_oos": _window_payload(
                self.locked_oos, "gate_report_only_after_train_validation_freeze"
            ),
        }


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(broad69._json_safe(payload), indent=2, sort_keys=True) + "\n")


def _parse_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(UTC).tz_localize(None)
    return ts


def _window_payload(window: tuple[pd.Timestamp, pd.Timestamp], role: str) -> dict[str, Any]:
    return {
        "start": window[0].isoformat(),
        "end": window[1].isoformat(),
        "role": role,
        "enabled": True,
    }


def _coerce_windows(args: argparse.Namespace) -> GateWindows:
    return GateWindows(
        train=(_parse_timestamp(args.train_start), _parse_timestamp(args.train_end)),
        validation=(
            _parse_timestamp(args.validation_start),
            _parse_timestamp(args.validation_end),
        ),
        locked_oos=(
            _parse_timestamp(args.locked_oos_start),
            _parse_timestamp(args.locked_oos_end),
        ),
    )


def _split_mask(index: pd.DatetimeIndex | pd.Series, window: tuple[pd.Timestamp, pd.Timestamp]) -> np.ndarray:
    values = pd.Series(index) if not isinstance(index, pd.Series) else index
    values = pd.to_datetime(values)
    return ((values >= window[0]) & (values <= window[1])).to_numpy()


def _periods_per_year_for_index(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 365.0 * 24.0
    diffs = pd.Series(index).diff().dropna().dt.total_seconds()
    if diffs.empty:
        return 365.0 * 24.0
    seconds = float(diffs.median())
    return 365.0 * 24.0 * 3600.0 / seconds if seconds > 0.0 else 365.0 * 24.0


def _stream_events_in_mask(position: pd.Series, mask: np.ndarray) -> int:
    values = position.to_numpy(dtype=float)
    if values.size == 0 or not mask.any():
        return 0
    indices = np.flatnonzero(mask)
    first_previous = values[indices[0] - 1] if indices[0] > 0 else 0.0
    subset = values[indices]
    return int(np.count_nonzero(np.abs(np.diff(np.r_[first_previous, subset])) > 1e-12))


def _calc_liquidation_flags(
    row: Mapping[str, Any],
    stream: broad69.CandidateStream,
    bars_by_symbol_tf: Mapping[tuple[str, str], pd.DataFrame],
) -> pd.Series:
    frame = bars_by_symbol_tf[(str(row["symbol"]), str(row["timeframe"]))]
    frame_indexed = (
        frame.assign(datetime=pd.to_datetime(frame["datetime"]))
        .set_index("datetime")
        .reindex(stream.position.index)
    )
    close = frame_indexed["close"].to_numpy(dtype=float)
    high = frame_indexed["high"].to_numpy(dtype=float)
    low = frame_indexed["low"].to_numpy(dtype=float)
    signal = stream.position.to_numpy(dtype=float)
    leverage = int(row.get("integer_leverage") or 0)
    long_liq = (signal > 0.0) & (((low / np.maximum(close, 1e-12)) - 1.0) * leverage <= -0.95)
    short_liq = (signal < 0.0) & (((high / np.maximum(close, 1e-12)) - 1.0) * leverage >= 0.95)
    return pd.Series(long_liq | short_liq, index=stream.position.index)


def _split_metrics(
    returns: pd.Series,
    *,
    windows: GateWindows,
    turnover_by_split: Mapping[str, float],
    events_by_split: Mapping[str, int],
    liquidation_by_split: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    index = pd.DatetimeIndex(returns.index)
    values = returns.to_numpy(dtype=float)
    annual = _periods_per_year_for_index(index)
    equity = (1.0 + returns).cumprod()
    for split in ("train", "validation", "locked_oos"):
        window = getattr(windows, split)
        mask = _split_mask(index, window)
        split_values = values[mask]
        total_return = float(np.prod(1.0 + split_values) - 1.0) if split_values.size else 0.0
        mdd = broad69.max_drawdown(split_values.astype(float)) if split_values.size else 0.0
        mean = float(np.mean(split_values)) if split_values.size else 0.0
        std = float(np.std(split_values, ddof=1)) if split_values.size > 1 else 0.0
        downside = split_values[split_values < 0.0]
        down_std = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
        turnover = float(turnover_by_split.get(split, 0.0))
        out[split] = {
            "start": window[0].isoformat() + "Z",
            "end": window[1].isoformat() + "Z",
            "bar_count": int(split_values.size),
            "total_return": total_return,
            "mdd": float(mdd),
            "sharpe": mean / std * math.sqrt(annual) if std > 0.0 else 0.0,
            "sortino": mean / down_std * math.sqrt(annual) if down_std > 0.0 else 0.0,
            "calmar": total_return / float(mdd) if float(mdd) > 0.0 else 0.0,
            "trade_event_count": int(events_by_split.get(split, 0)),
            "turnover_proxy": turnover,
            "return_per_turnover_proxy_bps": total_return * 10_000.0 / turnover
            if turnover > 0.0
            else None,
            "liquidation_count": int(liquidation_by_split.get(split, 0)),
            "account_wipeout_bar_count": int(
                ((equity <= 0.0).to_numpy(dtype=bool) & mask).sum()
            ),
        }
    return out


def _split_overlap(
    left: tuple[pd.Timestamp, pd.Timestamp], right: tuple[pd.Timestamp, pd.Timestamp]
) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def artifact_oos_contamination_reasons(
    split_policy: Mapping[str, Any], windows: GateWindows
) -> list[str]:
    reasons: list[str] = []
    locked_oos = windows.locked_oos
    for split in ("train", "validation"):
        raw = dict(split_policy.get(split) or {})
        if raw.get("start") is None or raw.get("end") is None:
            reasons.append(f"artifact_split_policy_missing_{split}")
            continue
        window = (_parse_timestamp(raw["start"]), _parse_timestamp(raw["end"]))
        if _split_overlap(window, locked_oos):
            reasons.append(f"artifact_{split}_window_overlaps_requested_locked_oos")
    return reasons


def locked_oos_gate_reasons(
    metrics: Mapping[str, Any], *, max_oos_mdd: float = DEFAULT_MAX_OOS_MDD
) -> list[str]:
    reasons: list[str] = []
    locked = dict(metrics.get("locked_oos") or {})
    if int(locked.get("bar_count") or 0) <= 0:
        reasons.append("locked_oos_bar_count_zero")
    if float(locked.get("total_return") or 0.0) <= 0.0:
        reasons.append("locked_oos_return_not_positive")
    if float(locked.get("mdd") or 0.0) > float(max_oos_mdd):
        reasons.append(f"locked_oos_mdd_above_{float(max_oos_mdd):.4f}")
    rpt = locked.get("return_per_turnover_proxy_bps")
    if rpt is None or float(rpt) <= broad69.RETURN_PER_TURNOVER_THRESHOLD_BPS:
        rendered = "missing" if rpt is None else f"{float(rpt):.3f}"
        reasons.append(f"locked_oos_rpt_{rendered}_not_above_10bps")
    if int(locked.get("trade_event_count") or 0) < MIN_LOCKED_OOS_TRADE_EVENTS:
        reasons.append(
            "locked_oos_trade_event_count_"
            f"{int(locked.get('trade_event_count') or 0)}_below_{MIN_LOCKED_OOS_TRADE_EVENTS}"
        )
    if int(locked.get("liquidation_count") or 0) != 0:
        reasons.append("locked_oos_liquidation_count_nonzero")
    if int(locked.get("account_wipeout_bar_count") or 0) != 0:
        reasons.append("locked_oos_account_wipeout_nonzero")
    return reasons


def _row_weight_sets(row: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in ("final_weights", "weights", "average_weights_train_validation"):
        weights = {str(k): float(v) for k, v in dict(row.get(key) or {}).items()}
        if weights:
            out[key] = weights
    return out


def _hybrid_rows(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for key in ("hybrid_v3_5_optuna", "hybrid_v3_6_optuna"):
        row = dict(dict(payload.get(key) or {}).get("row") or {})
        if row:
            rows[key] = row
    selected = dict(payload.get("selected_optuna_hybrid_profile") or {})
    if selected:
        rows["selected_optuna_hybrid_profile"] = selected
    return rows


def _build_profile_context(
    payload: Mapping[str, Any], windows: GateWindows
) -> dict[str, Any]:
    symbols = tuple(payload["universe"]["symbols"])
    timeframes = tuple(payload["timeframes"])
    data_root = Path(payload["data_coverage"]["data_root"])
    bars, coverage = broad69.load_all_bars(symbols, data_root=data_root, timeframes=timeframes)
    cache = profile69.FeatureCache(
        bars_by_symbol_tf=bars,
        symbols=symbols,
        timeframes=timeframes,
        _xsmom={},
        _anchor_returns={},
    )
    profile_returns: dict[str, pd.Series] = {}
    profile_turnover: dict[str, dict[str, float]] = {}
    profile_events: dict[str, dict[str, int]] = {}
    profile_liquidation: dict[str, dict[str, int]] = {}
    profile_gross: dict[str, float] = {}
    sleeve_streams: dict[str, list[tuple[dict[str, Any], broad69.CandidateStream]]] = defaultdict(list)
    for raw in payload.get("selected_sleeve_rows") or []:
        row = dict(raw)
        params = dict(row.get("optuna_params") or {})
        if not params:
            for key in (
                "family",
                "timeframe",
                "side",
                "integer_leverage",
                "min_hold_bars",
                "cooldown_bars",
                "lookback_bars",
                "threshold",
                "exit_threshold",
            ):
                params[key] = row[key]
        stream = profile69._candidate_from_params(
            symbol=str(row["symbol"]),
            profile_id=str(row["profile_id"]),
            params=params,
            cache=cache,
            windows=windows,  # type: ignore[arg-type]
            allocation_fraction=float(row.get("allocation_fraction") or 0.10),
        )
        sleeve_streams[str(row["profile_id"])].append((row, stream))

    for profile_id, items in sleeve_streams.items():
        index = pd.DatetimeIndex(sorted(set().union(*(set(stream.returns.index) for _, stream in items))))
        returns = pd.Series(0.0, index=index)
        turnover = dict.fromkeys(SPLIT_NAMES, 0.0)
        events = dict.fromkeys(SPLIT_NAMES, 0)
        liquidation = dict.fromkeys(SPLIT_NAMES, 0)
        gross = 0.0
        for row, stream in items:
            multiplier = float(row.get("sleeve_multiplier") or 0.0)
            notional = float(row.get("notional_fraction") or 0.0)
            returns = returns.add(
                stream.returns.reindex(index, fill_value=0.0) * multiplier,
                fill_value=0.0,
            )
            gross += multiplier * notional
            liq_flags = _calc_liquidation_flags(row, stream, bars)
            for split in SPLIT_NAMES:
                mask = _split_mask(pd.DatetimeIndex(stream.returns.index), getattr(windows, split))
                event_count = _stream_events_in_mask(stream.position, mask)
                events[split] += event_count
                turnover[split] += event_count * abs(notional) * multiplier
                liquidation[split] += int(liq_flags.to_numpy(dtype=bool)[mask].sum()) if mask.any() else 0
        profile_returns[profile_id] = returns.sort_index()
        profile_turnover[profile_id] = turnover
        profile_events[profile_id] = events
        profile_liquidation[profile_id] = liquidation
        profile_gross[profile_id] = gross
    return {
        "coverage": coverage,
        "profile_returns": profile_returns,
        "profile_turnover": profile_turnover,
        "profile_events": profile_events,
        "profile_liquidation": profile_liquidation,
        "profile_gross": profile_gross,
    }


def _evaluate_weights(
    context: Mapping[str, Any], weights: Mapping[str, float], windows: GateWindows
) -> dict[str, Any]:
    profile_returns: Mapping[str, pd.Series] = context["profile_returns"]
    index = pd.DatetimeIndex(sorted(set().union(*(set(series.index) for series in profile_returns.values()))))
    returns = pd.Series(0.0, index=index)
    turnover = dict.fromkeys(SPLIT_NAMES, 0.0)
    events = dict.fromkeys(SPLIT_NAMES, 0)
    liquidation = dict.fromkeys(SPLIT_NAMES, 0)
    gross = 0.0
    for profile_id, weight in weights.items():
        if weight <= 0.0 or profile_id not in profile_returns:
            continue
        returns = returns.add(profile_returns[profile_id].reindex(index, fill_value=0.0) * weight)
        gross += float(context["profile_gross"][profile_id]) * weight
        for split in SPLIT_NAMES:
            turnover[split] += float(context["profile_turnover"][profile_id][split]) * weight
            events[split] += int(context["profile_events"][profile_id][split])
            liquidation[split] += int(context["profile_liquidation"][profile_id][split])
    return {
        "gross_notional_fraction": gross,
        "metrics": _split_metrics(
            returns.sort_index(),
            windows=windows,
            turnover_by_split=turnover,
            events_by_split=events,
            liquidation_by_split=liquidation,
        ),
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    artifact = Path(args.artifact).expanduser().resolve()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    windows = _coerce_windows(args)
    contamination_reasons = artifact_oos_contamination_reasons(
        dict(payload.get("split_policy") or {}), windows
    )
    context = _build_profile_context(payload, windows)
    rows: dict[str, Any] = {}
    for row_key, row in _hybrid_rows(payload).items():
        weight_sets: dict[str, Any] = {}
        for weight_key, weights in _row_weight_sets(row).items():
            evaluated = _evaluate_weights(context, weights, windows)
            gate_reasons = [
                *contamination_reasons,
                *locked_oos_gate_reasons(evaluated["metrics"], max_oos_mdd=float(args.max_oos_mdd)),
            ]
            weight_sets[weight_key] = {
                "weights": weights,
                **evaluated,
                "locked_oos_gate_pass": not gate_reasons,
                "locked_oos_gate_reasons": gate_reasons,
            }
        rows[row_key] = {
            "profile_id": row.get("profile_id"),
            "hybrid_version": row.get("hybrid_version"),
            "artifact_train_return": row.get("train_return"),
            "artifact_validation_return": row.get("validation_return"),
            "artifact_train_mdd": row.get("train_mdd"),
            "artifact_validation_mdd": row.get("validation_mdd"),
            "fit_splits": row.get("fit_splits"),
            "test_set_policy": row.get("test_set_policy"),
            "selection_reasons": row.get("selection_reasons"),
            "weight_sets": weight_sets,
        }
    selected = dict(payload.get("selected_optuna_hybrid_profile") or {})
    selected_eval = rows.get("selected_optuna_hybrid_profile", {})
    selected_weights = dict(selected_eval.get("weight_sets") or {})
    primary_gate = dict(selected_weights.get("final_weights") or selected_weights.get("weights") or {})
    primary_reasons = list(primary_gate.get("locked_oos_gate_reasons") or [])
    primary_pass = bool(primary_gate.get("locked_oos_gate_pass"))
    return {
        "artifact_kind": "alpha_zoo_69_asset_clean_oos_gate",
        "generated_at_utc": _utc_now_iso(),
        "source_artifact": str(artifact),
        "evaluation_policy": {
            "candidate_freeze_before_locked_oos_gate": not contamination_reasons,
            "oos_used_for_selection": False,
            "oos_used_for_parameter_fitting": False,
            "oos_used_for_objective": False,
            "oos_used_for_pruning": False,
            "gate_return_must_be_positive": True,
            "gate_rpt_must_exceed_bps": broad69.RETURN_PER_TURNOVER_THRESHOLD_BPS,
            "gate_max_oos_mdd": float(args.max_oos_mdd),
            "gate_requires_zero_liquidations": True,
            "gate_requires_zero_wipeout": True,
            "gate_min_trade_events": MIN_LOCKED_OOS_TRADE_EVENTS,
        },
        "split_manifest": windows.as_payload(),
        "artifact_split_policy": payload.get("split_policy"),
        "contamination_reasons": contamination_reasons,
        "data_coverage": context["coverage"],
        "selected_profile_id": selected.get("profile_id"),
        "selected_hybrid_version": selected.get("hybrid_version"),
        "selected_primary_weight_set": "final_weights" if "final_weights" in selected_weights else "weights",
        "clean_oos_gate_pass": primary_pass,
        "clean_oos_gate_reasons": primary_reasons,
        "ready_for_paper_after_clean_oos_gate": primary_pass,
        "ready_for_real": False,
        "real_money_execution": False,
        "real_execution_allowed": False,
        "rows": rows,
        "runner_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def _render_pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):+.4%}"


def _render_num(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# 69-asset clean locked-OOS gate",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        f"Source artifact: `{payload.get('source_artifact')}`",
        f"Gate pass: `{payload.get('clean_oos_gate_pass')}`",
        f"Gate reasons: `{payload.get('clean_oos_gate_reasons')}`",
        "",
        "## Split manifest",
        "",
    ]
    for split, window in dict(payload.get("split_manifest") or {}).items():
        lines.append(f"- {split}: `{window.get('start')}` → `{window.get('end')}`")
    lines.extend(
        [
            "",
            "## Hybrid rows",
            "",
            "| row | version | weight set | train | validation | locked-OOS | OOS MDD | OOS Sharpe | OOS RPT bps | OOS trades | OOS liq | pass | reasons |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row_key, row in dict(payload.get("rows") or {}).items():
        for weight_key, evaluated in dict(row.get("weight_sets") or {}).items():
            metrics = dict(evaluated.get("metrics") or {})
            locked = dict(metrics.get("locked_oos") or {})
            lines.append(
                f"| `{row_key}` | `{row.get('hybrid_version')}` | `{weight_key}` | "
                f"{_render_pct(dict(metrics.get('train') or {}).get('total_return'))} | "
                f"{_render_pct(dict(metrics.get('validation') or {}).get('total_return'))} | "
                f"{_render_pct(locked.get('total_return'))} | {_render_pct(locked.get('mdd'))} | "
                f"{_render_num(locked.get('sharpe'), 3)} | "
                f"{_render_num(locked.get('return_per_turnover_proxy_bps'))} | "
                f"{int(locked.get('trade_event_count') or 0)} | "
                f"{int(locked.get('liquidation_count') or 0)} | "
                f"`{evaluated.get('locked_oos_gate_pass')}` | "
                f"`{evaluated.get('locked_oos_gate_reasons')}` |"
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--train-start", default=DEFAULT_TRAIN_START)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    parser.add_argument("--validation-start", default=DEFAULT_VALIDATION_START)
    parser.add_argument("--validation-end", default=DEFAULT_VALIDATION_END)
    parser.add_argument("--locked-oos-start", default=DEFAULT_LOCKED_OOS_START)
    parser.add_argument("--locked-oos-end", default=DEFAULT_LOCKED_OOS_END)
    parser.add_argument("--max-oos-mdd", type=float, default=DEFAULT_MAX_OOS_MDD)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(args)
    if args.check_only:
        print(
            json.dumps(
                {
                    "source_artifact": payload["source_artifact"],
                    "selected_profile_id": payload["selected_profile_id"],
                    "selected_hybrid_version": payload["selected_hybrid_version"],
                    "clean_oos_gate_pass": payload["clean_oos_gate_pass"],
                    "clean_oos_gate_reasons": payload["clean_oos_gate_reasons"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if payload["clean_oos_gate_pass"] else 1
    output = Path(args.output).expanduser().resolve()
    _write_json(output, payload)
    output.with_suffix(".md").write_text(render_markdown(payload), encoding="utf-8")
    print(output)
    return 0 if payload["clean_oos_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
