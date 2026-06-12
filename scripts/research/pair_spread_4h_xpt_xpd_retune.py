"""Run the focused 4h XPT/XPD pair retune and write a guarded follow-up artifact."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from lumina_quant.strategy_factory import research_runner as rr

REPORT_ROOT = Path(__file__).resolve().parents[2] / "var" / "reports" / "exact_window_backtests"
FOLLOWUP_ROOT = REPORT_ROOT / "followup_status"
MANIFEST_PATH = REPORT_ROOT / "xpt_xpd_4h_focus_20260310T115011Z" / "manifest.json"

MIN_TOTAL_COVERAGE_DAYS = 60
SURVIVAL_THRESHOLDS = {
    "oos_sharpe_min": 1.0,
    "oos_return_min": 0.0,
    "max_pbo": 0.5,
    "val_sharpe_min": 1.0,
    "oos_trade_count_min": 3.0,
}
RETUNE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "variant": "strict_2p8",
        "lookback_window": 120,
        "hedge_window": 240,
        "entry_z": 2.8,
        "exit_z": 0.80,
        "stop_z": 4.4,
        "max_hold_bars": 72,
        "min_correlation": 0.15,
        "cooldown_bars": 8,
        "reentry_z_buffer": 0.25,
        "stop_loss_pct": 0.020,
    },
    {
        "variant": "strict_3p0",
        "lookback_window": 144,
        "hedge_window": 288,
        "entry_z": 3.0,
        "exit_z": 0.90,
        "stop_z": 4.8,
        "max_hold_bars": 72,
        "min_correlation": 0.20,
        "cooldown_bars": 8,
        "reentry_z_buffer": 0.30,
        "stop_loss_pct": 0.015,
    },
    {
        "variant": "strict_3p2",
        "lookback_window": 144,
        "hedge_window": 288,
        "entry_z": 3.2,
        "exit_z": 0.95,
        "stop_z": 5.0,
        "max_hold_bars": 60,
        "min_correlation": 0.20,
        "cooldown_bars": 10,
        "reentry_z_buffer": 0.35,
        "stop_loss_pct": 0.015,
    },
    {
        "variant": "extreme_3p4",
        "lookback_window": 168,
        "hedge_window": 336,
        "entry_z": 3.4,
        "exit_z": 1.00,
        "stop_z": 5.2,
        "max_hold_bars": 48,
        "min_correlation": 0.25,
        "cooldown_bars": 12,
        "reentry_z_buffer": 0.40,
        "stop_loss_pct": 0.015,
    },
    {
        "variant": "extreme_3p6",
        "lookback_window": 168,
        "hedge_window": 336,
        "entry_z": 3.6,
        "exit_z": 1.10,
        "stop_z": 5.6,
        "max_hold_bars": 48,
        "min_correlation": 0.25,
        "cooldown_bars": 12,
        "reentry_z_buffer": 0.40,
        "stop_loss_pct": 0.015,
    },
    {
        "variant": "fast_strict_2p8",
        "lookback_window": 96,
        "hedge_window": 192,
        "entry_z": 2.8,
        "exit_z": 0.80,
        "stop_z": 4.4,
        "max_hold_bars": 48,
        "min_correlation": 0.15,
        "cooldown_bars": 8,
        "reentry_z_buffer": 0.25,
        "stop_loss_pct": 0.015,
    },
    {
        "variant": "slow_guarded_2p8",
        "lookback_window": 168,
        "hedge_window": 336,
        "entry_z": 2.8,
        "exit_z": 0.80,
        "stop_z": 4.4,
        "max_hold_bars": 96,
        "min_correlation": 0.20,
        "cooldown_bars": 10,
        "reentry_z_buffer": 0.35,
        "stop_loss_pct": 0.020,
    },
    {
        "variant": "slow_guarded_3p0",
        "lookback_window": 192,
        "hedge_window": 384,
        "entry_z": 3.0,
        "exit_z": 0.90,
        "stop_z": 4.8,
        "max_hold_bars": 96,
        "min_correlation": 0.20,
        "cooldown_bars": 10,
        "reentry_z_buffer": 0.35,
        "stop_loss_pct": 0.015,
    },
    {
        "variant": "corr_guarded_2p8",
        "lookback_window": 144,
        "hedge_window": 288,
        "entry_z": 2.8,
        "exit_z": 0.80,
        "stop_z": 4.4,
        "max_hold_bars": 72,
        "min_correlation": 0.30,
        "cooldown_bars": 10,
        "reentry_z_buffer": 0.30,
        "stop_loss_pct": 0.015,
    },
    {
        "variant": "corr_guarded_3p0",
        "lookback_window": 168,
        "hedge_window": 336,
        "entry_z": 3.0,
        "exit_z": 0.90,
        "stop_z": 4.8,
        "max_hold_bars": 72,
        "min_correlation": 0.30,
        "cooldown_bars": 12,
        "reentry_z_buffer": 0.35,
        "stop_loss_pct": 0.015,
    },
    {
        "variant": "hold_short_3p0",
        "lookback_window": 120,
        "hedge_window": 240,
        "entry_z": 3.0,
        "exit_z": 0.90,
        "stop_z": 4.8,
        "max_hold_bars": 36,
        "min_correlation": 0.20,
        "cooldown_bars": 10,
        "reentry_z_buffer": 0.30,
        "stop_loss_pct": 0.015,
    },
    {
        "variant": "hold_short_3p2",
        "lookback_window": 144,
        "hedge_window": 288,
        "entry_z": 3.2,
        "exit_z": 0.95,
        "stop_z": 5.0,
        "max_hold_bars": 36,
        "min_correlation": 0.20,
        "cooldown_bars": 10,
        "reentry_z_buffer": 0.35,
        "stop_loss_pct": 0.015,
    },
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except TypeError, ValueError:
        return float(default)


def _safe_bool(value: Any) -> bool:
    return bool(value)


def _load_manifest(manifest_payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if manifest_payload is not None:
        return dict(manifest_payload)
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _coverage_guard(manifest: Mapping[str, Any]) -> dict[str, Any]:
    adaptive = dict(manifest.get("adaptive_windows") or {})
    observed_days = int(adaptive.get("total_days") or 0)
    common_start = str(adaptive.get("common_start") or "")
    review_after = ""
    if common_start:
        review_after = (
            pd.to_datetime(common_start, utc=True) + timedelta(days=MIN_TOTAL_COVERAGE_DAYS)
        ).isoformat()
    return {
        "min_total_days": MIN_TOTAL_COVERAGE_DAYS,
        "observed_total_days": observed_days,
        "pass": observed_days >= MIN_TOTAL_COVERAGE_DAYS,
        "profile": str(adaptive.get("profile") or ""),
        "common_start": common_start,
        "common_end": str(adaptive.get("common_end") or ""),
        "review_not_before": review_after,
    }


def _build_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in RETUNE_SPECS:
        rows.append(
            {
                "name": (
                    f"pair_spread_4h_{spec['variant']}_xptusdt_xpdusdt_"
                    f"{float(spec['entry_z']):.1f}_{float(spec['exit_z']):.2f}"
                ),
                "family": "market_neutral",
                "strategy_class": "PairSpreadZScoreStrategy",
                "timeframe": "4h",
                "symbols": ["XPT/USDT", "XPD/USDT"],
                "params": {
                    "lookback_window": int(spec["lookback_window"]),
                    "hedge_window": int(spec["hedge_window"]),
                    "entry_z": float(spec["entry_z"]),
                    "exit_z": float(spec["exit_z"]),
                    "stop_z": float(spec["stop_z"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "min_correlation": float(spec["min_correlation"]),
                    "cooldown_bars": int(spec["cooldown_bars"]),
                    "reentry_z_buffer": float(spec["reentry_z_buffer"]),
                    "stop_loss_pct": float(spec["stop_loss_pct"]),
                    "symbol_x": "XPT/USDT",
                    "symbol_y": "XPD/USDT",
                },
                "metadata": {
                    "timeframe": "4h",
                    "pair": "XPT/USDT_XPD/USDT",
                    "pair_variant": str(spec["variant"]),
                },
            }
        )
    return rows


def _split_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    windows = dict(manifest.get("windows") or {})
    return {
        "train_start": windows.get("train_start"),
        "train_end": windows.get("train_end_exclusive"),
        "val_start": windows.get("val_start"),
        "val_end": windows.get("val_end_exclusive"),
        "oos_start": windows.get("val_end_exclusive"),
        "oos_end": windows.get("actual_oos_end_exclusive"),
        "strategy_timeframe": "4h",
        "mode": "exact_dates",
    }


def _run_report(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    report = rr.run_candidate_research(
        candidates=_build_candidates(),
        strategy_timeframes=["4h"],
        symbol_universe=["XPT/USDT", "XPD/USDT"],
        stage1_keep_ratio=1.0,
        max_candidates=len(RETUNE_SPECS),
        split=_split_config(manifest),
    )
    return [dict(row) for row in list(report.get("candidates") or [])]


def _survivor_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    val = dict(row.get("val") or {})
    oos = dict(row.get("oos") or {})
    if _safe_float(oos.get("sharpe"), 0.0) < SURVIVAL_THRESHOLDS["oos_sharpe_min"]:
        reasons.append("oos_sharpe")
    if _safe_float(oos.get("return"), 0.0) <= SURVIVAL_THRESHOLDS["oos_return_min"]:
        reasons.append("oos_return")
    if _safe_float(oos.get("pbo"), 1.0) > SURVIVAL_THRESHOLDS["max_pbo"]:
        reasons.append("pbo")
    if _safe_float(val.get("sharpe"), 0.0) < SURVIVAL_THRESHOLDS["val_sharpe_min"]:
        reasons.append("val_sharpe")
    if _safe_float(oos.get("trade_count"), 0.0) < SURVIVAL_THRESHOLDS["oos_trade_count_min"]:
        reasons.append("oos_trade_count")
    return reasons


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = list(rows)
    ranked.sort(
        key=lambda row: (
            _safe_float((row.get("oos") or {}).get("sharpe"), 0.0),
            _safe_float((row.get("oos") or {}).get("return"), 0.0),
            -_safe_float((row.get("oos") or {}).get("pbo"), 1.0),
        ),
        reverse=True,
    )
    return ranked


def build_pair_spread_4h_xpt_xpd_retune(
    *,
    manifest_payload: Mapping[str, Any] | None = None,
    report_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_payload)
    rows = [
        dict(row) for row in (report_rows if report_rows is not None else _run_report(manifest))
    ]
    coverage_guard = _coverage_guard(manifest)
    ranked = _rank_rows(rows)

    survivor: dict[str, Any] | None = None
    for row in ranked:
        candidate_reasons = _survivor_reasons(row)
        if not coverage_guard["pass"]:
            candidate_reasons.append("coverage_days")
        row["survivor_blockers"] = list(dict.fromkeys(candidate_reasons))
        if not row["survivor_blockers"] and survivor is None:
            survivor = row

    overall_blockers: list[str] = []
    if not coverage_guard["pass"]:
        overall_blockers.append("coverage_days")
    if survivor is None:
        overall_blockers.append("no_candidate_passed")

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_kind": "pair_spread_4h_xpt_xpd_retune",
        "schema_version": "2.0",
        "baseline_candidate": "pair_spread_4h_balanced_xptusdt_xpdusdt_1.6_0.35",
        "windows": dict(manifest.get("windows") or {}),
        "coverage_guard": coverage_guard,
        "survival_thresholds": dict(SURVIVAL_THRESHOLDS),
        "candidate_count": len(ranked),
        "survives": survivor is not None,
        "blockers": overall_blockers,
        "survivor": survivor,
        "top_candidates": ranked[:10],
    }


def write_pair_spread_4h_xpt_xpd_retune(
    *,
    report_root: Path | str = REPORT_ROOT,
    manifest_payload: Mapping[str, Any] | None = None,
    report_rows: list[dict[str, Any]] | None = None,
    run_name: str = "pair_spread_4h_xpt_xpd_retune",
) -> dict[str, Any]:
    payload = build_pair_spread_4h_xpt_xpd_retune(
        manifest_payload=manifest_payload,
        report_rows=report_rows,
    )
    followup_root = Path(report_root) / "followup_status"
    followup_root.mkdir(parents=True, exist_ok=True)
    json_path = followup_root / f"{run_name}_latest.json"
    md_path = followup_root / f"{run_name}_latest.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    coverage = dict(payload["coverage_guard"])
    lines = [
        "# pair spread 4h XPT/XPD retune",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- candidate_count: `{payload['candidate_count']}`",
        f"- survives: `{payload['survives']}`",
        f"- coverage_pass: `{coverage.get('pass')}`",
        f"- observed_total_days: `{coverage.get('observed_total_days')}`",
        f"- min_total_days: `{coverage.get('min_total_days')}`",
        f"- review_not_before: `{coverage.get('review_not_before')}`",
        "",
        "## top candidates",
    ]
    for row in list(payload.get("top_candidates") or []):
        oos = dict(row.get("oos") or {})
        val = dict(row.get("val") or {})
        lines.append(
            f"- `{row.get('name')}` | oos_return={_safe_float(oos.get('return'), 0.0):.4%} | "
            f"oos_sharpe={_safe_float(oos.get('sharpe'), 0.0):.3f} | "
            f"oos_sortino={_safe_float(oos.get('sortino'), 0.0):.3f} | "
            f"oos_calmar={_safe_float(oos.get('calmar'), 0.0):.3f} | "
            f"oos_max_dd={_safe_float(oos.get('max_drawdown'), 0.0):.4%} | "
            f"oos_pbo={_safe_float(oos.get('pbo'), 0.0):.3f} | "
            f"val_sharpe={_safe_float(val.get('sharpe'), 0.0):.3f} | "
            f"blockers={','.join(row.get('survivor_blockers') or []) or 'none'}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "payload": payload,
        "json_path": str(json_path.resolve()),
        "md_path": str(md_path.resolve()),
    }


if __name__ == "__main__":
    result = write_pair_spread_4h_xpt_xpd_retune()
    print(result["json_path"])
    print(result["md_path"])
