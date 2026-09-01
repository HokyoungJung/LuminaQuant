"""Materialize the metric-only legacy HYBRID winner into live-backed artifacts.

The full-universe report intentionally ranked metric-only saved candidates, but
real deployment can only use candidates with source streams and a live strategy
mode.  This runner reconstructs the No-HighVol legacy metric winner over saved
stream-backed sleeves, writes a deployable final-allocation artifact, and ranks
the resulting live-implementable modes with the same validation-primary scoring
policy used by the full-universe selection report.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
GROUP_ROOT = (
    REPO_ROOT
    / "var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped"
)
DEFAULT_FULL_UNIVERSE_REPORT = (
    GROUP_ROOT / "full_universe_selection_20260426" / "full_universe_selection_latest.json"
)
DEFAULT_OUTPUT_DIR = GROUP_ROOT / "legacy_metric_live_materialization_20260426"
LIVE_IMPLEMENTABLE_OUTPUT_DIR = GROUP_ROOT / "live_implementable_selection_20260426"
WAVE2_PAIR_RESEARCH_PATH = (
    REPO_ROOT
    / "var/reports/portfolio_superiority_wave2/candidate_research_wave2_filtered_latest.json"
)
WAVE2_PAIR_NAME = "pair_spread_1h_exec_tightstop_tp_bnbusdt_trxusdt_2.2_0.55"
MDD_CAP = 0.25
LEGACY_MATERIALIZATION_SLEEVES = frozenset(
    {
        "risk_off_cash",
        "soft_three_way_regime",
        "three_way_regime",
        "static_blend_76_24",
        "balanced_overlay_80_20",
        "pair_tactical_mode",
        "production_guarded_portfolio",
        "state_vwap_pair",
        "wave2_pair",
    }
)
CSV_FIELDS = (
    "rank",
    "name",
    "kind",
    "live_mode",
    "live_deployable",
    "selection_score",
    "val_scaled_score",
    "train_scaled_score",
    "oos_scaled_score_report_only",
    "train_total_return",
    "train_sharpe",
    "train_sortino",
    "train_calmar",
    "train_max_drawdown",
    "val_total_return",
    "val_sharpe",
    "val_sortino",
    "val_calmar",
    "val_max_drawdown",
    "oos_total_return",
    "oos_sharpe",
    "oos_sortino",
    "oos_calmar",
    "oos_max_drawdown",
    "cash_weight",
    "caveat",
)

_HYBRID_SPEC = importlib.util.spec_from_file_location(
    "run_hybrid_online_portfolio",
    Path(__file__).resolve().parent / "run_hybrid_online_portfolio.py",
)
if _HYBRID_SPEC is None or _HYBRID_SPEC.loader is None:
    raise RuntimeError("failed to import run_hybrid_online_portfolio helpers")
_HYBRID = importlib.util.module_from_spec(_HYBRID_SPEC)
sys.modules[_HYBRID_SPEC.name] = _HYBRID
_HYBRID_SPEC.loader.exec_module(_HYBRID)


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    if key == "total_return":
        return _safe_float(metrics.get("total_return", metrics.get("return")), 0.0)
    if key == "max_drawdown":
        return abs(_safe_float(metrics.get("max_drawdown", metrics.get("mdd")), 0.0))
    return _safe_float(metrics.get(key), 0.0)


def _scaled_score(metrics: dict[str, Any]) -> float:
    total_return = _metric_value(metrics, "total_return")
    sharpe = _metric_value(metrics, "sharpe")
    sortino = _metric_value(metrics, "sortino")
    calmar = _metric_value(metrics, "calmar")
    max_drawdown = _metric_value(metrics, "max_drawdown")
    mdd_headroom = 1.0 - min(max(max_drawdown, 0.0), MDD_CAP) / MDD_CAP
    return float(
        100.0
        * (
            0.30 * math.tanh(total_return / 0.18)
            + 0.30 * math.tanh(sharpe / 4.0)
            + 0.15 * math.tanh(sortino / 12.0)
            + 0.15 * math.tanh(calmar / 80.0)
            + 0.10 * mdd_headroom
        )
    )


def _score_from_metrics(metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
    train_scaled = _scaled_score(metrics.get("train") or {})
    val_scaled = _scaled_score(metrics.get("val") or {})
    oos_scaled = _scaled_score(metrics.get("oos") or {})
    return {
        "selection_score": float(val_scaled + 0.18 * train_scaled),
        "val_scaled_score": float(val_scaled),
        "train_scaled_score": float(train_scaled),
        "oos_scaled_score_report_only": float(oos_scaled),
    }


def _split_metrics_from_streams(
    streams: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    return _HYBRID._split_metrics_from_streams(streams)


def _streams_from_daily_map(
    daily_map: dict[str, float],
    *,
    split_config: Any,
) -> dict[str, list[dict[str, Any]]]:
    days = sorted(daily_map)
    returns = [_safe_float(daily_map[day], 0.0) for day in days]
    return _HYBRID._portfolio_return_streams_from_daily(days, returns, split_config=split_config)


def _make_cash_row(day_keys: list[str], *, split_config: Any) -> dict[str, Any]:
    streams = _HYBRID._portfolio_return_streams_from_daily(
        day_keys,
        [0.0] * len(day_keys),
        split_config=split_config,
    )
    metrics = _split_metrics_from_streams(streams)
    return {
        "candidate_id": "risk_off_cash",
        "name": "risk_off_cash",
        "strategy_class": "CashSleeve",
        "strategy_timeframe": "1d",
        "family": "cash",
        "symbols": [],
        "return_streams": streams,
        "metadata": {"source_payload_path": ""},
        **metrics,
    }


def _make_payload_sleeve_row(
    *,
    name: str,
    path: Path,
    split_config: Any,
    strategy_class: str,
    family: str,
    timeframe: str = "1d",
) -> dict[str, Any]:
    payload = _load_json(path)
    return _HYBRID._make_sleeve_row(
        sleeve_name=name,
        source_payload=payload,
        streams=_HYBRID._payload_daily_streams(payload, split_config=split_config),
        metadata={
            "strategy_class": strategy_class,
            "family": family,
            "timeframe": payload.get("strategy_timeframe") or payload.get("timeframe") or timeframe,
            "symbols": list(payload.get("symbols") or []),
            "source_payload_path": _display_path(path),
        },
    )


def _load_wave2_pair_row() -> tuple[dict[str, Any], Path]:
    payload = _load_json(WAVE2_PAIR_RESEARCH_PATH)
    candidates = [
        dict(row) for row in list(payload.get("candidates") or []) if isinstance(row, dict)
    ]
    for row in candidates:
        if str(row.get("name") or "") == WAVE2_PAIR_NAME:
            return row, WAVE2_PAIR_RESEARCH_PATH
    raise RuntimeError(
        f"wave2 pair candidate {WAVE2_PAIR_NAME} not found in {WAVE2_PAIR_RESEARCH_PATH}"
    )


def _make_wave2_sleeve_row(*, split_config: Any) -> dict[str, Any]:
    row, path = _load_wave2_pair_row()
    return _HYBRID._make_sleeve_row(
        sleeve_name="wave2_pair",
        source_payload=row,
        streams=_HYBRID._payload_daily_streams(row, split_config=split_config),
        metadata={
            "strategy_class": str(row.get("strategy_class") or "PairSpreadZScoreStrategy"),
            "family": "market_neutral_pair",
            "timeframe": str(row.get("strategy_timeframe") or row.get("timeframe") or "1h"),
            "symbols": list(row.get("symbols") or []),
            "source_payload_path": _display_path(path),
            "mapped_from_name": str(row.get("name") or ""),
        },
    )


def _build_sleeve_rows(*, split_config: Any) -> list[dict[str, Any]]:
    input_paths = {
        "soft_three_way_regime": (
            Path(_HYBRID.REFRESHED_INPUTS["soft_three_way_regime"]),
            "SoftThreeWayAllocator",
            "portfolio",
        ),
        "three_way_regime": (
            Path(_HYBRID.REFRESHED_INPUTS["three_way_regime"]),
            "ThreeWayAllocator",
            "portfolio",
        ),
        "static_blend_76_24": (
            Path(_HYBRID.REFRESHED_INPUTS["static_blend_76_24"]),
            "StaticBlendPortfolio",
            "portfolio",
        ),
        "balanced_overlay_80_20": (
            Path(_HYBRID.REFRESHED_INPUTS["balanced_overlay_80_20"]),
            "BalancedOverlayPortfolio",
            "portfolio_overlay",
        ),
        "pair_tactical_mode": (
            Path(_HYBRID.REFRESHED_INPUTS["pair_tactical_mode"]),
            "PairSpreadZScoreStrategy",
            "market_neutral_pair",
        ),
        "production_guarded_portfolio": (
            Path(_HYBRID.REFRESHED_INPUTS["production_guarded_portfolio"]),
            "ProductionGuardedPortfolio",
            "portfolio_overlay",
        ),
        "incumbent_only": (
            Path(_HYBRID.REFRESHED_INPUTS["incumbent_only"]),
            "IncumbentPortfolio",
            "portfolio",
        ),
        "autoresearch_55_45": (
            GROUP_ROOT
            / "current_switch_validation_current/refreshed_autoresearch_pair_55_45_portfolio_latest.json",
            "AutoresearchPortfolio",
            "portfolio",
        ),
        "strict_autoresearch_1x": (
            GROUP_ROOT
            / (
                "strict_blend_76_24_leverage_sweep_rerun_current/inc_1_auto_1/"
                "strict_autoresearch_portfolio_current/strict_autoresearch_portfolio_latest.json"
            ),
            "StrictAutoresearchPortfolio",
            "portfolio",
        ),
        "state_vwap_pair": (
            GROUP_ROOT
            / "portfolio_superiority_dense_pairs_current/state_vwap_pair_candidate_latest.json",
            "PairSpreadZScoreStrategy",
            "market_neutral_pair",
        ),
    }
    rows: list[dict[str, Any]] = []
    for name, (path, strategy_class, family) in input_paths.items():
        if not path.exists():
            continue
        rows.append(
            _make_payload_sleeve_row(
                name=name,
                path=path,
                split_config=split_config,
                strategy_class=strategy_class,
                family=family,
            )
        )
    rows.append(_make_wave2_sleeve_row(split_config=split_config))
    all_day_keys = sorted(
        set().union(*(_HYBRID._merged_daily_map(row["return_streams"]).keys() for row in rows))
    )
    return [_make_cash_row(all_day_keys, split_config=split_config), *rows]


def _source_sleeve_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        out[str(row.get("name") or "")] = {
            "strategy_class": str(row.get("strategy_class") or ""),
            "family": str(row.get("family") or ""),
            "strategy_timeframe": str(row.get("strategy_timeframe") or ""),
            "symbols": list(row.get("symbols") or []),
            "source_payload_path": str(
                dict(row.get("metadata") or {}).get("source_payload_path") or ""
            ),
            "train": dict(row.get("train") or {}),
            "val": dict(row.get("val") or {}),
            "oos": dict(row.get("oos") or {}),
        }
    return out


def _allocation_summary(allocations: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "oos": []}
    for allocation in allocations:
        split = str(allocation.get("split") or "")
        if split in by_split:
            by_split[split].append(dict(allocation))
    average_weights_by_split: dict[str, dict[str, float]] = {}
    average_cash_by_split: dict[str, float] = {}
    for split, items in by_split.items():
        if not items:
            average_weights_by_split[split] = {}
            average_cash_by_split[split] = 0.0
            continue
        names = sorted(
            {
                str(name)
                for item in items
                for name, weight in dict(item.get("weights") or {}).items()
                if _safe_float(weight, 0.0) > 0.0
            }
        )
        average_weights_by_split[split] = {
            name: float(
                np.mean(
                    [_safe_float(dict(item.get("weights") or {}).get(name), 0.0) for item in items]
                )
            )
            for name in names
        }
        average_cash_by_split[split] = float(
            np.mean([_safe_float(item.get("cash_weight"), 0.0) for item in items])
        )
    return {
        "average_cash_by_split": average_cash_by_split,
        "average_weights_by_split": average_weights_by_split,
    }


def _hybrid_config_from_legacy(legacy_candidate: dict[str, Any]) -> Any:
    valid_fields = {field.name for field in fields(_HYBRID.HybridOnlineConfig)}
    config_payload = {
        key: value
        for key, value in dict(legacy_candidate.get("config") or {}).items()
        if key in valid_fields
    }
    config_payload["use_current_health_priors"] = False
    return _HYBRID.HybridOnlineConfig(**config_payload)


def _run_legacy_materialization(
    *,
    rows: list[dict[str, Any]],
    full_report: dict[str, Any],
    split_config: Any,
) -> tuple[dict[str, Any], Any]:
    legacy_candidate = dict(
        dict(full_report.get("final_recommendations") or {}).get("best_full_universe_candidate")
        or {}
    )
    if str(legacy_candidate.get("name") or "") != "legacy_metric_no_highvol_baseline_raw_score":
        raise RuntimeError("full-universe report does not expose the expected legacy metric winner")
    legacy_rows = [
        row for row in rows if str(row.get("name") or "") in LEGACY_MATERIALIZATION_SLEEVES
    ]
    config = _hybrid_config_from_legacy(legacy_candidate)
    result = _HYBRID.run_hybrid_online_allocator(
        legacy_rows,
        config=config,
        refreshed_health_metrics=None,
        split_config=split_config,
    )
    result["allocation_summary"] = _allocation_summary(list(result.get("allocations") or []))
    result["materialized_sleeves"] = [str(row.get("name") or "") for row in legacy_rows]
    return result, config


def _frozen_final_allocation_replay(
    *,
    name: str,
    weights: dict[str, float],
    rows_by_name: dict[str, dict[str, Any]],
    split_config: Any,
    cash_weight: float = 0.0,
) -> dict[str, Any]:
    row_maps = {
        sleeve_name: _HYBRID._merged_daily_map(row["return_streams"])
        for sleeve_name, row in rows_by_name.items()
        if sleeve_name != "risk_off_cash"
    }
    day_keys = sorted(set().union(*(daily_map.keys() for daily_map in row_maps.values())))
    daily_map: dict[str, float] = {}
    normalized_weights = {
        str(sleeve): _safe_float(weight, 0.0)
        for sleeve, weight in dict(weights or {}).items()
        if _safe_float(weight, 0.0) > 0.0
    }
    for day_key in day_keys:
        if split_config.split_for_day_key(day_key) is None:
            continue
        daily_map[day_key] = float(
            sum(
                _safe_float(weight, 0.0) * _safe_float(row_maps.get(sleeve, {}).get(day_key), 0.0)
                for sleeve, weight in normalized_weights.items()
            )
        )
    streams = _streams_from_daily_map(daily_map, split_config=split_config)
    metrics = _split_metrics_from_streams(streams)
    return {
        "name": name,
        "weights": normalized_weights,
        "cash_weight": float(max(0.0, cash_weight)),
        "dates": sorted(daily_map),
        "daily_returns": [_safe_float(daily_map[day], 0.0) for day in sorted(daily_map)],
        "split_metrics": metrics,
        **_score_from_metrics(metrics),
    }


def _candidate_row(
    *,
    name: str,
    kind: str,
    metrics: dict[str, dict[str, Any]],
    live_mode: str = "",
    live_deployable: bool,
    cash_weight: float = 0.0,
    caveat: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scored = _score_from_metrics(metrics)
    return {
        "name": name,
        "kind": kind,
        "live_mode": live_mode,
        "live_deployable": bool(live_deployable),
        "cash_weight": float(cash_weight),
        "metrics": metrics,
        "caveat": caveat,
        **scored,
        **(extra or {}),
    }


def _live_candidate_rows(
    *,
    rows_by_name: dict[str, dict[str, Any]],
    materialized_result: dict[str, Any],
    frozen_replay: dict[str, Any],
    full_report: dict[str, Any],
    split_config: Any,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def add_static(
        name: str, weights: dict[str, float], cash_weight: float = 0.0, caveat: str = ""
    ) -> None:
        replay = _frozen_final_allocation_replay(
            name=name,
            weights=weights,
            rows_by_name=rows_by_name,
            split_config=split_config,
            cash_weight=cash_weight,
        )
        candidates.append(
            _candidate_row(
                name=name,
                kind="live_static_or_final_allocation_replay",
                live_mode=name,
                live_deployable=True,
                cash_weight=replay["cash_weight"],
                metrics=dict(replay["split_metrics"] or {}),
                caveat=caveat,
                extra={"weights": dict(replay.get("weights") or {})},
            )
        )

    add_static("core_mode", {"soft_three_way_regime": 1.0})
    add_static("balanced_overlay_mode", {"balanced_overlay_80_20": 1.0})
    add_static("defensive_overlay_mode", {"soft_three_way_regime": 0.7, "pair_tactical_mode": 0.3})
    add_static("aggressive_realized_mode", {"three_way_regime": 1.0})
    add_static("pair_tactical_mode", {"pair_tactical_mode": 1.0})
    add_static(
        "production_guarded_state_vwap_pair_mode",
        {"production_guarded_portfolio": 0.4, "state_vwap_pair": 0.25},
        cash_weight=0.35,
    )
    add_static(
        "strict_autoresearch_practical_mode",
        {"production_guarded_portfolio": 0.8, "strict_autoresearch_1x": 0.2},
    )
    add_static("risk_off_mode", {}, cash_weight=1.0)

    current_hybrid_path = (
        GROUP_ROOT / "portfolio_hybrid_online_current/hybrid_online_portfolio_latest.json"
    )
    if current_hybrid_path.exists():
        current_hybrid = _load_json(current_hybrid_path)
        current_final = dict(
            dict((current_hybrid.get("scenarios") or {}).get("refreshed_latest_tail") or {}).get(
                "final_allocation"
            )
            or {}
        )
        add_static(
            "hybrid_guarded_mode",
            {
                str(k): _safe_float(v, 0.0)
                for k, v in dict(current_final.get("weights") or {}).items()
            },
            cash_weight=_safe_float(current_final.get("cash_weight"), 0.0),
            caveat="current live HYBRID mode uses the latest saved final allocation, not the dynamic allocator path",
        )

    candidates.append(
        _candidate_row(
            name="legacy_metric_no_highvol_materialized_dynamic_stream_backtest",
            kind="stream_backed_dynamic_research_proxy",
            live_mode="",
            live_deployable=False,
            cash_weight=_safe_float(
                dict(materialized_result.get("final_allocation") or {}).get("cash_weight"),
                0.0,
            ),
            metrics=dict(materialized_result.get("split_metrics") or {}),
            caveat=(
                "stream-backed reconstruction of the metric-only winner; live strategy mode uses its final "
                "allocation unless/until a stateful live HYBRID allocator is implemented"
            ),
            extra={
                "final_allocation": dict(materialized_result.get("final_allocation") or {}),
                "allocation_summary": dict(materialized_result.get("allocation_summary") or {}),
            },
        )
    )
    candidates.append(
        _candidate_row(
            name="legacy_no_highvol_hybrid_mode",
            kind="live_final_allocation_replay",
            live_mode="legacy_no_highvol_hybrid_mode",
            live_deployable=True,
            cash_weight=_safe_float(frozen_replay.get("cash_weight"), 0.0),
            metrics=dict(frozen_replay.get("split_metrics") or {}),
            caveat=(
                "deployable through ArtifactPortfolioModeStrategy using the materialized final allocation; "
                "historical dynamic allocator score is reported separately"
            ),
            extra={"weights": dict(frozen_replay.get("weights") or {})},
        )
    )

    for generated in list(full_report.get("generated_hybrid_candidates") or []):
        if not isinstance(generated, dict):
            continue
        candidates.append(
            _candidate_row(
                name=str(generated.get("name") or ""),
                kind="full_universe_generated_stream_backtest_report_only",
                live_mode="",
                live_deployable=False,
                metrics=dict(generated.get("metrics") or {}),
                caveat="stream-backed in research report, but no committed live portfolio mode artifact yet",
                extra={
                    "source_score": _safe_float(generated.get("selection_score"), 0.0),
                    "stream_day_count": int(_safe_float(generated.get("stream_day_count"), 0.0)),
                },
            )
        )

    candidates.sort(
        key=lambda row: (
            1 if row.get("live_deployable") else 0,
            _safe_float(row.get("selection_score"), 0.0),
        ),
        reverse=True,
    )
    return candidates


def _metric_columns(row: dict[str, Any]) -> dict[str, float]:
    metrics = dict(row.get("metrics") or {})
    out: dict[str, float] = {}
    for split in ("train", "val", "oos"):
        split_metrics = dict(metrics.get(split) or {})
        for key in ("total_return", "sharpe", "sortino", "calmar", "max_drawdown"):
            out[f"{split}_{key}"] = _metric_value(split_metrics, key)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            metric_columns = _metric_columns(row)
            writer.writerow(
                {
                    "rank": idx,
                    "name": row.get("name"),
                    "kind": row.get("kind"),
                    "live_mode": row.get("live_mode"),
                    "live_deployable": row.get("live_deployable"),
                    "selection_score": row.get("selection_score"),
                    "val_scaled_score": row.get("val_scaled_score"),
                    "train_scaled_score": row.get("train_scaled_score"),
                    "oos_scaled_score_report_only": row.get("oos_scaled_score_report_only"),
                    "cash_weight": row.get("cash_weight", 0.0),
                    "caveat": row.get("caveat", ""),
                    **metric_columns,
                }
            )


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value, 0.0):+.4%}"


def _fmt_float(value: Any) -> str:
    return f"{_safe_float(value, 0.0):.4f}"


def _metrics_table(title: str, rows: list[dict[str, Any]], *, limit: int = 12) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Rank | Candidate | Live? | Score | Train ret/Sh/MDD | Val ret/Sh/MDD | OOS ret/Sh/MDD | Caveat |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for idx, row in enumerate(rows[:limit], start=1):
        metrics = dict(row.get("metrics") or {})
        train = dict(metrics.get("train") or {})
        val = dict(metrics.get("val") or {})
        oos = dict(metrics.get("oos") or {})
        lines.append(
            "| {rank} | `{name}` | {live} | {score:.2f} | {tr}/{ts}/{tmdd} | "
            "{vr}/{vs}/{vmdd} | {or_}/{os}/{omdd} | {caveat} |".format(
                rank=idx,
                name=row.get("name"),
                live="yes" if row.get("live_deployable") else "no",
                score=_safe_float(row.get("selection_score"), 0.0),
                tr=_fmt_pct(_metric_value(train, "total_return")),
                ts=_fmt_float(_metric_value(train, "sharpe")),
                tmdd=_fmt_pct(_metric_value(train, "max_drawdown")),
                vr=_fmt_pct(_metric_value(val, "total_return")),
                vs=_fmt_float(_metric_value(val, "sharpe")),
                vmdd=_fmt_pct(_metric_value(val, "max_drawdown")),
                or_=_fmt_pct(_metric_value(oos, "total_return")),
                os=_fmt_float(_metric_value(oos, "sharpe")),
                omdd=_fmt_pct(_metric_value(oos, "max_drawdown")),
                caveat=str(row.get("caveat") or ""),
            )
        )
    return lines


def _write_wave2_live_candidate(output_dir: Path) -> Path:
    row, source_path = _load_wave2_pair_row()
    thin = {
        "artifact_kind": "live_pair_component",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "candidate_id": "wave2_pair_leaf",
        "name": "wave2_pair_leaf",
        "source_candidate_id": str(row.get("candidate_id") or ""),
        "source_name": str(row.get("name") or ""),
        "strategy_class": str(row.get("strategy_class") or "PairSpreadZScoreStrategy"),
        "strategy": str(row.get("strategy") or ""),
        "family": str(row.get("family") or "market_neutral_pair"),
        "strategy_timeframe": str(row.get("strategy_timeframe") or row.get("timeframe") or "1h"),
        "symbols": list(row.get("symbols") or []),
        "params": dict(row.get("params") or {}),
        "train": dict(row.get("train") or {}),
        "val": dict(row.get("val") or {}),
        "oos": dict(row.get("oos") or {}),
        "source_path": _display_path(source_path),
    }
    path = output_dir / "wave2_pair_live_candidate_latest.json"
    path.write_text(
        json.dumps(thin, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    return path


def _write_outputs(
    *,
    output_dir: Path,
    live_output_dir: Path,
    payload: dict[str, Any],
    live_rows: list[dict[str, Any]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    live_output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"legacy_metric_live_materialization_{stamp}.json"
    latest_json_path = output_dir / "legacy_metric_live_materialization_latest.json"
    md_path = output_dir / f"legacy_metric_live_materialization_{stamp}.md"
    latest_md_path = output_dir / "legacy_metric_live_materialization_latest.md"
    csv_path = output_dir / "legacy_metric_live_materialization_candidates_20260426.csv"
    live_json_path = live_output_dir / "live_implementable_selection_latest.json"
    live_md_path = live_output_dir / "live_implementable_selection_latest.md"
    live_csv_path = live_output_dir / "live_implementable_selection_candidates_20260426.csv"

    json_text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
    json_path.write_text(json_text, encoding="utf-8")
    latest_json_path.write_text(json_text, encoding="utf-8")
    _write_csv(csv_path, live_rows)

    dynamic = payload["scenarios"]["refreshed_latest_tail"]
    frozen = payload["live_execution_model"]["final_allocation_replay"]
    gap = payload["live_execution_model"]["dynamic_vs_final_allocation_gap"]
    live_deployable = [row for row in live_rows if row.get("live_deployable")]
    best_live = live_deployable[0] if live_deployable else {}
    best_research = max(live_rows, key=lambda row: _safe_float(row.get("selection_score"), 0.0))
    lines = [
        "# Legacy Metric No-HighVol HYBRID Live Materialization",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        "- selection remains validation-primary; OOS is report-only.",
        "- cash efficiency is not scored.",
        "- MDD up to 25% remains eligible and MDD enters only through bounded headroom.",
        "",
        "## Result",
        "",
        (
            f"- Original research #1: `{payload['source_metric_candidate']['name']}` "
            f"(metric-only score `{payload['source_metric_candidate']['selection_score']:.2f}`)."
        ),
        (
            f"- Stream-backed materialized dynamic proxy: score "
            f"`{payload['materialized_scores']['selection_score']:.2f}`, val return "
            f"`{_fmt_pct(dynamic['split_metrics']['val']['total_return'])}`, "
            f"val Sharpe `{_fmt_float(dynamic['split_metrics']['val']['sharpe'])}`."
        ),
        (
            f"- Deployable final-allocation live mode: `legacy_no_highvol_hybrid_mode`, score "
            f"`{frozen['selection_score']:.2f}`, val return "
            f"`{_fmt_pct(frozen['split_metrics']['val']['total_return'])}`, "
            f"val Sharpe `{_fmt_float(frozen['split_metrics']['val']['sharpe'])}`."
        ),
        (
            f"- Best live-deployable candidate in this comparison: `{best_live.get('name')}` "
            f"(score `{_safe_float(best_live.get('selection_score'), 0.0):.2f}`)."
        ),
        (
            f"- Best research/backtest candidate shown here: `{best_research.get('name')}` "
            f"(score `{_safe_float(best_research.get('selection_score'), 0.0):.2f}`)."
        ),
        "",
        "## Why the original research #1 cannot be promoted verbatim",
        "",
        (
            "The original row has no saved daily stream and was marked `combinable=false`, so there was no "
            "strategy artifact for live execution.  This materialization rebuilds it over saved sleeves, then "
            "compares the dynamic research path against the actually deployable artifact mode."
        ),
        "",
        "## Backtest/live parity check",
        "",
        (
            f"- dynamic_vs_final_allocation val_return_gap: "
            f"`{_fmt_pct(gap['val_total_return_gap'])}`"
        ),
        (
            f"- dynamic_vs_final_allocation train_return_gap: "
            f"`{_fmt_pct(gap['train_total_return_gap'])}`"
        ),
        (
            f"- dynamic_vs_final_allocation oos_return_gap: "
            f"`{_fmt_pct(gap['oos_total_return_gap'])}`"
        ),
        (
            "A non-zero gap is expected because the live `ArtifactPortfolioModeStrategy` consumes the latest "
            "saved final allocation, while the research dynamic allocator changes allocations through history. "
            "The report therefore does not label the dynamic path as live-deployable."
        ),
        "",
        "## Final allocation for `legacy_no_highvol_hybrid_mode`",
        "",
        f"- date: `{dynamic['final_allocation'].get('date')}`",
        f"- cash_weight: `{_fmt_pct(dynamic['final_allocation'].get('cash_weight'))}`",
    ]
    for sleeve, weight in sorted(
        dict(dynamic["final_allocation"].get("weights") or {}).items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        lines.append(f"- `{sleeve}`: `{_fmt_pct(weight)}`")
    lines.extend([""])
    lines.extend(_metrics_table("Live-implementable comparison", live_rows, limit=20))
    lines.extend(["", "## Explicit caveats", ""])
    lines.extend(f"- {item}" for item in payload["explicit_caveats"])
    md_text = "\n".join(lines) + "\n"
    md_path.write_text(md_text, encoding="utf-8")
    latest_md_path.write_text(md_text, encoding="utf-8")

    live_payload = {
        "generated_at": payload["generated_at"],
        "selection_policy": payload["selection_policy"],
        "best_live_deployable": best_live,
        "best_research_backtest_report_only": best_research,
        "ranked_candidates": live_rows,
        "source_materialization_report": _display_path(latest_json_path),
    }
    live_json_path.write_text(
        json.dumps(live_payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    live_lines = [
        "# Live-Implementable Portfolio Selection",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        (
            f"- best_live_deployable: `{best_live.get('name')}` "
            f"(score `{_safe_float(best_live.get('selection_score'), 0.0):.2f}`)"
        ),
        (
            f"- best_research_backtest_report_only: `{best_research.get('name')}` "
            f"(score `{_safe_float(best_research.get('selection_score'), 0.0):.2f}`)"
        ),
        "",
    ]
    live_lines.extend(_metrics_table("Ranked candidates", live_rows, limit=30))
    live_md_path.write_text("\n".join(live_lines) + "\n", encoding="utf-8")
    _write_csv(live_csv_path, live_rows)

    return {
        "json_path": str(json_path.resolve()),
        "latest_json_path": str(latest_json_path.resolve()),
        "md_path": str(md_path.resolve()),
        "latest_md_path": str(latest_md_path.resolve()),
        "csv_path": str(csv_path.resolve()),
        "live_json_path": str(live_json_path.resolve()),
        "live_md_path": str(live_md_path.resolve()),
        "live_csv_path": str(live_csv_path.resolve()),
    }


def build_materialization_report(
    *,
    full_universe_report: Path = DEFAULT_FULL_UNIVERSE_REPORT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    live_output_dir: Path = LIVE_IMPLEMENTABLE_OUTPUT_DIR,
) -> dict[str, Any]:
    split_config = _HYBRID.HybridSplitConfig()
    full_report = _load_json(full_universe_report)
    rows = _build_sleeve_rows(split_config=split_config)
    rows_by_name = {str(row.get("name") or ""): row for row in rows}
    materialized_result, config = _run_legacy_materialization(
        rows=rows,
        full_report=full_report,
        split_config=split_config,
    )
    final_allocation = dict(materialized_result.get("final_allocation") or {})
    frozen_replay = _frozen_final_allocation_replay(
        name="legacy_no_highvol_hybrid_mode",
        weights={
            str(k): _safe_float(v, 0.0)
            for k, v in dict(final_allocation.get("weights") or {}).items()
        },
        rows_by_name=rows_by_name,
        split_config=split_config,
        cash_weight=_safe_float(final_allocation.get("cash_weight"), 0.0),
    )
    materialized_scores = _score_from_metrics(dict(materialized_result.get("split_metrics") or {}))
    live_rows = _live_candidate_rows(
        rows_by_name=rows_by_name,
        materialized_result=materialized_result,
        frozen_replay=frozen_replay,
        full_report=full_report,
        split_config=split_config,
    )
    legacy_candidate = dict(
        dict(full_report.get("final_recommendations") or {}).get("best_full_universe_candidate")
        or {}
    )
    gap = {
        "train_total_return_gap": _metric_value(
            dict(materialized_result["split_metrics"].get("train") or {}),
            "total_return",
        )
        - _metric_value(dict(frozen_replay["split_metrics"].get("train") or {}), "total_return"),
        "val_total_return_gap": _metric_value(
            dict(materialized_result["split_metrics"].get("val") or {}),
            "total_return",
        )
        - _metric_value(dict(frozen_replay["split_metrics"].get("val") or {}), "total_return"),
        "oos_total_return_gap": _metric_value(
            dict(materialized_result["split_metrics"].get("oos") or {}),
            "total_return",
        )
        - _metric_value(dict(frozen_replay["split_metrics"].get("oos") or {}), "total_return"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    wave2_live_path = _write_wave2_live_candidate(output_dir)
    payload = {
        "artifact_kind": "legacy_metric_live_materialization",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "selection_policy": {
            "selection_score": "val_scaled_score + 0.18 * train_scaled_score",
            "oos_role": "report_only",
            "cash_efficiency_scored": False,
            "train_val_mdd_cap": MDD_CAP,
            "scaled_score_formula": (
                "100*(0.30*tanh(return/0.18)+0.30*tanh(sharpe/4)+"
                "0.15*tanh(sortino/12)+0.15*tanh(calmar/80)+0.10*MDD_headroom)"
            ),
        },
        "split_windows": split_config.as_payload(),
        "source_metric_candidate": legacy_candidate,
        "materialized_scores": materialized_scores,
        "config": asdict(config),
        "source_sleeve_metrics": _source_sleeve_metrics(rows),
        "wave2_pair_live_candidate_path": _display_path(wave2_live_path),
        "live_execution_model": {
            "live_mode": "legacy_no_highvol_hybrid_mode",
            "strategy_class": "ArtifactPortfolioModeStrategy",
            "execution_model": "saved_final_allocation_expanded_to_live_child_strategies",
            "dynamic_allocator_live_supported": False,
            "final_allocation_replay": frozen_replay,
            "dynamic_vs_final_allocation_gap": gap,
        },
        "scenarios": {
            "refreshed_latest_tail": {
                "active_sleeves": list(materialized_result.get("materialized_sleeves") or []),
                **materialized_result,
                "comparison_rows": live_rows,
            }
        },
        "live_implementable_ranked_candidates": live_rows,
        "explicit_caveats": [
            "This is not investment advice and is not a live-trading authorization.",
            "The original research #1 was metric-only: no daily stream, non-combinable, and not directly executable.",
            "The restored stream-backed dynamic proxy materially differs from the original saved metrics, so the old score is not portable verbatim.",
            "The current live portfolio mode architecture executes saved final allocations; it does not yet run the HYBRID allocator statefully inside live trading.",
            "A stateful live HYBRID allocator would need portfolio/fill/PnL state integration before it can claim exact parity with the dynamic backtest path.",
            "Exchange fees, funding, partial fills, latency, order-size limits, and slippage still require paper/canary validation.",
        ],
    }
    paths = _write_outputs(
        output_dir=output_dir,
        live_output_dir=live_output_dir,
        payload=payload,
        live_rows=live_rows,
    )
    return {"payload": payload, **paths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-universe-report", type=Path, default=DEFAULT_FULL_UNIVERSE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--live-output-dir", type=Path, default=LIVE_IMPLEMENTABLE_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_materialization_report(
        full_universe_report=args.full_universe_report,
        output_dir=args.output_dir,
        live_output_dir=args.live_output_dir,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "payload"}, indent=2))


if __name__ == "__main__":
    main()
