#!/usr/bin/env python3
"""Build a sparse train/validation-selected 69-monitor core to repair clean-OOS return.

The selection rule intentionally ignores locked OOS:

* start from the clean source profile-refit sleeve rows;
* keep rows with positive train, validation >= threshold, validation MDD <= cap,
  train >= validation, and efficiency above the 10 bps proxy gate;
* keep at most one sleeve per symbol;
* rank by a train/validation robustness score and select a small core;
* leave all 69 assets in the monitor manifest for future promotion.

Locked OOS must be checked by ``run_alpha_zoo_69_asset_clean_oos_gate.py`` after
this artifact is frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_alpha_zoo_69_asset_diverse_salvage as diverse  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_optuna_hybrid_refit as broad69  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_profile_optuna_hybrid_refit as profile69  # noqa: E402

DEFAULT_SOURCE_ARTIFACT = (
    profile69.DEFAULT_OUTPUT_DIR / "alpha_zoo_69_asset_profile_optuna_hybrid_refit_latest.json"
)
DEFAULT_OUTPUT_DIR = broad69.ALPHA_V2_ROOT / "alpha_zoo_69_asset_subset_oos_repair_20260531"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "alpha_zoo_69_asset_subset_oos_repair_latest.json"
DEFAULT_OUTPUT_PROFILE_ID = "subset_tv_quality_core_gross2_69_monitor"
DEFAULT_TARGET_GROSS = 2.0
DEFAULT_MIN_VALIDATION_RETURN = 0.10
DEFAULT_MAX_VALIDATION_MDD = 0.06
DEFAULT_MAX_SLEEVES = 4
DEFAULT_MIN_RPT_BPS = broad69.RETURN_PER_TURNOVER_THRESHOLD_BPS
DEFAULT_SELECTION_MODE = "sparse_quality"
SELECTION_MODES = ("sparse_quality", "diversified_watch_core", "explicit_indices")


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(broad69._json_safe(payload), indent=2, sort_keys=True) + "\n")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def tv_quality_score(
    row: Mapping[str, Any], *, selection_mode: str = DEFAULT_SELECTION_MODE
) -> float:
    train_return = _safe_float(row.get("train_return"))
    validation_return = _safe_float(row.get("validation_return"))
    validation_mdd = _safe_float(row.get("validation_mdd"))
    train_mdd = _safe_float(row.get("train_mdd"))
    train_rpt = _safe_float(row.get("train_return_per_turnover_proxy_bps"), -100.0)
    validation_rpt = _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -100.0)
    if selection_mode == "diversified_watch_core":
        ratio = validation_return / max(train_return, 1e-9)
        return float(
            2.0 * min(train_return, validation_return)
            + 0.5 * validation_return
            + min(train_rpt, validation_rpt, 200.0) / 500.0
            - 2.0 * validation_mdd
            - 0.2 * train_mdd
            - 1.5 * abs(math.log(max(ratio, 1e-6)))
        )
    validation_spike = max(0.0, validation_return - train_return)
    return float(
        1.4 * min(train_return, validation_return)
        + 0.7 * validation_return
        + min(train_rpt, validation_rpt, 250.0) / 500.0
        - 2.0 * validation_mdd
        - 0.35 * train_mdd
        - 2.5 * validation_spike
    )


def row_passes_tv_quality_gate(
    row: Mapping[str, Any],
    *,
    selection_mode: str = DEFAULT_SELECTION_MODE,
    min_validation_return: float = DEFAULT_MIN_VALIDATION_RETURN,
    max_validation_mdd: float = DEFAULT_MAX_VALIDATION_MDD,
    min_rpt_bps: float = DEFAULT_MIN_RPT_BPS,
) -> bool:
    train_return = _safe_float(row.get("train_return"))
    validation_return = _safe_float(row.get("validation_return"))
    validation_mdd = _safe_float(row.get("validation_mdd"), 999.0)
    train_rpt = _safe_float(row.get("train_return_per_turnover_proxy_bps"), -100.0)
    validation_rpt = _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -100.0)
    common = (
        train_return > 0.0
        and validation_return >= float(min_validation_return)
        and validation_mdd <= float(max_validation_mdd)
        and train_rpt > float(min_rpt_bps)
        and validation_rpt > float(min_rpt_bps)
    )
    if selection_mode == "diversified_watch_core":
        return bool(common)
    return bool(common and train_return >= validation_return)


def select_tv_quality_core_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    selection_mode: str = DEFAULT_SELECTION_MODE,
    max_sleeves: int = DEFAULT_MAX_SLEEVES,
    min_validation_return: float = DEFAULT_MIN_VALIDATION_RETURN,
    max_validation_mdd: float = DEFAULT_MAX_VALIDATION_MDD,
    min_rpt_bps: float = DEFAULT_MIN_RPT_BPS,
) -> list[dict[str, Any]]:
    best_by_symbol: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        if not row_passes_tv_quality_gate(
            row,
            selection_mode=selection_mode,
            min_validation_return=min_validation_return,
            max_validation_mdd=max_validation_mdd,
            min_rpt_bps=min_rpt_bps,
        ):
            continue
        row["tv_quality_score"] = tv_quality_score(row, selection_mode=selection_mode)
        symbol = str(row.get("symbol"))
        current = best_by_symbol.get(symbol)
        if current is None or float(row["tv_quality_score"]) > float(current["tv_quality_score"]):
            best_by_symbol[symbol] = row
    return sorted(
        best_by_symbol.values(),
        key=lambda item: (
            float(item.get("tv_quality_score") or 0.0),
            float(item.get("validation_return") or 0.0),
            str(item.get("symbol")),
        ),
        reverse=True,
    )[: int(max_sleeves)]


def _parse_selected_row_indices(value: str | None) -> list[int]:
    if value is None or not str(value).strip():
        return []
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def select_explicit_index_rows(
    rows: Sequence[Mapping[str, Any]], selected_indices: Sequence[int]
) -> list[dict[str, Any]]:
    by_index = {int(index): dict(row) for index, row in enumerate(rows)}
    selected: list[dict[str, Any]] = []
    for index in selected_indices:
        if int(index) not in by_index:
            raise ValueError(f"selected row index not found: {index}")
        row = dict(by_index[int(index)])
        row["source_row_index"] = int(index)
        row["tv_quality_score"] = tv_quality_score(row, selection_mode="diversified_watch_core")
        selected.append(row)
    return selected


def _scale_selected_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_gross: float,
    output_profile_id: str,
    allow_upscale: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_gross = diverse.profile_gross_notional(rows)
    if source_gross <= 0.0:
        raise ValueError("selected TV-quality source gross is zero")
    effective_target = (
        float(target_gross) if allow_upscale else min(float(target_gross), source_gross)
    )
    scale = effective_target / source_gross
    scaled: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        row["source_profile_id"] = str(row.get("profile_id"))
        row["profile_id"] = output_profile_id
        row["sleeve_multiplier"] = _safe_float(row.get("sleeve_multiplier")) * scale
        row["weighted_notional_fraction"] = diverse.row_gross_notional(row)
        row["subset_oos_repair_selected"] = True
        row["subset_oos_repair_scale_factor"] = scale
        row["subset_oos_repair_policy"] = "train_validation_quality_sparse_core_69_monitor"
        scaled.append(row)
    return scaled, {
        "source_gross_notional_fraction": source_gross,
        "target_gross_notional_fraction": float(target_gross),
        "effective_gross_notional_fraction": diverse.profile_gross_notional(scaled),
        "scale_factor": scale,
        "allow_upscale": bool(allow_upscale),
    }


def build_monitor_manifest(
    *,
    universe_symbols: Sequence[str],
    selected_rows: Sequence[Mapping[str, Any]],
    train_eligible_symbols: Sequence[str],
    train_ineligible_symbols: Sequence[str],
) -> list[dict[str, Any]]:
    selected_by_symbol = {str(row.get("symbol")): dict(row) for row in selected_rows}
    train_eligible = {str(symbol) for symbol in train_eligible_symbols}
    train_ineligible = {str(symbol) for symbol in train_ineligible_symbols}
    manifest: list[dict[str, Any]] = []
    for symbol in universe_symbols:
        row = selected_by_symbol.get(str(symbol))
        if row:
            status = "sparse_core_tradable_now"
            action = "paper_core_candidate_after_clean_oos_gate"
        elif str(symbol) in train_ineligible:
            status = "future_watchlist_insufficient_train_history"
            action = "monitor_until_train_eligibility_refresh"
        elif str(symbol) in train_eligible:
            status = "eligible_shadow_monitor_not_selected"
            action = "monitor_shadow_and_retest_for_future_promotion"
        else:
            status = "unclassified_watchlist"
            action = "monitor_data_quality"
        manifest.append(
            {
                "symbol": str(symbol),
                "status": status,
                "action": action,
                "source_profile_id": row.get("source_profile_id") if row else None,
                "profile_id": row.get("profile_id") if row else None,
                "timeframe": row.get("timeframe") if row else None,
                "side": row.get("side") if row else None,
                "family": row.get("family") if row else None,
                "gross_notional_fraction": diverse.row_gross_notional(row) if row else 0.0,
                "train_return": row.get("train_return") if row else None,
                "validation_return": row.get("validation_return") if row else None,
                "validation_mdd": row.get("validation_mdd") if row else None,
                "tv_quality_score": row.get("tv_quality_score") if row else None,
            }
        )
    return manifest


def _candidate_pool_policy(
    *,
    universe_symbols: Sequence[str],
    selected_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    train_eligible_symbols: Sequence[str],
    train_ineligible_symbols: Sequence[str],
) -> dict[str, Any]:
    selected_symbols = sorted({str(row.get("symbol")) for row in selected_rows})
    stream_capable_symbols = sorted({str(row.get("symbol")) for row in source_rows})
    train_eligible = {str(symbol) for symbol in train_eligible_symbols}
    train_ineligible = {str(symbol) for symbol in train_ineligible_symbols}
    universe = [str(symbol) for symbol in universe_symbols]
    return {
        "candidate_pool_symbol_count": len(universe),
        "candidate_pool_symbols": universe,
        "all_universe_symbols_remain_candidates": True,
        "candidate_pool_is_not_equal_to_current_positions": True,
        "current_sparse_core_symbols": selected_symbols,
        "current_stream_capable_symbol_count": len(stream_capable_symbols),
        "current_stream_capable_symbols": stream_capable_symbols,
        "train_eligible_monitor_symbol_count": len(
            sorted(train_eligible.difference(selected_symbols))
        ),
        "train_ineligible_future_candidate_symbol_count": len(train_ineligible),
        "selection_contract": (
            "All 69 symbols remain in the candidate pool. Symbols without enough clean "
            "train history are monitor-only future candidates, not permanently excluded. "
            "The active allocator may hold only a small subset at any instant."
        ),
    }


def _hybrid_row(
    *,
    profile_id: str,
    gross: float,
    selected_symbol_count: int,
    future_watchlist_symbol_count: int,
    gate_policy: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "profile_id": profile_id,
        "hybrid_version": "sparse_tv_quality_oos_repair_v1",
        "source_strategy": "train_validation_quality_sparse_core_69_monitor",
        "gross_notional_fraction": gross,
        "selected_symbol_count": selected_symbol_count,
        "future_watchlist_symbol_count": future_watchlist_symbol_count,
        "fit_splits": ["train", "validation"],
        "test_set_policy": "locked_oos_report_only_after_train_validation_freeze",
        "oos_used_for_selection": False,
        "oos_used_for_parameter_fitting": False,
        "weights": {profile_id: 1.0},
        "final_weights": {profile_id: 1.0},
        "selection_reasons": [
            "sparse_core_allowed_instead_of_forcing_69_asset_investment",
            "row_selection_uses_train_validation_metrics_only",
            f"min_validation_return_{float(gate_policy['min_validation_return']):.4f}",
            f"max_validation_mdd_{float(gate_policy['max_validation_mdd']):.4f}",
            f"max_sleeves_{int(gate_policy['max_sleeves'])}",
            "all_69_assets_retained_in_monitor_manifest_for_future_promotion",
        ],
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.source_artifact).expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    output_profile_id = str(args.output_profile_id)
    raw_selected_row_indices = getattr(args, "selected_row_indices", "")
    gate_policy = {
        "selection_mode": str(args.selection_mode),
        "selected_row_indices": _parse_selected_row_indices(raw_selected_row_indices),
        "min_validation_return": float(args.min_validation_return),
        "max_validation_mdd": float(args.max_validation_mdd),
        "min_rpt_bps": float(args.min_rpt_bps),
        "max_sleeves": int(args.max_sleeves),
        "one_sleeve_per_symbol": True,
        "rank_key": "tv_quality_score",
        "locked_oos_used_for_selection": False,
    }
    source_sleeve_rows = source.get("selected_sleeve_rows") or []
    selected_row_indices = _parse_selected_row_indices(raw_selected_row_indices)
    if selected_row_indices:
        source_rows = select_explicit_index_rows(source_sleeve_rows, selected_row_indices)
    else:
        source_rows = select_tv_quality_core_rows(
            source_sleeve_rows,
            selection_mode=str(args.selection_mode),
            max_sleeves=int(args.max_sleeves),
            min_validation_return=float(args.min_validation_return),
            max_validation_mdd=float(args.max_validation_mdd),
            min_rpt_bps=float(args.min_rpt_bps),
        )
    if not source_rows:
        raise ValueError("no source sleeve passed the sparse TV-quality selection gate")
    scaled_rows, scale_policy = _scale_selected_rows(
        source_rows,
        target_gross=float(args.target_gross),
        output_profile_id=output_profile_id,
        allow_upscale=bool(args.allow_upscale),
    )
    universe_symbols = diverse._as_symbol_list(dict(source.get("universe") or {}).get("symbols"))
    train_eligible = diverse._train_eligible_symbols(source)
    train_ineligible = diverse._train_ineligible_symbols(source)
    selected_symbols = sorted({str(row.get("symbol")) for row in scaled_rows})
    monitor_manifest = build_monitor_manifest(
        universe_symbols=universe_symbols,
        selected_rows=scaled_rows,
        train_eligible_symbols=train_eligible,
        train_ineligible_symbols=train_ineligible,
    )
    hybrid = _hybrid_row(
        profile_id=output_profile_id,
        gross=float(scale_policy["effective_gross_notional_fraction"]),
        selected_symbol_count=len(selected_symbols),
        future_watchlist_symbol_count=len(train_ineligible),
        gate_policy=gate_policy,
    )
    return {
        "artifact_kind": "alpha_zoo_69_asset_subset_oos_repair",
        "generated_at_utc": _utc_now_iso(),
        "source_artifact": str(source_path),
        "source_artifact_kind": source.get("artifact_kind"),
        "universe": source.get("universe"),
        "timeframes": source.get("timeframes"),
        "data_coverage": source.get("data_coverage"),
        "split_policy": source.get("split_policy"),
        "train_eligibility": source.get("train_eligibility"),
        "subset_selection_policy": {
            "objective": "maximize clean-OOS robustness by investing only a sparse train/validation-quality core while monitoring all 69 assets",
            "output_profile_id": output_profile_id,
            **gate_policy,
            **scale_policy,
            "current_tradable_symbol_count": len(selected_symbols),
            "current_tradable_symbols": selected_symbols,
            "future_watchlist_symbol_count": len(train_ineligible),
            "future_watchlist_symbols": train_ineligible,
            "promotion_rule": (
                "non-core symbols remain monitor-only until a later train/validation refit marks them eligible, "
                "passes the same sparse sleeve gate, and then passes a clean locked-OOS report gate"
            ),
        },
        "candidate_pool_policy": _candidate_pool_policy(
            universe_symbols=universe_symbols,
            selected_rows=scaled_rows,
            source_rows=source.get("selected_sleeve_rows") or [],
            train_eligible_symbols=train_eligible,
            train_ineligible_symbols=train_ineligible,
        ),
        "asset_inclusion_manifest": monitor_manifest,
        "selected_sleeve_rows": scaled_rows,
        "selected_optuna_hybrid_profile": hybrid,
        "hybrid_v3_5_optuna": {"row": hybrid},
        "ready_for_paper": False,
        "ready_for_real": False,
        "paper_testnet_only": True,
        "real_money_execution": False,
        "real_execution_allowed": False,
        "runner_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def _render_pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):+.4%}"


def render_markdown(payload: Mapping[str, Any]) -> str:
    policy = dict(payload.get("subset_selection_policy") or {})
    lines = [
        "# 69-monitor sparse core OOS repair artifact",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        f"Source artifact: `{payload.get('source_artifact')}`",
        f"Output profile: `{policy.get('output_profile_id')}`",
        f"Effective gross: `{float(policy.get('effective_gross_notional_fraction') or 0.0):.4f}x`",
        f"Current core symbols: `{policy.get('current_tradable_symbol_count')}`",
        f"Future watchlist symbols: `{policy.get('future_watchlist_symbol_count')}`",
        "",
        "## Selection policy",
        "",
        f"- OOS used for selection: `{policy.get('locked_oos_used_for_selection')}`",
        f"- Min validation return: `{float(policy.get('min_validation_return') or 0.0):.4f}`",
        f"- Max validation MDD: `{float(policy.get('max_validation_mdd') or 0.0):.4f}`",
        f"- Max sleeves: `{policy.get('max_sleeves')}`",
        "",
        "## Selected sparse core",
        "",
        "| symbol | source profile | timeframe | side | gross | train | validation | val MDD | score |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("selected_sleeve_rows") or []:
        lines.append(
            f"| `{row.get('symbol')}` | `{row.get('source_profile_id')}` | `{row.get('timeframe')}` | "
            f"`{row.get('side')}` | {diverse.row_gross_notional(row):.4f} | "
            f"{_render_pct(row.get('train_return'))} | {_render_pct(row.get('validation_return'))} | "
            f"{_render_pct(row.get('validation_mdd'))} | {float(row.get('tv_quality_score') or 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Promotion rule",
            "",
            str(policy.get("promotion_rule")),
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--output-profile-id", default=DEFAULT_OUTPUT_PROFILE_ID)
    parser.add_argument("--target-gross", type=float, default=DEFAULT_TARGET_GROSS)
    parser.add_argument(
        "--min-validation-return", type=float, default=DEFAULT_MIN_VALIDATION_RETURN
    )
    parser.add_argument("--max-validation-mdd", type=float, default=DEFAULT_MAX_VALIDATION_MDD)
    parser.add_argument("--min-rpt-bps", type=float, default=DEFAULT_MIN_RPT_BPS)
    parser.add_argument("--max-sleeves", type=int, default=DEFAULT_MAX_SLEEVES)
    parser.add_argument("--selection-mode", choices=SELECTION_MODES, default=DEFAULT_SELECTION_MODE)
    parser.add_argument(
        "--selected-row-indices",
        default="",
        help=(
            "Optional comma-separated source selected_sleeve_rows indices. "
            "Use only for diagnostic/WFO-selected artifacts."
        ),
    )
    parser.add_argument("--allow-upscale", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(args)
    output = Path(args.output).expanduser().resolve()
    _write_json(output, payload)
    output.with_suffix(".md").write_text(render_markdown(payload), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
