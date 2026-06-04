#!/usr/bin/env python3
"""Build a max-diversity 69-asset salvage artifact from a clean profile refit.

This runner is intentionally conservative:

* only train/validation-eligible symbols receive live tradable sleeves now;
* train-ineligible symbols stay in a future-watchlist manifest instead of being
  deleted from the universe;
* the selected source profile is scaled by a train/validation-declared gross
  cap before any locked-OOS gate is run by the separate clean-OOS gate runner.
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

from scripts.research import run_alpha_zoo_69_asset_optuna_hybrid_refit as broad69  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_profile_optuna_hybrid_refit as profile69  # noqa: E402

DEFAULT_SOURCE_ARTIFACT = (
    profile69.DEFAULT_OUTPUT_DIR / "alpha_zoo_69_asset_profile_optuna_hybrid_refit_latest.json"
)
DEFAULT_OUTPUT_DIR = broad69.ALPHA_V2_ROOT / "alpha_zoo_69_asset_diverse_salvage_20260531"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "alpha_zoo_69_asset_diverse_salvage_latest.json"
DEFAULT_TARGET_GROSS = 2.0
DEFAULT_MAX_VALIDATION_MDD = 0.12
DEFAULT_DIVERSE_PROFILE_ID = "diverse_69_asset_balanced_core_gross2_profile"


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(broad69._json_safe(payload), indent=2, sort_keys=True) + "\n")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _as_symbol_list(value: Any) -> list[str]:
    return [str(item) for item in (value or []) if str(item)]


def row_gross_notional(row: Mapping[str, Any]) -> float:
    """Return the replayed gross notional contribution for one sleeve row."""
    return abs(_as_float(row.get("notional_fraction")) * _as_float(row.get("sleeve_multiplier")))


def profile_gross_notional(rows: Sequence[Mapping[str, Any]]) -> float:
    return float(sum(row_gross_notional(row) for row in rows))


def _profile_row_by_id(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("profile_id")): dict(row) for row in payload.get("profile_rows") or []}


def _rows_by_profile(payload: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in payload.get("selected_sleeve_rows") or []:
        row = dict(raw)
        grouped.setdefault(str(row.get("profile_id")), []).append(row)
    return grouped


def _train_eligible_symbols(payload: Mapping[str, Any]) -> list[str]:
    eligibility = dict(payload.get("train_eligibility") or {})
    explicit = _as_symbol_list(eligibility.get("train_eligible_symbols"))
    if explicit:
        return explicit
    return _as_symbol_list(dict(payload.get("universe") or {}).get("symbols"))


def _train_ineligible_symbols(payload: Mapping[str, Any]) -> list[str]:
    eligibility = dict(payload.get("train_eligibility") or {})
    return _as_symbol_list(eligibility.get("train_ineligible_symbols"))


def _profile_selection_tuple(
    *,
    profile_id: str,
    rows: Sequence[Mapping[str, Any]],
    profile_row: Mapping[str, Any],
    train_eligible: set[str],
    max_validation_mdd: float,
) -> tuple[int, int, int, float, float, float, str]:
    covered = {str(row.get("symbol")) for row in rows if str(row.get("symbol")) in train_eligible}
    positive_train_val = int(
        _as_float(profile_row.get("train_return")) > 0.0
        and _as_float(profile_row.get("validation_return")) > 0.0
    )
    ready_for_paper = int(bool(profile_row.get("ready_for_paper")))
    validation_mdd = _as_float(profile_row.get("validation_mdd"), default=999.0)
    validation_rpt = _as_float(profile_row.get("validation_return_per_turnover_proxy_bps"))
    train_rpt = _as_float(profile_row.get("train_return_per_turnover_proxy_bps"))
    max_mdd_ok = int(validation_mdd <= max_validation_mdd)
    return (
        len(covered),
        positive_train_val,
        ready_for_paper,
        max_mdd_ok,
        -validation_mdd,
        min(train_rpt, validation_rpt),
        profile_id,
    )


def select_source_profile(
    payload: Mapping[str, Any],
    *,
    requested_profile_id: str | None = None,
    max_validation_mdd: float = DEFAULT_MAX_VALIDATION_MDD,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Select the highest-coverage train/validation-safe source profile.

    The score intentionally does not inspect locked-OOS metrics.  When a
    profile is requested explicitly, this function only validates that it
    exists.
    """
    grouped = _rows_by_profile(payload)
    profile_rows = _profile_row_by_id(payload)
    if requested_profile_id:
        rows = grouped.get(requested_profile_id)
        if not rows:
            raise ValueError(f"requested source profile not found: {requested_profile_id}")
        return requested_profile_id, rows, dict(profile_rows.get(requested_profile_id) or {})

    train_eligible = set(_train_eligible_symbols(payload))
    candidates: list[tuple[tuple[int, int, int, float, float, float, str], str]] = []
    for profile_id, rows in grouped.items():
        profile_row = profile_rows.get(profile_id) or {}
        candidates.append(
            (
                _profile_selection_tuple(
                    profile_id=profile_id,
                    rows=rows,
                    profile_row=profile_row,
                    train_eligible=train_eligible,
                    max_validation_mdd=max_validation_mdd,
                ),
                profile_id,
            )
        )
    if not candidates:
        raise ValueError("source artifact has no selected_sleeve_rows to salvage")
    _, selected_profile_id = max(candidates, key=lambda item: item[0])
    return (
        selected_profile_id,
        grouped[selected_profile_id],
        dict(profile_rows.get(selected_profile_id) or {}),
    )


def build_asset_inclusion_manifest(
    *,
    universe_symbols: Sequence[str],
    selected_rows: Sequence[Mapping[str, Any]],
    train_eligible_symbols: Sequence[str],
    train_ineligible_symbols: Sequence[str],
) -> list[dict[str, Any]]:
    selected_by_symbol = {str(row.get("symbol")): dict(row) for row in selected_rows}
    train_eligible = set(train_eligible_symbols)
    train_ineligible = set(train_ineligible_symbols)
    manifest: list[dict[str, Any]] = []
    for symbol in universe_symbols:
        row = selected_by_symbol.get(str(symbol))
        if row:
            status = "tradable_now_train_eligible"
        elif str(symbol) in train_ineligible:
            status = "future_watchlist_insufficient_train_history"
        elif str(symbol) in train_eligible:
            status = "eligible_but_not_selected"
        else:
            status = "unclassified_watchlist"
        manifest.append(
            {
                "symbol": str(symbol),
                "status": status,
                "source_profile_id": row.get("source_profile_id") if row else None,
                "profile_id": row.get("profile_id") if row else None,
                "timeframe": row.get("timeframe") if row else None,
                "side": row.get("side") if row else None,
                "family": row.get("family") if row else None,
                "gross_notional_fraction": row_gross_notional(row) if row else 0.0,
            }
        )
    return manifest


def scale_rows_to_target_gross(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_gross: float,
    output_profile_id: str,
    allow_upscale: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_gross = profile_gross_notional(rows)
    if source_gross <= 0.0:
        raise ValueError("source profile gross notional is zero")
    effective_target = float(target_gross)
    if not allow_upscale:
        effective_target = min(effective_target, source_gross)
    scale = effective_target / source_gross
    scaled: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        source_profile_id = str(row.get("profile_id"))
        row["source_profile_id"] = source_profile_id
        row["profile_id"] = output_profile_id
        row["sleeve_multiplier"] = _as_float(row.get("sleeve_multiplier")) * scale
        row["weighted_notional_fraction"] = row_gross_notional(row)
        row["diverse_salvage_selected"] = True
        row["diverse_salvage_scale_factor"] = scale
        row["diverse_salvage_policy"] = "max_train_eligible_coverage_future_watchlist_gross_cap"
        scaled.append(row)
    return scaled, {
        "source_gross_notional_fraction": source_gross,
        "target_gross_notional_fraction": float(target_gross),
        "effective_gross_notional_fraction": profile_gross_notional(scaled),
        "scale_factor": scale,
        "allow_upscale": bool(allow_upscale),
    }


def _hybrid_row(
    *,
    profile_id: str,
    source_profile_id: str,
    source_profile_row: Mapping[str, Any],
    target_gross: float,
    selected_symbol_count: int,
    future_watchlist_symbol_count: int,
) -> dict[str, Any]:
    return {
        "profile_id": profile_id,
        "hybrid_version": "v3_5_diverse_balanced_core_salvage",
        "source_strategy": "hybrid_v3_5_optuna_three_profile_blend_salvage",
        "source_profile_id": source_profile_id,
        "source_profile_train_return": source_profile_row.get("train_return"),
        "source_profile_validation_return": source_profile_row.get("validation_return"),
        "source_profile_train_mdd": source_profile_row.get("train_mdd"),
        "source_profile_validation_mdd": source_profile_row.get("validation_mdd"),
        "gross_notional_fraction": target_gross,
        "selected_symbol_count": selected_symbol_count,
        "future_watchlist_symbol_count": future_watchlist_symbol_count,
        "fit_splits": ["train", "validation"],
        "test_set_policy": "locked_oos_report_only_after_train_validation_freeze",
        "oos_used_for_selection": False,
        "oos_used_for_parameter_fitting": False,
        "weights": {profile_id: 1.0},
        "final_weights": {profile_id: 1.0},
        "selection_reasons": [
            "max_current_train_eligible_symbol_coverage",
            "train_ineligible_assets_retained_as_future_watchlist",
            "source_profile_selected_by_train_validation_readiness_not_locked_oos",
            f"gross_notional_capped_at_{target_gross:.4f}",
        ],
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.source_artifact).expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_profile_id, source_rows, source_profile_row = select_source_profile(
        source,
        requested_profile_id=args.source_profile_id,
        max_validation_mdd=float(args.max_validation_mdd),
    )
    output_profile_id = str(args.output_profile_id)
    scaled_rows, scale_payload = scale_rows_to_target_gross(
        source_rows,
        target_gross=float(args.target_gross),
        output_profile_id=output_profile_id,
        allow_upscale=bool(args.allow_upscale),
    )
    universe_symbols = _as_symbol_list(dict(source.get("universe") or {}).get("symbols"))
    train_eligible = _train_eligible_symbols(source)
    train_ineligible = _train_ineligible_symbols(source)
    selected_symbols = sorted({str(row.get("symbol")) for row in scaled_rows})
    hybrid = _hybrid_row(
        profile_id=output_profile_id,
        source_profile_id=source_profile_id,
        source_profile_row=source_profile_row,
        target_gross=float(scale_payload["effective_gross_notional_fraction"]),
        selected_symbol_count=len(selected_symbols),
        future_watchlist_symbol_count=len(train_ineligible),
    )
    manifest = build_asset_inclusion_manifest(
        universe_symbols=universe_symbols,
        selected_rows=scaled_rows,
        train_eligible_symbols=train_eligible,
        train_ineligible_symbols=train_ineligible,
    )
    return {
        "artifact_kind": "alpha_zoo_69_asset_diverse_salvage",
        "generated_at_utc": _utc_now_iso(),
        "source_artifact": str(source_path),
        "source_artifact_kind": source.get("artifact_kind"),
        "universe": source.get("universe"),
        "timeframes": source.get("timeframes"),
        "data_coverage": source.get("data_coverage"),
        "split_policy": source.get("split_policy"),
        "train_eligibility": source.get("train_eligibility"),
        "diversity_policy": {
            "objective": "keep_69_asset_universe_and_trade_the_maximum_train_eligible_diverse_set_now",
            "locked_oos_used_for_selection": False,
            "source_profile_id": source_profile_id,
            "output_profile_id": output_profile_id,
            **scale_payload,
            "current_tradable_symbol_count": len(selected_symbols),
            "current_tradable_symbols": selected_symbols,
            "future_watchlist_symbol_count": len(train_ineligible),
            "future_watchlist_symbols": train_ineligible,
            "eligibility_refresh_rule": (
                "promote future-watchlist symbols only after a later refit marks them train-eligible "
                "and they pass train/validation sleeve gates"
            ),
        },
        "asset_inclusion_manifest": manifest,
        "selected_sleeve_rows": scaled_rows,
        "hybrid_v3_5_optuna": {"row": hybrid},
        "selected_optuna_hybrid_profile": hybrid,
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
    policy = dict(payload.get("diversity_policy") or {})
    lines = [
        "# 69-asset diverse salvage artifact",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        f"Source artifact: `{payload.get('source_artifact')}`",
        f"Source profile: `{policy.get('source_profile_id')}`",
        f"Output profile: `{policy.get('output_profile_id')}`",
        f"Effective gross: `{float(policy.get('effective_gross_notional_fraction') or 0.0):.4f}x`",
        f"Current tradable symbols: `{policy.get('current_tradable_symbol_count')}`",
        f"Future watchlist symbols: `{policy.get('future_watchlist_symbol_count')}`",
        "",
        "## Current tradable universe",
        "",
        ", ".join(f"`{symbol}`" for symbol in policy.get("current_tradable_symbols") or []),
        "",
        "## Future watchlist policy",
        "",
        str(policy.get("eligibility_refresh_rule")),
        "",
        "## Selected sleeves",
        "",
        "| symbol | source profile | timeframe | side | gross | train | validation | val MDD | status |",
        "|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    status_by_symbol = {
        str(row.get("symbol")): str(row.get("status"))
        for row in payload.get("asset_inclusion_manifest") or []
    }
    for row in payload.get("selected_sleeve_rows") or []:
        lines.append(
            f"| `{row.get('symbol')}` | `{row.get('source_profile_id')}` | "
            f"`{row.get('timeframe')}` | `{row.get('side')}` | "
            f"{row_gross_notional(row):.4f} | {_render_pct(row.get('train_return'))} | "
            f"{_render_pct(row.get('validation_return'))} | "
            f"{_render_pct(row.get('validation_mdd'))} | "
            f"`{status_by_symbol.get(str(row.get('symbol')), '')}` |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--source-profile-id", default=None)
    parser.add_argument("--output-profile-id", default=DEFAULT_DIVERSE_PROFILE_ID)
    parser.add_argument("--target-gross", type=float, default=DEFAULT_TARGET_GROSS)
    parser.add_argument("--max-validation-mdd", type=float, default=DEFAULT_MAX_VALIDATION_MDD)
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
