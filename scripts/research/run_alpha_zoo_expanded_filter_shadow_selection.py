#!/usr/bin/env python3
"""Summarize expanded Alpha Zoo 10bps filter retune shadow candidates.

The expanded retune can surface high-validation, positive locked-OOS candidates
that still fail live promotion gates.  This runner freezes those as shadow-only
research hypotheses while preserving the hard real-money prohibition.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_alpha_zoo_validation_march_high_leverage as high  # noqa: E402

DEFAULT_ALPHA_V2 = high.DEFAULT_ALPHA_V2
DEFAULT_EXPANDED_RETUNE_JSON = (
    DEFAULT_ALPHA_V2
    / "alpha_zoo_10bps_expanded_filter_retune_20260520"
    / "alpha_zoo_10bps_full_retune_latest.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ALPHA_V2 / "alpha_zoo_expanded_filter_shadow_selection_20260520"
PRIMARY_ROUND_TRIP_COST_BPS = 10.0

SHADOW_FIELDS = [
    "role",
    "rank",
    "model_id",
    "candidate_name",
    "variant_name",
    "leverage",
    "allocation_fraction",
    "target_notional_fraction_of_equity",
    "validation_return",
    "validation_mdd",
    "validation_sharpe",
    "validation_trade_event_count",
    "train_return",
    "train_mdd",
    "locked_oos_return",
    "locked_oos_mdd",
    "locked_oos_sharpe",
    "locked_oos_trade_event_count",
    "locked_oos_liquidation_count",
    "primary_10bps_promotion_gate_pass",
    "live_promotable_10bps",
    "ready_for_paper",
    "ready_for_real",
    "shadow_status",
    "gate_reasons",
    "trade_filter_params",
]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(high._json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return json.dumps(high._json_safe(value), sort_keys=True)
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    return high._json_safe(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _gate_reasons(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("primary_10bps_promotion_gate_reasons") or []
    if isinstance(raw, str):
        return [item for item in raw.split(";") if item]
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        return [str(item) for item in raw if str(item)]
    return [str(raw)] if raw else []


def _metrics_by_model(retune: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for row in retune.get("candidate_model_metrics") or []:
        if not isinstance(row, Mapping):
            continue
        model_id = str(row.get("model_id") or "")
        split = str(row.get("split") or "")
        if model_id and split in high.SPLIT_ORDER:
            by_model.setdefault(model_id, {})[split] = dict(row)
    return {
        model_id: splits
        for model_id, splits in by_model.items()
        if all(split in splits for split in high.SPLIT_ORDER)
    }


def _live_gate_pass(splits: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(
        _as_bool(dict(splits[split]).get("primary_10bps_promotion_gate_pass"))
        and _as_bool(dict(splits[split]).get("live_promotable_10bps"))
        for split in high.SPLIT_ORDER
    )


def _summary(
    model_id: str, splits: Mapping[str, Mapping[str, Any]], *, rank: int, role: str
) -> dict[str, Any]:
    train = dict(splits["train"])
    validation = dict(splits["validation"])
    locked = dict(splits["locked_oos"])
    reasons: set[str] = set()
    for row in (train, validation, locked):
        reasons.update(_gate_reasons(row))
    leverage = _safe_float(validation.get("leverage"))
    allocation = _safe_float(validation.get("allocation_fraction"))
    live_gate = _live_gate_pass(splits)
    return {
        "role": role,
        "rank": rank,
        "model_id": model_id,
        "candidate_name": validation.get("candidate_name"),
        "model_kind": validation.get("model_kind"),
        "variant_name": validation.get("variant_name"),
        "trade_filter_params": dict(validation.get("trade_filter_params") or {}),
        "leverage": leverage,
        "allocation_fraction": allocation,
        "target_notional_fraction_of_equity": leverage * allocation,
        "validation_return": _safe_float(validation.get("total_return")),
        "validation_mdd": _safe_float(validation.get("max_drawdown")),
        "validation_sharpe": _safe_float(validation.get("sharpe")),
        "validation_trade_event_count": validation.get("trade_event_count"),
        "train_return": _safe_float(train.get("total_return")),
        "train_mdd": _safe_float(train.get("max_drawdown")),
        "locked_oos_return": _safe_float(locked.get("total_return")),
        "locked_oos_mdd": _safe_float(locked.get("max_drawdown")),
        "locked_oos_sharpe": _safe_float(locked.get("sharpe")),
        "locked_oos_trade_event_count": locked.get("trade_event_count"),
        "locked_oos_liquidation_count": _safe_float(locked.get("liquidation_count")),
        "locked_oos_account_wipeout_count": _safe_float(locked.get("account_wipeout_count")),
        "primary_10bps_promotion_gate_pass": live_gate,
        "live_promotable_10bps": live_gate,
        "ready_for_paper": False,
        "ready_for_real": False,
        "real_money_execution": False,
        "shadow_status": "shadow_only_gate_failed" if not live_gate else "paper_candidate",
        "gate_reasons": sorted(reasons),
    }


def _candidate_rows(retune: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model_id, splits in _metrics_by_model(retune).items():
        rows.append(_summary(model_id, splits, rank=0, role="expanded_filter_candidate"))
    return rows


def _sort_key(row: Mapping[str, Any]) -> tuple[float, float, float, str]:
    return (
        _safe_float(row.get("validation_return")),
        _safe_float(row.get("locked_oos_return")),
        _safe_float(row.get("train_return")),
        str(row.get("model_id") or ""),
    )


def _rank(rows: Sequence[Mapping[str, Any]], *, role: str, limit: int) -> list[dict[str, Any]]:
    ranked = [dict(row, role=role) for row in sorted(rows, key=_sort_key, reverse=True)[:limit]]
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Alpha Zoo expanded-filter shadow selection",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        "These are shadow-only research hypotheses. `ready_for_paper=false`, `ready_for_real=false`, and `real_money_execution=false`.",
        "",
        "## Best positive-OOS validation candidates",
        "",
        "| Rank | Model | Val return | Val trades | Train return | Locked-OOS return | Gate status |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload.get("positive_oos_shadow_candidates") or []:
        lines.append(
            f"| {row.get('rank')} | `{row.get('model_id')}` | "
            f"{_safe_float(row.get('validation_return')):.4%} | "
            f"{row.get('validation_trade_event_count')} | "
            f"{_safe_float(row.get('train_return')):.4%} | "
            f"{_safe_float(row.get('locked_oos_return')):.4%} | {row.get('shadow_status')} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    expanded_path = Path(args.expanded_retune_json).expanduser().resolve()
    retune = _load_json(expanded_path)
    if (
        _safe_float(retune.get("round_trip_slippage_fee_bps_primary"))
        != PRIMARY_ROUND_TRIP_COST_BPS
    ):
        raise ValueError("expanded shadow selection requires a 10bps retune artifact")
    rows = _candidate_rows(retune)
    positive_oos = [
        row
        for row in rows
        if _safe_float(row.get("validation_return")) > float(args.min_validation_return)
        and _safe_float(row.get("locked_oos_return")) > 0.0
        and _safe_float(row.get("locked_oos_liquidation_count")) == 0.0
        and not bool(row.get("live_promotable_10bps"))
    ]
    positive_oos = _rank(positive_oos, role="positive_oos_shadow_candidate", limit=int(args.top_n))
    long_only_reversal = [
        row
        for row in positive_oos
        if row.get("candidate_name") == "alpha_zoo_high_confidence_long_only"
        and dict(row.get("trade_filter_params") or {}).get("dominant_factor_family")
        == "crypto_residual_reversal"
    ]
    conservative = [
        row
        for row in rows
        if "conservative_exit" in str(row.get("candidate_name") or "")
        and _safe_float(row.get("locked_oos_return")) > 0.0
        and _safe_float(row.get("locked_oos_liquidation_count")) == 0.0
    ]
    conservative = _rank(
        conservative, role="conservative_exit_positive_oos_candidate", limit=int(args.top_n)
    )
    live_rows = [row for row in rows if bool(row.get("live_promotable_10bps"))]
    timestamp = _timestamp()
    latest_json = output_dir / "alpha_zoo_expanded_filter_shadow_selection_latest.json"
    timestamped_json = output_dir / f"alpha_zoo_expanded_filter_shadow_selection_{timestamp}.json"
    latest_md = output_dir / "alpha_zoo_expanded_filter_shadow_selection_latest.md"
    positive_csv = output_dir / "positive_oos_shadow_candidates_latest.csv"
    conservative_csv = output_dir / "conservative_exit_positive_oos_candidates_latest.csv"
    generation_log = output_dir / "artifact_generation_validation_latest.log"
    payload: dict[str, Any] = {
        "artifact_kind": "alpha_zoo_expanded_filter_shadow_selection",
        "generated_at_utc": _utc_now_iso(),
        "research_primary_round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
        "source_expanded_retune_json": str(expanded_path),
        "ready_for_paper": False,
        "ready_for_real": False,
        "real_money_execution": False,
        "expanded_retune_summary": {
            "model_count": len(rows),
            "live_promotable_count": len(live_rows),
            "positive_oos_shadow_candidate_count": len(positive_oos),
            "positive_oos_shadow_family": "alpha_zoo_high_confidence_long_only / crypto_residual_reversal"
            if long_only_reversal
            else None,
            "conservative_exit_positive_oos_count": len(conservative),
            "retune_memory_summary": dict(retune.get("memory_summary") or {}),
            "candidate_universe_summary": dict(retune.get("candidate_universe_summary") or {}),
        },
        "positive_oos_shadow_candidates": positive_oos,
        "long_only_reversal_shadow_candidates": long_only_reversal,
        "conservative_exit_positive_oos_candidates": conservative,
        "decision": {
            "paper_execution_allowed": False,
            "shadow_observation_allowed": True,
            "why_not_paper": [
                "primary 10bps promotion gate failed",
                "train metrics are not above validation/locked-OOS metrics for the leading candidates",
                "validation and locked-OOS trade samples are small for the new high-validation family",
            ],
            "next_experiment": "Run a dedicated train+validation-only high_confidence_long_only crypto_residual_reversal retune with stricter minimum trade-count guards before any paper execution artifact.",
        },
        "selection_policy": {
            "selection_inputs": ["train", "validation"],
            "uses_locked_oos_for_selection": False,
            "locked_oos_role": "gate_report_only_after_expanded_filter_freeze",
            "shadow_candidate_filter": "validation_return > threshold, locked_oos_return > 0, locked_oos_liquidation_count == 0, live_promotable_10bps == false",
        },
        "output_paths": {
            "latest_json": str(latest_json),
            "timestamped_json": str(timestamped_json),
            "latest_markdown": str(latest_md),
            "positive_oos_shadow_csv": str(positive_csv),
            "conservative_exit_positive_oos_csv": str(conservative_csv),
            "artifact_generation_validation_log": str(generation_log),
        },
    }
    _write_json(latest_json, payload)
    _write_json(timestamped_json, payload)
    latest_md.write_text(_markdown(payload), encoding="utf-8")
    _write_csv(positive_csv, positive_oos, SHADOW_FIELDS)
    _write_csv(conservative_csv, conservative, SHADOW_FIELDS)
    generation_log.write_text(
        "\n".join(
            [
                f"generated_at_utc={payload['generated_at_utc']}",
                f"artifact_kind={payload['artifact_kind']}",
                f"model_count={len(rows)}",
                f"positive_oos_shadow_candidate_count={len(positive_oos)}",
                f"conservative_exit_positive_oos_count={len(conservative)}",
                "ready_for_paper=false",
                "ready_for_real=false",
                "real_money_execution=false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-retune-json", default=str(DEFAULT_EXPANDED_RETUNE_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-validation-return", type=float, default=0.01)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    payload = build_payload(parse_args(argv))
    print(json.dumps(payload["output_paths"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
