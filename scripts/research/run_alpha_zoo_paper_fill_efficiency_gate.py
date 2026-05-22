#!/usr/bin/env python3
"""Gate Alpha Zoo paper/testnet fill efficiency against BBO-spread evidence.

This runner is intentionally fail-closed.  It does not start execution and it
never enables real money.  When paper/testnet fill JSONL telemetry is absent it
writes a pending gate artifact; when telemetry is supplied it checks realized
PnL per turnover against average BBO spread times a configured multiplier.
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
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_alpha_zoo_validation_march_high_leverage as high  # noqa: E402

DEFAULT_ALPHA_V2 = high.DEFAULT_ALPHA_V2
DEFAULT_SAMPLE_GUARDED_JSON = (
    DEFAULT_ALPHA_V2
    / "alpha_zoo_sample_guarded_alpha_discovery_20260520"
    / "alpha_zoo_sample_guarded_alpha_discovery_latest.json"
)
DEFAULT_MONITORING_CONTRACT_JSON = (
    DEFAULT_ALPHA_V2
    / "alpha_zoo_7x_paper_forward_preflight_20260519"
    / "paper_forward_monitoring_contract_latest.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ALPHA_V2 / "alpha_zoo_paper_fill_efficiency_gate_20260522"

PRIMARY_ROUND_TRIP_COST_BPS = 10.0
DEFAULT_BBO_SPREAD_MULTIPLIER = 5.0
DEFAULT_MIN_FILL_COUNT = 30
DEFAULT_MAX_MEAN_ALL_IN_COST_BPS = 10.0
DEFAULT_MAX_P95_ALL_IN_COST_BPS = 15.0
DEFAULT_MAX_TIMEOUT_RATE = 0.0
DEFAULT_MAX_CANCEL_RATE = 0.0
DEFAULT_MAX_PARTIAL_FILL_RATE = 0.10

DECISION_FIELDS = [
    "decision_rank",
    "status",
    "fill_count",
    "total_abs_notional_quote",
    "realized_pnl_quote",
    "realized_return_per_turnover_bps",
    "avg_bbo_spread_bps",
    "bbo_spread_multiplier",
    "return_per_turnover_threshold_bps",
    "mean_all_in_cost_bps",
    "p95_all_in_cost_bps",
    "timeout_rate",
    "cancel_rate",
    "partial_fill_rate",
    "actual_fill_efficiency_gate_pass",
    "ready_for_real",
    "real_money_execution",
    "rejection_reasons",
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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        token = line.strip()
        if not token:
            continue
        try:
            row = json.loads(token)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows


def _field(row: Mapping[str, Any], *names: str) -> Any:
    metadata = dict(row.get("metadata") or {})
    for name in names:
        if row.get(name) is not None:
            return row.get(name)
        if metadata.get(name) is not None:
            return metadata.get(name)
    return None


def _row_abs_notional(row: Mapping[str, Any]) -> float | None:
    direct = _field(row, "round_trip_notional_quote", "notional", "notional_quote", "executed_notional")
    if direct is not None:
        value = abs(_safe_float(direct, math.nan))
        return value if math.isfinite(value) and value > 0.0 else None
    entry = _field(row, "entry_notional_quote")
    exit_ = _field(row, "exit_notional_quote")
    if entry is not None or exit_ is not None:
        value = abs(_safe_float(entry)) + abs(_safe_float(exit_))
        return value if value > 0.0 else None
    quantity = _field(row, "quantity", "filled_quantity")
    fill_price = _field(row, "fill_price", "entry_fill_price")
    if quantity is not None and fill_price is not None:
        value = abs(_safe_float(quantity) * _safe_float(fill_price))
        return value if value > 0.0 else None
    return None


def _row_pnl_quote(row: Mapping[str, Any]) -> float | None:
    value = _field(row, "realized_pnl_quote", "net_pnl_quote", "pnl_quote")
    if value is not None:
        parsed = _safe_float(value, math.nan)
        return parsed if math.isfinite(parsed) else None
    before = _field(row, "equity_before")
    after = _field(row, "equity_after")
    if before is not None and after is not None:
        parsed = _safe_float(after, math.nan) - _safe_float(before, math.nan)
        return parsed if math.isfinite(parsed) else None
    return None


def _row_spread_bps(row: Mapping[str, Any]) -> float | None:
    value = _field(row, "spread_bps_at_submit", "avg_bbo_spread_bps", "bbo_spread_bps", "spread_bps")
    if value is None:
        return None
    parsed = _safe_float(value, math.nan)
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else None


def _row_all_in_cost_bps(row: Mapping[str, Any]) -> float | None:
    value = _field(row, "all_in_cost_bps", "all_in_round_trip_bps", "total_bps")
    if value is not None:
        parsed = _safe_float(value, math.nan)
        return parsed if math.isfinite(parsed) else None
    fee = _field(row, "fee_bps", "realized_fee_bps")
    slip = _field(row, "realized_slippage_bps", "slippage_bps")
    if fee is not None or slip is not None:
        return _safe_float(fee) + _safe_float(slip)
    return None


def _percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = max(0.0, min(1.0, float(pct))) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(len(ordered) - 1, lower + 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _rate(count: int, total: int) -> float:
    return float(count) / float(total) if total > 0 else 0.0


def _weighted_average(pairs: Sequence[tuple[float, float]]) -> float | None:
    denominator = sum(weight for _, weight in pairs)
    if denominator <= 0.0:
        return None
    return sum(value * weight for value, weight in pairs) / denominator


def summarize_fill_efficiency(
    rows: Sequence[Mapping[str, Any]],
    *,
    spread_multiplier: float,
    min_fill_count: int,
    max_mean_all_in_cost_bps: float,
    max_p95_all_in_cost_bps: float,
    max_timeout_rate: float,
    max_cancel_rate: float,
    max_partial_fill_rate: float,
) -> dict[str, Any]:
    fill_count = len(rows)
    notionals = [_row_abs_notional(row) for row in rows]
    pnls = [_row_pnl_quote(row) for row in rows]
    spreads = [_row_spread_bps(row) for row in rows]
    costs = [_row_all_in_cost_bps(row) for row in rows]
    valid_notionals = [value for value in notionals if value is not None]
    valid_pnls = [value for value in pnls if value is not None]
    valid_costs = [value for value in costs if value is not None]
    weighted_spread_pairs = [
        (spread, notional)
        for spread, notional in zip(spreads, notionals, strict=False)
        if spread is not None and notional is not None
    ]
    avg_spread = _weighted_average(weighted_spread_pairs)
    total_abs_notional = sum(valid_notionals)
    realized_pnl = sum(valid_pnls)
    return_per_turnover = (realized_pnl * 10_000.0 / total_abs_notional) if total_abs_notional > 0.0 else None
    threshold = avg_spread * spread_multiplier if avg_spread is not None else None
    mean_cost = mean(valid_costs) if valid_costs else None
    p95_cost = _percentile(valid_costs, 0.95)
    timeout_count = sum(_as_bool(_field(row, "timeout_flag")) for row in rows)
    cancel_count = sum(_as_bool(_field(row, "cancel_flag")) for row in rows)
    partial_count = sum(_as_bool(_field(row, "partial_fill_flag")) for row in rows)
    liquidation_count = sum(_safe_int(_field(row, "liquidation_count")) for row in rows)
    account_wipeout_count = sum(_safe_int(_field(row, "account_wipeout_count")) for row in rows)

    checks = {
        "min_fill_count": fill_count >= min_fill_count,
        "notional_present": len(valid_notionals) == fill_count and total_abs_notional > 0.0,
        "pnl_present": len(valid_pnls) == fill_count,
        "spread_present": len(weighted_spread_pairs) == fill_count and avg_spread is not None,
        "all_in_cost_present": len(valid_costs) == fill_count,
        "return_per_turnover_above_spread_multiple": bool(
            return_per_turnover is not None and threshold is not None and return_per_turnover > threshold
        ),
        "mean_all_in_cost": bool(mean_cost is not None and mean_cost <= max_mean_all_in_cost_bps),
        "p95_all_in_cost": bool(p95_cost is not None and p95_cost <= max_p95_all_in_cost_bps),
        "timeout_rate": _rate(timeout_count, fill_count) <= max_timeout_rate,
        "cancel_rate": _rate(cancel_count, fill_count) <= max_cancel_rate,
        "partial_fill_rate": _rate(partial_count, fill_count) <= max_partial_fill_rate,
        "no_liquidation": liquidation_count == 0,
        "no_account_wipeout": account_wipeout_count == 0,
    }
    reasons: list[str] = []
    if fill_count == 0:
        reasons.append("missing_paper_testnet_fill_telemetry")
    if not checks["min_fill_count"]:
        reasons.append(f"fill_count_{fill_count}_below_{min_fill_count}")
    for name, reason in (
        ("notional_present", "missing_turnover_notional_telemetry"),
        ("pnl_present", "missing_realized_pnl_telemetry"),
        ("spread_present", "missing_bbo_spread_telemetry"),
        ("all_in_cost_present", "missing_all_in_cost_telemetry"),
    ):
        if not checks[name]:
            reasons.append(reason)
    if not checks["return_per_turnover_above_spread_multiple"] and return_per_turnover is not None and threshold is not None:
        reasons.append(f"return_per_turnover_bps_{return_per_turnover:.3f}_not_above_{threshold:.3f}")
    if not checks["mean_all_in_cost"]:
        reasons.append("mean_all_in_cost_bps_above_limit_or_missing")
    if not checks["p95_all_in_cost"]:
        reasons.append("p95_all_in_cost_bps_above_limit_or_missing")
    if not checks["timeout_rate"]:
        reasons.append("timeout_rate_above_limit")
    if not checks["cancel_rate"]:
        reasons.append("cancel_rate_above_limit")
    if not checks["partial_fill_rate"]:
        reasons.append("partial_fill_rate_above_limit")
    if not checks["no_liquidation"]:
        reasons.append("liquidation_count_nonzero")
    if not checks["no_account_wipeout"]:
        reasons.append("account_wipeout_count_nonzero")

    gate_pass = fill_count > 0 and all(checks.values())
    status = "actual_fill_efficiency_gate_passed" if gate_pass else "pending_or_failed_actual_fill_efficiency_gate"
    if fill_count == 0:
        status = "pending_paper_testnet_fill_telemetry"
    return {
        "status": status,
        "fill_count": fill_count,
        "min_fill_count": min_fill_count,
        "total_abs_notional_quote": total_abs_notional,
        "realized_pnl_quote": realized_pnl,
        "realized_return_per_turnover_bps": return_per_turnover,
        "avg_bbo_spread_bps": avg_spread,
        "bbo_spread_multiplier": spread_multiplier,
        "return_per_turnover_threshold_bps": threshold,
        "mean_all_in_cost_bps": mean_cost,
        "p95_all_in_cost_bps": p95_cost,
        "max_mean_all_in_cost_bps": max_mean_all_in_cost_bps,
        "max_p95_all_in_cost_bps": max_p95_all_in_cost_bps,
        "timeout_count": timeout_count,
        "cancel_count": cancel_count,
        "partial_fill_count": partial_count,
        "timeout_rate": _rate(timeout_count, fill_count),
        "cancel_rate": _rate(cancel_count, fill_count),
        "partial_fill_rate": _rate(partial_count, fill_count),
        "liquidation_count": liquidation_count,
        "account_wipeout_count": account_wipeout_count,
        "checks": checks,
        "actual_fill_efficiency_gate_pass": gate_pass,
        "rejection_reasons": sorted(dict.fromkeys(reasons)),
    }


def _decision_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "decision_rank": 1,
            "status": summary.get("status"),
            "fill_count": summary.get("fill_count"),
            "total_abs_notional_quote": summary.get("total_abs_notional_quote"),
            "realized_pnl_quote": summary.get("realized_pnl_quote"),
            "realized_return_per_turnover_bps": summary.get("realized_return_per_turnover_bps"),
            "avg_bbo_spread_bps": summary.get("avg_bbo_spread_bps"),
            "bbo_spread_multiplier": summary.get("bbo_spread_multiplier"),
            "return_per_turnover_threshold_bps": summary.get("return_per_turnover_threshold_bps"),
            "mean_all_in_cost_bps": summary.get("mean_all_in_cost_bps"),
            "p95_all_in_cost_bps": summary.get("p95_all_in_cost_bps"),
            "timeout_rate": summary.get("timeout_rate"),
            "cancel_rate": summary.get("cancel_rate"),
            "partial_fill_rate": summary.get("partial_fill_rate"),
            "actual_fill_efficiency_gate_pass": bool(summary.get("actual_fill_efficiency_gate_pass")),
            "ready_for_real": False,
            "real_money_execution": False,
            "rejection_reasons": ";".join(str(reason) for reason in summary.get("rejection_reasons") or []),
        }
    ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DECISION_FIELDS, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: high._json_safe(row.get(field)) for field in DECISION_FIELDS})


def _markdown(payload: Mapping[str, Any]) -> str:
    summary = dict(payload.get("actual_fill_efficiency_summary") or {})
    return "\n".join(
        [
            "# Alpha Zoo paper fill efficiency gate",
            "",
            f"Generated: `{payload.get('generated_at_utc')}`",
            "",
            "This artifact is fail-closed and never enables real-money execution.",
            "",
            f"- Status: `{summary.get('status')}`",
            f"- Fill count: `{summary.get('fill_count')}`",
            f"- Realized return/turnover: `{summary.get('realized_return_per_turnover_bps')}` bps",
            f"- Avg BBO spread: `{summary.get('avg_bbo_spread_bps')}` bps",
            f"- Threshold: `{summary.get('return_per_turnover_threshold_bps')}` bps",
            f"- Gate pass: `{summary.get('actual_fill_efficiency_gate_pass')}`",
            "- `ready_for_real=false`, `real_money_execution=false`",
            "",
        ]
    )


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    sample_path = Path(args.sample_guarded_json).expanduser().resolve()
    monitoring_path = Path(args.monitoring_contract_json).expanduser().resolve()
    fill_path = Path(args.fill_jsonl).expanduser().resolve() if args.fill_jsonl else None
    sample_guarded = _load_json(sample_path)
    monitoring = _load_json(monitoring_path) if monitoring_path.exists() else {}
    if _as_bool(sample_guarded.get("real_money_execution")):
        raise ValueError("source sample-guarded artifact unexpectedly allows real-money execution")
    fill_rows = _read_jsonl(fill_path) if fill_path else []
    summary = summarize_fill_efficiency(
        fill_rows,
        spread_multiplier=_safe_float(args.bbo_spread_multiplier, DEFAULT_BBO_SPREAD_MULTIPLIER),
        min_fill_count=int(args.min_fill_count),
        max_mean_all_in_cost_bps=_safe_float(args.max_mean_all_in_cost_bps, DEFAULT_MAX_MEAN_ALL_IN_COST_BPS),
        max_p95_all_in_cost_bps=_safe_float(args.max_p95_all_in_cost_bps, DEFAULT_MAX_P95_ALL_IN_COST_BPS),
        max_timeout_rate=_safe_float(args.max_timeout_rate, DEFAULT_MAX_TIMEOUT_RATE),
        max_cancel_rate=_safe_float(args.max_cancel_rate, DEFAULT_MAX_CANCEL_RATE),
        max_partial_fill_rate=_safe_float(args.max_partial_fill_rate, DEFAULT_MAX_PARTIAL_FILL_RATE),
    )
    timestamp = _timestamp()
    latest_json = output_dir / "alpha_zoo_paper_fill_efficiency_gate_latest.json"
    timestamped_json = output_dir / f"alpha_zoo_paper_fill_efficiency_gate_{timestamp}.json"
    latest_md = output_dir / "alpha_zoo_paper_fill_efficiency_gate_latest.md"
    decisions_csv = output_dir / "paper_fill_efficiency_decisions_latest.csv"
    generation_log = output_dir / "artifact_generation_validation_latest.log"
    payload: dict[str, Any] = {
        "artifact_kind": "alpha_zoo_paper_fill_efficiency_gate",
        "generated_at_utc": _utc_now_iso(),
        "ready_for_paper": False,
        "ready_for_real": False,
        "real_money_execution": False,
        "paper_execution_allowed": False,
        "paper_testnet_only": True,
        "source_sample_guarded_json": str(sample_path),
        "source_monitoring_contract_json": str(monitoring_path),
        "source_fill_jsonl": str(fill_path) if fill_path else None,
        "source_monitoring_status": monitoring.get("status"),
        "source_sample_guarded_decision": sample_guarded.get("decision"),
        "gate_policy": {
            "primary_round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
            "min_fill_count": int(args.min_fill_count),
            "bbo_spread_multiplier": _safe_float(args.bbo_spread_multiplier, DEFAULT_BBO_SPREAD_MULTIPLIER),
            "return_per_turnover_formula": "sum(realized_pnl_quote) * 10000 / sum(abs(notional_quote))",
            "avg_bbo_spread_formula": "notional-weighted average spread_bps_at_submit",
            "threshold_formula": "avg_bbo_spread_bps * bbo_spread_multiplier",
            "real_money_gate": "always_false",
            "missing_telemetry_policy": "fail_closed_pending_paper_testnet_fills",
        },
        "actual_fill_efficiency_summary": summary,
        "paper_fill_efficiency_decisions": _decision_rows(summary),
        "output_paths": {
            "latest_json": str(latest_json),
            "timestamped_json": str(timestamped_json),
            "latest_markdown": str(latest_md),
            "paper_fill_efficiency_decisions_csv": str(decisions_csv),
            "artifact_generation_validation_log": str(generation_log),
        },
    }
    _write_json(latest_json, payload)
    _write_json(timestamped_json, payload)
    latest_md.write_text(_markdown(payload), encoding="utf-8")
    _write_csv(decisions_csv, payload["paper_fill_efficiency_decisions"])
    generation_log.write_text(
        "\n".join(
            [
                f"generated_at_utc={payload['generated_at_utc']}",
                f"artifact_kind={payload['artifact_kind']}",
                f"status={summary['status']}",
                f"fill_count={summary['fill_count']}",
                f"actual_fill_efficiency_gate_pass={str(summary['actual_fill_efficiency_gate_pass']).lower()}",
                "ready_for_real=false",
                "real_money_execution=false",
                f"latest_json={latest_json}",
                f"timestamped_json={timestamped_json}",
                f"paper_fill_efficiency_decisions_csv={decisions_csv}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-guarded-json", default=str(DEFAULT_SAMPLE_GUARDED_JSON))
    parser.add_argument("--monitoring-contract-json", default=str(DEFAULT_MONITORING_CONTRACT_JSON))
    parser.add_argument("--fill-jsonl", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bbo-spread-multiplier", type=float, default=DEFAULT_BBO_SPREAD_MULTIPLIER)
    parser.add_argument("--min-fill-count", type=int, default=DEFAULT_MIN_FILL_COUNT)
    parser.add_argument("--max-mean-all-in-cost-bps", type=float, default=DEFAULT_MAX_MEAN_ALL_IN_COST_BPS)
    parser.add_argument("--max-p95-all-in-cost-bps", type=float, default=DEFAULT_MAX_P95_ALL_IN_COST_BPS)
    parser.add_argument("--max-timeout-rate", type=float, default=DEFAULT_MAX_TIMEOUT_RATE)
    parser.add_argument("--max-cancel-rate", type=float, default=DEFAULT_MAX_CANCEL_RATE)
    parser.add_argument("--max-partial-fill-rate", type=float, default=DEFAULT_MAX_PARTIAL_FILL_RATE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    payload = build_payload(parse_args(argv))
    print(json.dumps(payload["output_paths"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
