#!/usr/bin/env python3
"""Build a combined multi-asset monitoring slate from frozen Alpha Zoo artifacts.

This runner does not discover, fit, rank, or prune parameters with locked-OOS data.
It consolidates already-frozen train+validation discovery artifacts into a
paper/testnet-only monitoring book so that SOL/ETH leaders, TRX paper candidates,
and shadow-only cross-asset probes can be monitored together by symbol/asset group.
Real-money execution remains disabled and fails closed if any input artifact enables
real execution flags.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.research_universe import (  # noqa: E402
    BINANCE_TRADFI_ENERGY_INDUSTRIAL_COMMODITY_SYMBOLS,
    BINANCE_TRADFI_EQUITY_SYMBOLS,
    BINANCE_TRADFI_ETF_INDEX_SYMBOLS,
    BINANCE_TRADFI_PRECIOUS_METAL_SYMBOLS,
    BINANCE_TRADFI_PREMARKET_SYMBOLS,
)
from scripts.research import run_alpha_zoo_30m_plus_alpha_feedback_discovery as feedback  # noqa: E402

ALPHA_V2_ROOT = REPO_ROOT / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2"
DEFAULT_OUTPUT_DIR = ALPHA_V2_ROOT / "alpha_zoo_multi_asset_monitoring_slate_20260524"

DEFAULT_SOURCE_ARTIFACTS: tuple[tuple[str, Path], ...] = (
    (
        "debounced_efficiency_repair",
        ALPHA_V2_ROOT / "alpha_zoo_debounced_efficiency_repair_discovery_20260523/"
        "alpha_zoo_debounced_efficiency_repair_discovery_latest.json",
    ),
    (
        "thirty_m_plus_feedback",
        ALPHA_V2_ROOT / "alpha_zoo_30m_plus_alpha_feedback_discovery_20260523/"
        "alpha_zoo_30m_plus_alpha_feedback_discovery_latest.json",
    ),
    (
        "thirty_m_plus_booster",
        ALPHA_V2_ROOT / "alpha_zoo_30m_plus_alpha_booster_discovery_20260523/"
        "alpha_zoo_30m_plus_alpha_booster_discovery_latest.json",
    ),
    (
        "asset_diverse_strategy",
        ALPHA_V2_ROOT / "alpha_zoo_asset_diverse_strategy_discovery_20260523/"
        "alpha_zoo_asset_diverse_strategy_discovery_latest.json",
    ),
)

ASSET_GROUPS: dict[str, tuple[str, ...]] = {
    "crypto_major": ("BTCUSDT", "ETHUSDT"),
    "crypto_high_beta_alt": ("SOLUSDT", "AVAXUSDT", "DOGEUSDT", "TONUSDT", "ADAUSDT"),
    "crypto_payment_alt": ("TRXUSDT", "XRPUSDT"),
    "crypto_exchange_beta": ("BNBUSDT",),
    "precious_metal_proxy": BINANCE_TRADFI_PRECIOUS_METAL_SYMBOLS,
    "tradfi_energy_industrial_commodity": BINANCE_TRADFI_ENERGY_INDUSTRIAL_COMMODITY_SYMBOLS,
    "tradfi_etf_index": BINANCE_TRADFI_ETF_INDEX_SYMBOLS,
    "tradfi_equity": BINANCE_TRADFI_EQUITY_SYMBOLS,
    "tradfi_premarket": BINANCE_TRADFI_PREMARKET_SYMBOLS,
}

BLOCKED_REAL_MONEY_FLAG_KEYS = frozenset(
    {
        "ready_for_real",
        "real_money_execution",
        "real_execution_allowed",
    }
)

ROW_FIELDS = [
    "monitoring_rank",
    "symbol_monitoring_rank",
    "monitoring_status",
    "monitoring_action",
    "model_id",
    "source_artifact_kind",
    "source_label",
    "candidate_origin",
    "symbol",
    "asset_group",
    "timeframe",
    "family",
    "side",
    "leverage",
    "allocation_fraction",
    "notional_fraction",
    "train_return",
    "validation_return",
    "locked_oos_return",
    "train_mdd",
    "validation_mdd",
    "locked_oos_mdd",
    "train_trade_event_count",
    "validation_trade_event_count",
    "locked_oos_trade_event_count",
    "train_return_per_turnover_proxy_bps",
    "validation_return_per_turnover_proxy_bps",
    "locked_oos_return_per_turnover_proxy_bps",
    "train_validation_return_ratio",
    "locked_oos_liquidation_count",
    "locked_oos_account_wipeout_count",
    "paper_candidate_gate_pass",
    "primary_10bps_promotion_gate_pass",
    "execution_efficiency_proxy_gate_pass",
    "ready_for_paper",
    "ready_for_real",
    "real_money_execution",
    "replay_live_notional_parity",
    "monitoring_score_train_validation_only",
    "status_reasons",
    "rejection_reasons",
]

MATRIX_FIELDS = [
    "symbol",
    "asset_group",
    "source_artifact_kinds",
    "timeframes_observed",
    "families_observed",
    "total_candidate_rows",
    "paper_monitor_count",
    "shadow_watchlist_count",
    "coverage_blocked_shadow_count",
    "insufficient_candidate_evidence_count",
    "best_paper_model_id",
    "best_paper_family",
    "best_paper_timeframe",
    "best_paper_train_return",
    "best_paper_validation_return",
    "best_paper_locked_oos_return_report_only",
    "best_paper_train_rpt_bps",
    "best_paper_validation_rpt_bps",
    "best_paper_locked_oos_rpt_bps_report_only",
    "best_shadow_model_id",
    "best_shadow_family",
    "best_shadow_timeframe",
    "best_shadow_train_return",
    "best_shadow_validation_return",
    "best_shadow_locked_oos_return_report_only",
    "monitoring_action",
    "ready_for_real",
    "real_money_execution",
]


@dataclass(frozen=True)
class LoadedSource:
    label: str
    path: Path
    payload: Mapping[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "null"}
    return bool(value)


def _walk_blocked_real_money_flags(
    value: Any,
    *,
    location: str,
) -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key in BLOCKED_REAL_MONEY_FLAG_KEYS and _truthy(child):
                violations.append(f"{child_location}={child!r}")
            violations.extend(_walk_blocked_real_money_flags(child, location=child_location))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            violations.extend(_walk_blocked_real_money_flags(child, location=f"{location}[{idx}]"))
    return violations


def _assert_no_real_money_disabled(source: LoadedSource) -> None:
    violations = _walk_blocked_real_money_flags(source.payload, location=source.label)
    if violations:
        rendered = "; ".join(violations[:10])
        raise ValueError(
            f"real-money guard violation in {source.path}: {rendered}; "
            "monitoring slate must remain paper/testnet-only"
        )


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except TypeError, ValueError:
        return default


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except TypeError, ValueError:
        return default


def _split_float(row: Mapping[str, Any], split: str, suffix: str) -> float | None:
    direct = _as_float(row.get(f"{split}_{suffix}"))
    if direct is not None:
        return direct
    pct = _as_float(row.get(f"{split}_{suffix}_pct"))
    if pct is None:
        return None
    return pct / 100.0 if abs(pct) > 1.0 else pct


def _split_rpt(row: Mapping[str, Any], split: str) -> float | None:
    for key in (
        f"{split}_return_per_turnover_proxy_bps",
        f"{split}_return_per_turnover_bps",
        f"{split}_rpt_bps",
    ):
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


def _source_artifact_kind(source: LoadedSource) -> str:
    return str(source.payload.get("artifact_kind") or source.label)


def _asset_group_from_payloads(
    symbol: str,
    source_asset_groups: Sequence[Mapping[str, Any]],
) -> str:
    normalized = symbol.strip().upper()
    for groups in source_asset_groups:
        for group, members in groups.items():
            if normalized in {str(member).upper() for member in members or []}:
                return str(group)
    for group, members in ASSET_GROUPS.items():
        if normalized in members:
            return group
    return "other"


def _candidate_lists(payload: Mapping[str, Any]) -> list[tuple[str, list[Mapping[str, Any]]]]:
    lists: list[tuple[str, list[Mapping[str, Any]]]] = []
    handoff = payload.get("paper_testnet_handoff")
    if isinstance(handoff, Mapping):
        candidates = handoff.get("candidates")
        if isinstance(candidates, list):
            lists.append(
                ("paper_testnet_handoff", [row for row in candidates if isinstance(row, Mapping)])
            )
    top = payload.get("top_candidates")
    if isinstance(top, list):
        lists.append(("top_candidates", [row for row in top if isinstance(row, Mapping)]))
    shadow = payload.get("no_promotion_shadow_shortlist")
    if isinstance(shadow, Mapping):
        shadows = shadow.get("shadows")
        if isinstance(shadows, list):
            lists.append(
                (
                    "no_promotion_shadow_shortlist",
                    [row for row in shadows if isinstance(row, Mapping)],
                )
            )
    return lists


def _source_symbols(payload: Mapping[str, Any]) -> set[str]:
    symbols: set[str] = set()
    source_data = payload.get("source_data")
    if isinstance(source_data, Mapping):
        for key in ("symbols", "promotion_symbols", "shadow_symbols"):
            value = source_data.get(key)
            if isinstance(value, list):
                symbols.update(str(symbol).upper() for symbol in value)
        coverage = source_data.get("coverage_manifest")
        if isinstance(coverage, Mapping):
            symbols.update(str(symbol).upper() for symbol in coverage)
    asset_groups = payload.get("asset_groups")
    if isinstance(asset_groups, Mapping):
        for members in asset_groups.values():
            if isinstance(members, list):
                symbols.update(str(symbol).upper() for symbol in members)
    for _, rows in _candidate_lists(payload):
        for row in rows:
            symbol = row.get("symbol") or row.get("pair") or row.get("target_symbol")
            if symbol:
                symbols.add(str(symbol).upper())
    return symbols


def _monitoring_score(row: Mapping[str, Any]) -> float:
    """Train+validation-only priority; locked-OOS is deliberately ignored."""
    train = float(row.get("train_return") or 0.0)
    validation = float(row.get("validation_return") or 0.0)
    train_rpt = float(row.get("train_return_per_turnover_proxy_bps") or 0.0)
    validation_rpt = float(row.get("validation_return_per_turnover_proxy_bps") or 0.0)
    validation_mdd = float(row.get("validation_mdd") or 0.0)
    validation_trades = float(row.get("validation_trade_event_count") or 0.0)
    spike_penalty = max(0.0, validation - train)
    efficiency_bonus = min(train_rpt, validation_rpt, 80.0) / 150.0
    return (
        6.0 * validation
        + 2.0 * min(train, validation)
        + efficiency_bonus
        - 7.0 * spike_penalty
        - 2.0 * validation_mdd
        - 0.00012 * validation_trades
    )


def _is_locked_oos_coverage_blocked(reasons: Sequence[Any]) -> bool:
    rendered = " ".join(str(reason).lower() for reason in reasons)
    coverage_tokens = (
        "locked_oos_trade_event_count_0_below_20",
        "locked_oos_trade_event_count_none",
        "locked_oos_trade_event_count_missing",
        "locked_oos_bar_count_0",
        "locked_oos_no_bars",
        "no_locked_oos",
        "missing_locked_oos",
        "locked_oos_feature_coverage",
    )
    return any(token in rendered for token in coverage_tokens)


def _monitoring_status_and_action(row: Mapping[str, Any]) -> tuple[str, str, list[str]]:
    reasons = list(row.get("rejection_reasons") or [])
    if bool(row.get("paper_candidate_gate_pass")):
        return (
            "paper_testnet_monitor",
            "monitor_paper_testnet_with_realized_bbo_fill_liq_mdd_notional_parity",
            ["strict_paper_candidate_gate_pass"],
        )
    if _is_locked_oos_coverage_blocked(reasons):
        return (
            "coverage_blocked_shadow",
            "extend_locked_oos_data_coverage_before_any_paper_review",
            [str(reason) for reason in reasons],
        )
    if reasons:
        return (
            "shadow_watchlist_no_promotion",
            "shadow_monitor_research_only_no_allocation",
            [str(reason) for reason in reasons],
        )
    return (
        "insufficient_candidate_evidence",
        "record_source_coverage_without_candidate_allocation",
        ["no_paper_gate_and_no_specific_rejection_reason"],
    )


def _normalize_row(
    source: LoadedSource,
    row: Mapping[str, Any],
    *,
    candidate_origin: str,
    source_asset_groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    symbol = str(
        row.get("symbol") or row.get("pair") or row.get("target_symbol") or "UNKNOWN"
    ).upper()
    normalized: dict[str, Any] = {
        "model_id": str(
            row.get("model_id") or row.get("candidate_id") or f"{source.label}:{symbol}"
        ),
        "source_artifact_kind": _source_artifact_kind(source),
        "source_label": source.label,
        "source_artifact_path": str(source.path),
        "candidate_origin": candidate_origin,
        "source_rank": _as_int(row.get("rank")),
        "symbol": symbol,
        "asset_group": str(
            row.get("asset_group") or _asset_group_from_payloads(symbol, source_asset_groups)
        ),
        "timeframe": row.get("timeframe"),
        "family": row.get("family") or row.get("strategy_family"),
        "side": row.get("side"),
        "leverage": _as_float(row.get("leverage")),
        "allocation_fraction": _as_float(row.get("allocation_fraction")),
        "notional_fraction": _as_float(row.get("notional_fraction")),
        "train_return": _split_float(row, "train", "return"),
        "validation_return": _split_float(row, "validation", "return"),
        "locked_oos_return": _split_float(row, "locked_oos", "return"),
        "train_mdd": _split_float(row, "train", "mdd"),
        "validation_mdd": _split_float(row, "validation", "mdd"),
        "locked_oos_mdd": _split_float(row, "locked_oos", "mdd"),
        "train_trade_event_count": _as_int(row.get("train_trade_event_count")),
        "validation_trade_event_count": _as_int(row.get("validation_trade_event_count")),
        "locked_oos_trade_event_count": _as_int(row.get("locked_oos_trade_event_count")),
        "train_return_per_turnover_proxy_bps": _split_rpt(row, "train"),
        "validation_return_per_turnover_proxy_bps": _split_rpt(row, "validation"),
        "locked_oos_return_per_turnover_proxy_bps": _split_rpt(row, "locked_oos"),
        "locked_oos_liquidation_count": _as_int(row.get("locked_oos_liquidation_count"), 0),
        "locked_oos_account_wipeout_count": _as_int(row.get("locked_oos_account_wipeout_count"), 0),
        "paper_candidate_gate_pass": bool(row.get("paper_candidate_gate_pass")),
        "primary_10bps_promotion_gate_pass": bool(row.get("primary_10bps_promotion_gate_pass")),
        "execution_efficiency_proxy_gate_pass": bool(
            row.get("execution_efficiency_proxy_gate_pass")
        ),
        "ready_for_paper": bool(row.get("ready_for_paper")),
        "ready_for_real": False,
        "real_money_execution": False,
        "replay_live_notional_parity": row.get("replay_live_notional_parity"),
        "rejection_reasons": list(row.get("rejection_reasons") or []),
    }
    train = normalized["train_return"]
    validation = normalized["validation_return"]
    ratio = _as_float(row.get("train_validation_return_ratio"))
    if ratio is None and train is not None and validation not in (None, 0.0):
        ratio = train / validation
    normalized["train_validation_return_ratio"] = ratio
    status, action, status_reasons = _monitoring_status_and_action(normalized)
    normalized["monitoring_status"] = status
    normalized["monitoring_action"] = action
    normalized["status_reasons"] = status_reasons
    normalized["monitoring_score_train_validation_only"] = _monitoring_score(normalized)
    return normalized


def _candidate_origin_priority(origin: str) -> int:
    return {
        "paper_testnet_handoff": 3,
        "top_candidates": 2,
        "no_promotion_shadow_shortlist": 1,
    }.get(origin, 0)


def _merge_candidate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    origins: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("source_artifact_kind")), str(row.get("model_id")))
        origins[key].append(str(row.get("candidate_origin")))
        current = merged.get(key)
        if current is None:
            merged[key] = dict(row)
            continue
        if _candidate_origin_priority(
            str(row.get("candidate_origin"))
        ) > _candidate_origin_priority(str(current.get("candidate_origin"))):
            replacement = dict(row)
            replacement["source_origins_seen"] = sorted(set(origins[key]))
            merged[key] = replacement
        else:
            current["source_origins_seen"] = sorted(set(origins[key]))
            current["paper_candidate_gate_pass"] = bool(
                current.get("paper_candidate_gate_pass") or row.get("paper_candidate_gate_pass")
            )
    for key, row in merged.items():
        row["source_origins_seen"] = sorted(set(origins[key]))
    return list(merged.values())


def _rank_monitoring_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    status_priority = {
        "paper_testnet_monitor": 0,
        "shadow_watchlist_no_promotion": 1,
        "coverage_blocked_shadow": 2,
        "insufficient_candidate_evidence": 3,
    }
    ranked = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            status_priority.get(str(row.get("monitoring_status")), 99),
            str(row.get("asset_group")),
            str(row.get("symbol")),
            -float(row.get("monitoring_score_train_validation_only") or 0.0),
            str(row.get("model_id")),
        ),
    )
    symbol_counts: Counter[str] = Counter()
    for rank, row in enumerate(ranked, start=1):
        symbol = str(row.get("symbol"))
        symbol_counts[symbol] += 1
        row["monitoring_rank"] = rank
        row["symbol_monitoring_rank"] = symbol_counts[symbol]
    return ranked


def _best_by_score(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not rows:
        return None
    return max(
        rows, key=lambda row: float(row.get("monitoring_score_train_validation_only") or -1e18)
    )


def _matrix_row(symbol: str, rows: Sequence[Mapping[str, Any]], asset_group: str) -> dict[str, Any]:
    paper = [row for row in rows if row.get("monitoring_status") == "paper_testnet_monitor"]
    shadow = [
        row
        for row in rows
        if row.get("monitoring_status")
        in {"shadow_watchlist_no_promotion", "coverage_blocked_shadow"}
    ]
    best_paper = _best_by_score(paper)
    best_shadow = _best_by_score(shadow)
    status_counts = Counter(str(row.get("monitoring_status")) for row in rows)
    if paper:
        action = "monitor_all_paper_testnet_candidates_for_symbol"
    elif shadow:
        action = "shadow_monitor_or_extend_data_before_paper_review"
    else:
        action = "source_coverage_only_no_candidate_rows"

    def value(row: Mapping[str, Any] | None, key: str) -> Any:
        return None if row is None else row.get(key)

    return {
        "symbol": symbol,
        "asset_group": asset_group,
        "source_artifact_kinds": sorted({str(row.get("source_artifact_kind")) for row in rows}),
        "timeframes_observed": sorted(
            {str(row.get("timeframe")) for row in rows if row.get("timeframe")}
        ),
        "families_observed": sorted({str(row.get("family")) for row in rows if row.get("family")}),
        "total_candidate_rows": len(rows),
        "paper_monitor_count": status_counts["paper_testnet_monitor"],
        "shadow_watchlist_count": status_counts["shadow_watchlist_no_promotion"],
        "coverage_blocked_shadow_count": status_counts["coverage_blocked_shadow"],
        "insufficient_candidate_evidence_count": status_counts["insufficient_candidate_evidence"],
        "best_paper_model_id": value(best_paper, "model_id"),
        "best_paper_family": value(best_paper, "family"),
        "best_paper_timeframe": value(best_paper, "timeframe"),
        "best_paper_train_return": value(best_paper, "train_return"),
        "best_paper_validation_return": value(best_paper, "validation_return"),
        "best_paper_locked_oos_return_report_only": value(best_paper, "locked_oos_return"),
        "best_paper_train_rpt_bps": value(best_paper, "train_return_per_turnover_proxy_bps"),
        "best_paper_validation_rpt_bps": value(
            best_paper,
            "validation_return_per_turnover_proxy_bps",
        ),
        "best_paper_locked_oos_rpt_bps_report_only": value(
            best_paper,
            "locked_oos_return_per_turnover_proxy_bps",
        ),
        "best_shadow_model_id": value(best_shadow, "model_id"),
        "best_shadow_family": value(best_shadow, "family"),
        "best_shadow_timeframe": value(best_shadow, "timeframe"),
        "best_shadow_train_return": value(best_shadow, "train_return"),
        "best_shadow_validation_return": value(best_shadow, "validation_return"),
        "best_shadow_locked_oos_return_report_only": value(best_shadow, "locked_oos_return"),
        "monitoring_action": action,
        "ready_for_real": False,
        "real_money_execution": False,
    }


def _asset_monitoring_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    all_symbols: set[str],
    source_asset_groups: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_symbol: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_symbol[str(row.get("symbol"))].append(row)
    matrix: list[dict[str, Any]] = []
    for symbol in sorted(all_symbols | set(rows_by_symbol)):
        asset_group = _asset_group_from_payloads(symbol, source_asset_groups)
        matrix.append(_matrix_row(symbol, rows_by_symbol.get(symbol, []), asset_group))
    return matrix


def _summary(
    rows: Sequence[Mapping[str, Any]], matrix: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    status_counts = Counter(str(row.get("monitoring_status")) for row in rows)
    symbol_counts = Counter(str(row.get("symbol")) for row in rows)
    asset_group_counts = Counter(str(row.get("asset_group")) for row in rows)
    source_counts = Counter(str(row.get("source_artifact_kind")) for row in rows)
    paper = [row for row in rows if row.get("monitoring_status") == "paper_testnet_monitor"]
    return {
        "candidate_row_count": len(rows),
        "source_artifact_count": len(source_counts),
        "symbol_count": len(matrix),
        "symbols_with_candidate_rows": len(symbol_counts),
        "asset_group_count": len({row.get("asset_group") for row in matrix}),
        "monitoring_status_counts": dict(sorted(status_counts.items())),
        "symbol_candidate_counts": dict(sorted(symbol_counts.items())),
        "asset_group_candidate_counts": dict(sorted(asset_group_counts.items())),
        "source_artifact_candidate_counts": dict(sorted(source_counts.items())),
        "paper_monitor_candidate_count": len(paper),
        "paper_monitor_symbol_count": len({row.get("symbol") for row in paper}),
        "ready_for_paper": bool(paper),
        "ready_for_real": False,
        "real_money_execution": False,
    }


def _paper_monitoring_handoff(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    paper = [dict(row) for row in rows if row.get("monitoring_status") == "paper_testnet_monitor"]
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_asset_group: dict[str, list[str]] = defaultdict(list)
    by_family: dict[str, list[str]] = defaultdict(list)
    for row in paper:
        by_symbol[str(row.get("symbol"))].append(row)
        by_asset_group[str(row.get("asset_group"))].append(str(row.get("model_id")))
        by_family[str(row.get("family"))].append(str(row.get("model_id")))
    return {
        "handoff_kind": "multi_asset_paper_testnet_monitoring_slate",
        "status": "paper_testnet_candidates_available" if paper else "no_paper_candidates",
        "candidate_count": len(paper),
        "symbol_count": len(by_symbol),
        "asset_group_count": len(by_asset_group),
        "ready_for_paper": bool(paper),
        "ready_for_real": False,
        "real_money_execution": False,
        "paper_execution_allowed": bool(paper),
        "real_execution_allowed": False,
        "candidates": paper,
        "candidates_by_symbol": {symbol: rows for symbol, rows in sorted(by_symbol.items())},
        "candidate_model_ids_by_asset_group": {
            group: sorted(ids) for group, ids in sorted(by_asset_group.items())
        },
        "candidate_model_ids_by_family": {
            family: sorted(ids) for family, ids in sorted(by_family.items())
        },
        "preflight": {
            "required_mode": "paper_or_testnet_only",
            "ready_for_real": False,
            "real_money_execution": False,
            "monitor_all_symbols_together": True,
            "check_replay_live_notional_parity": True,
            "check_liquidation_account_wipeout": True,
            "check_liquidation_inclusive_mdd": True,
            "check_realized_all_in_cost_bps_mean_lte": feedback.PRIMARY_ROUND_TRIP_COST_BPS,
            "check_realized_all_in_cost_bps_p95_lte": 15.0,
        },
        "monitoring_contract": {
            "record_realized_fee_bps": True,
            "record_realized_slippage_bps": True,
            "record_all_in_round_trip_bps": True,
            "record_bbo_spread_bps_at_submit": True,
            "record_return_per_turnover_proxy_bps": True,
            "record_liquidation_inclusive_mdd": True,
            "record_account_wipeout": True,
            "record_symbol_asset_group_family": True,
            "primary_research_round_trip_cost_bps": feedback.PRIMARY_ROUND_TRIP_COST_BPS,
            "return_per_turnover_gate_bps": feedback.RETURN_PER_TURNOVER_THRESHOLD_BPS,
        },
    }


def _guard_payload(sources: Sequence[LoadedSource], row_count: int) -> dict[str, Any]:
    return {
        "guard_kind": "multi_asset_monitoring_no_real_money_guard",
        "status": "pass",
        "checked_source_count": len(sources),
        "checked_sources": [
            {
                "label": source.label,
                "path": str(source.path),
                "artifact_kind": _source_artifact_kind(source),
                "ready_for_real": False,
                "real_money_execution": False,
            }
            for source in sources
        ],
        "checked_candidate_row_count": row_count,
        "blocked_flag_keys": sorted(BLOCKED_REAL_MONEY_FLAG_KEYS),
        "ready_for_real": False,
        "real_money_execution": False,
        "paper_testnet_only": True,
    }


def build_payload_from_loaded_sources(
    sources: Sequence[LoadedSource],
    *,
    output_dir: Path,
    write_outputs: bool = True,
) -> dict[str, Any]:
    for source in sources:
        _assert_no_real_money_disabled(source)

    source_asset_groups = [
        payload.get("asset_groups")
        for payload in (source.payload for source in sources)
        if isinstance(payload.get("asset_groups"), Mapping)
    ]
    all_symbols: set[str] = set()
    raw_rows: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    for source in sources:
        all_symbols.update(_source_symbols(source.payload))
        per_source_rows = []
        for origin, candidates in _candidate_lists(source.payload):
            per_source_rows.extend(
                _normalize_row(
                    source,
                    row,
                    candidate_origin=origin,
                    source_asset_groups=source_asset_groups,
                )
                for row in candidates
            )
        raw_rows.extend(per_source_rows)
        source_summaries.append(
            {
                "label": source.label,
                "path": str(source.path),
                "artifact_kind": _source_artifact_kind(source),
                "candidate_rows_loaded": len(per_source_rows),
                "source_symbols": sorted(_source_symbols(source.payload)),
                "ready_for_real": False,
                "real_money_execution": False,
            }
        )

    rows = _rank_monitoring_rows(_merge_candidate_rows(raw_rows))
    matrix = _asset_monitoring_matrix(
        rows,
        all_symbols=all_symbols,
        source_asset_groups=source_asset_groups,
    )
    handoff = _paper_monitoring_handoff(rows)
    guard = _guard_payload(sources, len(rows))

    latest_json = output_dir / "multi_asset_monitoring_slate_latest.json"
    timestamped_json = output_dir / f"multi_asset_monitoring_slate_{_timestamp()}.json"
    latest_md = output_dir / "multi_asset_monitoring_slate_latest.md"
    rows_csv = output_dir / "multi_asset_monitoring_rows_latest.csv"
    matrix_csv = output_dir / "asset_monitoring_matrix_latest.csv"
    handoff_json = output_dir / "paper_monitoring_handoff_latest.json"
    handoff_md = output_dir / "paper_monitoring_handoff_latest.md"
    guard_json = output_dir / "no_real_money_guard_latest.json"
    generation_log = output_dir / "artifact_generation_validation_latest.log"

    payload: dict[str, Any] = {
        "artifact_kind": "alpha_zoo_multi_asset_monitoring_slate",
        "generated_at_utc": _utc_now_iso(),
        "ready_for_paper": bool(handoff["ready_for_paper"]),
        "ready_for_real": False,
        "real_money_execution": False,
        "paper_execution_allowed": bool(handoff["paper_execution_allowed"]),
        "real_execution_allowed": False,
        "paper_testnet_only": True,
        "research_primary_round_trip_cost_bps": feedback.PRIMARY_ROUND_TRIP_COST_BPS,
        "avg_bbo_spread_bps_assumption": feedback.AVG_BBO_SPREAD_BPS_ASSUMPTION,
        "bbo_spread_multiplier": feedback.BBO_SPREAD_MULTIPLIER,
        "return_per_turnover_threshold_bps": feedback.RETURN_PER_TURNOVER_THRESHOLD_BPS,
        "selection_policy": {
            "slate_source": "previously_frozen_discovery_artifacts_only",
            "monitoring_score_inputs": ["train", "validation"],
            "uses_locked_oos_for_discovery": False,
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
            "ranking_freeze_before_locked_oos_gate": True,
            "locked_oos_role": "gate_report_only_after_train_validation_candidate_freeze",
            "train_return_below_validation_return_is_promotion_reject": True,
            "monitor_all_symbols_together": True,
            "no_calendar_date_hack": True,
        },
        "baseline_lanes_preserved": feedback.BASELINE_LANES,
        "source_artifacts": source_summaries,
        "discovery_summary": _summary(rows, matrix),
        "asset_monitoring_matrix": matrix,
        "monitoring_rows": rows,
        "paper_monitoring_handoff": handoff,
        "no_real_money_guard": guard,
        "runner_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "output_paths": {
            "latest_json": str(latest_json),
            "timestamped_json": str(timestamped_json),
            "latest_markdown": str(latest_md),
            "multi_asset_monitoring_rows_csv": str(rows_csv),
            "asset_monitoring_matrix_csv": str(matrix_csv),
            "paper_monitoring_handoff_json": str(handoff_json),
            "paper_monitoring_handoff_markdown": str(handoff_md),
            "no_real_money_guard_json": str(guard_json),
            "artifact_generation_validation_log": str(generation_log),
        },
    }
    if write_outputs:
        _write_outputs(payload)
    return payload


def _format_pct(value: Any) -> str:
    numeric = _as_float(value)
    return "NA" if numeric is None else f"{numeric:.4%}"


def _format_bps(value: Any) -> str:
    numeric = _as_float(value)
    return "NA" if numeric is None else f"{numeric:.2f}"


def _markdown(payload: Mapping[str, Any]) -> str:
    summary = dict(payload.get("discovery_summary") or {})
    matrix = list(payload.get("asset_monitoring_matrix") or [])
    paper_rows = [
        row
        for row in payload.get("monitoring_rows", [])
        if row.get("monitoring_status") == "paper_testnet_monitor"
    ]
    lines = [
        "# Alpha Zoo multi-asset monitoring slate",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        "Combined paper/testnet-only monitoring view across all recent Alpha Zoo discovery",
        "artifacts. Locked-OOS remains gate/report-only; monitoring priority uses train+",
        "validation evidence only. Real-money execution is disabled.",
        "",
        "## Summary",
        "",
        f"- Candidate rows normalized: `{summary.get('candidate_row_count')}`",
        f"- Symbols covered by matrix: `{summary.get('symbol_count')}`",
        f"- Symbols with paper candidates: `{summary.get('paper_monitor_symbol_count')}`",
        f"- Paper/testnet monitor candidates: `{summary.get('paper_monitor_candidate_count')}`",
        f"- Status counts: `{summary.get('monitoring_status_counts')}`",
        "- `ready_for_real=false`",
        "- `real_money_execution=false`",
        f"- Runner peak RSS MiB: `{float(payload.get('runner_peak_rss_mib') or 0.0):.3f}`",
        "",
        "## Asset monitoring matrix",
        "",
        "| Symbol | Group | Paper | Shadow | Coverage blocked | Best paper | Best shadow | Action |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in matrix:
        lines.append(
            f"| {row.get('symbol')} | {row.get('asset_group')} | "
            f"{row.get('paper_monitor_count')} | {row.get('shadow_watchlist_count')} | "
            f"{row.get('coverage_blocked_shadow_count')} | "
            f"`{row.get('best_paper_model_id') or '-'}` | "
            f"`{row.get('best_shadow_model_id') or '-'}` | "
            f"{row.get('monitoring_action')} |"
        )
    lines.extend(
        [
            "",
            "## Paper/testnet candidates by monitoring rank",
            "",
            "| Rank | Symbol | Group | TF | Family | Train | Val | OOS report | RPT train/val/OOS |",
            "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    if not paper_rows:
        lines.append("| - | - | - | - | - | - | - | - |")
    for row in paper_rows[:80]:
        rpt = "/".join(
            [
                _format_bps(row.get("train_return_per_turnover_proxy_bps")),
                _format_bps(row.get("validation_return_per_turnover_proxy_bps")),
                _format_bps(row.get("locked_oos_return_per_turnover_proxy_bps")),
            ]
        )
        lines.append(
            f"| {row.get('monitoring_rank')} | {row.get('symbol')} | "
            f"{row.get('asset_group')} | {row.get('timeframe')} | {row.get('family')} | "
            f"{_format_pct(row.get('train_return'))} | {_format_pct(row.get('validation_return'))} | "
            f"{_format_pct(row.get('locked_oos_return'))} | {rpt} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Paper/testnet-only monitoring handoff; real-money execution remains prohibited.",
            "- Replay/live notional parity, realized BBO spread, realized fee/slippage/all-in",
            "  round-trip cost, liquidation-inclusive MDD, and account wipeout must be",
            "  recorded for every monitored symbol and candidate.",
            "- Existing four `quality_single_pair` baseline lanes are preserved unchanged.",
        ]
    )
    return "\n".join(lines) + "\n"


def _handoff_markdown(handoff: Mapping[str, Any]) -> str:
    by_symbol = dict(handoff.get("candidates_by_symbol") or {})
    lines = [
        "# Paper/testnet handoff — multi-asset monitoring slate",
        "",
        f"- Status: `{handoff.get('status')}`",
        f"- Candidate count: `{handoff.get('candidate_count')}`",
        f"- Symbol count: `{handoff.get('symbol_count')}`",
        "- `ready_for_real=false`",
        "- `real_money_execution=false`",
        "- Monitor all paper symbols together; do not cherry-pick only one or two lanes.",
        "",
    ]
    for symbol, rows in sorted(by_symbol.items()):
        lines.extend(
            [
                f"## {symbol}",
                "",
                "| Rank | Model | TF | Family | Train | Val | OOS report | RPT train/val/OOS |",
                "| ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in rows[:20]:
            rpt = "/".join(
                [
                    _format_bps(row.get("train_return_per_turnover_proxy_bps")),
                    _format_bps(row.get("validation_return_per_turnover_proxy_bps")),
                    _format_bps(row.get("locked_oos_return_per_turnover_proxy_bps")),
                ]
            )
            lines.append(
                f"| {row.get('monitoring_rank')} | `{row.get('model_id')}` | "
                f"{row.get('timeframe')} | {row.get('family')} | "
                f"{_format_pct(row.get('train_return'))} | "
                f"{_format_pct(row.get('validation_return'))} | "
                f"{_format_pct(row.get('locked_oos_return'))} | {rpt} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_outputs(payload: Mapping[str, Any]) -> None:
    paths = dict(payload["output_paths"])
    feedback._write_json(Path(paths["latest_json"]), payload)
    feedback._write_json(Path(paths["timestamped_json"]), payload)
    Path(paths["latest_markdown"]).write_text(_markdown(payload), encoding="utf-8")
    feedback._write_csv(
        Path(paths["multi_asset_monitoring_rows_csv"]),
        list(payload.get("monitoring_rows") or []),
        ROW_FIELDS,
    )
    feedback._write_csv(
        Path(paths["asset_monitoring_matrix_csv"]),
        list(payload.get("asset_monitoring_matrix") or []),
        MATRIX_FIELDS,
    )
    feedback._write_json(
        Path(paths["paper_monitoring_handoff_json"]),
        payload.get("paper_monitoring_handoff") or {},
    )
    Path(paths["paper_monitoring_handoff_markdown"]).write_text(
        _handoff_markdown(payload.get("paper_monitoring_handoff") or {}),
        encoding="utf-8",
    )
    feedback._write_json(
        Path(paths["no_real_money_guard_json"]), payload.get("no_real_money_guard") or {}
    )
    Path(paths["artifact_generation_validation_log"]).write_text(
        "\n".join(
            [
                f"generated_at_utc={payload['generated_at_utc']}",
                f"artifact_kind={payload['artifact_kind']}",
                f"candidate_row_count={payload['discovery_summary']['candidate_row_count']}",
                f"symbol_count={payload['discovery_summary']['symbol_count']}",
                f"paper_monitor_candidate_count={payload['discovery_summary']['paper_monitor_candidate_count']}",
                "uses_locked_oos_for_discovery=false",
                "uses_locked_oos_for_selection=false",
                "uses_locked_oos_for_objective=false",
                "uses_locked_oos_for_pruning=false",
                "uses_locked_oos_for_parameter_fitting=false",
                "monitor_all_symbols_together=true",
                "ready_for_real=false",
                "real_money_execution=false",
                f"runner_peak_rss_mib={float(payload.get('runner_peak_rss_mib') or 0.0):.3f}",
                f"latest_json={paths['latest_json']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _source_specs_from_args(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if not args.source_json:
        return list(DEFAULT_SOURCE_ARTIFACTS)
    specs: list[tuple[str, Path]] = []
    for idx, source in enumerate(args.source_json, start=1):
        path = Path(source).expanduser().resolve()
        specs.append((f"source_{idx}_{path.stem}", path))
    return specs


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    sources = [
        LoadedSource(label=label, path=path.expanduser().resolve(), payload=_load_json(path))
        for label, path in _source_specs_from_args(args)
    ]
    return build_payload_from_loaded_sources(
        sources,
        output_dir=output_dir,
        write_outputs=not bool(args.no_write),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--source-json",
        action="append",
        default=None,
        help="Override source artifact JSON path. May be passed multiple times.",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="Build payload without writing artifacts."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(args)
    summary = payload["discovery_summary"]
    print(
        json.dumps(
            {
                "artifact": payload["output_paths"]["latest_json"],
                "candidate_row_count": summary["candidate_row_count"],
                "symbol_count": summary["symbol_count"],
                "paper_monitor_candidate_count": summary["paper_monitor_candidate_count"],
                "paper_monitor_symbol_count": summary["paper_monitor_symbol_count"],
                "ready_for_real": False,
                "real_money_execution": False,
                "runner_peak_rss_mib": payload["runner_peak_rss_mib"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
