"""Build a leakage-safe exact-window sleeve freeze artifact for portfolio tuning."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumina_quant.portfolio_split_contract import (
    FOLLOWUP_ROOT,
    OOS_START,
    REPORT_ROOT,
    TRAIN_END_EXCLUSIVE,
    VAL_END_EXCLUSIVE,
    VAL_START,
    memory_policy_payload,
    split_windows,
)

ONE_SHOT_BASELINE_PATH = (
    Path(__file__).resolve().parents[2] / "reports" / "portfolio_optimization_latest.json"
)
EQUAL_WEIGHT_BASELINE_PATH = FOLLOWUP_ROOT / "committee_portfolio_followup_latest.json"
PAIR_REVIEW_NOT_BEFORE = "2026-03-31"

DEFAULT_SLEEVE_SOURCES: dict[str, dict[str, Any]] = {
    "composite_trend_30m": {
        "path": REPORT_ROOT
        / "expansion_crypto_15m_30m_1h_20260310T115853Z"
        / "15m-30m-1h"
        / "exact_window_candidate_details_20260310T120626Z.json",
        "strategy_class": "CompositeTrendStrategy",
        "timeframe": "30m",
    },
    "topcap_tsmom_1h": {
        "path": REPORT_ROOT
        / "topcap_crypto_1h_focus_20260310T123813Z"
        / "1h"
        / "exact_window_candidate_details_20260310T124047Z.json",
        "strategy_class": "TopCapTimeSeriesMomentumStrategy",
        "timeframe": "1h",
    },
    "regime_breakout_1h": {
        "path": REPORT_ROOT
        / "topcap_crypto_1h_focus_20260310T123813Z"
        / "1h"
        / "exact_window_candidate_details_20260310T124047Z.json",
        "strategy_class": "RegimeBreakoutCandidateStrategy",
        "timeframe": "1h",
    },
    "rolling_breakout_30m": {
        "path": REPORT_ROOT
        / "expansion_crypto_15m_30m_1h_20260310T115853Z"
        / "15m-30m-1h"
        / "exact_window_candidate_details_20260310T120626Z.json",
        "strategy_class": "RollingBreakoutStrategy",
        "timeframe": "30m",
    },
}

ROLLING_GATE_PATH = FOLLOWUP_ROOT / "rolling_breakout_30m_gate_latest.json"
PAIR_RETUNE_PATH = FOLLOWUP_ROOT / "pair_spread_4h_xpt_xpd_retune_latest.json"
INCUMBENT_BUNDLE_PATH = FOLLOWUP_ROOT / "portfolio_one_shot_incumbent_bundle_latest.json"
FREEZE_MANIFEST_PATH = FOLLOWUP_ROOT / "portfolio_exact_window_freeze_manifest_latest.json"

DEFAULT_SELECTION_MODE = "validation_only"
INCUMBENT_ANCHORED_SELECTION_MODE = "incumbent_anchor_rolling_gate"

DENYLIST_FIELDS = {
    "committee",
    "candidate_pool_eligible",
    "promoted",
    "hurdle_fields",
    "btc_beating_candidate",
    "oos_btc_hurdle_pass",
    "oos_monthly_hurdle",
    "recent_three_month_two_pct_pass",
    "recent_three_months",
    "three_month_two_pct_candidate",
    "validation_btc_hurdle_pass",
    "validation_hurdle_pass",
    "validation_monthly_hurdle",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str) and value.strip():
        token = value.strip()
        try:
            numeric = float(token)
        except ValueError:
            normalized = token.replace("Z", "+00:00") if token.endswith("Z") else token
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
    else:
        return None

    magnitude = abs(numeric)
    if magnitude >= 1e15:
        return datetime.fromtimestamp(numeric / 1_000_000.0, tz=UTC)
    if magnitude >= 1e12:
        return datetime.fromtimestamp(numeric / 1_000.0, tz=UTC)
    if magnitude >= 1e9:
        return datetime.fromtimestamp(numeric, tz=UTC)
    return None


def _isoformat_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _normalize_stream(stream: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, raw_point in enumerate(list(stream or [])):
        raw_ts = raw_point.get("datetime", raw_point.get("t", raw_point.get("timestamp")))
        dt = _canonical_timestamp(raw_ts)
        if dt is not None:
            ts = _isoformat_utc(dt)
            normalized.append(
                {
                    "token": f"dt:{ts}",
                    "sort_key": (0, float(dt.timestamp()), float(idx)),
                    "t": ts,
                    "datetime": ts,
                    "v": _safe_float(raw_point.get("v"), 0.0),
                }
            )
            continue
        normalized.append(
            {
                "token": f"seq:{idx}",
                "sort_key": (1, float(idx), float(idx)),
                "t": float(idx),
                "datetime": None,
                "v": _safe_float(raw_point.get("v"), 0.0),
            }
        )
    normalized.sort(key=lambda item: item["sort_key"])
    return normalized


def _aggregate_stream(stream: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for point in _normalize_stream(stream):
        token = str(point["token"])
        if token not in aggregated:
            aggregated[token] = {
                "token": token,
                "sort_key": point["sort_key"],
                "t": point["t"],
                "datetime": point.get("datetime"),
                "v": 0.0,
            }
        aggregated[token]["v"] += float(point["v"])

    rows: list[dict[str, Any]] = []
    for point in sorted(aggregated.values(), key=lambda item: item["sort_key"]):
        row = {"t": point["t"], "v": float(point["v"])}
        if point.get("datetime"):
            row["datetime"] = point["datetime"]
        rows.append(row)
    return rows


def _weighted_stream(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for row in rows:
        weight = _safe_float(row.get("_portfolio_weight"), 0.0)
        if weight <= 0.0:
            continue
        for point in _normalize_stream(list(((row.get("return_streams") or {}).get(split)) or [])):
            token = str(point["token"])
            if token not in aggregated:
                aggregated[token] = {
                    "token": token,
                    "sort_key": point["sort_key"],
                    "t": point["t"],
                    "datetime": point.get("datetime"),
                    "v": 0.0,
                }
            aggregated[token]["v"] += weight * _safe_float(point.get("v"), 0.0)

    out: list[dict[str, Any]] = []
    for point in sorted(aggregated.values(), key=lambda item: item["sort_key"]):
        row = {"t": point["t"], "v": float(point["v"])}
        if point.get("datetime"):
            row["datetime"] = point["datetime"]
        out.append(row)
    return out


def _split_window_summary(stream: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = _normalize_stream(stream)
    timestamps = [point.get("datetime") for point in normalized if point.get("datetime")]
    return {
        "point_count": len(normalized),
        "start": timestamps[0] if timestamps else None,
        "end": timestamps[-1] if timestamps else None,
    }


def _strict_reslice_streams(
    streams: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    train_end = _canonical_timestamp(TRAIN_END_EXCLUSIVE)
    val_start = _canonical_timestamp(VAL_START)
    val_end = _canonical_timestamp(VAL_END_EXCLUSIVE)
    oos_start = _canonical_timestamp(OOS_START)
    if train_end is None or val_start is None or val_end is None or oos_start is None:
        return (
            {
                split: _aggregate_stream(list(streams.get(split) or []))
                for split in ("train", "val", "oos")
            },
            [],
        )

    bucket: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "oos": [],
    }
    adjustments: list[str] = []
    for original_split in ("train", "val", "oos"):
        for point in _aggregate_stream(list(streams.get(original_split) or [])):
            dt = _canonical_timestamp(point.get("datetime", point.get("t")))
            if dt is None:
                continue
            if dt < train_end:
                target_split = "train"
            elif val_start <= dt < val_end:
                target_split = "val"
            elif dt >= oos_start:
                target_split = "oos"
            else:
                continue
            if target_split != original_split:
                adjustments.append(
                    f"moved {original_split} point at {point.get('datetime')} into {target_split} to enforce strict windows"
                )
            bucket[target_split].append(point)
    return {split: _aggregate_stream(points) for split, points in bucket.items()}, adjustments


def _canonical_row_id(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or row.get("name") or "")


def _freeze_score(row: dict[str, Any]) -> float:
    val = dict(row.get("val") or {})
    return float(
        (3.0 * _safe_float(val.get("sharpe"), 0.0))
        + (2.0 * _safe_float(val.get("deflated_sharpe"), 0.0))
        + (25.0 * _safe_float(val.get("return"), 0.0))
        - (2.5 * _safe_float(val.get("pbo"), 1.0))
        - (0.25 * _safe_float(val.get("turnover"), 0.0))
    )


def _secondary_score(row: dict[str, Any]) -> float:
    train = dict(row.get("train") or {})
    return float(
        _safe_float(train.get("deflated_sharpe"), 0.0)
        + (0.5 * _safe_float(train.get("sharpe"), 0.0))
    )


def _anchor_score(row: dict[str, Any]) -> float:
    train = dict(row.get("train") or {})
    val = dict(row.get("val") or {})
    return float(
        (3.0 * _safe_float(val.get("sharpe"), 0.0))
        + (2.0 * _safe_float(val.get("deflated_sharpe"), 0.0))
        + (18.0 * _safe_float(val.get("return"), 0.0))
        - (2.5 * _safe_float(val.get("pbo"), 1.0))
        - (0.15 * _safe_float(val.get("turnover"), 0.0))
        + (0.5 * _safe_float(train.get("deflated_sharpe"), 0.0))
        + (0.25 * _safe_float(train.get("sharpe"), 0.0))
    )


def _strategy_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("strategy_class") or "").strip(),
        str(row.get("strategy_timeframe") or row.get("timeframe") or "").strip(),
    )


def _ranking_key(
    row: dict[str, Any],
) -> tuple[float, float, float, float, float, float, float, str]:
    val = dict(row.get("val") or {})
    return (
        -_freeze_score(row),
        -_secondary_score(row),
        _safe_float(val.get("pbo"), 1.0),
        -_safe_float(val.get("deflated_sharpe"), 0.0),
        -_safe_float(val.get("sharpe"), 0.0),
        -_safe_float(val.get("return"), 0.0),
        _safe_float(val.get("turnover"), 0.0),
        _canonical_row_id(row),
    )


def _trim_selected_row(
    row: dict[str, Any],
    *,
    selection_basis: str = DEFAULT_SELECTION_MODE,
) -> dict[str, Any]:
    allowed = {
        "candidate_id",
        "name",
        "family",
        "strategy_class",
        "strategy_timeframe",
        "timeframe",
        "symbols",
        "params",
        "pass",
        "metadata",
        "train",
        "val",
        "oos",
        "return_streams",
        "source_details_path",
        "source_summary_path",
        "portfolio_weight",
        "_portfolio_weight",
        "incumbent_rank",
    }
    trimmed = {key: value for key, value in row.items() if key in allowed}
    trimmed.pop("committee", None)
    trimmed.pop("candidate_pool_eligible", None)
    trimmed.pop("promoted", None)
    trimmed.pop("hurdle_fields", None)
    trimmed["pass"] = bool(trimmed.get("pass", True))
    if "timeframe" not in trimmed and "strategy_timeframe" in trimmed:
        trimmed["timeframe"] = trimmed["strategy_timeframe"]
    trimmed["selection_basis"] = selection_basis
    trimmed["freeze_score"] = _freeze_score(row)
    trimmed["freeze_secondary_score"] = _secondary_score(row)
    trimmed["anchor_score"] = _anchor_score(row)
    trimmed["selection_inputs"] = {
        "candidate_id": row.get("candidate_id"),
        "name": row.get("name"),
        "train": {
            "sharpe": _safe_float(dict(row.get("train") or {}).get("sharpe"), 0.0),
            "deflated_sharpe": _safe_float(
                dict(row.get("train") or {}).get("deflated_sharpe"), 0.0
            ),
            "return": _safe_float(dict(row.get("train") or {}).get("return"), 0.0),
        },
        "val": {
            "sharpe": _safe_float(dict(row.get("val") or {}).get("sharpe"), 0.0),
            "deflated_sharpe": _safe_float(dict(row.get("val") or {}).get("deflated_sharpe"), 0.0),
            "return": _safe_float(dict(row.get("val") or {}).get("return"), 0.0),
            "pbo": _safe_float(dict(row.get("val") or {}).get("pbo"), 1.0),
            "turnover": _safe_float(dict(row.get("val") or {}).get("turnover"), 0.0),
        },
    }
    trimmed["present_denied_fields_in_source"] = sorted(
        field for field in DENYLIST_FIELDS if field in row
    )
    for key in (
        "anchor_decision",
        "anchor_decision_reason",
        "source_of_truth",
        "rolling_gate_status",
        "rolling_gate_selection_basis",
    ):
        if key in row:
            trimmed[key] = row[key]
    return trimmed


def _load_default_grouped_rows(
    report_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    manifest: dict[str, str] = {}
    for sleeve, source in DEFAULT_SLEEVE_SOURCES.items():
        path = Path(source["path"])
        payload = _load_json(path)
        grouped[sleeve] = [
            dict(row)
            for row in list(payload or [])
            if str(row.get("strategy_class") or "") == str(source["strategy_class"])
            and str(row.get("strategy_timeframe") or row.get("timeframe") or "")
            == str(source["timeframe"])
        ]
        manifest[sleeve] = str(path.resolve())
    return grouped, manifest


def _coerce_grouped_rows(
    *,
    report_root: Path,
    grouped_rows: dict[str, list[dict[str, Any]]] | None,
    grouped_candidate_rows: dict[str, list[dict[str, Any]]] | None,
    candidate_groups: dict[str, list[dict[str, Any]]] | None,
    rows_by_sleeve: dict[str, list[dict[str, Any]]] | None,
    sleeve_rows: dict[str, list[dict[str, Any]]] | None,
    sleeve_groups: dict[str, list[dict[str, Any]]] | None,
    source_payloads: dict[str, list[dict[str, Any]]] | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    for candidate in (
        grouped_rows,
        grouped_candidate_rows,
        candidate_groups,
        rows_by_sleeve,
        sleeve_rows,
        sleeve_groups,
        source_payloads,
    ):
        if isinstance(candidate, dict) and candidate:
            return {
                str(key): [dict(row) for row in list(value or [])]
                for key, value in candidate.items()
            }, {}
    return _load_default_grouped_rows(report_root)


def _coerce_pair_payload(
    *,
    pair_payload: dict[str, Any] | None,
    pair_spread_payload: dict[str, Any] | None,
    pairspread_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    for payload in (pair_payload, pair_spread_payload, pairspread_payload):
        if isinstance(payload, dict) and payload:
            return dict(payload)
    if PAIR_RETUNE_PATH.exists():
        return dict(_load_json(PAIR_RETUNE_PATH))
    return {}


def _coerce_rolling_gate_payload(
    *,
    rolling_gate_payload: dict[str, Any] | None,
    rolling_breakout_gate_payload: dict[str, Any] | None,
    gate_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    for payload in (rolling_gate_payload, rolling_breakout_gate_payload, gate_payload):
        if isinstance(payload, dict) and payload:
            return dict(payload)
    if ROLLING_GATE_PATH.exists():
        return dict(_load_json(ROLLING_GATE_PATH))
    return {}


def _load_grouped_rows_from_manifest(
    manifest_path: Path,
    *,
    report_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    if not manifest_path.exists():
        return _load_default_grouped_rows(report_root)

    manifest_payload = dict(_load_json(manifest_path))
    source_manifest = {
        str(key): str(value)
        for key, value in dict(manifest_payload.get("source_manifest") or {}).items()
        if str(value).strip()
    }
    if not source_manifest:
        return _load_default_grouped_rows(report_root)

    grouped: dict[str, list[dict[str, Any]]] = {}
    resolved_manifest: dict[str, str] = {}
    for sleeve, raw_path in source_manifest.items():
        path = Path(raw_path)
        payload = _load_json(path)
        source_spec = DEFAULT_SLEEVE_SOURCES.get(sleeve, {})
        strategy_class = str(source_spec.get("strategy_class") or "")
        timeframe = str(source_spec.get("timeframe") or "")
        rows = [dict(row) for row in list(payload or []) if isinstance(row, dict)]
        if strategy_class or timeframe:
            rows = [
                row
                for row in rows
                if (not strategy_class or str(row.get("strategy_class") or "") == strategy_class)
                and (
                    not timeframe
                    or str(row.get("strategy_timeframe") or row.get("timeframe") or "") == timeframe
                )
            ]
        grouped[sleeve] = rows
        resolved_manifest[sleeve] = str(path.resolve())
    return grouped, resolved_manifest


def _coerce_incumbent_bundle_payload(
    *,
    incumbent_bundle_payload: dict[str, Any] | None,
    anchor_bundle_payload: dict[str, Any] | None,
    incumbent_payload: dict[str, Any] | None,
    incumbent_bundle_path: Path | str,
    anchor_bundle_path: Path | str | None = None,
) -> dict[str, Any]:
    for payload in (incumbent_bundle_payload, anchor_bundle_payload, incumbent_payload):
        if isinstance(payload, dict) and payload:
            return dict(payload)
    for raw_path in (incumbent_bundle_path, anchor_bundle_path):
        if raw_path is None:
            continue
        path = Path(raw_path)
        if path.exists():
            return dict(_load_json(path))
    return {}


def _rolling_gate_survives(gate_payload: dict[str, Any]) -> bool:
    if not gate_payload:
        return False
    if "survives_train_val" in gate_payload:
        return bool(gate_payload.get("survives_train_val"))
    selected_rule = dict(gate_payload.get("selected_rule") or {})
    if "survives_train_val" in selected_rule:
        return bool(selected_rule.get("survives_train_val"))
    return bool(gate_payload.get("survives"))


def _rolling_gate_selection_basis(gate_payload: dict[str, Any]) -> str:
    if not gate_payload:
        return ""
    if gate_payload.get("selection_basis"):
        return str(gate_payload.get("selection_basis"))
    selected_rule = dict(gate_payload.get("selected_rule") or {})
    return str(selected_rule.get("selection_basis") or "")


def _match_candidate_row(
    rows: list[dict[str, Any]],
    template: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not template:
        return None
    target_id = _canonical_row_id(template)
    target_name = str(template.get("name") or "").strip()
    for row in rows:
        row_id = _canonical_row_id(row)
        row_name = str(row.get("name") or "").strip()
        if target_id and row_id == target_id:
            return dict(row)
        if target_name and row_name == target_name:
            return dict(row)
    return None


def _challenger_clears_anchor(
    anchor: dict[str, Any], challenger: dict[str, Any]
) -> tuple[bool, dict[str, float]]:
    anchor_val = dict(anchor.get("val") or {})
    anchor_train = dict(anchor.get("train") or {})
    challenger_val = dict(challenger.get("val") or {})
    challenger_train = dict(challenger.get("train") or {})
    delta = _anchor_score(challenger) - _anchor_score(anchor)
    diagnostics = {
        "score_delta": float(delta),
        "candidate_anchor_score": _anchor_score(challenger),
        "incumbent_anchor_score": _anchor_score(anchor),
        "candidate_val_pbo": _safe_float(challenger_val.get("pbo"), 1.0),
        "candidate_val_deflated_sharpe": _safe_float(challenger_val.get("deflated_sharpe"), 0.0),
        "candidate_train_return": _safe_float(challenger_train.get("return"), 0.0),
    }
    return (
        delta >= 0.25
        and _safe_float(challenger_val.get("pbo"), 1.0)
        <= min(0.45, _safe_float(anchor_val.get("pbo"), 1.0) + 0.05)
        and _safe_float(challenger_val.get("deflated_sharpe"), 0.0)
        >= (_safe_float(anchor_val.get("deflated_sharpe"), 0.0) - 0.02)
        and _safe_float(challenger_train.get("return"), 0.0)
        >= (_safe_float(anchor_train.get("return"), 0.0) - 0.02),
        diagnostics,
    )


def _select_anchor_or_local_challenger(
    anchor_row: dict[str, Any],
    sleeve_candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    anchor = dict(anchor_row)
    best_choice = dict(anchor)
    best_decision = {
        "decision": "keep_anchor",
        "reason": "no challenger cleared incumbent-aware thresholds",
        "score_delta": 0.0,
    }
    promotable: list[tuple[dict[str, Any], dict[str, float]]] = []
    for challenger in sleeve_candidates:
        challenger_row = dict(challenger)
        if _canonical_row_id(challenger_row) == _canonical_row_id(anchor):
            continue
        passes, diagnostics = _challenger_clears_anchor(anchor, challenger_row)
        if passes:
            promotable.append((challenger_row, diagnostics))

    if promotable:
        best_choice, best_diag = max(
            promotable,
            key=lambda item: (
                float(item[1].get("score_delta", 0.0)),
                _anchor_score(item[0]),
                -_safe_float(dict(item[0].get("val") or {}).get("pbo"), 1.0),
            ),
        )
        best_decision = {
            "decision": "promote_local_challenger",
            "reason": "local challenger cleared incumbent-aware thresholds",
            **best_diag,
        }

    selected = dict(best_choice)
    selected["anchor_decision"] = best_decision["decision"]
    selected["anchor_decision_reason"] = best_decision["reason"]
    selected["source_of_truth"] = (
        "incumbent_bundle_anchor"
        if best_decision["decision"] == "keep_anchor"
        else "exact_window_candidate_details_local_challenger"
    )
    return selected, best_decision


def _select_rolling_candidate(
    sleeve_candidates: list[dict[str, Any]],
    gate_payload: dict[str, Any],
) -> dict[str, Any]:
    gated = dict(gate_payload.get("gated_candidate_row") or {})
    base_row = _match_candidate_row(sleeve_candidates, gated)
    if base_row is None and gated:
        base_row = dict(gated)
    if base_row is None and sleeve_candidates:
        base_row = dict(min(sleeve_candidates, key=_ranking_key))
    if base_row is None:
        raise RuntimeError("rolling gate survived but no rolling candidate row was available")
    supplemented = _supplement_rolling_gate(base_row, gate_payload)
    supplemented["anchor_decision"] = "rolling_gate_admitted"
    supplemented["anchor_decision_reason"] = (
        "rolling breakout admitted only after train+val gate survival"
    )
    supplemented["source_of_truth"] = "rolling_breakout_train_val_gate"
    supplemented["rolling_gate_status"] = "admitted"
    supplemented["rolling_gate_selection_basis"] = _rolling_gate_selection_basis(gate_payload)
    return supplemented


def _ranking_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted([dict(row) for row in rows], key=_ranking_key)
    table: list[dict[str, Any]] = []
    for row in ranked:
        val = dict(row.get("val") or {})
        table.append(
            {
                "candidate_id": row.get("candidate_id"),
                "name": row.get("name"),
                "freeze_score": _freeze_score(row),
                "secondary_score": _secondary_score(row),
                "val_sharpe": _safe_float(val.get("sharpe"), 0.0),
                "val_deflated_sharpe": _safe_float(val.get("deflated_sharpe"), 0.0),
                "val_return": _safe_float(val.get("return"), 0.0),
                "val_pbo": _safe_float(val.get("pbo"), 1.0),
                "val_turnover": _safe_float(val.get("turnover"), 0.0),
            }
        )
    return table


def _supplement_rolling_gate(
    selected_row: dict[str, Any], gate_payload: dict[str, Any]
) -> dict[str, Any]:
    if not gate_payload or not _rolling_gate_survives(gate_payload):
        return selected_row
    gated = dict(gate_payload.get("gated_candidate_row") or {})
    if not gated:
        return selected_row
    selected_id = _canonical_row_id(selected_row)
    gated_id = _canonical_row_id(gated)
    if selected_id and gated_id and selected_id != gated_id:
        return selected_row

    supplemented = dict(selected_row)
    split_adjustments: list[str] = []
    for split in ("train", "val", "oos"):
        if split in gated and isinstance(gated.get(split), dict):
            supplemented[split] = dict(gated.get(split) or {})
    if isinstance(gated.get("return_streams"), dict):
        gated_streams, split_adjustments = _strict_reslice_streams(
            dict(gated.get("return_streams") or {})
        )
        supplemented["return_streams"] = gated_streams
    metadata = dict(supplemented.get("metadata") or {})
    gated_metadata = dict(gated.get("metadata") or {})
    for key in (
        "selection_basis",
        "activation_rule_id",
        "activation_rule_survives",
        "activation_rule_survives_train_val",
        "activation_rule_conditions",
        "activation_rule_label",
        "activation_signal_lag_days",
    ):
        if key in gated_metadata:
            metadata[key] = gated_metadata[key]
    if split_adjustments:
        metadata["activation_split_adjustments"] = list(split_adjustments)
    supplemented["metadata"] = metadata
    return supplemented


def _summary_from_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = dict(_load_json(path))
    return {
        "path": str(path.resolve()),
        "generated_at": payload.get("generated_at"),
        "portfolio_metrics": dict(payload.get("portfolio_metrics") or payload.get("metrics") or {}),
        "weight_count": len(list(payload.get("weights") or [])),
    }


def _equal_weight_baseline_support(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = dict(_load_json(path))
    selection_rows = [
        dict(row) for row in list(payload.get("selection") or []) if isinstance(row, dict)
    ]
    component_count = len(selection_rows)
    if component_count > 0:
        equal_weight = 1.0 / float(component_count)
        for row in selection_rows:
            row["_portfolio_weight"] = equal_weight
    combined_streams = (
        {split: _weighted_stream(selection_rows, split) for split in ("train", "val", "oos")}
        if selection_rows
        else {}
    )
    has_streams = any(
        bool(list((combined_streams.get(split)) or [])) for split in ("train", "val", "oos")
    )
    normalization_method = (
        "rebuilt_from_component_streams_default"
        if selection_rows and has_streams
        else "summary_only_missing_component_streams"
    )
    return {
        "path": str(path.resolve()),
        "generated_at": payload.get("generated_at"),
        "normalization_method": normalization_method,
        "component_count": component_count,
        "component_names": [row.get("name") for row in selection_rows],
        "portfolio_metrics": dict(payload.get("portfolio_metrics") or payload.get("metrics") or {}),
        "combined_streams": combined_streams if has_streams else {},
        "split_windows": {
            split: _split_window_summary(list((combined_streams.get(split)) or []))
            for split in ("train", "val", "oos")
        }
        if has_streams
        else {},
    }


def _one_shot_baseline_support(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = dict(_load_json(path))
    combined_streams = {
        split: _aggregate_stream(
            list(((payload.get("portfolio_return_streams") or {}).get(split)) or [])
        )
        for split in ("train", "val", "oos")
    }
    has_streams = any(combined_streams.values())
    return {
        "path": str(path.resolve()),
        "generated_at": payload.get("generated_at"),
        "normalization_method": "normalized_existing_portfolio_streams"
        if has_streams
        else "summary_only_missing_streams",
        "portfolio_metrics": dict(payload.get("portfolio_metrics") or payload.get("metrics") or {}),
        "combined_streams": combined_streams if has_streams else {},
        "split_windows": {
            split: _split_window_summary(list((combined_streams.get(split)) or []))
            for split in ("train", "val", "oos")
        }
        if has_streams
        else {},
        "weight_count": len(list(payload.get("weights") or [])),
    }


def _build_manifest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    selected_rows = [
        dict(row) for row in list(payload.get("candidates") or []) if isinstance(row, dict)
    ]
    baseline_support = dict(payload.get("baseline_support") or {})
    pair_payload = dict(payload.get("pair_payload") or {})
    selection = dict(payload.get("selection") or {})
    return {
        "generated_at": payload.get("generated_at"),
        "artifact_kind": "portfolio_exact_window_freeze_manifest",
        "schema_version": payload.get("schema_version"),
        "selection_basis": payload.get("selection_basis"),
        "split_windows": split_windows(),
        "memory_policy": dict(payload.get("memory_policy") or {}),
        "comparison_scope": list(payload.get("comparison_scope") or []),
        "source_manifest": dict(payload.get("source_manifest") or {}),
        "source_of_truth": dict(selection.get("source_of_truth") or {}),
        "rolling_admission_blocked": bool(payload.get("rolling_admission_blocked")),
        "rolling_gate": dict(payload.get("rolling_gate") or {}),
        "anchor_decisions": list(selection.get("anchor_decisions") or []),
        "pairspread_exclusion": {
            "excluded": True,
            "review_not_before": pair_payload.get("review_not_before") or PAIR_REVIEW_NOT_BEFORE,
            "coverage_guard": dict(pair_payload.get("coverage_guard") or {}),
        },
        "baseline_support": baseline_support,
        "selected_rows": [
            {
                "candidate_id": row.get("candidate_id"),
                "name": row.get("name"),
                "strategy_class": row.get("strategy_class"),
                "timeframe": row.get("timeframe") or row.get("strategy_timeframe"),
                "selection_basis": row.get("selection_basis"),
                "freeze_score": row.get("freeze_score"),
                "freeze_secondary_score": row.get("freeze_secondary_score"),
                "anchor_score": row.get("anchor_score"),
                "anchor_decision": row.get("anchor_decision"),
                "split_windows": {
                    split: _split_window_summary(
                        list(((row.get("return_streams") or {}).get(split)) or [])
                    )
                    for split in ("train", "val", "oos")
                },
            }
            for row in selected_rows
        ],
        "notes": list(payload.get("notes") or []),
    }


def build_portfolio_exact_window_freeze(
    *,
    report_root: Path | str = REPORT_ROOT,
    output_dir: Path | str | None = None,
    grouped_rows: dict[str, list[dict[str, Any]]] | None = None,
    grouped_candidate_rows: dict[str, list[dict[str, Any]]] | None = None,
    candidate_groups: dict[str, list[dict[str, Any]]] | None = None,
    rows_by_sleeve: dict[str, list[dict[str, Any]]] | None = None,
    sleeve_rows: dict[str, list[dict[str, Any]]] | None = None,
    sleeve_groups: dict[str, list[dict[str, Any]]] | None = None,
    source_payloads: dict[str, list[dict[str, Any]]] | None = None,
    component_rows: list[dict[str, Any]] | None = None,
    candidate_rows: list[dict[str, Any]] | None = None,
    rows: list[dict[str, Any]] | None = None,
    rolling_gate_payload: dict[str, Any] | None = None,
    rolling_breakout_gate_payload: dict[str, Any] | None = None,
    gate_payload: dict[str, Any] | None = None,
    pair_payload: dict[str, Any] | None = None,
    pair_spread_payload: dict[str, Any] | None = None,
    pairspread_payload: dict[str, Any] | None = None,
    include_one_shot_baseline: bool = True,
    selection_mode: str = DEFAULT_SELECTION_MODE,
    source_manifest_path: Path | str = FREEZE_MANIFEST_PATH,
    incumbent_bundle_path: Path | str = INCUMBENT_BUNDLE_PATH,
    anchor_bundle_path: Path | str | None = None,
    incumbent_bundle_payload: dict[str, Any] | None = None,
    anchor_bundle_payload: dict[str, Any] | None = None,
    incumbent_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_root = Path(report_root)
    _ = output_dir  # reserved for future use in callers/tests
    _ = component_rows
    _ = candidate_rows
    _ = rows
    normalized_selection_mode = str(selection_mode).strip().lower() or DEFAULT_SELECTION_MODE
    if normalized_selection_mode not in {
        DEFAULT_SELECTION_MODE,
        INCUMBENT_ANCHORED_SELECTION_MODE,
    }:
        raise ValueError(f"unsupported selection_mode={selection_mode!r}")

    explicit_grouped_inputs = any(
        isinstance(candidate, dict) and candidate
        for candidate in (
            grouped_rows,
            grouped_candidate_rows,
            candidate_groups,
            rows_by_sleeve,
            sleeve_rows,
            sleeve_groups,
            source_payloads,
        )
    )
    if explicit_grouped_inputs:
        grouped, source_manifest = _coerce_grouped_rows(
            report_root=resolved_root,
            grouped_rows=grouped_rows,
            grouped_candidate_rows=grouped_candidate_rows,
            candidate_groups=candidate_groups,
            rows_by_sleeve=rows_by_sleeve,
            sleeve_rows=sleeve_rows,
            sleeve_groups=sleeve_groups,
            source_payloads=source_payloads,
        )
    elif normalized_selection_mode == INCUMBENT_ANCHORED_SELECTION_MODE:
        grouped, source_manifest = _load_grouped_rows_from_manifest(
            Path(source_manifest_path),
            report_root=resolved_root,
        )
    else:
        grouped, source_manifest = _coerce_grouped_rows(
            report_root=resolved_root,
            grouped_rows=grouped_rows,
            grouped_candidate_rows=grouped_candidate_rows,
            candidate_groups=candidate_groups,
            rows_by_sleeve=rows_by_sleeve,
            sleeve_rows=sleeve_rows,
            sleeve_groups=sleeve_groups,
            source_payloads=source_payloads,
        )
    resolved_pair_payload = _coerce_pair_payload(
        pair_payload=pair_payload,
        pair_spread_payload=pair_spread_payload,
        pairspread_payload=pairspread_payload,
    )
    resolved_gate_payload = _coerce_rolling_gate_payload(
        rolling_gate_payload=rolling_gate_payload,
        rolling_breakout_gate_payload=rolling_breakout_gate_payload,
        gate_payload=gate_payload,
    )

    selected_rows: list[dict[str, Any]] = []
    ranking_tables: dict[str, list[dict[str, Any]]] = {}
    source_of_truth: dict[str, str] = {}
    excluded_sleeves: list[dict[str, Any]] = []
    anchor_decisions: list[dict[str, Any]] = []
    rolling_admission_blocked = False

    if normalized_selection_mode == INCUMBENT_ANCHORED_SELECTION_MODE:
        incumbent_bundle = _coerce_incumbent_bundle_payload(
            incumbent_bundle_payload=incumbent_bundle_payload,
            anchor_bundle_payload=anchor_bundle_payload,
            incumbent_payload=incumbent_payload,
            incumbent_bundle_path=incumbent_bundle_path,
            anchor_bundle_path=anchor_bundle_path,
        )
        incumbent_rows = [
            dict(row)
            for row in list(incumbent_bundle.get("candidates") or [])
            if isinstance(row, dict)
        ]
        if not incumbent_rows:
            raise RuntimeError(
                "incumbent-aware selection_mode requires incumbent bundle candidates"
            )

        grouped_by_strategy: dict[tuple[str, str], tuple[str, list[dict[str, Any]]]] = {}
        for sleeve_name, sleeve_candidates in grouped.items():
            if sleeve_candidates:
                ranking_tables[sleeve_name] = _ranking_table(sleeve_candidates)
            strategy_classes = {
                str(row.get("strategy_class") or "") for row in list(sleeve_candidates or [])
            }
            if "PairSpreadZScoreStrategy" in strategy_classes and not bool(
                resolved_pair_payload.get("survives")
            ):
                excluded_sleeves.append(
                    {
                        "sleeve": sleeve_name,
                        "reason": "pair_spread_excluded_by_followup_guard",
                        "review_not_before": resolved_pair_payload.get("review_not_before"),
                    }
                )
            for row in sleeve_candidates:
                key = _strategy_key(dict(row))
                if key != ("", "") and key not in grouped_by_strategy:
                    grouped_by_strategy[key] = (
                        sleeve_name,
                        [dict(item) for item in sleeve_candidates],
                    )
                    break

        for incumbent_row in incumbent_rows:
            strategy_key = _strategy_key(incumbent_row)
            sleeve_name, sleeve_candidates = grouped_by_strategy.get(
                strategy_key,
                (" / ".join(part for part in strategy_key if part), []),
            )
            if sleeve_candidates:
                chosen, decision = _select_anchor_or_local_challenger(
                    incumbent_row,
                    sleeve_candidates,
                )
            else:
                chosen = dict(incumbent_row)
                chosen["anchor_decision"] = "keep_anchor"
                chosen["anchor_decision_reason"] = (
                    "incumbent anchor retained because no exact-window local sleeve source was available"
                )
                chosen["source_of_truth"] = "incumbent_bundle_anchor"
                decision = {
                    "decision": chosen["anchor_decision"],
                    "reason": chosen["anchor_decision_reason"],
                    "score_delta": 0.0,
                }
            chosen.setdefault("rolling_gate_status", "not_applicable")
            source_of_truth[sleeve_name] = str(
                chosen.get("source_of_truth") or "incumbent_bundle_anchor"
            )
            selected_rows.append(
                _trim_selected_row(
                    chosen,
                    selection_basis=normalized_selection_mode,
                )
            )
            anchor_decisions.append(
                {
                    "sleeve": sleeve_name,
                    "strategy_class": chosen.get("strategy_class"),
                    "timeframe": chosen.get("timeframe") or chosen.get("strategy_timeframe"),
                    "selected_candidate_id": chosen.get("candidate_id"),
                    "selected_name": chosen.get("name"),
                    **decision,
                }
            )

        rolling_key = ("RollingBreakoutStrategy", "30m")
        rolling_sleeve_name, rolling_candidates = grouped_by_strategy.get(
            rolling_key,
            ("rolling_breakout_30m", []),
        )
        if rolling_candidates or dict(resolved_gate_payload.get("gated_candidate_row") or {}):
            if _rolling_gate_survives(resolved_gate_payload):
                chosen = _select_rolling_candidate(rolling_candidates, resolved_gate_payload)
                source_of_truth[rolling_sleeve_name] = str(
                    chosen.get("source_of_truth") or "rolling_breakout_train_val_gate"
                )
                selected_rows.append(
                    _trim_selected_row(
                        chosen,
                        selection_basis=normalized_selection_mode,
                    )
                )
                anchor_decisions.append(
                    {
                        "sleeve": rolling_sleeve_name,
                        "strategy_class": chosen.get("strategy_class"),
                        "timeframe": chosen.get("timeframe") or chosen.get("strategy_timeframe"),
                        "selected_candidate_id": chosen.get("candidate_id"),
                        "selected_name": chosen.get("name"),
                        "decision": chosen.get("anchor_decision"),
                        "reason": chosen.get("anchor_decision_reason"),
                        "score_delta": None,
                    }
                )
            else:
                rolling_admission_blocked = True
                excluded_sleeves.append(
                    {
                        "sleeve": rolling_sleeve_name,
                        "reason": "rolling_breakout_blocked_by_train_val_gate",
                        "selection_basis": _rolling_gate_selection_basis(resolved_gate_payload),
                        "survives_train_val": False,
                    }
                )
    else:
        for sleeve_name, sleeve_candidates in grouped.items():
            strategy_classes = {
                str(row.get("strategy_class") or "") for row in list(sleeve_candidates or [])
            }
            if "PairSpreadZScoreStrategy" in strategy_classes and not bool(
                resolved_pair_payload.get("survives")
            ):
                excluded_sleeves.append(
                    {
                        "sleeve": sleeve_name,
                        "reason": "pair_spread_excluded_by_followup_guard",
                        "review_not_before": resolved_pair_payload.get("review_not_before"),
                    }
                )
                continue
            if not sleeve_candidates:
                continue

            ranking_tables[sleeve_name] = _ranking_table(sleeve_candidates)
            chosen = dict(min(sleeve_candidates, key=_ranking_key))
            if str(chosen.get("strategy_class") or "") == "RollingBreakoutStrategy":
                source_of_truth[sleeve_name] = "exact_window_candidate_details_then_gate_supplement"
                chosen = _supplement_rolling_gate(chosen, resolved_gate_payload)
            else:
                source_of_truth[sleeve_name] = "exact_window_candidate_details"

            selected_rows.append(
                _trim_selected_row(
                    chosen,
                    selection_basis=normalized_selection_mode,
                )
            )

    selected_rows.sort(
        key=lambda row: (
            str(row.get("strategy_class") or ""),
            str(row.get("timeframe") or row.get("strategy_timeframe") or ""),
        )
    )

    equal_weight_summary = _summary_from_payload(EQUAL_WEIGHT_BASELINE_PATH)
    one_shot_summary = (
        _summary_from_payload(ONE_SHOT_BASELINE_PATH) if include_one_shot_baseline else {}
    )
    equal_weight_baseline_support = _equal_weight_baseline_support(EQUAL_WEIGHT_BASELINE_PATH)
    one_shot_baseline_support = (
        _one_shot_baseline_support(ONE_SHOT_BASELINE_PATH) if include_one_shot_baseline else {}
    )

    if normalized_selection_mode == INCUMBENT_ANCHORED_SELECTION_MODE:
        notes = [
            "Incumbent-aware mode seeds the existing one-shot incumbent sleeves and only promotes local challengers when the explicit anchor thresholds are met.",
            "RollingBreakout admission is conditional on the rebuilt train+val-only gate; locked OOS is excluded from sleeve admission.",
            "PairSpread 4h remains excluded unless the follow-up guard explicitly survives.",
            "Equal-weight baseline default handling is rebuild/normalize from component streams before comparison.",
        ]
        if rolling_admission_blocked:
            notes.append(
                "RollingBreakout gate failed the train+val-only admission thresholds, so the anchored bundle remains 3-sleeve and the optimizer lane should stay blocked."
            )
        else:
            notes.append(
                "RollingBreakout cleared the train+val-only gate and is admitted as the conditional fourth sleeve."
            )
    else:
        notes = [
            "Exact-window sleeve freeze uses the deterministic train/val-only formula; OOS-derived helper fields are excluded from selection.",
            "RollingBreakout is frozen from exact-window candidate-detail rows first, then optionally supplemented by the gate artifact for stream/metadata only.",
            "PairSpread 4h is excluded unless the follow-up guard explicitly survives.",
            "Equal-weight baseline default handling is rebuild/normalize from component streams before 3-way comparison.",
        ]
    if include_one_shot_baseline:
        notes.append(
            "One-shot optimized baseline defaults to reports/portfolio_optimization_latest.json."
        )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_kind": "portfolio_exact_window_freeze",
        "schema_version": "1.0",
        "selection_basis": normalized_selection_mode,
        "comparison_scope": (
            [
                "equal_weight_diagnostic",
                "one_shot_optimized",
                "portfolio_four_sleeve_anchored_bundle",
            ]
            if normalized_selection_mode == INCUMBENT_ANCHORED_SELECTION_MODE
            else [
                "equal_weight_diagnostic",
                "one_shot_optimized",
                "exact_window_frozen",
            ]
        ),
        "candidates": selected_rows,
        "selected_team": selected_rows,
        "frozen_rows": selected_rows,
        "rolling_admission_blocked": rolling_admission_blocked,
        "rolling_gate": {
            "selection_basis": _rolling_gate_selection_basis(resolved_gate_payload),
            "survives_train_val": _rolling_gate_survives(resolved_gate_payload),
            "recommended_action": resolved_gate_payload.get("recommended_action"),
        },
        "optimizer_bundle": {
            "artifact_kind": "portfolio_exact_window_freeze_optimizer_bundle",
            "selection_basis": normalized_selection_mode,
            "candidates": selected_rows,
            "selected_team": selected_rows,
            "source_artifact_kind": "portfolio_exact_window_freeze",
            "rolling_admission_blocked": rolling_admission_blocked,
        },
        "ranking_tables": ranking_tables,
        "excluded_sleeves": excluded_sleeves,
        "pair_survives": bool(resolved_pair_payload.get("survives")),
        "pair_payload": {
            "survives": bool(resolved_pair_payload.get("survives")),
            "coverage_guard": dict(resolved_pair_payload.get("coverage_guard") or {}),
            "review_not_before": resolved_pair_payload.get("review_not_before")
            or PAIR_REVIEW_NOT_BEFORE,
        },
        "selection": {
            "selection_basis": normalized_selection_mode,
            "formula": (
                "(3*val.sharpe)+(2*val.deflated_sharpe)+(18*val.return)"
                "-(2.5*val.pbo)-(0.15*val.turnover)"
                "+(0.5*train.deflated_sharpe)+(0.25*train.sharpe)"
                if normalized_selection_mode == INCUMBENT_ANCHORED_SELECTION_MODE
                else "(3*val.sharpe)+(2*val.deflated_sharpe)+(25*val.return)-(2.5*val.pbo)-(0.25*val.turnover)"
            ),
            "secondary_rule": (
                "promote local challenger only if score_delta>=0.25, val.pbo<=min(0.45, anchor.pbo+0.05), "
                "val.deflated_sharpe>=anchor.val.deflated_sharpe-0.02, and train.return>=anchor.train.return-0.02"
                if normalized_selection_mode == INCUMBENT_ANCHORED_SELECTION_MODE
                else "train.deflated_sharpe + 0.5*train.sharpe"
            ),
            "denylisted_fields": sorted(DENYLIST_FIELDS),
            "source_of_truth": source_of_truth,
            "count": len(selected_rows),
            "anchor_decisions": anchor_decisions,
            "rolling_admission_blocked": rolling_admission_blocked,
        },
        "split_windows": split_windows(),
        "memory_policy": memory_policy_payload(),
        "incumbent_bundle_path": str(Path(incumbent_bundle_path).resolve()),
        "baseline_paths": {
            "equal_weight_diagnostic": str(EQUAL_WEIGHT_BASELINE_PATH.resolve()),
            "one_shot_optimized": str(ONE_SHOT_BASELINE_PATH.resolve()),
        },
        "baseline_summaries": {
            "equal_weight_diagnostic": equal_weight_summary,
            "one_shot_optimized": one_shot_summary,
        },
        "baseline_support": {
            "equal_weight_diagnostic": equal_weight_baseline_support,
            "one_shot_optimized": one_shot_baseline_support,
        },
        "source_manifest": source_manifest,
        "notes": notes,
    }
    return payload


def write_portfolio_exact_window_freeze(
    *,
    report_root: Path | str = REPORT_ROOT,
    output_dir: Path | str | None = None,
    run_name: str = "portfolio_exact_window_freeze",
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_portfolio_exact_window_freeze(
        report_root=report_root,
        output_dir=output_dir,
        **kwargs,
    )
    manifest_payload = _build_manifest_payload(payload)
    followup_root = Path(output_dir or report_root) / "followup_status"
    followup_root.mkdir(parents=True, exist_ok=True)
    json_path = followup_root / f"{run_name}_latest.json"
    md_path = followup_root / f"{run_name}_latest.md"
    manifest_json_path = followup_root / f"{run_name}_manifest_latest.json"
    manifest_md_path = followup_root / f"{run_name}_manifest_latest.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    manifest_json_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    lines = [
        "# portfolio exact-window freeze",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- selection_basis: `{payload['selection_basis']}`",
        f"- frozen_count: `{len(list(payload.get('candidates') or []))}`",
        f"- rolling_admission_blocked: `{payload.get('rolling_admission_blocked')}`",
        f"- pair_survives: `{payload.get('pair_survives')}`",
        f"- equal_weight_baseline: `{payload['baseline_paths']['equal_weight_diagnostic']}`",
        f"- one_shot_baseline: `{payload['baseline_paths']['one_shot_optimized']}`",
        "",
        "## frozen sleeves",
    ]
    for row in list(payload.get("candidates") or []):
        val = dict(row.get("val") or {})
        lines.append(
            f"- `{row.get('name')}` | strategy={row.get('strategy_class')} | tf={row.get('timeframe') or row.get('strategy_timeframe')} | "
            f"val_sharpe={_safe_float(val.get('sharpe'), 0.0):.3f} | "
            f"val_return={_safe_float(val.get('return'), 0.0):.4%} | "
            f"val_pbo={_safe_float(val.get('pbo'), 1.0):.3f}"
        )
    lines.extend(["", "## notes"])
    for note in list(payload.get("notes") or []):
        lines.append(f"- {note}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_lines = [
        "# portfolio exact-window freeze manifest",
        "",
        f"- generated_at: `{manifest_payload.get('generated_at')}`",
        f"- selection_basis: `{manifest_payload.get('selection_basis')}`",
        f"- oos_start: `{dict(manifest_payload.get('split_windows') or {}).get('oos_start')}`",
        f"- heavy_lock_path: `{dict(manifest_payload.get('memory_policy') or {}).get('heavy_lock_path')}`",
        "",
        "## sources",
    ]
    for sleeve, source in sorted(dict(manifest_payload.get("source_manifest") or {}).items()):
        manifest_lines.append(f"- {sleeve}: `{source}`")
    pair = dict(manifest_payload.get("pairspread_exclusion") or {})
    manifest_lines.extend(
        [
            "",
            "## baselines",
            f"- equal_weight: `{dict((manifest_payload.get('baseline_support') or {}).get('equal_weight_diagnostic') or {}).get('normalization_method')}`",
            f"- one_shot_optimized: `{dict((manifest_payload.get('baseline_support') or {}).get('one_shot_optimized') or {}).get('normalization_method')}`",
            "",
            "## exclusions",
            f"- PairSpread 4h excluded: `{pair.get('excluded')}` | review_not_before=`{pair.get('review_not_before')}`",
        ]
    )
    manifest_md_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return {
        "payload": payload,
        "json_path": str(json_path.resolve()),
        "md_path": str(md_path.resolve()),
        "manifest_payload": manifest_payload,
        "manifest_json_path": str(manifest_json_path.resolve()),
        "manifest_md_path": str(manifest_md_path.resolve()),
    }


def main() -> int:
    result = write_portfolio_exact_window_freeze()
    print(result["json_path"])
    print(result["md_path"])
    print(result["manifest_json_path"])
    print(result["manifest_md_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
