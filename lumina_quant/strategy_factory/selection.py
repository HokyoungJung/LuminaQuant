"""Selection helpers for building a diversified portfolio shortlist."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def strategy_family(name: str, fallback: str = "other") -> str:
    token = str(name).strip().lower()
    if token.startswith(("pair_", "lag_convergence", "mean_rev", "mean_reversion", "vwap_rev")):
        return "market_neutral"
    if token.startswith(("topcap_", "ma_cross", "breakout_", "rolling_breakout", "rsi_")):
        return "trend"
    if token.startswith(("carry_", "funding_")):
        return "carry"
    if token:
        return fallback
    return "other"


def candidate_identity(candidate: dict[str, Any]) -> str:
    payload = {
        "name": str(candidate.get("name", "")),
        "timeframe": str(candidate.get("strategy_timeframe") or candidate.get("timeframe") or ""),
        "symbols": list(candidate.get("symbols") or []),
        "params": candidate.get("params") if isinstance(candidate.get("params"), dict) else {},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def hurdle_score(candidate: dict[str, Any], *, mode: str = "oos") -> float:
    mode_token = str(mode).strip().lower()
    hurdle_key = "val" if mode_token == "live" else "oos"
    hurdle = ((candidate.get("hurdle_fields") or {}).get(hurdle_key)) or {}

    score = safe_float(hurdle.get("score"), -1_000_000.0)
    excess_return = safe_float(hurdle.get("excess_return"), -1_000_000.0)
    passed = bool(hurdle.get("pass", False))
    if passed:
        return score

    metric_key = "val" if mode_token == "live" else "oos"
    metrics = candidate.get(metric_key)
    if isinstance(metrics, dict):
        metric_return = safe_float(metrics.get("return"), -1_000_000.0)
        return -500_000.0 + metric_return

    return -1_000_000.0 + excess_return


def select_diversified_shortlist(
    candidates: Iterable[dict[str, Any]],
    *,
    mode: str = "oos",
    max_total: int = 24,
    max_per_family: int = 8,
    max_per_timeframe: int = 6,
) -> list[dict[str, Any]]:
    ranked = sorted(candidates, key=lambda row: hurdle_score(row, mode=mode), reverse=True)

    selected: list[dict[str, Any]] = []
    family_count: dict[str, int] = {}
    timeframe_count: dict[str, int] = {}
    seen_identities: set[str] = set()

    for row in ranked:
        if len(selected) >= int(max_total):
            break

        timeframe = str(row.get("strategy_timeframe") or row.get("timeframe") or "").lower()
        family = strategy_family(str(row.get("name", "")), fallback=str(row.get("family", "other")))
        identity = str(row.get("identity") or candidate_identity(row))

        if identity in seen_identities:
            continue
        if family_count.get(family, 0) >= int(max_per_family):
            continue
        if timeframe_count.get(timeframe, 0) >= int(max_per_timeframe):
            continue

        enriched = dict(row)
        enriched["family"] = family
        enriched["identity"] = identity
        enriched["shortlist_score"] = float(hurdle_score(row, mode=mode))
        selected.append(enriched)

        seen_identities.add(identity)
        family_count[family] = family_count.get(family, 0) + 1
        timeframe_count[timeframe] = timeframe_count.get(timeframe, 0) + 1

    return selected


def summarize_shortlist(shortlist: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_family: dict[str, int] = {}
    by_timeframe: dict[str, int] = {}

    for row in shortlist:
        family = str(row.get("family", "other"))
        timeframe = str(row.get("strategy_timeframe") or row.get("timeframe") or "")
        by_family[family] = by_family.get(family, 0) + 1
        by_timeframe[timeframe] = by_timeframe.get(timeframe, 0) + 1

    return {
        "family": by_family,
        "timeframe": by_timeframe,
    }
