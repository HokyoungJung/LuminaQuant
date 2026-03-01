"""Selection helpers for building a diversified portfolio shortlist."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from typing import Any

DEFAULT_ROBUST_SCORE_WEIGHTS: dict[str, float] = {
    "sharpe_weight": 2.8,
    "deflated_sharpe_weight": 1.5,
    "pbo_penalty": 2.0,
    "return_weight": 35.0,
    "drawdown_penalty": 3.0,
    "turnover_penalty": 2.5,
    "cross_corr_penalty": 0.8,
}
DEFAULT_ROBUST_SCORE_PARAMS: dict[str, float] = {
    "turnover_threshold": 2.5,
    "failed_candidate_scale": 0.1,
    **DEFAULT_ROBUST_SCORE_WEIGHTS,
}


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
    if "composite_trend" in token:
        return "trend"
    if "volcomp" in token:
        return "mean_reversion"
    if "leadlag" in token:
        return "intraday_alpha"
    if "perp_crowding" in token or "carry" in token:
        return "carry"
    if "micro_range" in token:
        return "micro"
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


def _resolve_robust_score_params(overrides: dict[str, Any] | None = None) -> dict[str, float]:
    merged = dict(DEFAULT_ROBUST_SCORE_PARAMS)
    if not isinstance(overrides, dict):
        return merged
    for key, value in overrides.items():
        if key in merged:
            merged[key] = safe_float(value, merged[key])
    return merged


def robust_score_from_metrics(
    metrics: dict[str, Any],
    *,
    params: dict[str, Any] | None = None,
) -> float:
    cfg = _resolve_robust_score_params(params)
    turnover_threshold = float(cfg["turnover_threshold"])
    return float(
        (float(cfg["sharpe_weight"]) * safe_float(metrics.get("sharpe"), 0.0))
        + (float(cfg["deflated_sharpe_weight"]) * safe_float(metrics.get("deflated_sharpe"), 0.0))
        - (float(cfg["pbo_penalty"]) * safe_float(metrics.get("pbo"), 1.0))
        + (float(cfg["return_weight"]) * safe_float(metrics.get("return"), 0.0))
        - (float(cfg["drawdown_penalty"]) * safe_float(metrics.get("mdd"), 0.0))
        - (
            float(cfg["turnover_penalty"])
            * max(0.0, safe_float(metrics.get("turnover"), 0.0) - turnover_threshold)
        )
        - (float(cfg["cross_corr_penalty"]) * safe_float(metrics.get("cross_candidate_corr"), 0.0))
    )


def hurdle_score(
    candidate: dict[str, Any],
    *,
    mode: str = "oos",
    robust_score_params: dict[str, Any] | None = None,
) -> float:
    mode_token = str(mode).strip().lower()
    hurdle_key = "val" if mode_token == "live" else "oos"
    hurdle = ((candidate.get("hurdle_fields") or {}).get(hurdle_key)) or {}

    score = safe_float(hurdle.get("score"), -1_000_000.0)
    excess_return = safe_float(hurdle.get("excess_return"), -1_000_000.0)
    passed = bool(hurdle.get("pass", False))

    metric_key = "val" if mode_token == "live" else "oos"
    metrics = candidate.get(metric_key)
    if isinstance(metrics, dict):
        robust_score = robust_score_from_metrics(metrics, params=robust_score_params)
        cfg = _resolve_robust_score_params(robust_score_params)
        failed_scale = float(cfg["failed_candidate_scale"])
        if passed:
            return max(score, robust_score)
        metric_return = safe_float(metrics.get("return"), -1_000_000.0)
        return -500_000.0 + metric_return + (failed_scale * robust_score)

    return -1_000_000.0 + excess_return


def candidate_mix_type(candidate: dict[str, Any]) -> str:
    name_token = str(candidate.get("name", "")).strip().lower()
    symbols = [str(item).strip() for item in list(candidate.get("symbols") or []) if str(item).strip()]
    if name_token.startswith(("pair_", "lag_convergence")):
        return "pair"
    if len(symbols) >= 3:
        return "multi"
    if len(symbols) == 2:
        return "pair"
    return "single"


def _has_mode_metrics(candidate: dict[str, Any], *, mode: str) -> bool:
    mode_token = str(mode).strip().lower()
    key = "val" if mode_token == "live" else "oos"
    metrics = candidate.get(key)
    if not isinstance(metrics, dict):
        return False
    # At least one concrete metric should exist.
    for field in ("return", "sharpe", "mdd", "trades"):
        if field in metrics and metrics.get(field) is not None:
            return True
    return False


def _mode_metrics(candidate: dict[str, Any], *, mode: str) -> dict[str, Any]:
    mode_token = str(mode).strip().lower()
    key = "val" if mode_token == "live" else "oos"
    metrics = candidate.get(key)
    if isinstance(metrics, dict):
        return metrics
    return {}


def allocate_portfolio_weights(
    shortlist: Iterable[dict[str, Any]],
    *,
    score_key: str = "shortlist_score",
    temperature: float = 0.35,
    max_weight: float = 0.35,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in shortlist]
    if not rows:
        return rows

    capped_max = min(1.0, max(0.05, float(max_weight)))
    temp = max(0.05, float(temperature))

    family_counts: dict[str, int] = {}
    timeframe_counts: dict[str, int] = {}
    for row in rows:
        family = strategy_family(str(row.get("name", "")), fallback=str(row.get("family", "other")))
        timeframe = str(row.get("strategy_timeframe") or row.get("timeframe") or "")
        family_counts[family] = family_counts.get(family, 0) + 1
        timeframe_counts[timeframe] = timeframe_counts.get(timeframe, 0) + 1

    scores = [safe_float(row.get(score_key), safe_float(row.get("selection_score"), -1_000_000.0)) for row in rows]
    max_score = max(scores)

    raw_weights: list[float] = []
    for row, score in zip(rows, scores, strict=True):
        family = strategy_family(str(row.get("name", "")), fallback=str(row.get("family", "other")))
        timeframe = str(row.get("strategy_timeframe") or row.get("timeframe") or "")
        mix_type = candidate_mix_type(row)

        scaled = (score - max_score) / temp
        base = math.exp(max(-60.0, min(0.0, scaled)))
        diversity_penalty = 1.0 / math.sqrt(max(1, family_counts.get(family, 1)))
        timeframe_penalty = 1.0 / math.sqrt(max(1, timeframe_counts.get(timeframe, 1)))
        mix_bonus = 1.05 if mix_type in {"pair", "multi"} else 1.0

        mdd = abs(safe_float((row.get("oos") or {}).get("mdd"), 0.0))
        risk_penalty = 1.0 / (1.0 + (2.5 * max(0.0, mdd)))
        raw_weights.append(base * diversity_penalty * timeframe_penalty * mix_bonus * risk_penalty)

    weight_sum = float(sum(raw_weights))
    if weight_sum <= 0.0:
        equal = 1.0 / float(len(rows))
        for row in rows:
            row["portfolio_weight"] = equal
        return rows

    normalized = [value / weight_sum for value in raw_weights]

    # One-pass capping.
    capped = [min(capped_max, value) for value in normalized]
    capped_sum = float(sum(capped))
    if capped_sum <= 0.0:
        equal = 1.0 / float(len(rows))
        for row in rows:
            row["portfolio_weight"] = equal
        return rows
    normalized = [value / capped_sum for value in capped]

    for row, value in zip(rows, normalized, strict=True):
        row["portfolio_weight"] = float(value)

    rows.sort(key=lambda item: float(item.get("portfolio_weight", 0.0)), reverse=True)
    return rows


def build_single_asset_portfolio_sets(
    shortlist: Iterable[dict[str, Any]],
    *,
    mode: str = "oos",
    max_per_asset: int = 2,
    max_sets: int = 16,
    robust_score_params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build portfolio-set candidates from successful single-asset strategies."""
    rows = [dict(row) for row in shortlist]
    if not rows:
        return []

    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if candidate_mix_type(row) != "single":
            continue
        symbols = [str(item).strip().upper() for item in list(row.get("symbols") or []) if str(item).strip()]
        if len(symbols) != 1:
            continue
        symbol = symbols[0]
        by_symbol.setdefault(symbol, []).append(row)

    if not by_symbol:
        return []

    max_per_asset_i = max(1, int(max_per_asset))
    for symbol, symbol_rows in by_symbol.items():
        symbol_rows.sort(
            key=lambda row: float(
                row.get(
                    "shortlist_score",
                    hurdle_score(row, mode=mode, robust_score_params=robust_score_params),
                )
            ),
            reverse=True,
        )
        by_symbol[symbol] = symbol_rows[:max_per_asset_i]

    symbols_sorted = sorted(by_symbol.keys())
    top_members = [by_symbol[symbol][0] for symbol in symbols_sorted if by_symbol[symbol]]
    if not top_members:
        return []

    out: list[dict[str, Any]] = []
    max_sets_i = max(1, int(max_sets))

    def _normalize_set_weights(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        scores = [
            float(
                item.get(
                    "shortlist_score",
                    hurdle_score(item, mode=mode, robust_score_params=robust_score_params),
                )
            )
            for item in items
        ]
        max_score = max(scores)
        raw = [math.exp(max(-60.0, min(0.0, score - max_score))) for score in scores]
        total = float(sum(raw))
        if total <= 0.0:
            equal = 1.0 / float(len(items))
            return [{**item, "portfolio_weight": equal} for item in items]
        return [{**item, "portfolio_weight": float(weight / total)} for item, weight in zip(items, raw, strict=True)]

    out.append(
        {
            "set_id": "single_asset_top_set",
            "member_count": len(top_members),
            "members": _normalize_set_weights(top_members),
        }
    )

    # Additional sets: rotate second-best candidate for each symbol (if available).
    if max_sets_i > 1:
        for symbol in symbols_sorted:
            symbol_rows = by_symbol[symbol]
            if len(symbol_rows) < 2:
                continue
            variant_members: list[dict[str, Any]] = []
            for current_symbol in symbols_sorted:
                entries = by_symbol[current_symbol]
                if current_symbol == symbol and len(entries) >= 2:
                    variant_members.append(entries[1])
                else:
                    variant_members.append(entries[0])
            out.append(
                {
                    "set_id": f"single_asset_variant_{symbol.replace('/', '')}",
                    "member_count": len(variant_members),
                    "members": _normalize_set_weights(variant_members),
                }
            )
            if len(out) >= max_sets_i:
                break

    return out


def select_diversified_shortlist(
    candidates: Iterable[dict[str, Any]],
    *,
    mode: str = "oos",
    max_total: int = 24,
    max_per_family: int = 8,
    max_per_timeframe: int = 6,
    single_min_score: float | None = None,
    drop_single_without_metrics: bool = False,
    single_min_return: float | None = None,
    single_min_sharpe: float | None = None,
    single_min_trades: int | None = None,
    allow_multi_asset: bool = True,
    include_weights: bool = False,
    weight_temperature: float = 0.35,
    max_weight: float = 0.35,
    robust_score_params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda row: hurdle_score(row, mode=mode, robust_score_params=robust_score_params),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    family_count: dict[str, int] = {}
    timeframe_count: dict[str, int] = {}
    seen_identities: set[str] = set()

    for row in ranked:
        if len(selected) >= int(max_total):
            break

        timeframe = str(row.get("strategy_timeframe") or row.get("timeframe") or "").lower()
        family = strategy_family(str(row.get("name", "")), fallback=str(row.get("family", "other")))
        mix_type = candidate_mix_type(row)
        identity = str(row.get("identity") or candidate_identity(row))
        score = float(hurdle_score(row, mode=mode, robust_score_params=robust_score_params))

        if identity in seen_identities:
            continue
        if not bool(allow_multi_asset) and mix_type == "multi":
            continue
        if mix_type == "single":
            if bool(drop_single_without_metrics) and not _has_mode_metrics(row, mode=mode):
                continue
            if single_min_score is not None and score < float(single_min_score):
                continue
            metrics = _mode_metrics(row, mode=mode)
            if single_min_return is not None:
                metric_return = safe_float(metrics.get("return"), float("-inf"))
                if metric_return < float(single_min_return):
                    continue
            if single_min_sharpe is not None:
                metric_sharpe = safe_float(metrics.get("sharpe"), float("-inf"))
                if metric_sharpe < float(single_min_sharpe):
                    continue
            if single_min_trades is not None:
                metric_trades = safe_float(metrics.get("trades"), float("-inf"))
                if metric_trades < float(single_min_trades):
                    continue
        if family_count.get(family, 0) >= int(max_per_family):
            continue
        if timeframe_count.get(timeframe, 0) >= int(max_per_timeframe):
            continue

        enriched = dict(row)
        enriched["family"] = family
        enriched["identity"] = identity
        enriched["mix_type"] = mix_type
        enriched["shortlist_score"] = score
        selected.append(enriched)

        seen_identities.add(identity)
        family_count[family] = family_count.get(family, 0) + 1
        timeframe_count[timeframe] = timeframe_count.get(timeframe, 0) + 1

    if include_weights:
        return allocate_portfolio_weights(
            selected,
            score_key="shortlist_score",
            temperature=weight_temperature,
            max_weight=max_weight,
        )
    return selected


def summarize_shortlist(shortlist: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_family: dict[str, int] = {}
    by_timeframe: dict[str, int] = {}
    by_mix: dict[str, int] = {}

    for row in shortlist:
        family = str(row.get("family", "other"))
        timeframe = str(row.get("strategy_timeframe") or row.get("timeframe") or "")
        mix = str(row.get("mix_type") or candidate_mix_type(row))
        by_family[family] = by_family.get(family, 0) + 1
        by_timeframe[timeframe] = by_timeframe.get(timeframe, 0) + 1
        by_mix[mix] = by_mix.get(mix, 0) + 1

    return {
        "family": by_family,
        "mix": by_mix,
        "timeframe": by_timeframe,
    }
