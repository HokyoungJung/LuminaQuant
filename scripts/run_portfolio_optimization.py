"""Constrained portfolio optimization over shortlisted strategy return streams."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_METALS = {"XAU/USDT", "XAG/USDT"}

DEFAULT_PORTFOLIO_SCORING_CONFIG: dict[str, Any] = {
    "candidate_rank_score_weights": {
        "sharpe_weight": 2.8,
        "deflated_sharpe_weight": 1.5,
        "pbo_penalty": 2.0,
        "return_weight": 25.0,
    },
    "allocation_quality_params": {
        "deflated_sharpe_floor": 0.01,
        "deflated_sharpe_offset": 0.5,
    },
    "vol_targeting": {
        "target_vol_floor": 0.01,
        "vol_scale_cap": 2.0,
        "vol_scale_epsilon": 1e-12,
    },
    "sensitivity": {
        "cost_stress_x2_multiplier": 2.0,
        "cost_stress_x3_multiplier": 3.0,
        "signal_drift_down_multiplier": 0.9,
        "signal_drift_up_multiplier": 1.1,
    },
    "constraints": {
        "max_strategy": 0.15,
        "max_family": 0.40,
        "max_asset": 0.20,
        "max_metals": 0.15,
    },
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _stream_to_array(stream: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([_safe_float(item.get("v"), 0.0) for item in stream], dtype=float)


def _load_score_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"score config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid score config JSON ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"score config must be a JSON object: {path}")
    return dict(payload)


def _resolve_portfolio_score_config(overrides: dict[str, Any] | None) -> dict[str, Any]:
    resolved = deepcopy(DEFAULT_PORTFOLIO_SCORING_CONFIG)
    if not isinstance(overrides, dict):
        return resolved
    source = overrides
    nested = source.get("portfolio_optimization")
    if isinstance(nested, dict):
        source = nested
    for key, default_value in resolved.items():
        override_value = source.get(key)
        if isinstance(default_value, dict) and isinstance(override_value, dict):
            for sub_key in default_value:
                if sub_key in override_value:
                    default_value[sub_key] = override_value[sub_key]
    return resolved


def _resolved_cli_or_config_float(cli_value: float | None, config_value: Any, *, default: float) -> float:
    if cli_value is not None:
        return max(0.0, _safe_float(cli_value, default))
    return max(0.0, _safe_float(config_value, default))


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    n = min(x.size, y.size)
    if n < 8:
        return 0.0
    xa = x[-n:]
    ya = y[-n:]
    sx = float(np.std(xa, ddof=1))
    sy = float(np.std(ya, ddof=1))
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    value = float(np.corrcoef(xa, ya)[0, 1])
    if not math.isfinite(value):
        return 0.0
    return value


def _max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    dd = 1.0 - np.divide(equity, np.maximum(peaks, 1e-12))
    return float(np.max(dd)) if dd.size else 0.0


def _metrics(returns: np.ndarray, periods_per_year: int = 365) -> dict[str, float]:
    if returns.size == 0:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
        }

    total_return = float(np.prod(1.0 + returns) - 1.0)
    years = max(1.0 / periods_per_year, returns.size / periods_per_year)
    cagr = float(math.exp(math.log1p(max(-0.999999, total_return)) / years) - 1.0)
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    sharpe = 0.0 if sigma <= 1e-12 else (mu / sigma) * math.sqrt(periods_per_year)

    downside = returns[returns < 0.0]
    dsigma = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    sortino = 0.0 if dsigma <= 1e-12 else (mu / dsigma) * math.sqrt(periods_per_year)

    mdd = _max_drawdown(returns)
    calmar = 0.0 if mdd <= 1e-12 else cagr / mdd

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": mdd,
        "volatility": float(sigma * math.sqrt(periods_per_year)),
    }


def _cluster_by_correlation(ids: list[str], oos_map: dict[str, np.ndarray], threshold: float = 0.60) -> list[list[str]]:
    clusters: list[list[str]] = []
    for cid in ids:
        added = False
        for cluster in clusters:
            if any(abs(_corr(oos_map[cid], oos_map[member])) >= threshold for member in cluster):
                cluster.append(cid)
                added = True
                break
        if not added:
            clusters.append([cid])
    return clusters


def _inverse_vol_weight(returns: np.ndarray) -> float:
    sigma = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    if sigma <= 1e-9:
        return 1.0
    return 1.0 / sigma


def _normalized_symbols(record: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(symbol).strip()
            for symbol in list(record.get("symbols") or [])
            if str(symbol).strip()
        }
    )


def _project_simplex_with_upper_bounds(
    weights: dict[str, float],
    *,
    upper: dict[str, float],
    target_sum: float = 1.0,
) -> dict[str, float]:
    if not weights:
        return {}

    keys = list(weights.keys())
    target = max(0.0, float(target_sum))
    clipped_upper = {key: max(0.0, float(upper.get(key, target))) for key in keys}
    capacity = sum(clipped_upper.values())
    if capacity <= 0.0:
        return dict.fromkeys(keys, 0.0)
    if capacity < target:
        target = capacity

    pref = {key: max(0.0, float(weights.get(key, 0.0))) for key in keys}
    pref_sum = sum(pref.values())
    if pref_sum <= 0.0:
        pref = dict.fromkeys(keys, 1.0)
        pref_sum = float(len(keys))
    for key in keys:
        pref[key] /= pref_sum

    out = dict.fromkeys(keys, 0.0)
    active = set(keys)
    remaining = target

    # Water-filling with upper bounds.
    for _ in range(len(keys) + 2):
        if not active or remaining <= 1e-12:
            break
        active_pref = sum(pref[key] for key in active)
        if active_pref <= 1e-12:
            even = remaining / max(1, len(active))
            for key in list(active):
                alloc = min(even, clipped_upper[key] - out[key])
                if alloc > 0.0:
                    out[key] += alloc
                    remaining -= alloc
            break

        saturated: list[str] = []
        for key in list(active):
            alloc = remaining * (pref[key] / active_pref)
            room = clipped_upper[key] - out[key]
            if alloc >= room - 1e-12:
                if room > 0.0:
                    out[key] += room
                    remaining -= room
                saturated.append(key)
        if not saturated:
            for key in list(active):
                alloc = remaining * (pref[key] / active_pref)
                room = clipped_upper[key] - out[key]
                out[key] += min(alloc, room)
            remaining = 0.0
            break
        for key in saturated:
            active.discard(key)

    # Final numerical top-up where feasible.
    if remaining > 1e-10:
        for key in keys:
            room = clipped_upper[key] - out[key]
            if room <= 0.0:
                continue
            add = min(room, remaining)
            out[key] += add
            remaining -= add
            if remaining <= 1e-12:
                break

    return out


def _min_required_exposure(
    ratios_by_candidate: dict[str, float],
    *,
    strategy_cap: float,
) -> float:
    if not ratios_by_candidate:
        return 0.0
    cap = max(0.0, float(strategy_cap))
    ordered = sorted(float(ratios_by_candidate[key]) for key in ratios_by_candidate)
    remaining = 1.0
    total = 0.0
    for ratio in ordered:
        alloc = min(cap, remaining)
        total += alloc * ratio
        remaining -= alloc
        if remaining <= 1e-12:
            break
    if remaining > 1e-8:
        # Infeasible; treat as full exposure upper bound.
        return 1.0
    return total


def _asset_exposure(weights: dict[str, float], records: dict[str, dict[str, Any]], asset: str) -> float:
    exposure = 0.0
    token = str(asset)
    for key, weight in weights.items():
        symbols = _normalized_symbols(records.get(key) or {})
        if not symbols or token not in symbols:
            continue
        exposure += float(weight) / float(len(symbols))
    return exposure


def _metals_exposure(weights: dict[str, float], records: dict[str, dict[str, Any]]) -> float:
    exposure = 0.0
    for key, weight in weights.items():
        symbols = _normalized_symbols(records.get(key) or {})
        if not symbols:
            continue
        metals_count = sum(1 for symbol in symbols if symbol in _METALS)
        if metals_count <= 0:
            continue
        exposure += float(weight) * (float(metals_count) / float(len(symbols)))
    return exposure


def _apply_caps(
    weights: dict[str, float],
    *,
    records: dict[str, dict[str, Any]],
    max_strategy: float = 0.15,
    max_family: float = 0.40,
    max_asset: float = 0.20,
    max_metals: float = 0.15,
) -> tuple[dict[str, float], dict[str, Any]]:
    if not weights:
        return {}, {
            "max_strategy": float(max_strategy),
            "max_family": float(max_family),
            "max_asset": float(max_asset),
            "max_metals": float(max_metals),
            "family_caps": {},
        }

    out = {key: max(0.0, float(value)) for key, value in weights.items()}
    n = len(out)
    strategy_cap = max(float(max_strategy), 1.0 / max(1, n))

    families = {
        key: str((records.get(key) or {}).get("family", "other"))
        for key in out
    }
    family_counts: dict[str, int] = defaultdict(int)
    for fam in families.values():
        family_counts[fam] += 1

    family_caps: dict[str, float] = {}
    for fam in family_counts:
        other_capacity = 0.0
        for other_fam, other_count in family_counts.items():
            if other_fam == fam:
                continue
            other_capacity += float(other_count) * strategy_cap
        required = max(0.0, 1.0 - other_capacity)
        family_caps[fam] = min(1.0, max(float(max_family), required))

    # Relax caps only when mathematically required by available candidates.
    assets = sorted(
        {
            symbol
            for key in out
            for symbol in _normalized_symbols(records.get(key) or {})
        }
    )
    min_required_asset = 0.0
    for asset in assets:
        ratios = {}
        for key in out:
            symbols = _normalized_symbols(records.get(key) or {})
            if not symbols:
                ratios[key] = 0.0
            elif asset in symbols:
                ratios[key] = 1.0 / float(len(symbols))
            else:
                ratios[key] = 0.0
        min_required_asset = max(
            min_required_asset,
            _min_required_exposure(ratios, strategy_cap=strategy_cap),
        )
    asset_cap = max(float(max_asset), float(min_required_asset))

    metal_ratios: dict[str, float] = {}
    for key in out:
        symbols = _normalized_symbols(records.get(key) or {})
        if not symbols:
            metal_ratios[key] = 0.0
            continue
        metals_count = sum(1 for symbol in symbols if symbol in _METALS)
        metal_ratios[key] = float(metals_count) / float(len(symbols))
    metals_cap = max(float(max_metals), _min_required_exposure(metal_ratios, strategy_cap=strategy_cap))

    upper = dict.fromkeys(out, strategy_cap)
    out = _project_simplex_with_upper_bounds(out, upper=upper, target_sum=1.0)

    for _ in range(24):
        changed = False

        # Family caps.
        family_sums: dict[str, float] = defaultdict(float)
        for key, weight in out.items():
            family_sums[families[key]] += float(weight)
        for family, total in family_sums.items():
            limit = float(family_caps.get(family, 1.0))
            if total <= limit + 1e-9:
                continue
            scale = limit / max(1e-12, total)
            for key in out:
                if families[key] == family:
                    out[key] *= scale
            changed = True

        # Asset caps (exposure share).
        for asset in assets:
            exposure = _asset_exposure(out, records, asset)
            if exposure <= asset_cap + 1e-9:
                continue
            scale = asset_cap / max(1e-12, exposure)
            for key in out:
                symbols = _normalized_symbols(records.get(key) or {})
                if asset in symbols:
                    out[key] *= scale
            changed = True

        # Metals combined cap (share exposure).
        metal_exposure = _metals_exposure(out, records)
        if metal_exposure > metals_cap + 1e-9:
            scale = metals_cap / max(1e-12, metal_exposure)
            for key in out:
                if metal_ratios.get(key, 0.0) > 0.0:
                    out[key] *= scale
            changed = True

        out = _project_simplex_with_upper_bounds(out, upper=upper, target_sum=1.0)
        if not changed:
            break

    return out, {
        "max_strategy": float(strategy_cap),
        "max_family": float(max(family_caps.values()) if family_caps else max_family),
        "max_asset": float(asset_cap),
        "max_metals": float(metals_cap),
        "family_caps": {key: float(value) for key, value in sorted(family_caps.items())},
    }


def _build_portfolio_returns(
    weights: dict[str, float],
    rows: dict[str, dict[str, Any]],
    *,
    split: str,
) -> np.ndarray:
    arrays: list[tuple[float, np.ndarray]] = []
    min_len = None
    for cid, weight in weights.items():
        stream = list(((rows.get(cid) or {}).get("return_streams") or {}).get(split) or [])
        arr = _stream_to_array(stream)
        if arr.size == 0 or weight <= 0.0:
            continue
        arrays.append((float(weight), arr))
        min_len = arr.size if min_len is None else min(min_len, arr.size)

    if not arrays or min_len is None or min_len <= 0:
        return np.asarray([], dtype=float)

    out = np.zeros(min_len, dtype=float)
    for weight, arr in arrays:
        out += weight * arr[-min_len:]
    return out


def _load_rows(args) -> tuple[list[dict[str, Any]], str]:
    if args.team_report and Path(args.team_report).exists():
        payload = json.loads(Path(args.team_report).read_text(encoding="utf-8"))
        rows = list(payload.get("selected_team") or [])
        source = str(Path(args.team_report).resolve())
        if rows:
            return [dict(row) for row in rows if isinstance(row, dict)], source

    payload = json.loads(Path(args.research_report).read_text(encoding="utf-8"))
    rows = list(payload.get("candidates") or [])
    source = str(Path(args.research_report).resolve())
    return [dict(row) for row in rows if isinstance(row, dict)], source


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize shortlisted strategy portfolio.")
    parser.add_argument("--research-report", default="reports/candidate_research_latest.json")
    parser.add_argument("--team-report", default="reports/strategy_factory_report_latest.json")
    parser.add_argument("--score-config", default="", help="Optional scoring config JSON path.")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--max-strategies", type=int, default=24)
    parser.add_argument("--target-vol", type=float, default=0.12)
    parser.add_argument("--correlation-threshold", type=float, default=0.60)
    parser.add_argument("--cost-penalty", type=float, default=0.35)
    parser.add_argument("--max-strategy-cap", type=float, default=None)
    parser.add_argument("--max-family-cap", type=float, default=None)
    parser.add_argument("--max-asset-cap", type=float, default=None)
    parser.add_argument("--max-metals-cap", type=float, default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    score_config_payload: dict[str, Any] | None = None
    score_config_path = None
    if str(args.score_config).strip():
        score_config_path = Path(str(args.score_config)).resolve()
        try:
            score_config_payload = _load_score_config(score_config_path)
        except ValueError as exc:
            raise SystemExit(f"[PORTFOLIO] {exc}")
    optimization_config = _resolve_portfolio_score_config(score_config_payload)
    rank_weights = dict(optimization_config.get("candidate_rank_score_weights") or {})
    allocation_quality_params = dict(optimization_config.get("allocation_quality_params") or {})
    vol_targeting_params = dict(optimization_config.get("vol_targeting") or {})
    sensitivity_params = dict(optimization_config.get("sensitivity") or {})
    constraint_defaults = dict(optimization_config.get("constraints") or {})

    rows_raw, source_path = _load_rows(args)
    if not rows_raw:
        raise RuntimeError("No candidate rows available for optimization.")

    # Filter by pass + non-empty OOS stream.
    filtered = [
        row
        for row in rows_raw
        if bool(row.get("pass", True))
        and list(((row.get("return_streams") or {}).get("oos")) or [])
    ]
    if not filtered:
        filtered = rows_raw

    # Rank by robust OOS quality.
    def _score(row: dict[str, Any]) -> float:
        oos = dict(row.get("oos") or {})
        return float(
            (_safe_float(rank_weights.get("sharpe_weight"), 2.8) * _safe_float(oos.get("sharpe"), 0.0))
            + (
                _safe_float(rank_weights.get("deflated_sharpe_weight"), 1.5)
                * _safe_float(oos.get("deflated_sharpe"), 0.0)
            )
            - (_safe_float(rank_weights.get("pbo_penalty"), 2.0) * _safe_float(oos.get("pbo"), 1.0))
            + (_safe_float(rank_weights.get("return_weight"), 25.0) * _safe_float(oos.get("return"), 0.0))
        )

    filtered.sort(key=_score, reverse=True)
    selected = filtered[: max(1, int(args.max_strategies))]

    rows = {str(row.get("candidate_id") or row.get("name")): row for row in selected}
    ids = list(rows.keys())
    oos_map = {
        cid: _stream_to_array(list((rows[cid].get("return_streams") or {}).get("oos") or []))
        for cid in ids
    }

    # 1) Correlation clustering.
    clusters = _cluster_by_correlation(ids, oos_map, threshold=float(args.correlation_threshold))

    # 2) HRP-like allocation (inverse-vol cluster + member weights with turnover penalty).
    cluster_weight_raw: dict[int, float] = {}
    member_weight_raw: dict[str, float] = {}
    for c_idx, cluster in enumerate(clusters):
        cluster_rets = [oos_map[cid] for cid in cluster if oos_map[cid].size > 0]
        min_len = min((arr.size for arr in cluster_rets), default=0)
        if min_len <= 0:
            cluster_weight_raw[c_idx] = 1.0
            for cid in cluster:
                member_weight_raw[cid] = 1.0 / max(1, len(cluster))
            continue

        cluster_port = np.zeros(min_len, dtype=float)
        invs = np.asarray([_inverse_vol_weight(arr[-min_len:]) for arr in cluster_rets], dtype=float)
        inv_sum = float(np.sum(invs))
        invs = invs / inv_sum if inv_sum > 0 else np.ones_like(invs) / len(invs)

        for arr_weight, arr in zip(invs, cluster_rets, strict=True):
            cluster_port += arr_weight * arr[-min_len:]

        cluster_weight_raw[c_idx] = _inverse_vol_weight(cluster_port)

        for cid in cluster:
            row = rows[cid]
            oos = dict(row.get("oos") or {})
            quality = max(
                _safe_float(allocation_quality_params.get("deflated_sharpe_floor"), 0.01),
                _safe_float(oos.get("deflated_sharpe"), 0.0)
                + _safe_float(allocation_quality_params.get("deflated_sharpe_offset"), 0.5),
            )
            inv_vol = _inverse_vol_weight(oos_map[cid])
            turnover = _safe_float(oos.get("turnover"), 0.0)
            penalty = 1.0 + (float(args.cost_penalty) * turnover)
            member_weight_raw[cid] = (quality * inv_vol) / max(1e-9, penalty)

    cluster_total = float(sum(cluster_weight_raw.values()))
    cluster_weights = {
        key: (value / cluster_total if cluster_total > 0 else 1.0 / max(1, len(cluster_weight_raw)))
        for key, value in cluster_weight_raw.items()
    }

    weights: dict[str, float] = {}
    for c_idx, cluster in enumerate(clusters):
        raw = np.asarray([member_weight_raw[cid] for cid in cluster], dtype=float)
        raw_sum = float(np.sum(raw))
        if raw_sum <= 0.0:
            raw = np.ones(len(cluster), dtype=float)
            raw_sum = float(len(cluster))
        normalized = raw / raw_sum
        for cid, w in zip(cluster, normalized, strict=True):
            weights[cid] = float(cluster_weights[c_idx] * w)

    # 3) Apply constraints and caps.
    configured_caps = {
        "max_strategy": _resolved_cli_or_config_float(
            args.max_strategy_cap,
            constraint_defaults.get("max_strategy"),
            default=0.15,
        ),
        "max_family": _resolved_cli_or_config_float(
            args.max_family_cap,
            constraint_defaults.get("max_family"),
            default=0.40,
        ),
        "max_asset": _resolved_cli_or_config_float(
            args.max_asset_cap,
            constraint_defaults.get("max_asset"),
            default=0.20,
        ),
        "max_metals": _resolved_cli_or_config_float(
            args.max_metals_cap,
            constraint_defaults.get("max_metals"),
            default=0.15,
        ),
    }
    weights, effective_caps = _apply_caps(
        weights,
        records=rows,
        max_strategy=configured_caps["max_strategy"],
        max_family=configured_caps["max_family"],
        max_asset=configured_caps["max_asset"],
        max_metals=configured_caps["max_metals"],
    )

    # 4) Vol targeting.
    target_vol_floor = max(0.0, _safe_float(vol_targeting_params.get("target_vol_floor"), 0.01))
    vol_scale_cap = max(0.0, _safe_float(vol_targeting_params.get("vol_scale_cap"), 2.0))
    vol_scale_epsilon = max(0.0, _safe_float(vol_targeting_params.get("vol_scale_epsilon"), 1e-12))
    portfolio_oos = _build_portfolio_returns(weights, rows, split="oos")
    oos_vol = _safe_float(np.std(portfolio_oos, ddof=1), 0.0)
    target_vol = max(target_vol_floor, float(args.target_vol))
    vol_scale = 1.0 if oos_vol <= vol_scale_epsilon else min(vol_scale_cap, target_vol / max(vol_scale_epsilon, oos_vol))
    for key in weights:
        weights[key] *= vol_scale

    # Re-normalize after vol scaling to keep full allocation.
    total = float(sum(weights.values()))
    if total > 0:
        for key in weights:
            weights[key] /= total

    # Portfolio metrics across splits.
    portfolio_train = _build_portfolio_returns(weights, rows, split="train")
    portfolio_val = _build_portfolio_returns(weights, rows, split="val")
    portfolio_oos = _build_portfolio_returns(weights, rows, split="oos")

    train_metrics = _metrics(portfolio_train)
    val_metrics = _metrics(portfolio_val)
    oos_metrics = _metrics(portfolio_oos)

    # Cost sensitivity (x2/x3).
    weighted_turnover = 0.0
    weighted_cost = 0.0
    for cid, weight in weights.items():
        row = rows[cid]
        oos = dict(row.get("oos") or {})
        cost = _safe_float(((row.get("metadata") or {}).get("cost_rate")), 0.0005)
        weighted_turnover += weight * _safe_float(oos.get("turnover"), 0.0)
        weighted_cost += weight * cost

    cost_stress_x2_multiplier = _safe_float(sensitivity_params.get("cost_stress_x2_multiplier"), 2.0)
    cost_stress_x3_multiplier = _safe_float(sensitivity_params.get("cost_stress_x3_multiplier"), 3.0)
    signal_drift_down_multiplier = _safe_float(sensitivity_params.get("signal_drift_down_multiplier"), 0.9)
    signal_drift_up_multiplier = _safe_float(sensitivity_params.get("signal_drift_up_multiplier"), 1.1)

    oos_x2 = portfolio_oos - (max(0.0, cost_stress_x2_multiplier - 1.0) * weighted_turnover * weighted_cost)
    oos_x3 = portfolio_oos - (max(0.0, cost_stress_x3_multiplier - 1.0) * weighted_turnover * weighted_cost)

    sensitivity = {
        "cost_stress": {
            "x2": _metrics(oos_x2),
            "x3": _metrics(oos_x3),
        },
        "param_drift": {
            "minus_10pct_signal": _metrics(portfolio_oos * signal_drift_down_multiplier),
            "plus_10pct_signal": _metrics(portfolio_oos * signal_drift_up_multiplier),
        },
    }

    # Sleeve budgets.
    sleeve_budget: dict[str, float] = defaultdict(float)
    for cid, weight in weights.items():
        sleeve = str(rows[cid].get("family") or "other")
        sleeve_budget[sleeve] += float(weight)

    ranked_weights = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    allocation_rows = []
    for cid, weight in ranked_weights:
        row = rows[cid]
        allocation_rows.append(
            {
                "candidate_id": cid,
                "name": row.get("name"),
                "strategy_class": row.get("strategy_class"),
                "family": row.get("family"),
                "symbols": list(row.get("symbols") or []),
                "timeframe": row.get("strategy_timeframe") or row.get("timeframe"),
                "weight": float(weight),
                "oos_sharpe": _safe_float((row.get("oos") or {}).get("sharpe"), 0.0),
                "oos_return": _safe_float((row.get("oos") or {}).get("return"), 0.0),
            }
        )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_report": source_path,
        "cluster_count": len(clusters),
        "clusters": clusters,
        "constraints": {
            "max_strategy": float(effective_caps.get("max_strategy", configured_caps["max_strategy"])),
            "max_family": float(effective_caps.get("max_family", configured_caps["max_family"])),
            "max_asset": float(effective_caps.get("max_asset", configured_caps["max_asset"])),
            "max_metals": float(effective_caps.get("max_metals", configured_caps["max_metals"])),
            "family_caps": dict(effective_caps.get("family_caps") or {}),
            "configured": configured_caps,
        },
        "scoring": {
            "candidate_rank_score_weights": {
                "sharpe_weight": _safe_float(rank_weights.get("sharpe_weight"), 2.8),
                "deflated_sharpe_weight": _safe_float(rank_weights.get("deflated_sharpe_weight"), 1.5),
                "pbo_penalty": _safe_float(rank_weights.get("pbo_penalty"), 2.0),
                "return_weight": _safe_float(rank_weights.get("return_weight"), 25.0),
            },
            "vol_targeting": {
                "target_vol_floor": float(target_vol_floor),
                "vol_scale_cap": float(vol_scale_cap),
                "vol_scale_epsilon": float(vol_scale_epsilon),
            },
            "sensitivity": {
                "cost_stress_x2_multiplier": float(cost_stress_x2_multiplier),
                "cost_stress_x3_multiplier": float(cost_stress_x3_multiplier),
                "signal_drift_down_multiplier": float(signal_drift_down_multiplier),
                "signal_drift_up_multiplier": float(signal_drift_up_multiplier),
            },
            "source": str(score_config_path) if score_config_path is not None else "",
        },
        "weights": allocation_rows,
        "sleeve_budget": dict(sorted(sleeve_budget.items())),
        "portfolio_metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "oos": oos_metrics,
        },
        "sensitivity": sensitivity,
    }

    json_path = output_dir / f"portfolio_optimization_{stamp}.json"
    json_latest = output_dir / "portfolio_optimization_latest.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    json_latest.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = output_dir / f"portfolio_optimization_{stamp}.md"
    lines = [
        "# Portfolio Optimization Report",
        "",
        f"- Source report: `{source_path}`",
        f"- Clusters: {len(clusters)}",
        "",
        "## Sleeve budgets",
        "",
    ]
    for family, weight in sorted(sleeve_budget.items()):
        lines.append(f"- {family}: {weight:.2%}")

    lines.extend(
        [
            "",
            "## Top strategy weights",
            "",
            "| # | Name | Strategy | Family | TF | Weight | OOS Sharpe | OOS Return |",
            "|---:|---|---|---|---|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(allocation_rows[:20], start=1):
        lines.append(
            "| "
            f"{idx} | {row.get('name', '')} | {row.get('strategy_class', '')} | {row.get('family', '')} | "
            f"{row.get('timeframe', '')} | {float(row.get('weight', 0.0)):.2%} | "
            f"{float(row.get('oos_sharpe', 0.0)):.3f} | {float(row.get('oos_return', 0.0)):.2%} |"
        )

    lines.extend(
        [
            "",
            "## Portfolio metrics",
            "",
            "- Train: " + json.dumps(train_metrics, sort_keys=True),
            "- Val: " + json.dumps(val_metrics, sort_keys=True),
            "- OOS: " + json.dumps(oos_metrics, sort_keys=True),
            "",
            "## Sensitivity",
            "",
            "```json",
            json.dumps(sensitivity, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {json_path}")
    print(f"Saved latest: {json_latest}")
    print(f"Saved markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
