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

_METALS = {"XAU/USDT", "XAG/USDT", "XPT/USDT", "XPD/USDT"}

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


def _canonical_split(value: Any, *, default: str) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return default
    alias_map = {
        "train": "train",
        "validation": "val",
        "val": "val",
        "oos": "oos",
        "test": "oos",
        "test_oos": "oos",
    }
    return alias_map.get(token, default)


def _coerce_stream_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    numeric_value: float | None = None
    if isinstance(value, (int, float)):
        numeric_value = float(value)
    elif isinstance(value, str) and value.strip():
        token = value.strip()
        try:
            numeric_value = float(token)
        except Exception:
            normalized = token.replace("Z", "+00:00") if token.endswith("Z") else token
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)

    if numeric_value is None or not math.isfinite(numeric_value):
        return None

    magnitude = abs(numeric_value)
    if magnitude >= 1e15:
        return datetime.fromtimestamp(numeric_value / 1_000_000.0, tz=UTC)
    if magnitude >= 1e12:
        return datetime.fromtimestamp(numeric_value / 1_000.0, tz=UTC)
    if magnitude >= 1e9:
        return datetime.fromtimestamp(numeric_value, tz=UTC)
    return None


def _isoformat_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _stream_point_timestamp(raw_point: dict[str, Any]) -> Any:
    return raw_point.get("t", raw_point.get("timestamp", raw_point.get("datetime")))


def _normalize_stream(stream: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, raw_point in enumerate(stream):
        value = _safe_float(raw_point.get("v"), 0.0)
        raw_timestamp = _stream_point_timestamp(raw_point)
        dt = _coerce_stream_datetime(raw_timestamp)
        if dt is not None:
            timestamp = _isoformat_utc(dt)
            normalized.append(
                {
                    "token": f"dt:{timestamp}",
                    "sort_key": (0, float(dt.timestamp()), float(idx)),
                    "t": timestamp,
                    "datetime": timestamp,
                    "v": float(value),
                }
            )
            continue

        numeric_timestamp = None
        if isinstance(raw_timestamp, (int, float)):
            numeric_timestamp = float(raw_timestamp)
        elif isinstance(raw_timestamp, str) and raw_timestamp.strip():
            try:
                numeric_timestamp = float(raw_timestamp.strip())
            except Exception:
                numeric_timestamp = None

        if numeric_timestamp is not None and math.isfinite(numeric_timestamp):
            normalized.append(
                {
                    "token": f"num:{numeric_timestamp:.12g}",
                    "sort_key": (1, float(numeric_timestamp), float(idx)),
                    "t": float(numeric_timestamp),
                    "datetime": None,
                    "v": float(value),
                }
            )
            continue

        seq = float(idx)
        normalized.append(
            {
                "token": f"seq:{seq:.12g}",
                "sort_key": (2, float(seq), float(idx)),
                "t": float(seq),
                "datetime": None,
                "v": float(value),
            }
        )
    normalized.sort(key=lambda item: item["sort_key"])
    return normalized


def _aggregate_stream(stream: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
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
    return aggregated


def _aligned_stream_arrays(lhs_stream: list[dict[str, Any]], rhs_stream: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    lhs = _aggregate_stream(lhs_stream)
    rhs = _aggregate_stream(rhs_stream)
    if not lhs and not rhs:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    merged = dict(lhs)
    for token, point in rhs.items():
        if token not in merged:
            merged[token] = point

    ordered_tokens = [token for token, _ in sorted(merged.items(), key=lambda item: item[1]["sort_key"])]
    lhs_values = np.asarray([_safe_float((lhs.get(token) or {}).get("v"), 0.0) for token in ordered_tokens], dtype=float)
    rhs_values = np.asarray([_safe_float((rhs.get(token) or {}).get("v"), 0.0) for token in ordered_tokens], dtype=float)
    return lhs_values, rhs_values


def _split_metrics(row: dict[str, Any], split: str) -> dict[str, Any]:
    token = _canonical_split(split, default="oos")
    if token == "val":
        return dict(row.get("val") or {})
    if token == "train":
        return dict(row.get("train") or {})
    return dict(row.get("oos") or {})


def _split_stream(row: dict[str, Any], split: str) -> list[dict[str, Any]]:
    token = _canonical_split(split, default="oos")
    return list(((row.get("return_streams") or {}).get(token)) or [])


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


def _corr_streams(lhs_stream: list[dict[str, Any]], rhs_stream: list[dict[str, Any]]) -> float:
    lhs, rhs = _aligned_stream_arrays(lhs_stream, rhs_stream)
    return _corr(lhs, rhs)


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


def _cluster_by_correlation(
    ids: list[str],
    stream_map: dict[str, list[dict[str, Any]]],
    threshold: float = 0.60,
) -> list[list[str]]:
    clusters: list[list[str]] = []
    for cid in ids:
        added = False
        for cluster in clusters:
            if any(abs(_corr_streams(stream_map[cid], stream_map[member])) >= threshold for member in cluster):
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


def _build_portfolio_stream(
    weights: dict[str, float],
    rows: dict[str, dict[str, Any]],
    *,
    split: str,
) -> list[dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for cid, weight in weights.items():
        stream = _split_stream(rows.get(cid) or {}, split)
        if not stream or weight <= 0.0:
            continue
        for token, point in _aggregate_stream(stream).items():
            if token not in aggregated:
                aggregated[token] = {
                    "token": token,
                    "sort_key": point["sort_key"],
                    "t": point["t"],
                    "datetime": point.get("datetime"),
                    "v": 0.0,
                }
            aggregated[token]["v"] += float(weight) * _safe_float(point.get("v"), 0.0)

    out: list[dict[str, Any]] = []
    for point in sorted(aggregated.values(), key=lambda item: item["sort_key"]):
        row = {"t": point["t"], "v": float(point["v"])}
        if point.get("datetime"):
            row["datetime"] = point["datetime"]
        out.append(row)
    return out


def _build_portfolio_returns(
    weights: dict[str, float],
    rows: dict[str, dict[str, Any]],
    *,
    split: str,
) -> np.ndarray:
    return _stream_to_array(_build_portfolio_stream(weights, rows, split=split))


def _load_rows(args) -> tuple[list[dict[str, Any]], str]:
    research_path = Path(args.research_report)
    if research_path.exists():
        payload = json.loads(research_path.read_text(encoding="utf-8"))
        rows = [dict(row) for row in list(payload.get("candidates") or []) if isinstance(row, dict)]
        if rows:
            return rows, str(research_path.resolve())

    if args.team_report and Path(args.team_report).exists():
        payload = json.loads(Path(args.team_report).read_text(encoding="utf-8"))
        rows = list(payload.get("selected_team") or [])
        source = str(Path(args.team_report).resolve())
        if rows:
            return [dict(row) for row in rows if isinstance(row, dict)], source

    raise RuntimeError("No candidate rows available in research report or team report.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize shortlisted strategy portfolio.")
    parser.add_argument("--research-report", default="reports/candidate_research_latest.json")
    parser.add_argument("--team-report", default="reports/strategy_factory_report_latest.json")
    parser.add_argument("--score-config", default="", help="Optional scoring config JSON path.")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--max-strategies", type=int, default=24)
    parser.add_argument("--fit-split", default="val")
    parser.add_argument("--report-split", default="oos")
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
    fit_split = _canonical_split(args.fit_split, default="val")
    report_split = _canonical_split(args.report_split, default="oos")

    rows_raw, source_path = _load_rows(args)
    if not rows_raw:
        raise RuntimeError("No candidate rows available for optimization.")

    # Filter by pass + non-empty fit/report streams.
    filtered = [
        row
        for row in rows_raw
        if bool(row.get("pass", True))
        and _split_stream(row, fit_split)
        and _split_stream(row, report_split)
    ]
    if not filtered:
        filtered = [
            row
            for row in rows_raw
            if _split_stream(row, fit_split) or _split_stream(row, report_split)
        ] or rows_raw

    # Rank by robust fit-split quality.
    def _score(row: dict[str, Any]) -> float:
        fit_metrics = _split_metrics(row, fit_split)
        return float(
            (_safe_float(rank_weights.get("sharpe_weight"), 2.8) * _safe_float(fit_metrics.get("sharpe"), 0.0))
            + (
                _safe_float(rank_weights.get("deflated_sharpe_weight"), 1.5)
                * _safe_float(fit_metrics.get("deflated_sharpe"), 0.0)
            )
            - (_safe_float(rank_weights.get("pbo_penalty"), 2.0) * _safe_float(fit_metrics.get("pbo"), 1.0))
            + (_safe_float(rank_weights.get("return_weight"), 25.0) * _safe_float(fit_metrics.get("return"), 0.0))
        )

    filtered.sort(key=_score, reverse=True)
    selected = filtered[: max(1, int(args.max_strategies))]

    rows = {str(row.get("candidate_id") or row.get("name")): row for row in selected}
    ids = list(rows.keys())
    fit_streams = {cid: _split_stream(rows[cid], fit_split) for cid in ids}
    fit_map = {cid: _stream_to_array(fit_streams[cid]) for cid in ids}

    # 1) Correlation clustering.
    clusters = _cluster_by_correlation(ids, fit_streams, threshold=float(args.correlation_threshold))

    # 2) HRP-like allocation (inverse-vol cluster + member weights with turnover penalty).
    cluster_weight_raw: dict[int, float] = {}
    member_weight_raw: dict[str, float] = {}
    for c_idx, cluster in enumerate(clusters):
        cluster_rets = [fit_map[cid] for cid in cluster if fit_map[cid].size > 0]
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
            fit_metrics = _split_metrics(row, fit_split)
            quality = max(
                _safe_float(allocation_quality_params.get("deflated_sharpe_floor"), 0.01),
                _safe_float(fit_metrics.get("deflated_sharpe"), 0.0)
                + _safe_float(allocation_quality_params.get("deflated_sharpe_offset"), 0.5),
            )
            inv_vol = _inverse_vol_weight(fit_map[cid])
            turnover = _safe_float(fit_metrics.get("turnover"), 0.0)
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
    portfolio_fit = _build_portfolio_returns(weights, rows, split=fit_split)
    fit_vol = _safe_float(np.std(portfolio_fit, ddof=1), 0.0)
    target_vol = max(target_vol_floor, float(args.target_vol))
    vol_scale = 1.0 if fit_vol <= vol_scale_epsilon else min(vol_scale_cap, target_vol / max(vol_scale_epsilon, fit_vol))
    for key in weights:
        weights[key] *= vol_scale

    # Re-normalize after vol scaling to keep full allocation.
    total = float(sum(weights.values()))
    if total > 0:
        for key in weights:
            weights[key] /= total

    # Portfolio metrics across splits.
    portfolio_train_stream = _build_portfolio_stream(weights, rows, split="train")
    portfolio_val_stream = _build_portfolio_stream(weights, rows, split="val")
    portfolio_oos_stream = _build_portfolio_stream(weights, rows, split="oos")
    portfolio_fit_stream = _build_portfolio_stream(weights, rows, split=fit_split)
    portfolio_report_stream = _build_portfolio_stream(weights, rows, split=report_split)

    portfolio_train = _stream_to_array(portfolio_train_stream)
    portfolio_val = _stream_to_array(portfolio_val_stream)
    portfolio_oos = _stream_to_array(portfolio_oos_stream)
    portfolio_fit = _stream_to_array(portfolio_fit_stream)
    portfolio_report = _stream_to_array(portfolio_report_stream)

    train_metrics = _metrics(portfolio_train)
    val_metrics = _metrics(portfolio_val)
    oos_metrics = _metrics(portfolio_oos)
    fit_metrics = _metrics(portfolio_fit)
    report_metrics = _metrics(portfolio_report)

    # Cost sensitivity (x2/x3).
    weighted_turnover = 0.0
    weighted_cost = 0.0
    for cid, weight in weights.items():
        row = rows[cid]
        report_row_metrics = _split_metrics(row, report_split)
        cost = _safe_float(((row.get("metadata") or {}).get("cost_rate")), 0.0005)
        weighted_turnover += weight * _safe_float(report_row_metrics.get("turnover"), 0.0)
        weighted_cost += weight * cost

    cost_stress_x2_multiplier = _safe_float(sensitivity_params.get("cost_stress_x2_multiplier"), 2.0)
    cost_stress_x3_multiplier = _safe_float(sensitivity_params.get("cost_stress_x3_multiplier"), 3.0)
    signal_drift_down_multiplier = _safe_float(sensitivity_params.get("signal_drift_down_multiplier"), 0.9)
    signal_drift_up_multiplier = _safe_float(sensitivity_params.get("signal_drift_up_multiplier"), 1.1)

    report_x2 = portfolio_report - (max(0.0, cost_stress_x2_multiplier - 1.0) * weighted_turnover * weighted_cost)
    report_x3 = portfolio_report - (max(0.0, cost_stress_x3_multiplier - 1.0) * weighted_turnover * weighted_cost)

    sensitivity = {
        "cost_stress": {
            "x2": _metrics(report_x2),
            "x3": _metrics(report_x3),
        },
        "param_drift": {
            "minus_10pct_signal": _metrics(portfolio_report * signal_drift_down_multiplier),
            "plus_10pct_signal": _metrics(portfolio_report * signal_drift_up_multiplier),
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
        fit_row_metrics = _split_metrics(row, fit_split)
        report_row_metrics = _split_metrics(row, report_split)
        allocation_rows.append(
            {
                "candidate_id": cid,
                "name": row.get("name"),
                "strategy_class": row.get("strategy_class"),
                "family": row.get("family"),
                "symbols": list(row.get("symbols") or []),
                "timeframe": row.get("strategy_timeframe") or row.get("timeframe"),
                "weight": float(weight),
                "fit_split": fit_split,
                "fit_sharpe": _safe_float(fit_row_metrics.get("sharpe"), 0.0),
                "fit_return": _safe_float(fit_row_metrics.get("return"), 0.0),
                "report_split": report_split,
                "report_sharpe": _safe_float(report_row_metrics.get("sharpe"), 0.0),
                "report_return": _safe_float(report_row_metrics.get("return"), 0.0),
                "oos_sharpe": _safe_float((row.get("oos") or {}).get("sharpe"), 0.0),
                "oos_return": _safe_float((row.get("oos") or {}).get("return"), 0.0),
            }
        )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_report": source_path,
        "selection": {
            "fit_split": fit_split,
            "report_split": report_split,
            "selection_basis": "validation_only" if fit_split == "val" and report_split == "oos" else f"{fit_split}_fit",
        },
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
        "portfolio_return_streams": {
            "train": portfolio_train_stream,
            "val": portfolio_val_stream,
            "oos": portfolio_oos_stream,
            fit_split: portfolio_fit_stream,
            report_split: portfolio_report_stream,
        },
        "portfolio_metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "oos": oos_metrics,
            fit_split: fit_metrics,
            report_split: report_metrics,
        },
        "fit_metrics": fit_metrics,
        "report_metrics": report_metrics,
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
        f"- Fit split: `{fit_split}`",
        f"- Report split: `{report_split}`",
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
            "| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |",
            "|---:|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(allocation_rows[:20], start=1):
        lines.append(
            "| "
            f"{idx} | {row.get('name', '')} | {row.get('strategy_class', '')} | {row.get('family', '')} | "
            f"{row.get('timeframe', '')} | {float(row.get('weight', 0.0)):.2%} | "
            f"{float(row.get('fit_sharpe', 0.0)):.3f} | {float(row.get('fit_return', 0.0)):.2%} | "
            f"{float(row.get('report_sharpe', 0.0)):.3f} | {float(row.get('report_return', 0.0)):.2%} |"
        )

    lines.extend(
        [
            "",
            "## Portfolio metrics",
            "",
            f"- Fit ({fit_split}): " + json.dumps(fit_metrics, sort_keys=True),
            f"- Report ({report_split}): " + json.dumps(report_metrics, sort_keys=True),
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
