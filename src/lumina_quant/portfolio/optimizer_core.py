"""Shared portfolio optimizer helpers.

The script-level optimizers keep ownership of CLI/reporting behavior.  This
module owns the reusable, allocation-hot-path pieces that must stay consistent
across portfolio tuning, Optuna, and validator-facing reports.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import numpy as np

LOCKED_OOS_OBJECTIVE_POLICY = "train_val_only_locked_oos_report"
DIAGNOSTIC_OOS_OBJECTIVE_POLICY = "diagnostic_oos_in_objective_not_selection_authority"
LOCKED_OOS_LABEL = "locked_oos_report_only"
METALS = {"XAU/USDT", "XAG/USDT", "XPT/USDT", "XPD/USDT"}

AggregatedStream = dict[str, dict[str, Any]]


class PortfolioConstraintInfeasibleError(RuntimeError):
    """Raised when configured portfolio caps cannot be satisfied."""

    def __init__(self, message: str, *, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = dict(details)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def canonical_split(value: Any, *, default: str) -> str:
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


def coerce_stream_datetime(value: Any) -> datetime | None:
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


def isoformat_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def stream_point_timestamp(raw_point: dict[str, Any]) -> Any:
    return raw_point.get("t", raw_point.get("timestamp", raw_point.get("datetime")))


def normalize_stream(stream: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, raw_point in enumerate(stream):
        value = safe_float(raw_point.get("v"), 0.0)
        raw_timestamp = stream_point_timestamp(raw_point)
        dt = coerce_stream_datetime(raw_timestamp)
        if dt is not None:
            timestamp = isoformat_utc(dt)
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


def aggregate_stream(stream: list[dict[str, Any]]) -> AggregatedStream:
    aggregated: AggregatedStream = {}
    for point in normalize_stream(stream):
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


def stream_to_array(stream: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([safe_float(item.get("v"), 0.0) for item in stream], dtype=float)


def split_metrics(row: dict[str, Any], split: str) -> dict[str, Any]:
    token = canonical_split(split, default="oos")
    if token == "val":
        return dict(row.get("val") or row.get("validation") or {})
    if token == "train":
        return dict(row.get("train") or {})
    return dict(row.get("oos") or {})


def split_stream(row: dict[str, Any], split: str) -> list[dict[str, Any]]:
    token = canonical_split(split, default="oos")
    streams = row.get("return_streams") or {}
    raw_key = str(split or "").strip()
    aliases = {
        "train": ("train",),
        "val": ("val", "validation"),
        "oos": ("oos", "test", "test_oos"),
    }
    for key in (*aliases.get(token, (token,)), raw_key):
        if key and streams.get(key):
            return list(streams.get(key) or [])
    return []


def _as_aggregated(stream: list[dict[str, Any]] | AggregatedStream) -> AggregatedStream:
    if isinstance(stream, dict):
        return stream
    return aggregate_stream(stream)


def aligned_stream_arrays(
    lhs_stream: list[dict[str, Any]] | AggregatedStream,
    rhs_stream: list[dict[str, Any]] | AggregatedStream,
) -> tuple[np.ndarray, np.ndarray]:
    lhs = _as_aggregated(lhs_stream)
    rhs = _as_aggregated(rhs_stream)
    if not lhs and not rhs:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    merged = dict(lhs)
    for token, point in rhs.items():
        if token not in merged:
            merged[token] = point

    ordered_tokens = [
        token for token, _ in sorted(merged.items(), key=lambda item: item[1]["sort_key"])
    ]
    lhs_values = np.asarray(
        [safe_float((lhs.get(token) or {}).get("v"), 0.0) for token in ordered_tokens],
        dtype=float,
    )
    rhs_values = np.asarray(
        [safe_float((rhs.get(token) or {}).get("v"), 0.0) for token in ordered_tokens],
        dtype=float,
    )
    return lhs_values, rhs_values


def corr_arrays(x: np.ndarray, y: np.ndarray) -> float:
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


def corr_streams(
    lhs_stream: list[dict[str, Any]] | AggregatedStream,
    rhs_stream: list[dict[str, Any]] | AggregatedStream,
) -> float:
    lhs, rhs = aligned_stream_arrays(lhs_stream, rhs_stream)
    return corr_arrays(lhs, rhs)


def max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    dd = 1.0 - np.divide(equity, np.maximum(peaks, 1e-12))
    return float(np.max(dd)) if dd.size else 0.0


def metrics(returns: np.ndarray, periods_per_year: int = 365) -> dict[str, float]:
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

    mdd = max_drawdown(returns)
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


class StreamCache:
    """Cache normalized stream payloads for repeated portfolio combinations."""

    def __init__(self) -> None:
        self._aggregates: dict[tuple[str, str], AggregatedStream] = {}
        self._arrays: dict[tuple[str, str], np.ndarray] = {}

    @staticmethod
    def _row_key(row: dict[str, Any]) -> str:
        key = row.get("candidate_id") or row.get("name")
        if key is not None and str(key).strip():
            return str(key)
        return f"row:{id(row)}"

    def aggregate_for_row(self, row: dict[str, Any], split: str) -> AggregatedStream:
        token = canonical_split(split, default="oos")
        key = (self._row_key(row), token)
        cached = self._aggregates.get(key)
        if cached is None:
            cached = aggregate_stream(split_stream(row, split))
            self._aggregates[key] = cached
        return cached

    def array_for_row(self, row: dict[str, Any], split: str) -> np.ndarray:
        token = canonical_split(split, default="oos")
        key = (self._row_key(row), token)
        cached = self._arrays.get(key)
        if cached is None:
            aggregate = self.aggregate_for_row(row, token)
            ordered = sorted(aggregate.values(), key=lambda item: item["sort_key"])
            cached = np.asarray([safe_float(point.get("v"), 0.0) for point in ordered], dtype=float)
            self._arrays[key] = cached
        return cached


def build_portfolio_stream(
    weights: dict[str, float],
    rows: dict[str, dict[str, Any]],
    *,
    split: str,
    cache: StreamCache | None = None,
) -> list[dict[str, Any]]:
    stream_cache = cache or StreamCache()
    aggregated: AggregatedStream = {}
    for cid, weight in weights.items():
        if weight <= 0.0:
            continue
        row = rows.get(cid) or {}
        for token, point in stream_cache.aggregate_for_row(row, split).items():
            if token not in aggregated:
                aggregated[token] = {
                    "token": token,
                    "sort_key": point["sort_key"],
                    "t": point["t"],
                    "datetime": point.get("datetime"),
                    "v": 0.0,
                }
            aggregated[token]["v"] += float(weight) * safe_float(point.get("v"), 0.0)

    out: list[dict[str, Any]] = []
    for point in sorted(aggregated.values(), key=lambda item: item["sort_key"]):
        row = {"t": point["t"], "v": float(point["v"])}
        if point.get("datetime"):
            row["datetime"] = point["datetime"]
        out.append(row)
    return out


def build_portfolio_returns(
    weights: dict[str, float],
    rows: dict[str, dict[str, Any]],
    *,
    split: str,
    cache: StreamCache | None = None,
) -> np.ndarray:
    return stream_to_array(build_portfolio_stream(weights, rows, split=split, cache=cache))


def cluster_by_correlation(
    ids: list[str],
    stream_map: dict[str, list[dict[str, Any]] | AggregatedStream],
    threshold: float = 0.60,
) -> list[list[str]]:
    clusters: list[list[str]] = []
    for cid in ids:
        added = False
        for cluster in clusters:
            if any(
                abs(corr_streams(stream_map[cid], stream_map[member])) >= threshold
                for member in cluster
            ):
                cluster.append(cid)
                added = True
                break
        if not added:
            clusters.append([cid])
    return clusters


def objective_policy_payload(profile: str, *, oos_is_objective_input: bool) -> dict[str, Any]:
    policy = (
        DIAGNOSTIC_OOS_OBJECTIVE_POLICY if oos_is_objective_input else LOCKED_OOS_OBJECTIVE_POLICY
    )
    return {
        "objective_profile": str(profile),
        "objective_policy": policy,
        "selection_label": "diagnostic_oos_in_objective"
        if oos_is_objective_input
        else "train_val_validation_only",
        "locked_oos_label": LOCKED_OOS_LABEL,
        "oos_is_objective_input": bool(oos_is_objective_input),
    }


def normalized_symbols(record: dict[str, Any]) -> list[str]:
    return sorted(
        {str(symbol).strip() for symbol in list(record.get("symbols") or []) if str(symbol).strip()}
    )


def project_simplex_with_upper_bounds(
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


def asset_exposure(
    weights: dict[str, float], records: dict[str, dict[str, Any]], asset: str
) -> float:
    exposure = 0.0
    token = str(asset)
    for key, weight in weights.items():
        symbols = normalized_symbols(records.get(key) or {})
        if not symbols or token not in symbols:
            continue
        exposure += float(weight) / float(len(symbols))
    return exposure


def metals_exposure(weights: dict[str, float], records: dict[str, dict[str, Any]]) -> float:
    exposure = 0.0
    for key, weight in weights.items():
        symbols = normalized_symbols(records.get(key) or {})
        if not symbols:
            continue
        metals_count = sum(1 for symbol in symbols if symbol in METALS)
        if metals_count <= 0:
            continue
        exposure += float(weight) * (float(metals_count) / float(len(symbols)))
    return exposure


def portfolio_constraint_violations(
    weights: dict[str, float],
    *,
    records: dict[str, dict[str, Any]],
    max_strategy: float,
    max_family: float,
    max_asset: float,
    max_metals: float,
) -> dict[str, Any]:
    strategy_violations = [
        {
            "candidate_id": key,
            "weight": float(weight),
            "max_strategy": float(max_strategy),
        }
        for key, weight in sorted(weights.items())
        if float(weight) > float(max_strategy) + 1e-9
    ]

    family_weights: dict[str, float] = defaultdict(float)
    for key, weight in weights.items():
        family = str((records.get(key) or {}).get("family", "other"))
        family_weights[family] += float(weight)
    family_violations = {
        family: {
            "weight": float(weight),
            "max_family": float(max_family),
        }
        for family, weight in sorted(family_weights.items())
        if float(weight) > float(max_family) + 1e-9
    }

    assets = sorted(
        {symbol for key in weights for symbol in normalized_symbols(records.get(key) or {})}
    )
    asset_violations = {}
    for asset in assets:
        exposure = asset_exposure(weights, records, asset)
        if exposure <= float(max_asset) + 1e-9:
            continue
        asset_violations[asset] = {
            "exposure": float(exposure),
            "max_asset": float(max_asset),
        }

    metal_exposure = metals_exposure(weights, records)
    metals_violation = None
    if metal_exposure > float(max_metals) + 1e-9:
        metals_violation = {
            "exposure": float(metal_exposure),
            "max_metals": float(max_metals),
        }

    violations: dict[str, Any] = {}
    if strategy_violations:
        violations["strategy"] = strategy_violations
    if family_violations:
        violations["family"] = family_violations
    if asset_violations:
        violations["asset"] = asset_violations
    if metals_violation is not None:
        violations["metals"] = metals_violation
    return violations


def apply_caps(
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
            "target_active_weight": 0.0,
            "active_weight": 0.0,
            "cash_reserve_weight": 1.0,
            "family_caps": {},
        }

    out = {key: max(0.0, float(value)) for key, value in weights.items()}
    n = len(out)
    strategy_cap = max(0.0, float(max_strategy))
    family_cap = max(0.0, float(max_family))
    asset_cap = max(0.0, float(max_asset))
    metals_cap = max(0.0, float(max_metals))

    families = {key: str((records.get(key) or {}).get("family", "other")) for key in out}
    family_counts: dict[str, int] = defaultdict(int)
    for fam in families.values():
        family_counts[fam] += 1

    family_caps = dict.fromkeys(family_counts, family_cap)
    family_capacity = sum(
        min(float(count) * strategy_cap, family_cap) for count in family_counts.values()
    )
    target_active_weight = min(1.0, float(n) * strategy_cap, family_capacity)

    assets = sorted(
        {symbol for key in out for symbol in normalized_symbols(records.get(key) or {})}
    )
    metal_ratios: dict[str, float] = {}
    for key in out:
        symbols = normalized_symbols(records.get(key) or {})
        if not symbols:
            metal_ratios[key] = 0.0
            continue
        metals_count = sum(1 for symbol in symbols if symbol in METALS)
        metal_ratios[key] = float(metals_count) / float(len(symbols))

    upper = dict.fromkeys(out, strategy_cap)
    out = project_simplex_with_upper_bounds(out, upper=upper, target_sum=target_active_weight)

    for _ in range(24):
        changed = False

        family_sums: dict[str, float] = defaultdict(float)
        for key, weight in out.items():
            family_sums[families[key]] += float(weight)
        for family, total in family_sums.items():
            limit = float(family_caps.get(family, family_cap))
            if total <= limit + 1e-9:
                continue
            scale = limit / max(1e-12, total)
            for key in out:
                if families[key] == family:
                    out[key] *= scale
            changed = True

        for asset in assets:
            exposure = asset_exposure(out, records, asset)
            if exposure <= asset_cap + 1e-9:
                continue
            scale = asset_cap / max(1e-12, exposure)
            for key in out:
                symbols = normalized_symbols(records.get(key) or {})
                if asset in symbols:
                    out[key] *= scale
            changed = True

        metal_exposure = metals_exposure(out, records)
        if metal_exposure > metals_cap + 1e-9:
            scale = metals_cap / max(1e-12, metal_exposure)
            for key in out:
                if metal_ratios.get(key, 0.0) > 0.0:
                    out[key] *= scale
            changed = True

        if not changed:
            break

    violations = portfolio_constraint_violations(
        out,
        records=records,
        max_strategy=strategy_cap,
        max_family=family_cap,
        max_asset=asset_cap,
        max_metals=metals_cap,
    )
    if violations:
        target_active_weight = float(sum(out.values()))
        out = project_simplex_with_upper_bounds(out, upper=upper, target_sum=target_active_weight)
        violations = portfolio_constraint_violations(
            out,
            records=records,
            max_strategy=strategy_cap,
            max_family=family_cap,
            max_asset=asset_cap,
            max_metals=metals_cap,
        )
        if violations:
            raise PortfolioConstraintInfeasibleError(
                "configured diversification caps are infeasible for the shortlisted candidates",
                details={
                    "configured": {
                        "max_strategy": float(strategy_cap),
                        "max_family": float(family_cap),
                        "max_asset": float(asset_cap),
                        "max_metals": float(metals_cap),
                    },
                    "violations": violations,
                },
            )

    active_weight = float(sum(out.values()))
    return out, {
        "max_strategy": float(strategy_cap),
        "max_family": float(family_cap),
        "max_asset": float(asset_cap),
        "max_metals": float(metals_cap),
        "target_active_weight": float(target_active_weight),
        "active_weight": active_weight,
        "cash_reserve_weight": max(0.0, 1.0 - active_weight),
        "family_caps": {key: float(value) for key, value in sorted(family_caps.items())},
    }
