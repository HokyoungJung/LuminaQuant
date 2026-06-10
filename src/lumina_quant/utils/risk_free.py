"""Risk-free configuration and per-period conversion helpers."""

from __future__ import annotations

import csv
from bisect import bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class RiskFreePolicy:
    mode: str = "us_treasury_constant"
    tenor: str = "3m"
    annual: float = 0.0
    series_path: str = ""
    sortino_target_mode: str = "same_as_rf"
    sortino_target_annual: float = 0.0


@dataclass(slots=True)
class ResolvedRiskFree:
    mode: str
    tenor: str
    annual_rate: float
    per_period_rate: float
    periodic_rates: np.ndarray
    sortino_target_mode: str
    sortino_target_annual: float
    sortino_target_per_period: float
    periodic_sortino_targets: np.ndarray


_MISSING = object()


def _read_attr(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    # Try the canonical uppercase name first (BacktestConfigView / old metaclass classes).
    val = getattr(config, name, _MISSING)
    if val is not _MISSING:
        return val
    # Typed dataclass fields are lowercase — BacktestRuntimeConfig uses risk_free_annual etc.
    return getattr(config, name.lower(), default)


def risk_free_policy_from_config(config: Any | None = None) -> RiskFreePolicy:
    """Resolve risk-free policy from Backtest/BaseConfig-like objects."""
    annual = float(
        _read_attr(
            config,
            "RISK_FREE_ANNUAL",
            _read_attr(config, "RISK_FREE_RATE", 0.0),
        )
        or 0.0
    )
    return RiskFreePolicy(
        mode=str(
            _read_attr(config, "RISK_FREE_MODE", "us_treasury_constant") or "us_treasury_constant"
        )
        .strip()
        .lower(),
        tenor=str(_read_attr(config, "RISK_FREE_TENOR", "3m") or "3m").strip().lower(),
        annual=float(annual),
        series_path=str(_read_attr(config, "RISK_FREE_SERIES_PATH", "") or "").strip(),
        sortino_target_mode=str(
            _read_attr(config, "SORTINO_TARGET_MODE", "same_as_rf") or "same_as_rf"
        )
        .strip()
        .lower(),
        sortino_target_annual=float(_read_attr(config, "SORTINO_TARGET_ANNUAL", 0.0) or 0.0),
    )


def annual_to_periodic_rate(annual_rate: float, periods_per_year: int) -> float:
    """Convert annualized rate to per-period rate."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    return float((1.0 + float(annual_rate)) ** (1.0 / float(periods_per_year)) - 1.0)


def _coerce_timestamp_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return int(dt.astimezone(UTC).timestamp() * 1000)
    if isinstance(value, np.datetime64):
        return int(value.astype("datetime64[ms]").astype(np.int64))
    if isinstance(value, (int, float)):
        numeric = int(value)
        if abs(numeric) < 100_000_000_000:
            return numeric * 1000
        return numeric
    token = str(value).strip()
    if not token:
        return None
    try:
        parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.astimezone(UTC).timestamp() * 1000)


@lru_cache(maxsize=8)
def _load_series_points(path: str) -> tuple[list[int], list[float]]:
    series_path = Path(path).expanduser().resolve()
    if not series_path.exists():
        raise FileNotFoundError(f"Risk-free series path does not exist: {series_path}")

    timestamps: list[int] = []
    annual_rates: list[float] = []
    with series_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        field_map = {str(name or "").strip().lower(): name for name in reader.fieldnames or []}
        date_key = field_map.get("date") or field_map.get("datetime") or field_map.get("timestamp")
        rate_key = (
            field_map.get("annual")
            or field_map.get("annual_rate")
            or field_map.get("yield")
            or field_map.get("rate")
            or field_map.get("value")
        )
        if date_key is None or rate_key is None:
            raise ValueError(
                "Risk-free series CSV must contain date/datetime/timestamp and annual/yield/rate columns."
            )
        for row in reader:
            timestamp_ms = _coerce_timestamp_ms(row.get(date_key))
            if timestamp_ms is None:
                continue
            try:
                annual_rate = float(row.get(rate_key) or 0.0)
            except Exception:
                continue
            if annual_rate > 1.0:
                annual_rate /= 100.0
            timestamps.append(int(timestamp_ms))
            annual_rates.append(float(annual_rate))

    if not timestamps:
        raise ValueError(f"Risk-free series contains no usable rows: {series_path}")
    ordered = sorted(zip(timestamps, annual_rates), key=lambda item: item[0])
    return [int(item[0]) for item in ordered], [float(item[1]) for item in ordered]


def periodic_risk_free_array(
    *,
    policy: RiskFreePolicy,
    periods_per_year: int,
    size: int,
    timestamps: Any | None = None,
) -> np.ndarray:
    """Return per-period risk-free rates aligned to the returns stream."""
    count = max(0, int(size))
    if count == 0:
        return np.asarray([], dtype=float)

    if policy.mode == "zero":
        return np.zeros(count, dtype=float)
    if policy.mode == "us_treasury_constant":
        return np.full(
            count,
            annual_to_periodic_rate(float(policy.annual), int(periods_per_year)),
            dtype=float,
        )
    if policy.mode != "us_treasury_series":
        raise ValueError(f"Unsupported risk-free mode: {policy.mode}")

    if timestamps is None:
        raise ValueError("us_treasury_series mode requires timestamps.")
    if not policy.series_path:
        raise ValueError("us_treasury_series mode requires RISK_FREE_SERIES_PATH.")

    series_ts, series_ann = _load_series_points(str(policy.series_path))
    resolved: list[float] = []
    for raw in list(timestamps):
        timestamp_ms = _coerce_timestamp_ms(raw)
        if timestamp_ms is None:
            raise ValueError("Risk-free series alignment requires concrete timestamps.")
        idx = bisect_right(series_ts, int(timestamp_ms)) - 1
        if idx < 0:
            raise ValueError(
                f"Risk-free series has no entry on or before {timestamp_ms} ms for tenor {policy.tenor}."
            )
        resolved.append(annual_to_periodic_rate(float(series_ann[idx]), int(periods_per_year)))
    return np.asarray(resolved, dtype=float)


def periodic_sortino_target_array(
    *,
    policy: RiskFreePolicy,
    periods_per_year: int,
    size: int,
    timestamps: Any | None = None,
) -> np.ndarray:
    """Return per-period Sortino target aligned to the returns stream."""
    count = max(0, int(size))
    if count == 0:
        return np.asarray([], dtype=float)

    mode = str(policy.sortino_target_mode or "same_as_rf").strip().lower()
    if mode == "zero":
        return np.zeros(count, dtype=float)
    if mode == "explicit":
        return np.full(
            count,
            annual_to_periodic_rate(float(policy.sortino_target_annual), int(periods_per_year)),
            dtype=float,
        )
    return periodic_risk_free_array(
        policy=policy,
        periods_per_year=periods_per_year,
        size=count,
        timestamps=timestamps,
    )


def mean_excess_return(
    returns: np.ndarray,
    *,
    policy: RiskFreePolicy,
    periods_per_year: int,
    timestamps: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned returns, risk-free series, and excess returns."""
    clean_returns = np.asarray(returns, dtype=float)
    rf = periodic_risk_free_array(
        policy=policy,
        periods_per_year=periods_per_year,
        size=int(clean_returns.size),
        timestamps=timestamps,
    )
    if clean_returns.size != rf.size:
        raise ValueError("Risk-free series length mismatch.")
    return clean_returns, rf, clean_returns - rf


def resolve_risk_free_config(
    config: Any | None,
    *,
    periods_per_year: int,
    timestamps: Any | None = None,
    size: int | None = None,
) -> ResolvedRiskFree:
    """Resolve config into scalar + aligned series risk-free values."""
    policy = risk_free_policy_from_config(config)
    if size is None:
        if timestamps is None:
            size = 1
        else:
            size = len(list(timestamps))
    periodic_rates = periodic_risk_free_array(
        policy=policy,
        periods_per_year=int(periods_per_year),
        size=int(size),
        timestamps=timestamps,
    )
    periodic_sortino_targets = periodic_sortino_target_array(
        policy=policy,
        periods_per_year=int(periods_per_year),
        size=int(size),
        timestamps=timestamps,
    )
    resolved_annual = (
        float((1.0 + float(periodic_rates[-1])) ** float(periods_per_year) - 1.0)
        if policy.mode == "us_treasury_series" and periodic_rates.size
        else float(policy.annual)
    )
    resolved_sortino_annual = (
        float(policy.sortino_target_annual)
        if str(policy.sortino_target_mode or "same_as_rf").strip().lower() == "explicit"
        else (
            0.0
            if str(policy.sortino_target_mode or "same_as_rf").strip().lower() == "zero"
            else float(resolved_annual)
        )
    )
    per_period_rate = float(periodic_rates[-1] if periodic_rates.size else 0.0)
    sortino_target_per_period = float(
        periodic_sortino_targets[-1] if periodic_sortino_targets.size else 0.0
    )
    return ResolvedRiskFree(
        mode=str(policy.mode),
        tenor=str(policy.tenor),
        annual_rate=float(resolved_annual),
        per_period_rate=float(per_period_rate),
        periodic_rates=np.asarray(periodic_rates, dtype=float),
        sortino_target_mode=str(policy.sortino_target_mode or "same_as_rf"),
        sortino_target_annual=float(resolved_sortino_annual),
        sortino_target_per_period=float(sortino_target_per_period),
        periodic_sortino_targets=np.asarray(periodic_sortino_targets, dtype=float),
    )


def sharpe_ratio(
    returns: np.ndarray,
    *,
    periods_per_year: int,
    risk_free_per_period: float | np.ndarray = 0.0,
) -> float:
    """Annualized Sharpe ratio from per-period excess returns."""
    series = np.asarray(returns, dtype=float)
    if series.size == 0:
        return 0.0
    rf = (
        np.full(series.size, float(risk_free_per_period), dtype=float)
        if np.isscalar(risk_free_per_period)
        else np.asarray(risk_free_per_period, dtype=float)
    )
    if rf.size != series.size:
        raise ValueError("risk_free_per_period must match returns length.")
    mask = np.isfinite(series) & np.isfinite(rf)
    if not np.any(mask):
        return 0.0
    clean = series[mask]
    excess = clean - rf[mask]
    sigma = float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0
    if sigma <= 1e-12:
        return 0.0
    value = (float(np.mean(excess)) / sigma) * float(np.sqrt(periods_per_year))
    return float(value) if np.isfinite(value) else 0.0


def sortino_ratio(
    returns: np.ndarray,
    *,
    periods_per_year: int,
    target_per_period: float | np.ndarray = 0.0,
) -> float:
    """Annualized Sortino ratio from downside deviation vs target."""
    series = np.asarray(returns, dtype=float)
    if series.size == 0:
        return 0.0
    target = (
        np.full(series.size, float(target_per_period), dtype=float)
        if np.isscalar(target_per_period)
        else np.asarray(target_per_period, dtype=float)
    )
    if target.size != series.size:
        raise ValueError("target_per_period must match returns length.")
    mask = np.isfinite(series) & np.isfinite(target)
    if not np.any(mask):
        return 0.0
    clean = series[mask]
    target_clean = target[mask]
    downside = clean - target_clean
    downside = downside[downside < 0.0]
    downside_std = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    if downside_std <= 1e-12:
        return 0.0
    value = (float(np.mean(clean - target_clean)) / downside_std) * float(np.sqrt(periods_per_year))
    return float(value) if np.isfinite(value) else 0.0


__all__ = [
    "ResolvedRiskFree",
    "RiskFreePolicy",
    "annual_to_periodic_rate",
    "mean_excess_return",
    "periodic_risk_free_array",
    "periodic_sortino_target_array",
    "resolve_risk_free_config",
    "risk_free_policy_from_config",
    "sharpe_ratio",
    "sortino_ratio",
]
