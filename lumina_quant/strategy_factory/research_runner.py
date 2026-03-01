"""Candidate research runner with train/val/oos evaluation and robust metrics."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from copy import deepcopy
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from lumina_quant.config import BaseConfig
from lumina_quant.parquet_market_data import load_data_dict_from_parquet
from lumina_quant.symbols import (
    CANONICAL_STRATEGY_TIMEFRAMES,
    canonical_symbol,
    canonicalize_symbol_list,
    normalize_strategy_timeframes,
)

_PERIODS_PER_YEAR = {
    "1s": 31_536_000,
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "30m": 17_520,
    "1h": 8_760,
    "4h": 2_190,
    "1d": 365,
}

_MIN_BARS = 360
_CROWDING_SUPPORT_PATH = Path("data") / "market_parquet" / "feature_points"
_METALS = {"XAU/USDT", "XAG/USDT"}
_LEADERS = ("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT")

DEFAULT_RESEARCH_SCORING_CONFIG: dict[str, Any] = {
    "stage1_prefilter_weights": {
        "sharpe_weight": 2.0,
        "return_weight": 20.0,
        "pbo_penalty": 2.0,
    },
    "candidate_rank_score_weights": {
        "sharpe_weight": 2.8,
        "deflated_sharpe_weight": 1.4,
        "pbo_penalty": 2.0,
        "return_weight": 35.0,
        "turnover_penalty": 2.5,
        "drawdown_penalty": 3.0,
        "turnover_threshold": 2.5,
    },
    "hurdle_score_weights": {
        "sharpe_weight": 2.4,
        "return_weight": 35.0,
        "deflated_sharpe_weight": 1.2,
        "pbo_penalty": 2.0,
        "turnover_penalty": 4.0,
        "drawdown_penalty": 5.0,
        "spa_pvalue_penalty": 1.0,
    },
    "reject_thresholds": {
        "oos_sharpe_min": 0.35,
        "max_pbo": 0.45,
        "max_turnover": 2.5,
        "max_drawdown": 0.45,
        "min_trade_count": 5.0,
    },
    "cost_stress_thresholds": {
        "x2_sharpe_min": 0.0,
        "x3_sharpe_min": -0.25,
    },
    "keep_ratio_bounds": {
        "min": 0.05,
        "max": 1.0,
    },
}


def _resolve_score_config(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    resolved = deepcopy(DEFAULT_RESEARCH_SCORING_CONFIG)
    if not isinstance(overrides, Mapping):
        return resolved
    for key, default_value in resolved.items():
        override_value = overrides.get(key)
        if isinstance(default_value, dict) and isinstance(override_value, Mapping):
            for sub_key in default_value:
                if sub_key in override_value:
                    default_value[sub_key] = override_value[sub_key]
        elif override_value is not None and not isinstance(default_value, dict):
            resolved[key] = override_value
    return resolved


@dataclass(slots=True)
class SeriesBundle:
    symbol: str
    timeframe: str
    datetime: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray


def _hash_unit_interval(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) / float(0xFFFFFFFFFFFFFFFF)


def _hash_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return int(digest, 16)


def _candidate_identity(candidate: dict[str, Any]) -> str:
    payload = {
        "name": str(candidate.get("name", "")),
        "strategy_class": str(candidate.get("strategy_class", "")),
        "strategy_timeframe": str(candidate.get("strategy_timeframe") or candidate.get("timeframe") or ""),
        "symbols": list(candidate.get("symbols") or []),
        "params": dict(candidate.get("params") or {}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def adapt_legacy_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Adapt legacy candidate fields to v2 contract without data loss."""

    row = dict(candidate)

    if not row.get("strategy_timeframe") and row.get("timeframe"):
        row["strategy_timeframe"] = str(row.get("timeframe"))
    if not row.get("timeframe") and row.get("strategy_timeframe"):
        row["timeframe"] = str(row.get("strategy_timeframe"))

    if not row.get("strategy_class") and row.get("strategy"):
        row["strategy_class"] = str(row.get("strategy"))
    if not row.get("strategy") and row.get("strategy_class"):
        row["strategy"] = str(row.get("strategy_class"))

    row["strategy_timeframe"] = str(row.get("strategy_timeframe") or "1m").strip().lower()
    row["timeframe"] = str(row.get("timeframe") or row["strategy_timeframe"]).strip().lower()

    symbols = canonicalize_symbol_list(list(row.get("symbols") or []))
    row["symbols"] = symbols

    if not row.get("candidate_id"):
        row["candidate_id"] = _candidate_identity(row)

    row["params"] = dict(row.get("params") or {})
    row["name"] = str(row.get("name") or row.get("candidate_id"))
    row["strategy_class"] = str(row.get("strategy_class") or row.get("strategy") or "")
    row["strategy"] = str(row.get("strategy") or row.get("strategy_class") or "")
    row["family"] = str(row.get("family") or _family_from_strategy(row["strategy_class"]))
    return row


def _family_from_strategy(strategy_class: str) -> str:
    token = str(strategy_class).strip().lower()
    if "composite" in token or "trend" in token:
        return "trend"
    if "vwap" in token or "reversion" in token:
        return "mean_reversion"
    if "leadlag" in token:
        return "intraday_alpha"
    if "pair" in token:
        return "market_neutral"
    if "perp" in token or "carry" in token:
        return "carry"
    if "micro" in token:
        return "micro"
    return "other"


def _split_lengths(total: int, *, train_frac: float = 0.60, val_frac: float = 0.20) -> tuple[slice, slice, slice]:
    if total <= 0:
        return slice(0, 0), slice(0, 0), slice(0, 0)
    train_end = max(1, int(total * train_frac))
    val_end = max(train_end + 1, int(total * (train_frac + val_frac)))
    val_end = min(total - 1, val_end)
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, total)


def _safe_std(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    out = float(np.std(values, ddof=1))
    if not math.isfinite(out):
        return 0.0
    return out


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    out = float(np.mean(values))
    if not math.isfinite(out):
        return 0.0
    return out


def _max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdown = 1.0 - np.divide(equity, np.maximum(peaks, 1e-12))
    return float(np.max(drawdown)) if drawdown.size else 0.0


def _rolling_sharpe_min(returns: np.ndarray, *, window: int = 64, periods_per_year: int = 365) -> float:
    if returns.size < max(8, window):
        return 0.0
    vals: list[float] = []
    for idx in range(window, returns.size + 1):
        tail = returns[idx - window : idx]
        mu = _safe_mean(tail)
        sigma = _safe_std(tail)
        if sigma <= 1e-12:
            continue
        vals.append((mu / sigma) * math.sqrt(periods_per_year))
    if not vals:
        return 0.0
    return float(min(vals))


def _worst_month(returns: np.ndarray, *, bars_per_month: int) -> float:
    if returns.size == 0:
        return 0.0
    bars = max(4, int(bars_per_month))
    monthly: list[float] = []
    for idx in range(0, returns.size, bars):
        tail = returns[idx : idx + bars]
        if tail.size == 0:
            continue
        monthly.append(float(np.prod(1.0 + tail) - 1.0))
    if not monthly:
        return 0.0
    return float(min(monthly))


def _deflated_sharpe_ratio(returns: np.ndarray, *, num_trials: int = 1) -> float:
    if returns.size < 16:
        return 0.0
    mu = _safe_mean(returns)
    sigma = _safe_std(returns)
    if sigma <= 1e-12:
        return 0.0

    sharpe = mu / sigma
    n = float(max(2, returns.size))
    k = float(max(1, num_trials))
    expected_max = math.sqrt(2.0 * math.log(k)) / math.sqrt(n)

    centered = returns - mu
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))
    skew = 0.0 if sigma <= 1e-12 else m3 / (sigma**3)
    kurt = 3.0 if sigma <= 1e-12 else m4 / (sigma**4)

    denom_term = 1.0 - (skew * sharpe) + (((kurt - 1.0) / 4.0) * (sharpe**2))
    denom_term = max(1e-8, denom_term)
    denom = math.sqrt(denom_term / max(1.0, n - 1.0))
    z = (sharpe - expected_max) / max(1e-8, denom)

    # Normal CDF.
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return float(max(0.0, min(1.0, cdf)))


def _approx_pbo(returns: np.ndarray) -> float:
    """Approximate probability of backtest overfitting from fold instability."""
    n = returns.size
    if n < 64:
        return 1.0
    folds = min(8, max(4, n // 32))
    fold_size = n // folds
    if fold_size <= 0:
        return 1.0

    failures = 0
    trials = 0
    for idx in range(folds):
        test_start = idx * fold_size
        test_end = n if idx == folds - 1 else (idx + 1) * fold_size
        test = returns[test_start:test_end]
        train = np.concatenate((returns[:test_start], returns[test_end:]))
        if train.size < 8 or test.size < 8:
            continue
        train_sharpe = 0.0
        test_sharpe = 0.0
        train_std = _safe_std(train)
        test_std = _safe_std(test)
        if train_std > 1e-12:
            train_sharpe = _safe_mean(train) / train_std
        if test_std > 1e-12:
            test_sharpe = _safe_mean(test) / test_std
        trials += 1
        if train_sharpe > 0.0 and test_sharpe <= 0.0:
            failures += 1
    if trials <= 0:
        return 1.0
    return float(failures / trials)


def _spa_like_pvalue(returns: np.ndarray, *, bootstrap_rounds: int = 200) -> float:
    """Simple bootstrap p-value proxy for data-snooping correction."""
    if returns.size < 16:
        return 1.0
    observed = _safe_mean(returns)
    if observed <= 0.0:
        return 1.0

    rng = np.random.default_rng(12345)
    exceed = 0
    centered = returns - _safe_mean(returns)
    n = centered.size
    for _ in range(max(64, int(bootstrap_rounds))):
        idx = rng.integers(0, n, size=n)
        sample = centered[idx]
        if _safe_mean(sample) >= observed:
            exceed += 1
    return float(exceed / max(1, int(bootstrap_rounds)))


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    n = min(x.size, y.size)
    if n < 8:
        return 0.0
    xa = x[-n:]
    ya = y[-n:]
    xs = _safe_std(xa)
    ys = _safe_std(ya)
    if xs <= 1e-12 or ys <= 1e-12:
        return 0.0
    corr = float(np.corrcoef(xa, ya)[0, 1])
    if not math.isfinite(corr):
        return 0.0
    return corr


def _compute_metrics(
    returns: np.ndarray,
    *,
    turnover: np.ndarray,
    exposure: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int,
    num_trials: int,
) -> dict[str, float]:
    if returns.size == 0:
        return {
            "return": 0.0,
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "mdd": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
            "trades": 0.0,
            "trade_count": 0.0,
            "win_rate": 0.0,
            "avg_trade": 0.0,
            "exposure": 0.0,
            "volatility": 0.0,
            "stability": 0.0,
            "rolling_sharpe_min": 0.0,
            "worst_month": 0.0,
            "benchmark_corr": 0.0,
            "deflated_sharpe": 0.0,
            "pbo": 1.0,
            "spa_pvalue": 1.0,
        }

    total_return = float(np.prod(1.0 + returns) - 1.0)
    years = max(1.0 / float(periods_per_year), returns.size / float(periods_per_year))
    cagr = float(math.exp(math.log1p(max(-0.999999, total_return)) / years) - 1.0)

    mean_r = _safe_mean(returns)
    sigma = _safe_std(returns)
    sharpe = 0.0 if sigma <= 1e-12 else (mean_r / sigma) * math.sqrt(periods_per_year)
    downside = returns[returns < 0.0]
    downside_std = _safe_std(downside)
    sortino = 0.0 if downside_std <= 1e-12 else (mean_r / downside_std) * math.sqrt(periods_per_year)

    max_dd = _max_drawdown(returns)
    calmar = 0.0 if max_dd <= 1e-12 else cagr / max_dd

    trade_count = float(np.sum(turnover > 1e-9))
    win_rate = float(np.sum(returns > 0.0) / returns.size)
    avg_trade = float(np.sum(returns) / max(1.0, trade_count))

    rolling_sharpe_min = _rolling_sharpe_min(
        returns,
        window=min(128, max(32, returns.size // 8)),
        periods_per_year=periods_per_year,
    )
    worst_month = _worst_month(
        returns,
        bars_per_month=max(4, int(periods_per_year // 12)),
    )

    stability = 0.5 * max(-3.0, min(3.0, rolling_sharpe_min)) + 0.5 * worst_month

    return {
        "return": total_return,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "mdd": float(max_dd),
        "max_drawdown": float(max_dd),
        "turnover": float(_safe_mean(turnover)),
        "trades": trade_count,
        "trade_count": trade_count,
        "win_rate": win_rate,
        "avg_trade": avg_trade,
        "exposure": float(_safe_mean(np.abs(exposure))),
        "volatility": float(sigma * math.sqrt(periods_per_year)),
        "stability": float(stability),
        "rolling_sharpe_min": float(rolling_sharpe_min),
        "worst_month": float(worst_month),
        "benchmark_corr": float(_correlation(returns, benchmark_returns)),
        "deflated_sharpe": float(_deflated_sharpe_ratio(returns, num_trials=num_trials)),
        "pbo": float(_approx_pbo(returns)),
        "spa_pvalue": float(_spa_like_pvalue(returns)),
    }


def _hurdle_fields(
    train: dict[str, float],
    val: dict[str, float],
    oos: dict[str, float],
    *,
    scoring_config: Mapping[str, Any] | None = None,
    oos_sharpe_min: float = 0.35,
    max_pbo: float = 0.45,
    max_turnover: float = 2.5,
    max_drawdown: float = 0.45,
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    cfg = _resolve_score_config(scoring_config)
    thresholds = dict(cfg.get("reject_thresholds") or {})
    weights = dict(cfg.get("hurdle_score_weights") or {})
    oos_sharpe_min = float(thresholds.get("oos_sharpe_min", oos_sharpe_min))
    max_pbo = float(thresholds.get("max_pbo", max_pbo))
    max_turnover = float(thresholds.get("max_turnover", max_turnover))
    max_drawdown = float(thresholds.get("max_drawdown", max_drawdown))
    min_trade_count = float(thresholds.get("min_trade_count", 5.0))

    def _pack(metrics: dict[str, float], *, stage: str) -> dict[str, Any]:
        score = (
            (float(weights.get("sharpe_weight", 2.4)) * float(metrics.get("sharpe", 0.0)))
            + (float(weights.get("return_weight", 35.0)) * float(metrics.get("return", 0.0)))
            + (float(weights.get("deflated_sharpe_weight", 1.2)) * float(metrics.get("deflated_sharpe", 0.0)))
            - (float(weights.get("pbo_penalty", 2.0)) * float(metrics.get("pbo", 1.0)))
            - (
                float(weights.get("turnover_penalty", 4.0))
                * max(0.0, float(metrics.get("turnover", 0.0)) - max_turnover)
            )
            - (
                float(weights.get("drawdown_penalty", 5.0))
                * max(0.0, float(metrics.get("mdd", 0.0)) - max_drawdown)
            )
            - (float(weights.get("spa_pvalue_penalty", 1.0)) * float(metrics.get("spa_pvalue", 1.0)))
        )

        passed = bool(
            float(metrics.get("sharpe", 0.0)) >= (-0.1 if stage != "oos" else oos_sharpe_min)
            and float(metrics.get("pbo", 1.0)) <= max_pbo
            and float(metrics.get("turnover", 0.0)) <= max_turnover
            and float(metrics.get("mdd", 0.0)) <= max_drawdown
            and float(metrics.get("trade_count", 0.0)) >= min_trade_count
        )

        return {
            "pass": passed,
            "score": float(score),
            "excess_return": float(metrics.get("return", 0.0)),
        }

    fields = {
        "train": _pack(train, stage="train"),
        "val": _pack(val, stage="val"),
        "oos": _pack(oos, stage="oos"),
    }

    cost_ok = bool(
        float(oos.get("sharpe", 0.0)) >= oos_sharpe_min
        and float(oos.get("pbo", 1.0)) <= max_pbo
    )

    hard_reject_reasons: dict[str, Any] = {}
    if float(oos.get("sharpe", 0.0)) < oos_sharpe_min:
        hard_reject_reasons["oos_sharpe"] = float(oos.get("sharpe", 0.0))
    if float(oos.get("pbo", 1.0)) > max_pbo:
        hard_reject_reasons["pbo"] = float(oos.get("pbo", 1.0))
    if float(oos.get("turnover", 0.0)) > max_turnover:
        hard_reject_reasons["turnover"] = float(oos.get("turnover", 0.0))
    if float(oos.get("mdd", 0.0)) > max_drawdown:
        hard_reject_reasons["max_drawdown"] = float(oos.get("mdd", 0.0))
    if float(oos.get("trade_count", 0.0)) < min_trade_count:
        hard_reject_reasons["trade_count"] = float(oos.get("trade_count", 0.0))

    return fields, bool(cost_ok and not hard_reject_reasons), hard_reject_reasons


def _series_to_stream(values: np.ndarray, *, offset: int = 0) -> list[dict[str, float]]:
    return [{"t": float(offset + idx), "v": float(value)} for idx, value in enumerate(values)]


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def _rolling_z(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    win = max(8, int(window))
    if values.size < win:
        return out
    for idx in range(win, values.size + 1):
        tail = values[idx - win : idx]
        hist = tail[:-1]
        latest = tail[-1]
        std = _safe_std(hist)
        if std <= 1e-12:
            out[idx - 1] = 0.0
        else:
            out[idx - 1] = (float(latest) - _safe_mean(hist)) / std
    return out


def _trend_efficiency_series(closes: np.ndarray, window: int = 55) -> np.ndarray:
    out = np.full(closes.shape, np.nan, dtype=float)
    win = max(8, int(window))
    if closes.size < win + 1:
        return out
    for idx in range(win, closes.size):
        span = closes[idx - win : idx + 1]
        net = abs(float(span[-1] - span[0]))
        path = float(np.sum(np.abs(np.diff(span))))
        out[idx] = 0.0 if path <= 1e-12 else min(1.0, max(0.0, net / path))
    return out


def _vol_ratio_series(closes: np.ndarray, fast: int = 8, slow: int = 55) -> np.ndarray:
    out = np.full(closes.shape, np.nan, dtype=float)
    rets = np.diff(np.log(np.clip(closes, 1e-12, np.inf)), prepend=np.log(max(1e-12, closes[0])))
    f = max(4, int(fast))
    s = max(f + 2, int(slow))
    if rets.size < s:
        return out
    for idx in range(s, rets.size + 1):
        fast_std = _safe_std(rets[idx - f : idx])
        slow_std = _safe_std(rets[idx - s : idx])
        out[idx - 1] = 0.0 if slow_std <= 1e-12 else fast_std / slow_std
    return out


def _vwap_dev_z(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, window: int = 60, z_window: int = 120) -> np.ndarray:
    n = close.size
    out = np.full(close.shape, np.nan, dtype=float)
    win = max(8, int(window))
    if n < win:
        return out
    dev = np.full(close.shape, np.nan, dtype=float)
    typical = (high + low + close) / 3.0
    vol = np.clip(volume, 0.0, np.inf)
    for idx in range(win, n + 1):
        p = typical[idx - win : idx]
        v = vol[idx - win : idx]
        den = float(np.sum(v))
        vw = float(np.sum(p * v) / den) if den > 1e-12 else float(np.mean(p))
        if vw <= 0.0:
            dev[idx - 1] = 0.0
        else:
            dev[idx - 1] = (close[idx - 1] / vw) - 1.0
    out = _rolling_z(np.nan_to_num(dev, nan=0.0), window=max(16, int(z_window)))
    return out


def _pair_spread_z(px: np.ndarray, py: np.ndarray, window: int = 96) -> np.ndarray:
    n = min(px.size, py.size)
    out = np.full(n, np.nan, dtype=float)
    x = np.log(np.clip(px[-n:], 1e-12, np.inf))
    y = np.log(np.clip(py[-n:], 1e-12, np.inf))
    win = max(16, int(window))
    if n < win + 2:
        return out
    for idx in range(win, n + 1):
        x_tail = x[idx - win : idx]
        y_tail = y[idx - win : idx]
        vx = float(np.var(x_tail))
        if vx <= 1e-12:
            beta = 1.0
        else:
            beta = float(np.cov(x_tail, y_tail)[0, 1] / vx)
        spread_tail = y_tail - (beta * x_tail)
        mean = float(np.mean(spread_tail[:-1]))
        std = _safe_std(spread_tail[:-1])
        if std <= 1e-12:
            out[idx - 1] = 0.0
        else:
            out[idx - 1] = (float(spread_tail[-1]) - mean) / std
    return out


def _align_bundles(bundles: Sequence[SeriesBundle]) -> dict[str, np.ndarray] | None:
    if not bundles:
        return None
    min_len = min(bundle.close.size for bundle in bundles)
    if min_len < _MIN_BARS:
        return None

    aligned: dict[str, np.ndarray] = {
        "datetime": bundles[0].datetime[-min_len:],
    }
    for bundle in bundles:
        prefix = bundle.symbol
        aligned[f"{prefix}:open"] = bundle.open[-min_len:]
        aligned[f"{prefix}:high"] = bundle.high[-min_len:]
        aligned[f"{prefix}:low"] = bundle.low[-min_len:]
        aligned[f"{prefix}:close"] = bundle.close[-min_len:]
        aligned[f"{prefix}:volume"] = bundle.volume[-min_len:]
    return aligned


def _returns_from_close(closes: np.ndarray) -> np.ndarray:
    if closes.size < 2:
        return np.zeros(closes.shape, dtype=float)
    return np.diff(closes, prepend=closes[0]) / np.clip(np.r_[closes[0], closes[:-1]], 1e-12, np.inf)


def _strategy_signal(
    candidate: dict[str, Any],
    *,
    aligned: dict[str, np.ndarray],
    symbols: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    strategy_class = str(candidate.get("strategy_class") or candidate.get("strategy") or "")
    params = dict(candidate.get("params") or {})

    n = len(next(iter(aligned.values()))) if aligned else 0
    if n <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float), {}

    exposures = np.zeros((len(symbols), n), dtype=float)
    returns = np.zeros((len(symbols), n), dtype=float)

    for s_idx, symbol in enumerate(symbols):
        close = aligned[f"{symbol}:close"]
        returns[s_idx] = _returns_from_close(close)

    meta: dict[str, Any] = {}

    if strategy_class == "CompositeTrendStrategy":
        long_th = float(params.get("long_threshold", 0.55))
        short_th = float(params.get("short_threshold", 0.55))
        te_min = float(params.get("te_min", 0.25))
        vr_min = float(params.get("vr_min", 0.85))

        for s_idx, symbol in enumerate(symbols):
            close = aligned[f"{symbol}:close"]
            high = aligned[f"{symbol}:high"]
            low = aligned[f"{symbol}:low"]
            volume = aligned[f"{symbol}:volume"]

            ret8 = np.log(np.clip(close, 1e-12, np.inf) / np.clip(np.roll(close, 8), 1e-12, np.inf))
            ret21 = np.log(np.clip(close, 1e-12, np.inf) / np.clip(np.roll(close, 21), 1e-12, np.inf))
            ret55 = np.log(np.clip(close, 1e-12, np.inf) / np.clip(np.roll(close, 55), 1e-12, np.inf))
            z8 = _rolling_z(ret8, 120)
            z21 = _rolling_z(ret21, 120)
            z55 = _rolling_z(ret55, 120)
            mom = (0.5 * np.nan_to_num(z8)) + (0.3 * np.nan_to_num(z21)) + (0.2 * np.nan_to_num(z55))
            te = np.nan_to_num(_trend_efficiency_series(close, 55), nan=0.0)
            vr = np.nan_to_num(_vol_ratio_series(close, 8, 55), nan=0.0)

            vol_shock = _rolling_z(np.nan_to_num(volume, nan=0.0), 64)
            score = mom * (1.0 + 0.25 * np.tanh(np.nan_to_num(vol_shock) / 2.0)) * np.clip(te, 0.0, 1.0)
            gate = (te >= te_min) & (vr >= vr_min)

            position = np.where(score >= long_th, 1.0, np.where(score <= -short_th, -1.0, 0.0))
            position = np.where(gate, position, 0.0)
            exposures[s_idx] = position

            _ = high, low  # kept for symmetry with requested factor inputs.

    elif strategy_class in {"VolCompressionVWAPReversionStrategy", "VolCompressionVwapReversionStrategy", "VolatilityCompressionReversionStrategy"}:
        entry_z = float(params.get("entry_z", 1.5))
        exit_z = float(params.get("exit_z", 0.35))
        compression_pct = float(params.get("compression_percentile", 0.30))
        comp_vol_ratio = float(params.get("compression_vol_ratio", params.get("compression_threshold", 0.85)))

        for s_idx, symbol in enumerate(symbols):
            close = aligned[f"{symbol}:close"]
            high = aligned[f"{symbol}:high"]
            low = aligned[f"{symbol}:low"]
            volume = aligned[f"{symbol}:volume"]

            dev_z = np.nan_to_num(_vwap_dev_z(high, low, close, volume, window=60, z_window=120), nan=0.0)
            vr = np.nan_to_num(_vol_ratio_series(close, 12, 60), nan=0.0)
            bw = np.nan_to_num(_rolling_z(np.abs(_returns_from_close(close)), 64), nan=0.0)
            compression = (vr <= comp_vol_ratio) & (bw <= _normal_cdf(compression_pct) * 2.0)

            pos = np.zeros(close.shape, dtype=float)
            pos = np.where(compression & (dev_z <= -entry_z), 1.0, pos)
            pos = np.where(compression & (dev_z >= entry_z), -1.0, pos)
            pos = np.where(np.abs(dev_z) <= exit_z, 0.0, pos)
            exposures[s_idx] = pos

    elif strategy_class == "LeadLagSpilloverStrategy":
        # Build leader lag predictor for non-leader assets only.
        leader_symbols = [symbol for symbol in symbols if symbol in _LEADERS]
        laggards = [symbol for symbol in symbols if symbol not in _LEADERS and symbol not in _METALS]
        entry_score = float(params.get("entry_score", params.get("entry_spillover", 0.35)))
        exit_score = float(params.get("exit_score", params.get("exit_spillover", 0.08)))
        lag_order = max(1, int(params.get("max_lag", 3)))

        if leader_symbols and laggards:
            leader_ret = np.zeros((len(leader_symbols), n), dtype=float)
            for idx, sym in enumerate(leader_symbols):
                leader_ret[idx] = _returns_from_close(aligned[f"{sym}:close"])

            decay = np.exp(-np.arange(1, lag_order + 1, dtype=float))
            decay /= np.sum(decay)

            for symbol in laggards:
                s_idx = symbols.index(symbol)
                pred = np.zeros(n, dtype=float)
                for lag in range(1, lag_order + 1):
                    shifted = np.roll(np.mean(leader_ret, axis=0), lag)
                    pred += decay[lag - 1] * shifted
                follower_ret = _returns_from_close(aligned[f"{symbol}:close"])
                score = np.zeros(n, dtype=float)
                sigma = _safe_std(follower_ret)
                if sigma > 1e-12:
                    score = pred / sigma
                pos = np.where(score >= entry_score, 1.0, np.where(score <= -entry_score, -1.0, 0.0))
                pos = np.where(np.abs(score) <= exit_score, 0.0, pos)
                exposures[s_idx] = pos

    elif strategy_class in {"PairSpreadZScoreStrategy", "PairTradingZScoreStrategy"} and len(symbols) >= 2:
        symbol_x = canonical_symbol(str(params.get("symbol_x") or symbols[0]))
        symbol_y = canonical_symbol(str(params.get("symbol_y") or symbols[1]))
        if symbol_x not in symbols:
            symbol_x = symbols[0]
        if symbol_y not in symbols:
            symbol_y = symbols[1]
        x_idx = symbols.index(symbol_x)
        y_idx = symbols.index(symbol_y)

        entry_z = float(params.get("entry_z", 2.0))
        exit_z = float(params.get("exit_z", 0.35))
        lookback = int(params.get("lookback_window", 96))

        z = np.nan_to_num(
            _pair_spread_z(
                aligned[f"{symbol_x}:close"],
                aligned[f"{symbol_y}:close"],
                window=lookback,
            ),
            nan=0.0,
        )

        long_spread = z <= -entry_z
        short_spread = z >= entry_z
        exit_cond = np.abs(z) <= exit_z

        x_pos = np.zeros(n, dtype=float)
        y_pos = np.zeros(n, dtype=float)
        x_pos = np.where(long_spread, -1.0, x_pos)
        y_pos = np.where(long_spread, 1.0, y_pos)
        x_pos = np.where(short_spread, 1.0, x_pos)
        y_pos = np.where(short_spread, -1.0, y_pos)
        x_pos = np.where(exit_cond, 0.0, x_pos)
        y_pos = np.where(exit_cond, 0.0, y_pos)

        exposures[x_idx] = x_pos
        exposures[y_idx] = y_pos

    elif strategy_class == "PerpCrowdingCarryStrategy":
        if not _CROWDING_SUPPORT_PATH.exists():
            meta["missing_support_data"] = True
            exposures[:] = 0.0
        else:
            # Placeholder support-data proxy from price/volume dynamics.
            entry = float(params.get("entry_threshold", 0.30))
            for s_idx, symbol in enumerate(symbols):
                close = aligned[f"{symbol}:close"]
                ret = _returns_from_close(close)
                vol = np.nan_to_num(_rolling_z(np.abs(ret), 96), nan=0.0)
                crowd = np.tanh(_rolling_z(ret, 64) + 0.5 * vol)
                exposures[s_idx] = np.where(crowd >= entry, 1.0, np.where(crowd <= -entry, -1.0, 0.0))

    elif strategy_class == "MicroRangeExpansion1sStrategy":
        lookback = max(8, int(params.get("lookback", 30)))
        range_z_threshold = float(params.get("range_z_threshold", 1.5))
        volume_z_threshold = float(params.get("volume_z_threshold", 1.0))
        for s_idx, symbol in enumerate(symbols):
            high = aligned[f"{symbol}:high"]
            low = aligned[f"{symbol}:low"]
            close = aligned[f"{symbol}:close"]
            volume = aligned[f"{symbol}:volume"]
            range_pct = np.divide(high - low, np.clip(close, 1e-12, np.inf))
            range_z = np.nan_to_num(_rolling_z(range_pct, lookback), nan=0.0)
            vol_z = np.nan_to_num(_rolling_z(volume, lookback), nan=0.0)
            ret = _returns_from_close(close)
            breakout = np.where(ret >= 0.0, 1.0, -1.0)
            active = (range_z >= range_z_threshold) & (vol_z >= volume_z_threshold)
            exposures[s_idx] = np.where(active, breakout, 0.0)

    else:
        # Generic fallback: momentum sign.
        for s_idx, symbol in enumerate(symbols):
            close = aligned[f"{symbol}:close"]
            ret = _returns_from_close(close)
            mom = np.nan_to_num(_rolling_z(ret, 64), nan=0.0)
            exposures[s_idx] = np.where(mom >= 0.4, 1.0, np.where(mom <= -0.4, -1.0, 0.0))

    exposure = np.nanmean(exposures, axis=0)
    portfolio_ret = np.nanmean(np.roll(exposures, 1, axis=1) * returns, axis=0)
    turnover = np.nanmean(np.abs(exposures - np.roll(exposures, 1, axis=1)), axis=0)

    return portfolio_ret, turnover, exposure, meta


def _candidate_cost_rate(candidate: dict[str, Any]) -> float:
    strategy = str(candidate.get("strategy_class") or candidate.get("strategy") or "").lower()
    if "micro" in strategy:
        return 0.0012
    if "pair" in strategy:
        return 0.0008
    if "leadlag" in strategy:
        return 0.0007
    return 0.0005


def _evaluate_candidate(
    candidate: dict[str, Any],
    *,
    cache: Mapping[tuple[str, str], SeriesBundle],
    benchmark_cache: Mapping[str, np.ndarray],
    candidate_count: int,
    scoring_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    symbols = canonicalize_symbol_list(list(candidate.get("symbols") or []))
    timeframe = str(candidate.get("strategy_timeframe") or candidate.get("timeframe") or "1m")
    bundles: list[SeriesBundle] = []
    for symbol in symbols:
        bundle = cache.get((symbol, timeframe))
        if bundle is None:
            continue
        bundles.append(bundle)

    aligned = _align_bundles(bundles)
    if aligned is None:
        return {
            "error": "insufficient_data",
            "candidate": candidate,
            "returns": np.asarray([], dtype=float),
            "turnover": np.asarray([], dtype=float),
            "exposure": np.asarray([], dtype=float),
            "metadata": {"missing_symbols": [symbol for symbol in symbols if (symbol, timeframe) not in cache]},
        }

    returns_raw, turnover, exposure, meta = _strategy_signal(candidate, aligned=aligned, symbols=symbols)
    cost_rate = _candidate_cost_rate(candidate)
    returns = returns_raw - (turnover * cost_rate)

    split_train, split_val, split_oos = _split_lengths(returns.size)
    train_returns = returns[split_train]
    val_returns = returns[split_val]
    oos_returns = returns[split_oos]

    train_turnover = turnover[split_train]
    val_turnover = turnover[split_val]
    oos_turnover = turnover[split_oos]

    train_exposure = exposure[split_train]
    val_exposure = exposure[split_val]
    oos_exposure = exposure[split_oos]

    periods_per_year = int(_PERIODS_PER_YEAR.get(timeframe, 365))
    benchmark = benchmark_cache.get(timeframe)
    if benchmark is None or benchmark.size < returns.size:
        benchmark = np.zeros_like(returns)
    benchmark = benchmark[-returns.size :]

    train_bench = benchmark[split_train]
    val_bench = benchmark[split_val]
    oos_bench = benchmark[split_oos]

    train_metrics = _compute_metrics(
        train_returns,
        turnover=train_turnover,
        exposure=train_exposure,
        benchmark_returns=train_bench,
        periods_per_year=periods_per_year,
        num_trials=candidate_count,
    )
    val_metrics = _compute_metrics(
        val_returns,
        turnover=val_turnover,
        exposure=val_exposure,
        benchmark_returns=val_bench,
        periods_per_year=periods_per_year,
        num_trials=candidate_count,
    )
    oos_metrics = _compute_metrics(
        oos_returns,
        turnover=oos_turnover,
        exposure=oos_exposure,
        benchmark_returns=oos_bench,
        periods_per_year=periods_per_year,
        num_trials=candidate_count,
    )

    # Cost stress tests on OOS.
    cost_rate_x2 = cost_rate * 2.0
    cost_rate_x3 = cost_rate * 3.0
    oos_x2 = returns_raw[split_oos] - (oos_turnover * cost_rate_x2)
    oos_x3 = returns_raw[split_oos] - (oos_turnover * cost_rate_x3)
    oos_stress_x2 = _compute_metrics(
        oos_x2,
        turnover=oos_turnover,
        exposure=oos_exposure,
        benchmark_returns=oos_bench,
        periods_per_year=periods_per_year,
        num_trials=candidate_count,
    )
    oos_stress_x3 = _compute_metrics(
        oos_x3,
        turnover=oos_turnover,
        exposure=oos_exposure,
        benchmark_returns=oos_bench,
        periods_per_year=periods_per_year,
        num_trials=candidate_count,
    )

    hurdle_fields, passed, hard_reject = _hurdle_fields(
        train_metrics,
        val_metrics,
        oos_metrics,
        scoring_config=scoring_config,
    )

    cfg = _resolve_score_config(scoring_config)
    stress_cfg = dict(cfg.get("cost_stress_thresholds") or {})
    x2_sharpe_min = float(stress_cfg.get("x2_sharpe_min", 0.0))
    x3_sharpe_min = float(stress_cfg.get("x3_sharpe_min", -0.25))

    # Enforce hard reject when stress collapses.
    if float(oos_stress_x2.get("sharpe", 0.0)) < x2_sharpe_min:
        hard_reject["stress_x2_sharpe"] = float(oos_stress_x2.get("sharpe", 0.0))
    if float(oos_stress_x3.get("sharpe", 0.0)) < x3_sharpe_min:
        hard_reject["stress_x3_sharpe"] = float(oos_stress_x3.get("sharpe", 0.0))

    passed = passed and not hard_reject
    for stage in ("train", "val", "oos"):
        hurdle_fields[stage]["pass"] = bool(hurdle_fields[stage]["pass"] and passed)

    return {
        "candidate": candidate,
        "returns": returns,
        "turnover": turnover,
        "exposure": exposure,
        "train": train_metrics,
        "val": val_metrics,
        "oos": oos_metrics,
        "oos_cost_stress": {
            "x2": {
                "sharpe": float(oos_stress_x2.get("sharpe", 0.0)),
                "return": float(oos_stress_x2.get("return", 0.0)),
            },
            "x3": {
                "sharpe": float(oos_stress_x3.get("sharpe", 0.0)),
                "return": float(oos_stress_x3.get("return", 0.0)),
            },
        },
        "hurdle_fields": hurdle_fields,
        "pass": bool(passed),
        "hard_reject_reasons": hard_reject,
        "metadata": {
            "strategy_family": _family_from_strategy(str(candidate.get("strategy_class") or "")),
            "cost_rate": float(cost_rate),
            **meta,
        },
    }


def _candidate_rank_score(row: dict[str, Any], *, scoring_config: Mapping[str, Any] | None = None) -> float:
    cfg = _resolve_score_config(scoring_config)
    weights = dict(cfg.get("candidate_rank_score_weights") or {})
    oos = dict(row.get("oos") or {})
    return float(
        (float(weights.get("sharpe_weight", 2.8)) * float(oos.get("sharpe", 0.0)))
        + (float(weights.get("deflated_sharpe_weight", 1.4)) * float(oos.get("deflated_sharpe", 0.0)))
        - (float(weights.get("pbo_penalty", 2.0)) * float(oos.get("pbo", 1.0)))
        + (float(weights.get("return_weight", 35.0)) * float(oos.get("return", 0.0)))
        - (
            float(weights.get("turnover_penalty", 2.5))
            * max(0.0, float(oos.get("turnover", 0.0)) - float(weights.get("turnover_threshold", 2.5)))
        )
        - (float(weights.get("drawdown_penalty", 3.0)) * float(oos.get("mdd", 0.0)))
    )


def _read_csv_ohlcv(path: Path) -> pl.DataFrame:
    frame = pl.read_csv(path)
    cols = {str(col).lower(): col for col in frame.columns}
    required = {
        "open": cols.get("open"),
        "high": cols.get("high"),
        "low": cols.get("low"),
        "close": cols.get("close"),
        "volume": cols.get("volume"),
    }
    if any(value is None for value in required.values()):
        return pl.DataFrame()

    if "datetime" in cols:
        dt_col = cols["datetime"]
        dt_expr = pl.col(dt_col).cast(pl.Datetime(time_unit="ms"), strict=False)
    elif "timestamp" in cols:
        dt_col = cols["timestamp"]
        dt_expr = pl.from_epoch(pl.col(dt_col).cast(pl.Int64), time_unit="s").cast(pl.Datetime(time_unit="ms"))
    else:
        # Synthesize monotonic timestamps.
        dt_expr = pl.int_range(0, frame.height, eager=False).cast(pl.Int64) * 1000
        dt_expr = pl.from_epoch(dt_expr, time_unit="ms").cast(pl.Datetime(time_unit="ms"))

    out = (
        frame.select(
            [
                dt_expr.alias("datetime"),
                pl.col(required["open"]).cast(pl.Float64).alias("open"),
                pl.col(required["high"]).cast(pl.Float64).alias("high"),
                pl.col(required["low"]).cast(pl.Float64).alias("low"),
                pl.col(required["close"]).cast(pl.Float64).alias("close"),
                pl.col(required["volume"]).cast(pl.Float64).alias("volume"),
            ]
        )
        .drop_nulls()
        .sort("datetime")
    )
    return out


def _synthetic_bundle(symbol: str, timeframe: str, *, bars: int = 2400) -> SeriesBundle:
    seed = _hash_seed("synthetic", symbol, timeframe)
    rng = random.Random(seed)

    step_seconds = {
        "1s": 1,
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14_400,
        "1d": 86_400,
    }.get(timeframe, 60)

    start = datetime.now(UTC) - timedelta(seconds=bars * step_seconds)

    close = np.zeros(bars, dtype=float)
    high = np.zeros(bars, dtype=float)
    low = np.zeros(bars, dtype=float)
    open_ = np.zeros(bars, dtype=float)
    volume = np.zeros(bars, dtype=float)
    dt = np.zeros(bars, dtype="datetime64[ms]")

    base = 100.0 + (20.0 * _hash_unit_interval(symbol, timeframe, "base"))
    price = base
    for idx in range(bars):
        drift = (0.00002 + (0.00008 * _hash_unit_interval(symbol, timeframe, "drift")))
        shock = rng.gauss(0.0, 0.0025)
        regime = math.sin((idx / max(50.0, bars / 18.0)) + (2.0 * math.pi * _hash_unit_interval(symbol, timeframe, "phase")))
        step = drift + (0.001 * regime) + shock

        o = max(0.1, price)
        c = max(0.1, o * (1.0 + step))
        wiggle = abs(rng.gauss(0.0, 0.0018)) + 0.0003
        h = max(o, c) * (1.0 + wiggle)
        l = min(o, c) * (1.0 - wiggle)

        open_[idx] = o
        high[idx] = h
        low[idx] = l
        close[idx] = c
        volume[idx] = max(1.0, 1200.0 * (1.0 + abs(regime)) + rng.uniform(-200.0, 200.0))

        ts = start + timedelta(seconds=idx * step_seconds)
        dt[idx] = np.datetime64(ts.replace(tzinfo=None), "ms")
        price = c

    return SeriesBundle(
        symbol=symbol,
        timeframe=timeframe,
        datetime=dt,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _frame_to_bundle(symbol: str, timeframe: str, frame: pl.DataFrame) -> SeriesBundle:
    sorted_frame = frame.sort("datetime")
    return SeriesBundle(
        symbol=symbol,
        timeframe=timeframe,
        datetime=sorted_frame["datetime"].to_numpy(),
        open=sorted_frame["open"].to_numpy(),
        high=sorted_frame["high"].to_numpy(),
        low=sorted_frame["low"].to_numpy(),
        close=sorted_frame["close"].to_numpy(),
        volume=sorted_frame["volume"].to_numpy(),
    )


def _load_bundle_cache(
    *,
    symbols: Sequence[str],
    timeframes: Sequence[str],
) -> tuple[dict[tuple[str, str], SeriesBundle], dict[str, list[str]]]:
    cache: dict[tuple[str, str], SeriesBundle] = {}
    source_map: dict[str, list[str]] = {"parquet": [], "csv": [], "synthetic": []}

    parquet_root = str(getattr(BaseConfig, "MARKET_DATA_PARQUET_PATH", "data/market_parquet"))
    exchange = str(getattr(BaseConfig, "MARKET_DATA_EXCHANGE", "binance") or "binance")

    for timeframe in timeframes:
        loaded: dict[str, pl.DataFrame] = {}
        try:
            loaded = load_data_dict_from_parquet(
                parquet_root,
                exchange=exchange,
                symbol_list=list(symbols),
                timeframe=timeframe,
            )
        except Exception:
            loaded = {}

        for symbol in symbols:
            key = (symbol, timeframe)
            frame = loaded.get(symbol)
            if frame is not None and not frame.is_empty() and frame.height >= _MIN_BARS:
                cache[key] = _frame_to_bundle(symbol, timeframe, frame)
                source_map["parquet"].append(f"{symbol}@{timeframe}")
                continue

            # CSV fallback (compact token layout).
            compact = symbol.replace("/", "")
            csv_candidates = [
                Path("data") / f"{compact}.csv",
                Path("data") / f"{symbol}.csv",
                Path("data") / f"{symbol.replace('/', '_')}.csv",
            ]
            csv_bundle: SeriesBundle | None = None
            for csv_path in csv_candidates:
                if not csv_path.exists():
                    continue
                try:
                    frame_csv = _read_csv_ohlcv(csv_path)
                except Exception:
                    frame_csv = pl.DataFrame()
                if frame_csv.is_empty() or frame_csv.height < _MIN_BARS:
                    continue
                csv_bundle = _frame_to_bundle(symbol, timeframe, frame_csv)
                break

            if csv_bundle is not None:
                cache[key] = csv_bundle
                source_map["csv"].append(f"{symbol}@{timeframe}")
                continue

            cache[key] = _synthetic_bundle(symbol, timeframe)
            source_map["synthetic"].append(f"{symbol}@{timeframe}")

    return cache, source_map


def _benchmark_cache(cache: Mapping[tuple[str, str], SeriesBundle], timeframes: Sequence[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for tf in timeframes:
        bundle = cache.get(("BTC/USDT", tf))
        if bundle is None:
            out[tf] = np.asarray([], dtype=float)
        else:
            out[tf] = _returns_from_close(bundle.close)
    return out


def _build_default_split(strategy_timeframe: str) -> dict[str, Any]:
    now = datetime.now(UTC)
    train_start = now - timedelta(days=360)
    train_end = now - timedelta(days=150)
    val_start = train_end + timedelta(days=1)
    val_end = now - timedelta(days=60)
    oos_start = val_end + timedelta(days=1)
    oos_end = now
    return {
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "val_start": val_start.isoformat(),
        "val_end": val_end.isoformat(),
        "oos_start": oos_start.isoformat(),
        "oos_end": oos_end.isoformat(),
        "strategy_timeframe": str(strategy_timeframe),
    }


def run_candidate_research(
    *,
    candidates: Iterable[dict[str, Any]],
    base_timeframe: str = "1s",
    strategy_timeframes: Sequence[str] | None = None,
    symbol_universe: Sequence[str] | None = None,
    stage1_keep_ratio: float = 0.35,
    max_candidates: int = 512,
    score_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate candidate manifest into train/val/OOS report contract (v2)."""

    base_tf = str(base_timeframe).strip().lower() or "1s"
    if base_tf != "1s":
        base_tf = "1s"

    resolved_scoring_config = _resolve_score_config(score_config)
    keep_ratio_cfg = dict(resolved_scoring_config.get("keep_ratio_bounds") or {})
    keep_ratio_min = float(keep_ratio_cfg.get("min", 0.05))
    keep_ratio_max = float(keep_ratio_cfg.get("max", 1.0))
    keep_ratio_applied = max(keep_ratio_min, min(keep_ratio_max, float(stage1_keep_ratio)))

    adapted = [adapt_legacy_candidate(item) for item in candidates]
    if int(max_candidates) > 0:
        adapted = adapted[: int(max_candidates)]

    if not adapted:
        normalized_timeframes = normalize_strategy_timeframes(
            strategy_timeframes or CANONICAL_STRATEGY_TIMEFRAMES,
            required=CANONICAL_STRATEGY_TIMEFRAMES,
            strict_subset=True,
        )
        return {
            "schema_version": "v2",
            "generated_at": datetime.now(UTC).isoformat(),
            "base_timeframe": base_tf,
            "strategy_timeframes": normalized_timeframes,
            "symbol_universe": canonicalize_symbol_list(symbol_universe or BaseConfig.SYMBOLS),
            "split": _build_default_split(normalized_timeframes[0] if normalized_timeframes else "1m"),
            "candidates": [],
            "stage1": {
                "input_count": 0,
                "selected_count": 0,
                "keep_ratio": float(stage1_keep_ratio),
                "keep_ratio_applied": float(keep_ratio_applied),
            },
            "scoring_config": resolved_scoring_config,
        }

    discovered_timeframes = sorted(
        {
            str(row.get("strategy_timeframe") or row.get("timeframe") or "1m").strip().lower()
            for row in adapted
        }
    )
    normalized_timeframes = normalize_strategy_timeframes(
        strategy_timeframes or discovered_timeframes or CANONICAL_STRATEGY_TIMEFRAMES,
        required=CANONICAL_STRATEGY_TIMEFRAMES,
        strict_subset=True,
    )
    universe = canonicalize_symbol_list(symbol_universe or BaseConfig.SYMBOLS)

    candidate_symbols = canonicalize_symbol_list(
        itertools.chain.from_iterable(list(row.get("symbols") or []) for row in adapted)
    )
    if candidate_symbols:
        universe = canonicalize_symbol_list(list(dict.fromkeys(list(universe) + list(candidate_symbols))))

    cache, data_sources = _load_bundle_cache(
        symbols=universe,
        timeframes=normalized_timeframes,
    )
    benchmark = _benchmark_cache(cache, normalized_timeframes)

    # Stage-1 fast prefilter: evaluate on early train region approximation.
    scored_stage1: list[tuple[float, dict[str, Any]]] = []
    for row in adapted:
        result = _evaluate_candidate(
            row,
            cache=cache,
            benchmark_cache=benchmark,
            candidate_count=max(1, len(adapted)),
            scoring_config=resolved_scoring_config,
        )
        if result.get("error"):
            score = float("-inf")
        else:
            train = dict(result.get("train") or {})
            stage1_weights = dict(resolved_scoring_config.get("stage1_prefilter_weights") or {})
            score = (
                (float(stage1_weights.get("sharpe_weight", 2.0)) * float(train.get("sharpe", 0.0)))
                + (float(stage1_weights.get("return_weight", 20.0)) * float(train.get("return", 0.0)))
                - (float(stage1_weights.get("pbo_penalty", 2.0)) * float(train.get("pbo", 1.0)))
            )
        scored_stage1.append((float(score), result))

    ranked = sorted(scored_stage1, key=lambda item: item[0], reverse=True)
    keep_ratio = keep_ratio_applied
    keep_count = max(1, int(round(len(ranked) * keep_ratio)))
    stage2 = [item[1] for item in ranked[:keep_count]]

    report_candidates: list[dict[str, Any]] = []
    for result in stage2:
        row = dict(result.get("candidate") or {})
        timeframe = str(row.get("strategy_timeframe") or row.get("timeframe") or "1m")

        if result.get("error"):
            empty_metrics = _compute_metrics(
                np.asarray([], dtype=float),
                turnover=np.asarray([], dtype=float),
                exposure=np.asarray([], dtype=float),
                benchmark_returns=np.asarray([], dtype=float),
                periods_per_year=int(_PERIODS_PER_YEAR.get(timeframe, 365)),
                num_trials=max(1, len(adapted)),
            )
            hurdles, passed, hard_reject = _hurdle_fields(
                empty_metrics,
                empty_metrics,
                empty_metrics,
                scoring_config=resolved_scoring_config,
            )
            report_candidates.append(
                {
                    "candidate_id": str(row.get("candidate_id")),
                    "name": str(row.get("name")),
                    "strategy_class": str(row.get("strategy_class")),
                    "strategy": str(row.get("strategy") or row.get("strategy_class") or ""),
                    "family": str(row.get("family") or _family_from_strategy(str(row.get("strategy_class")))),
                    "strategy_timeframe": timeframe,
                    "timeframe": timeframe,
                    "symbols": canonicalize_symbol_list(list(row.get("symbols") or [])),
                    "params": dict(row.get("params") or {}),
                    "train": empty_metrics,
                    "val": empty_metrics,
                    "oos": empty_metrics,
                    "hurdle_fields": hurdles,
                    "return_streams": {"train": [], "val": [], "oos": []},
                    "cost_metrics": {"turnover": 0.0, "fee_cost": 0.0, "slippage_cost": 0.0},
                    "oos_cost_stress": {"x2": {"sharpe": 0.0, "return": 0.0}, "x3": {"sharpe": 0.0, "return": 0.0}},
                    "hard_reject": True,
                    "hard_reject_reasons": {"insufficient_data": True},
                    "selection_score": -1_000_000.0,
                    "pass": bool(passed),
                    "metadata": dict(result.get("metadata") or {}),
                }
            )
            continue

        returns = np.asarray(result.get("returns"), dtype=float)
        split_train, split_val, split_oos = _split_lengths(returns.size)

        train = dict(result.get("train") or {})
        val = dict(result.get("val") or {})
        oos = dict(result.get("oos") or {})

        hard_reject = dict(result.get("hard_reject_reasons") or {})
        hard_reject_flag = bool(hard_reject)

        candidate_payload = {
            "candidate_id": str(row.get("candidate_id")),
            "name": str(row.get("name")),
            "strategy_class": str(row.get("strategy_class")),
            "strategy": str(row.get("strategy") or row.get("strategy_class") or ""),
            "family": str(row.get("family") or _family_from_strategy(str(row.get("strategy_class")))),
            "strategy_timeframe": timeframe,
            "timeframe": timeframe,
            "symbols": canonicalize_symbol_list(list(row.get("symbols") or [])),
            "params": dict(row.get("params") or {}),
            "train": train,
            "val": val,
            "oos": oos,
            "hurdle_fields": dict(result.get("hurdle_fields") or {}),
            "return_streams": {
                "train": _series_to_stream(returns[split_train]),
                "val": _series_to_stream(returns[split_val], offset=split_train.stop or 0),
                "oos": _series_to_stream(returns[split_oos], offset=split_val.stop or 0),
            },
            "cost_metrics": {
                "turnover": float(oos.get("turnover", 0.0)),
                "fee_cost": float(result.get("metadata", {}).get("cost_rate", 0.0)),
                "slippage_cost": float(result.get("metadata", {}).get("cost_rate", 0.0) * 0.7),
            },
            "oos_cost_stress": dict(result.get("oos_cost_stress") or {}),
            "hard_reject": hard_reject_flag,
            "hard_reject_reasons": hard_reject,
            "pass": bool(result.get("pass", False)) and not hard_reject_flag,
            "metadata": dict(result.get("metadata") or {}),
        }
        candidate_payload["selection_score"] = _candidate_rank_score(
            candidate_payload,
            scoring_config=resolved_scoring_config,
        )
        report_candidates.append(candidate_payload)

    # Cross-candidate diversification diagnostics.
    oos_series = {
        row["candidate_id"]: np.asarray([point["v"] for point in row["return_streams"]["oos"]], dtype=float)
        for row in report_candidates
        if row.get("return_streams", {}).get("oos")
    }

    for row in report_candidates:
        cid = str(row.get("candidate_id"))
        base = oos_series.get(cid)
        if base is None or base.size < 8:
            row.setdefault("oos", {})["cross_candidate_corr"] = 0.0
            continue
        corr_values: list[float] = []
        for other_id, other in oos_series.items():
            if other_id == cid:
                continue
            corr_values.append(_correlation(base, other))
        row.setdefault("oos", {})["cross_candidate_corr"] = float(np.mean(corr_values)) if corr_values else 0.0

    # Stable sort by robust score.
    report_candidates.sort(key=lambda item: float(item.get("selection_score", -1e9)), reverse=True)

    split_timeframe = normalized_timeframes[0] if normalized_timeframes else "1m"
    split = _build_default_split(split_timeframe)

    return {
        "schema_version": "v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "base_timeframe": base_tf,
        "strategy_timeframes": normalized_timeframes,
        "symbol_universe": universe,
        "split": split,
        "candidates": report_candidates,
        "stage1": {
            "input_count": len(adapted),
            "selected_count": len(stage2),
            "keep_ratio": float(stage1_keep_ratio),
            "keep_ratio_applied": float(keep_ratio_applied),
        },
        "scoring_config": resolved_scoring_config,
        "data_sources": data_sources,
    }


def build_default_candidate_rows(
    *,
    symbols: Sequence[str] | None = None,
    timeframes: Sequence[str] | None = None,
    max_candidates: int = 512,
) -> list[dict[str, Any]]:
    """Build candidate rows from strategy-factory candidate library."""

    from lumina_quant.strategy_factory.candidate_library import build_binance_futures_candidates

    rows = build_binance_futures_candidates(
        symbols=symbols or BaseConfig.SYMBOLS,
        timeframes=timeframes or CANONICAL_STRATEGY_TIMEFRAMES,
    )
    out = [adapt_legacy_candidate(item.to_dict()) for item in rows]
    if int(max_candidates) > 0:
        out = out[: int(max_candidates)]
    return out


__all__ = [
    "adapt_legacy_candidate",
    "build_default_candidate_rows",
    "run_candidate_research",
]
