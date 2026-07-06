"""Advanced research alpha factors for multi-sleeve strategy pipelines.

This module centralizes the higher-level factor math used by:
- Composite trend momentum (RG_PVTM)
- Volatility-compression VWAP reversion
- Cross-asset lead/lag spillover
- Perp crowding / positioning-momentum filters (``perp_crowding_score``)

Note on "carry": ``perp_crowding_score`` measures crowding / positioning
momentum, NOT funding-carry harvest.  A positive funding level means longs PAY
funding (crowded longs), so collecting funding requires the OPPOSITE side; see
the opt-in ``true_carry_sign`` output on ``perp_crowding_score``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from lumina_quant.indicators.factory_fast import (
    rolling_range_position_latest,
    volatility_ratio_latest,
)
from lumina_quant.indicators.futures_fast import (
    trend_efficiency_latest,
    volume_shock_zscore_latest,
)
from lumina_quant.indicators.rare_event import rare_event_scores_latest
from lumina_quant.indicators.volatility import bollinger_bandwidth, choppiness_index
from lumina_quant.symbols import canonical_symbol

_METALS = frozenset({"XAU/USDT", "XAG/USDT", "XPT/USDT", "XPD/USDT"})


def _to_array(values: Sequence[float] | np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _finite_tail(arr: np.ndarray, min_size: int) -> np.ndarray | None:
    if arr.size < int(min_size):
        return None
    tail = arr[-int(min_size) :]
    if not np.all(np.isfinite(tail)):
        return None
    return tail


def _latest_zscore(arr: np.ndarray, *, window: int, eps: float = 1e-12) -> float | None:
    win = max(8, int(window))
    if arr.size < win:
        return None
    tail = arr[-win:]
    if not np.all(np.isfinite(tail)):
        return None
    hist = tail[:-1]
    latest = float(tail[-1])
    mean = float(np.mean(hist))
    std = float(np.std(hist, ddof=1)) if hist.size > 1 else 0.0
    if std <= eps:
        delta = latest - mean
        if abs(delta) <= eps:
            return 0.0
        # Preserve directional information for abrupt jumps from near-flat history.
        return float(6.0 if delta > 0.0 else -6.0)
    return float((latest - mean) / std)


def _log_returns(closes: np.ndarray) -> np.ndarray:
    if closes.size < 2:
        return np.asarray([], dtype=float)
    safe = np.clip(closes, 1e-12, np.inf)
    return np.diff(np.log(safe))


def _rolling_series_percentile(series: np.ndarray, *, window: int) -> float | None:
    win = max(8, int(window))
    if series.size < win:
        return None
    tail = series[-win:]
    if not np.all(np.isfinite(tail)):
        return None
    latest = float(tail[-1])
    rank = float(np.sum(tail <= latest)) / float(tail.size)
    return min(1.0, max(0.0, rank))


def _rolling_bandwidth_series(closes: np.ndarray, *, window: int) -> np.ndarray:
    win = max(8, int(window))
    if closes.size < win:
        return np.asarray([], dtype=float)
    values = np.asarray(closes, dtype=float)
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    sums = np.r_[0.0, np.cumsum(safe)]
    sums_sq = np.r_[0.0, np.cumsum(safe * safe)]
    counts = np.r_[0, np.cumsum(finite.astype(np.int64))]

    window_sums = sums[win:] - sums[:-win]
    window_sums_sq = sums_sq[win:] - sums_sq[:-win]
    window_counts = counts[win:] - counts[:-win]

    out = np.full(window_sums.shape, np.nan, dtype=float)
    valid = window_counts == win
    if not np.any(valid):
        return out

    means = window_sums[valid] / float(win)
    variances = (
        window_sums_sq[valid] - (window_sums[valid] * window_sums[valid] / float(win))
    ) / float(win - 1)
    valid_values = (np.abs(means) > 1e-12) & (variances > 0.0)
    computed = np.full(means.shape, np.nan, dtype=float)
    computed[valid_values] = (4.0 * np.sqrt(variances[valid_values])) / means[valid_values]
    out[valid] = computed
    return out


def _rolling_vwap_deviation_series(
    typical: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    win = max(2, int(window))
    n = min(typical.size, closes.size, volumes.size)
    if n < win:
        return np.asarray([], dtype=float)

    typical = np.asarray(typical[-n:], dtype=float)
    closes = np.asarray(closes[-n:], dtype=float)
    weights = np.clip(np.asarray(volumes[-n:], dtype=float), 0.0, np.inf)
    finite = np.isfinite(typical) & np.isfinite(closes) & np.isfinite(weights)
    safe_weights = np.where(finite, weights, 0.0)
    safe_notional = np.where(finite, typical * safe_weights, 0.0)

    volume_sums = np.r_[0.0, np.cumsum(safe_weights)]
    notional_sums = np.r_[0.0, np.cumsum(safe_notional)]
    counts = np.r_[0, np.cumsum(finite.astype(np.int64))]

    window_volume = volume_sums[win:] - volume_sums[:-win]
    window_notional = notional_sums[win:] - notional_sums[:-win]
    window_counts = counts[win:] - counts[:-win]
    close_tail = closes[win - 1 :]

    out = np.full(window_volume.shape, np.nan, dtype=float)
    valid = (window_counts == win) & (window_volume > 1e-12) & np.isfinite(close_tail)
    if np.any(valid):
        vwap = window_notional[valid] / window_volume[valid]
        positive = vwap > 1e-12
        computed = np.full(vwap.shape, np.nan, dtype=float)
        computed[positive] = (close_tail[valid][positive] / vwap[positive]) - 1.0
        out[valid] = computed
    return out


def pv_trend_score(
    close: Sequence[float] | np.ndarray,
    volume: Sequence[float] | np.ndarray,
    *,
    high: Sequence[float] | np.ndarray | None = None,
    low: Sequence[float] | np.ndarray | None = None,
    z_window: int = 144,
    momentum_legs: Sequence[int] = (8, 21, 55),
    momentum_weights: Sequence[float] = (0.5, 0.3, 0.2),
    te_window: int = 55,
    vol_fast_window: int = 8,
    vol_slow_window: int = 55,
    volume_window: int = 55,
    range_window: int = 55,
    chop_window: int = 21,
    te_min: float = 0.25,
    vr_min: float = 0.85,
    chop_max: float = 62.0,
) -> dict[str, Any]:
    """Compute regime-gated price-volume trend momentum (RG_PVTM).

    Returns a dict containing `score` and `gate`, plus diagnostics.
    """
    closes = _to_array(close)
    volumes = _to_array(volume)
    n = min(closes.size, volumes.size)
    if n < 64:
        return {
            "available": False,
            "score": 0.0,
            "gate": False,
            "mom": 0.0,
            "legs": {},
            "trend_efficiency": None,
            "volatility_ratio": None,
            "volume_shock_z": None,
            "range_position": None,
            "choppiness": None,
            "reason": "insufficient_history",
        }

    closes = closes[-n:]
    volumes = volumes[-n:]

    if len(momentum_legs) != len(momentum_weights) or not momentum_legs:
        momentum_legs = (8, 21, 55)
        momentum_weights = (0.5, 0.3, 0.2)

    legs: dict[str, float] = {}
    mom = 0.0
    weight_norm = float(sum(abs(float(weight)) for weight in momentum_weights))
    if weight_norm <= 1e-12:
        weight_norm = 1.0

    for leg, weight in zip(momentum_legs, momentum_weights, strict=True):
        lag = max(1, int(leg))
        if closes.size <= lag + 2:
            z_leg = 0.0
        else:
            ret = np.log(
                np.clip(closes[lag:], 1e-12, np.inf) / np.clip(closes[:-lag], 1e-12, np.inf)
            )
            z_leg = _latest_zscore(ret, window=max(16, int(z_window)))
            if z_leg is None:
                z_leg = 0.0
        legs[f"ret_{lag}"] = float(z_leg)
        mom += float(weight) * float(z_leg)

    mom /= weight_norm

    te = trend_efficiency_latest(closes, window=max(4, int(te_window)))
    vol_shock = volume_shock_zscore_latest(volumes, window=max(4, int(volume_window)))

    rets = _log_returns(closes)
    vol_ratio = None
    if rets.size >= max(12, int(vol_slow_window) + 2):
        vol_ratio = volatility_ratio_latest(
            rets,
            fast_window=max(4, int(vol_fast_window)),
            slow_window=max(8, int(vol_slow_window)),
        )

    highs = _to_array(high)
    lows = _to_array(low)
    range_pos = None
    chop = None
    if highs.size and lows.size:
        m = min(highs.size, lows.size, closes.size)
        if m >= 8:
            hh = highs[-m:]
            ll = lows[-m:]
            cc = closes[-m:]
            range_pos = rolling_range_position_latest(hh, ll, cc, window=max(4, int(range_window)))
            chop = choppiness_index(hh, ll, cc, period=max(4, int(chop_window)))

    te_clamped = min(1.0, max(0.0, float(te if te is not None else 0.0)))
    vshock = float(vol_shock if vol_shock is not None else 0.0)
    vr = float(vol_ratio if vol_ratio is not None else 0.0)
    rp = float(range_pos if range_pos is not None else 0.5)

    score = float(mom)
    score *= 1.0 + (0.25 * math.tanh(vshock / 2.0))
    score *= te_clamped
    score *= 1.0 + (0.10 * max(-1.0, min(1.0, vr - 1.0)))
    score *= 0.90 + (0.20 * rp)

    gate = te_clamped >= float(te_min) and vr >= float(vr_min)
    if chop is not None:
        gate = gate and float(chop) <= float(chop_max)

    return {
        "available": True,
        "score": float(score),
        "gate": bool(gate),
        "mom": float(mom),
        "legs": legs,
        "trend_efficiency": None if te is None else float(te),
        "volatility_ratio": None if vol_ratio is None else float(vol_ratio),
        "volume_shock_z": None if vol_shock is None else float(vol_shock),
        "range_position": None if range_pos is None else float(range_pos),
        "choppiness": None if chop is None else float(chop),
        "thresholds": {
            "te_min": float(te_min),
            "vr_min": float(vr_min),
            "chop_max": float(chop_max),
        },
    }


def volcomp_vwap_pressure(
    high: Sequence[float] | np.ndarray,
    low: Sequence[float] | np.ndarray,
    close: Sequence[float] | np.ndarray,
    volume: Sequence[float] | np.ndarray,
    *,
    vwap_window: int = 60,
    z_window: int = 120,
    bandwidth_window: int = 48,
    compression_percentile: float = 0.30,
    percentile_window: int = 240,
    vol_fast_window: int = 12,
    vol_slow_window: int = 60,
    compression_vol_ratio: float = 0.85,
    rare_extreme_weight: float = 0.50,
) -> dict[str, Any]:
    """Compute volatility-compression VWAP mean-reversion pressure."""
    highs = _to_array(high)
    lows = _to_array(low)
    closes = _to_array(close)
    volumes = _to_array(volume)

    n = min(highs.size, lows.size, closes.size, volumes.size)
    min_points = max(
        24,
        int(vol_slow_window) + 3,
        int(vwap_window) + 4,
    )
    if n < min_points:
        return {
            "available": False,
            "active": False,
            "score": 0.0,
            "deviation": None,
            "deviation_z": None,
            "bandwidth": None,
            "bandwidth_percentile": None,
            "volatility_ratio": None,
            "rare_event_score": None,
            "rare_extreme": None,
            "reason": "insufficient_history",
            "required_points": int(min_points),
        }

    highs = highs[-n:]
    lows = lows[-n:]
    closes = closes[-n:]
    volumes = np.clip(volumes[-n:], 0.0, np.inf)

    vw = max(8, int(vwap_window))
    typical = (highs + lows + closes) / 3.0
    dev_arr = _rolling_vwap_deviation_series(typical, closes, volumes, window=vw)
    deviation = float(dev_arr[-1]) if dev_arr.size and math.isfinite(float(dev_arr[-1])) else None
    dev_z = _latest_zscore(dev_arr, window=max(16, int(z_window)))

    bw = bollinger_bandwidth(closes, window=max(8, int(bandwidth_window)))
    bw_series = _rolling_bandwidth_series(closes, window=max(8, int(bandwidth_window)))
    bw_pct = _rolling_series_percentile(
        bw_series,
        window=max(32, int(percentile_window)),
    )

    rets = _log_returns(closes)
    vr = None
    if rets.size >= max(16, int(vol_slow_window) + 2):
        vr = volatility_ratio_latest(
            rets,
            fast_window=max(4, int(vol_fast_window)),
            slow_window=max(8, int(vol_slow_window)),
        )

    rare = rare_event_scores_latest(closes, max_points=max(4096, n + 8))
    rare_score = float(rare.composite_score) if rare is not None else 0.5
    rare_extreme = min(1.0, max(0.0, 1.0 - rare_score))

    deviation_value = float(deviation if deviation is not None else 0.0)
    deviation_z = float(dev_z if dev_z is not None else 0.0)

    active = True
    if bw_pct is not None:
        active = active and bw_pct <= float(compression_percentile)
    if vr is not None:
        active = active and vr <= float(compression_vol_ratio)

    signal = -deviation_z
    signal *= 1.0 + (float(rare_extreme_weight) * rare_extreme)
    if not active:
        signal = 0.0

    return {
        "available": True,
        "active": bool(active),
        "score": float(signal),
        "deviation": float(deviation_value),
        "deviation_z": float(deviation_z),
        "bandwidth": None if bw is None else float(bw),
        "bandwidth_percentile": None if bw_pct is None else float(bw_pct),
        "volatility_ratio": None if vr is None else float(vr),
        "rare_event_score": float(rare_score),
        "rare_extreme": float(rare_extreme),
        "thresholds": {
            "compression_percentile": float(compression_percentile),
            "compression_vol_ratio": float(compression_vol_ratio),
        },
    }


def cross_leadlag_spillover(
    price_by_symbol: Mapping[str, Sequence[float] | np.ndarray],
    *,
    leaders: Sequence[str] = ("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"),
    max_lag: int = 3,
    ridge_alpha: float = 1.0,
    window: int = 240,
) -> dict[str, Any]:
    """Estimate cross-asset lead/lag spillover via ridge lag regression.

    Metals are excluded by design.
    """
    clean_prices: dict[str, np.ndarray] = {}
    excluded: list[str] = []
    for raw_symbol, values in dict(price_by_symbol or {}).items():
        symbol = canonical_symbol(str(raw_symbol))
        if not symbol:
            continue
        if symbol in _METALS:
            excluded.append(symbol)
            continue
        arr = _to_array(values)
        if arr.size < 32:
            continue
        clean_prices[symbol] = arr

    if len(clean_prices) < 3:
        return {
            "available": False,
            "predictions": {},
            "leaders": [],
            "excluded_symbols": sorted(set(excluded)),
            "reason": "insufficient_symbols",
        }

    leader_set = [canonical_symbol(item) for item in leaders]
    leader_set = [item for item in leader_set if item and item in clean_prices]
    if not leader_set:
        # fallback: pick top-4 liquid names by history length.
        leader_set = sorted(
            clean_prices.keys(), key=lambda token: clean_prices[token].size, reverse=True
        )[:4]

    min_len = min(arr.size for arr in clean_prices.values())
    span = max(48, min(int(window), min_len))
    returns: dict[str, np.ndarray] = {}
    for symbol, prices in clean_prices.items():
        series = prices[-span:]
        ret = _log_returns(series)
        if ret.size >= 16:
            returns[symbol] = ret

    if len(returns) < 3:
        return {
            "available": False,
            "predictions": {},
            "leaders": leader_set,
            "excluded_symbols": sorted(set(excluded)),
            "reason": "insufficient_return_history",
        }

    min_ret = min(arr.size for arr in returns.values())
    lag = max(1, int(max_lag))
    min_ret = min_ret - lag - 1
    if min_ret < 12:
        return {
            "available": False,
            "predictions": {},
            "leaders": leader_set,
            "excluded_symbols": sorted(set(excluded)),
            "reason": "insufficient_lag_samples",
        }

    aligned_returns = {symbol: arr[-(min_ret + lag + 1) :] for symbol, arr in returns.items()}

    def _heuristic_prediction(follower: str) -> float:
        weights = np.exp(-np.arange(1, lag + 1, dtype=float))
        weights /= np.sum(weights)
        pred = 0.0
        used = 0
        follower_vol = float(np.std(aligned_returns[follower]))
        if follower_vol <= 1e-12:
            follower_vol = 1.0
        for leader in leader_set:
            if leader == follower or leader not in aligned_returns:
                continue
            lead_arr = aligned_returns[leader]
            lead_vol = float(np.std(lead_arr))
            scale = 1.0 if lead_vol <= 1e-12 else min(3.0, follower_vol / lead_vol)
            lagged = np.array([lead_arr[-int(l_idx)] for l_idx in range(1, lag + 1)], dtype=float)
            pred += float(np.dot(weights, lagged)) * scale
            used += 1
        if used <= 0:
            return 0.0
        return float(pred / used)

    predictions: dict[str, dict[str, Any]] = {}
    for follower in sorted(aligned_returns.keys()):
        if follower in leader_set:
            continue

        y_all = aligned_returns[follower]
        rows = min_ret
        feature_names: list[str] = []
        feature_cols: list[np.ndarray] = []
        for leader in leader_set:
            if leader == follower or leader not in aligned_returns:
                continue
            lead = aligned_returns[leader]
            for lag_idx in range(1, lag + 1):
                start = lag - lag_idx
                stop = start + rows
                feature_cols.append(lead[start:stop])
                feature_names.append(f"{leader}:lag{lag_idx}")

        if not feature_cols:
            pred = _heuristic_prediction(follower)
            denom = float(np.std(y_all))
            score = 0.0 if denom <= 1e-12 else pred / denom
            predictions[follower] = {
                "predicted_return": float(pred),
                "score": float(score),
                "method": "heuristic",
                "leaders_used": list(leader_set),
            }
            continue

        X = np.column_stack(feature_cols)
        y = y_all[lag : lag + rows]

        if X.shape[0] != y.shape[0] or X.shape[0] < 12:
            pred = _heuristic_prediction(follower)
            denom = float(np.std(y_all))
            score = 0.0 if denom <= 1e-12 else pred / denom
            predictions[follower] = {
                "predicted_return": float(pred),
                "score": float(score),
                "method": "heuristic",
                "leaders_used": list(leader_set),
            }
            continue

        try:
            x_mean = np.mean(X, axis=0)
            x_std = np.std(X, axis=0, ddof=1)
            x_std = np.where(x_std <= 1e-12, 1.0, x_std)
            Xn = (X - x_mean) / x_std
            y_mean = float(np.mean(y))
            y_std = float(np.std(y, ddof=1))
            if y_std <= 1e-12:
                raise ValueError("degenerate_y")
            yn = (y - y_mean) / y_std

            gram = Xn.T @ Xn
            reg = float(max(1e-6, ridge_alpha)) * np.eye(gram.shape[0], dtype=float)
            beta = np.linalg.solve(gram + reg, Xn.T @ yn)

            x_next = np.array(
                [
                    aligned_returns[name.split(":", 1)[0]][-(int(name.rsplit("lag", 1)[1]))]
                    for name in feature_names
                ],
                dtype=float,
            )
            x_next_n = (x_next - x_mean) / x_std
            pred_n = float(np.dot(x_next_n, beta))
            pred = (pred_n * y_std) + y_mean
            score = pred_n
            method = "ridge"
        except Exception:
            pred = _heuristic_prediction(follower)
            denom = float(np.std(y_all))
            score = 0.0 if denom <= 1e-12 else pred / denom
            method = "heuristic"

        predictions[follower] = {
            "predicted_return": float(pred),
            "score": float(score),
            "method": method,
            "leaders_used": list(leader_set),
        }

    return {
        "available": bool(predictions),
        "predictions": predictions,
        "leaders": list(leader_set),
        "excluded_symbols": sorted(set(excluded)),
        "lag_order": int(lag),
    }


def perp_crowding_score(
    *,
    funding_rate: Sequence[float] | np.ndarray | float | None = None,
    open_interest: Sequence[float] | np.ndarray | float | None = None,
    mark_price: Sequence[float] | np.ndarray | float | None = None,
    index_price: Sequence[float] | np.ndarray | float | None = None,
    basis: Sequence[float] | np.ndarray | float | None = None,
    liquidation_long_notional: Sequence[float] | np.ndarray | float | None = None,
    liquidation_short_notional: Sequence[float] | np.ndarray | float | None = None,
    window: int = 96,
    true_carry_sign: bool = False,
) -> dict[str, Any]:
    """Compute perp crowding / positioning-momentum diagnostics.

    High positive score implies crowded longs (who PAY funding when the funding
    level is positive); high negative score implies crowded shorts.  This is a
    CROWDING / positioning-momentum signal, NOT a funding-carry-harvest signal:
    to *collect* funding one must take the side OPPOSITE the funding sign.

    When ``true_carry_sign`` is True (default False => output byte-identical) the
    result gains a genuine funding-sign-aware carry hint derived from the latest
    funding level: ``carry_harvest_sign`` (+1 long / -1 short / 0 flat -- the
    side that actually collects funding), ``carry_harvest_side`` and
    ``latest_funding_rate``.  With the flag OFF no extra keys are added.
    """

    def _arr_or_scalar(value: Sequence[float] | np.ndarray | float | None) -> np.ndarray:
        if value is None:
            return np.asarray([], dtype=float)
        if isinstance(value, (int, float)):
            return np.asarray([float(value)], dtype=float)
        return _to_array(value)

    fr = _arr_or_scalar(funding_rate)
    oi = _arr_or_scalar(open_interest)
    long_liq = _arr_or_scalar(liquidation_long_notional)
    short_liq = _arr_or_scalar(liquidation_short_notional)

    if fr.size == 0 or oi.size == 0:
        return {
            "available": False,
            "score": 0.0,
            "reason": "missing_core_inputs",
            "components": {},
        }

    n = min(fr.size, oi.size)
    fr = fr[-n:]
    oi = oi[-n:]

    if basis is not None:
        basis_arr = _arr_or_scalar(basis)
        basis_arr = basis_arr[-n:] if basis_arr.size >= n else np.asarray([], dtype=float)
    elif mark_price is not None and index_price is not None:
        mark = _arr_or_scalar(mark_price)
        idx = _arr_or_scalar(index_price)
        m = min(mark.size, idx.size)
        if m > 0:
            mark = mark[-m:]
            idx = np.clip(idx[-m:], 1e-12, np.inf)
            basis_arr = (mark - idx) / idx
            if basis_arr.size > n:
                basis_arr = basis_arr[-n:]
        else:
            basis_arr = np.asarray([], dtype=float)
    else:
        basis_arr = np.asarray([], dtype=float)

    oi_delta = _log_returns(np.clip(oi, 1e-12, np.inf))
    funding_z = _latest_zscore(fr, window=max(16, int(window)))
    oi_delta_z = (
        _latest_zscore(oi_delta, window=max(16, int(window // 2))) if oi_delta.size else None
    )
    basis_z = (
        _latest_zscore(basis_arr, window=max(12, int(window // 2))) if basis_arr.size else None
    )

    liq_imbalance_z = None
    if long_liq.size and short_liq.size:
        m = min(long_liq.size, short_liq.size)
        liq_num = long_liq[-m:] - short_liq[-m:]
        liq_den = np.abs(long_liq[-m:]) + np.abs(short_liq[-m:]) + 1e-12
        liq_imbalance = liq_num / liq_den
        liq_imbalance_z = _latest_zscore(liq_imbalance, window=max(12, int(window // 2)))

    fz = float(funding_z if funding_z is not None else 0.0)
    oz = float(oi_delta_z if oi_delta_z is not None else 0.0)
    bz = float(basis_z if basis_z is not None else 0.0)
    lz = float(liq_imbalance_z if liq_imbalance_z is not None else 0.0)

    score = math.tanh((0.45 * fz) + (0.35 * oz) + (0.15 * bz) + (0.05 * lz))

    result = {
        "available": True,
        "score": float(score),
        "crowding_score": float(score),
        "components": {
            "funding_z": float(fz),
            "oi_delta_z": float(oz),
            "basis_z": float(bz),
            "liquidation_imbalance_z": float(lz),
        },
        "funding_z": float(fz),
        "oi_delta_z": float(oz),
    }
    if true_carry_sign:
        # The sign that actually COLLECTS funding: short when funding > 0, long
        # when funding < 0 (opposite the crowded side). Additive keys only; the
        # flag-OFF path above is byte-identical to the legacy output.
        latest_funding = float(fr[-1]) if fr.size else 0.0
        harvest_sign = -1.0 if latest_funding > 0.0 else (1.0 if latest_funding < 0.0 else 0.0)
        result["carry_harvest_sign"] = harvest_sign
        result["carry_harvest_side"] = (
            "SHORT" if harvest_sign < 0.0 else "LONG" if harvest_sign > 0.0 else "FLAT"
        )
        result["latest_funding_rate"] = latest_funding
    return result


__all__ = [
    "cross_leadlag_spillover",
    "perp_crowding_score",
    "pv_trend_score",
    "volcomp_vwap_pressure",
]
