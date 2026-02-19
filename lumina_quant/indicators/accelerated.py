"""Accelerated alpha-factor helpers with TA-Lib, NumPy/Polars, and optional Numba."""

from __future__ import annotations

import math

import numpy as np

try:
    import polars as pl
except Exception:  # pragma: no cover - optional runtime acceleration backend
    pl = None

try:
    import talib
except Exception:  # pragma: no cover - optional runtime acceleration backend
    talib = None

try:
    from numba import njit
except Exception:  # pragma: no cover - optional runtime acceleration backend
    njit = None

TALIB_AVAILABLE = talib is not None
POLARS_AVAILABLE = pl is not None
NUMBA_AVAILABLE = njit is not None


def _to_np(values) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float64)


def _last_finite(values) -> float | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    for idx in range(arr.size - 1, -1, -1):
        value = float(arr[idx])
        if math.isfinite(value):
            return value
    return None


def _linear_decay_latest_python(arr: np.ndarray, window: int) -> float:
    window_i = max(1, int(window))
    if arr.size < window_i:
        return float("nan")
    tail = arr[-window_i:]
    if not np.all(np.isfinite(tail)):
        return float("nan")
    weights = np.arange(1, window_i + 1, dtype=np.float64)
    denom = float(window_i * (window_i + 1) // 2)
    return float(np.dot(tail, weights) / denom)


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _linear_decay_latest_numba(arr: np.ndarray, window: int) -> float:  # pragma: no cover
        window_i = 1 if window < 1 else window
        n = arr.shape[0]
        if n < window_i:
            return np.nan
        start = n - window_i
        denom = window_i * (window_i + 1) / 2.0
        numer = 0.0
        weight = 1.0
        for idx in range(start, n):
            value = arr[idx]
            if not np.isfinite(value):
                return np.nan
            numer += weight * value
            weight += 1.0
        return numer / denom

else:

    def _linear_decay_latest_numba(arr: np.ndarray, window: int) -> float:
        return _linear_decay_latest_python(arr, window)


def linear_decay_latest(values, window: int = 10) -> float | None:
    """Return latest linearly-decayed value with optional Numba acceleration."""
    arr = _to_np(values)
    value = _linear_decay_latest_numba(arr, int(window))
    return float(value) if math.isfinite(float(value)) else None


def rolling_mean_latest_numpy(values, window: int = 20) -> float | None:
    """Return latest rolling mean via NumPy."""
    window_i = max(1, int(window))
    arr = _to_np(values)
    if arr.size < window_i:
        return None
    tail = arr[-window_i:]
    if not np.all(np.isfinite(tail)):
        return None
    return float(np.mean(tail))


def rolling_std_latest_numpy(values, window: int = 20, *, ddof: int = 1) -> float | None:
    """Return latest rolling stddev via NumPy."""
    window_i = max(2, int(window))
    arr = _to_np(values)
    if arr.size < window_i:
        return None
    tail = arr[-window_i:]
    if not np.all(np.isfinite(tail)):
        return None
    value = float(np.std(tail, ddof=int(ddof)))
    return value if math.isfinite(value) and value > 0.0 else None


def rolling_corr_latest_numpy(x_values, y_values, window: int = 20) -> float | None:
    """Return latest rolling correlation via NumPy."""
    window_i = max(2, int(window))
    x_arr = _to_np(x_values)
    y_arr = _to_np(y_values)
    n = min(x_arr.size, y_arr.size)
    if n < window_i:
        return None
    x_tail = x_arr[-window_i:]
    y_tail = y_arr[-window_i:]
    if not np.all(np.isfinite(x_tail)) or not np.all(np.isfinite(y_tail)):
        return None
    corr = float(np.corrcoef(x_tail, y_tail)[0, 1])
    return corr if math.isfinite(corr) else None


def _log_safe_ratio(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.full(numer.shape, np.nan, dtype=np.float64)
    mask = (numer > 0.0) & (denom > 0.0)
    out[mask] = np.log(numer[mask] / denom[mask])
    return out


def close_to_close_volatility(
    closes, *, window: int = 20, annualization: float = 252.0
) -> float | None:
    """Return annualized close-to-close volatility from log returns."""
    arr = _to_np(closes)
    window_i = max(2, int(window))
    if arr.size < window_i + 1:
        return None
    rets = _log_safe_ratio(arr[1:], arr[:-1])
    tail = rets[-window_i:]
    if not np.all(np.isfinite(tail)):
        return None
    value = float(np.std(tail, ddof=1) * math.sqrt(float(annualization)))
    return value if math.isfinite(value) else None


def garman_klass_volatility(
    opens,
    highs,
    lows,
    closes,
    *,
    window: int = 20,
    annualization: float = 252.0,
) -> float | None:
    """Return annualized Garman-Klass range-based volatility."""
    o = _to_np(opens)
    h = _to_np(highs)
    low_arr = _to_np(lows)
    c = _to_np(closes)
    n = min(o.size, h.size, low_arr.size, c.size)
    window_i = max(2, int(window))
    if n < window_i:
        return None
    o = o[-n:]
    h = h[-n:]
    low_arr = low_arr[-n:]
    c = c[-n:]

    log_hl = _log_safe_ratio(h, low_arr)
    log_co = _log_safe_ratio(c, o)
    var = 0.5 * (log_hl**2) - ((2.0 * math.log(2.0)) - 1.0) * (log_co**2)
    tail = var[-window_i:]
    if not np.all(np.isfinite(tail)):
        return None
    mean_var = float(np.mean(tail))
    if mean_var <= 0.0:
        return None
    value = math.sqrt(mean_var * float(annualization))
    return value if math.isfinite(value) else None


def rogers_satchell_volatility(
    opens,
    highs,
    lows,
    closes,
    *,
    window: int = 20,
    annualization: float = 252.0,
) -> float | None:
    """Return annualized Rogers-Satchell range-based volatility."""
    o = _to_np(opens)
    h = _to_np(highs)
    low_arr = _to_np(lows)
    c = _to_np(closes)
    n = min(o.size, h.size, low_arr.size, c.size)
    window_i = max(2, int(window))
    if n < window_i:
        return None
    o = o[-n:]
    h = h[-n:]
    low_arr = low_arr[-n:]
    c = c[-n:]

    log_ho = _log_safe_ratio(h, o)
    log_lo = _log_safe_ratio(low_arr, o)
    log_co = _log_safe_ratio(c, o)
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    tail = rs[-window_i:]
    if not np.all(np.isfinite(tail)):
        return None
    mean_var = float(np.mean(tail))
    if mean_var <= 0.0:
        return None
    value = math.sqrt(mean_var * float(annualization))
    return value if math.isfinite(value) else None


def yang_zhang_volatility(
    opens,
    highs,
    lows,
    closes,
    *,
    window: int = 20,
    annualization: float = 252.0,
) -> float | None:
    """Return annualized Yang-Zhang volatility estimator."""
    o = _to_np(opens)
    h = _to_np(highs)
    low_arr = _to_np(lows)
    c = _to_np(closes)
    n = min(o.size, h.size, low_arr.size, c.size)
    window_i = max(3, int(window))
    if n < window_i + 1:
        return None

    o = o[-n:]
    h = h[-n:]
    low_arr = low_arr[-n:]
    c = c[-n:]

    log_oc = _log_safe_ratio(o[1:], c[:-1])
    log_cc = _log_safe_ratio(c[1:], c[:-1])
    log_ho = _log_safe_ratio(h[1:], o[1:])
    log_lo = _log_safe_ratio(low_arr[1:], o[1:])
    log_co = _log_safe_ratio(c[1:], o[1:])
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    oc_tail = log_oc[-window_i:]
    cc_tail = log_cc[-window_i:]
    rs_tail = rs[-window_i:]
    if (
        not np.all(np.isfinite(oc_tail))
        or not np.all(np.isfinite(cc_tail))
        or not np.all(np.isfinite(rs_tail))
    ):
        return None

    sigma_o2 = float(np.var(oc_tail, ddof=1))
    sigma_c2 = float(np.var(cc_tail, ddof=1))
    sigma_rs = float(np.mean(rs_tail))

    if sigma_o2 < 0.0 or sigma_c2 < 0.0:
        return None
    k = 0.34 / (1.34 + (float(window_i + 1) / float(window_i - 1)))
    sigma2 = sigma_o2 + (k * sigma_c2) + ((1.0 - k) * sigma_rs)
    if sigma2 <= 0.0:
        return None
    value = math.sqrt(sigma2 * float(annualization))
    return value if math.isfinite(value) else None


def talib_feature_pack(
    opens,
    highs,
    lows,
    closes,
    volumes,
    *,
    rsi_period: int = 14,
    roc_period: int = 12,
    cci_period: int = 20,
    mfi_period: int = 14,
    adx_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_window: int = 20,
    bb_dev_up: float = 2.0,
    bb_dev_down: float = 2.0,
    stoch_k: int = 14,
    stoch_d: int = 3,
    stoch_smooth_k: int = 3,
    willr_period: int = 14,
    ult_short: int = 7,
    ult_medium: int = 14,
    ult_long: int = 28,
    trix_period: int = 15,
    aroon_period: int = 25,
    natr_period: int = 14,
) -> dict[str, float | None]:
    """Return a dense bundle of TA-Lib-like features with safe fallback paths."""
    o = _to_np(opens)
    h = _to_np(highs)
    low_arr = _to_np(lows)
    c = _to_np(closes)
    v = _to_np(volumes)
    n = min(o.size, h.size, low_arr.size, c.size, v.size)
    if n == 0:
        return {}

    o = o[-n:]
    h = h[-n:]
    low_arr = low_arr[-n:]
    c = c[-n:]
    v = v[-n:]

    if TALIB_AVAILABLE:
        macd, macd_signal_arr, macd_hist = talib.MACD(
            c,
            fastperiod=max(1, int(macd_fast)),
            slowperiod=max(max(1, int(macd_fast)) + 1, int(macd_slow)),
            signalperiod=max(1, int(macd_signal)),
        )
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            c,
            timeperiod=max(2, int(bb_window)),
            nbdevup=float(bb_dev_up),
            nbdevdn=float(bb_dev_down),
            matype=0,
        )
        stoch_k_arr, stoch_d_arr = talib.STOCH(
            h,
            low_arr,
            c,
            fastk_period=max(1, int(stoch_k)),
            slowk_period=max(1, int(stoch_smooth_k)),
            slowk_matype=0,
            slowd_period=max(1, int(stoch_d)),
            slowd_matype=0,
        )
        aroon_down, aroon_up = talib.AROON(h, low_arr, timeperiod=max(2, int(aroon_period)))

        return {
            "rsi": _last_finite(talib.RSI(c, timeperiod=max(1, int(rsi_period)))),
            "roc": _last_finite(talib.ROC(c, timeperiod=max(1, int(roc_period)))),
            "cci": _last_finite(talib.CCI(h, low_arr, c, timeperiod=max(2, int(cci_period)))),
            "mfi": _last_finite(talib.MFI(h, low_arr, c, v, timeperiod=max(1, int(mfi_period)))),
            "adx": _last_finite(talib.ADX(h, low_arr, c, timeperiod=max(2, int(adx_period)))),
            "plus_di": _last_finite(
                talib.PLUS_DI(h, low_arr, c, timeperiod=max(2, int(adx_period)))
            ),
            "minus_di": _last_finite(
                talib.MINUS_DI(h, low_arr, c, timeperiod=max(2, int(adx_period)))
            ),
            "macd": _last_finite(macd),
            "macd_signal": _last_finite(macd_signal_arr),
            "macd_hist": _last_finite(macd_hist),
            "bb_upper": _last_finite(bb_upper),
            "bb_middle": _last_finite(bb_middle),
            "bb_lower": _last_finite(bb_lower),
            "stoch_k": _last_finite(stoch_k_arr),
            "stoch_d": _last_finite(stoch_d_arr),
            "willr": _last_finite(talib.WILLR(h, low_arr, c, timeperiod=max(1, int(willr_period)))),
            "ultosc": _last_finite(
                talib.ULTOSC(
                    h,
                    low_arr,
                    c,
                    timeperiod1=max(1, int(ult_short)),
                    timeperiod2=max(max(1, int(ult_short)) + 1, int(ult_medium)),
                    timeperiod3=max(max(2, int(ult_medium)) + 1, int(ult_long)),
                )
            ),
            "trix": _last_finite(talib.TRIX(c, timeperiod=max(1, int(trix_period)))),
            "aroon_up": _last_finite(aroon_up),
            "aroon_down": _last_finite(aroon_down),
            "natr": _last_finite(talib.NATR(h, low_arr, c, timeperiod=max(1, int(natr_period)))),
            "obv": _last_finite(talib.OBV(c, v)),
        }

    from .bands import bollinger_bands
    from .oscillators import (
        commodity_channel_index,
        money_flow_index,
        rate_of_change,
        relative_strength_index,
        stochastic_oscillator,
        ultimate_oscillator,
        williams_r,
    )
    from .trend import (
        aroon_indicator,
        average_directional_index,
        moving_average_convergence_divergence,
        triple_exponential_average_rate_of_change,
    )
    from .volatility import atr_percent
    from .volume import on_balance_volume

    macd_value, macd_signal_value, macd_hist_value = moving_average_convergence_divergence(
        c,
        fast_period=max(1, int(macd_fast)),
        slow_period=max(max(1, int(macd_fast)) + 1, int(macd_slow)),
        signal_period=max(1, int(macd_signal)),
    )
    bb_middle, bb_upper, bb_lower = bollinger_bands(
        c,
        window=max(2, int(bb_window)),
        num_std=float(max(bb_dev_up, bb_dev_down)),
    )
    stoch_k_value, stoch_d_value = stochastic_oscillator(
        h,
        low_arr,
        c,
        k_period=max(1, int(stoch_k)),
        d_period=max(1, int(stoch_d)),
        smooth_k=max(1, int(stoch_smooth_k)),
    )
    adx_value, plus_di_value, minus_di_value = average_directional_index(
        h,
        low_arr,
        c,
        period=max(2, int(adx_period)),
    )
    aroon_up_value, aroon_down_value, _ = aroon_indicator(
        h,
        low_arr,
        period=max(2, int(aroon_period)),
    )
    trix_value, _ = triple_exponential_average_rate_of_change(
        c,
        period=max(1, int(trix_period)),
        signal_period=max(1, int(macd_signal)),
    )

    return {
        "rsi": relative_strength_index(c, period=max(1, int(rsi_period))),
        "roc": rate_of_change(c, period=max(1, int(roc_period))),
        "cci": commodity_channel_index(h, low_arr, c, period=max(2, int(cci_period))),
        "mfi": money_flow_index(h, low_arr, c, v, period=max(1, int(mfi_period))),
        "adx": adx_value,
        "plus_di": plus_di_value,
        "minus_di": minus_di_value,
        "macd": macd_value,
        "macd_signal": macd_signal_value,
        "macd_hist": macd_hist_value,
        "bb_upper": bb_upper,
        "bb_middle": bb_middle,
        "bb_lower": bb_lower,
        "stoch_k": stoch_k_value,
        "stoch_d": stoch_d_value,
        "willr": williams_r(h, low_arr, c, period=max(1, int(willr_period))),
        "ultosc": ultimate_oscillator(
            h,
            low_arr,
            c,
            short_period=max(1, int(ult_short)),
            medium_period=max(max(1, int(ult_short)) + 1, int(ult_medium)),
            long_period=max(max(2, int(ult_medium)) + 1, int(ult_long)),
        ),
        "trix": trix_value,
        "aroon_up": aroon_up_value,
        "aroon_down": aroon_down_value,
        "natr": atr_percent(h, low_arr, c, period=max(1, int(natr_period))),
        "obv": on_balance_volume(c, v),
    }


def rolling_feature_frame_polars(
    opens,
    highs,
    lows,
    closes,
    volumes,
    *,
    sma_fast: int = 10,
    sma_slow: int = 30,
    atr_window: int = 14,
    vol_short: int = 10,
    vol_long: int = 50,
    momentum_window: int = 20,
    z_window: int = 20,
) -> pl.DataFrame:
    """Return a Polars feature frame for high-throughput rolling factor pipelines."""
    if not POLARS_AVAILABLE:
        raise RuntimeError("polars is not available")

    df = pl.DataFrame(
        {
            "open": list(opens),
            "high": list(highs),
            "low": list(lows),
            "close": list(closes),
            "volume": list(volumes),
        }
    )

    sma_fast_i = max(1, int(sma_fast))
    sma_slow_i = max(sma_fast_i + 1, int(sma_slow))
    atr_i = max(1, int(atr_window))
    vol_short_i = max(2, int(vol_short))
    vol_long_i = max(vol_short_i + 1, int(vol_long))
    momentum_i = max(1, int(momentum_window))
    z_i = max(2, int(z_window))

    df = df.with_columns(
        [
            pl.col("close").pct_change().alias("ret"),
            pl.max_horizontal(
                [
                    (pl.col("high") - pl.col("low")).abs(),
                    (pl.col("high") - pl.col("close").shift(1)).abs(),
                    (pl.col("low") - pl.col("close").shift(1)).abs(),
                ]
            ).alias("tr"),
        ]
    )

    df = df.with_columns(
        [
            pl.col("close").rolling_mean(window_size=sma_fast_i).alias("sma_fast"),
            pl.col("close").rolling_mean(window_size=sma_slow_i).alias("sma_slow"),
            pl.col("tr").rolling_mean(window_size=atr_i).alias("atr"),
            pl.col("ret").rolling_std(window_size=vol_short_i).alias("vol_short"),
            pl.col("ret").rolling_std(window_size=vol_long_i).alias("vol_long"),
            ((pl.col("close") / pl.col("close").shift(momentum_i)) - 1.0).alias("momentum"),
            (
                (pl.col("close") - pl.col("close").rolling_mean(window_size=z_i))
                / pl.col("close").rolling_std(window_size=z_i)
            ).alias("close_zscore"),
            (
                (pl.col("volume") - pl.col("volume").rolling_mean(window_size=z_i))
                / pl.col("volume").rolling_std(window_size=z_i)
            ).alias("volume_zscore"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("sma_fast") - pl.col("sma_slow")).alias("ma_spread"),
            (pl.col("vol_short") / pl.col("vol_long")).alias("vol_regime"),
            (pl.col("atr") / pl.col("close")).alias("atr_percent"),
        ]
    )

    return df


def compute_fast_alpha_bundle(
    opens,
    highs,
    lows,
    closes,
    volumes,
    *,
    decay_window: int = 10,
    corr_window: int = 20,
) -> dict[str, float | None]:
    """Return a broad, tunable alpha bundle with accelerated computation paths."""
    from .formulaic_alpha import (
        alpha_001,
        alpha_002,
        alpha_005,
        alpha_020,
        alpha_025,
        alpha_042,
        alpha_043,
        alpha_101,
    )

    out = talib_feature_pack(opens, highs, lows, closes, volumes)
    vwap_proxy = [
        (float(high) + float(low) + float(close)) / 3.0
        for high, low, close in zip(highs, lows, closes, strict=False)
    ]
    out.update(
        {
            "alpha_001": alpha_001(closes),
            "alpha_002": alpha_002(closes, opens, volumes),
            "alpha_005": alpha_005(opens, closes, vwap_proxy),
            "alpha_020": alpha_020(opens, highs, lows, closes),
            "alpha_025": alpha_025(highs, closes, volumes, vwap_proxy),
            "alpha_042": alpha_042(closes, vwap_proxy),
            "alpha_043": alpha_043(closes, volumes),
            "alpha_101": alpha_101(opens, highs, lows, closes),
            "decay_close": linear_decay_latest(closes, window=max(1, int(decay_window))),
            "corr_close_volume": rolling_corr_latest_numpy(
                closes,
                volumes,
                window=max(2, int(corr_window)),
            ),
            "vol_close_to_close": close_to_close_volatility(closes),
            "vol_garman_klass": garman_klass_volatility(opens, highs, lows, closes),
            "vol_rogers_satchell": rogers_satchell_volatility(opens, highs, lows, closes),
            "vol_yang_zhang": yang_zhang_volatility(opens, highs, lows, closes),
        }
    )
    return out
