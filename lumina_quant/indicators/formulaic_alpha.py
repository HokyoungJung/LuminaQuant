"""WorldQuant-style formulaic alpha subset for OHLCV data."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise

import numpy as np
import pandas as pd

from .formulaic_definitions import ALPHA_FORMULAS
from .formulaic_operators import (
    as_window,
    decay_linear_series,
    delay_series,
    delta,
    delta_series,
    indneutralize_series,
    rank_pct,
    rank_series,
    returns_from_close,
    scale_series,
    signed_power,
    signed_power_series,
    to_series,
    ts_argmax,
    ts_argmax_series,
    ts_argmin_series,
    ts_corr_series,
    ts_correlation,
    ts_cov_series,
    ts_covariance,
    ts_max_series,
    ts_min_series,
    ts_product_series,
    ts_rank,
    ts_rank_series,
    ts_stddev,
    ts_stddev_series,
    ts_sum_series,
    where_series,
)

try:  # Optional vectorized backend.
    import polars as pl
except Exception:  # pragma: no cover - optional dependency behavior
    pl = None


def alpha_001(closes, *, std_window: int = 20, argmax_window: int = 5) -> float | None:
    """Adaptation of Formulaic Alpha#1."""
    closes_f = [float(value) for value in closes]
    returns = returns_from_close(closes_f)
    if len(closes_f) < max(int(std_window) + 1, int(argmax_window)):
        return None

    conditional: list[float] = []
    for idx in range(1, len(closes_f)):
        std_value = ts_stddev(returns[:idx], std_window)
        latest_return = returns[idx - 1]
        chosen = std_value if latest_return < 0.0 and std_value is not None else closes_f[idx]
        conditional.append(signed_power(chosen, 2.0))

    argmax_value = ts_argmax(conditional, argmax_window)
    if argmax_value is None:
        return None
    return (argmax_value / float(max(1, int(argmax_window)))) - 0.5


def alpha_002(
    closes, opens, volumes, *, delta_period: int = 2, corr_window: int = 6
) -> float | None:
    """Adaptation of Formulaic Alpha#2."""
    n = min(len(closes), len(opens), len(volumes))
    if n < max(int(delta_period) + 2, int(corr_window)):
        return None

    closes_f = [float(value) for value in closes][-n:]
    opens_f = [float(value) for value in opens][-n:]
    volumes_f = [max(1e-12, float(value)) for value in volumes][-n:]

    log_volume = [math.log(value) for value in volumes_f]
    d = max(1, int(delta_period))
    x = [log_volume[idx] - log_volume[idx - d] for idx in range(d, n)]
    y = []
    for idx in range(d, n):
        open_value = opens_f[idx]
        y.append(0.0 if abs(open_value) <= 1e-12 else (closes_f[idx] - open_value) / open_value)

    corr = ts_correlation(x, y, corr_window)
    return None if corr is None else -corr


def alpha_003(opens, volumes, *, corr_window: int = 10) -> float | None:
    """Adaptation of Formulaic Alpha#3."""
    corr = ts_correlation(opens, volumes, corr_window)
    return None if corr is None else -corr


def alpha_004(lows, *, rank_window: int = 9) -> float | None:
    """Adaptation of Formulaic Alpha#4."""
    ranked = ts_rank(lows, rank_window)
    return None if ranked is None else -ranked


def alpha_005(
    opens, closes, vwaps, *, mean_window: int = 10, rank_window: int = 20
) -> float | None:
    """Adaptation of Formulaic Alpha#5."""
    n = min(len(opens), len(closes), len(vwaps))
    if n < max(int(mean_window), int(rank_window)):
        return None

    opens_f = [float(value) for value in opens][-n:]
    closes_f = [float(value) for value in closes][-n:]
    vwaps_f = [float(value) for value in vwaps][-n:]

    spread_one = []
    w = max(1, int(mean_window))
    for idx, open_value in enumerate(opens_f):
        start = max(0, idx - w + 1)
        vwap_mean = sum(vwaps_f[start : idx + 1]) / float(idx - start + 1)
        spread_one.append(open_value - vwap_mean)
    spread_two = [
        close_value - vwap_value for close_value, vwap_value in zip(closes_f, vwaps_f, strict=False)
    ]

    rank_one = rank_pct(spread_one, rank_window)
    rank_two = rank_pct(spread_two, rank_window)
    if rank_one is None or rank_two is None:
        return None
    return rank_one * (-abs(rank_two))


def alpha_006(opens, volumes, *, corr_window: int = 10) -> float | None:
    """Adaptation of Formulaic Alpha#6."""
    corr = ts_correlation(opens, volumes, corr_window)
    return None if corr is None else -corr


def alpha_007(
    closes, volumes, *, adv_window: int = 20, delta_period: int = 7, rank_window: int = 60
) -> float | None:
    """Adaptation of Formulaic Alpha#7."""
    n = min(len(closes), len(volumes))
    if n < max(int(adv_window), int(delta_period) + 1, int(rank_window) + int(delta_period)):
        return None

    closes_f = [float(value) for value in closes][-n:]
    volumes_f = [max(0.0, float(value)) for value in volumes][-n:]

    latest_adv = sum(volumes_f[-int(adv_window) :]) / float(int(adv_window))
    if volumes_f[-1] <= latest_adv:
        return -1.0

    d = max(1, int(delta_period))
    delta_series = [closes_f[idx] - closes_f[idx - d] for idx in range(d, len(closes_f))]
    rank_value = ts_rank([abs(value) for value in delta_series], rank_window)
    if rank_value is None:
        return None

    latest_delta = delta_series[-1]
    sign_value = 1.0 if latest_delta > 0.0 else (-1.0 if latest_delta < 0.0 else 0.0)
    return (-rank_value) * sign_value


def alpha_008(
    opens, closes, *, sum_window: int = 5, delay_period: int = 10, rank_window: int = 20
) -> float | None:
    """Adaptation of Formulaic Alpha#8."""
    n = min(len(opens), len(closes))
    w = max(1, int(sum_window))
    d = max(1, int(delay_period))
    if n < w + d + 1:
        return None

    opens_f = [float(value) for value in opens][-n:]
    closes_f = [float(value) for value in closes][-n:]
    rets = returns_from_close(closes_f)

    core_series: list[float] = []
    for idx in range(w, len(rets) + 1):
        open_sum = sum(opens_f[idx - w + 1 : idx + 1])
        ret_sum = sum(rets[idx - w : idx])
        core_series.append(open_sum * ret_sum)

    if len(core_series) <= d:
        return None
    shifted = [core_series[idx] - core_series[idx - d] for idx in range(d, len(core_series))]
    ranked = rank_pct(shifted, rank_window)
    return None if ranked is None else -ranked


def alpha_009(closes, *, minmax_window: int = 5) -> float | None:
    """Adaptation of Formulaic Alpha#9."""
    window_i = max(2, int(minmax_window))
    if len(closes) < window_i + 1:
        return None
    closes_f = [float(value) for value in closes]
    deltas = [curr - prev for prev, curr in pairwise(closes_f)]
    tail = deltas[-window_i:]
    latest = deltas[-1]
    if min(tail) > 0.0 or max(tail) < 0.0:
        return latest
    return -latest


def alpha_010(closes, *, minmax_window: int = 4, rank_window: int = 20) -> float | None:
    """Adaptation of Formulaic Alpha#10."""
    window_i = max(2, int(minmax_window))
    closes_f = [float(value) for value in closes]
    if len(closes_f) < window_i + rank_window + 1:
        return None

    series: list[float] = []
    for idx in range(window_i + 1, len(closes_f) + 1):
        value = alpha_009(closes_f[:idx], minmax_window=window_i)
        if value is not None:
            series.append(value)
    return rank_pct(series, rank_window)


def alpha_011(
    closes, vwaps, volumes, *, spread_window: int = 3, delta_period: int = 3, rank_window: int = 20
) -> float | None:
    """Adaptation of Formulaic Alpha#11."""
    n = min(len(closes), len(vwaps), len(volumes))
    if n < max(int(spread_window), int(delta_period) + 1, int(rank_window)):
        return None

    closes_f = [float(value) for value in closes][-n:]
    vwaps_f = [float(value) for value in vwaps][-n:]
    volumes_f = [max(0.0, float(value)) for value in volumes][-n:]

    spread = [
        vwap_value - close_value for close_value, vwap_value in zip(closes_f, vwaps_f, strict=False)
    ]
    s = max(1, int(spread_window))
    spread_max_series = [max(spread[max(0, idx - s + 1) : idx + 1]) for idx in range(len(spread))]
    spread_min_series = [min(spread[max(0, idx - s + 1) : idx + 1]) for idx in range(len(spread))]

    d = max(1, int(delta_period))
    volume_delta = [volumes_f[idx] - volumes_f[idx - d] for idx in range(d, len(volumes_f))]

    r1 = rank_pct(spread_max_series, rank_window)
    r2 = rank_pct(spread_min_series, rank_window)
    r3 = rank_pct(volume_delta, rank_window)
    if r1 is None or r2 is None or r3 is None:
        return None
    return (r1 + r2) * r3


def alpha_012(closes, volumes) -> float | None:
    """Adaptation of Formulaic Alpha#12."""
    if min(len(closes), len(volumes)) < 2:
        return None
    close_delta = float(closes[-1]) - float(closes[-2])
    volume_delta = float(volumes[-1]) - float(volumes[-2])
    sign_value = 1.0 if volume_delta > 0.0 else (-1.0 if volume_delta < 0.0 else 0.0)
    return sign_value * (-close_delta)


def alpha_013(closes, volumes, *, cov_window: int = 5) -> float | None:
    """Adaptation of Formulaic Alpha#13."""
    cov = ts_covariance(closes, volumes, cov_window)
    return None if cov is None else -cov


def alpha_014(
    closes, opens, volumes, *, delta_period: int = 3, corr_window: int = 10
) -> float | None:
    """Adaptation of Formulaic Alpha#14."""
    returns = returns_from_close(closes)
    d = delta(returns, delta_period)
    corr = ts_correlation(opens, volumes, corr_window)
    if d is None or corr is None:
        return None
    return -d * corr


def alpha_015(highs, volumes, *, corr_window: int = 3, sum_window: int = 3) -> float | None:
    """Adaptation of Formulaic Alpha#15."""
    n = min(len(highs), len(volumes))
    c = max(2, int(corr_window))
    s = max(1, int(sum_window))
    if n < c + s:
        return None

    highs_f = [float(value) for value in highs][-n:]
    volumes_f = [max(0.0, float(value)) for value in volumes][-n:]
    corr_series: list[float] = []
    for idx in range(c, n + 1):
        corr = ts_correlation(highs_f[:idx], volumes_f[:idx], c)
        if corr is not None:
            corr_series.append(rank_pct([corr], 2) or 0.0)
    if len(corr_series) < s:
        return None
    return -sum(corr_series[-s:])


def alpha_016(highs, volumes, *, cov_window: int = 5) -> float | None:
    """Adaptation of Formulaic Alpha#16."""
    cov = ts_covariance(highs, volumes, cov_window)
    return None if cov is None else -cov


def alpha_018(closes, opens, *, std_window: int = 5, corr_window: int = 10) -> float | None:
    """Adaptation of Formulaic Alpha#18."""
    n = min(len(closes), len(opens))
    if n < max(int(std_window), int(corr_window)):
        return None

    closes_f = [float(value) for value in closes][-n:]
    opens_f = [float(value) for value in opens][-n:]
    spread = [abs(c - o) for c, o in zip(closes_f, opens_f, strict=False)]
    std_value = ts_stddev(spread, std_window)
    corr_value = ts_correlation(closes_f, opens_f, corr_window)
    if corr_value is None:
        return None
    if std_value is None:
        std_value = 0.0
    return -(std_value + (closes_f[-1] - opens_f[-1]) + corr_value)


def alpha_019(
    closes, *, delay_period: int = 7, ret_sum_window: int = 250, rank_window: int = 20
) -> float | None:
    """Adaptation of Formulaic Alpha#19."""
    closes_f = [float(value) for value in closes]
    d = max(1, int(delay_period))
    if len(closes_f) < d + 2:
        return None

    delayed = closes_f[-1 - d]
    change = closes_f[-1] - delayed
    sign_value = 1.0 if change > 0.0 else (-1.0 if change < 0.0 else 0.0)

    rets = returns_from_close(closes_f)
    w = min(len(rets), max(1, int(ret_sum_window)))
    if len(rets) < w:
        return None
    tail_sum = sum(rets[-w:])
    rank_component = rank_pct([1.0 + tail_sum], rank_window)
    if rank_component is None:
        rank_component = 0.5
    return (-sign_value) * (1.0 + rank_component)


def alpha_020(opens, highs, lows, closes, *, rank_window: int = 20) -> float | None:
    """Adaptation of Formulaic Alpha#20."""
    n = min(len(opens), len(highs), len(lows), len(closes))
    if n < 2:
        return None

    opens_f = [float(value) for value in opens][-n:]
    highs_f = [float(value) for value in highs][-n:]
    lows_f = [float(value) for value in lows][-n:]
    closes_f = [float(value) for value in closes][-n:]

    x = [opens_f[idx] - highs_f[idx - 1] for idx in range(1, n)]
    y = [opens_f[idx] - closes_f[idx - 1] for idx in range(1, n)]
    z = [opens_f[idx] - lows_f[idx - 1] for idx in range(1, n)]

    r1 = rank_pct(x, rank_window)
    r2 = rank_pct(y, rank_window)
    r3 = rank_pct(z, rank_window)
    if r1 is None or r2 is None or r3 is None:
        return None
    return (-r1) * r2 * r3


def alpha_041(highs, lows, vwaps) -> float | None:
    """Adaptation of Formulaic Alpha#41."""
    n = min(len(highs), len(lows), len(vwaps))
    if n == 0:
        return None
    high_value = float(highs[-1])
    low_value = float(lows[-1])
    vwap_value = float(vwaps[-1])
    return math.sqrt(max(high_value * low_value, 0.0)) - vwap_value


def alpha_042(closes, vwaps, *, rank_window: int = 20) -> float | None:
    """Adaptation of Formulaic Alpha#42."""
    n = min(len(closes), len(vwaps))
    if n < rank_window:
        return None
    closes_f = [float(value) for value in closes][-n:]
    vwaps_f = [float(value) for value in vwaps][-n:]
    numerator = rank_pct(
        [vwap - close for close, vwap in zip(closes_f, vwaps_f, strict=False)], rank_window
    )
    denominator = rank_pct(
        [vwap + close for close, vwap in zip(closes_f, vwaps_f, strict=False)], rank_window
    )
    if numerator is None or denominator is None or abs(denominator) <= 1e-12:
        return None
    return numerator / denominator


def alpha_043(
    closes, volumes, *, adv_window: int = 20, rank_window: int = 20, delta_period: int = 7
) -> float | None:
    """Adaptation of Formulaic Alpha#43."""
    n = min(len(closes), len(volumes))
    if n < max(int(adv_window), int(rank_window), int(delta_period) + 1):
        return None

    closes_f = [float(value) for value in closes][-n:]
    volumes_f = [max(0.0, float(value)) for value in volumes][-n:]
    a = max(1, int(adv_window))

    adv_series = []
    for idx in range(len(volumes_f)):
        start = max(0, idx - a + 1)
        adv_series.append(sum(volumes_f[start : idx + 1]) / float(idx - start + 1))
    ratio_series = [
        vol / max(adv_v, 1e-12) for vol, adv_v in zip(volumes_f, adv_series, strict=False)
    ]

    d = max(1, int(delta_period))
    delta_series = [-(closes_f[idx] - closes_f[idx - d]) for idx in range(d, len(closes_f))]

    r1 = ts_rank(ratio_series, rank_window)
    r2 = ts_rank(delta_series, 8)
    if r1 is None or r2 is None:
        return None
    return r1 * r2


def alpha_044(highs, volumes, *, corr_window: int = 5) -> float | None:
    """Adaptation of Formulaic Alpha#44."""
    corr = ts_correlation(highs, volumes, corr_window)
    return None if corr is None else -corr


def alpha_053(highs, lows, closes, *, delta_period: int = 9) -> float | None:
    """Adaptation of Formulaic Alpha#53."""
    n = min(len(highs), len(lows), len(closes))
    if n < int(delta_period) + 1:
        return None

    highs_f = [float(value) for value in highs][-n:]
    lows_f = [float(value) for value in lows][-n:]
    closes_f = [float(value) for value in closes][-n:]

    core: list[float] = []
    for high, low, close in zip(highs_f, lows_f, closes_f, strict=False):
        denom = close - low
        if abs(denom) <= 1e-12:
            core.append(0.0)
        else:
            core.append(((close - low) - (high - close)) / denom)

    d = max(1, int(delta_period))
    return -(core[-1] - core[-1 - d])


def alpha_054(opens, highs, lows, closes, *, eps: float = 1e-12) -> float | None:
    """Adaptation of Formulaic Alpha#54."""
    n = min(len(opens), len(highs), len(lows), len(closes))
    if n == 0:
        return None

    open_value = float(opens[-1])
    high_value = float(highs[-1])
    low_value = float(lows[-1])
    close_value = float(closes[-1])

    numer = -((low_value - close_value) * (open_value**5))
    denom = (low_value - high_value) * (close_value**5)
    if abs(denom) <= float(eps):
        return None
    return numer / denom


def alpha_055(
    highs, lows, closes, volumes, *, range_window: int = 12, corr_window: int = 6
) -> float | None:
    """Adaptation of Formulaic Alpha#55."""
    n = min(len(highs), len(lows), len(closes), len(volumes))
    if n < max(int(range_window), int(corr_window)):
        return None

    highs_f = [float(value) for value in highs][-n:]
    lows_f = [float(value) for value in lows][-n:]
    closes_f = [float(value) for value in closes][-n:]
    volumes_f = [max(0.0, float(value)) for value in volumes][-n:]

    w = max(2, int(range_window))
    ratio_series = []
    for idx in range(w - 1, len(closes_f)):
        high_window = highs_f[idx - w + 1 : idx + 1]
        low_window = lows_f[idx - w + 1 : idx + 1]
        high_value = max(high_window)
        low_value = min(low_window)
        denom = high_value - low_value
        if abs(denom) <= 1e-12:
            ratio_series.append(0.0)
        else:
            ratio_series.append((closes_f[idx] - low_value) / denom)

    volume_aligned = volumes_f[-len(ratio_series) :]
    corr = ts_correlation(ratio_series, volume_aligned, corr_window)
    return None if corr is None else -corr


def alpha_025(
    highs, closes, volumes, vwaps, *, adv_window: int = 20, rank_window: int = 20
) -> float | None:
    """Adaptation of Formulaic Alpha#25."""
    n = min(len(highs), len(closes), len(volumes), len(vwaps))
    if n < max(int(adv_window), int(rank_window), 2):
        return None

    highs_f = [float(value) for value in highs][-n:]
    closes_f = [float(value) for value in closes][-n:]
    volumes_f = [max(0.0, float(value)) for value in volumes][-n:]
    vwaps_f = [float(value) for value in vwaps][-n:]
    returns = returns_from_close(closes_f)

    adv_series: list[float] = []
    w = max(1, int(adv_window))
    for idx in range(n):
        start = max(0, idx - w + 1)
        dollars = [closes_f[i] * volumes_f[i] for i in range(start, idx + 1)]
        adv_series.append(sum(dollars) / float(len(dollars)))

    value_series: list[float] = []
    for idx in range(1, n):
        value_series.append(
            (-returns[idx - 1]) * adv_series[idx] * vwaps_f[idx] * (highs_f[idx] - closes_f[idx])
        )

    return rank_pct(value_series, rank_window)


def alpha_101(opens, highs, lows, closes, *, eps: float = 0.001) -> float | None:
    """Adaptation of Formulaic Alpha#101."""
    n = min(len(opens), len(highs), len(lows), len(closes))
    if n == 0:
        return None

    open_value = float(opens[-1])
    high_value = float(highs[-1])
    low_value = float(lows[-1])
    close_value = float(closes[-1])
    return (close_value - open_value) / ((high_value - low_value) + float(eps))


_MANUAL_ALPHA_IDS = {
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    18,
    19,
    20,
    25,
    41,
    42,
    43,
    44,
    53,
    54,
    55,
    101,
}


def _last_finite_value(series: pd.Series) -> float | None:
    for value in reversed(series.to_list()):
        value_f = float(value)
        if math.isfinite(value_f):
            return value_f
    return None


def _to_series(value, index: pd.Index) -> pd.Series:
    return to_series(value, index)


def _as_window(value) -> int:
    return as_window(value)


def _make_context(
    *,
    opens,
    highs,
    lows,
    closes,
    volumes,
    vwaps=None,
    returns=None,
    cap=None,
    sector=None,
    industry=None,
    subindustry=None,
) -> dict[str, pd.Series]:
    n = max(
        len(closes),
        len(opens) if opens is not None else 0,
        len(highs) if highs is not None else 0,
        len(lows) if lows is not None else 0,
        len(volumes) if volumes is not None else 0,
        len(vwaps) if vwaps is not None else 0,
        len(returns) if returns is not None else 0,
    )
    index = pd.RangeIndex(start=0, stop=n, step=1)

    close_s = _to_series(closes, index)
    open_s = _to_series(opens if opens is not None else closes, index)
    high_s = _to_series(highs if highs is not None else closes, index)
    low_s = _to_series(lows if lows is not None else closes, index)
    volume_s = _to_series(volumes if volumes is not None else [0.0] * n, index)
    if vwaps is None:
        vwap_s = (high_s + low_s + close_s) / 3.0
    else:
        vwap_s = _to_series(vwaps, index)

    if returns is None:
        ret_s = close_s.pct_change().replace([np.inf, -np.inf], np.nan)
    else:
        ret_s = _to_series(returns, index)

    cap_s = _to_series(cap if cap is not None else [1.0] * n, index)
    sector_s = _to_series(sector if sector is not None else [0.0] * n, index)
    industry_s = _to_series(industry if industry is not None else [0.0] * n, index)
    subindustry_s = _to_series(subindustry if subindustry is not None else [0.0] * n, index)

    context: dict[str, pd.Series] = {
        "open": open_s,
        "high": high_s,
        "low": low_s,
        "close": close_s,
        "volume": volume_s,
        "vwap": vwap_s,
        "returns": ret_s,
        "cap": cap_s,
        "sector": sector_s,
        "industry": industry_s,
        "subindustry": subindustry_s,
    }

    for adv_window in (5, 10, 15, 20, 30, 40, 50, 60, 81, 120, 150, 180):
        context[f"adv{adv_window}"] = (close_s * volume_s).rolling(adv_window).mean()

    return context


def _convert_ternary(expr: str) -> str:
    text = expr
    while "?" in text:
        q_pos = text.rfind("?")
        colon = -1
        nested = 0
        depth = 0
        for idx in range(q_pos + 1, len(text)):
            char = text[idx]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "?" and depth >= 0:
                nested += 1
            elif char == ":" and depth >= 0:
                if nested == 0:
                    colon = idx
                    break
                nested -= 1
        if colon < 0:
            break

        left = q_pos
        paren_depth = 0
        while left >= 0:
            char = text[left]
            if char == ")":
                paren_depth += 1
            elif char == "(":
                if paren_depth == 0:
                    break
                paren_depth -= 1
            left -= 1
        if left < 0:
            break

        right = colon
        paren_depth = 0
        while right < len(text):
            char = text[right]
            if char == "(":
                paren_depth += 1
            elif char == ")":
                if paren_depth == 0:
                    break
                paren_depth -= 1
            right += 1
        if right >= len(text):
            break

        cond = text[left + 1 : q_pos].strip()
        if_true = text[q_pos + 1 : colon].strip()
        if_false = text[colon + 1 : right].strip()
        replacement = f"where({cond}, {if_true}, {if_false})"
        text = text[:left] + replacement + text[right + 1 :]
    return text.strip()


@dataclass(frozen=True, slots=True)
class _ConstantSlot:
    key: str
    default: float


@dataclass(frozen=True, slots=True)
class _CompiledFormula:
    alpha_id: int
    source: str
    normalized: str
    tree: ast.Expression
    constant_slots: dict[int, _ConstantSlot]
    polars_capable: bool


class ParamRegistry:
    """Key/value store for runtime-tunable formula constants."""

    def __init__(self, initial: Mapping[str, float] | None = None):
        self._values: dict[str, float] = {}
        if initial:
            self.update(initial)

    def update(self, mapping: Mapping[str, float]) -> None:
        for key, value in mapping.items():
            self._values[str(key)] = float(value)

    def set(self, key: str, value: float) -> None:
        self._values[str(key)] = float(value)

    def get(self, key: str, default: float) -> float:
        return float(self._values.get(str(key), default))

    def clear_prefix(self, prefix: str = "alpha101.") -> None:
        prefix_s = str(prefix)
        keys = [key for key in self._values if key.startswith(prefix_s)]
        for key in keys:
            self._values.pop(key, None)

    def snapshot(self, prefix: str = "") -> dict[str, float]:
        prefix_s = str(prefix)
        if not prefix_s:
            return dict(self._values)
        return {key: value for key, value in self._values.items() if key.startswith(prefix_s)}


ALPHA101_PARAM_REGISTRY = ParamRegistry()
_EXEMPT_CONSTANTS = frozenset({-1.0, 0.0, 1.0})
_ADV_NAME_RE = re.compile(r"^adv\d+$")
_ROOT_NAMES = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "returns",
        "cap",
        "sector",
        "industry",
        "subindustry",
    }
)
_CALL_NAMES = frozenset(
    {
        "abs",
        "log",
        "sign",
        "rank",
        "ts_rank",
        "ts_sum",
        "ts_stddev",
        "ts_corr",
        "ts_cov",
        "ts_min",
        "ts_max",
        "ts_product",
        "delay",
        "delta",
        "ts_argmax",
        "ts_argmin",
        "decay_linear",
        "scale",
        "signed_power",
        "where",
        "indneutralize",
        "max",
        "min",
    }
)
_POLARS_CALLS = frozenset({"abs", "log", "sign", "where", "max", "min"})


def set_alpha101_param_overrides(overrides: Mapping[str, float]) -> None:
    """Update global registry for Alpha101 tunable constants."""
    ALPHA101_PARAM_REGISTRY.update(overrides)


def clear_alpha101_param_overrides(prefix: str = "alpha101.") -> None:
    """Clear global Alpha101 overrides by prefix."""
    ALPHA101_PARAM_REGISTRY.clear_prefix(prefix=prefix)


def _normalize_formula(expr: str) -> str:
    code = _convert_ternary(expr)
    replacements = {
        "Ts_ArgMax": "ts_argmax",
        "Ts_ArgMin": "ts_argmin",
        "Ts_Rank": "ts_rank",
        "SignedPower": "signed_power",
        "IndNeutralize": "indneutralize",
        "indneutralize": "indneutralize",
        "IndClass.subindustry": "subindustry",
        "IndClass.industry": "industry",
        "IndClass.sector": "sector",
        "Sign": "sign",
        "Log": "log",
    }
    for source, target in replacements.items():
        code = code.replace(source, target)

    code = re.sub(r"\bsum\(", "ts_sum(", code)
    code = re.sub(r"\bstddev\(", "ts_stddev(", code)
    code = re.sub(r"\bcorrelation\(", "ts_corr(", code)
    code = re.sub(r"\bcovariance\(", "ts_cov(", code)
    code = re.sub(r"\bproduct\(", "ts_product(", code)
    code = code.replace("^", "**")
    code = code.replace("||", "|")
    code = code.replace("&&", "&")
    return code


def _is_exempt_constant(value: float) -> bool:
    return any(abs(float(value) - float(exempt)) <= 1e-12 for exempt in _EXEMPT_CONSTANTS)


class _FormulaValidator(ast.NodeVisitor):
    def __init__(self, alpha_id: int, constant_slots: dict[int, _ConstantSlot]):
        self.alpha_id = int(alpha_id)
        self.constant_slots = constant_slots
        self.constant_index = 0

    def generic_visit(self, node):  # type: ignore[override]
        raise ValueError(f"Unsupported formula node: {type(node).__name__}")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(
            node.op, ast.Add | ast.Sub | ast.Mult | ast.Div | ast.Pow | ast.BitAnd | ast.BitOr
        ):
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, ast.UAdd | ast.USub | ast.Not | ast.Invert):
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        self.visit(node.operand)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if not isinstance(node.op, ast.And | ast.Or):
            raise ValueError(f"Unsupported boolean operator: {type(node.op).__name__}")
        for value in node.values:
            self.visit(value)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op in node.ops:
            if not isinstance(op, ast.Lt | ast.LtE | ast.Gt | ast.GtE | ast.Eq | ast.NotEq):
                raise ValueError(f"Unsupported compare operator: {type(op).__name__}")
        self.visit(node.left)
        for comparator in node.comparators:
            self.visit(comparator)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed.")
        func_name = node.func.id
        if func_name not in _CALL_NAMES:
            raise ValueError(f"Unsupported formula function: {func_name}")
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Name(self, node: ast.Name) -> None:
        name = node.id
        if name in _CALL_NAMES or name in _ROOT_NAMES or _ADV_NAME_RE.match(name):
            return
        raise ValueError(f"Unsupported formula symbol: {name}")

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, int | float) or isinstance(node.value, bool):
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
        value = float(node.value)
        if _is_exempt_constant(value):
            return
        self.constant_index += 1
        key = f"alpha101.{self.alpha_id}.const.{self.constant_index:03d}"
        self.constant_slots[id(node)] = _ConstantSlot(key=key, default=value)


def _polars_supported(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int | float) and not isinstance(node.value, bool)
    if isinstance(node, ast.Name):
        return node.id in _ROOT_NAMES or bool(_ADV_NAME_RE.match(node.id))
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.UAdd | ast.USub):
            return False
        return _polars_supported(node.operand)
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, ast.Add | ast.Sub | ast.Mult | ast.Div | ast.Pow):
            return False
        return _polars_supported(node.left) and _polars_supported(node.right)
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return False
        if not isinstance(node.ops[0], ast.Lt | ast.LtE | ast.Gt | ast.GtE | ast.Eq | ast.NotEq):
            return False
        return _polars_supported(node.left) and _polars_supported(node.comparators[0])
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            return False
        name = node.func.id
        if name not in _POLARS_CALLS:
            return False
        if name in {"max", "min"} and len(node.args) != 2:
            return False
        if name == "where" and len(node.args) != 3:
            return False
        if name in {"abs", "log", "sign"} and len(node.args) != 1:
            return False
        return all(_polars_supported(arg) for arg in node.args)
    return False


@lru_cache(maxsize=256)
def _compile_formula(alpha_id: int, expr: str) -> _CompiledFormula:
    normalized = _normalize_formula(expr)
    parsed = ast.parse(normalized, mode="eval")
    if not isinstance(parsed, ast.Expression):
        raise ValueError("Formula parser expected expression tree.")
    slots: dict[int, _ConstantSlot] = {}
    _FormulaValidator(alpha_id, slots).visit(parsed.body)
    return _CompiledFormula(
        alpha_id=int(alpha_id),
        source=expr,
        normalized=normalized,
        tree=parsed,
        constant_slots=slots,
        polars_capable=_polars_supported(parsed.body),
    )


def list_alpha101_tunable_params(alpha_id: int | None = None) -> dict[str, float]:
    """Return discovered tunable constant keys with defaults."""
    out: dict[str, float] = {}
    if alpha_id is None:
        formula_ids = sorted(ALPHA_FORMULAS)
    else:
        formula_ids = [int(alpha_id)]
    for alpha_key in formula_ids:
        formula = ALPHA_FORMULAS.get(alpha_key)
        if formula is None:
            continue
        compiled = _compile_formula(alpha_key, formula)
        for slot in compiled.constant_slots.values():
            out[slot.key] = float(slot.default)
    return out


def _resolve_constant(
    node: ast.Constant,
    *,
    compiled: _CompiledFormula,
    param_overrides: Mapping[str, float] | None,
) -> float:
    base = float(node.value)
    slot = compiled.constant_slots.get(id(node))
    if slot is None:
        return base
    if param_overrides is not None and slot.key in param_overrides:
        try:
            return float(param_overrides[slot.key])
        except (TypeError, ValueError):
            return slot.default
    return ALPHA101_PARAM_REGISTRY.get(slot.key, slot.default)


def _bool_series(value, index: pd.Index) -> pd.Series:
    return _to_series(value, index).fillna(0.0).astype(bool)


def _apply_compare(op: ast.cmpop, left, right, *, index: pd.Index):
    if isinstance(op, ast.Lt):
        return left < right
    if isinstance(op, ast.LtE):
        return left <= right
    if isinstance(op, ast.Gt):
        return left > right
    if isinstance(op, ast.GtE):
        return left >= right
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    raise ValueError(f"Unsupported comparison operator: {type(op).__name__}")


def _eval_ast_node(
    node: ast.AST,
    *,
    env: dict[str, object],
    compiled: _CompiledFormula,
    index: pd.Index,
    param_overrides: Mapping[str, float] | None,
):
    if isinstance(node, ast.Constant):
        return _resolve_constant(node, compiled=compiled, param_overrides=param_overrides)
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(
            node.operand,
            env=env,
            compiled=compiled,
            index=index,
            param_overrides=param_overrides,
        )
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.Not | ast.Invert):
            return ~_bool_series(operand, index)
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        left = _eval_ast_node(
            node.left, env=env, compiled=compiled, index=index, param_overrides=param_overrides
        )
        right = _eval_ast_node(
            node.right, env=env, compiled=compiled, index=index, param_overrides=param_overrides
        )
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        if isinstance(node.op, ast.BitAnd):
            return _bool_series(left, index) & _bool_series(right, index)
        if isinstance(node.op, ast.BitOr):
            return _bool_series(left, index) | _bool_series(right, index)
        raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
    if isinstance(node, ast.BoolOp):
        values = [
            _eval_ast_node(v, env=env, compiled=compiled, index=index, param_overrides=param_overrides)
            for v in node.values
        ]
        if not values:
            return False
        result = _bool_series(values[0], index)
        for value in values[1:]:
            if isinstance(node.op, ast.And):
                result = result & _bool_series(value, index)
            elif isinstance(node.op, ast.Or):
                result = result | _bool_series(value, index)
            else:
                raise ValueError(f"Unsupported boolean op: {type(node.op).__name__}")
        return result
    if isinstance(node, ast.Compare):
        left = _eval_ast_node(
            node.left, env=env, compiled=compiled, index=index, param_overrides=param_overrides
        )
        result = None
        for op, comparator in zip(node.ops, node.comparators, strict=False):
            right = _eval_ast_node(
                comparator, env=env, compiled=compiled, index=index, param_overrides=param_overrides
            )
            current = _apply_compare(op, left, right, index=index)
            result = current if result is None else (_bool_series(result, index) & _bool_series(current, index))
            left = right
        return result
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Unsupported callable node.")
        fn_name = node.func.id
        fn = env[fn_name]
        args = [
            _eval_ast_node(arg, env=env, compiled=compiled, index=index, param_overrides=param_overrides)
            for arg in node.args
        ]
        kwargs = {
            kw.arg: _eval_ast_node(
                kw.value,
                env=env,
                compiled=compiled,
                index=index,
                param_overrides=param_overrides,
            )
            for kw in node.keywords
            if kw.arg is not None
        }
        return fn(*args, **kwargs)
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def _build_polars_expr(
    node: ast.AST,
    *,
    compiled: _CompiledFormula,
    param_overrides: Mapping[str, float] | None,
):
    if pl is None:
        raise RuntimeError("Polars backend unavailable.")
    if isinstance(node, ast.Constant):
        return pl.lit(_resolve_constant(node, compiled=compiled, param_overrides=param_overrides))
    if isinstance(node, ast.Name):
        return pl.col(node.id)
    if isinstance(node, ast.UnaryOp):
        operand = _build_polars_expr(node.operand, compiled=compiled, param_overrides=param_overrides)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        raise NotImplementedError
    if isinstance(node, ast.BinOp):
        left = _build_polars_expr(node.left, compiled=compiled, param_overrides=param_overrides)
        right = _build_polars_expr(node.right, compiled=compiled, param_overrides=param_overrides)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left.pow(right)
        raise NotImplementedError
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise NotImplementedError
        left = _build_polars_expr(node.left, compiled=compiled, param_overrides=param_overrides)
        right = _build_polars_expr(
            node.comparators[0], compiled=compiled, param_overrides=param_overrides
        )
        op = node.ops[0]
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        raise NotImplementedError
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        args = [
            _build_polars_expr(arg, compiled=compiled, param_overrides=param_overrides)
            for arg in node.args
        ]
        if name == "abs":
            return args[0].abs()
        if name == "log":
            return args[0].log()
        if name == "sign":
            return args[0].sign()
        if name == "where":
            return pl.when(args[0]).then(args[1]).otherwise(args[2])
        if name == "max":
            return pl.max_horizontal(args[0], args[1])
        if name == "min":
            return pl.min_horizontal(args[0], args[1])
    raise NotImplementedError


def _eval_formula(
    alpha_id: int,
    expr: str,
    context: dict[str, pd.Series],
    *,
    rank_window: int = 20,
    param_overrides: Mapping[str, float] | None = None,
    vector_backend: str = "auto",
) -> float | None:
    compiled = _compile_formula(alpha_id, expr)
    index = next(iter(context.values())).index

    backend = str(vector_backend).strip().lower()
    if backend not in {"auto", "numpy", "polars"}:
        raise ValueError("vector_backend must be one of: auto, numpy, polars")

    if pl is not None and backend in {"auto", "polars"} and compiled.polars_capable:
        try:
            frame = pl.DataFrame(
                {name: np.asarray(series, dtype=float) for name, series in context.items()}
            )
            expr_pl = _build_polars_expr(
                compiled.tree.body,
                compiled=compiled,
                param_overrides=param_overrides,
            )
            out = frame.lazy().select(expr_pl.alias("__alpha__")).collect()
            values = out["__alpha__"].to_numpy()
            result_series = pd.Series(values, index=index, dtype=float)
            result_series = result_series.replace([np.inf, -np.inf], np.nan)
            latest = _last_finite_value(result_series.dropna())
            if latest is not None:
                return latest
        except Exception:
            # Fallback to numpy/pandas evaluator.
            pass

    env: dict[str, object] = {
        **context,
        "abs": np.abs,
        "log": np.log,
        "sign": np.sign,
        "rank": lambda s: rank_series(s, index=index, window=max(2, int(rank_window))),
        "ts_rank": lambda s, w: ts_rank_series(s, w, index=index),
        "ts_sum": lambda s, w: ts_sum_series(s, w, index=index),
        "ts_stddev": lambda s, w: ts_stddev_series(s, w, index=index),
        "ts_corr": lambda left, right, window: ts_corr_series(left, right, window, index=index),
        "ts_cov": lambda left, right, window: ts_cov_series(left, right, window, index=index),
        "ts_min": lambda s, w: ts_min_series(s, w, index=index),
        "ts_max": lambda s, w: ts_max_series(s, w, index=index),
        "ts_product": lambda s, w: ts_product_series(s, w, index=index),
        "delay": lambda s, p=1: delay_series(s, p, index=index),
        "delta": lambda s, p=1: delta_series(s, p, index=index),
        "ts_argmax": lambda s, w: ts_argmax_series(s, w, index=index),
        "ts_argmin": lambda s, w: ts_argmin_series(s, w, index=index),
        "decay_linear": lambda s, p=10: decay_linear_series(s, p, index=index),
        "scale": lambda s, a=1.0: scale_series(s, rank_window=int(rank_window), index=index, a=a),
        "signed_power": lambda s, p: signed_power_series(s, p, index=index),
        "where": lambda cond, left, right: where_series(cond, left, right, index=index),
        "indneutralize": lambda s, g: indneutralize_series(s, g, index=index),
        "max": np.maximum,
        "min": np.minimum,
    }
    result = _eval_ast_node(
        compiled.tree.body,
        env=env,
        compiled=compiled,
        index=index,
        param_overrides=param_overrides,
    )
    result_series = _to_series(result, index).replace([np.inf, -np.inf], np.nan)
    if result_series.empty:
        return None
    latest = _last_finite_value(result_series.dropna())
    if latest is None:
        return None
    latest_float = float(latest)
    return latest_float if math.isfinite(latest_float) else None


def compute_alpha101(
    alpha_id: int,
    *,
    opens,
    highs,
    lows,
    closes,
    volumes,
    vwaps=None,
    returns=None,
    cap=None,
    sector=None,
    industry=None,
    subindustry=None,
    rank_window: int = 20,
    param_overrides: Mapping[str, float] | None = None,
    vector_backend: str = "auto",
) -> float | None:
    """Evaluate Alpha101 formula by ID with tunable rank window and optional neutralizers."""
    alpha_int = int(alpha_id)
    formula = ALPHA_FORMULAS.get(alpha_int)
    if formula is None:
        raise ValueError(f"Unknown Alpha101 id: {alpha_id}")
    context = _make_context(
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes,
        vwaps=vwaps,
        returns=returns,
        cap=cap,
        sector=sector,
        industry=industry,
        subindustry=subindustry,
    )
    return _eval_formula(
        alpha_int,
        formula,
        context,
        rank_window=rank_window,
        param_overrides=param_overrides,
        vector_backend=vector_backend,
    )


def _make_formula_alpha(alpha_id: int):
    def _alpha(
        *,
        opens,
        highs,
        lows,
        closes,
        volumes,
        vwaps=None,
        returns=None,
        cap=None,
        sector=None,
        industry=None,
        subindustry=None,
        rank_window: int = 20,
        param_overrides: Mapping[str, float] | None = None,
        vector_backend: str = "auto",
    ) -> float | None:
        return compute_alpha101(
            alpha_id,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
            vwaps=vwaps,
            returns=returns,
            cap=cap,
            sector=sector,
            industry=industry,
            subindustry=subindustry,
            rank_window=rank_window,
            param_overrides=param_overrides,
            vector_backend=vector_backend,
        )

    _alpha.__name__ = f"alpha_{alpha_id:03d}"
    _alpha.__doc__ = f"Formula-evaluated WorldQuant Alpha#{alpha_id}."
    return _alpha


for _alpha_id in range(1, 102):
    _name = f"alpha_{_alpha_id:03d}"
    if _alpha_id in _MANUAL_ALPHA_IDS:
        continue
    if _name not in globals():
        globals()[_name] = _make_formula_alpha(_alpha_id)


ALPHA_101_FUNCTIONS = {
    alpha_id: globals()[f"alpha_{alpha_id:03d}"]
    for alpha_id in range(1, 102)
    if f"alpha_{alpha_id:03d}" in globals()
}
