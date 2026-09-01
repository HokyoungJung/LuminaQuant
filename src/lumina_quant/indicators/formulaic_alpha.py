"""WorldQuant-style formulaic alpha subset for OHLCV data."""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import pairwise

from lumina_quant.indicators.alpha101 import compiler as alpha101_compiler
from lumina_quant.indicators.alpha101 import registry as alpha101_registry

from .formulaic_operators import (
    delta,
    rank_pct,
    returns_from_close,
    signed_power,
    ts_argmax,
    ts_correlation,
    ts_covariance,
    ts_rank,
    ts_stddev,
)

pl = alpha101_compiler.pl


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


ALPHA101_PARAM_REGISTRY = alpha101_registry.ALPHA101_PARAM_REGISTRY


def set_alpha101_param_overrides(overrides: Mapping[str, float]) -> None:
    """Update global registry for Alpha101 tunable constants."""
    alpha101_registry.set_param_overrides(overrides)


def clear_alpha101_param_overrides(prefix: str = "alpha101.") -> None:
    """Clear global Alpha101 overrides by prefix."""
    alpha101_registry.clear_param_overrides(prefix=prefix)


def list_alpha101_tunable_params(alpha_id: int | None = None) -> dict[str, float]:
    """Return discovered tunable constant keys with defaults."""
    return alpha101_registry.list_tunable_params(alpha_id=alpha_id)


def get_alpha101_optuna_search_space(alpha_id: int | None = None, *, n_trials: int = 20) -> dict:
    """Build Optuna-ready search space for discovered Alpha101 constants."""
    return alpha101_registry.build_optuna_search_space(alpha_id=alpha_id, n_trials=n_trials)


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
    context = alpha101_compiler.build_context(
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
    return alpha101_registry.evaluate_alpha(
        alpha_int,
        context=context,
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
