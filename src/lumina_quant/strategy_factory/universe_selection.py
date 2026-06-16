"""Multi-factor target-pool selector for LuminaQuant research universes.

This module screens and ranks a candidate symbol universe on four orthogonal
factors -- price-position, volume, volatility, and VWAP distance -- using only
the verified pure indicator primitives from :mod:`lumina_quant.indicators`.
It is standalone and opt-in: importing it has no side effects, and the default
research pipeline never invokes it unless a caller wires the result into the
existing ``symbol_universe`` seam.

Design contract:
    * Every factor is ``None``-safe: a raw factor returns ``float | None`` and
      ``None`` means "could not compute" (short or degenerate input).
    * Cross-sectional normalization treats unscored factors as neutral.
    * On *any* exception or insufficient data the selector returns the input
      order unchanged (identity), so a missing-data run behaves like today.
    * All tunables live in module-level ``DEFAULT_*`` constants and the frozen
      :class:`UniverseSelectionConfig`.

The reused indicator primitives are:
    ``donchian_channel`` / ``bollinger_bands`` (bands), ``atr_percent``
    (volatility), ``realized_volatility`` / ``volume_zscore`` /
    ``rolling_zscore`` / ``clipped_tanh_score`` / ``finite_floats``
    (alpha_features), ``rolling_vwap`` / ``vwap_deviation`` (vwap),
    ``percentile_rank`` (oscillators), and ``rank_pct`` (formulaic_operators).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lumina_quant.indicators import (
    atr_percent,
    bollinger_bands,
    clipped_tanh_score,
    donchian_channel,
    finite_floats,
    percentile_rank,
    rank_pct,
    realized_volatility,
    rolling_vwap,
    volume_zscore,
    vwap_deviation,
)

# ``rolling_zscore`` is defined in the alpha_features submodule but not
# re-exported from the indicators package root, so import it directly.
from lumina_quant.indicators.alpha_features import rolling_zscore

if TYPE_CHECKING:
    import polars as pl

# ---------------------------------------------------------------------------
# Tunable defaults (single source of truth)
# ---------------------------------------------------------------------------
DEFAULT_FACTOR_WEIGHTS: dict[str, float] = {
    "price_position": 1.0,
    "volume": 1.5,
    "volatility": 1.0,
    "vwap": 0.8,
}
"""Default cross-sectional weight per factor (higher = more influence)."""

DEFAULT_HARD_FILTERS: dict[str, float] = {
    "min_dollar_volume": 0.0,  # quote-currency median dollar-volume floor
    "min_history_bars": 200.0,  # minimum usable bars; below -> short_history
    "vol_band_low": 0.0,  # ATR% lower tradeable bound (inclusive)
    "vol_band_high": 1.0,  # ATR% upper tradeable bound (inclusive)
}
"""Default hard-filter thresholds; symbols failing these are gated out."""

DEFAULT_WINDOWS: dict[str, int] = {
    "price_position": 60,
    "volume": 64,
    "volatility": 14,
    "vwap": 64,
}
"""Default rolling windows (in bars) per factor."""

DEFAULT_TARGET_POOL_SIZE: int = 12
"""Default number of symbols retained in the ranked pool."""

DEFAULT_VWAP_SIGN: float = -1.0
"""Sign applied to VWAP deviation magnitude. ``-1`` = mild mean-reversion."""

DEFAULT_VOLUME_SURGE_SCALE: float = 2.0
"""Tanh scale for the clipped volume-surge z-score component."""

DEFAULT_KEEP_UNSCORED: bool = True
"""When true, never drop a symbol solely because a factor was unscored."""

_EPS = 1e-12
_NEUTRAL_RANK = 0.5


# ---------------------------------------------------------------------------
# Config / result dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class UniverseSelectionConfig:
    """Configuration for multi-factor target-pool selection.

    Attributes:
        weights: Per-factor cross-sectional weights. Defaults to
            :data:`DEFAULT_FACTOR_WEIGHTS`.
        hard_filters: Threshold map gating symbols out of the pool. Defaults to
            :data:`DEFAULT_HARD_FILTERS`.
        windows: Per-factor rolling windows in bars. Defaults to
            :data:`DEFAULT_WINDOWS`.
        pool_size: Maximum number of data-selected symbols retained in the
            ranked pool. This is a soft cap: protected symbols (passed to
            :func:`select_target_pool_from_frames`) are unioned in afterward and
            may push the final pool above this size so required anchors survive.
        keep_unscored: When true, an unscored factor never on its own removes a
            symbol; it contributes a neutral rank instead.
        vwap_sign: Multiplier on the VWAP deviation magnitude. ``-1`` rewards
            proximity to fair value (mean-reversion); ``+1`` rewards distance
            (momentum sleeves).
        volume_surge_scale: Tanh scale for the clipped volume-surge component.
    """

    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_FACTOR_WEIGHTS))
    hard_filters: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_HARD_FILTERS))
    windows: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_WINDOWS))
    pool_size: int = DEFAULT_TARGET_POOL_SIZE
    keep_unscored: bool = DEFAULT_KEEP_UNSCORED
    vwap_sign: float = DEFAULT_VWAP_SIGN
    volume_surge_scale: float = DEFAULT_VOLUME_SURGE_SCALE

    def fingerprint(self) -> str:
        """Return a stable short hash of the configuration for provenance."""
        payload = {
            "weights": {key: float(val) for key, val in sorted(self.weights.items())},
            "hard_filters": {key: float(val) for key, val in sorted(self.hard_filters.items())},
            "windows": {key: int(val) for key, val in sorted(self.windows.items())},
            "pool_size": int(self.pool_size),
            "keep_unscored": bool(self.keep_unscored),
            "vwap_sign": float(self.vwap_sign),
            "volume_surge_scale": float(self.volume_surge_scale),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class SymbolFactorBreakdown:
    """Per-symbol raw and normalized factor scores plus gating result.

    Attributes:
        symbol: Canonical symbol identifier.
        raw: Raw (pre-normalization) factor values; ``None`` when unscored.
            Also carries provenance keys such as ``dollar_volume``,
            ``atr_percent``, ``history_bars``, and ``vwap_above``.
        normalized: Cross-sectionally normalized factor values in ``[0, 1]``.
        composite: Weighted composite of the normalized factors.
        passed_hard_filters: True when the symbol cleared all hard filters.
        reason: Empty string when kept; otherwise the first failing filter name
            (e.g. ``"vol_band"``, ``"short_history"``, ``"min_dollar_volume"``).
    """

    symbol: str
    raw: Mapping[str, float | None]
    normalized: Mapping[str, float]
    composite: float
    passed_hard_filters: bool
    reason: str


@dataclass(frozen=True, slots=True)
class TargetPoolResult:
    """Ranked target pool plus full per-symbol provenance.

    Attributes:
        pool: Selected symbols, ranked best to worst.
        breakdowns: Per-symbol breakdown for *every* input symbol, in ranked
            order (passing symbols first, then gated symbols).
        config_fingerprint: Stable hash of the config used, for candidate
            provenance metadata.
    """

    pool: tuple[str, ...]
    breakdowns: tuple[SymbolFactorBreakdown, ...]
    config_fingerprint: str


# ---------------------------------------------------------------------------
# Local helpers (kept in-module so they stay outside the audited tree)
# ---------------------------------------------------------------------------
def _dollar_volume(close: Sequence[float], volume: Sequence[float], window: int) -> float | None:
    """Return median close*volume over the trailing ``window`` bars.

    Args:
        close: Close-price series.
        volume: Volume series aligned to ``close``.
        window: Trailing window length in bars.

    Returns:
        Median dollar-volume, or ``None`` when there is insufficient data.
    """
    closes = finite_floats(close)
    volumes = finite_floats(volume)
    win = max(1, int(window))
    n = min(len(closes), len(volumes))
    if n < win:
        return None
    products = [closes[idx] * max(0.0, volumes[idx]) for idx in range(n - win, n)]
    if not products:
        return None
    ordered = sorted(products)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _trailing_range_percentile(close: Sequence[float], window: int) -> float | None:
    """Return the percentile rank of the latest close in its trailing window.

    Args:
        close: Close-price series.
        window: Trailing window length in bars.

    Returns:
        Percentile in ``[0, 1]``, or ``None`` when insufficient data.
    """
    return percentile_rank(finite_floats(close), window=max(2, int(window)))


# ---------------------------------------------------------------------------
# Factor functions (None-safe, pure)
# ---------------------------------------------------------------------------
def price_position_factor(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    *,
    window: int,
) -> float | None:
    """Return a mid-band-preference price-position score in ``[0, 1]``.

    Uses the Donchian channel position of the latest close. A symbol pinned at
    a range extreme (squeeze/blow-off) scores low; one near the channel midpoint
    scores high via ``1 - 2*|pos - 0.5|``.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        window: Donchian lookback window.

    Returns:
        Mid-band preference score in ``[0, 1]`` or ``None`` when degenerate.
    """
    highs = finite_floats(high)
    lows = finite_floats(low)
    closes = finite_floats(close)
    if not closes:
        return None
    upper, _mid, lower = donchian_channel(highs, lows, window=max(1, int(window)))
    if upper is None or lower is None or (upper - lower) <= _EPS:
        return None
    pos = (closes[-1] - lower) / (upper - lower)
    pos = min(1.0, max(0.0, pos))
    return 1.0 - 2.0 * abs(pos - 0.5)


def _bollinger_percent_b(close: Sequence[float], *, window: int) -> float | None:
    """Return Bollinger %B for the latest close (provenance cross-check)."""
    closes = finite_floats(close)
    if not closes:
        return None
    _middle, upper, lower = bollinger_bands(closes, window=max(2, int(window)))
    if upper is None or lower is None or (upper - lower) <= _EPS:
        return None
    return (closes[-1] - lower) / (upper - lower)


def volume_factor(
    close: Sequence[float],
    volume: Sequence[float],
    *,
    window: int,
    surge_scale: float,
    dollar_volume_rank: float | None,
) -> float | None:
    """Return the combined liquidity/surge volume score.

    The raw score blends a cross-sectional dollar-volume rank with a bounded
    volume-surge z-score: ``0.5*dv_rank + 0.5*tanh(surge_z / scale)``, mapped to
    ``[0, 1]``.

    Args:
        close: Close-price series.
        volume: Volume series aligned to ``close``.
        window: Surge z-score window.
        surge_scale: Tanh scale applied to the surge z-score.
        dollar_volume_rank: Pre-computed cross-sectional dollar-volume rank in
            ``[0, 1]``, or ``None`` when unavailable.

    Returns:
        Combined volume score in ``[0, 1]`` or ``None`` when both components are
        unavailable.
    """
    surge_z = volume_zscore(finite_floats(volume), window=max(3, int(window)))
    surge_component: float | None = None
    if surge_z is not None:
        surge_component = 0.5 * (
            clipped_tanh_score(surge_z, scale=max(_EPS, float(surge_scale))) + 1.0
        )
    dv_component = dollar_volume_rank
    if dv_component is None and surge_component is None:
        return None
    if dv_component is None:
        return surge_component
    if surge_component is None:
        return dv_component
    return 0.5 * dv_component + 0.5 * surge_component


def volatility_factor(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    *,
    window: int,
    vol_band_low: float,
    vol_band_high: float,
) -> tuple[float | None, bool]:
    """Return an inverted-U volatility score plus a tradeable-band flag.

    ATR% inside ``[vol_band_low, vol_band_high]`` scores via an inverted-U that
    peaks at the band midpoint and decays to zero at both edges. ATR% outside
    the band fails the volatility hard filter.

    Args:
        high: High-price series.
        low: Low-price series.
        close: Close-price series.
        window: ATR period.
        vol_band_low: Lower tradeable ATR% bound (inclusive).
        vol_band_high: Upper tradeable ATR% bound (inclusive).

    Returns:
        ``(score, in_band)``. ``score`` is ``None`` when ATR% is unavailable.
        ``in_band`` is False when ATR% lies outside the band.
    """
    atr_pct = atr_percent(
        finite_floats(high),
        finite_floats(low),
        finite_floats(close),
        period=max(1, int(window)),
    )
    if atr_pct is None:
        return None, True
    if atr_pct <= 0.0:
        # A dead-flat / zero-volatility instrument is untradeable; gate it out
        # even when the configured lower band is an inclusive 0.0.
        return None, False
    low_bound = float(vol_band_low)
    high_bound = float(vol_band_high)
    if atr_pct < low_bound or atr_pct > high_bound:
        return None, False
    half_width = (high_bound - low_bound) / 2.0
    if half_width <= _EPS:
        return 1.0, True
    mid = (high_bound + low_bound) / 2.0
    score = 1.0 - abs(atr_pct - mid) / half_width
    return max(0.0, score), True


def vwap_factor(
    close: Sequence[float],
    volume: Sequence[float],
    *,
    window: int,
    sign: float,
) -> tuple[float | None, bool | None]:
    """Return a VWAP-distance score plus an above/below flag.

    The raw score is ``sign * |vwap_deviation|`` so the default
    ``sign = -1`` rewards proximity to the rolling VWAP (mild mean-reversion).

    Args:
        close: Close-price series.
        volume: Volume series aligned to ``close``.
        window: Rolling VWAP window.
        sign: Multiplier applied to the deviation magnitude.

    Returns:
        ``(score, above)``. ``score`` is ``None`` when VWAP cannot be computed.
        ``above`` is True when the latest close is above VWAP, else False, or
        ``None`` when undetermined.
    """
    closes = finite_floats(close)
    volumes = finite_floats(volume)
    if not closes:
        return None, None
    vwap_value = rolling_vwap(closes, volumes, max(2, int(window)))
    dev = vwap_deviation(closes[-1], vwap_value)
    if dev is None:
        return None, None
    above = dev > 0.0
    return float(sign) * abs(dev), above


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def _rank_normalize(values: Mapping[str, float | None]) -> dict[str, float]:
    """Cross-sectionally rank-normalize a factor map to ``[0, 1]``.

    ``None`` values map to a neutral mid-rank. With a single scored value the
    result is the neutral mid-rank (no spread to rank against).

    Args:
        values: Symbol -> raw factor value (or ``None``).

    Returns:
        Symbol -> normalized value in ``[0, 1]``.
    """
    scored = {
        symbol: float(value)
        for symbol, value in values.items()
        if value is not None and math.isfinite(float(value))
    }
    if len(scored) <= 1:
        return dict.fromkeys(values, _NEUTRAL_RANK)
    ordered = sorted(scored.items(), key=lambda item: (item[1], item[0]))
    denom = float(len(ordered) - 1)
    # Fractional ranking: tied raw values share the average of their positions
    # so equal inputs normalize identically (no symbol-name leakage).
    rank_of: dict[str, float] = {}
    idx = 0
    n = len(ordered)
    while idx < n:
        end = idx
        while end + 1 < n and ordered[end + 1][1] == ordered[idx][1]:
            end += 1
        avg_position = (idx + end) / 2.0
        normalized_rank = avg_position / denom
        for pos in range(idx, end + 1):
            rank_of[ordered[pos][0]] = normalized_rank
        idx = end + 1
    out: dict[str, float] = {}
    for symbol in values:
        out[symbol] = rank_of.get(symbol, _NEUTRAL_RANK)
    return out


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------
class TargetPoolSelector:
    """Score a symbol universe on four factors and rank it deterministically."""

    def __init__(self, config: UniverseSelectionConfig | None = None) -> None:
        """Initialize the selector.

        Args:
            config: Optional configuration; defaults to
                :class:`UniverseSelectionConfig` with module defaults.
        """
        self._config = config or UniverseSelectionConfig()

    @property
    def config(self) -> UniverseSelectionConfig:
        """Return the active configuration."""
        return self._config

    def _window(self, key: str) -> int:
        return int(self._config.windows.get(key, DEFAULT_WINDOWS[key]))

    def _hard_filter(self, key: str) -> float:
        return float(self._config.hard_filters.get(key, DEFAULT_HARD_FILTERS[key]))

    def score_symbol(
        self,
        symbol: str,
        *,
        high: Sequence[float],
        low: Sequence[float],
        close: Sequence[float],
        volume: Sequence[float],
        dollar_volume_rank: float | None = None,
    ) -> SymbolFactorBreakdown:
        """Compute None-safe raw factors and hard-filter gating for one symbol.

        Cross-sectional normalization is applied later by :meth:`select`; this
        method fills ``normalized`` with neutral placeholders.

        Args:
            symbol: Canonical symbol identifier.
            high: High-price series.
            low: Low-price series.
            close: Close-price series.
            volume: Volume series aligned to ``close``.
            dollar_volume_rank: Optional pre-computed cross-sectional
                dollar-volume rank in ``[0, 1]``.

        Returns:
            A :class:`SymbolFactorBreakdown` with raw factors and gating result.
        """
        closes = finite_floats(close)
        history_bars = len(closes)
        dollar_volume = _dollar_volume(close, volume, self._window("volume"))

        price_pos = price_position_factor(high, low, close, window=self._window("price_position"))
        volume_score = volume_factor(
            close,
            volume,
            window=self._window("volume"),
            surge_scale=self._config.volume_surge_scale,
            dollar_volume_rank=dollar_volume_rank,
        )
        vol_score, in_band = volatility_factor(
            high,
            low,
            close,
            window=self._window("volatility"),
            vol_band_low=self._hard_filter("vol_band_low"),
            vol_band_high=self._hard_filter("vol_band_high"),
        )
        vwap_score, vwap_above = vwap_factor(
            close,
            volume,
            window=self._window("vwap"),
            sign=self._config.vwap_sign,
        )
        atr_pct = atr_percent(
            finite_floats(high),
            finite_floats(low),
            closes,
            period=self._window("volatility"),
        )

        passed, reason = self._gate(
            history_bars=history_bars,
            dollar_volume=dollar_volume,
            in_band=in_band,
        )

        raw: dict[str, float | None] = {
            "price_position": price_pos,
            "volume": volume_score,
            "volatility": vol_score,
            "vwap": vwap_score,
            "dollar_volume": dollar_volume,
            "atr_percent": atr_pct,
            "history_bars": float(history_bars),
            "vwap_above": None if vwap_above is None else float(vwap_above),
            "bollinger_percent_b": _bollinger_percent_b(
                close, window=self._window("price_position")
            ),
            "realized_volatility": realized_volatility(closes, window=self._window("volatility")),
            "volume_surge_z": rolling_zscore(
                finite_floats(volume), window=max(3, self._window("volume"))
            ),
            "close_trailing_rank": rank_pct(
                finite_floats(close), window=max(2, self._window("price_position"))
            ),
            "trailing_range_percentile": _trailing_range_percentile(
                close, self._window("price_position")
            ),
        }
        normalized: dict[str, float] = {
            "price_position": _NEUTRAL_RANK,
            "volume": _NEUTRAL_RANK,
            "volatility": _NEUTRAL_RANK,
            "vwap": _NEUTRAL_RANK,
        }
        return SymbolFactorBreakdown(
            symbol=symbol,
            raw=raw,
            normalized=normalized,
            composite=0.0,
            passed_hard_filters=passed,
            reason=reason,
        )

    def _gate(
        self,
        *,
        history_bars: int,
        dollar_volume: float | None,
        in_band: bool,
    ) -> tuple[bool, str]:
        """Apply hard filters in deterministic order, returning first failure."""
        if history_bars < int(self._hard_filter("min_history_bars")):
            return False, "short_history"
        if not in_band:
            return False, "vol_band"
        min_dv = self._hard_filter("min_dollar_volume")
        if min_dv > 0.0 and (dollar_volume is None or dollar_volume < min_dv):
            return False, "min_dollar_volume"
        return True, ""

    def select(
        self,
        series: Mapping[str, Mapping[str, Sequence[float]]],
    ) -> TargetPoolResult:
        """Score all symbols, cross-sectionally normalize, weight, rank, gate.

        Args:
            series: Symbol -> mapping with ``high``/``low``/``close``/``volume``
                sequences.

        Returns:
            A :class:`TargetPoolResult`. On any failure the result preserves the
            input order (identity).
        """
        symbols = list(series.keys())
        if not symbols:
            return TargetPoolResult(
                pool=(), breakdowns=(), config_fingerprint=self._config.fingerprint()
            )
        try:
            return self._select_impl(series, symbols)
        except Exception:
            return self._identity_result(symbols)

    def _select_impl(
        self,
        series: Mapping[str, Mapping[str, Sequence[float]]],
        symbols: list[str],
    ) -> TargetPoolResult:
        dv_window = self._window("volume")
        raw_dv: dict[str, float | None] = {}
        for symbol in symbols:
            frame = series[symbol]
            raw_dv[symbol] = _dollar_volume(
                frame.get("close", ()), frame.get("volume", ()), dv_window
            )
        dv_ranks = _rank_normalize(raw_dv)

        breakdowns: list[SymbolFactorBreakdown] = []
        for symbol in symbols:
            frame = series[symbol]
            breakdowns.append(
                self.score_symbol(
                    symbol,
                    high=frame.get("high", ()),
                    low=frame.get("low", ()),
                    close=frame.get("close", ()),
                    volume=frame.get("volume", ()),
                    dollar_volume_rank=dv_ranks.get(symbol),
                )
            )

        normalized_by_factor = {
            factor: _rank_normalize({bd.symbol: bd.raw.get(factor) for bd in breakdowns})
            for factor in ("price_position", "volume", "volatility", "vwap")
        }

        finalized: list[SymbolFactorBreakdown] = []
        for breakdown in breakdowns:
            normalized = {
                factor: normalized_by_factor[factor][breakdown.symbol]
                for factor in ("price_position", "volume", "volatility", "vwap")
            }
            composite = sum(
                float(self._config.weights.get(factor, 0.0)) * value
                for factor, value in normalized.items()
            )
            finalized.append(
                SymbolFactorBreakdown(
                    symbol=breakdown.symbol,
                    raw=breakdown.raw,
                    normalized=normalized,
                    composite=composite,
                    passed_hard_filters=breakdown.passed_hard_filters,
                    reason=breakdown.reason,
                )
            )

        ranked = sorted(finalized, key=self._sort_key)
        passing = [bd for bd in ranked if bd.passed_hard_filters]
        if passing:
            pool_source = passing
        elif self._config.keep_unscored:
            pool_source = ranked
        else:
            pool_source = []
        pool = tuple(bd.symbol for bd in pool_source[: max(0, int(self._config.pool_size))])
        return TargetPoolResult(
            pool=pool,
            breakdowns=tuple(ranked),
            config_fingerprint=self._config.fingerprint(),
        )

    def _sort_key(self, breakdown: SymbolFactorBreakdown) -> tuple[bool, float, float, float, str]:
        """Return the deterministic sort key.

        Ordering: passing-first, composite desc, dollar-volume rank desc,
        ATR% asc, symbol lexicographic asc.
        """
        dv_rank = breakdown.normalized.get("volume", _NEUTRAL_RANK)
        atr_pct = breakdown.raw.get("atr_percent")
        atr_sort = float(atr_pct) if atr_pct is not None else math.inf
        return (
            not breakdown.passed_hard_filters,
            -breakdown.composite,
            -float(dv_rank),
            atr_sort,
            breakdown.symbol,
        )

    def _identity_result(self, symbols: Sequence[str]) -> TargetPoolResult:
        """Return an identity result preserving the input symbol order."""
        breakdowns = tuple(
            SymbolFactorBreakdown(
                symbol=symbol,
                raw={},
                normalized={},
                composite=0.0,
                passed_hard_filters=True,
                reason="",
            )
            for symbol in symbols
        )
        return TargetPoolResult(
            pool=tuple(symbols),
            breakdowns=breakdowns,
            config_fingerprint=self._config.fingerprint(),
        )

    def select_static(self, symbols: Sequence[str]) -> TargetPoolResult:
        """Data-free fallback: identity ordering with availability dedupe.

        Used when no usable bars are available. Preserves input order, dropping
        only empty/duplicate identifiers so pair builders never starve.

        Args:
            symbols: Candidate symbol identifiers.

        Returns:
            A :class:`TargetPoolResult` with the deduplicated input order.
        """
        seen: set[str] = set()
        ordered: list[str] = []
        for symbol in symbols or ():
            text = str(symbol).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
        return self._identity_result(ordered)


# ---------------------------------------------------------------------------
# Functional adapter (research_runner / CLI seam)
# ---------------------------------------------------------------------------
_OHLCV_COLUMNS: tuple[str, ...] = ("high", "low", "close", "volume")


def _extract_columns(frame: pl.DataFrame) -> dict[str, list[float]] | None:
    """Extract case-insensitive ``high``/``low``/``close``/``volume`` columns.

    Args:
        frame: A Polars OHLCV frame (schema ``datetime, open, high, low,
            close, volume``).

    Returns:
        Mapping of factor inputs to float lists, or ``None`` when a required
        column is absent or extraction fails.
    """
    try:
        lookup = {str(name).lower(): name for name in frame.columns}
    except Exception:
        return None
    resolved: dict[str, str] = {}
    for wanted in _OHLCV_COLUMNS:
        actual = lookup.get(wanted)
        if actual is None:
            return None
        resolved[wanted] = actual
    try:
        # Drop any row where ANY of the four selected columns is null so the
        # extracted series stay row-aligned (high[i] corresponds to close[i],
        # low[i], volume[i]). Stripping nulls per-column independently would
        # desynchronize the lists on real frames with gaps in one column.
        cleaned = frame.select([resolved[wanted] for wanted in _OHLCV_COLUMNS]).drop_nulls()
        out: dict[str, list[float]] = {
            wanted: [float(value) for value in cleaned.get_column(resolved[wanted]).to_list()]
            for wanted in _OHLCV_COLUMNS
        }
    except Exception:
        return None
    return out


def select_target_pool_from_frames(
    loaded: Mapping[str, pl.DataFrame],
    *,
    config: UniverseSelectionConfig | None = None,
    fallback_symbols: Sequence[str] | None = None,
    protected: Sequence[str] | None = None,
) -> TargetPoolResult:
    """Run the selector over already-loaded Polars OHLCV frames.

    Extracts ``high``/``low``/``close``/``volume`` columns defensively from each
    frame. On empty or degenerate input it returns
    :meth:`TargetPoolSelector.select_static` over ``fallback_symbols`` (or the
    ``loaded`` keys). Protected symbols are always retained in the pool, even if
    they would otherwise be gated out, so pair/cross builders never empty.

    Args:
        loaded: Symbol -> Polars OHLCV frame, as produced by the bundle loader.
        config: Optional selection configuration.
        fallback_symbols: Symbols to use for the static fallback path. Defaults
            to the ``loaded`` keys.
        protected: Symbols that must always survive screening (e.g. pair
            anchors). They are unioned into the pool, ranked ahead of the
            data-driven tail.

    Returns:
        A :class:`TargetPoolResult`. Never raises; degrades to a static result.
    """
    selector = TargetPoolSelector(config)
    fallback = list(fallback_symbols) if fallback_symbols is not None else list(loaded or {})
    protected_list = [str(sym).strip() for sym in (protected or ()) if str(sym).strip()]

    if not loaded:
        return _apply_protected(selector.select_static(fallback), protected_list)

    series: dict[str, dict[str, list[float]]] = {}
    try:
        for symbol, frame in loaded.items():
            columns = _extract_columns(frame)
            if columns is None or not columns["close"]:
                continue
            series[str(symbol)] = columns
    except Exception:
        return _apply_protected(selector.select_static(fallback), protected_list)

    if not series:
        return _apply_protected(selector.select_static(fallback), protected_list)

    result = selector.select(series)
    return _apply_protected(result, protected_list)


def _apply_protected(result: TargetPoolResult, protected: Sequence[str]) -> TargetPoolResult:
    """Union protected symbols into the pool, preserving order and ranking.

    Protected symbols already present keep their rank; missing ones are appended
    to the front-of-tail so screening never removes a required anchor.
    """
    if not protected:
        return result
    pool = list(result.pool)
    pool_set = set(pool)
    known = {bd.symbol for bd in result.breakdowns}
    additions = [sym for sym in protected if sym not in pool_set and sym in known]
    brand_new = [sym for sym in protected if sym not in pool_set and sym not in known]
    if not additions and not brand_new:
        return result
    new_pool = tuple(pool + additions + brand_new)
    new_breakdowns = list(result.breakdowns)
    for sym in brand_new:
        new_breakdowns.append(
            SymbolFactorBreakdown(
                symbol=sym,
                raw={},
                normalized={},
                composite=0.0,
                passed_hard_filters=True,
                reason="protected",
            )
        )
    return TargetPoolResult(
        pool=new_pool,
        breakdowns=tuple(new_breakdowns),
        config_fingerprint=result.config_fingerprint,
    )


def breakdown_to_dict(breakdown: SymbolFactorBreakdown) -> dict[str, Any]:
    """Return a JSON-serializable view of a :class:`SymbolFactorBreakdown`."""
    return {
        "symbol": breakdown.symbol,
        "raw": {key: value for key, value in breakdown.raw.items()},
        "normalized": dict(breakdown.normalized),
        "composite": breakdown.composite,
        "passed_hard_filters": breakdown.passed_hard_filters,
        "reason": breakdown.reason,
    }


def result_to_dict(result: TargetPoolResult) -> dict[str, Any]:
    """Return a JSON-serializable view of a :class:`TargetPoolResult`."""
    return {
        "pool": list(result.pool),
        "config_fingerprint": result.config_fingerprint,
        "breakdowns": [breakdown_to_dict(bd) for bd in result.breakdowns],
    }


__all__ = [
    "DEFAULT_FACTOR_WEIGHTS",
    "DEFAULT_HARD_FILTERS",
    "DEFAULT_KEEP_UNSCORED",
    "DEFAULT_TARGET_POOL_SIZE",
    "DEFAULT_VOLUME_SURGE_SCALE",
    "DEFAULT_VWAP_SIGN",
    "DEFAULT_WINDOWS",
    "SymbolFactorBreakdown",
    "TargetPoolResult",
    "TargetPoolSelector",
    "UniverseSelectionConfig",
    "breakdown_to_dict",
    "price_position_factor",
    "result_to_dict",
    "select_target_pool_from_frames",
    "volatility_factor",
    "volume_factor",
    "vwap_factor",
]
