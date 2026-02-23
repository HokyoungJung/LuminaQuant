"""Candidate-library builder for Binance futures strategy research."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from typing import Any

DEFAULT_BINANCE_TOP10_PLUS_METALS: tuple[str, ...] = (
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "TRX/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "XAU/USDT",
    "XAG/USDT",
)

DEFAULT_TIMEFRAMES: tuple[str, ...] = ("1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d")


@dataclass(frozen=True, slots=True)
class StrategyCandidate:
    """Serializable strategy-candidate definition."""

    candidate_id: str
    name: str
    family: str
    strategy_class: str
    timeframe: str
    symbols: tuple[str, ...]
    params: dict[str, Any]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "family": self.family,
            "strategy_class": self.strategy_class,
            "timeframe": self.timeframe,
            "symbols": list(self.symbols),
            "params": dict(self.params),
            "notes": self.notes,
        }


def _normalize_unique(values: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    for raw in values:
        token = str(raw).strip().upper()
        if not token:
            continue
        if token not in out:
            out.append(token)
    return tuple(out)


def _candidate_id(*, name: str, timeframe: str, params: dict[str, Any], symbols: tuple[str, ...]) -> str:
    payload = {
        "name": name,
        "timeframe": str(timeframe),
        "params": params,
        "symbols": list(symbols),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _build_symbol_pairs(symbols: Sequence[str]) -> list[tuple[str, str]]:
    topcap = [symbol for symbol in symbols if symbol not in {"XAU/USDT", "XAG/USDT"}]
    pairs: list[tuple[str, str]] = []

    anchor_pairs = [
        ("BTC/USDT", "ETH/USDT"),
        ("ETH/USDT", "SOL/USDT"),
        ("BNB/USDT", "SOL/USDT"),
        ("ADA/USDT", "XRP/USDT"),
        ("DOGE/USDT", "TRX/USDT"),
        ("AVAX/USDT", "LINK/USDT"),
        ("XAU/USDT", "XAG/USDT"),
    ]
    for left, right in anchor_pairs:
        if left in symbols and right in symbols:
            pairs.append((left, right))

    if len(pairs) < 6:
        for idx in range(max(0, len(topcap) - 1)):
            pair = (topcap[idx], topcap[idx + 1])
            if pair not in pairs:
                pairs.append(pair)
            if len(pairs) >= 6:
                break

    return pairs


def build_binance_futures_candidates(
    *,
    timeframes: Sequence[str] = DEFAULT_TIMEFRAMES,
    symbols: Sequence[str] = DEFAULT_BINANCE_TOP10_PLUS_METALS,
) -> list[StrategyCandidate]:
    """Build a diversified candidate universe for Binance futures research."""
    normalized_timeframes = tuple(str(token).strip().lower() for token in timeframes if str(token).strip())
    normalized_symbols = _normalize_unique(symbols)
    if not normalized_timeframes:
        raise ValueError("timeframes must not be empty")
    if len(normalized_symbols) < 2:
        raise ValueError("symbols must include at least two instruments")

    pair_symbols = _build_symbol_pairs(normalized_symbols)
    topcap_symbols = tuple(symbol for symbol in normalized_symbols if symbol not in {"XAU/USDT", "XAG/USDT"})
    if not topcap_symbols:
        topcap_symbols = normalized_symbols

    candidates: list[StrategyCandidate] = []

    for timeframe in normalized_timeframes:
        timeframe_slug = timeframe.replace("/", "-")

        for rsi_period, oversold, overbought in product((7, 14, 21), (22, 28, 34), (66, 72, 78)):
            name = f"rsi_reversion_{timeframe_slug}_{rsi_period}_{oversold}_{overbought}"
            params = {
                "rsi_period": int(rsi_period),
                "oversold": int(oversold),
                "overbought": int(overbought),
                "allow_short": True,
            }
            candidates.append(
                StrategyCandidate(
                    candidate_id=_candidate_id(
                        name=name,
                        timeframe=timeframe,
                        params=params,
                        symbols=normalized_symbols,
                    ),
                    name=name,
                    family="mean_reversion",
                    strategy_class="RsiStrategy",
                    timeframe=timeframe,
                    symbols=normalized_symbols,
                    params=params,
                    notes="Single-symbol RSI mean reversion sweep over the full futures universe.",
                )
            )

        for short_window, long_window in (
            (8, 34),
            (12, 48),
            (21, 64),
            (34, 96),
            (55, 144),
        ):
            name = f"ma_cross_{timeframe_slug}_{short_window}_{long_window}"
            params = {
                "short_window": int(short_window),
                "long_window": int(long_window),
                "allow_short": True,
            }
            candidates.append(
                StrategyCandidate(
                    candidate_id=_candidate_id(
                        name=name,
                        timeframe=timeframe,
                        params=params,
                        symbols=normalized_symbols,
                    ),
                    name=name,
                    family="trend",
                    strategy_class="MovingAverageCrossStrategy",
                    timeframe=timeframe,
                    symbols=normalized_symbols,
                    params=params,
                    notes="Classic momentum crossover with long/short enabled for futures.",
                )
            )

        for lookback, entry_z, exit_z in product((32, 64, 96, 128), (1.4, 1.8, 2.2), (0.35, 0.55)):
            name = f"mean_rev_std_{timeframe_slug}_{lookback}_{entry_z:.2f}_{exit_z:.2f}"
            params = {
                "window": int(lookback),
                "entry_z": float(entry_z),
                "exit_z": float(exit_z),
                "stop_loss_pct": 0.03,
                "allow_short": True,
            }
            candidates.append(
                StrategyCandidate(
                    candidate_id=_candidate_id(
                        name=name,
                        timeframe=timeframe,
                        params=params,
                        symbols=normalized_symbols,
                    ),
                    name=name,
                    family="mean_reversion",
                    strategy_class="MeanReversionStdStrategy",
                    timeframe=timeframe,
                    symbols=normalized_symbols,
                    params=params,
                    notes="Z-score reversal with symmetric long/short entries.",
                )
            )

        for lookback_bars, atr_stop_multiplier in product((20, 48, 96), (1.5, 2.2, 3.0)):
            name = f"breakout_{timeframe_slug}_{lookback_bars}_{atr_stop_multiplier:.1f}"
            params = {
                "lookback_bars": int(lookback_bars),
                "breakout_buffer": 0.001,
                "atr_window": 14,
                "atr_stop_multiplier": float(atr_stop_multiplier),
                "stop_loss_pct": 0.03,
                "allow_short": True,
            }
            candidates.append(
                StrategyCandidate(
                    candidate_id=_candidate_id(
                        name=name,
                        timeframe=timeframe,
                        params=params,
                        symbols=normalized_symbols,
                    ),
                    name=name,
                    family="breakout",
                    strategy_class="RollingBreakoutStrategy",
                    timeframe=timeframe,
                    symbols=normalized_symbols,
                    params=params,
                    notes="Donchian breakout with ATR trailing stop.",
                )
            )

        for window, entry_dev, exit_dev in product((32, 64, 96), (0.010, 0.018, 0.026), (0.002, 0.005)):
            name = f"vwap_rev_{timeframe_slug}_{window}_{entry_dev:.3f}_{exit_dev:.3f}"
            params = {
                "window": int(window),
                "entry_dev": float(entry_dev),
                "exit_dev": float(exit_dev),
                "stop_loss_pct": 0.025,
                "allow_short": True,
            }
            candidates.append(
                StrategyCandidate(
                    candidate_id=_candidate_id(
                        name=name,
                        timeframe=timeframe,
                        params=params,
                        symbols=normalized_symbols,
                    ),
                    name=name,
                    family="mean_reversion",
                    strategy_class="VwapReversionStrategy",
                    timeframe=timeframe,
                    symbols=normalized_symbols,
                    params=params,
                    notes="VWAP distance reversion with volatility-scaled exits.",
                )
            )

        for lookback_bars, rebalance_bars, signal_threshold in product(
            (8, 16, 24),
            (8, 16),
            (0.02, 0.04, 0.06),
        ):
            name = (
                f"topcap_tsmom_{timeframe_slug}_{lookback_bars}_{rebalance_bars}_"
                f"{signal_threshold:.2f}"
            )
            params = {
                "lookback_bars": int(lookback_bars),
                "rebalance_bars": int(rebalance_bars),
                "signal_threshold": float(signal_threshold),
                "stop_loss_pct": 0.07,
                "max_longs": min(6, len(topcap_symbols)),
                "max_shorts": min(5, len(topcap_symbols)),
                "min_price": 0.10,
                "btc_regime_ma": 48,
                "btc_symbol": "BTC/USDT" if "BTC/USDT" in topcap_symbols else topcap_symbols[0],
            }
            candidates.append(
                StrategyCandidate(
                    candidate_id=_candidate_id(
                        name=name,
                        timeframe=timeframe,
                        params=params,
                        symbols=topcap_symbols,
                    ),
                    name=name,
                    family="trend",
                    strategy_class="TopCapTimeSeriesMomentumStrategy",
                    timeframe=timeframe,
                    symbols=topcap_symbols,
                    params=params,
                    notes="Cross-sectional top-cap momentum rotation with BTC regime filter.",
                )
            )

        for symbol_x, symbol_y in pair_symbols:
            for entry_z, exit_z, stop_z in ((1.8, 0.45, 3.4), (2.2, 0.55, 3.9), (2.6, 0.7, 4.2)):
                pair_token = (
                    f"{symbol_x.replace('/', '').lower()}_{symbol_y.replace('/', '').lower()}"
                )
                name = f"pair_z_{timeframe_slug}_{pair_token}_{entry_z:.1f}_{exit_z:.2f}"
                params = {
                    "lookback_window": 96,
                    "hedge_window": 192,
                    "entry_z": float(entry_z),
                    "exit_z": float(exit_z),
                    "stop_z": float(stop_z),
                    "min_correlation": 0.1,
                    "max_hold_bars": 240,
                    "cooldown_bars": 4,
                    "reentry_z_buffer": 0.2,
                    "min_z_turn": 0.05,
                    "stop_loss_pct": 0.03,
                    "min_abs_beta": 0.02,
                    "max_abs_beta": 5.0,
                    "min_volume_window": 24,
                    "min_volume_ratio": 0.0,
                    "symbol_x": symbol_x,
                    "symbol_y": symbol_y,
                    "use_log_price": True,
                }
                pair_universe = tuple(sorted({symbol_x, symbol_y}))
                candidates.append(
                    StrategyCandidate(
                        candidate_id=_candidate_id(
                            name=name,
                            timeframe=timeframe,
                            params=params,
                            symbols=pair_universe,
                        ),
                        name=name,
                        family="market_neutral",
                        strategy_class="PairTradingZScoreStrategy",
                        timeframe=timeframe,
                        symbols=pair_universe,
                        params=params,
                        notes="Pair mean-reversion with beta + spread z-score gating.",
                    )
                )

    return candidates


def build_candidate_manifest(
    *,
    timeframes: Sequence[str] = DEFAULT_TIMEFRAMES,
    symbols: Sequence[str] = DEFAULT_BINANCE_TOP10_PLUS_METALS,
) -> dict[str, Any]:
    """Build a JSON-ready manifest with aggregate metadata."""
    candidates = build_binance_futures_candidates(timeframes=timeframes, symbols=symbols)

    family_counts: dict[str, int] = {}
    strategy_counts: dict[str, int] = {}
    timeframe_counts: dict[str, int] = {}

    for candidate in candidates:
        family_counts[candidate.family] = family_counts.get(candidate.family, 0) + 1
        strategy_counts[candidate.strategy_class] = strategy_counts.get(candidate.strategy_class, 0) + 1
        timeframe_counts[candidate.timeframe] = timeframe_counts.get(candidate.timeframe, 0) + 1

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "symbol_universe": list(_normalize_unique(symbols)),
        "timeframes": [str(token).strip().lower() for token in timeframes if str(token).strip()],
        "candidate_count": len(candidates),
        "family_counts": family_counts,
        "strategy_counts": strategy_counts,
        "timeframe_counts": timeframe_counts,
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
