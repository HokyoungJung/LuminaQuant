"""Candidate-library builder for advanced multi-sleeve quant research."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

from lumina_quant.symbols import (
    CANONICAL_STRATEGY_TIMEFRAMES,
    canonicalize_symbol_list,
    normalize_strategy_timeframes,
)

_DEFAULT_SYMBOL_FALLBACK: tuple[str, ...] = (
    "BTC/USDT",
    "ETH/USDT",
    "XRP/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "TRX/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "TON/USDT",
    "AVAX/USDT",
    "XAU/USDT",
    "XAG/USDT",
)

_PAIR_ANCHORS: tuple[tuple[str, str], ...] = (
    ("BTC/USDT", "ETH/USDT"),
    ("ETH/USDT", "SOL/USDT"),
    ("XAU/USDT", "XAG/USDT"),
)

_CRYPTO_LEADERS = {"BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"}
_METALS = {"XAU/USDT", "XAG/USDT"}


try:  # pragma: no cover - guarded import for bootstrap contexts
    from lumina_quant.config import BaseConfig

    DEFAULT_BINANCE_TOP10_PLUS_METALS: tuple[str, ...] = tuple(
        canonicalize_symbol_list(list(getattr(BaseConfig, "SYMBOLS", _DEFAULT_SYMBOL_FALLBACK)))
    )
except Exception:  # pragma: no cover - safe fallback during partial imports
    DEFAULT_BINANCE_TOP10_PLUS_METALS = _DEFAULT_SYMBOL_FALLBACK

DEFAULT_TIMEFRAMES: tuple[str, ...] = CANONICAL_STRATEGY_TIMEFRAMES


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
        timeframe = str(self.timeframe)
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "family": self.family,
            "strategy_class": self.strategy_class,
            "strategy": self.strategy_class,
            "strategy_timeframe": timeframe,
            # Legacy alias retained for compatibility.
            "timeframe": timeframe,
            "symbols": list(self.symbols),
            "params": dict(self.params),
            "notes": self.notes,
        }


def _normalize_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(canonicalize_symbol_list(values))


def _candidate_id(*, name: str, timeframe: str, params: dict[str, Any], symbols: tuple[str, ...]) -> str:
    payload = {
        "name": name,
        "timeframe": str(timeframe),
        "params": params,
        "symbols": list(symbols),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _has_perp_support_data() -> bool:
    path = Path("data") / "market_parquet" / "feature_points"
    return path.exists()


def _add_candidate(
    out: list[StrategyCandidate],
    *,
    name: str,
    family: str,
    strategy_class: str,
    timeframe: str,
    symbols: Sequence[str],
    params: dict[str, Any],
    notes: str,
) -> None:
    symbol_tuple = tuple(canonicalize_symbol_list(symbols))
    if not symbol_tuple:
        return
    out.append(
        StrategyCandidate(
            candidate_id=_candidate_id(
                name=name,
                timeframe=timeframe,
                params=params,
                symbols=symbol_tuple,
            ),
            name=name,
            family=family,
            strategy_class=strategy_class,
            timeframe=str(timeframe),
            symbols=symbol_tuple,
            params=dict(params),
            notes=notes,
        )
    )


def _pairs_in_universe(symbols: Sequence[str]) -> list[tuple[str, str]]:
    universe = set(symbols)
    out: list[tuple[str, str]] = []
    for left, right in _PAIR_ANCHORS:
        if left in universe and right in universe:
            out.append((left, right))
    return out


def build_binance_futures_candidates(
    *,
    timeframes: Sequence[str] = DEFAULT_TIMEFRAMES,
    symbols: Sequence[str] = DEFAULT_BINANCE_TOP10_PLUS_METALS,
) -> list[StrategyCandidate]:
    """Build candidate universe for RG_PVTM and diversifier sleeves."""
    normalized_timeframes = tuple(
        normalize_strategy_timeframes(
            list(timeframes),
            required=CANONICAL_STRATEGY_TIMEFRAMES,
            strict_subset=True,
        )
    )
    normalized_symbols = _normalize_unique(symbols)
    if not normalized_timeframes:
        raise ValueError("timeframes must not be empty")
    if len(normalized_symbols) < 2:
        raise ValueError("symbols must include at least two instruments")

    candidates: list[StrategyCandidate] = []
    pairs = _pairs_in_universe(normalized_symbols)

    trend_tfs = [tf for tf in ("30m", "1h", "4h", "1d") if tf in normalized_timeframes]
    mean_rev_tfs = [tf for tf in ("1m", "5m", "15m") if tf in normalized_timeframes]
    pair_tfs = [tf for tf in ("5m", "15m", "1h") if tf in normalized_timeframes]
    carry_tfs = [tf for tf in ("30m", "1h", "4h") if tf in normalized_timeframes]
    micro_tfs = [tf for tf in ("1s",) if tf in normalized_timeframes]

    crypto_symbols = [symbol for symbol in normalized_symbols if symbol not in _METALS]
    laggard_symbols = [symbol for symbol in crypto_symbols if symbol not in _CRYPTO_LEADERS]

    # Primary trend sleeve (RG_PVTM).
    for timeframe in trend_tfs:
        tf_tag = timeframe.replace("/", "-")
        for long_th, short_th, te_min, vr_min in product((0.45, 0.60, 0.75), (0.45, 0.60, 0.75), (0.20, 0.25), (0.80, 0.95)):
            params = {
                "long_threshold": float(long_th),
                "short_threshold": float(short_th),
                "te_min": float(te_min),
                "vr_min": float(vr_min),
                "chop_max": 62.0,
                "risk_target_vol": 0.004,
                "atr_stop_mult": 2.0,
                "trail_atr_mult": 2.8,
                "allow_short": True,
            }
            _add_candidate(
                candidates,
                name=f"composite_trend_{tf_tag}_{long_th:.2f}_{short_th:.2f}_{te_min:.2f}_{vr_min:.2f}",
                family="trend",
                strategy_class="CompositeTrendStrategy",
                timeframe=timeframe,
                symbols=normalized_symbols,
                params=params,
                notes="Primary RG_PVTM trend sleeve across full universe.",
            )

    # Vol-compression VWAP reversion sleeve.
    for timeframe in mean_rev_tfs:
        tf_tag = timeframe.replace("/", "-")
        for entry_z, exit_z, comp_pct in product((1.2, 1.5, 1.9), (0.25, 0.35, 0.50), (0.20, 0.30, 0.40)):
            params = {
                "entry_z": float(entry_z),
                "exit_z": float(exit_z),
                "compression_percentile": float(comp_pct),
                "compression_vol_ratio": 0.85,
                "atr_stop_pct": 0.02,
                "max_hold_bars": 64,
                "allow_short": True,
            }
            _add_candidate(
                candidates,
                name=f"volcomp_vwap_rev_{tf_tag}_{entry_z:.2f}_{exit_z:.2f}_{comp_pct:.2f}",
                family="mean_reversion",
                strategy_class="VolCompressionVWAPReversionStrategy",
                timeframe=timeframe,
                symbols=normalized_symbols,
                params=params,
                notes="Compression-gated VWAP mean reversion diversifier.",
            )

    # Lead/lag spillover sleeve (metals excluded).
    if laggard_symbols:
        for timeframe in mean_rev_tfs:
            tf_tag = timeframe.replace("/", "-")
            for entry_score, max_lag in product((0.25, 0.35, 0.50), (2, 3, 4)):
                params = {
                    "entry_score": float(entry_score),
                    "exit_score": 0.08,
                    "max_lag": int(max_lag),
                    "ridge_alpha": 1.0,
                    "max_hold_bars": 24,
                    "stop_loss_pct": 0.02,
                    "allow_short": True,
                }
                _add_candidate(
                    candidates,
                    name=f"leadlag_spillover_{tf_tag}_{entry_score:.2f}_lag{max_lag}",
                    family="intraday_alpha",
                    strategy_class="LeadLagSpilloverStrategy",
                    timeframe=timeframe,
                    symbols=tuple(sorted(set(_CRYPTO_LEADERS).intersection(normalized_symbols)) + laggard_symbols),
                    params=params,
                    notes="Cross-asset lead-lag predictor (crypto only, metals excluded).",
                )

    # Pair spread sleeve.
    for timeframe in pair_tfs:
        tf_tag = timeframe.replace("/", "-")
        for symbol_x, symbol_y in pairs:
            pair_token = f"{symbol_x.replace('/', '').lower()}_{symbol_y.replace('/', '').lower()}"
            for entry_z, exit_z, stop_z in ((1.8, 0.45, 3.4), (2.2, 0.55, 3.9), (2.6, 0.70, 4.2)):
                params = {
                    "lookback_window": 96,
                    "hedge_window": 192,
                    "entry_z": float(entry_z),
                    "exit_z": float(exit_z),
                    "stop_z": float(stop_z),
                    "max_hold_bars": 240,
                    "min_correlation": 0.1,
                    "stop_loss_pct": 0.03,
                    "symbol_x": symbol_x,
                    "symbol_y": symbol_y,
                }
                _add_candidate(
                    candidates,
                    name=f"pair_spread_{tf_tag}_{pair_token}_{entry_z:.1f}_{exit_z:.2f}",
                    family="market_neutral",
                    strategy_class="PairSpreadZScoreStrategy",
                    timeframe=timeframe,
                    symbols=(symbol_x, symbol_y),
                    params=params,
                    notes="Rolling-beta spread z-score with correlation stability filters.",
                )

    # Optional carry/crowding sleeve.
    if _has_perp_support_data() and carry_tfs:
        for timeframe in carry_tfs:
            tf_tag = timeframe.replace("/", "-")
            for entry, exit_th in ((0.25, 0.08), (0.35, 0.10), (0.45, 0.15)):
                params = {
                    "entry_threshold": float(entry),
                    "exit_threshold": float(exit_th),
                    "mild_funding": 0.0002,
                    "extreme_funding": 0.0012,
                    "stop_loss_pct": 0.02,
                    "max_hold_bars": 72,
                    "allow_short": True,
                }
                _add_candidate(
                    candidates,
                    name=f"perp_crowding_carry_{tf_tag}_{entry:.2f}_{exit_th:.2f}",
                    family="carry",
                    strategy_class="PerpCrowdingCarryStrategy",
                    timeframe=timeframe,
                    symbols=crypto_symbols,
                    params=params,
                    notes="Funding/OI crowding-aware carry sleeve.",
                )

    # Research-only micro sleeve.
    for timeframe in micro_tfs:
        tf_tag = timeframe.replace("/", "-")
        for lookback, range_z, vol_z in ((20, 1.2, 0.8), (30, 1.5, 1.0), (45, 2.0, 1.2)):
            params = {
                "lookback": int(lookback),
                "range_z_threshold": float(range_z),
                "volume_z_threshold": float(vol_z),
                "max_hold_bars": 20,
                "allow_short": True,
            }
            _add_candidate(
                candidates,
                name=f"micro_range_expansion_{tf_tag}_{lookback}_{range_z:.1f}_{vol_z:.1f}",
                family="micro",
                strategy_class="MicroRangeExpansion1sStrategy",
                timeframe=timeframe,
                symbols=crypto_symbols,
                params=params,
                notes="Research-only micro breakout sleeve with strict turnover controls.",
            )

    return candidates


def build_candidate_manifest(
    *,
    timeframes: Sequence[str] = DEFAULT_TIMEFRAMES,
    symbols: Sequence[str] = DEFAULT_BINANCE_TOP10_PLUS_METALS,
) -> dict[str, Any]:
    """Build a JSON-ready manifest with aggregate metadata."""
    normalized_symbols = tuple(canonicalize_symbol_list(symbols))
    normalized_timeframes = tuple(
        normalize_strategy_timeframes(
            list(timeframes),
            required=CANONICAL_STRATEGY_TIMEFRAMES,
            strict_subset=True,
        )
    )
    candidates = build_binance_futures_candidates(
        timeframes=normalized_timeframes,
        symbols=normalized_symbols,
    )

    family_counts: dict[str, int] = {}
    strategy_counts: dict[str, int] = {}
    timeframe_counts: dict[str, int] = {}

    for candidate in candidates:
        family_counts[candidate.family] = family_counts.get(candidate.family, 0) + 1
        strategy_counts[candidate.strategy_class] = strategy_counts.get(candidate.strategy_class, 0) + 1
        timeframe_counts[candidate.timeframe] = timeframe_counts.get(candidate.timeframe, 0) + 1

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "symbol_universe": list(normalized_symbols),
        "timeframes": list(normalized_timeframes),
        "candidate_count": len(candidates),
        "family_counts": family_counts,
        "strategy_counts": strategy_counts,
        "timeframe_counts": timeframe_counts,
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
