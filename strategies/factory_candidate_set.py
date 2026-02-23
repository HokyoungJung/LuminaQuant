"""Strategy-factory candidate set construction helpers.

This module provides a deterministic candidate universe spanning:
- top-10 crypto majors + XAU/XAG
- multiple timeframes from 1s to 1d
- trend, breakout, and mean-reversion families
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

DEFAULT_TOP10_PLUS_METALS: tuple[str, ...] = (
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
    "XAU/USDT:USDT",
    "XAG/USDT:USDT",
)

DEFAULT_TIMEFRAMES: tuple[str, ...] = ("1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d")


@dataclass(frozen=True, slots=True)
class StrategyTemplate:
    name: str
    family: str
    symbol_mode: str  # "single" | "multi"
    param_grid: dict[str, Sequence[object]]
    tags: tuple[str, ...] = ()


def _grid_rows(param_grid: dict[str, Sequence[object]]) -> list[dict[str, object]]:
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    values = [list(param_grid[key]) for key in keys]
    out: list[dict[str, object]] = []
    for row in itertools.product(*values):
        out.append(dict(zip(keys, row, strict=True)))
    return out


def _candidate_id(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _strategy_templates() -> list[StrategyTemplate]:
    return [
        StrategyTemplate(
            name="RegimeBreakoutCandidateStrategy",
            family="trend_breakout",
            symbol_mode="single",
            param_grid={
                "lookback_window": (32, 48),
                "slope_window": (13, 21),
                "range_entry_threshold": (0.66, 0.72),
                "max_volatility_ratio": (1.4, 1.8),
                "stop_loss_pct": (0.02, 0.03),
            },
            tags=("candidate", "worker3", "futures"),
        ),
        StrategyTemplate(
            name="VolatilityCompressionReversionStrategy",
            family="mean_reversion",
            symbol_mode="single",
            param_grid={
                "z_window": (36, 48),
                "compression_threshold": (0.68, 0.75),
                "entry_z": (1.5, 1.8),
                "exit_z": (0.25, 0.40),
                "stop_loss_pct": (0.02, 0.03),
            },
            tags=("candidate", "worker3", "futures"),
        ),
        StrategyTemplate(
            name="RollingBreakoutStrategy",
            family="trend_breakout",
            symbol_mode="single",
            param_grid={
                "lookback_bars": (24, 48),
                "atr_window": (10, 14),
                "atr_stop_multiplier": (1.8, 2.5),
                "breakout_buffer": (0.0, 0.002),
                "stop_loss_pct": (0.02, 0.03),
                "allow_short": (True,),
            },
            tags=("existing", "futures"),
        ),
        StrategyTemplate(
            name="TopCapTimeSeriesMomentumStrategy",
            family="cross_sectional_momentum",
            symbol_mode="multi",
            param_grid={
                "lookback_bars": (12, 24),
                "rebalance_bars": (8, 16),
                "signal_threshold": (0.015, 0.03),
                "max_longs": (4, 6),
                "max_shorts": (4, 6),
                "btc_regime_ma": (48,),
                "stop_loss_pct": (0.06, 0.08),
            },
            tags=("existing", "futures", "topcap"),
        ),
    ]


def build_candidate_set(
    *,
    symbols: Iterable[str] | None = None,
    timeframes: Iterable[str] | None = None,
    max_candidates: int = 0,
) -> list[dict[str, object]]:
    """Build a deterministic strategy candidate set."""
    symbol_list = [str(symbol).strip().upper() for symbol in symbols or DEFAULT_TOP10_PLUS_METALS]
    symbol_list = [symbol.replace("-", "/").replace("_", "/") for symbol in symbol_list if symbol]

    timeframe_list = [str(timeframe).strip().lower() for timeframe in timeframes or DEFAULT_TIMEFRAMES]
    timeframe_list = [token for token in timeframe_list if token]

    templates = _strategy_templates()
    out: list[dict[str, object]] = []

    for timeframe in timeframe_list:
        for template in templates:
            rows = _grid_rows(template.param_grid)
            if template.symbol_mode == "multi":
                for params in rows:
                    payload = {
                        "strategy": template.name,
                        "family": template.family,
                        "timeframe": timeframe,
                        "symbols": list(symbol_list),
                        "params": params,
                        "tags": list(template.tags),
                    }
                    payload["candidate_id"] = _candidate_id(payload)
                    out.append(payload)
                    if int(max_candidates) > 0 and len(out) >= int(max_candidates):
                        return out
            else:
                for symbol in symbol_list:
                    for params in rows:
                        payload = {
                            "strategy": template.name,
                            "family": template.family,
                            "timeframe": timeframe,
                            "symbols": [symbol],
                            "params": params,
                            "tags": list(template.tags),
                        }
                        payload["candidate_id"] = _candidate_id(payload)
                        out.append(payload)
                        if int(max_candidates) > 0 and len(out) >= int(max_candidates):
                            return out

    return out


def summarize_candidate_set(candidates: Sequence[dict[str, object]]) -> dict[str, object]:
    """Return compact summary stats for a candidate list."""
    summary: dict[str, object] = {
        "count": len(candidates),
        "families": {},
        "timeframes": {},
        "strategies": {},
    }

    family_counts: dict[str, int] = {}
    timeframe_counts: dict[str, int] = {}
    strategy_counts: dict[str, int] = {}

    for row in candidates:
        family = str(row.get("family", ""))
        timeframe = str(row.get("timeframe", ""))
        strategy = str(row.get("strategy", ""))
        family_counts[family] = family_counts.get(family, 0) + 1
        timeframe_counts[timeframe] = timeframe_counts.get(timeframe, 0) + 1
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

    summary["families"] = dict(sorted(family_counts.items(), key=lambda item: item[0]))
    summary["timeframes"] = dict(sorted(timeframe_counts.items(), key=lambda item: item[0]))
    summary["strategies"] = dict(sorted(strategy_counts.items(), key=lambda item: item[0]))
    return summary


__all__ = [
    "DEFAULT_TIMEFRAMES",
    "DEFAULT_TOP10_PLUS_METALS",
    "build_candidate_set",
    "summarize_candidate_set",
]
