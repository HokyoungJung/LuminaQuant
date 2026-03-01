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
from typing import Any

from lumina_quant.symbols import (
    CANONICAL_STRATEGY_TIMEFRAMES,
    canonicalize_symbol_list,
    normalize_strategy_timeframes,
)

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
    "TON/USDT",
    "XAU/USDT",
    "XAG/USDT",
)

DEFAULT_TIMEFRAMES: tuple[str, ...] = CANONICAL_STRATEGY_TIMEFRAMES


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


def _family_for_strategy(name: str) -> str:
    token = str(name).strip().lower()
    if "pair" in token or token.startswith("lagconvergence"):
        return "market_neutral"
    if "breakout" in token:
        return "trend_breakout"
    if (
        "reversion" in token
        or "mean" in token
        or token.startswith("rsi")
        or token.startswith("rareevent")
    ):
        return "mean_reversion"
    if "momentum" in token or "movingaverage" in token or token.startswith("topcap"):
        return "trend"
    if "buyhold" in token:
        return "benchmark"
    return "other"


def _symbol_mode_for_strategy(name: str) -> str:
    token = str(name).strip()
    if token in {"TopCapTimeSeriesMomentumStrategy"}:
        return "multi"
    if token in {"PairTradingZScoreStrategy", "LagConvergenceStrategy"}:
        return "pair"
    return "single"


def _build_symbol_pairs(symbols: Sequence[str]) -> list[tuple[str, str]]:
    clean = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    clean = [symbol.replace("-", "/").replace("_", "/") for symbol in clean]
    out: list[tuple[str, str]] = []

    preferred = [
        ("BTC/USDT", "ETH/USDT"),
        ("ETH/USDT", "SOL/USDT"),
        ("BNB/USDT", "SOL/USDT"),
        ("ADA/USDT", "XRP/USDT"),
        ("DOGE/USDT", "TRX/USDT"),
        ("AVAX/USDT", "TON/USDT"),
        ("XAU/USDT", "XAG/USDT"),
    ]
    for left, right in preferred:
        if left in clean and right in clean:
            out.append((left, right))

    if len(clean) >= 2:
        for idx in range(len(clean) - 1):
            pair = (clean[idx], clean[idx + 1])
            if pair not in out:
                out.append(pair)

    if len(clean) >= 2 and not out:
        out.append((clean[0], clean[1]))
    return out


def _normalize_param_values(value: Any) -> tuple[object, ...]:
    if isinstance(value, (list, tuple)):
        cleaned = tuple(item for item in value if item is not None)
        if cleaned:
            return cleaned
        return (0,)
    return (value,)


def _default_grid_for_strategy(name: str) -> dict[str, Sequence[object]]:
    from lumina_quant.strategies import registry as strategy_registry

    grid = strategy_registry.get_default_grid_config(name)
    params = grid.get("params") if isinstance(grid, dict) else None
    if isinstance(params, dict) and params:
        return {str(key): _normalize_param_values(value) for key, value in params.items()}

    defaults = strategy_registry.get_default_strategy_params(name)
    if not isinstance(defaults, dict):
        return {}
    return {str(key): _normalize_param_values(value) for key, value in defaults.items()}


def _limit_grid_rows(
    rows: list[dict[str, object]],
    *,
    max_rows: int,
) -> list[dict[str, object]]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    if max_rows == 1:
        return [rows[0]]

    step = (len(rows) - 1) / float(max_rows - 1)
    selected: list[dict[str, object]] = []
    for idx in range(max_rows):
        pick = rows[round(idx * step)]
        selected.append(pick)
    # preserve order while removing duplicates from rounded picks
    deduped: list[dict[str, object]] = []
    seen = set()
    for row in selected:
        marker = json.dumps(row, sort_keys=True, separators=(",", ":"))
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(row)
    return deduped


def _strategy_templates() -> list[StrategyTemplate]:
    from lumina_quant.strategies import registry as strategy_registry

    templates: list[StrategyTemplate] = []
    for name in strategy_registry.get_strategy_names():
        templates.append(
            StrategyTemplate(
                name=name,
                family=_family_for_strategy(name),
                symbol_mode=_symbol_mode_for_strategy(name),
                param_grid=_default_grid_for_strategy(name),
                tags=("registry", "futures"),
            )
        )
    return templates


def build_candidate_set(
    *,
    symbols: Iterable[str] | None = None,
    timeframes: Iterable[str] | None = None,
    max_candidates: int = 0,
    max_param_rows_per_strategy: int = 24,
) -> list[dict[str, object]]:
    """Build a deterministic strategy candidate set."""
    symbol_list = canonicalize_symbol_list(symbols or DEFAULT_TOP10_PLUS_METALS)
    timeframe_list = normalize_strategy_timeframes(
        timeframes or DEFAULT_TIMEFRAMES,
        required=CANONICAL_STRATEGY_TIMEFRAMES,
        strict_subset=True,
    )

    templates = _strategy_templates()
    out: list[dict[str, object]] = []

    for timeframe in timeframe_list:
        for template in templates:
            rows = _limit_grid_rows(
                _grid_rows(template.param_grid),
                max_rows=max(1, int(max_param_rows_per_strategy)),
            )
            if template.symbol_mode == "multi":
                for params in rows:
                    params = dict(params)
                    if "btc_symbol" in params:
                        params["btc_symbol"] = (
                            "BTC/USDT" if "BTC/USDT" in symbol_list else symbol_list[0]
                        )
                    if "symbol" in params and symbol_list:
                        params["symbol"] = symbol_list[0]
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
            elif template.symbol_mode == "pair":
                symbol_pairs = _build_symbol_pairs(symbol_list)
                for left, right in symbol_pairs:
                    for params in rows:
                        params = dict(params)
                        if "symbol_x" in params:
                            params["symbol_x"] = left
                        if "symbol_y" in params:
                            params["symbol_y"] = right
                        payload = {
                            "strategy": template.name,
                            "family": template.family,
                            "timeframe": timeframe,
                            "symbols": [left, right],
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
                        params = dict(params)
                        if "symbol" in params:
                            params["symbol"] = symbol
                        if "symbol_x" in params:
                            params["symbol_x"] = symbol
                        if "symbol_y" in params:
                            params["symbol_y"] = symbol
                        if "btc_symbol" in params:
                            params["btc_symbol"] = (
                                "BTC/USDT" if "BTC/USDT" in symbol_list else symbol_list[0]
                            )
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
