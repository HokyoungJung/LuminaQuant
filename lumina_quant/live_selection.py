"""Helpers to load and apply live-selection artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def resolve_selection_file(selection_file: str = "") -> Path | None:
    token = str(selection_file or "").strip()
    if token:
        path = Path(token).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"Live selection file not found: {path}")
        return path

    root = Path("best_optimized_parameters/live")
    if not root.exists():
        return None
    files = sorted(root.glob("live_selection_*.json"), key=lambda item: item.stat().st_mtime)
    if not files:
        return None
    return files[-1]


def load_selection_payload(path: Path) -> dict:
    with path.open(encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid selection payload shape: {path}")
    return payload


def infer_strategy_class_name(candidate_name: str) -> str | None:
    token = str(candidate_name or "").strip().lower()
    if not token:
        return None
    if token.startswith("bitcoin_buy_hold"):
        return "BitcoinBuyHoldStrategy"
    if token.startswith("lag_convergence"):
        return "LagConvergenceStrategy"
    if token.startswith("mean_reversion_std"):
        return "MeanReversionStdStrategy"
    if token.startswith("rolling_breakout"):
        return "RollingBreakoutStrategy"
    if token.startswith("topcap_tsmom"):
        return "TopCapTimeSeriesMomentumStrategy"
    if token.startswith("pair_"):
        return "PairTradingZScoreStrategy"
    if token.startswith("vwap_reversion"):
        return "VwapReversionStrategy"
    if token.startswith("rsi"):
        return "RsiStrategy"
    if token.startswith("moving"):
        return "MovingAverageCrossStrategy"
    return None


def extract_selection_config(payload: dict) -> dict:
    selected = payload.get("selected_candidate") or {}
    selected = selected if isinstance(selected, dict) else {}

    symbols = selected.get("symbols")
    if not isinstance(symbols, list):
        symbols = []
    symbols = [str(item).strip().upper() for item in symbols if str(item).strip()]

    params = selected.get("params")
    if not isinstance(params, dict):
        params = {}

    split = payload.get("split") or {}
    split = split if isinstance(split, dict) else {}
    strategy_timeframe = str(
        split.get("strategy_timeframe") or payload.get("best_timeframe") or ""
    ).strip()
    if not strategy_timeframe:
        strategy_timeframe = None

    return {
        "candidate_name": str(selected.get("name") or "").strip(),
        "symbols": symbols,
        "params": params,
        "strategy_timeframe": strategy_timeframe,
        "base_timeframe": str(payload.get("base_timeframe") or "").strip() or None,
        "mode": str(payload.get("mode") or "").strip() or None,
    }
