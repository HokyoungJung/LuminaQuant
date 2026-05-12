#!/usr/bin/env python3
"""Smoke replay CryptoFxAlphaZooStateStrategy from OHLCV CSV/parquet rows."""

from __future__ import annotations

import argparse
import json
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lumina_quant.core.events import MarketBatchEvent, MarketEvent
from lumina_quant.strategies.crypto_fx_alpha_zoo_state import CryptoFxAlphaZooStateStrategy


@dataclass(slots=True)
class _Bars:
    symbol_list: list[str]


def _load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def replay_frame(frame: pd.DataFrame, *, require_calibrated_edge: bool = False) -> dict[str, Any]:
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    symbols = sorted(str(item) for item in data["symbol"].dropna().unique())
    events: queue.Queue = queue.Queue()
    strategy = CryptoFxAlphaZooStateStrategy(
        _Bars(symbols),
        events,
        require_calibrated_edge=require_calibrated_edge,
        calibrated_edges={"default:LONG": 1.0, "default:SHORT": 1.0},
    )
    for ts, group in data.sort_values(["timestamp", "symbol"]).groupby("timestamp", sort=True):
        bars = tuple(
            MarketEvent(
                time=ts,
                symbol=str(row.symbol),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )
            for row in group.itertuples(index=False)
        )
        strategy.calculate_signals(MarketBatchEvent(time=ts, bars=bars))
    signal_rows = []
    while not events.empty():
        signal = events.get()
        signal_rows.append(
            {
                "datetime": str(signal.datetime),
                "symbol": signal.symbol,
                "signal_type": signal.signal_type,
                "strength": signal.strength,
                "metadata": signal.metadata or {},
            }
        )
    return {
        "artifact_kind": "crypto_fx_alpha_zoo_state_replay_smoke",
        "row_count": len(data),
        "signal_count": len(signal_rows),
        "signals": signal_rows,
        "strategy_validity": CryptoFxAlphaZooStateStrategy.strategy_validity,
        "uses_locked_oos_for_selection": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="var/reports/crypto_fx_alpha_zoo_v0/state_replay_latest.json")
    parser.add_argument("--require-calibrated-edge", action="store_true")
    args = parser.parse_args()
    payload = replay_frame(_load(Path(args.input).expanduser().resolve()), require_calibrated_edge=bool(args.require_calibrated_edge))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
