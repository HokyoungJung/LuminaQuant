#!/usr/bin/env python3
"""Materialize point-in-time named-quant universes from local snapshots."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_TIMESTAMP_KEYS = ("timestamp", "snapshot_timestamp", "as_of", "time", "serverTime")
_FILTER_TYPES = {
    "PRICE_FILTER",
    "LOT_SIZE",
    "MARKET_LOT_SIZE",
    "MIN_NOTIONAL",
    "PERCENT_PRICE",
    "PERCENT_PRICE_BY_SIDE",
}
_STABLE_BASES = {
    "BUSD",
    "DAI",
    "FDUSD",
    "PYUSD",
    "TUSD",
    "USDC",
    "USDD",
    "USDE",
    "USDP",
    "USDT",
}


def _parse_timestamp(value: Any, *, label: str) -> datetime:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = float(value) / 1000 if abs(float(value)) >= 10_000_000_000 else float(value)
        parsed = datetime.fromtimestamp(seconds, tz=UTC)
    elif isinstance(value, str):
        raw = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(f"invalid {label} timestamp: {value!r}") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        parsed = parsed.astimezone(UTC)
    else:
        raise ValueError(f"missing or invalid {label} timestamp")
    return parsed


def _timestamp(snapshot: dict[str, Any], *, label: str) -> datetime:
    for key in _TIMESTAMP_KEYS:
        if key in snapshot:
            return _parse_timestamp(snapshot[key], label=label)
    raise ValueError(f"missing {label} snapshot timestamp")


def _load_snapshots(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        if path.suffix.lower() == ".jsonl":
            payload: Any = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} snapshot file: {path}") from exc
    if isinstance(payload, dict) and "snapshots" in payload:
        payload = payload["snapshots"]
    if not isinstance(payload, list) or not payload or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"{label} snapshots must be a non-empty list")
    timestamps = [_timestamp(row, label=label) for row in payload]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError(f"duplicate {label} snapshot timestamp")
    return payload


def _latest(snapshots: list[dict[str, Any]], as_of: datetime, *, label: str) -> dict[str, Any]:
    valid = [row for row in snapshots if _timestamp(row, label=label) <= as_of]
    if not valid:
        raise ValueError(f"no {label} snapshot at or before --as-of")
    return max(valid, key=lambda row: _timestamp(row, label=label))


def _rows(snapshot: dict[str, Any], keys: tuple[str, ...], *, label: str) -> list[dict[str, Any]]:
    for key in keys:
        value = snapshot.get(key)
        if isinstance(value, list) and all(isinstance(row, dict) for row in value):
            return value
    raise ValueError(f"missing or invalid {label} rows")


def _market_cap_ranking(snapshot: dict[str, Any]) -> list[str]:
    rows = _rows(snapshot, ("assets", "coins", "data", "rankings"), label="market-cap")
    ranked: list[tuple[int, str]] = []
    for row in rows:
        symbol = str(row.get("symbol") or "").upper().strip()
        rank = row.get("rank", row.get("market_cap_rank"))
        if not symbol or isinstance(rank, bool):
            raise ValueError("invalid market-cap row")
        try:
            rank_int = int(rank)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid market-cap rank") from exc
        if rank_int <= 0:
            raise ValueError("invalid market-cap rank")
        ranked.append((rank_int, symbol))
    if len({rank for rank, _ in ranked}) != len(ranked) or len({symbol for _, symbol in ranked}) != len(ranked):
        raise ValueError("duplicate market-cap rank or symbol")
    return [symbol for _, symbol in sorted(ranked)]


def _exchange_symbols(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _rows(snapshot, ("symbols", "data"), label="exchangeInfo")
    seen: set[str] = set()
    for row in rows:
        symbol = str(row.get("symbol") or "").upper().strip()
        if not symbol or symbol in seen:
            raise ValueError("missing or duplicate exchangeInfo symbol")
        seen.add(symbol)
    return rows


def _eligible(row: dict[str, Any], contract_type: str) -> bool:
    return (
        str(row.get("contractType") or "").upper() == contract_type
        and str(row.get("status") or "").upper() == "TRADING"
        and str(row.get("quoteAsset") or "").upper() == "USDT"
        and bool(str(row.get("baseAsset") or "").strip())
    )


def _slash_symbol(row: dict[str, Any]) -> str:
    return f"{str(row['baseAsset']).upper()}/{str(row['quoteAsset']).upper()}"


def materialize(
    suite: dict[str, Any],
    market_cap_snapshot: dict[str, Any],
    exchange_snapshot: dict[str, Any],
    *,
    as_of: datetime,
    market_cap_source: Path,
    exchange_info_source: Path,
    crypto_top_n: int = 10,
) -> dict[str, Any]:
    """Return a suite with explicitly bound universes replaced point-in-time."""
    if crypto_top_n <= 0:
        raise ValueError("--crypto-top-n must be positive")
    ranking = _market_cap_ranking(market_cap_snapshot)
    exchange_rows = _exchange_symbols(exchange_snapshot)
    perpetual_rows = [
        row
        for row in exchange_rows
        if _eligible(row, "PERPETUAL")
        and str(row["baseAsset"]).upper() not in _STABLE_BASES
    ]
    perpetual = {str(row["baseAsset"]).upper(): row for row in perpetual_rows}
    if len(perpetual) != len(perpetual_rows):
        raise ValueError("duplicate eligible crypto base asset")
    ranked_eligible = [base for base in ranking if base in perpetual and base not in _STABLE_BASES]
    if len(ranked_eligible) < crypto_top_n:
        raise ValueError(
            f"only {len(ranked_eligible)} eligible ranked crypto names; need {crypto_top_n}"
        )
    crypto_rows = [perpetual[base] for base in ranked_eligible[:crypto_top_n]]
    tradfi_rows = sorted(
        (row for row in exchange_rows if _eligible(row, "TRADIFI_PERPETUAL")),
        key=lambda row: str(row["symbol"]),
    )
    crypto = [_slash_symbol(row) for row in crypto_rows]
    tradfi = [_slash_symbol(row) for row in tradfi_rows]

    result = deepcopy(suite)
    bindings = {
        "crypto_top10": crypto,
        "tradfi_all": tradfi,
        "crypto_top10_plus_tradfi": crypto + tradfi,
    }
    patched: list[str] = []
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("suite candidates must be a list")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("invalid suite candidate")
        metadata = candidate.get("metadata")
        binding = metadata.get("universe_binding") if isinstance(metadata, dict) else None
        if binding is None:
            continue
        if binding not in bindings:
            raise ValueError(f"unknown universe binding: {binding!r}")
        candidate["symbols"] = list(bindings[binding])
        patched.append(str(candidate.get("candidate_id") or candidate.get("name") or "<unnamed>"))

    selected_rows = crypto_rows + tradfi_rows
    result["universe_materialization_receipt"] = {
        "as_of": as_of.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "sources": {
            "market_caps": str(market_cap_source.resolve()),
            "exchange_info": str(exchange_info_source.resolve()),
        },
        "snapshot_timestamps": {
            "market_caps": _timestamp(market_cap_snapshot, label="market-cap")
            .isoformat()
            .replace("+00:00", "Z"),
            "exchange_info": _timestamp(exchange_snapshot, label="exchangeInfo")
            .isoformat()
            .replace("+00:00", "Z"),
        },
        "counts": {
            "market_cap_ranked": len(ranking),
            "eligible_crypto_contracts": len(perpetual),
            "eligible_ranked_crypto": len(ranked_eligible),
            "eligible_tradfi_contracts": len(tradfi_rows),
            "selected_crypto": len(crypto),
            "selected_tradfi": len(tradfi),
            "patched_candidates": len(patched),
        },
        "selected_symbols": {"crypto_top10": crypto, "tradfi_all": tradfi},
        "patched_candidates": patched,
        "binance_filters": {
            _slash_symbol(row): [
                deepcopy(item)
                for item in row.get("filters", [])
                if isinstance(item, dict) and item.get("filterType") in _FILTER_TYPES
            ]
            for row in selected_rows
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--market-caps", type=Path, required=True)
    parser.add_argument("--exchange-info", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crypto-top-n", type=int, default=10)
    args = parser.parse_args()
    try:
        suite = json.loads(args.suite.read_text())
        if not isinstance(suite, dict):
            raise ValueError("suite must be a JSON object")
        as_of = _parse_timestamp(args.as_of, label="--as-of")
        market_caps = _load_snapshots(args.market_caps, label="market-cap")
        exchange_info = _load_snapshots(args.exchange_info, label="exchangeInfo")
        for snapshot in market_caps:
            _market_cap_ranking(snapshot)
        for snapshot in exchange_info:
            _exchange_symbols(snapshot)
        output = materialize(
            suite,
            _latest(market_caps, as_of, label="market-cap"),
            _latest(exchange_info, as_of, label="exchangeInfo"),
            as_of=as_of,
            market_cap_source=args.market_caps,
            exchange_info_source=args.exchange_info,
            crypto_top_n=args.crypto_top_n,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
