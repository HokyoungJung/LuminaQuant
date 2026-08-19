from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/research/materialize_named_quant_universe.py"
SPEC = importlib.util.spec_from_file_location("materialize_named_quant_universe", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _market_snapshot(timestamp: str, symbols: list[str]) -> dict:
    return {
        "timestamp": timestamp,
        "assets": [{"symbol": symbol, "rank": rank} for rank, symbol in enumerate(symbols, 1)],
    }


def _exchange_snapshot(timestamp: str, *, tradfi: tuple[str, ...] = ("XAU", "SPX")) -> dict:
    symbols = []
    for base in ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "TON", "BNB", "TRX", "USDC"]:
        symbols.append(
            {
                "symbol": f"{base}USDT",
                "baseAsset": base,
                "quoteAsset": "USDT",
                "contractType": "PERPETUAL",
                "status": "TRADING",
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                    {"filterType": "NOT_COPIED", "x": "y"},
                ],
            }
        )
    for base in tradfi:
        symbols.append(
            {
                "symbol": f"{base}USDT",
                "baseAsset": base,
                "quoteAsset": "USDT",
                "contractType": "TRADIFI_PERPETUAL",
                "status": "TRADING",
                "filters": [{"filterType": "MIN_NOTIONAL", "notional": "5"}],
            }
        )
    return {"timestamp": timestamp, "symbols": symbols}


def _suite() -> dict:
    return {
        "candidates": [
            {"candidate_id": "crypto", "metadata": {"universe_binding": "crypto_top10"}, "symbols": ["OLD"]},
            {"candidate_id": "tradfi", "metadata": {"universe_binding": "tradfi_all"}, "symbols": ["OLD"]},
            {
                "candidate_id": "both",
                "metadata": {"universe_binding": "crypto_top10_plus_tradfi"},
                "symbols": ["OLD"],
            },
            {"candidate_id": "fixed", "metadata": {}, "symbols": ["BTC/USDT", "ETH/USDT"]},
        ]
    }


def test_materializes_ranked_intersection_tradfi_filters_and_bindings(tmp_path: Path) -> None:
    ranking = ["USDC", "BTC", "MISSING", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "TON", "BNB", "TRX"]
    market = _market_snapshot("2026-01-01T00:00:00Z", ranking)
    exchange = _exchange_snapshot("2026-01-01T00:00:00Z")
    output = MODULE.materialize(
        _suite(),
        market,
        exchange,
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
        market_cap_source=tmp_path / "caps.json",
        exchange_info_source=tmp_path / "exchange.json",
    )

    candidates = {row["candidate_id"]: row for row in output["candidates"]}
    crypto = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "TON/USDT", "BNB/USDT", "TRX/USDT"]
    assert candidates["crypto"]["symbols"] == crypto
    assert candidates["tradfi"]["symbols"] == ["SPX/USDT", "XAU/USDT"]
    assert candidates["both"]["symbols"] == [*crypto, "SPX/USDT", "XAU/USDT"]
    assert candidates["fixed"]["symbols"] == ["BTC/USDT", "ETH/USDT"]
    receipt = output["universe_materialization_receipt"]
    assert receipt["counts"]["eligible_ranked_crypto"] == 10
    assert receipt["binance_filters"]["BTC/USDT"] == [
        {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
        {"filterType": "LOT_SIZE", "stepSize": "0.001"},
    ]
    assert receipt["binance_filters"]["XAU/USDT"] == [
        {"filterType": "MIN_NOTIONAL", "notional": "5"}
    ]


def test_cli_uses_latest_non_future_snapshot_and_jsonl(tmp_path: Path) -> None:
    suite = tmp_path / "suite.json"
    caps = tmp_path / "caps.jsonl"
    exchange = tmp_path / "exchange.json"
    output = tmp_path / "out.json"
    suite.write_text(json.dumps(_suite()))
    caps.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _market_snapshot("2026-01-01T00:00:00Z", ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "TON", "BNB", "TRX"]),
                _market_snapshot("2026-02-01T00:00:00Z", ["TRX", "BNB", "TON", "AVAX", "ADA", "DOGE", "XRP", "SOL", "ETH", "BTC"]),
            ]
        )
    )
    exchange.write_text(
        json.dumps(
            {
                "snapshots": [
                    _exchange_snapshot("2026-01-01T00:00:00Z", tradfi=("XAU",)),
                    _exchange_snapshot("2026-02-01T00:00:00Z", tradfi=("SPX",)),
                ]
            }
        )
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--suite",
            str(suite),
            "--market-caps",
            str(caps),
            "--exchange-info",
            str(exchange),
            "--as-of",
            "2026-01-15T00:00:00Z",
            "--output",
            str(output),
        ],
        check=True,
    )
    result = json.loads(output.read_text())
    assert result["candidates"][0]["symbols"][0] == "BTC/USDT"
    assert result["candidates"][1]["symbols"] == ["XAU/USDT"]
    assert result["universe_materialization_receipt"]["snapshot_timestamps"] == {
        "exchange_info": "2026-01-01T00:00:00Z",
        "market_caps": "2026-01-01T00:00:00Z",
    }


@pytest.mark.parametrize("problem", ["future", "duplicate", "missing"])
def test_fails_closed_on_unusable_snapshots(tmp_path: Path, problem: str) -> None:
    valid_ranking = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "TON", "BNB", "TRX"]
    if problem == "future":
        snapshots = [_market_snapshot("2026-02-01T00:00:00Z", valid_ranking)]
    elif problem == "duplicate":
        snapshots = [
            _market_snapshot("2026-01-01T00:00:00Z", valid_ranking),
            _market_snapshot("2026-01-01T00:00:00Z", valid_ranking),
        ]
    else:
        snapshots = [_market_snapshot("2026-01-01T00:00:00Z", valid_ranking[:-1])]
    path = tmp_path / "caps.json"
    path.write_text(json.dumps(snapshots))
    as_of = datetime(2026, 1, 15, tzinfo=UTC)
    if problem == "duplicate":
        with pytest.raises(ValueError):
            MODULE._load_snapshots(path, label="market-cap")
        return
    loaded = MODULE._load_snapshots(path, label="market-cap")
    if problem == "future":
        with pytest.raises(ValueError):
            MODULE._latest(loaded, as_of, label="market-cap")
    else:
        with pytest.raises(ValueError, match="only 9"):
            MODULE.materialize(
                _suite(),
                MODULE._latest(loaded, as_of, label="market-cap"),
                _exchange_snapshot("2026-01-01T00:00:00Z"),
                as_of=as_of,
                market_cap_source=path,
                exchange_info_source=tmp_path / "exchange.json",
            )
