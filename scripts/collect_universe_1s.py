"""Collect 1s data symbol-by-symbol with resume-friendly reporting."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import ccxt
from lumina_quant.config import BaseConfig, LiveConfig
from lumina_quant.data_collector import auto_collect_market_data


def _parse_dt(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.isdigit():
        numeric = int(text)
        if abs(numeric) < 100_000_000_000:
            numeric *= 1000
        return datetime.fromtimestamp(numeric / 1000.0, tz=UTC)
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _build_top20_universe(
    *,
    desired_count: int = 10,
    candidate_count: int = 200,
    market_type: str = "future",
) -> list[str]:
    stable = {
        "USDT",
        "USDC",
        "USDS",
        "USDE",
        "DAI",
        "FDUSD",
        "USD1",
        "TUSD",
        "BUSD",
        "PYUSD",
        "FRAX",
        "USD0",
        "USDD",
    }
    blacklist = {"WBT", "LEO", "WBTC", "STETH", "WEETH", "WETH", "USDT0", "FIGR_HELOC", "CC"}

    url = (
        "https://api.coingecko.com/api/v3/coins/markets"
        f"?vs_currency=usd&order=market_cap_desc&per_page={int(candidate_count)}&page=1&sparkline=false"
    )
    payload = json.loads(urllib.request.urlopen(url, timeout=30).read().decode())

    exchange = ccxt.binance({"enableRateLimit": True})
    options = dict(getattr(exchange, "options", {}) or {})
    options["defaultType"] = str(market_type).strip().lower()
    exchange.options = options
    market_symbols = set(exchange.load_markets().keys())

    out = []
    seen = set()
    for coin in payload:
        symbol = str(coin.get("symbol", "")).upper().strip()
        if not symbol or symbol in stable or symbol in blacklist or symbol in seen:
            continue
        pair = f"{symbol}/USDT"
        alt_pair = f"1000{symbol}/USDT"
        if pair in market_symbols:
            out.append(pair)
            seen.add(symbol)
        elif alt_pair in market_symbols:
            out.append(alt_pair)
            seen.add(symbol)
        if len(out) >= int(desired_count):
            break

    close_fn = getattr(exchange, "close", None)
    if callable(close_fn):
        close_fn()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect universe 1s data with progress reports.")
    parser.add_argument(
        "--db-path",
        default=BaseConfig.MARKET_DATA_SQLITE_PATH,
    )
    parser.add_argument("--exchange-id", default=BaseConfig.MARKET_DATA_EXCHANGE)
    parser.add_argument(
        "--market-type", choices=["spot", "future"], default=str(LiveConfig.MARKET_TYPE)
    )
    parser.add_argument("--timeframe", default="1s")
    parser.add_argument("--since", default="")
    parser.add_argument("--until", default="")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-batches", type=int, default=100000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--include-xau-xag", action="store_true")
    parser.add_argument("--symbols", nargs="+", default=[])
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    symbols = list(args.symbols)
    if not symbols:
        symbols = _build_top20_universe(
            desired_count=10, candidate_count=200, market_type=str(args.market_type)
        )
    if args.include_xau_xag:
        for item in ["XAU/USDT", "XAG/USDT"]:
            if item not in symbols:
                symbols.append(item)

    since_dt = _parse_dt(args.since)
    until_dt = _parse_dt(args.until)

    started = datetime.now(UTC)
    report = {
        "started_at": started.isoformat(),
        "db_path": str(args.db_path),
        "exchange_id": str(args.exchange_id),
        "market_type": str(args.market_type),
        "timeframe": str(args.timeframe),
        "since": since_dt.isoformat() if since_dt else None,
        "until": until_dt.isoformat() if until_dt else None,
        "symbols": symbols,
        "results": [],
    }

    for index, symbol in enumerate(symbols, start=1):
        t0 = time.time()
        print(f"[{index}/{len(symbols)}] collect {symbol}")
        try:
            rows = auto_collect_market_data(
                symbol_list=[symbol],
                timeframe=str(args.timeframe),
                db_path=str(args.db_path),
                exchange_id=str(args.exchange_id),
                market_type=str(args.market_type),
                since_dt=since_dt,
                until_dt=until_dt,
                api_key=str(LiveConfig.BINANCE_API_KEY or ""),
                secret_key=str(LiveConfig.BINANCE_SECRET_KEY or ""),
                testnet=bool(LiveConfig.IS_TESTNET),
                limit=max(1, int(args.limit)),
                max_batches=max(1, int(args.max_batches)),
                retries=max(0, int(args.retries)),
            )
            elapsed = time.time() - t0
            entry = rows[0] if rows else {"symbol": symbol}
            entry["elapsed_sec"] = round(float(elapsed), 3)
            entry["status"] = "ok"
            report["results"].append(entry)
            print(
                f"  ok fetched={entry.get('fetched_rows', 0)} upserted={entry.get('upserted_rows', 0)} sec={entry['elapsed_sec']}"
            )
        except Exception as exc:
            elapsed = time.time() - t0
            report["results"].append(
                {
                    "symbol": symbol,
                    "status": "error",
                    "error": str(exc),
                    "elapsed_sec": round(float(elapsed), 3),
                }
            )
            print(f"  error: {exc}")

    report["finished_at"] = datetime.now(UTC).isoformat()
    out_path = (
        Path(args.out)
        if str(args.out).strip()
        else Path("reports")
        / ("collect_universe_1s_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + ".json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
