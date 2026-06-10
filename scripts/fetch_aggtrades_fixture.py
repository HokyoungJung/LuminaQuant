"""One-time aggTrades fixture fetch for lumina_quant Phase 0b.

Fetches a fixed historical window of BTCUSDT aggTrades from the public
Binance Futures REST endpoint (no API key required) and writes a frozen
parquet fixture to tests/fixtures/aggtrades/.

Usage:
    uv run python scripts/fetch_aggtrades_fixture.py

Run ONCE. Commit the resulting parquet file. Never re-run unless creating
a NEW baseline (which requires re-approval per baseline/README.md).

The fixture path and SHA-256 are printed at completion for entry into
baseline/golden/PROVENANCE.json.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

import polars as pl

# Fixed historical 1-hour window — never changes after first capture.
SYMBOL = "BTCUSDT"
START_MS = 1781046000000
END_MS = 1781049600000
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "aggtrades"
FIXTURE_NAME = f"{SYMBOL}_1h_{START_MS}_{END_MS}.parquet"
API_BASE = "https://fapi.binance.com/fapi/v1/aggTrades"
PAGE_LIMIT = 1000
REQUEST_DELAY = 0.12  # stay well under rate limit


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_window(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    """Paginate aggTrades by fromId through the fixed time window."""
    records: list[dict] = []

    # Bootstrap: get the first fromId at start_ms.
    url = f"{API_BASE}?symbol={symbol}&startTime={start_ms}&limit={PAGE_LIMIT}"
    with urllib.request.urlopen(url, timeout=20) as r:
        batch = json.loads(r.read())
    if not batch:
        raise RuntimeError(f"No aggTrades returned for start_ms={start_ms}")

    records.extend(batch)
    last_id = batch[-1]["a"]
    last_t = batch[-1]["T"]
    print(f"  boot: {len(batch)} records, T_last={last_t}, id_last={last_id}")

    page = 1
    while last_t < end_ms:
        time.sleep(REQUEST_DELAY)
        next_from = last_id + 1
        url = f"{API_BASE}?symbol={symbol}&fromId={next_from}&limit={PAGE_LIMIT}"
        with urllib.request.urlopen(url, timeout=20) as r:
            batch = json.loads(r.read())
        if not batch:
            break
        # Trim records past end_ms.
        trimmed = [r for r in batch if r["T"] <= end_ms]
        records.extend(trimmed)
        last_id = batch[-1]["a"]
        last_t = batch[-1]["T"]
        page += 1
        if page % 10 == 0:
            print(f"  page {page}: {len(records)} total, T_last={last_t}")
        if len(trimmed) < len(batch):
            break  # Last batch overshot the window — done.

    return records


def build_dataframe(records: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "agg_trade_id": [int(r["a"]) for r in records],
            "price": [float(r["p"]) for r in records],
            "quantity": [float(r["q"]) for r in records],
            "timestamp_ms": [int(r["T"]) for r in records],
            "is_buyer_maker": [bool(r["m"]) for r in records],
            "first_trade_id": [int(r["f"]) for r in records],
            "last_trade_id": [int(r["l"]) for r in records],
        }
    )


def main() -> None:
    fixture_path = FIXTURE_DIR / FIXTURE_NAME
    if fixture_path.exists():
        print(f"Fixture already exists: {fixture_path}")
        print(f"SHA-256: {_sha256(fixture_path)}")
        print("Re-run only if creating a NEW baseline (requires re-approval).")
        return

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching BTCUSDT aggTrades [{START_MS}, {END_MS}] from public REST...")

    records = fetch_window(SYMBOL, START_MS, END_MS)
    print(f"Fetched {len(records)} records total.")

    df = build_dataframe(records)
    df.write_parquet(fixture_path)

    sha = _sha256(fixture_path)
    print(f"\nFixture written: {fixture_path}")
    print(f"Records: {len(df)}")
    print(f"T_first: {df['timestamp_ms'][0]}, T_last: {df['timestamp_ms'][-1]}")
    print(f"SHA-256: {sha}")
    print("\nAdd to PROVENANCE.json aggtrades_fixture:")
    print(json.dumps({
        "status": "captured",
        "path": f"tests/fixtures/aggtrades/{FIXTURE_NAME}",
        "symbol": SYMBOL,
        "start_ms": START_MS,
        "end_ms": END_MS,
        "record_count": len(df),
        "sha256": sha,
    }, indent=2))


if __name__ == "__main__":
    main()
