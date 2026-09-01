#!/usr/bin/env python3
"""Backtest dacapogo's public daily formula on Binance USD-M futures.

The formula is source-parity with ``crypto_backtest.py`` at upstream HEAD
``633ba5d6bc0c84a20696af6b2bf807cf55d21248``. Official Binance daily klines
drive selection and signals; local 1-minute bars only audit executable exits,
funding, and leverage/liquidation risk.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import resource
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, date, datetime, time as dt_time, timedelta
from functools import cache
from itertools import batched
from pathlib import Path
from typing import Any, cast

import polars as pl
import lumina_quant.backtesting.execution_model as execution_model_module
import lumina_quant.strategies.dacapogo_daily_source as strategy_module

from lumina_quant.backtesting.execution_model import ExecutionModel, ExecutionModelConfig
from lumina_quant.configuration import get_default_runtime_config
from lumina_quant.research.run_card import (
    atomic_output_path,
    atomic_write_text,
    runtime_provenance,
)
from lumina_quant.research_universe import BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS
from lumina_quant.strategies.dacapogo_daily_source import (
    COST,
    FEE,
    SL,
    SLIP,
    TOPK,
    TP,
    backtest_daily,
)

SOURCE_URL = "https://github.com/HokyoungJung/dacapogo"
SOURCE_COMMIT = "633ba5d6bc0c84a20696af6b2bf807cf55d21248"
SOURCE_UNCHANGED_SINCE = "c89fac5d0f64243c49589409900e085f397ee929"
SOURCE_FILE_SHA256 = "17516d9457540e978d4828620c99794df0617c154faead5acee9f0847b5fcd8e"
FAPI_BASE_URL = "https://fapi.binance.com"
FAPI_LOW_WEIGHT_LIMIT = 499
MINUTE_MS = 60_000
DAY_MS = 86_400_000
MINUTES_PER_DAY = 1_440
SOURCE_END_DATE = date(2026, 7, 21)


def _parse_date(value: str) -> date:
    return date.fromisoformat(str(value).strip())


def _file_identity(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _compact_symbol(value: str) -> str:
    token = str(value).strip().upper().replace("/", "").replace("-", "")
    if not token.endswith("USDT"):
        raise ValueError(f"expected a USDT symbol, got {value!r}")
    return token


def _parse_leverages(value: str) -> tuple[int, ...]:
    leverages = tuple(sorted({int(item) for item in str(value).split(",") if item.strip()}))
    if not leverages or leverages[0] < 1 or leverages[-1] > 125:
        raise ValueError("leverages must be comma-separated integers from 1 to 125")
    return leverages


def _parse_symbols(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(_compact_symbol(item) for item in str(value).split(",") if item.strip())
    )


def _date_paths(base: Path, start: date, end: date) -> list[str]:
    paths: list[str] = []
    current = start
    while current <= end:
        paths.extend(str(path) for path in sorted((base / f"date={current}").glob("*.parquet")))
        current += timedelta(days=1)
    return paths


def _price_bounds(data_root: Path, exchange: str, symbol: str) -> tuple[datetime, datetime]:
    base = data_root / f"exchange={exchange}" / f"symbol={symbol}" / "timeframe=1m"
    paths = sorted(str(path) for path in base.glob("date=*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no 1m OHLCV for {symbol}: {base}")
    row = (
        pl.scan_parquet(paths)
        .select(
            pl.col("datetime").min().alias("first"),
            pl.col("datetime").max().alias("last"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    if row["first"] is None or row["last"] is None:
        raise ValueError(f"no 1m OHLCV rows for {symbol}")
    return row["first"].replace(tzinfo=UTC), row["last"].replace(tzinfo=UTC)


def resolve_default_window(
    data_root: Path,
    exchange: str,
    symbols: tuple[str, ...],
) -> tuple[date, date]:
    """Return the local common full-day window, preserving one prior rank day."""
    bounds = [_price_bounds(data_root, exchange, symbol) for symbol in symbols]
    first = max(item[0] for item in bounds)
    last = min(item[1] for item in bounds)
    first_full = first.date() if first.time() == dt_time.min else first.date() + timedelta(days=1)
    last_full = last.date() if last.time() >= dt_time(23, 59) else last.date() - timedelta(days=1)
    start = first_full + timedelta(days=1)
    if start > last_full:
        raise ValueError("local 1m coverage has fewer than two common full UTC days")
    return start, last_full


def _fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "LuminaQuant/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
            if isinstance(payload, dict) and isinstance(payload.get("code"), int):
                raise RuntimeError(f"Binance API error: {payload}")
            return payload
        except urllib.error.HTTPError as exc:
            if exc.code not in {408, 418, 429, 500, 502, 503, 504} or attempt == 3:
                raise
            time.sleep(float(exc.headers.get("Retry-After", attempt + 1)))
        except urllib.error.URLError, TimeoutError, ConnectionResetError:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("unreachable Binance API retry state")


def _active_coin_perpetuals(end: date) -> tuple[tuple[str, ...], dict[str, Any]]:
    payload = _fetch_json(f"{FAPI_BASE_URL}/fapi/v1/exchangeInfo")
    if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
        raise ValueError("invalid Binance exchangeInfo response")
    end_ms = int(
        datetime.combine(end + timedelta(days=1), dt_time.min, tzinfo=UTC).timestamp() * 1000
    )
    symbols = tuple(
        sorted(
            str(row["symbol"])
            for row in payload["symbols"]
            if row.get("status") == "TRADING"
            and row.get("contractType") == "PERPETUAL"
            and row.get("quoteAsset") == "USDT"
            and row.get("underlyingType") == "COIN"
            and int(row.get("onboardDate") or 0) < end_ms
        )
    )
    if len(symbols) < TOPK:
        raise ValueError(f"Binance active coin-perpetual universe has only {len(symbols)} symbols")
    return symbols, {
        "mode": "current_active_binance_coin_perpetuals",
        "endpoint": f"{FAPI_BASE_URL}/fapi/v1/exchangeInfo",
        "symbols": len(symbols),
        "snapshot_sha256": hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        "survivorship_bias": "current-active snapshot excludes contracts delisted before the run",
    }


def _fetch_fapi_daily(
    symbol: str,
    *,
    load_start: date,
    end: date,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if end >= datetime.now(UTC).date():
        raise ValueError("end must be a fully closed UTC day")
    start_ms = int(datetime.combine(load_start, dt_time.min, tzinfo=UTC).timestamp() * 1000)
    end_ms = (
        int(datetime.combine(end + timedelta(days=1), dt_time.min, tzinfo=UTC).timestamp() * 1000)
        - 1
    )
    cursor = start_ms
    rows: list[list[Any]] = []
    while cursor <= end_ms:
        query = urllib.parse.urlencode(
            {
                "symbol": symbol,
                "interval": "1d",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": FAPI_LOW_WEIGHT_LIMIT,
            }
        )
        batch = _fetch_json(f"{FAPI_BASE_URL}/fapi/v1/klines?{query}")
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + DAY_MS
        if len(batch) < FAPI_LOW_WEIGHT_LIMIT:
            break
    records = [
        {
            "market": symbol,
            "date": datetime.fromtimestamp(int(row[0]) / 1000, tz=UTC).date(),
            "value": float(row[7]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "open_time_ms": int(row[0]),
            "close_time_ms": int(row[6]),
        }
        for row in rows
        if isinstance(row, list) and len(row) >= 8
    ]
    daily = pl.DataFrame(records).sort("date") if records else pl.DataFrame()
    if daily.is_empty():
        raise ValueError(f"{symbol} has no FAPI daily data through {end}")
    first_day = daily[0, "date"]
    expected_days = (end - first_day).days + 1
    invalid = (
        daily.filter(
            pl.any_horizontal(
                pl.col(name).is_null() | ~pl.col(name).is_finite()
                for name in ("value", "open", "high", "low", "close")
            )
            | (pl.col("value") < 0)
            | (pl.min_horizontal("open", "high", "low", "close") <= 0)
            | (pl.col("high") < pl.max_horizontal("open", "close"))
            | (pl.col("low") > pl.min_horizontal("open", "close"))
            | (pl.col("close_time_ms") != pl.col("open_time_ms") + DAY_MS - 1)
        )
        if not daily.is_empty()
        else daily
    )
    if (
        daily.height != expected_days
        or invalid.height
        or first_day < load_start
        or daily[-1, "date"] != end
    ):
        raise ValueError(
            f"{symbol} FAPI daily coverage failed for "
            f"{load_start}..{end}: days={daily.height}/{expected_days}, invalid={invalid.height}"
        )
    first_ms = int(datetime.combine(first_day, dt_time.min, tzinfo=UTC).timestamp() * 1000)
    expected_open_times = [first_ms + index * DAY_MS for index in range(expected_days)]
    if daily.get_column("open_time_ms").to_list() != expected_open_times:
        raise ValueError(f"{symbol} FAPI daily timestamps are not contiguous UTC days")
    response_sha256 = hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return daily, {
        "symbol": symbol,
        "days": int(daily.height),
        "requested_start": str(load_start),
        "start": str(first_day),
        "end": str(daily[-1, "date"]),
        "turnover_field": "FAPI kline quote asset volume (response index 7)",
        "zero_turnover_days": int(daily.filter(pl.col("value") == 0).height),
        "source": f"{FAPI_BASE_URL}/fapi/v1/klines interval=1d",
        "response_sha256": response_sha256,
    }


def load_daily_panel(
    symbols: tuple[str, ...],
    *,
    start: date,
    end: date,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    if start > end:
        raise ValueError("start must be on or before end")
    load_start = start - timedelta(days=1)
    parts: list[pl.DataFrame] = []
    audits: list[dict[str, Any]] = []

    def fetch(symbol: str) -> tuple[pl.DataFrame, dict[str, Any]]:
        return _fetch_fapi_daily(symbol, load_start=load_start, end=end)

    with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as pool:
        for chunk in batched(symbols, 32, strict=False):
            results = list(pool.map(fetch, chunk))
            parts.append(pl.concat([frame for frame, _ in results], how="vertical"))
            audits.extend(audit for _, audit in results)
    return pl.concat(parts, how="vertical").sort(["market", "date"]), audits


def _fetch_funding_rates(symbol: str, start: date, end: date) -> list[tuple[int, float, float]]:
    start_ms = int(datetime.combine(start, dt_time.min, tzinfo=UTC).timestamp() * 1000)
    end_ms = (
        int(datetime.combine(end + timedelta(days=1), dt_time.min, tzinfo=UTC).timestamp() * 1000)
        - 1
    )
    cursor = start_ms
    rates: list[tuple[int, float, float]] = []
    while cursor <= end_ms:
        query = urllib.parse.urlencode(
            {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000}
        )
        batch = _fetch_json(f"{FAPI_BASE_URL}/fapi/v1/fundingRate?{query}")
        if not isinstance(batch, list):
            raise ValueError(f"invalid Binance funding response for {symbol}")
        if not batch:
            break
        for row in batch:
            timestamp = int(row["fundingTime"])
            rate = float(row["fundingRate"])
            mark_price = float(row["markPrice"])
            if not math.isfinite(rate) or not math.isfinite(mark_price):
                raise ValueError(f"invalid Binance funding values for {symbol}")
            rates.append((timestamp, rate, mark_price))
        cursor = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < 1000:
            break
    if not rates:
        raise ValueError(f"empty Binance funding history for traded symbol {symbol}")
    return sorted(set(rates))


def _fetch_fapi_minute_day(
    symbol: str,
    day: date,
    *,
    endpoint: str = "klines",
) -> pl.DataFrame:
    start_ms = int(datetime.combine(day, dt_time.min, tzinfo=UTC).timestamp() * 1000)
    end_ms = start_ms + DAY_MS - 1
    cursor = start_ms
    rows: list[list[Any]] = []
    while cursor <= end_ms:
        query = urllib.parse.urlencode(
            {
                "symbol": symbol,
                "interval": "1m",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": FAPI_LOW_WEIGHT_LIMIT,
            }
        )
        batch = _fetch_json(f"{FAPI_BASE_URL}/fapi/v1/{endpoint}?{query}")
        if not isinstance(batch, list):
            raise ValueError(f"invalid Binance {endpoint} minute response for {symbol} {day}")
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + MINUTE_MS
        if len(batch) < FAPI_LOW_WEIGHT_LIMIT:
            break
    records = [
        {
            "datetime": datetime.fromtimestamp(int(row[0]) / 1000, tz=UTC).replace(tzinfo=None),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        }
        for row in rows
        if isinstance(row, list) and len(row) >= 6
    ]
    if not records:
        raise ValueError(f"no Binance {endpoint} minute data for {symbol} {day}")
    return pl.DataFrame(records).sort("datetime")


def _validate_minute_frame(frame: pl.DataFrame, symbol: str, day: date, label: str) -> pl.DataFrame:
    start = datetime.combine(day, dt_time.min)
    end = start + timedelta(days=1)
    frame = frame.filter((pl.col("datetime") >= start) & (pl.col("datetime") < end)).sort(
        "datetime"
    )
    expected = [start + timedelta(minutes=index) for index in range(MINUTES_PER_DAY)]
    if frame.height != MINUTES_PER_DAY or frame["datetime"].to_list() != expected:
        raise ValueError(f"incomplete {label} 1m day: {symbol} {day}")
    invalid = frame.filter(
        ~pl.col("open").is_finite()
        | ~pl.col("high").is_finite()
        | ~pl.col("low").is_finite()
        | ~pl.col("close").is_finite()
        | (pl.col("open") <= 0)
        | (pl.col("high") < pl.max_horizontal("open", "close"))
        | (pl.col("low") > pl.min_horizontal("open", "close"))
    )
    if invalid.height:
        raise ValueError(f"invalid {label} 1m OHLC for {symbol} {day}")
    return frame


def _load_minute_day(
    data_root: Path,
    exchange: str,
    symbol: str,
    day: date,
    expected_daily: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    base = data_root / f"exchange={exchange}" / f"symbol={symbol}" / "timeframe=1m"
    paths = _date_paths(base, day, day)
    if paths:
        frame = (
            pl.scan_parquet(paths)
            .select("datetime", "open", "high", "low", "close", "volume")
            .sort("datetime")
            .collect(engine="streaming")
        )
        source = "local_parquet"
    else:
        frame = _fetch_fapi_minute_day(symbol, day)
        source = "fapi_1m"
    frame = _validate_minute_frame(frame, symbol, day, "trade-price")
    actual = {
        "open": float(frame[0, "open"]),
        "high": float(cast(float, frame["high"].max())),
        "low": float(cast(float, frame["low"].min())),
        "close": float(frame[-1, "close"]),
    }
    for field, value in actual.items():
        if not math.isclose(value, float(expected_daily[field]), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"local 1m/FAPI 1d {field} mismatch for {symbol} {day}: "
                f"{value} != {expected_daily[field]}"
            )
    mark_frame = _validate_minute_frame(
        _fetch_fapi_minute_day(symbol, day, endpoint="markPriceKlines"),
        symbol,
        day,
        "mark-price",
    )
    if mark_frame["datetime"].to_list() != frame["datetime"].to_list():
        raise ValueError(f"trade/mark 1m timestamps mismatch for {symbol} {day}")
    return frame.to_dicts(), mark_frame.to_dicts(), source


@cache
def _execution_model(leverage: int) -> ExecutionModel:
    base = ExecutionModelConfig.from_runtime(get_default_runtime_config())
    return ExecutionModel(
        replace(
            base,
            leverage=max(1, int(leverage)),
            margin_mode="isolated",
            taker_fee_rate=FEE,
        )
    )


def _execution_assumptions() -> dict[str, float | str]:
    cfg = _execution_model(1).cfg
    return {
        "margin_mode": cfg.margin_mode,
        "maintenance_margin_rate": cfg.maintenance_margin_rate,
        "liquidation_buffer_rate": cfg.liquidation_buffer_rate,
        "liquidation_taker_fee_rate": cfg.taker_fee_rate,
    }


def _simulate_trade(
    bars: list[dict[str, Any]],
    mark_bars: list[dict[str, Any]],
    *,
    entry_trigger: float,
    leverage: int,
    variant: str,
    same_bar_priority: str,
    funding_rates: list[tuple[int, float, float]],
    stop_pct: float = SL,
    take_profit_pct: float = TP,
    round_trip_cost: float = COST,
) -> dict[str, Any]:
    if variant not in {"close", "tp_sl"} or same_bar_priority not in {
        "stop_first",
        "tp_first",
    }:
        raise ValueError("invalid execution scenario")
    if len(mark_bars) != len(bars) or any(
        trade["datetime"] != mark["datetime"] for trade, mark in zip(bars, mark_bars, strict=True)
    ):
        raise ValueError("trade/mark 1m bars must be complete and timestamp-aligned")
    entry_index = next(
        (index for index, bar in enumerate(bars) if float(bar["high"]) >= entry_trigger),
        None,
    )
    if entry_index is None:
        raise ValueError("daily trigger was not present in audited minute bars")
    entry_bar = bars[entry_index]
    entry_price = max(float(entry_trigger), float(entry_bar["open"]))
    stop_price = entry_price * (1.0 - stop_pct)
    take_profit_price = entry_price * (1.0 + take_profit_pct)
    model = _execution_model(leverage)
    base_liquidation_price = model.liquidation_price(qty=1.0, entry_price=entry_price)
    liquidation_price = base_liquidation_price
    liquidation_denominator = 1.0 - model.cfg.maintenance_margin_rate
    if liquidation_denominator <= 0:
        raise ValueError("maintenance margin rate must be below one")
    exit_bar = bars[-1]
    exit_price = float(exit_bar["close"])
    reason = "daily_close"
    ambiguous_minute = False
    mark_liquidation_breach = False
    entry_ms = int(entry_bar["datetime"].replace(tzinfo=UTC).timestamp() * 1000)
    day_end_ms = int(bars[-1]["datetime"].replace(tzinfo=UTC).timestamp() * 1000) + MINUTE_MS
    pending_funding = sorted(event for event in funding_rates if entry_ms < event[0] < day_end_ms)
    funding_index = 0
    funding_rate = 0.0
    funding_cash = 0.0
    funding_event_count = 0
    funding_margin_shift = 0.0

    for index in range(entry_index, len(bars)):
        bar = bars[index]
        mark_bar = mark_bars[index]
        open_price = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        mark_open = float(mark_bar["open"])
        mark_low = float(mark_bar["low"])
        mark_close = float(mark_bar["close"])
        bar_ms = int(bar["datetime"].replace(tzinfo=UTC).timestamp() * 1000)

        entered_at_bar_open = index == entry_index and open_price >= entry_trigger
        if (
            (index > entry_index or entered_at_bar_open)
            and liquidation_price is not None
            and mark_open <= liquidation_price
        ):
            mark_liquidation_breach = True
            exit_bar, exit_price, reason = bar, open_price, "liquidation_gap"
            break

        funding_liquidated = False
        while funding_index < len(pending_funding) and pending_funding[funding_index][0] < (
            bar_ms + MINUTE_MS
        ):
            _, rate, funding_mark = pending_funding[funding_index]
            funding_index += 1
            if funding_mark <= 0:
                raise ValueError("nonpositive funding mark price while position is open")
            if liquidation_price is not None and funding_mark <= liquidation_price:
                mark_liquidation_breach = True
                exit_bar, exit_price, reason = bar, open_price, "liquidation_funding"
                funding_liquidated = True
                break
            funding_rate += rate
            funding_cash += rate * funding_mark
            funding_event_count += 1
            funding_margin_shift = funding_cash / liquidation_denominator
            if base_liquidation_price is not None:
                liquidation_price = max(
                    0.0,
                    base_liquidation_price + funding_margin_shift,
                )
                if funding_mark <= liquidation_price:
                    mark_liquidation_breach = True
                    exit_bar, exit_price, reason = bar, open_price, "liquidation_funding"
                    funding_liquidated = True
                    break
        if funding_liquidated:
            break

        if index > entry_index:
            if open_price <= stop_price:
                exit_bar, exit_price, reason = bar, open_price, "stop_gap"
                break
            if variant == "tp_sl" and open_price >= take_profit_price:
                exit_bar, exit_price, reason = bar, open_price, "take_profit_gap"
                break

        hit_stop = low <= stop_price
        hit_tp = variant == "tp_sl" and high >= take_profit_price
        hit_liquidation = liquidation_price is not None and (
            mark_low <= liquidation_price or mark_close <= liquidation_price
        )
        mark_liquidation_breach |= hit_liquidation
        downside_reason = "stop_loss"
        downside_price = stop_price
        if hit_liquidation and liquidation_price is not None and not hit_stop:
            downside_reason = "liquidation"
            downside_price = liquidation_price
        elif hit_liquidation and liquidation_price is not None and hit_stop:
            ambiguous_minute = True
            if same_bar_priority == "stop_first":
                downside_reason = "liquidation"
                downside_price = liquidation_price

        entry_downside_ambiguous = (
            index == entry_index
            and open_price < entry_trigger
            and (hit_stop or downside_reason == "liquidation")
        )
        ambiguous_minute |= entry_downside_ambiguous

        if hit_tp and (hit_stop or downside_reason == "liquidation"):
            ambiguous_minute = True
            if same_bar_priority == "tp_first":
                exit_bar, exit_price, reason = bar, take_profit_price, "take_profit"
            else:
                exit_bar, exit_price, reason = bar, downside_price, downside_reason
            break
        if downside_reason == "liquidation" or hit_stop:
            entry_minute_recovered = (not hit_stop or close > stop_price) and (
                not hit_liquidation
                or (liquidation_price is not None and mark_close > liquidation_price)
            )
            if (
                entry_downside_ambiguous
                and same_bar_priority == "tp_first"
                and entry_minute_recovered
            ):
                continue
            exit_bar, exit_price, reason = bar, downside_price, downside_reason
            break
        if hit_tp:
            exit_bar, exit_price, reason = bar, take_profit_price, "take_profit"
            break

    funding_rate = float(funding_rate)
    funding_return = float(funding_cash / entry_price)
    raw_return = exit_price / entry_price - 1.0
    slot_return = (
        -1.0
        if reason.startswith("liquidation")
        else max(-1.0, leverage * (raw_return - round_trip_cost - funding_return))
    )
    return {
        "entry_time": entry_bar["datetime"],
        "exit_time": exit_bar["datetime"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "reason": reason,
        "raw_return": raw_return,
        "funding_rate": funding_rate,
        "funding_return": funding_return,
        "funding_events": funding_event_count,
        "funding_margin_shift": funding_margin_shift,
        "slot_return": slot_return,
        "liquidated": reason.startswith("liquidation"),
        "mark_liquidation_breach": mark_liquidation_breach,
        "ambiguous_minute": ambiguous_minute,
    }


def _execution_backtest(
    panel: pl.DataFrame,
    source_trades: pl.DataFrame,
    *,
    data_root: Path,
    exchange: str,
    start: date,
    end: date,
    leverages: tuple[int, ...],
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    trade_rows = list(source_trades.iter_rows(named=True))
    trade_keys = source_trades.select("market", "date").unique()
    daily_lookup = {
        (str(row["market"]), row["date"]): row
        for row in panel.join(trade_keys, on=["market", "date"], how="inner").iter_rows(named=True)
    }
    traded_symbols = sorted(set(source_trades["market"].to_list()))
    trade_dates: dict[str, set[date]] = {symbol: set() for symbol in traded_symbols}
    for trade in trade_rows:
        trade_dates[str(trade["market"])].add(trade["date"])

    def fetch_funding(symbol: str) -> list[tuple[int, float, float]]:
        wanted = trade_dates[symbol]
        return [
            item
            for item in _fetch_funding_rates(symbol, min(wanted), max(wanted))
            if datetime.fromtimestamp(item[0] / 1000, tz=UTC).date() in wanted
        ]

    funding: dict[str, list[tuple[int, float, float]]] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(traded_symbols))) as pool:
        for chunk in batched(traded_symbols, 32, strict=False):
            funding.update(zip(chunk, pool.map(fetch_funding, chunk), strict=True))
    scenarios = (
        ("close_exit_stop_first", "close", "stop_first"),
        ("close_exit_entry_last", "close", "tp_first"),
        ("tp_sl_stop_first", "tp_sl", "stop_first"),
        ("tp_sl_tp_first", "tp_sl", "tp_first"),
    )
    records: list[dict[str, Any]] = []
    minute_sources: dict[str, int] = {}
    mark_price_source = "fapi_mark_price_1m"

    def load_trade(
        trade: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        symbol = str(trade["market"])
        day = trade["date"]
        return _load_minute_day(data_root, exchange, symbol, day, daily_lookup[(symbol, day)])

    with ThreadPoolExecutor(max_workers=8) as pool:
        for chunk in batched(trade_rows, 32, strict=False):
            for trade, (bars, mark_bars, minute_source) in zip(
                chunk, pool.map(load_trade, chunk), strict=True
            ):
                symbol = str(trade["market"])
                day = trade["date"]
                minute_sources[minute_source] = minute_sources.get(minute_source, 0) + 1
                for leverage in leverages:
                    for scenario, variant, priority in scenarios:
                        result = _simulate_trade(
                            bars,
                            mark_bars,
                            entry_trigger=float(trade["entry"]),
                            leverage=leverage,
                            variant=variant,
                            same_bar_priority=priority,
                            funding_rates=funding[symbol],
                        )
                        records.append(
                            {
                                "market": symbol,
                                "date": day,
                                "scenario": scenario,
                                "leverage": leverage,
                                "minute_source": minute_source,
                                "mark_price_source": mark_price_source,
                                **result,
                            }
                        )
    execution_trades = pl.DataFrame(records)
    grouped = {
        (str(row["scenario"]), int(row["leverage"]), row["date"]): float(row["daily_return"])
        for row in execution_trades.group_by("scenario", "leverage", "date")
        .agg((pl.col("slot_return").sum() / TOPK).alias("daily_return"))
        .iter_rows(named=True)
    }
    daily_records = []
    current = start
    while current <= end:
        for leverage in leverages:
            for scenario, _, _ in scenarios:
                daily_records.append(
                    {
                        "date": current,
                        "scenario": scenario,
                        "leverage": leverage,
                        "daily_return": grouped.get((scenario, leverage, current), 0.0),
                    }
                )
        current += timedelta(days=1)
    execution_daily = pl.DataFrame(daily_records).sort(["scenario", "leverage", "date"])
    return (
        execution_trades,
        execution_daily,
        {
            "audited_trigger_symbol_days": int(source_trades.height),
            "minute_sources": minute_sources,
            "mark_price_sources": {mark_price_source: int(source_trades.height)},
            "mark_price_endpoint": f"{FAPI_BASE_URL}/fapi/v1/markPriceKlines interval=1m",
            "funding_source": f"{FAPI_BASE_URL}/fapi/v1/fundingRate",
            "summaries": _execution_slice(execution_trades, execution_daily, start, end),
        },
    )


def _summarize_execution(trades: pl.DataFrame, daily: pl.DataFrame) -> dict[str, Any]:
    returns = [float(value) for value in daily["daily_return"].to_list()]
    trade_returns = [float(value) for value in trades["slot_return"].to_list()]
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, 1.0 - equity / peak)
    mean = sum(returns) / len(returns) if returns else 0.0
    variance = (
        sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        if len(returns) > 1
        else 0.0
    )
    return {
        "trades": len(trade_returns),
        "total_return": equity - 1.0,
        "max_drawdown": max_drawdown,
        "sharpe_365": math.sqrt(365.0) * mean / math.sqrt(variance) if variance > 0 else 0.0,
        "win_rate": sum(value > 0 for value in trade_returns) / len(trade_returns)
        if trade_returns
        else 0.0,
        "liquidations": int(trades["liquidated"].sum()),
        "possible_liquidations": int(trades["mark_liquidation_breach"].sum()),
        "ambiguous_minute_trades": int(trades["ambiguous_minute"].sum()),
        "funding_events": int(trades["funding_events"].sum()),
        "funding_rate_sum": float(trades["funding_rate"].sum()),
        "funding_return_sum": float(trades["funding_return"].sum()),
        "worst_day": min(returns, default=0.0),
    }


def _execution_slice(
    trades: pl.DataFrame,
    daily: pl.DataFrame,
    start: date,
    end: date,
) -> dict[str, Any]:
    if start > end:
        return {"start": str(start), "end": str(end), "status": "empty"}
    selected_trades = trades.filter((pl.col("date") >= start) & (pl.col("date") <= end))
    selected_daily = daily.filter((pl.col("date") >= start) & (pl.col("date") <= end))
    summaries: dict[str, Any] = {}
    for row in selected_daily.select("scenario", "leverage").unique().iter_rows(named=True):
        scenario = str(row["scenario"])
        leverage = int(row["leverage"])
        predicate = (pl.col("scenario") == scenario) & (pl.col("leverage") == leverage)
        summaries[f"{scenario}_{leverage}x"] = _summarize_execution(
            selected_trades.filter(predicate), selected_daily.filter(predicate)
        )
    return {
        "start": str(start),
        "end": str(end),
        "status": "ok",
        "summaries": dict(sorted(summaries.items())),
    }


def _summarize(
    trades: pl.DataFrame,
    daily: pl.DataFrame,
    ret_col: str,
    *,
    calendar_days: int,
) -> dict[str, Any]:
    event_returns = [float(value) for value in daily.get_column(ret_col).to_list()]
    returns = event_returns + [0.0] * max(0, calendar_days - len(event_returns))
    trade_returns = [float(value) for value in trades.get_column(ret_col).to_list()]
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, 1.0 - equity / peak)
    mean = sum(returns) / len(returns) if returns else 0.0
    variance = (
        sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        if len(returns) > 1
        else 0.0
    )
    return {
        "trades": len(trade_returns),
        "days_with_trades": len(event_returns),
        "win_rate": sum(value > 0 for value in trade_returns) / len(trade_returns)
        if trade_returns
        else 0.0,
        "avg_trade_return": sum(trade_returns) / len(trade_returns) if trade_returns else 0.0,
        "total_return": equity - 1.0,
        "positive_days": sum(value > 0 for value in returns),
        "worst_day": min(returns, default=0.0),
        "max_drawdown": max_drawdown,
        "sharpe_365": math.sqrt(365.0) * mean / math.sqrt(variance) if variance > 0 else 0.0,
    }


def _benchmark(panel: pl.DataFrame, start: date, end: date) -> float | None:
    btc = panel.filter(
        (pl.col("market") == "BTCUSDT") & (pl.col("date") >= start) & (pl.col("date") <= end)
    ).sort("date")
    if btc.is_empty():
        return None
    return float(btc[-1, "close"] / btc[0, "open"] - 1.0)


def _slice_result(panel: pl.DataFrame, start: date, end: date) -> dict[str, Any]:
    if start > end:
        return {"start": str(start), "end": str(end), "status": "empty"}
    trades, daily = backtest_daily(panel, start_date=start, end_date=end)
    calendar_days = (end - start).days + 1
    return {
        "start": str(start),
        "end": str(end),
        "status": "ok",
        "ret_pess": _summarize(trades, daily, "ret_pess", calendar_days=calendar_days),
        "ret_opt": _summarize(trades, daily, "ret_opt", calendar_days=calendar_days),
        "benchmark_btc_hold": _benchmark(panel, start, end),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    provenance = runtime_provenance(
        repo_root=Path(__file__).resolve().parents[2],
        packages=("polars",),
        source_files=(
            Path(__file__),
            Path(strategy_module.__file__),
            Path(execution_model_module.__file__),
        ),
    )
    source_files = provenance["source_files"]
    data_root = Path(args.data_root).resolve()
    exchange = str(args.exchange).strip().lower()
    leverages = _parse_leverages(args.leverages)
    if args.start and args.end:
        start, end = _parse_date(args.start), _parse_date(args.end)
    else:
        default_start, default_end = resolve_default_window(
            data_root, exchange, BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS
        )
        start = _parse_date(args.start) if args.start else default_start
        end = _parse_date(args.end) if args.end else default_end
    if start > end:
        raise ValueError("start must be on or before end")
    requested_symbols = _parse_symbols(args.symbols)
    if requested_symbols:
        symbols = requested_symbols
        universe_audit = {
            "mode": "explicit_fixed_basket",
            "symbols": len(symbols),
            "warning": "TOPK ranking is ineffective when the basket has TOPK or fewer symbols",
        }
    else:
        symbols, universe_audit = _active_coin_perpetuals(end)
    panel, audits = load_daily_panel(symbols, start=start - timedelta(days=6), end=end)
    trades, daily = backtest_daily(panel, start_date=start, end_date=end)
    if trades.is_empty():
        raise ValueError("no daily +4% triggers in the requested window")
    execution_trades, execution_daily, execution_meta = _execution_backtest(
        panel,
        trades,
        data_root=data_root,
        exchange=exchange,
        start=start,
        end=end,
        leverages=leverages,
    )

    latest_30_start = max(start, end - timedelta(days=29))
    post_source_start = max(start, SOURCE_END_DATE + timedelta(days=1))
    slices = {
        "full": _slice_result(panel, start, end),
        "latest_30d": _slice_result(panel, latest_30_start, end),
        "through_source_end": _slice_result(panel, start, min(end, SOURCE_END_DATE)),
        "strictly_after_source_end": _slice_result(panel, post_source_start, end),
    }
    execution_slices = {
        "full": execution_meta["summaries"],
        "latest_30d": _execution_slice(execution_trades, execution_daily, latest_30_start, end),
        "through_source_end": _execution_slice(
            execution_trades, execution_daily, start, min(end, SOURCE_END_DATE)
        ),
        "strictly_after_source_end": _execution_slice(
            execution_trades, execution_daily, post_source_start, end
        ),
    }
    elapsed = time.perf_counter() - started
    payload = {
        "artifact_kind": "dacapogo_daily_source_parity_backtest",
        "strategy_tier": "research_only",
        "promotion_eligible": False,
        "deploy_action": "cash",
        "publication": {
            "contract": "atomic file replacement; summary seal written last",
            "authority": "summary.json",
        },
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": {
            "url": SOURCE_URL,
            "commit": SOURCE_COMMIT,
            "unchanged_since_commit": SOURCE_UNCHANGED_SINCE,
            "file": "crypto_backtest.py",
            "file_sha256": SOURCE_FILE_SHA256,
            "lines": "22-56",
            "model": "daily ret_pess and ret_opt",
        },
        "adapter": {"file": Path(__file__).name, **source_files[str(Path(__file__).resolve())]},
        "local_code": {
            "strategy": source_files[str(Path(strategy_module.__file__).resolve())],
            "execution_model": source_files[str(Path(execution_model_module.__file__).resolve())],
        },
        "rules": {
            "topk": TOPK,
            "selection": "previous UTC day's native quote turnover rank",
            "entry": "daily open * 1.04 when daily high >= entry",
            "take_profit": TP,
            "loss_floor": SL,
            "fee_each_side": FEE,
            "slippage": SLIP,
            "round_trip_cost": COST,
            "portfolio_return": "sum(triggered trade returns) / TOPK",
        },
        "execution": {
            "leverages": list(leverages),
            "entry_fill": "first 1m trigger; max(daily trigger, minute open)",
            "stop": "post-entry -0.5%, gap-aware",
            "take_profit": "optional post-entry +0.8%, gap-aware",
            "scenarios": {
                "close_exit_stop_first": "no TP; adverse downside ordering, otherwise 23:59 close",
                "close_exit_entry_last": "no TP; recovered pre-entry lows are ignored when order is ambiguous",
                "tp_sl_stop_first": "adverse collision ordering: liquidation, SL, then TP",
                "tp_sl_tp_first": "favorable collision ordering: TP before downside",
            },
            "liquidation": "repository fixed-MMR isolated-margin approximation on official Binance USD-M 1m mark-price bars; chosen liquidation loses the full strategy slot",
            "liquidation_model": _execution_assumptions(),
            "funding": "actual Binance USD-M timestamps/rates processed chronologically; positive payments conservatively reduce isolated slot margin and move the modeled liquidation boundary",
            "cost": "source COST applied to every round trip; funding added separately",
            "audited_trigger_symbol_days": execution_meta["audited_trigger_symbol_days"],
            "minute_sources": execution_meta["minute_sources"],
            "mark_price_sources": execution_meta["mark_price_sources"],
            "mark_price_endpoint": execution_meta["mark_price_endpoint"],
            "funding_source": execution_meta["funding_source"],
            "slices": execution_slices,
        },
        "data": {
            "root": str(data_root),
            "exchange": exchange,
            "market": "Binance USD-M futures",
            "symbols": list(symbols),
            "universe": universe_audit,
            "start": str(start),
            "end": str(end),
            "panel_history_start": str(start - timedelta(days=7)),
            "signal_input": "official FAPI 1d klines",
            "execution_input": "complete 1m trade-price klines for fills plus official FAPI 1m mark-price klines for liquidation",
            "turnover": "official FAPI native quote asset volume (kline field 7)",
            "audits": audits,
        },
        "slices": slices,
        "limitations": [
            "The public repository calls this a transferred/reconstructed skeleton, not the private Dacapogo algorithm.",
            "ret_pess floors a close-based loss at -0.5%; it is not a path-aware executable stop fill.",
            "ret_opt uses the completed day's high before applying TP, so it is descriptive daily-OHLC accounting.",
            "Upbit KRW was replaced by Binance USD-M futures; rule parity does not imply venue/result parity.",
            "Liquidation uses official 1m mark-price OHLC and the repository maintenance-margin model; sub-minute mark-price paths remain unknown.",
            "Liquidation is not exchange-exact: the repository uses one fixed maintenance-margin rate rather than historical symbol/notional leverage brackets.",
            "Funding is conservatively debited from each isolated strategy slot instead of allowing subsidy from an external Futures wallet balance.",
            "A 1m candle cannot reveal entry/TP/SL order, so multiple path-order scenarios are reported without claiming guaranteed ordering.",
            "The default current-active universe has survivorship bias because historical delistings are absent from exchangeInfo.",
        ],
        "runtime": {
            "elapsed_seconds": elapsed,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "loading": "daily/API inputs in bounded 32-symbol-day chunks; execution models cached by leverage",
            "provenance": provenance,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "trades.csv": trades,
        "daily.csv": daily,
        "execution_trades.csv": execution_trades,
        "execution_daily.csv": execution_daily,
    }
    for name, frame in outputs.items():
        with atomic_output_path(output_dir / name) as temporary:
            frame.write_csv(temporary)
    panel_path = output_dir / "daily_panel.parquet"
    panel_manifest_path = output_dir / "daily_panel.parquet.manifest.json"
    with atomic_output_path(panel_path) as temporary:
        panel.write_parquet(temporary)
    atomic_write_text(
        panel_manifest_path,
        json.dumps({"file": _file_identity(panel_path), "audits": audits}, indent=2) + "\n",
    )
    payload["artifacts"] = {
        name: _file_identity(output_dir / name)
        for name in (*outputs, panel_path.name, panel_manifest_path.name)
    }
    atomic_write_text(
        output_dir / "summary.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".run.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _run(args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/market_parquet")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument(
        "--symbols",
        default="",
        help="Optional fixed USDT basket; default is all current active Binance coin perpetuals",
    )
    parser.add_argument("--start", help="Inclusive UTC date; default latest common full window")
    parser.add_argument("--end", help="Inclusive UTC date; default latest common full day")
    parser.add_argument("--leverages", default="1,3,5,10,20")
    parser.add_argument("--output-dir", default="var/reports/dacapogo_binance/daily_source")
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_arg_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
